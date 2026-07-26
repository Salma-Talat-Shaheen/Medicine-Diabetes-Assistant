#!/usr/bin/env python
"""
Multimodal PDF ingestion script.
Ingestion pipeline that pulling out the text and the visual content of a PDF -- embedded raster images
(figures, charts) AND vector-drawn diagrams that don't exist as an embedded image.
Each extracted image is:
  1. Saved to disk under --image-dir.
  2. Sent to a vision-capable LLM (via OpenRouter, same pattern as llm.py)
     which produces a clinically-useful text description -- for algorithms
     and tables it's asked to transcribe the actual decision steps/values,
     not just describe the picture.
  3. Wrapped in a langchain Document whose page_content is that caption and
     whose metadata points back to the saved image file, page number, and
     source PDF.
Downstream, src/agent.py's context-builder can check metadata["content_type"]
== "image" and include metadata["image_path"] when it retrieves one of
these chunks, and src/web/app.py's WeasyPrint report renderer can embed that
image file directly in the generated PDF report.
Usage:
    python scripts/ingest_pdf_multimodal.py path/to/guideline.pdf
    python scripts/ingest_pdf_multimodal.py path/to/dir --image-dir data/images
Requires (add to pyproject.toml):
    pymupdf>=1.24.0
    Pillow>=10.0.0
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PIL import Image
from tqdm import tqdm
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medicine_docs")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "openai/gpt-4o-mini")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
# Skip anything smaller than this -- bullets, checkbox glyphs, tiny icons.
MIN_IMAGE_DIM_PX = 120
# Full-page render resolution when we fall back to rasterizing a page.
PAGE_RENDER_DPI = 200
# If the exact same image (by content hash) shows up more than this many
# times across the document, treat it as a repeating decorative element
# (header/footer logo, watermark) and drop it after the first occurrence.
REPEAT_DECORATIVE_THRESHOLD = 2
FIGURE_KEYWORDS = re.compile(
    r"\b(figure|fig\.|algorithm|table|flowchart|diagram)\s*\d*", re.IGNORECASE
)
CAPTION_PROMPT = """You are helping build a searchable index of figures from a \
clinical practice guideline for a medical RAG system.
Look at this image extracted from the document and respond in ONE of two ways:
If this is a genuine clinical figure (an algorithm/flowchart, a decision \
tree, a dosing/titration table, a data chart, a diagram of a process, etc.), \
write a dense, factual description a clinician could search for and act on. \
Explicitly transcribe: the steps/branches of any algorithm in order, the \
row/column values of any table, and any numeric thresholds, dosages, or \
units shown. Do not summarize vaguely -- capture the actual decision logic \
and numbers. Keep it under 300 words, plain text, no markdown.
If this is NOT clinically meaningful content -- a logo, a page border, an \
icon, a watermark, decorative artwork, or a blank/near-blank image -- \
respond with exactly the single word: DECORATIVE
Respond with the description or DECORATIVE only, nothing else."""
@dataclass
class ExtractedImage:
    """A candidate image pulled from the PDF, before captioning."""
    page_num: int  # 0-indexed
    image_path: Path
    source_type: str  # "embedded" or "full_page"
    content_hash: str
    caption: str = field(default="", repr=False)
def validate_config() -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )
def get_vision_llm() -> ChatOpenAI:
    """Vision-capable chat model via OpenRouter (same pattern as src/llm.py)."""
    return ChatOpenAI(
        model=VISION_MODEL_NAME,
        api_key=lambda: OPENROUTER_API_KEY,  # type: ignore
        base_url=OPENROUTER_BASE_URL,
        temperature=0.0,
        max_tokens=500,
    )
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=lambda: OPENROUTER_API_KEY,  # type: ignore
        base_url=OPENROUTER_BASE_URL,
    )
def get_vector_store(embeddings: OpenAIEmbeddings, persist_directory: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
# --------------------------------------------------------------------------
# Step 1: extract candidate images from the PDF
# --------------------------------------------------------------------------
def _page_looks_like_figure(page: fitz.Page) -> bool:
    """Heuristic: does the page text mention Figure/Algorithm/Table/etc.?"""
    return bool(FIGURE_KEYWORDS.search(page.get_text()))
def _save_png(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(out_path, format="PNG")
def extract_images_from_pdf(
    pdf_path: Path, image_dir: Path, min_dim: int = MIN_IMAGE_DIM_PX
) -> list[ExtractedImage]:
    """
    Pull out both embedded raster images and, for pages that look like they
    contain a figure/algorithm/table but had no (large enough) embedded
    image, a full-page render -- this catches vector-drawn diagrams.
    """
    doc = fitz.open(str(pdf_path))
    doc_stem = pdf_path.stem
    candidates: list[ExtractedImage] = []
    hash_counts: dict[str, int] = {}
    print(f" Scanning {pdf_path.name} for images across {len(doc)} pages...")
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_had_large_embedded_image = False
        # --- 1a. embedded raster images ---
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                with Image.open(io.BytesIO(image_bytes)) as im:
                    width, height = im.size
                    if width < min_dim or height < min_dim:
                        continue  # icon/bullet/decoration, too small to matter
                    page_had_large_embedded_image = True
                    content_hash = hashlib.md5(image_bytes).hexdigest()
                    hash_counts[content_hash] = hash_counts.get(content_hash, 0) + 1
                    out_path = (
                        image_dir
                        / f"{doc_stem}_p{page_num + 1}_img{img_index}_{content_hash[:8]}.png"
                    )
                    if not out_path.exists():
                        _save_png(im, out_path)
                    candidates.append(
                        ExtractedImage(
                            page_num=page_num,
                            image_path=out_path,
                            source_type="embedded",
                            content_hash=content_hash,
                        )
                    )
            except Exception as e:  # corrupt / unsupported image stream
                print(f"    Skipping image xref {xref} on page {page_num + 1}: {e}")
        # --- 1b. fallback: full-page render for vector-drawn figures ---
        # If the page text suggests a figure/algorithm but we found no
        # substantial embedded raster image, the figure is very likely a
        # vector drawing (lines/boxes/arrows) -- rasterize the whole page.
        if _page_looks_like_figure(page) and not page_had_large_embedded_image:
            pix = page.get_pixmap(dpi=PAGE_RENDER_DPI)
            image_bytes = pix.tobytes("png")
            content_hash = hashlib.md5(image_bytes).hexdigest()
            hash_counts[content_hash] = hash_counts.get(content_hash, 0) + 1
            out_path = image_dir / f"{doc_stem}_p{page_num + 1}_fullpage.png"
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pix.save(str(out_path))
            candidates.append(
                ExtractedImage(
                    page_num=page_num,
                    image_path=out_path,
                    source_type="full_page",
                    content_hash=content_hash,
                )
            )
    doc.close()
    # --- Drop repeating decorative elements (logos/watermarks) ---
    # Keep only the first occurrence of any hash that repeats more than the
    # threshold; everything after that is almost certainly not a unique
    # clinical figure.
    seen: dict[str, int] = {}
    deduped: list[ExtractedImage] = []
    for cand in candidates:
        seen[cand.content_hash] = seen.get(cand.content_hash, 0) + 1
        is_repeating_decoration = (
            hash_counts[cand.content_hash] > REPEAT_DECORATIVE_THRESHOLD
        )
        if is_repeating_decoration and seen[cand.content_hash] > 1:
            continue
        deduped.append(cand)
    print(
        f"  ✓ Found {len(candidates)} candidate image(s), "
        f"{len(deduped)} after removing repeated decorative elements"
    )
    return deduped
# --------------------------------------------------------------------------
# Step 2: caption each image with a vision LLM
# --------------------------------------------------------------------------
def _encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
def caption_image(llm: ChatOpenAI, image: ExtractedImage, retries: int = 2) -> str:
    """Call the vision model once, with a couple of retries on failure."""
    b64 = _encode_image_b64(image.image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = llm.invoke([message])
            return response.content.strip()
        except Exception as e:  # transient API/network errors
            last_error = e
            if attempt < retries:
                continue
    print(f"    Captioning failed for {image.image_path.name}: {last_error}")
    return "DECORATIVE"  # fail safe: don't index something we couldn't read
def caption_images_concurrently(
    images: list[ExtractedImage], max_workers: int = MAX_WORKERS
) -> list[ExtractedImage]:
    """Caption all images in parallel (network-bound), return the kept ones."""
    llm = get_vision_llm()
    kept: list[ExtractedImage] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(caption_image, llm, img): img for img in images
        }
        for future in tqdm(
            as_completed(future_to_image), total=len(images), desc="Captioning images"
        ):
            img = future_to_image[future]
            caption = future.result()
            if caption.strip().upper() == "DECORATIVE":
                continue
            img.caption = caption
            kept.append(img)
    print(f"  ✓ {len(kept)}/{len(images)} images kept as clinically meaningful")
    return kept
# --------------------------------------------------------------------------
# Step 3: build Documents and add them to the vector store
# --------------------------------------------------------------------------
def build_image_documents(
    images: list[ExtractedImage], pdf_path: Path
) -> list[Document]:
    documents = []
    for img in images:
        documents.append(
            Document(
                page_content=(
                    f"[Figure on page {img.page_num + 1} of {pdf_path.name}]\n"
                    f"{img.caption}"
                ),
                metadata={
                    "source": str(pdf_path),
                    "page": img.page_num + 1,
                    "content_type": "image",
                    "image_path": str(img.image_path),
                    "extraction_method": img.source_type,
                },
            )
        )
    return documents
def add_documents_concurrently(
    vector_store: Chroma, documents: list[Document], max_workers: int, batch_size: int
) -> int:
    if not documents:
        return 0
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(vector_store.add_documents, batch) for batch in batches
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Indexing image captions"
        ):
            future.result()
    return len(documents)
# --------------------------------------------------------------------------
# End-to-end pipeline for a single PDF
# --------------------------------------------------------------------------
def ingest_pdf_multimodal(
    pdf_path: Path, image_dir: Path, vector_store: Chroma, min_dim: int
) -> None:
    print(f"\n Processing (multimodal): {pdf_path.name}")
    candidates = extract_images_from_pdf(pdf_path, image_dir, min_dim=min_dim)
    if not candidates:
        print("  (no candidate images found)")
        return
    kept = caption_images_concurrently(candidates)
    if not kept:
        print("  (no images were clinically meaningful after captioning)")
        return
    documents = build_image_documents(kept, pdf_path)
    total = add_documents_concurrently(
        vector_store, documents, max_workers=MAX_WORKERS, batch_size=32
    )
    print(f"Indexed {total} image caption(s) from {pdf_path.name}")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, caption, and index images/diagrams from PDF(s) "
        "into the same Chroma store used for text chunks."
    )
    parser.add_argument("path", type=str, help="PDF file or directory of PDFs")
    parser.add_argument(
        "--db-path", type=str, default=None, help="Chroma persist directory"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images",
        help="Where extracted image files are saved (default: data/images)",
    )
    parser.add_argument(
        "--min-image-dim",
        type=int,
        default=MIN_IMAGE_DIM_PX,
        help="Minimum width/height in pixels to consider an embedded image "
        "(filters out icons/bullets)",
    )
    args = parser.parse_args()
    try:
        validate_config()
        path = Path(args.path)
        image_dir = Path(args.image_dir)
        db_path = args.db_path or CHROMA_PERSIST_DIRECTORY
        embeddings = get_embeddings()
        vector_store = get_vector_store(embeddings, db_path)
        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"File is not a PDF: {path}")
            ingest_pdf_multimodal(path, image_dir, vector_store, args.min_image_dim)
        elif path.is_dir():
            pdf_files = sorted(path.glob("**/*.pdf"))
            if not pdf_files:
                raise ValueError(f"No PDF files found in: {path}")
            for pdf_file in pdf_files:
                ingest_pdf_multimodal(
                    pdf_file, image_dir, vector_store, args.min_image_dim
                )
        else:
            print(f" Error: Path does not exist: {args.path}", file=sys.stderr)
            sys.exit(1)
        print("\n---")
        print("Multimodal ingestion complete.")
        print(f" Vector store persisted to: {os.path.abspath(db_path)}")
        print(f" Image files saved under: {os.path.abspath(image_dir)}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__":
    main()
