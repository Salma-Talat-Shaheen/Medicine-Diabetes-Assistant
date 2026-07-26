#!/usr/bin/env python
"""
ingest_pdf_OCR.py
=================
Standalone OCR pipeline for scanned PDFs (image-only, no text layer).

Pipeline
--------
1. Extract embedded images directly from each PDF page (PyMuPDF).
   No rasterisation — the page IS the image.
2. Run EasyOCR on every page image and cache the result as JSON
   so OCR never runs twice for the same file.
3. Chunk the extracted text and build a FAISS vector store
   (also cached — rebuilt only when the OCR cache changes).
4. Accept a natural-language question and return THREE outputs
   side-by-side so you can compare them:

   a) RAG answer   — LLM answer grounded in retrieved context.
   b) No-RAG answer — same LLM, no context, pure parametric memory.
   c) Similarity scores — cosine distances for every retrieved chunk,
      plus a human-readable quality label.

Usage
-----
# Ingest a single scanned PDF then ask a question interactively:
    python ingest_pdf_OCR.py path/to/scanned.pdf

# Ingest a whole directory:
    python ingest_pdf_OCR.py path/to/dir/

# Skip rebuilding the index (already done) and just ask:
    python ingest_pdf_OCR.py path/to/scanned.pdf --query "What is the HbA1c target?"

# Force re-OCR even if cache exists:
    python ingest_pdf_OCR.py path/to/scanned.pdf --force-ocr

Environment variables (same .env as the rest of the project)
-------------------------------------------------------------
    OPENROUTER_API_KEY   — required
    OPENROUTER_BASE_URL  — default: https://openrouter.ai/api/v1
    CHROMA_PERSIST_DIR   — NOT used here; we use FAISS locally
    OCR_CACHE_DIR        — default: ./ocr_cache_easyocr
    FAISS_INDEX_DIR      — default: ./faiss_index_ocr
    LLM_MODEL            — default: mistralai/mixtral-8x7b-instruct
    EMBEDDING_MODEL      — default: sentence-transformers/all-MiniLM-L6-v2
    TOP_K                — chunks to retrieve (default: 4)
    CHUNK_SIZE           — default: 1 500
    CHUNK_OVERLAP        — default: 200
    MIN_CONFIDENCE       — EasyOCR word-level confidence floor (default: 0.4)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OCR_CACHE_DIR       = Path(os.getenv("OCR_CACHE_DIR", "./ocr_cache_easyocr"))
FAISS_INDEX_DIR     = Path(os.getenv("FAISS_INDEX_DIR", "./faiss_index_ocr"))
LLM_MODEL           = os.getenv("LLM_MODEL", "mistralai/mixtral-8x7b-instruct")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K               = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CONFIDENCE      = float(os.getenv("MIN_CONFIDENCE", "0.4"))

# ── Prompts ───────────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """\
You are an expert medical assistant. Answer the question below using ONLY
the context extracted from the provided medical standards document.
If the answer cannot be found in the context, say so explicitly.

Context:
{context}

Question: {question}

Answer (concise, evidence-based):"""

NO_RAG_PROMPT_TEMPLATE = """\
You are an expert medical assistant. Answer the following question using
your general medical knowledge. Be concise and factual.

Question: {question}

Answer:"""


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_config() -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set.\n"
            "Add it to your .env file:  OPENROUTER_API_KEY=sk-or-v1-..."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  OCR — extract images from PDF and run EasyOCR (cached)
# ═══════════════════════════════════════════════════════════════════════════════

def _ocr_cache_path(pdf_path: Path) -> Path:
    """
    Stable cache key: PDF stem + MD5 of (resolved path + file size + mtime).
    Changes if the file is replaced on disk; stable otherwise.
    """
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stat = pdf_path.stat()
    key  = f"{pdf_path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    digest = hashlib.md5(key.encode()).hexdigest()[:16]
    return OCR_CACHE_DIR / f"{pdf_path.stem}.{digest}.ocr.json"


def _extract_page_images(pdf_path: Path) -> list[tuple[int, np.ndarray]]:
    """
    Extract one image per page from a scanned PDF using PyMuPDF.

    Strategy (per page):
      1. `page.get_images()` — grab the first embedded raster image.
         Scanned PDFs store each page as a single embedded JPEG/PNG.
      2. Fallback: if no embedded image is found, render the page at
         2× zoom (≈144 dpi) — handles edge cases like OCR-d PDFs that
         re-embedded the text but kept the background as a vector page.

    Returns list of (1-indexed page number, RGB numpy array).
    """
    doc    = fitz.open(str(pdf_path))
    result = []

    for page_num in range(len(doc)):
        page     = doc[page_num]
        img_list = page.get_images(full=True)

        if img_list:
            # ── Scanned page: use the embedded image as-is ───────────
            xref       = img_list[0][0]
            base_image = doc.extract_image(xref)
            pil_img    = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
        else:
            # ── Fallback: rasterise the page ─────────────────────────
            mat     = fitz.Matrix(2.0, 2.0)
            pix     = page.get_pixmap(matrix=mat)
            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        result.append((page_num + 1, np.array(pil_img)))

    doc.close()
    return result


def ocr_pdf(pdf_path: Path, force: bool = False) -> list[dict]:
    """
    Run EasyOCR on every page of a scanned PDF.

    Returns a list of dicts:
        {"page": int, "text": str, "words": [{"text": str, "conf": float}]}

    Results are cached as JSON — OCR runs only once per PDF unless `force=True`.
    """
    cache_path = _ocr_cache_path(pdf_path)

    if cache_path.exists() and not force:
        print(f"  ↺  OCR cache hit — loading: {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"  Running EasyOCR on: {pdf_path.name}")

    # Import lazily so the module loads fast when OCR is not needed.
    import easyocr  # noqa: PLC0415
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    page_images = _extract_page_images(pdf_path)
    print(f"  {len(page_images)} page image(s) extracted.")

    pages_data: list[dict] = []

    for page_num, img_array in tqdm(page_images, desc="  OCR pages", unit="page"):
        # detail=1  → returns [(bbox, text, confidence), ...]
        raw = reader.readtext(img_array, detail=1, paragraph=False)

        words = [
            {"text": text, "conf": round(float(conf), 4)}
            for (_, text, conf) in raw
            if conf >= MIN_CONFIDENCE
        ]

        # Sort top-to-bottom by the y-coordinate of the bounding box's top-left
        raw_filtered = [r for r in raw if r[2] >= MIN_CONFIDENCE]
        raw_filtered.sort(key=lambda r: r[0][0][1])   # r[0][0][1] = top-left y
        page_text = "\n".join(r[1] for r in raw_filtered)

        pages_data.append({"page": page_num, "text": page_text, "words": words})

    # ── Persist cache ─────────────────────────────────────────────────────────
    cache_path.write_text(
        json.dumps(pages_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✓  OCR complete. Cached to: {cache_path.name}")
    return pages_data


def pages_to_documents(pages: list[dict], source: str) -> list[Document]:
    """Convert per-page OCR dicts into LangChain Documents."""
    return [
        Document(
            page_content=p["text"],
            metadata={"source": source, "page": p["page"]},
        )
        for p in pages
        if p["text"].strip()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Embeddings + FAISS vector store (cached on disk)
# ═══════════════════════════════════════════════════════════════════════════════

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _faiss_index_path(pdf_paths: list[Path]) -> Path:
    """
    Unique FAISS index directory per set of input PDFs.
    Key = sorted resolved paths + sizes + mtimes, so the index is
    rebuilt automatically whenever any source file changes.
    """
    key_parts = []
    for p in sorted(pdf_paths):
        stat = p.stat()
        key_parts.append(f"{p.resolve()}::{stat.st_size}::{int(stat.st_mtime)}")
    digest = hashlib.md5("\n".join(key_parts).encode()).hexdigest()[:16]
    return FAISS_INDEX_DIR / digest


def build_or_load_vector_store(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    pdf_paths: list[Path],
    force: bool = False,
) -> FAISS:
    """
    Build a FAISS vector store from documents, or load it from disk if
    an up-to-date index already exists.
    """
    index_dir = _faiss_index_path(pdf_paths)
    index_file = index_dir / "index.faiss"

    if index_file.exists() and not force:
        print(f"  ↺  FAISS index cache hit — loading: {index_dir}")
        store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"  ✓  Loaded {store.index.ntotal} vectors.")
        return store

    print("  Building FAISS index …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  ✓  {len(chunks)} chunks created from {len(documents)} page(s).")

    store = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    print(f"  ✓  FAISS index saved to: {index_dir}  ({store.index.ntotal} vectors)")
    return store


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  LLM helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=lambda: OPENROUTER_API_KEY,  # type: ignore[arg-type]
        base_url=OPENROUTER_BASE_URL,
        temperature=0.1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  Similarity scoring
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score_label(score: float) -> str:
    """Human-readable quality label for a cosine similarity score."""
    if score >= 0.85:
        return "🟢 Excellent"
    if score >= 0.70:
        return "🟡 Good"
    if score >= 0.55:
        return "🟠 Moderate"
    return "🔴 Weak"


def compute_similarity_scores(
    query: str,
    retrieved_docs: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> list[dict]:
    """
    Embed the query and each retrieved chunk, then compute cosine similarity.

    Returns a list of dicts (one per chunk):
        {
            "rank": int,
            "source": str,
            "page": int,
            "score": float,          # cosine similarity ∈ [-1, 1]
            "label": str,            # Excellent / Good / Moderate / Weak
            "preview": str,          # first 120 chars of the chunk
        }
    """
    query_vec  = np.array(embeddings.embed_query(query))
    chunk_vecs = np.array(embeddings.embed_documents(
        [doc.page_content for doc in retrieved_docs]
    ))

    scores = []
    for rank, (doc, vec) in enumerate(zip(retrieved_docs, chunk_vecs), start=1):
        sim = cosine_similarity(query_vec, vec)
        scores.append({
            "rank":    rank,
            "source":  doc.metadata.get("source", "unknown"),
            "page":    doc.metadata.get("page", "?"),
            "score":   round(sim, 4),
            "label":   score_label(sim),
            "preview": doc.page_content[:120].replace("\n", " "),
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  Query — RAG vs No-RAG with similarity scores
# ═══════════════════════════════════════════════════════════════════════════════

def query_pipeline(
    question: str,
    vector_store: FAISS,
    embeddings: HuggingFaceEmbeddings,
) -> dict:
    """
    Run the question through both RAG and No-RAG pipelines.

    Returns
    -------
    {
        "question":          str,
        "rag_answer":        str,
        "no_rag_answer":     str,
        "retrieved_chunks":  list[dict],   # with similarity scores
        "avg_similarity":    float,
    }
    """
    llm = get_llm()

    # ── Retrieve top-K chunks ─────────────────────────────────────────────────
    retriever     = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)

    # ── RAG answer ────────────────────────────────────────────────────────────
    rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    rag_chain  = (
        {
            "context":  lambda _: "\n\n".join(d.page_content for d in retrieved_docs),
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    rag_answer = rag_chain.invoke(question)

    # ── No-RAG answer ─────────────────────────────────────────────────────────
    no_rag_prompt = PromptTemplate.from_template(NO_RAG_PROMPT_TEMPLATE)
    no_rag_chain  = no_rag_prompt | llm | StrOutputParser()
    no_rag_answer = no_rag_chain.invoke({"question": question})

    # ── Similarity scores ─────────────────────────────────────────────────────
    sim_scores   = compute_similarity_scores(question, retrieved_docs, embeddings)
    avg_sim      = round(sum(s["score"] for s in sim_scores) / len(sim_scores), 4) \
                   if sim_scores else 0.0

    return {
        "question":         question,
        "rag_answer":       rag_answer,
        "no_rag_answer":    no_rag_answer,
        "retrieved_chunks": sim_scores,
        "avg_similarity":   avg_sim,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  Pretty-print results
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results: dict) -> None:
    W = 80
    SEP = "═" * W

    print(f"\n{SEP}")
    print(f"  QUESTION")
    print(SEP)
    print(f"  {results['question']}")

    print(f"\n{SEP}")
    print(f"  RAG ANSWER  (grounded in retrieved context)")
    print(SEP)
    print(results["rag_answer"])

    print(f"\n{SEP}")
    print(f"  NO-RAG ANSWER  (LLM parametric memory only)")
    print(SEP)
    print(results["no_rag_answer"])

    print(f"\n{SEP}")
    print(f"  RETRIEVED CHUNKS — SIMILARITY SCORES  "
          f"(avg: {results['avg_similarity']:.4f})")
    print(SEP)
    for c in results["retrieved_chunks"]:
        print(
            f"  #{c['rank']}  score={c['score']:.4f}  {c['label']}"
            f"  |  page {c['page']}  |  {Path(c['source']).name}"
        )
        print(f"       \"{c['preview']}…\"")
    print(SEP)


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  Ingest helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_path(
    path: Path,
    force_ocr: bool = False,
) -> tuple[FAISS, HuggingFaceEmbeddings, list[Path]]:
    """
    Ingest a single PDF or a directory of PDFs.

    Returns (vector_store, embeddings, list_of_pdf_paths)
    so the caller can pass them straight to `query_pipeline`.
    """
    # ── Collect PDFs ──────────────────────────────────────────────────────────
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        pdf_paths = [path]
    elif path.is_dir():
        pdf_paths = sorted(path.glob("**/*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in: {path}")
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    print(f"\nFound {len(pdf_paths)} PDF file(s) to ingest.\n")

    # ── OCR each PDF ──────────────────────────────────────────────────────────
    all_documents: list[Document] = []
    for pdf_path in pdf_paths:
        print(f"→ {pdf_path.name}")
        pages = ocr_pdf(pdf_path, force=force_ocr)
        docs  = pages_to_documents(pages, source=str(pdf_path))
        all_documents.extend(docs)
        print(f"  {len(docs)} non-empty page(s) added.\n")

    if not all_documents:
        raise ValueError("OCR produced no usable text from the provided PDF(s).")

    # ── Build / load FAISS index ──────────────────────────────────────────────
    embeddings   = get_embeddings()
    vector_store = build_or_load_vector_store(
        all_documents, embeddings, pdf_paths, force=force_ocr
    )

    return vector_store, embeddings, pdf_paths


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OCR pipeline for scanned PDFs with RAG / No-RAG comparison "
            "and cosine similarity scoring."
        )
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a scanned PDF file or directory of PDFs.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to ask. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Re-run OCR and rebuild the FAISS index even if caches exist.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K}).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the results as a JSON file.",
    )

    args = parser.parse_args()

    # Override globals from CLI flags
    global TOP_K
    TOP_K = args.top_k

    try:
        validate_config()

        # ── Ingest ────────────────────────────────────────────────────────────
        vector_store, embeddings, _ = ingest_path(
            Path(args.path), force_ocr=args.force_ocr
        )

        # ── Question ──────────────────────────────────────────────────────────
        question = args.query
        if not question:
            print("\nIngestion complete. Enter your question (Ctrl+C to exit).")
            question = input("Question: ").strip()
            if not question:
                print("No question provided — exiting.")
                sys.exit(0)

        # ── Query ─────────────────────────────────────────────────────────────
        print("\nRunning RAG and No-RAG pipelines …")
        results = query_pipeline(question, vector_store, embeddings)

        # ── Output ────────────────────────────────────────────────────────────
        print_results(results)

        if args.output_json:
            out = Path(args.output_json)
            out.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\nResults saved to: {out}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
