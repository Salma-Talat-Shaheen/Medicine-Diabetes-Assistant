
!/usr/bin/env python
"""Script to ingest PDF files into Chroma vector store using OpenRouter embeddings."""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF - used to rasterize pages for OCR
import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medicine_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))

# OCR configuration (used for scanned PDFs with no text layer)
OCR_CACHE_DIR = os.getenv("OCR_CACHE_DIR", "./ocr_cache")
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+ara")
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
# If the average extracted characters per page falls below this, the PDF is
# treated as scanned (image-only) and routed through OCR instead.
MIN_TEXT_CHARS_PER_PAGE = int(os.getenv("MIN_TEXT_CHARS_PER_PAGE", "20"))


def validate_config() -> None:
    """Validate that required configuration is set."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )
    if shutil.which("tesseract") is None:
        print(
            " Warning: the 'tesseract' binary was not found on PATH. "
            "OCR fallback for scanned PDFs will fail until it's installed "
            "(e.g. `sudo apt-get install tesseract-ocr tesseract-ocr-ara`).",
            file=sys.stderr,
        )


def get_embeddings() -> OpenAIEmbeddings:
    """Create and return OpenAI embeddings configured for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set")
    
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=lambda: OPENROUTER_API_KEY,  # type: ignore
        base_url=OPENROUTER_BASE_URL,
    )


def get_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    """Get or create the Chroma vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )


def add_documents_concurrently(
    vector_store: Chroma, documents: list, max_workers: int, batch_size: int
) -> int:
    """
    Add documents to the vector store in parallel using a thread pool.

    Args:
        vector_store: The Chroma vector store instance.
        documents: A list of documents to add.
        max_workers: The maximum number of concurrent threads.
        batch_size: The number of documents to process in each thread.

    Returns:
        The total number of chunks added.
    """
    total_chunks = len(documents)
    # Split documents into batches
    batches = [
        documents[i : i + batch_size] for i in range(0, total_chunks, batch_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(
            f"Starting concurrent ingestion with {max_workers} workers and {len(batches)} batches..."
        )
        # Create a future for each batch
        futures = [
            executor.submit(vector_store.add_documents, batch) for batch in batches
        ]

        # Use tqdm for a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ingesting Batches"):
            future.result()  # Wait for the batch to complete and raise any exceptions

    return total_chunks


def needs_ocr(documents: list) -> bool:
    """
    Heuristic to detect scanned PDFs.

    If a PDF has no real text layer (i.e. every page is a picture of a page,
    like a scanned book), PyPDFLoader will still "succeed" but return pages
    with empty or near-empty text. We flag that here so it can be routed
    through OCR instead of being silently ingested as blank chunks.
    """
    if not documents:
        return True
    total_chars = sum(len(doc.page_content.strip()) for doc in documents)
    avg_chars_per_page = total_chars / len(documents)
    return avg_chars_per_page < MIN_TEXT_CHARS_PER_PAGE


def _ocr_cache_path(pdf_path: Path, ocr_lang: str) -> Path:
    """Build a stable cache file path for a given PDF's OCR output.

    The key includes file size + modified time + language, so if the source
    PDF or the OCR language changes, the cache is naturally invalidated and
    OCR reruns; otherwise we reuse the saved text forever.
    """
    cache_dir = Path(OCR_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = pdf_path.stat()
    key = f"{pdf_path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}::{ocr_lang}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{pdf_path.stem}.{digest}.ocr.json"


def ocr_pdf_to_documents(pdf_path: str, ocr_lang: str = OCR_LANGUAGES) -> list:
    """
    Extract text from a scanned PDF by rasterizing each page and running
    Tesseract OCR on it. Results are cached to disk (OCR_CACHE_DIR) as JSON
    so the expensive OCR pass only ever runs once per PDF file/language.

    Returns a list of langchain Document objects, one per page, with
    metadata identical in shape to what PyPDFLoader produces (source, page).
    """
    pdf_file = Path(pdf_path)
    cache_path = _ocr_cache_path(pdf_file, ocr_lang)

    if cache_path.exists():
        print(f"  ↺ Found cached OCR text, skipping re-OCR: {cache_path.name}")
        cached_pages = json.loads(cache_path.read_text(encoding="utf-8"))
        return [
            Document(
                page_content=p["text"],
                metadata={"source": str(pdf_file), "page": p["page"]},
            )
            for p in cached_pages
        ]

    print(f"  No usable text layer found — running OCR ({ocr_lang}) on {pdf_file.name}")
    pdf_doc = fitz.open(str(pdf_file))
    zoom = OCR_DPI / 72.0  # PyMuPDF renders at 72 DPI by default
    matrix = fitz.Matrix(zoom, zoom)

    pages_data = []
    documents = []

    for page_num in tqdm(range(len(pdf_doc)), desc="  OCR pages", unit="page"):
        page = pdf_doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, lang=ocr_lang)

        pages_data.append({"page": page_num + 1, "text": text})
        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(pdf_file), "page": page_num + 1},
            )
        )

    pdf_doc.close()

    # Persist the extracted text once, so future runs (or re-ingests) never
    # need to OCR this file again.
    cache_path.write_text(
        json.dumps(pages_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  OCR text cached to: {cache_path}")

    return documents


def load_pdf_documents(
    pdf_path: str, force_ocr: bool = False, ocr_lang: str = OCR_LANGUAGES
) -> list:
    """
    Load a PDF's page-level text, automatically falling back to OCR when the
    PDF has no (or a negligible) extractable text layer — i.e. it's a
    scanned document rather than a "digital" PDF.
    """
    pdf_file = Path(pdf_path)

    if not force_ocr:
        documents = PyPDFLoader(str(pdf_file)).load()
        if not needs_ocr(documents):
            print(f"  ✓ Loaded {len(documents)} pages (native text layer)")
            return documents
        print("   Little to no extractable text detected — treating as a scanned PDF")
    else:
        print("  --force-ocr set — skipping native text extraction")

    return ocr_pdf_to_documents(str(pdf_file), ocr_lang=ocr_lang)


def ingest_pdf(
    pdf_path: str,
    vector_store: Chroma,
    force_ocr: bool = False,
    ocr_lang: str = OCR_LANGUAGES,
) -> None:
    """
    Ingest a single PDF file into the vector store.

    Args:
        pdf_path: Path to the PDF file to ingest.
        vector_store: The Chroma vector store instance.
        force_ocr: If True, always OCR the PDF instead of trying to reuse
            an existing text layer.
        ocr_lang: Tesseract language string to use if OCR is triggered.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the file is not a PDF.
    """
    pdf_file = Path(pdf_path)

    print(f"Loading PDF: {pdf_file.name}")

    # Load PDF documents (falls back to OCR automatically for scanned PDFs)
    documents = load_pdf_documents(str(pdf_file), force_ocr=force_ocr, ocr_lang=ocr_lang)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = text_splitter.split_documents(documents)
    print(f"  ✓ Split into {len(splits)} chunks")

    # Add to vector store concurrently
    add_documents_concurrently(
        vector_store, splits, max_workers=MAX_WORKERS, batch_size=INGEST_BATCH_SIZE
    )
    print(f"Successfully ingested {pdf_file.name}")


def ingest_directory(
    directory_path: str,
    vector_store: Chroma,
    force_ocr: bool = False,
    ocr_lang: str = OCR_LANGUAGES,
) -> None:
    """
    Ingest all PDF files from a directory into the vector store.

    Args:
        directory_path: Path to the directory containing PDF files.
        vector_store: The Chroma vector store instance.
        force_ocr: If True, always OCR every PDF instead of trying to reuse
            an existing text layer.
        ocr_lang: Tesseract language string to use if OCR is triggered.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        ValueError: If the directory contains no PDF files.
    """
    directory = Path(directory_path)

    # Validate directory exists
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Find all PDF files
    pdf_files = list(directory.glob("**/*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {directory_path}")

    print(f"Found {len(pdf_files)} PDF file(s)")

    # Setup text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_splits = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        documents = load_pdf_documents(
            str(pdf_file), force_ocr=force_ocr, ocr_lang=ocr_lang
        )
        splits = text_splitter.split_documents(documents)
        all_splits.extend(splits)
        print(f"  ✓ Split into {len(splits)} chunks")

    print("\n---")
    print(f"Total documents to ingest: {len(all_splits)}")
    total_chunks = add_documents_concurrently(
        vector_store, all_splits, max_workers=MAX_WORKERS, batch_size=INGEST_BATCH_SIZE
    )
    print(f"\nSuccessfully ingested {total_chunks} chunks from directory.")


def main() -> None:
    """Main entry point for the ingest script."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into Chroma vector store using OpenRouter embeddings"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a PDF file or directory containing PDF files",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=f"Custom Chroma database path (default: {CHROMA_PERSIST_DIRECTORY})",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Always OCR PDFs, even if a native text layer is detected "
        "(useful if a PDF has a garbled or partial text layer).",
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        default=None,
        help=f"Tesseract language(s) to use for OCR, e.g. 'eng+ara' "
        f"(default: {OCR_LANGUAGES})",
    )

    args = parser.parse_args()
    ocr_lang = args.ocr_lang or OCR_LANGUAGES

    try:
        validate_config()
        path = Path(args.path)

        # Centralize setup of embeddings and vector store
        embeddings = get_embeddings()
        db_path = args.db_path or CHROMA_PERSIST_DIRECTORY
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=db_path,
        )

        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"File is not a PDF: {path}")
            ingest_pdf(
                str(path), vector_store, force_ocr=args.force_ocr, ocr_lang=ocr_lang
            )
        elif path.is_dir():
            ingest_directory(
                str(path), vector_store, force_ocr=args.force_ocr, ocr_lang=ocr_lang
            )
        else:
            print(f"Error: Path does not exist: {args.path}", file=sys.stderr)
            sys.exit(1)

        # Persist and print final summary
        print("\n---")
        print("Ingestion complete.")
        print(f"Vector store persisted to: {os.path.abspath(db_path)}")

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
