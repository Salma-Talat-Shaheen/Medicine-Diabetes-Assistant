"""
PDF processing via OCR and querying with RAG and No-RAG:
  1) Perform OCR on a PDF file when needed and extract text (cached to disk).
  2) Split text and index into a ChromaDB vector store (cosine distance).
  3) Provide query functions returning RAG answer, No-RAG answer, and
     precise similarity scores.

Drop-in replacement: same public function names/signatures as before
(ingest_path, query_pipeline, answer_without_rag, answer_with_rag,
validate_config, get_embeddings, get_vector_store) so your existing Flask
routes that import this module keep working unchanged.

Fixes applied vs. the previous version:
  - Chroma now uses cosine distance explicitly (collection_metadata), instead
    of the default L2 space that Chroma's relevance-score normalization
    silently assumed was cosine-with-a-specific-range - causing negative,
    meaningless "similarity" scores with custom OpenRouter embeddings.
  - Retrieval now uses similarity_search_with_score (raw cosine distance)
    and converts it to similarity with the exact formula for cosine space:
    similarity = 1 - distance / 2. No more negative scores.
  - A relevance threshold (MAX_RELEVANT_DISTANCE) filters out chunks that
    are too dissimilar before building the RAG context, so the model isn't
    forced to reason over noise. Default raised to 0.9 after testing showed
    0.6 was too strict for this embedding model and rejected correct matches.
  - ingest_path() now skips re-indexing a PDF that is already present in the
    vector store (matched by source filename), preventing duplicate chunks
    from accumulating every time the same file is ingested again.
  - query_pipeline() now also returns a single overall_similarity_score
    (0-1) summarizing retrieval quality for the question, alongside the
    full per-chunk breakdown - useful for a simple UI indicator.
  - Added SharedSystemClient cache clearing to prevent 'Could not connect to
    tenant default_tenant' errors upon concurrent restarts or re-indexing.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from tqdm import tqdm

# Load environment variables
load_dotenv()

# ---- Core Environment Variables ----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medicine_docs")

OCR_CACHE_DIR = os.getenv("OCR_CACHE_DIR", "./ocr_cache")
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+ara")
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
MIN_TEXT_CHARS_PER_PAGE = int(os.getenv("MIN_TEXT_CHARS_PER_PAGE", "20"))

OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "4"))

# A cosine distance above this is treated as "not actually relevant" and
# excluded from the RAG context. 0.9 was chosen after testing: 0.6 was too
# strict and rejected genuinely correct matches for this embedding model.
MAX_RELEVANT_DISTANCE = float(os.getenv("MAX_RELEVANT_DISTANCE", "0.9"))


def validate_config() -> None:
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY is not defined in environment variables.", file=sys.stderr)

    if shutil.which("tesseract") is None:
        print(
            "Warning: tesseract was not found in the system PATH. "
            "If text is not cached, OCR processing may fail.",
            file=sys.stderr,
        )


def get_embeddings() -> OpenAIEmbeddings:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")

    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def get_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    # تفريغ الذاكرة المؤقتة لـ Chroma لمنع أخطاء الاتصال بـ default_tenant عند إعادة التحميل
    try:
        import chromadb.api.client
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        # Explicit cosine distance. Without this, Chroma defaults to L2 and
        # its built-in relevance-score normalization produces meaningless
        # (often negative) scores for these embeddings.
        collection_metadata={"hnsw:space": "cosine"},
    )


def cosine_distance_to_similarity(distance: float) -> float:
    """
    Chroma's cosine distance = 1 - cosine_similarity, ranging 0 (identical)
    to 2 (opposite). Exact inverse: similarity = 1 - distance / 2.
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def _ocr_cache_path(pdf_path: Path, ocr_lang: str) -> Path:
    cache_dir = Path(OCR_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = pdf_path.stat()
    key = f"{pdf_path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}::{ocr_lang}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{pdf_path.stem}.{digest}.ocr.json"


def ocr_pdf_to_documents(pdf_path: str, ocr_lang: str = OCR_LANGUAGES) -> list:
    pdf_file = Path(pdf_path)
    cache_path = _ocr_cache_path(pdf_file, ocr_lang)

    if cache_path.exists():
        print(f"Using pre-cached OCR data: {cache_path.name}")
        cached_pages = json.loads(cache_path.read_text(encoding="utf-8"))
        return [
            Document(
                page_content=p["text"],
                metadata={"source": str(pdf_file.name), "page": p["page"]},
            )
            for p in cached_pages
        ]

    print(f"Processing OCR on file: {pdf_file.name}")
    pdf_doc = fitz.open(str(pdf_file))
    zoom = OCR_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pages_data = []
    documents = []

    for page_num in tqdm(range(len(pdf_doc)), desc="OCR Progress", unit="page"):
        page = pdf_doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, lang=ocr_lang)

        pages_data.append({"page": page_num + 1, "text": text})
        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(pdf_file.name), "page": page_num + 1},
            )
        )

    pdf_doc.close()

    cache_path.write_text(
        json.dumps(pages_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return documents


def call_chat_model(prompt: str, temperature: float = 0.0) -> str:
    """temperature=0.0 for reproducible, precise answers."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def answer_without_rag(question: str) -> str:
    return call_chat_model(question)


def answer_with_rag(question: str, vector_store: Chroma, top_k: int = TOP_K):
    """
    Returns:
        answer: str
        scored_chunks: list of dicts, one per retrieved chunk, each with
            "similarity_score" (0-1, corrected cosine similarity - kept
            under this key name for frontend compatibility), "distance"
            (raw cosine distance), "used_in_context" (bool), "page",
            "source", "preview".
        overall_similarity_score: float, the best (highest) similarity
            among retrieved chunks - a single number summarizing how
            relevant the indexed document was to this question.
    """
    results = vector_store.similarity_search_with_score(question, k=top_k)

    if not results:
        return "No relevant context found within the documents.", [], 0.0

    scored_chunks = []
    context_parts = []
    for doc, distance in results:
        similarity = cosine_distance_to_similarity(distance)
        is_relevant = distance <= MAX_RELEVANT_DISTANCE
        scored_chunks.append(
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "distance": round(float(distance), 6),
                "similarity_score": round(similarity, 6),
                "used_in_context": is_relevant,
                "preview": doc.page_content[:200].replace("\n", " "),
            }
        )
        if is_relevant:
            context_parts.append(doc.page_content)

    overall_similarity_score = max(c["similarity_score"] for c in scored_chunks)

    if not context_parts:
        return (
            "No retrieved chunk was similar enough to the question "
            f"(all distances exceeded the {MAX_RELEVANT_DISTANCE} threshold), "
            "so no answer could be grounded in the documents.",
            scored_chunks,
            overall_similarity_score,
        )

    context = "\n\n---\n\n".join(context_parts)
    rag_prompt = (
        "Based strictly on the following provided medical context, answer the "
        "clinical question. Quote exact figures or values from the context "
        "where relevant. If the answer is not contained in the context, "
        "explicitly state that it is not available in the provided guidelines.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    answer = call_chat_model(rag_prompt)
    return answer, scored_chunks, overall_similarity_score


def ingest_path(pdf_path: str, ocr_lang: str = OCR_LANGUAGES) -> int:
    """
    Main ingestion function for the web app. Skips re-indexing if this exact
    PDF (by filename) is already present in the vector store, to avoid
    duplicate chunks accumulating on repeated uploads/calls.
    Returns the number of chunks actually added (0 if skipped).
    """
    pdf_file = Path(pdf_path)
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    existing = vector_store.get(where={"source": pdf_file.name})
    if existing and existing.get("ids"):
        print(f"Skipping ingest: '{pdf_file.name}' already indexed ({len(existing['ids'])} chunks).")
        return 0

    documents = ocr_pdf_to_documents(pdf_path, ocr_lang=ocr_lang)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)

    vector_store.add_documents(split_docs)
    return len(split_docs)


def query_pipeline(question: str, top_k: int = TOP_K) -> dict:
    """Main query function for integration with the Flask app."""
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    no_rag_ans = answer_without_rag(question)
    rag_ans, chunks, overall_similarity_score = answer_with_rag(
        question, vector_store, top_k=top_k
    )

    return {
        "no_rag_answer": no_rag_ans,
        "rag_answer": rag_ans,
        "overall_similarity_score": overall_similarity_score,
        "retrieved_chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process PDF with OCR and query with RAG vs No-RAG."
    )
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument("--question", type=str, default=None, help="Question to ask")
    parser.add_argument("--ocr-lang", type=str, default=OCR_LANGUAGES, help="Tesseract languages")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of retrieved chunks")
    # parse_known_args ignores extra args injected by notebook/Colab kernels
    args, _unknown = parser.parse_known_args()

    validate_config()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        print(f"Error: File does not exist or is not a valid PDF: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting and indexing file: {pdf_path.name}")
    num_chunks = ingest_path(str(pdf_path), ocr_lang=args.ocr_lang)
    print(f"Indexed {num_chunks} new text segment(s) (0 means it was already indexed).")

    question = args.question or input("\nEnter your question: ").strip()

    result = query_pipeline(question, top_k=args.top_k)

    print("\n--- Answer without RAG ---")
    print(result["no_rag_answer"])

    print("\n--- Answer with RAG ---")
    print(result["rag_answer"])

    print(f"\nOverall similarity score: {result['overall_similarity_score']:.4f} (0=not relevant, 1=identical match)")

    print("\n--- Retrieved Chunks & Similarity Scores ---")
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        flag = "used" if chunk["used_in_context"] else "filtered out"
        print(f"{i}. similarity={chunk['similarity_score']:.4f} distance={chunk['distance']:.4f} [{flag}] page={chunk['page']} source={chunk['source']}")
        print(f"    Preview: {chunk['preview']}...\n")


if __name__ == "__main__":
    main()
