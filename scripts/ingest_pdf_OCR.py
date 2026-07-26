"""
ingest_pdf_OCR.py  —  Production-safe OCR-RAG pipeline for Render.com
======================================================================

Root cause of the "re-OCR every restart" bug
---------------------------------------------
Render free-tier containers use an ephemeral filesystem: every restart wipes
./ocr_cache/ and ./chroma_db/ unless they are on a mounted Persistent Disk.
The old code cached OCR to disk and checked disk for "already indexed" —
both failed on restart, causing 8-minute OCR re-runs on every boot.

Fix: collection.count() as the single gatekeeper
-------------------------------------------------
Before running OCR we call collection.count().
  > 0  → already indexed, skip OCR entirely (fast path, ~1 second).
  == 0 → run OCR once, index, persist.

For this to survive Render restarts you MUST set CHROMA_PERSIST_DIRECTORY
to a path on a mounted Persistent Disk (e.g. /data/chroma_db).
If you use the default ./chroma_db on the ephemeral disk, OCR will
re-run on every restart — that is a deployment config issue, not a code bug.

_type Chroma error fix
-----------------------
Drop + recreate the affected collection instead of wiping the whole directory.

Public API (drop-in compatible with app.py)
-------------------------------------------
  validate_config()
  get_embeddings()
  get_vector_store()
  ingest_path(pdf_path, ocr_lang) -> int
  query_pipeline(question, top_k)  -> dict
  answer_without_rag(question)     -> str
  answer_with_rag(question, vs, k) -> (str, list, float)
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
from pathlib import Path

import fitz
import pytesseract
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from tqdm import tqdm

load_dotenv()

# ── Environment variables ────────────────────────────────────────────────────
OPENROUTER_API_KEY       = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL      = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME          = os.getenv("COLLECTION_NAME", "medicine_docs")
OPENROUTER_CHAT_MODEL    = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
TOP_K                    = int(os.getenv("TOP_K", "4"))
MAX_RELEVANT_DISTANCE    = float(os.getenv("MAX_RELEVANT_DISTANCE", "0.9"))
OCR_LANGUAGES            = os.getenv("OCR_LANGUAGES", "eng+ara")
OCR_DPI                  = int(os.getenv("OCR_DPI", "300"))


# === 1  Config ================================================================

def validate_config() -> None:
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY is not set.", file=sys.stderr)
    if shutil.which("tesseract") is None:
        print("Warning: tesseract not found on PATH.", file=sys.stderr)


# === 2  Embeddings ============================================================

def get_embeddings() -> OpenAIEmbeddings:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


# === 3  Chroma vector store ===================================================

def _clear_chroma_cache() -> None:
    try:
        import chromadb.api.client
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def get_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    """
    Build a Chroma store with cosine distance.
    If the _type error is detected, drop and recreate the collection.
    """
    _clear_chroma_cache()

    def _build() -> Chroma:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            collection_metadata={"hnsw:space": "cosine"},
        )

    try:
        store = _build()
        store._collection.count()   # surface any latent _type error now
        return store
    except Exception as exc:
        if "_type" in str(exc):
            print(f"[Chroma] Stale metadata detected ({exc}). Dropping collection...", flush=True)
            _clear_chroma_cache()
            try:
                import chromadb
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
                client.delete_collection(COLLECTION_NAME)
                print("[Chroma] Collection dropped. Rebuilding...", flush=True)
            except Exception as del_exc:
                print(f"[Chroma] Drop failed ({del_exc}). Wiping directory.", flush=True)
                shutil.rmtree(CHROMA_PERSIST_DIRECTORY, ignore_errors=True)
            _clear_chroma_cache()
            return _build()
        raise


def cosine_distance_to_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


# === 4  OCR ===================================================================

def _run_tesseract_ocr(pdf_path: Path, ocr_lang: str) -> list[Document]:
    """Rasterise PDF pages and run Tesseract. No local file cache."""
    print(f"[OCR] Starting Tesseract on: {pdf_path.name}", flush=True)
    pdf_doc = fitz.open(str(pdf_path))
    zoom    = OCR_DPI / 72.0
    matrix  = fitz.Matrix(zoom, zoom)
    docs: list[Document] = []

    for page_num in tqdm(range(len(pdf_doc)), desc="OCR Progress", unit="page"):
        page  = pdf_doc[page_num]
        pix   = page.get_pixmap(matrix=matrix)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text  = pytesseract.image_to_string(image, lang=ocr_lang)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path.name, "page": page_num + 1},
            )
        )

    pdf_doc.close()
    print(f"[OCR] Done — {len(docs)} pages extracted.", flush=True)
    return docs


# === 5  ingest_path  (idempotent, Chroma-as-truth) ===========================

def ingest_path(pdf_path: str, ocr_lang: str = OCR_LANGUAGES) -> int:
    """
    Index a scanned PDF into Chroma.

    SINGLE GATEKEEPER: collection.count()
      > 0  →  skip OCR, return 0.
      == 0 →  run OCR, index, persist.

    This is the only place where "already indexed?" is decided.
    No filesystem checks. No metadata hash checks.
    Chroma persistence is the guarantee — configure CHROMA_PERSIST_DIRECTORY
    to a Render Persistent Disk path for this to work across restarts.
    """
    pdf_file   = Path(pdf_path)
    embeddings = get_embeddings()
    vs         = get_vector_store(embeddings)

    # ── Single gatekeeper ────────────────────────────────────────────────────
    try:
        count = vs._collection.count()
    except Exception:
        count = 0

    if count > 0:
        print(
            f"[OCR] Collection already has {count} chunks. Skipping OCR.",
            flush=True,
        )
        return 0

    # ── First-time indexing ──────────────────────────────────────────────────
    documents  = _run_tesseract_ocr(pdf_file, ocr_lang=ocr_lang)
    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(documents)

    print(f"[OCR] Indexing {len(split_docs)} chunks...", flush=True)
    vs.add_documents(split_docs)
    print(f"[OCR] Indexed {len(split_docs)} chunks successfully.", flush=True)
    return len(split_docs)


# === 6  LLM ==================================================================

def call_chat_model(prompt: str, temperature: float = 0.0) -> str:
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model":       OPENROUTER_CHAT_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def answer_without_rag(question: str) -> str:
    return call_chat_model(question)


def answer_with_rag(
    question: str,
    vector_store: Chroma,
    top_k: int = TOP_K,
) -> tuple[str, list[dict], float]:
    raw = vector_store.similarity_search_with_score(question, k=top_k)
    if not raw:
        return "No relevant context found.", [], 0.0

    scored_chunks: list[dict] = []
    context_parts: list[str]  = []

    for doc, distance in raw:
        sim         = cosine_distance_to_similarity(distance)
        is_relevant = distance <= MAX_RELEVANT_DISTANCE
        scored_chunks.append({
            "source":           doc.metadata.get("source", "Unknown"),
            "page":             doc.metadata.get("page", 1),
            "distance":         round(float(distance), 6),
            "similarity_score": round(sim, 6),
            "used_in_context":  is_relevant,
            "preview":          doc.page_content[:200].replace("\n", " "),
        })
        if is_relevant:
            context_parts.append(doc.page_content)

    overall = max(c["similarity_score"] for c in scored_chunks)

    if not context_parts:
        return (
            f"No chunk was close enough (all distances > {MAX_RELEVANT_DISTANCE}). "
            "Cannot provide a guideline-grounded answer.",
            scored_chunks,
            overall,
        )

    context = "\n\n---\n\n".join(context_parts)
    prompt  = (
        "Based strictly on the following medical context, answer the clinical "
        "question. Quote exact figures where relevant. If the answer is not in "
        "the context, state that explicitly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return call_chat_model(prompt), scored_chunks, overall


# === 7  query_pipeline  (called by Flask) =====================================

def query_pipeline(question: str, top_k: int = TOP_K) -> dict:
    embeddings   = get_embeddings()
    vector_store = get_vector_store(embeddings)
    no_rag_ans   = answer_without_rag(question)
    rag_ans, chunks, overall = answer_with_rag(question, vector_store, top_k=top_k)
    return {
        "no_rag_answer":            no_rag_ans,
        "rag_answer":               rag_ans,
        "overall_similarity_score": overall,
        "retrieved_chunks":         chunks,
    }


# === 8  CLI ==================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="OCR-RAG pipeline for scanned PDFs.")
    parser.add_argument("pdf_path")
    parser.add_argument("--question", default=None)
    parser.add_argument("--ocr-lang", default=OCR_LANGUAGES)
    parser.add_argument("--top-k",    type=int, default=TOP_K)
    args, _ = parser.parse_known_args()

    validate_config()

    pdf = Path(args.pdf_path)
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        print(f"Error: {pdf}", file=sys.stderr); sys.exit(1)

    n = ingest_path(str(pdf), ocr_lang=args.ocr_lang)
    print(f"Indexed {n} new chunk(s).")

    q = args.question or input("\nQuestion: ").strip()
    r = query_pipeline(q, top_k=args.top_k)

    print("\n--- No-RAG ---"); print(r["no_rag_answer"])
    print("\n--- RAG ---");    print(r["rag_answer"])
    print(f"\nOverall similarity: {r['overall_similarity_score']:.4f}")
    for i, c in enumerate(r["retrieved_chunks"], 1):
        flag = "used" if c["used_in_context"] else "filtered"
        print(f"  {i}. sim={c['similarity_score']:.4f} [{flag}] p={c['page']} {c['preview'][:80]}...")


if __name__ == "__main__":
    main()
