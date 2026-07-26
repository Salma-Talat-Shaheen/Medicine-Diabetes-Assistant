"""
PDF processing script via OCR and querying with RAG and No-RAG:
  1) Perform OCR on PDF file when needed and extract text.
  2) Split text and index into ChromaDB vector store.
  3) Provide query functions returning RAG, No-RAG answers, and similarity scores.
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
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )


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


def call_chat_model(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
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
    results_with_scores = vector_store.similarity_search_with_relevance_scores(
        question, k=top_k
    )

    if not results_with_scores:
        return "No relevant context found within the documents.", []

    context_parts = []
    scored_chunks = []
    for doc, score in results_with_scores:
        context_parts.append(doc.page_content)
        scored_chunks.append(
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "similarity_score": round(float(score), 4),
                "preview": doc.page_content[:200].replace("\n", " "),
            }
        )

    context = "\n\n---\n\n".join(context_parts)
    rag_prompt = (
        "Based strictly on the following provided medical context, answer the clinical question. "
        "If the answer is not contained in the context, explicitly state that it is not available in the provided guidelines.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    answer = call_chat_model(rag_prompt)
    return answer, scored_chunks


def ingest_path(pdf_path: str, ocr_lang: str = OCR_LANGUAGES):
    """Main ingestion function to process documents for the Web App"""
    documents = ocr_pdf_to_documents(pdf_path, ocr_lang=ocr_lang)
    
    # Split text for optimal retrieval performance
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    vector_store.add_documents(split_docs)
    return len(split_docs)


def query_pipeline(question: str, top_k: int = TOP_K):
    """Main query function for integration with Flask App"""
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    no_rag_ans = answer_without_rag(question)
    rag_ans, chunks = answer_with_rag(question, vector_store, top_k=top_k)

    return {
        "no_rag_answer": no_rag_ans,
        "rag_answer": rag_ans,
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
    args = parser.parse_args()

    validate_config()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        print(f"Error: File does not exist or is not a valid PDF: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting and indexing file: {pdf_path.name}")
    num_chunks = ingest_path(str(pdf_path), ocr_lang=args.ocr_lang)
    print(f"Successfully chunked and indexed {num_chunks} text segments.")

    question = args.question or input("\nEnter your question: ").strip()

    print("\n--- Answer without RAG ---")
    print(answer_without_rag(question))

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    rag_ans, chunks = answer_with_rag(question, vector_store, top_k=args.top_k)

    print("\n--- Answer with RAG ---")
    print(rag_ans)

    print("\n--- Retrieved Chunks & Similarity Scores ---")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. Score: {chunk['similarity_score']} | Page: {chunk['page']} | Source: {chunk['source']}")
        print(f"    Preview: {chunk['preview']}...\n")


if __name__ == "__main__":
    main()
