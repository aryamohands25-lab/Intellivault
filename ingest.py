"""Phase 1: parse -> chunk -> embed -> store.

Usage:
    python ingest.py                 # index everything in ./data
    python ingest.py path/to/file.pdf
"""
import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (fitz is the legacy alias)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, get_collection, get_embedder

SUPPORTED = {".pdf", ".txt", ".md"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def read_pages(path: Path):
    """Yield (page_number, text). Text/markdown files count as one page."""
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    yield i, text
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def build_chunks(path: Path):
    ids, docs, metas = [], [], []
    for page_no, text in read_pages(path):
        for j, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{path.name}-p{page_no}-c{j}")
            docs.append(chunk)
            metas.append({"source": path.name, "page": page_no, "chunk": j})
    return ids, docs, metas


def ingest_file(path: Path) -> int:
    ids, docs, metas = build_chunks(path)
    if not docs:
        print(f"  ! no text found in {path.name} (scanned PDF? needs OCR)")
        return 0
    embeddings = get_embedder().encode(
        docs, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    ).tolist()
    get_collection().upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return len(docs)


def main():
    if len(sys.argv) > 1:
        files = [Path(p) for p in sys.argv[1:]]
    else:
        DATA_DIR.mkdir(exist_ok=True)
        files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in SUPPORTED)

    if not files:
        print(f"No files found. Put PDFs/notes in {DATA_DIR} and run again.")
        return

    total = 0
    for f in files:
        print(f"Indexing {f.name} ...")
        n = ingest_file(f)
        print(f"  -> {n} chunks")
        total += n
    print(f"\nDone. {total} chunks added. Collection size: {get_collection().count()}")


if __name__ == "__main__":
    main()
