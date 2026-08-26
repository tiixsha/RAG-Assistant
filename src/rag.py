"""
src/rag.py — RAG pipeline: ingestion, chunking, (later) retrieval.


"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Source corpus: voice/speaker embedding papers (identity preservation domain).
# Swapped out an unrelated OFDM channel-estimation paper that didn't fit the topic.
SOURCE_FILES = [
    "xvectors_speaker_recognition.pdf",
    "ecapa_tdnn_speaker_verification.pdf",
]


def load_documents(data_dir: Path = DATA_DIR, filenames: list[str] = SOURCE_FILES):
    """Load each PDF with PyPDFLoader. Returns a flat list of LangChain
    Document objects — one per PDF page, each carrying page-level metadata
    (source path, page number) by default."""
    all_docs = []
    for filename in filenames:
        pdf_path = data_dir / filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"Expected source PDF not found: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded {filename}: {len(docs)} pages")
    return all_docs


def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Chunk documents with RecursiveCharacterTextSplitter (fixed size + overlap).
    Metadata (source, page) is inherited automatically by each chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def verify_metadata(chunks):
    """Day 3 checkpoint: confirm every chunk carries populated source metadata."""
    missing = [i for i, c in enumerate(chunks) if not c.metadata.get("source")]
    if missing:
        raise ValueError(f"{len(missing)} chunks are missing 'source' metadata: {missing[:5]}...")
    sources = sorted({Path(c.metadata["source"]).name for c in chunks})
    print(f"\nMetadata check passed: all {len(chunks)} chunks have source + page metadata.")
    print(f"Sources present: {sources}")


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)
    verify_metadata(chunks)

    # Sanity print: show one chunk's structure so it's obvious what
    # goes into embedding on Day 4.
    sample = chunks[0]
    print("\n--- Sample chunk ---")
    print(f"Source: {Path(sample.metadata['source']).name}")
    print(f"Page:   {sample.metadata.get('page')}")
    print(f"Text (first 200 chars): {sample.page_content[:200]!r}")
    print(f"\nTotal chunks ready for embedding: {len(chunks)}")