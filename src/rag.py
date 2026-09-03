"""
src/rag.py — RAG pipeline: ingestion, chunking, retrieval.
"""

import sys
from pathlib import Path
import re
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import ChatOpenAI 

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCE_FILES = [
    "xvectors_speaker_recognition.pdf",
    "ecapa_tdnn_speaker_verification.pdf",
]

REFERENCES_HEADING = re.compile(
    r"(?:^|\n)\s*\d*\.?\s*(references|bibliography)\s*\n",
    re.IGNORECASE
)

COLLECTION_NAME = "speaker_embedding_papers"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")

local_llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key="not-needed",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    temperature=0.7,
)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore(chunks, recreate: bool = False):
    embeddings = get_embeddings()
    client = QdrantClient(url=QDRANT_URL)

    if recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    vectorstore = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    print(
        f"Upserted {len(chunks)} chunks into "
        f"Qdrant collection '{COLLECTION_NAME}'."
    )

    return vectorstore


def get_retriever(vectorstore, k: int = 3):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def strip_references(documents):
    cleaned = []

    from itertools import groupby

    for source, pages in groupby(
        documents,
        key=lambda d: d.metadata["source"]
    ):
        pages = list(pages)
        truncated_here = False

        for doc in pages:
            if truncated_here:
                continue

            match = REFERENCES_HEADING.search(doc.page_content)

            if match:
                doc.page_content = doc.page_content[:match.start()]
                truncated_here = True
                print(
                    f"Stripped references from {Path(source).name} "
                    f"at page {doc.metadata.get('page')}"
                )

            cleaned.append(doc)

    return cleaned


def load_documents(
    data_dir: Path = DATA_DIR,
    filenames: list[str] = SOURCE_FILES
):
    all_docs = []

    for filename in filenames:
        pdf_path = data_dir / filename

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Expected source PDF not found: {pdf_path}"
            )

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)

        print(f"Loaded {filename}: {len(docs)} pages")

    return strip_references(all_docs)


def chunk_documents(
    documents,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


def verify_metadata(chunks):
    missing = [
        i
        for i, chunk in enumerate(chunks)
        if not chunk.metadata.get("source")
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} chunks are missing 'source' metadata: "
            f"{missing[:5]}..."
        )

    sources = sorted({
        Path(chunk.metadata["source"]).name
        for chunk in chunks
    })

    print(
        f"\nMetadata check passed: all {len(chunks)} "
        f"chunks have source + page metadata."
    )
    print(f"Sources present: {sources}")


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)
    verify_metadata(chunks)

    sample = chunks[0]

    print("\n--- Sample chunk ---")
    print(f"Source: {Path(sample.metadata['source']).name}")
    print(f"Page:   {sample.metadata.get('page')}")
    print(f"Text (first 200 chars): {sample.page_content[:200]!r}")
    print(f"\nTotal chunks ready for embedding: {len(chunks)}")

    vectorstore = build_vectorstore(chunks, recreate=True)
    retriever = get_retriever(vectorstore)

    test_queries = [
        "How does the x-vector system extract speaker embeddings?",
        "What is the ECAPA-TDNN architecture?",
        "How is speaker verification evaluated in these papers?",
    ]

    for query in test_queries:
        print(f"\n--- Query: {query!r} ---")

        results = retriever.invoke(query)

        for i, doc in enumerate(results):
            src = Path(doc.metadata["source"]).name
            page = doc.metadata.get("page")

            print(
                f"  [{i}] {src} (p.{page}): "
                f"{doc.page_content[:120]!r}"
            )