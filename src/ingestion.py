import os
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config

BATCH_SIZE = 20          # Gemini free tier: ~100 embed requests/min
PAUSE_BETWEEN_BATCHES = 15  # seconds


def download_pdf(path: str = config.PDF_PATH) -> None:
    if os.path.exists(path):
        return
    import gdown
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gdown.download(id=config.PDF_DRIVE_ID, output=path, quiet=False)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Download failed. Save the PDF manually to {path}")


def ensure_index(index_name: str) -> None:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if index_name in pc.list_indexes().names():
        dim = pc.describe_index(index_name).dimension
        if dim == config.EMBEDDING_DIM:
            return
        print(f"Index dimension {dim} != {config.EMBEDDING_DIM}; recreating index")
        pc.delete_index(index_name)
        time.sleep(5)
    pc.create_index(
        name=index_name,
        dimension=config.EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


def run_ingestion(pdf_path: str = config.PDF_PATH, index_name: str = config.PINECONE_INDEX_NAME):
    download_pdf(pdf_path)
    docs = PyPDFLoader(pdf_path).load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    print(f"Created {len(chunks)} chunks")
    if not chunks:
        raise RuntimeError("No text extracted. The PDF may be scanned images (needs OCR).")

    ensure_index(index_name)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL, output_dimensionality=config.EMBEDDING_DIM
    )
    store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        for attempt in range(6):
            try:
                store.add_documents(batch)
                break
            except Exception as e:
                wait = 20 + attempt * 15
                print(f"  batch {i} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Batch starting at {i} failed after retries")
        print(f"  upserted {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")
        time.sleep(PAUSE_BETWEEN_BATCHES)

    time.sleep(5)  # Pinecone stats are eventually consistent
    stats = Pinecone(api_key=config.PINECONE_API_KEY).Index(index_name).describe_index_stats()
    print("Index stats:", stats)
    return store


if __name__ == "__main__":
    run_ingestion()