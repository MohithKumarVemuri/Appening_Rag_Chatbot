import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
PDF_DRIVE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768  # Pinecone index dimension must match this
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6
# Below this top cosine similarity, treat the question as out-of-scope
MIN_RELEVANCE = 0.25

REFUSAL = "I cannot answer based on the provided document."

for name, val in [("GOOGLE_API_KEY", GOOGLE_API_KEY), ("PINECONE_API_KEY", PINECONE_API_KEY)]:
    if not val:
        raise RuntimeError(f"Missing {name}. Copy .env.example to .env and fill it in.")