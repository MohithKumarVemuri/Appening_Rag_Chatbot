# RAG Chatbot: Agentic AI eBook

Answers questions strictly from the Agentic AI eBook using LangGraph + Pinecone + Gemini.

**Flow:** PDF -> chunks (1000/200) -> Gemini embeddings -> Pinecone -> LangGraph (`retrieve -> generate`) -> FastAPI / Streamlit.

## Setup
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add GOOGLE_API_KEY and PINECONE_API_KEY
```

## 1. Ingest (one time)
Auto-downloads the PDF to `data/Ebook-Agentic-AI.pdf` (or place it there manually), creates the Pinecone index (768 dims, cosine) and upserts chunks.
```bash
python -m src.ingestion
```

## 2. Run
```bash
uvicorn app:app --reload          # API at http://127.0.0.1:8000/docs
streamlit run streamlit_app.py    # optional UI
```

## 3. Test
```bash
python tests_sample_queries.py
```

## API
`POST /chat` with `{"query": "..."}` returns:
```json
{"answer": "...", "retrieved_chunks": ["..."], "confidence_score": 0.62}
```

## Design notes
- **Grounding:** strict prompt + refusal string; if top similarity < `MIN_RELEVANCE` (0.25) the LLM is skipped and the bot refuses.
- **Confidence score:** top-1 cosine similarity from Pinecone (0 when refused). Real retrieval signal, not a fixed number.
- **Tuning:** `TOP_K`, `MIN_RELEVANCE`, chunk sizes live in `src/config.py`.
