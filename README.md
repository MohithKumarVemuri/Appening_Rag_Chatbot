# RAG Chatbot: Agentic AI eBook

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **strictly from the Agentic AI eBook**. Built with **LangGraph**, **Pinecone**, **Google Gemini**, and **FastAPI / Streamlit**.

**Live demo:** `<https://mohithkumarvemuri-appening-rag-chatbot-streamlit-app-mvxas0.streamlit.app/>`

Every response returns:
1. The generated answer
2. The retrieved context chunks
3. A confidence score

## Architecture

```
PDF -> PyPDFLoader -> chunks (1000 / 200 overlap) -> Gemini embeddings (768-d)
   -> Pinecone (cosine)

Question -> LangGraph: START -> retrieve -> generate -> END
                         |           |
                  top-k chunks   grounded Gemini answer + confidence
```

| Component | Choice |
|---|---|
| Orchestration | LangGraph `StateGraph` (`retrieve` -> `generate`) |
| Vector DB | Pinecone serverless, 768 dims, cosine |
| Embeddings | `gemini-embedding-001` (768 dims) |
| LLM | Gemini (model set via `GEMINI_MODEL`) |
| Interfaces | FastAPI (`POST /chat`) and Streamlit UI |

## Project Structure

```
rag-agentic-ai/
├── data/                     # Ebook-Agentic-AI.pdf (auto-downloaded)
├── src/
│   ├── config.py             # Env vars and constants
│   ├── ingestion.py          # PDF load, chunk, embed, upsert to Pinecone
│   └── graph.py              # LangGraph workflow and grounding logic
├── app.py                    # FastAPI backend
├── streamlit_app.py          # Streamlit chat UI
├── check_index.py            # Diagnostic: index stats and sample scores
├── tests_sample_queries.py   # 5 benchmark queries against the API
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

Requires Python 3.10+ (developed on 3.11).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Create your `.env` from the template and fill in your keys:

```bash
cp .env.example .env           # Windows: copy .env.example .env
```

```env
GOOGLE_API_KEY=your_gemini_api_key        # from aistudio.google.com
PINECONE_API_KEY=your_pinecone_api_key    # from pinecone.io
PINECONE_INDEX_NAME=agentic-ai-index
GEMINI_MODEL=gemini-2.5-flash             # use a model ID available to your key
```

## 1. Ingest (one time)

Downloads the PDF to `data/Ebook-Agentic-AI.pdf` (or place it there manually), creates the Pinecone index if needed, and upserts the chunks.

```bash
python -m src.ingestion
```

Expected output: 60 pages, 119 chunks, then an index with 119 vectors.

Notes:
- If an existing index has a different dimension, it is deleted and recreated at 768.
- Upserts run in small batches with pauses and retries to stay under the Gemini free-tier limit (about 100 embedding requests per minute). Ingestion takes roughly 2 minutes.

## 2. Run

**FastAPI**
```bash
uvicorn app:app --reload       # docs at http://127.0.0.1:8000/docs
```

**Streamlit**
```bash
streamlit run streamlit_app.py # http://localhost:8501
```

The Streamlit sidebar shows the confidence score and the retrieved chunks.

## 3. Test

With the API running:

```bash
python tests_sample_queries.py
```

Benchmark queries:
1. What is Agentic AI according to the eBook?
2. How do AI agents differ from traditional automation systems?
3. What are the core components of an Agentic Architecture?
4. What role does memory play in Agentic AI workflows?
5. Who won the 2022 FIFA World Cup? (out-of-scope, should be refused)

### Sample results

| Query | Result | Confidence |
|---|---|---|
| 1. What is Agentic AI | Grounded answer | ~0.78 |
| 2. Agents vs automation | Grounded answer | ~0.75 |
| 3. Core components | Grounded answer | ~0.79 |
| 4. Role of memory | Grounded answer | ~0.77 |
| 5. FIFA World Cup 2022 | "I cannot answer based on the provided document." | 0.0 |

## API

`POST /chat`

```json
{ "query": "What role does memory play in Agentic AI workflows?" }
```

Response:

```json
{
  "answer": "Memory plays a crucial role...",
  "retrieved_chunks": ["...", "..."],
  "confidence_score": 0.77
}
```

`GET /health` returns `{"status": "ok"}`. LLM or retrieval failures return a 502 with the error message.

## Design Notes

- **Strict grounding:** the prompt restricts the model to the retrieved context and defines an exact refusal string. If the top similarity score is below `MIN_RELEVANCE` (0.25), the LLM is skipped and the bot refuses.
- **Confidence score:** the top-1 cosine similarity from Pinecone. It is 0.0 when the bot refuses.
- **Query cleaning:** phrases such as "according to the eBook" are stripped before embedding, since they add noise to retrieval.
- **Retrieval depth:** `TOP_K = 6` chunks per question.
- **Partial answers:** the prompt allows answers supported by partial context and refuses only when the context is unrelated.
- **Resilience:** LLM calls use a 30s timeout with limited retries. Ingestion batches and retries embedding calls.

Tunable settings (`TOP_K`, `MIN_RELEVANCE`, chunk size, models) are in `src/config.py`.

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (keep `.env` out of git).
2. On share.streamlit.io, create a new app with `streamlit_app.py` as the main file.
3. In Advanced settings, choose Python 3.11 and add secrets:

```toml
GOOGLE_API_KEY = "..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "agentic-ai-index"
GEMINI_MODEL = "gemini-2.5-flash"
```

The Pinecone index must already be populated (run ingestion locally first).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Vector dimension 768 does not match the index` | Re-run `python -m src.ingestion` (recreates the index at 768) |
| `429 RESOURCE_EXHAUSTED` | Gemini free-tier rate limit; wait a minute and retry |
| Every answer is a refusal, 0 chunks | Index is empty; run ingestion, then `python check_index.py` |
| Slow or timed-out `/chat` | Check that `GEMINI_MODEL` is a valid model ID for your key |
