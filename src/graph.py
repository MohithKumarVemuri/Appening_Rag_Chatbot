import re
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]
    scores: List[float]
    answer: str
    score: float


SYSTEM_PROMPT = """You are a strict assistant answering questions about a document (the eBook).
The context below contains excerpts from that document. Phrases like "the eBook" or
"the document" refer to this context.

Rules:
- Use ONLY the context. Never use outside knowledge.
- If the context has relevant information, answer as completely as it supports, even if it is
  only a partial answer. Synthesize definitions from the excerpts when needed.
- Only if the context is unrelated to the question, reply exactly: "{refusal}"

Context:
{context}

Question: {question}

Answer:"""

_FILLER = re.compile(r"\b(according to|based on|as per|in|from)\s+the\s+(e-?book|document|pdf|text)\b[?.!]*", re.I)


def clean_query(q: str) -> str:
    """Drop phrases like 'according to the eBook' that add noise to the embedding."""
    return _FILLER.sub("", q).strip(" ?.,") or q


def build_rag_graph(index_name: str = config.PINECONE_INDEX_NAME):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL, output_dimensionality=config.EMBEDDING_DIM
    )
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL, temperature=0, timeout=30, max_retries=2
    )

    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(clean_query(state["question"]), k=config.TOP_K)
        return {
            "context": [doc.page_content for doc, _ in results],
            "scores": [float(s) for _, s in results],
        }

    def generate_node(state: AgentState):
        scores = state["scores"]
        top = max(scores) if scores else 0.0

        # Guardrail: nothing relevant retrieved -> refuse without calling the LLM
        if not state["context"] or top < config.MIN_RELEVANCE:
            return {"answer": config.REFUSAL, "score": round(top, 4)}

        prompt = SYSTEM_PROMPT.format(
            refusal=config.REFUSAL,
            context="\n\n---\n\n".join(state["context"]),
            question=state["question"],
        )
        content = llm.invoke(prompt).content
        if isinstance(content, list):  # some Gemini models return content blocks
            content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        answer = content.strip()

        # Confidence = top cosine similarity (0 if the model refused)
        confidence = 0.0 if config.REFUSAL.lower() in answer.lower() else top
        return {"answer": answer, "score": round(confidence, 4)}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()