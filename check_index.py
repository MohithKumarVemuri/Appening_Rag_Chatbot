from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src import config

pc = Pinecone(api_key=config.PINECONE_API_KEY)
print("Indexes:", pc.list_indexes().names())
idx = pc.Index(config.PINECONE_INDEX_NAME)
print("Stats:", idx.describe_index_stats())

emb = GoogleGenerativeAIEmbeddings(
    model=config.EMBEDDING_MODEL, output_dimensionality=config.EMBEDDING_DIM
)
store = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=emb)
for q in ["What is Agentic AI?", "Who won the 2022 FIFA World Cup?"]:
    res = store.similarity_search_with_score(q, k=3)
    print(f"\nQ: {q} -> {len(res)} results")
    for d, s in res:
        print(f"  {s:.3f} | {d.page_content[:80]!r}")