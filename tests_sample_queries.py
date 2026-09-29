import requests

URL = "http://127.0.0.1:8000/chat"
QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # should be refused
]

for q in QUERIES:
    r = requests.post(URL, json={"query": q}, timeout=120)
    if r.status_code != 200:
        print("ERROR", r.status_code, r.text[:400])
        continue
    data = r.json()
    print("=" * 80)
    print("Q:", q)
    print("A:", data["answer"])
    print("Confidence:", data["confidence_score"])
    print("Chunks retrieved:", len(data["retrieved_chunks"]))