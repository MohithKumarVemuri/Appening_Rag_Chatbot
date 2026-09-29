import streamlit as st
from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")


@st.cache_resource
def get_graph():
    return build_rag_graph()


st.title("Agentic AI eBook Chatbot")
query = st.chat_input("Ask about the Agentic AI eBook...")

if query:
    result = get_graph().invoke({"question": query})
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(result["answer"])
    with st.sidebar:
        st.metric("Confidence", f"{result['score']:.2f}")
        st.subheader("Retrieved chunks")
        for i, chunk in enumerate(result["context"], 1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)

# Run: streamlit run streamlit_app.py
