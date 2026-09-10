import os
import streamlit as st

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from generate import generate_answer
from retrieve import retrieve_passages


import streamlit as st
from generate import generate_answer
from retrieve import retrieve_passages

from setup_vectorstore import ensure_vectorstore

@st.cache_resource
def setup():
    ensure_vectorstore()
    return True

setup()

from generate import generate_answer
from retrieve import retrieve_passages

st.set_page_config(page_title="SQuAD RAG Chatbot", page_icon="🔎", layout="centered")

st.markdown("""
<style>
.stChatMessage {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&display=swap');

h1 {
    font-family: 'Poppins', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🔎 SQuAD RAG Chatbot")
st.caption("Ask a question and I'll answer using retrieved Wikipedia passages, grounded and cited.")

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander("View retrieved passages"):
            for i, passage in enumerate(entry["passages"]):
                st.markdown(f"**Passage {i + 1}**")
                st.write(passage)

question = st.chat_input("Ask a question")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages and generating answer..."):
            passages = retrieve_passages(question, k=3)
            answer = generate_answer(question, k=3)

        st.write(answer)
        with st.expander("View retrieved passages"):
            for i, passage in enumerate(passages):
                st.markdown(f"**Passage {i + 1}**")
                st.write(passage)

    st.session_state.history.append({
        "question": question,
        "answer": answer,
        "passages": passages
    })
