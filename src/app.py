import os
import logging
import streamlit as st
from dotenv import load_dotenv
from rag_query_log import query_log  # must return (answer, source_docs)
from langchain_core.messages import HumanMessage  # only used to create RAG context if you need it

# ---- MUST be first Streamlit call ----
st.set_page_config(page_title="Query Observability Logs", layout="wide", page_icon=":mag_right:")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

st.title("Query Observability Logs")

# Helpers for consistent message shape
def human_msg(content: str) -> dict:
    return {"role": "human", "content": content}

def ai_msg(content: str, sources: list | None = None) -> dict:
    return {"role": "ai", "content": content, "sources": sources or []}

def serialize_sources(source_docs) -> list[dict]:
    """Turn LangChain Documents into simple dicts that survive reruns."""
    out = []
    for i, doc in enumerate(source_docs or [], start=1):
        meta = getattr(doc, "metadata", {}) or {}
        out.append({
            "label": meta.get("source", f"Source {i}"),
            "snippet": (getattr(doc, "page_content", "") or "")[:100],  # keep it short
            "metadata": meta
        })
    return out

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ai_msg("Hello! You are ready to query on the logs. Please enter your query below.")
    ]

# Render history
for msg in st.session_state.chat_history:
    if msg["role"] == "human":
        st.chat_message("Human").markdown(msg["content"])
    else:  # AI
        with st.chat_message("AI"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Source Documents"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s.get('label', 'Unknown')}**")
                        if s.get("snippet"):
                            st.code(s["snippet"])

# Input at bottom
user_query = st.chat_input("Type your message here...")

if user_query:
    # Append and show human message
    st.session_state.chat_history.append(human_msg(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Get AI answer
    with st.spinner("Querying..."):
        with st.chat_message("AI"):
            try:
                # Pass chat_history as context if your query_log uses it
                answer, source_docs = query_log(user_query, chat_history=st.session_state.chat_history)

                # Serialize sources to plain dicts before storing
                sources = serialize_sources(source_docs)

                # Append to history (this is what makes it persist on screen)
                st.session_state.chat_history.append(ai_msg(answer, sources))

                # Render immediately for this turn
                st.markdown(answer)
                if sources:
                    with st.expander("Source Documents"):
                        for s in sources:
                            st.markdown(f"- **{s.get('label', 'Unknown')}**")
                            if s.get("snippet"):
                                st.code(s["snippet"])

            except Exception as e:
                st.error(f"Error querying logs: {e}")
                logger.exception("Error querying logs")
