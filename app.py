import asyncio
import sys
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.storage import InMemoryStore
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any
import json
import re

# Fix for "no current event loop" issue in Streamlit
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------- CONFIG ----------------
MODEL_NAME = "gemini-1.5-flash"

# ---------------- STATE -----------------
class State(TypedDict):
    query: str
    context: str
    answer_json: Dict[str, Any]

# ---------------- HELPERS ----------------
def build_vectorstore(api_key: str, pdf_file) -> FAISS:
    """Read PDF and build FAISS retriever"""
    pdf_reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    if not text.strip():
        raise ValueError("PDF is empty or not readable.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    store = FAISS.from_texts(chunks, embeddings)
    return store

def call_llm(api_key: str, prompt: str) -> dict:
    """Call Gemini and parse JSON safely"""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=500,
        convert_system_message_to_human=True,
    )
    resp = llm.invoke(prompt)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    # Extract JSON with regex
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {
            "Environmental": {"Score": 3, "Notes": "No structured output."},
            "Social": {"Score": 3, "Notes": "No structured output."},
            "Governance": {"Score": 3, "Notes": "No structured output."},
        }

    try:
        return json.loads(m.group(0))
    except Exception:
        return {
            "Environmental": {"Score": 3, "Notes": "Invalid JSON output."},
            "Social": {"Score": 3, "Notes": "Invalid JSON output."},
            "Governance": {"Score": 3, "Notes": "Invalid JSON output."},
        }

# ---------------- GRAPH NODES ----------------
def node_retrieve(state: State, retriever=None):
    """Retrieve context if retriever available"""
    if retriever:
        docs = retriever.get_relevant_documents(state["query"])
        if not docs:
            context = "No relevant context found."
        else:
            context = "\n\n---\n\n".join(d.page_content for d in docs)
    else:
        context = "No PDF provided. Using only query."
    return {"context": context}

def node_esg(state: State, api_key: str):
    """LLM ESG analysis"""
    query, context = state["query"], state["context"]

    prompt = f"""
You are an ESG risk analyst. Assess ESG risks and provide **scores from 1 to 5** for Environmental, Social, and Governance.
Respond strictly in JSON like:
{{
  "Environmental": {{"Score": 4, "Notes": "…"}},
  "Social": {{"Score": 3, "Notes": "…"}},
  "Governance": {{"Score": 2, "Notes": "…"}}
}}

Company Info: {query}
Context from documents: {context}
"""

    result = call_llm(api_key, prompt)
    return {"answer_json": result}

# ---------------- STREAMLIT APP ----------------
def main():
    st.set_page_config(page_title="🌍 ESG Risk Analyzer", layout="wide")
    st.title("🌱 ESG Risk Analyzer with Gemini + LangGraph")

    api_key = st.text_input("🔑 Enter Google API Key", type="password")

    if not api_key:
        st.warning("Please enter your Google API Key.")
        st.stop()

    # Mode Selection
    mode = st.radio("Choose input mode:", ["Direct Input", "PDF + Input"])

    query = st.text_area("📌 Enter company name & description (e.g., 'Tata Steel, Indian multinational steel manufacturing company').")

    retriever = None
    if mode == "PDF + Input":
        pdf_file = st.file_uploader("📄 Upload a company sustainability PDF", type=["pdf"])
        if pdf_file:
            try:
                store = build_vectorstore(api_key, pdf_file)
                retriever = store.as_retriever()
                st.success("✅ PDF processed and embedded.")
            except Exception as e:
                st.error(f"PDF processing failed: {e}")

    if st.button("🚀 Run ESG Assessment"):
        if not query.strip():
            st.error("Please enter a company description.")
            return

        # Build LangGraph
        graph = StateGraph(State)
        graph.add_node("retrieve", lambda s: node_retrieve(s, retriever))
        graph.add_node("esg", lambda s: node_esg(s, api_key))

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "esg")
        graph.add_edge("esg", END)

        app = graph.compile()

        with st.spinner("Analyzing ESG risks..."):
            result = app.invoke({"query": query})

        out = result.get("answer_json", {})

        # Pretty Display
        st.subheader("📊 ESG Assessment Results")

        try:
            env = int(out.get("Environmental", {}).get("Score", 3))
            soc = int(out.get("Social", {}).get("Score", 3))
            gov = int(out.get("Governance", {}).get("Score", 3))

            st.markdown(
                f"""
                🌱 **Environmental (E): {env}/5**  
                → {out.get("Environmental", {}).get("Notes", "No notes")}

                🤝 **Social (S): {soc}/5**  
                → {out.get("Social", {}).get("Notes", "No notes")}

                🏛️ **Governance (G): {gov}/5**  
                → {out.get("Governance", {}).get("Notes", "No notes")}
                """
            )
        except Exception as e:
            st.warning(f"Could not extract scores properly. ({e})")

        # Download Results
        report_text = f"""ESG Assessment Report

Company: {query}

Environmental (E): {out.get("Environmental", {}).get("Score", 'N/A')}  
Notes: {out.get("Environmental", {}).get("Notes", 'N/A')}

Social (S): {out.get("Social", {}).get("Score", 'N/A')}  
Notes: {out.get("Social", {}).get("Notes", 'N/A')}

Governance (G): {out.get("Governance", {}).get("Score", 'N/A')}  
Notes: {out.get("Governance", {}).get("Notes", 'N/A')}
"""
        st.download_button("📥 Download Report", report_text, file_name="esg_report.txt")

if __name__ == "__main__":
    main()
