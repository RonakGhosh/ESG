import streamlit as st
from PyPDF2 import PdfReader
import faiss 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from typing import TypedDict, Optional
import json, re, pandas as pd

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌱 ESG Risk Assessment", layout="wide")

st.title("🌱 ESG Risk Assessment Agent")

api_key = st.text_input("🔑 Enter your Google API Key:", type="password")

query = st.text_area("📌 Enter Company Description or ESG Policy:")

use_rag = st.checkbox("📂 Use PDF for RAG (optional)")

pdf_text = ""
if use_rag:
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")
    if pdf_file:
        pdf_reader = PdfReader(pdf_file)
        pdf_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])

if st.button("🚀 Run ESG Assessment"):
    if not api_key:
        st.error("Please enter your Google API Key!")
    elif not query:
        st.error("Please enter some company description or ESG policy!")
    else:
        # -------------------------------
        # LangGraph State Definition
        # -------------------------------
        class ESGState(TypedDict):
            query: str
            pdf_text: str
            use_rag: bool
            answer_json: Optional[dict]
            error: Optional[str]

        # -------------------------------
        # Build Vectorstore for RAG
        # -------------------------------
        def build_vectorstore(text, api_key):
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.create_documents([text])
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            return FAISS.from_documents(docs, embeddings)

        # -------------------------------
        # ESG Node
        # -------------------------------
        def esg_assessment_node(state: ESGState) -> ESGState:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)

                context = ""
                if state["use_rag"] and state["pdf_text"]:
                    vs = build_vectorstore(state["pdf_text"], api_key)
                    retriever = vs.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.get_relevant_documents(state["query"])
                    context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""
                You are an ESG risk assessment assistant.
                Task: Assess ESG risks and provide Environmental, Social, and Governance scores (1–5) with brief notes.

                If context is provided, use it. Otherwise, rely on general knowledge.

                Input Query:
                {state['query']}

                Context:
                {context}

                Output strictly as JSON in this format:
                {{
                  "Environmental": {{"Score": <1-5>, "Notes": "<brief note>"}},
                  "Social": {{"Score": <1-5>, "Notes": "<brief note>"}},
                  "Governance": {{"Score": <1-5>, "Notes": "<brief note>"}}
                }}
                """

                response = llm.invoke(prompt).content.strip()

                match = re.search(r"\{.*\}", response, re.DOTALL)
                if not match:
                    raise ValueError(f"Model did not return valid JSON:\n{response}")

                json_str = match.group(0)
                answer = json.loads(json_str)

                return {**state, "answer_json": answer, "error": None}

            except Exception as e:
                return {**state, "error": str(e)}

        # -------------------------------
        # LangGraph Workflow
        # -------------------------------
        workflow = StateGraph(ESGState)
        workflow.add_node("esg_node", esg_assessment_node)
        workflow.set_entry_point("esg_node")
        workflow.set_finish_point("esg_node")
        app = workflow.compile()

        # -------------------------------
        # Run Workflow
        # -------------------------------
        initial_state: ESGState = {"query": query, "pdf_text": pdf_text, "use_rag": use_rag, "answer_json": None, "error": None}
        result = app.invoke(initial_state)

        # -------------------------------
        # Display Results
        # -------------------------------
        if result["error"]:
            st.error(f"❌ Error: {result['error']}")
        elif result["answer_json"]:
            answer = result["answer_json"]

            st.success("✅ ESG Assessment Results")

            # --- Paragraph style output ---
            st.markdown("### 🌍 Environmental")
            st.write(f"**Score:** {answer['Environmental']['Score']}")
            st.write(answer['Environmental']['Notes'])

            st.markdown("### 👥 Social")
            st.write(f"**Score:** {answer['Social']['Score']}")
            st.write(answer['Social']['Notes'])

            st.markdown("### 🏛 Governance")
            st.write(f"**Score:** {answer['Governance']['Score']}")
            st.write(answer['Governance']['Notes'])

            # --- Download buttons ---
            # JSON
            st.download_button("⬇️ Download JSON",
                               data=json.dumps(answer, indent=2),
                               file_name="esg_assessment.json",
                               mime="application/json")

            # TXT
            txt_output = "\n\n".join([
                f"Environmental ({answer['Environmental']['Score']}): {answer['Environmental']['Notes']}",
                f"Social ({answer['Social']['Score']}): {answer['Social']['Notes']}",
                f"Governance ({answer['Governance']['Score']}): {answer['Governance']['Notes']}"
            ])
            st.download_button("⬇️ Download TXT",
                               data=txt_output,
                               file_name="esg_assessment.txt",
                               mime="text/plain")

            # CSV
            df = pd.DataFrame([
                {"Category": "Environmental", "Score": answer["Environmental"]["Score"], "Notes": answer["Environmental"]["Notes"]},
                {"Category": "Social", "Score": answer["Social"]["Score"], "Notes": answer["Social"]["Notes"]},
                {"Category": "Governance", "Score": answer["Governance"]["Score"], "Notes": answer["Governance"]["Notes"]}
            ])
            st.download_button("⬇️ Download CSV",
                               data=df.to_csv(index=False),
                               file_name="esg_assessment.csv",
                               mime="text/csv")
