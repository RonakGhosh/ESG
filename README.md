# 🌍 ESG Assessment Agent (LangGraph + Gemini + Streamlit)

An interactive **ESG (Environmental, Social, Governance) assessment agent** built using **LangGraph, Gemini 1.5 Flash, FAISS, and Streamlit**.  
The app can:     
✅ Take **direct user input** for ESG evaluation.  
✅ Optionally use **RAG (Retrieval-Augmented Generation)** with company-provided PDFs.  
✅ Output results in **attractive paragraphs**, plus provide **downloadable reports**.  

---

## 🚀 Features
- **Two Modes**:
  - 🔹 **Normal Mode** → ESG assessment from user input only.  
  - 🔹 **PDF + RAG Mode** → Enhances response using company sustainability reports or policies.  
- **Token-efficient pipeline** → cost-effective Gemini usage.  
- **Output formats**:
  - Human-readable **paragraphs**  
  - Downloadable **JSON & text reports**  
- **Caching with FAISS** → avoids re-embedding PDFs.  

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/esg-agent.git
cd esg-agent
