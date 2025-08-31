import asyncio
import sys
import json
import re
from typing import TypedDict, Dict, Any, Optional, Callable

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from langgraph.graph import StateGraph, END


# -----------------------------
# OS / Event-loop (Windows fix)
# -----------------------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# -----------------------------
# Model / Token settings
# -----------------------------
MODEL_NAME = "gemini-1.5-flash"   # cost-efficient, fast


# -----------------------------
# LangGraph state
# -----------------------------
class ESGState(TypedDict, total=False):
    query: str
    report: Dict[str, Any]


# -----------------------------
# Gemini helpers (token-efficient)
# -----------------------------
def setup_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"temperature": 0, "max_output_tokens": 800},
    )

JSON_FENCE = """Return ONLY valid minified JSON, no extra words, no markdown fences."""

def extract_json(text: str) -> Dict[str, Any]:
    """Robust JSON extractor from model output."""
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def call_gemini_json(model, prompt: str) -> Dict[str, Any]:
    """Call Gemini with a JSON-only instruction and parse reply."""
    resp = model.generate_content(prompt + "\n\n" + JSON_FENCE)
    # google-generativeai returns .text
    return extract_json(getattr(resp, "text", "") or "")


# -----------------------------
# Node builders (each adds one section)
# -----------------------------
def node_understanding(model) -> Callable[[ESGState], Dict[str, Any]]:
    def _run(state: ESGState):
        prompt = (
            'Explain ESG in simple terms (2–4 sentences). '
            'Output JSON: {"UnderstandingESG": "<text>"}'
        )
        out = call_gemini_json(model, prompt)
        return {"report": {**state.get("report", {}), **out}}
    return _run

def node_client_needs(model) -> Callable[[ESGState], Dict[str, Any]]:
    def _run(state: ESGState):
        q = state["query"]
        prompt = (
            f'Company description: "{q}". '
            "Identify key ESG needs (max 6 concise bullets). "
            'Output JSON: {"ClientNeeds": "<bullets or short paragraph>"}'
        )
        out = call_gemini_json(model, prompt)
        return {"report": {**state.get("report", {}), **out}}
    return _run

def node_client_analysis(model) -> Callable[[ESGState], Dict[str, Any]]:
    def _run(state: ESGState):
        q = state["query"]
        prompt = (
            f'Company description: "{q}". '
            "Give risks & opportunities (balanced; 4–6 bullets total). "
            'Output JSON: {"ClientAnalysis": "<bullets or short paragraph>"}'
        )
        out = call_gemini_json(model, prompt)
        return {"report": {**state.get("report", {}), **out}}
    return _run

def node_solutions(model) -> Callable[[ESGState], Dict[str, Any]]:
    def _run(state: ESGState):
        q = state["query"]
        prompt = (
            f'Company description: "{q}". '
            "Describe how sustainability solutions address those risks (3–5 crisp points). "
            'Output JSON: {"SustainabilitySolutionAssessment": "<bullets or short paragraph>"}'
        )
        out = call_gemini_json(model, prompt)
        return {"report": {**state.get("report", {}), **out}}
    return _run

def node_fitment(model) -> Callable[[ESGState], Dict[str, Any]]:
    def _run(state: ESGState):
        q = state["query"]
        prompt = (
            f'Company description: "{q}". '
            "Propose 4–5 specific ESG initiatives with evaluation. "
            'Only these enums: Cost={High,Medium,Low}, Feasibility={High,Medium,Low}, '
            'Impact={High,Medium,Low}, Fitment={Strong,Good,Moderate,Weak}. '
            'Output JSON: {"FitmentMatrix":[{"Option":"...","Cost":"High|Medium|Low",'
            '"Feasibility":"High|Medium|Low","Impact":"High|Medium|Low","Fitment":"Strong|Good|Moderate|Weak"}]}'
        )
        out = call_gemini_json(model, prompt)
        return {"report": {**state.get("report", {}), **out}}
    return _run

def node_scores(model) -> Callable[[ESGState], Dict[str, Any]]:
    def _run(state: ESGState):
        q = state["query"]
        prompt = (
            f'Company description: "{q}". '
            "Give integer ESG scores (1–5) with short notes (<25 words each). "
            'Output JSON: {"ESG":{"Environmental":{"Score":int,"Notes":"..."},'
            '"Social":{"Score":int,"Notes":"..."},"Governance":{"Score":int,"Notes":"..."}}}'
        )
        out = call_gemini_json(model, prompt)
        return {"report": {**state.get("report", {}), **out}}
    return _run


# -----------------------------
# UI helpers
# -----------------------------
def pretty_fitment(df: pd.DataFrame) -> pd.DataFrame:
    """Add emoji to fitment + order columns for clarity."""
    if df.empty:
        return df
    icon = {"Strong": "✅ Strong", "Good": "🟩 Good", "Moderate": "🟨 Moderate", "Weak": "❌ Weak"}
    df = df.copy()
    if "Fitment" in df.columns:
        df["Fitment"] = df["Fitment"].map(lambda x: icon.get(str(x), str(x)))
    want_cols = [c for c in ["Option", "Cost", "Feasibility", "Impact", "Fitment"] if c in df.columns]
    return df[want_cols]

def radar_chart(scores: Dict[str, Any]):
    labels = ["Environmental", "Social", "Governance"]
    vals = [
        int(scores.get("Environmental", {}).get("Score", 3)),
        int(scores.get("Social", {}).get("Score", 3)),
        int(scores.get("Governance", {}).get("Score", 3)),
    ]
    vals += vals[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, vals, alpha=0.25)
    ax.plot(angles, vals, linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([1, 2, 3, 4, 5])
    return fig


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="ESG Analyzer (Gemini + LangGraph)", layout="wide")
    st.title("🌍 ESG Risk Analyzer — Gemini + LangGraph (6-step)")

    api_key = st.text_input("🔑 Google Gemini API Key", type="password")
    query = st.text_area(
        "📌 Company description (one paragraph)",
        placeholder="e.g., Shyam Steel is an Indian steel manufacturer producing TMT bars and structural steel for construction, operating multiple rolling mills across Eastern India...",
        height=120,
    )

    run = st.button("🚀 Run ESG Analysis")

    if not run:
        return

    if not api_key:
        st.error("Please provide your Google Gemini API key.")
        st.stop()

    if not query.strip():
        st.error("Please enter a brief company description.")
        st.stop()

    # Init model
    model = setup_gemini(api_key)

    # Build graph
    g = StateGraph(ESGState)
    g.add_node("understanding", node_understanding(model))
    g.add_node("needs",         node_client_needs(model))
    g.add_node("analysis",      node_client_analysis(model))
    g.add_node("solutions",     node_solutions(model))
    g.add_node("fitment",       node_fitment(model))
    g.add_node("scores",        node_scores(model))

    g.set_entry_point("understanding")
    g.add_edge("understanding", "needs")
    g.add_edge("needs", "analysis")
    g.add_edge("analysis", "solutions")
    g.add_edge("solutions", "fitment")
    g.add_edge("fitment", "scores")
    g.add_edge("scores", END)

    app = g.compile()

    with st.spinner("Analyzing…"):
        result = app.invoke({"query": query, "report": {}})

    report = result.get("report", {})

    # ---- Display (6 parts) ----
    st.markdown("## 1️⃣ Understanding ESG")
    st.write(report.get("UnderstandingESG", "N/A"))

    st.markdown("## 2️⃣ Client Needs")
    st.write(report.get("ClientNeeds", "N/A"))

    st.markdown("## 3️⃣ Client’s Analysis")
    st.write(report.get("ClientAnalysis", "N/A"))

    st.markdown("## 4️⃣ Sustainability Solution Assessment")
    st.write(report.get("SustainabilitySolutionAssessment", "N/A"))

    st.markdown("## 5️⃣ Fitment Matrix")
    df_fit = pretty_fitment(pd.DataFrame(report.get("FitmentMatrix", [])))
    st.dataframe(df_fit, use_container_width=True)

    st.markdown("## 6️⃣ ESG Scores")
    esg = report.get("ESG", {})
    if esg:
        col1, col2, col3 = st.columns(3)
        with col1:
            e = esg.get("Environmental", {})
            st.metric("Environmental", e.get("Score", "N/A"))
            st.caption(e.get("Notes", ""))
        with col2:
            s = esg.get("Social", {})
            st.metric("Social", s.get("Score", "N/A"))
            st.caption(s.get("Notes", ""))
        with col3:
            g_ = esg.get("Governance", {})
            st.metric("Governance", g_.get("Score", "N/A"))
            st.caption(g_.get("Notes", ""))

        st.pyplot(radar_chart(esg))
    else:
        st.write("N/A")

    # Downloads
    st.download_button(
        "📥 Download Full JSON",
        data=json.dumps(report, indent=2),
        file_name="esg_report.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
