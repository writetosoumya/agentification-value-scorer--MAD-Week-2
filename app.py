"""
app.py
Streamlit UI for the Agentification Value Scorer
ARM™ × AVRE™ Governance Intelligence Engine
"""

import sys
import json
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine import build_vectorstore, get_retriever, retrieve_context
from llm_scorer import score_use_case
from evaluator import run_evaluation_suite, EVAL_QUESTIONS, evaluate_retrieval

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentification Value Scorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0f1117; }

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 1rem;
}
.badge-purple { background: #2d1b69; color: #a78bfa; border: 1px solid #4c1d95; }
.badge-blue { background: #1e3a5f; color: #60a5fa; border: 1px solid #1d4ed8; }
.badge-green { background: #14532d; color: #4ade80; border: 1px solid #166534; }

.metric-card {
    background: #1a1d2e;
    border: 1px solid #2d2f45;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
}
.metric-sub {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.3rem;
}

.risk-card {
    background: #1a1d2e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid;
}
.risk-label { font-size: 0.8rem; color: #aaa; margin-bottom: 0.2rem; }
.risk-score { font-size: 1.6rem; font-weight: 700; }
.risk-rationale { font-size: 0.75rem; color: #888; margin-top: 0.4rem; line-height: 1.4; }

.tier-banner {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.tier-label { font-size: 0.85rem; font-weight: 600; opacity: 0.8; margin-bottom: 0.3rem; }
.tier-value { font-size: 1.3rem; font-weight: 700; }

.nev-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.nev-label { font-size: 0.9rem; font-weight: 600; opacity: 0.8; margin-bottom: 0.5rem; }
.nev-value { font-size: 4rem; font-weight: 800; line-height: 1; }
.nev-rating { font-size: 1rem; font-weight: 500; margin-top: 0.5rem; opacity: 0.9; }

.rec-card {
    background: #1a1d2e;
    border: 1px solid #2d2f45;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.rec-num {
    background: #2d1b69;
    color: #a78bfa;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.rec-text { font-size: 0.85rem; color: #ccc; line-height: 1.5; }

.source-tag {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    border: 1px solid #1d4ed8;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-right: 6px;
    margin-top: 4px;
}

.preset-btn {
    background: #1a1d2e;
    border: 1px solid #2d2f45;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    cursor: pointer;
    font-size: 0.8rem;
    color: #ccc;
}

.eval-pass { color: #4ade80; font-weight: 600; }
.eval-fail { color: #f87171; font-weight: 600; }

.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2d2f45;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def score_color(s):
    if s >= 7.6: return "#ef4444"
    elif s >= 5.6: return "#f97316"
    elif s >= 3.1: return "#eab308"
    return "#22c55e"

def value_color(s):
    if s >= 7: return "#22c55e"
    elif s >= 5: return "#eab308"
    elif s >= 3: return "#f97316"
    return "#ef4444"

def tier_style(composite):
    if composite >= 7.6:
        return {"bg": "#2d0a0a", "border": "#ef4444", "color": "#ef4444", "label": "⛔ CRITICAL", "text": "Do not deploy without executive sign-off and external audit"}
    elif composite >= 5.6:
        return {"bg": "#2d1500", "border": "#f97316", "color": "#f97316", "label": "🔴 RED", "text": "Phased deployment only. Independent review required."}
    elif composite >= 3.1:
        return {"bg": "#2d2200", "border": "#eab308", "color": "#eab308", "label": "🟡 AMBER", "text": "Proceed with enhanced controls and HITL gates"}
    return {"bg": "#0a2d0a", "border": "#22c55e", "color": "#22c55e", "label": "🟢 GREEN", "text": "Proceed with standard governance practices"}

def nev_style(nev):
    if nev >= 7:
        return {"bg": "#0a2d0a", "color": "#22c55e", "rating": "⭐ HIGH VALUE — Prioritize for agentification"}
    elif nev >= 5:
        return {"bg": "#2d2200", "color": "#eab308", "rating": "✅ MODERATE VALUE — Proceed with phased approach"}
    elif nev >= 3:
        return {"bg": "#2d1500", "color": "#f97316", "rating": "⚠️ CONDITIONAL VALUE — Resolve blockers first"}
    return {"bg": "#2d0a0a", "color": "#ef4444", "rating": "❌ LOW VALUE — Consider alternative automation"}

# ── Load vectorstore (cached) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag():
    store = build_vectorstore()
    retriever = get_retriever(store)
    return retriever

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem'>
        <div style='font-size:1.1rem;font-weight:700;color:#a78bfa'>⚡ AVS</div>
        <div style='font-size:0.75rem;color:#666'>Agentification Value Scorer</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🎯 Score a Use Case", "📊 Evaluation Report", "📖 About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#555;line-height:1.6'>
        <div style='color:#888;font-weight:600;margin-bottom:0.5rem'>FRAMEWORKS</div>
        <div>ARM™ — Agentification Risk Model</div>
        <div>AVRE™ — Agentic Value Realization Engine</div>
        <div style='margin-top:1rem;color:#888;font-weight:600;margin-bottom:0.5rem'>CORPUS</div>
        <div>EU AI Act (2024/1689)</div>
        <div>NIST AI RMF 1.0</div>
        <div>ISO 42001:2023</div>
        <div>ARM™ & AVRE™ Frameworks</div>
        <div style='margin-top:1rem;color:#888;font-weight:600;margin-bottom:0.5rem'>STACK</div>
        <div>LangChain · TF-IDF · MMR</div>
        <div>Llama 3.3 70B · Nebius</div>
        <div>Streamlit · Python</div>
    </div>
    """, unsafe_allow_html=True)

# ── Load RAG ──────────────────────────────────────────────────────────────────

with st.spinner("Loading governance knowledge base..."):
    retriever = load_rag()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: SCORE A USE CASE
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎯 Score a Use Case":

    # Hero
    st.markdown('<div class="hero-title">Agentification Value Scorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Score any enterprise AI use case for agentification readiness using ARM™ risk dimensions and AVRE™ value lenses</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="badge badge-purple">ARM™ Risk Model</span>
    <span class="badge badge-blue">AVRE™ Value Engine</span>
    <span class="badge badge-green">RAG-Grounded</span>
    """, unsafe_allow_html=True)

    # Presets
    st.markdown('<div class="section-header">Quick select — example use cases</div>', unsafe_allow_html=True)

    PRESETS = [
        ("HR Resume Screening Agent", "An AI agent that automatically screens and ranks resumes against job requirements, scores candidates, and sends automated rejection emails. Deployed in Workday across 50 open requisitions monthly."),
        ("Enterprise Policy Q&A Bot", "A RAG-powered chatbot answering HR, IT, and compliance policy questions for 8,000 employees via Microsoft Teams. Read-only, cites source documents in every answer. No write actions."),
        ("Fraud Detection Agent", "Real-time transaction monitoring agent that flags anomalies and autonomously places holds on accounts showing high fraud signals. Operates 24/7 in retail banking."),
        ("IT Service Management Triage", "Agent that classifies incoming IT tickets, auto-resolves low-complexity requests like password resets, and routes complex issues with suggested resolution paths."),
        ("Contract Review Agent", "Extracts key terms from vendor contracts, flags non-standard clauses against a legal playbook, generates attorney review summary. Attorney makes all final decisions."),
        ("Regulatory Intelligence Agent", "Monitors regulatory changes (EU AI Act, SEC, FDA) and generates organizational impact assessments. Read-only synthesis. Legal team reviews before any compliance action."),
    ]

    cols = st.columns(3)
    for i, (name, desc) in enumerate(PRESETS):
        with cols[i % 3]:
            if st.button(f"**{name}**", key=f"preset_{i}", use_container_width=True):
                st.session_state["uc_name"] = name
                st.session_state["uc_desc"] = desc

    # Input form
    st.markdown('<div class="section-header">Describe your use case</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        uc_name = st.text_input(
            "Use Case Name",
            value=st.session_state.get("uc_name", ""),
            placeholder="e.g. HR Resume Screening Agent"
        )
    with col2:
        uc_desc = st.text_area(
            "Use Case Description",
            value=st.session_state.get("uc_desc", ""),
            placeholder="Describe what the agent does, who uses it, what it can act on, and any system integrations...",
            height=100
        )

    score_btn = st.button("⚡ Score this use case", type="primary", use_container_width=True)

    # Scoring
    if score_btn:
        if not uc_name or not uc_desc:
            st.error("Please fill in both the use case name and description.")
        else:
            with st.spinner("Retrieving governance context from knowledge base..."):
                rag_result = retrieve_context(
                    f"ARM AVRE compliance equity risk scoring: {uc_name} {uc_desc}",
                    retriever
                )

            with st.spinner("Scoring with ARM™ and AVRE™ frameworks via Llama 3.3 70B..."):
                try:
                    result = score_use_case(
                        use_case_name=uc_name,
                        use_case_description=uc_desc,
                        rag_context=rag_result["context"],
                        rag_sources=rag_result["sources"],
                    )

                    st.success("Scoring complete!")
                    st.markdown("---")

                    # ── ARM™ Section ──────────────────────────────────────────
                    st.markdown('<div class="section-header">ARM™ Risk Assessment</div>', unsafe_allow_html=True)

                    dim_map = [
                        ("Autonomy Risk", result.arm.autonomy_risk, "autonomy_risk", "25%"),
                        ("Data Governance Risk", result.arm.data_governance_risk, "data_governance_risk", "20%"),
                        ("Compliance Exposure Risk", result.arm.compliance_exposure_risk, "compliance_exposure_risk", "20%"),
                        ("Capability Debt Risk", result.arm.capability_debt_risk, "capability_debt_risk", "20%"),
                        ("Equity & Bias Risk", result.arm.equity_bias_risk, "equity_bias_risk", "15%"),
                    ]

                    col_a, col_b = st.columns(2)
                    for i, (label, score, key, weight) in enumerate(dim_map):
                        col = col_a if i % 2 == 0 else col_b
                        color = score_color(score)
                        rationale = result.rationale.get(key, "")
                        with col:
                            st.markdown(f"""
                            <div class="risk-card" style="border-left-color:{color}">
                                <div class="risk-label">{label} <span style="color:#555">{weight}</span></div>
                                <div class="risk-score" style="color:{color}">{score:.1f}<span style="font-size:1rem;color:#555">/10</span></div>
                                <div style="background:#111;border-radius:4px;height:4px;margin:0.5rem 0">
                                    <div style="background:{color};height:4px;border-radius:4px;width:{score*10}%"></div>
                                </div>
                                <div class="risk-rationale">{rationale[:120]}{"..." if len(rationale)>120 else ""}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Composite + tier
                    composite = result.arm.composite()
                    ts = tier_style(composite)
                    st.markdown(f"""
                    <div class="tier-banner" style="background:{ts['bg']};border:2px solid {ts['border']}">
                        <div class="tier-label" style="color:{ts['color']}">COMPOSITE ARM™ SCORE</div>
                        <div style="font-size:3rem;font-weight:800;color:{ts['color']}">{composite}</div>
                        <div class="tier-value" style="color:{ts['color']}">{ts['label']}</div>
                        <div style="font-size:0.85rem;color:#aaa;margin-top:0.3rem">{ts['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── AVRE™ Section ─────────────────────────────────────────
                    st.markdown('<div class="section-header">AVRE™ Value Assessment</div>', unsafe_allow_html=True)

                    avre_map = [
                        ("ROI — Return on Investment", result.avre.roi_score, "roi", "40%"),
                        ("ROE — Return on Effort", result.avre.roe_score, "roe", "30%"),
                        ("ROF — Strategic Optionality", result.avre.rof_score, "rof", "30%"),
                        ("Opportunity Cost", result.avre.opportunity_cost, "opportunity_cost", "ref"),
                    ]

                    for label, score, key, weight in avre_map:
                        color = value_color(score)
                        rationale = result.rationale.get(key, "")
                        st.markdown(f"""
                        <div class="risk-card" style="border-left-color:{color};display:flex;align-items:center;gap:1.5rem">
                            <div style="min-width:180px">
                                <div class="risk-label">{label} <span style="color:#555">{weight}</span></div>
                                <div class="risk-score" style="color:{color}">{score:.1f}</div>
                            </div>
                            <div style="flex:1">
                                <div style="background:#111;border-radius:4px;height:6px;margin-bottom:0.5rem">
                                    <div style="background:{color};height:6px;border-radius:4px;width:{score*10}%"></div>
                                </div>
                                <div class="risk-rationale">{rationale[:140]}{"..." if len(rationale)>140 else ""}</div>
                            </div>
                            <div style="font-size:1.8rem;font-weight:700;color:{color};min-width:50px;text-align:right">{score:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    br = result.avre.benefit_realization()
                    st.markdown(f"""
                    <div style="background:#1a1d2e;border:1px solid #2d2f45;border-radius:10px;padding:1rem 1.5rem;margin:0.5rem 0;display:flex;justify-content:space-between;align-items:center">
                        <div style="color:#888;font-size:0.85rem;font-weight:600">BENEFIT REALIZATION SCORE</div>
                        <div style="font-size:1.8rem;font-weight:700;color:#a78bfa">{br:.2f} / 10</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── NEV ───────────────────────────────────────────────────
                    nev = result.net_enterprise_value()
                    ns = nev_style(nev)
                    mult = result.arm.risk_multiplier()

                    st.markdown(f"""
                    <div class="nev-card" style="background:{ns['bg']};border:2px solid {ns['color']}">
                        <div class="nev-label" style="color:{ns['color']}">NET ENTERPRISE VALUE</div>
                        <div class="nev-value" style="color:{ns['color']}">{nev}<span style="font-size:1.5rem">/10</span></div>
                        <div class="nev-rating" style="color:{ns['color']}">{ns['rating']}</div>
                        <div style="font-size:0.8rem;color:#666;margin-top:0.8rem">
                            ARM™ Risk Multiplier: <strong style="color:#aaa">{mult:.3f}</strong> &nbsp;|&nbsp; 
                            Benefit Realization: <strong style="color:#aaa">{br:.2f}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Recommendations ───────────────────────────────────────
                    if result.recommendations:
                        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
                        for i, rec in enumerate(result.recommendations, 1):
                            st.markdown(f"""
                            <div class="rec-card">
                                <div class="rec-num">{i}</div>
                                <div class="rec-text">{rec}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # ── Sources ───────────────────────────────────────────────
                    st.markdown('<div class="section-header">RAG Sources Used</div>', unsafe_allow_html=True)
                    sources_html = "".join(f'<span class="source-tag">{s}</span>' for s in result.rag_sources)
                    st.markdown(f"<div>{sources_html}</div>", unsafe_allow_html=True)

                    # ── Save JSON ─────────────────────────────────────────────
                    out_path = Path(__file__).parent / "outputs" / f"{uc_name.replace(' ','_')[:40]}_score.json"
                    out_path.parent.mkdir(exist_ok=True)
                    with open(out_path, "w") as f:
                        json.dump(result.to_dict(), f, indent=2)
                    st.caption(f"Score saved to {out_path}")

                except Exception as e:
                    st.error(f"Scoring failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Evaluation Report":

    st.markdown('<div class="hero-title">RAG Evaluation Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">15-question faithfulness and retrieval accuracy assessment</div>', unsafe_allow_html=True)

    # Check for existing report
    report_path = Path(__file__).parent / "outputs" / "eval_report.json"

    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        summary = report["evaluation_summary"]

        # Big metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Faithfulness</div>
                <div class="metric-value" style="color:#22c55e">{summary['avg_faithfulness_pct']}%</div>
                <div class="metric-sub">across 14 scoreable questions</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Source Accuracy</div>
                <div class="metric-value" style="color:#22c55e">{summary['source_retrieval_accuracy_pct']}%</div>
                <div class="metric-sub">correct document retrieved</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Questions Passing</div>
                <div class="metric-value" style="color:#a78bfa">{summary['questions_passing_50pct']}</div>
                <div class="metric-sub">at ≥50% faithfulness</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Retrieval Strategy</div>
                <div class="metric-value" style="font-size:1.2rem;color:#60a5fa">MMR k=5</div>
                <div class="metric-sub">{summary['embed_model']}</div>
            </div>""", unsafe_allow_html=True)

        # Results table
        st.markdown('<div class="section-header">Question-by-Question Results</div>', unsafe_allow_html=True)

        for r in report["results"]:
            faith = r["faithfulness_score"]
            color = "#22c55e" if faith >= 0.75 else "#eab308" if faith >= 0.5 else "#ef4444"
            icon = "✅" if faith >= 0.5 else "❌"
            diff_color = {"easy":"#22c55e","medium":"#eab308","hard":"#f97316","out-of-corpus":"#888"}.get(r["difficulty"],"#aaa")

            with st.expander(f"{icon} {r['id']} — {r['category']} · {r['query'][:70]}..."):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Faithfulness:** <span style='color:{color}'>{faith*100:.0f}% ({r['concept_hits']}/{r['concept_total']})</span>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**Source hit:** {'✅' if r['source_hit'] else '❌'}")
                with c3:
                    st.markdown(f"**Difficulty:** <span style='color:{diff_color}'>{r['difficulty']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Query:** {r['query']}")
                st.markdown(f"**Sources retrieved:** {', '.join(r['retrieved_sources'])}")
                if r.get("notes"):
                    st.info(r["notes"])

    else:
        st.warning("No evaluation report found. Run the evaluation suite first.")
        if st.button("▶ Run 15-Question Evaluation Suite", type="primary"):
            with st.spinner("Running evaluation — this takes about 30 seconds..."):
                report = run_evaluation_suite(retriever)
            st.success("Evaluation complete! Refresh this page to see results.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📖 About":
    st.markdown('<div class="hero-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#888;line-height:1.8;max-width:700px">
    <p><strong style="color:#ccc">The Agentification Value Scorer</strong> is a RAG-powered governance intelligence tool 
    that scores enterprise business use cases for AI agent readiness using two proprietary frameworks:</p>
    
    <p><strong style="color:#a78bfa">ARM™ (Agentification Risk Model)</strong> — Five risk dimensions: 
    Autonomy Risk, Data Governance Risk, Compliance Exposure Risk, Capability Debt Risk, and Equity & Bias Risk. 
    Each scored 0–10, weighted into a composite ARM™ score mapping to GREEN → AMBER → RED → CRITICAL tiers.</p>
    
    <p><strong style="color:#60a5fa">AVRE™ (Agentic Value Realization Engine)</strong> — Three value lenses: 
    ROI, ROE (Return on Effort), and ROF (Return on Freedom / Strategic Optionality), combined into a 
    Benefit Realization score adjusted by the ARM™ Risk Multiplier to produce a Net Enterprise Value (NEV).</p>
    
    <p><strong style="color:#ccc">RAG Architecture:</strong> TF-IDF vectorstore with MMR retrieval (k=5, fetch_k=20, λ=0.6) 
    over a 59-chunk governance corpus spanning ARM™, AVRE™, EU AI Act, NIST AI RMF, and ISO 42001.</p>
    
    <p><strong style="color:#ccc">Week 2 Project</strong> | The Gen Academy Mastering Agentic AI Bootcamp<br>
    <strong style="color:#ccc">Author:</strong> Soumya V Jom | Enterprise AI Strategy & Governance, Cognizant PS&E</p>
    </div>
    """, unsafe_allow_html=True)
