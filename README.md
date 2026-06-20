# Agentification Value Scorer
### ARM™ × AVRE™ Intelligence Engine — RAG-Powered Governance Assessment

**Week 2 Project | The Gen Academy Mastering Agentic AI Certification**  
**Author:** Soumya V Jom | Enterprise AI Strategy & Governance, Cognizant PS&E

---

## Objective

The **Agentification Value Scorer** is a RAG-powered governance intelligence tool that scores any enterprise business use case for AI agent readiness across two proprietary frameworks:

- **ARM™ (Agentification Risk Model):** Five-dimension risk assessment — Autonomy Risk, Data Governance Risk, Compliance Exposure Risk, Capability Debt Risk, and Equity & Bias Risk. Each scored 0–10, weighted into a composite ARM™ score that maps to a risk tier (GREEN → AMBER → RED → CRITICAL).

- **AVRE™ (Agentic Value Realization Engine):** Three value lenses — ROI, ROE (Return on Effort), and ROF (Return on Freedom / Strategic Optionality) — combined into a Benefit Realization score, then adjusted by the ARM™ Risk Multiplier to produce a **Net Enterprise Value (NEV)** score.

The system uses **Retrieval-Augmented Generation (RAG)** to ground all scoring in a governance knowledge corpus spanning:
- ARM™ and AVRE™ framework documents (proprietary IP)
- EU AI Act (Regulation 2024/1689) — prohibited practices, high-risk categories, GPAI provisions
- NIST AI Risk Management Framework 1.0 — GOVERN, MAP, MEASURE, MANAGE functions
- ISO 42001:2023 — AI Management System Standard
- Enterprise agentification use case reference library

---

## RAG One-Liner

> My RAG app helps **enterprise AI governance practitioners and business leaders** answer **"should we agentify this use case, and at what risk?"** from **governance frameworks, regulatory documents, and use case reference patterns** in a **Streamlit web interface** with **≥75% faithfulness** and **≤10 second response latency**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  AGENTIFICATION VALUE SCORER             │
│              Streamlit Web UI + CLI Interface            │
│                                                          │
│  ┌─────────────┐    ┌──────────────────────────────┐    │
│  │  User Input │    │    TF-IDF Vector Store        │    │
│  │  (use case  │    │  ┌──────────────────────────┐ │    │
│  │  name +     │    │  │ ARM™ framework chunks    │ │    │
│  │  description│    │  │ AVRE™ framework chunks   │ │    │
│  └──────┬──────┘    │  │ EU AI Act provisions     │ │    │
│         │           │  │ NIST AI RMF functions    │ │    │
│         ▼           │  │ Use case reference lib   │ │    │
│  ┌──────────────┐   │  └──────────────────────────┘ │    │
│  │ RAG Retrieval│◄──┤   Embedding: TF-IDF + bigrams  │    │
│  │ (MMR, k=5)  │   │   Retrieval: MMR (diversity)   │    │
│  └──────┬───────┘   └──────────────────────────────┘    │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │         LLM SCORING AGENT (Llama 3.3 70B)        │   │
│  │         Served via Nebius Token Factory           │   │
│  │  System: ARM™ + AVRE™ scoring rubrics            │   │
│  │  Input:  Use case + retrieved governance context  │   │
│  │  Output: Structured JSON scores + rationale       │   │
│  └──────┬───────────────────────────────────────────┘   │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │              SCORING ENGINE                       │   │
│  │  ARM™ Composite Score → Risk Tier                │   │
│  │  AVRE™ Benefit Realization                       │   │
│  │  NEV = BR - (CDR × ARM™ Multiplier)             │   │
│  │  Streamlit UI + JSON export                      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| RAG Framework | LangChain |
| Vector Store | Custom TF-IDF (scikit-learn, pickle-serialized) |
| Embeddings | TF-IDF + bigrams, 8,000 features, sublinear TF scaling (local, no API key) |
| Retrieval Strategy | MMR (Maximal Marginal Relevance), k=5, fetch_k=20, λ=0.6 |
| Chunking | Recursive character splitter, 800 tokens, 120 overlap |
| LLM Scoring | Llama 3.3 70B via Nebius Token Factory (OpenAI-compatible API) |
| Web UI | Streamlit |
| Terminal UI | Rich |
| Evaluation | Custom 15-question faithfulness suite |

---

## Setup

```bash
git clone <repo>
cd agentification-value-scorer
pip install -r requirements.txt
```

Set your Nebius API key:
```bash
export NEBIUS_API_KEY="your-nebius-secret-key"
```

---

## Usage

### Web UI (recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with three pages:
- **Score a Use Case** — interactive scoring with 6 presets
- **Evaluation Report** — 15-question RAG faithfulness results
- **About** — project overview

### CLI
```bash
# Interactive scoring mode
python main.py

# Run 15-question evaluation suite
python main.py --eval

# Score a single use case
python main.py --score "Agent that screens resumes and ranks candidates for HR"

# Rebuild vector store (if corpus changes)
python main.py --rebuild
```

---

## RAG Design Decisions

### Why TF-IDF over Dense Embeddings
For a governance corpus with precise technical terminology (ARM™, AVRE™, EU AI Act Article references, NIST AI RMF function names), TF-IDF with bigrams is a strong choice:
- Exact term matching on regulatory language ("Annex III", "SR 11-7", "DPIA") is high-precision
- Bigrams capture multi-word governance concepts ("autonomy risk", "benefit realization", "capability debt")
- Runs fully locally — no API calls, no latency, no data exfiltration risk

### Chunking Strategy
**Recursive character splitter** at 800 tokens with 120-token overlap. Chosen because:
- Framework documents have variable structure requiring adaptive splitting
- 800 tokens keeps ARM™ scoring rubrics intact (score ranges span multiple tiers)
- 120-token overlap ensures concepts spanning chunk boundaries are captured

### Retrieval Strategy
**MMR (Maximal Marginal Relevance)** with k=5, fetch_k=20, λ=0.6. Ensures retrieved chunks span multiple framework documents (ARM™ + AVRE™ + regulatory) rather than returning near-duplicate chunks from the same section.

### "I Don't Know" Path
When retrieval does not surface relevant context, the LLM is prompted to flag uncertainty rather than hallucinate. The faithfulness evaluation tests this explicitly (Q14 — out-of-corpus question).

---

## Evaluation Results

15-question evaluation suite testing retrieval faithfulness, source accuracy, edge cases, and multi-hop reasoning.

| Metric | Result |
|--------|--------|
| Avg faithfulness (14 scoreable) | **81.2%** |
| Source retrieval accuracy | **100%** |
| Questions passing ≥50% faithfulness | **13/14** |
| Only failure | Q11 — GPAI synthetic data edge case (40%) |
| Chunking strategy | Recursive, 800 tokens, 120 overlap |
| Embedding | TF-IDF + bigrams, 8,000 features |
| Retrieval | MMR, k=5, fetch_k=20 |

---

## Frameworks (Proprietary IP)

ARM™ and AVRE™ are original frameworks developed by Soumya V Jom. They are not Cognizant IP. Use under attribution.

---

## Week 3 Extension Plan

This project serves as the **RAG backbone** for the Week 3 agentic system: an **AI Governance Intelligence Agent** with:
- `score_use_case` tool (this RAG pipeline)
- `compliance_check` tool (regulatory lookup by industry + jurisdiction)
- `web_search` tool (real-time regulatory news retrieval)
- `generate_governance_brief` tool (structured report output)
- Orchestrated via LangGraph state machine with HITL gate before final report delivery
