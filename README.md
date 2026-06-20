# Agentification Value Scorer
### ARM™ × AVRE™ Intelligence Engine — RAG-Powered Governance Assessment

**Week 2 Project | The Gen Academy Mastering Agentic AI Bootcamp**  
**Author:** Agni Sivaraman | Enterprise AI Governance, Cognizant PS&E

---

## What This Is

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

> My RAG app helps **enterprise AI governance practitioners and business leaders** answer **"should we agentify this use case, and at what risk?"** from **governance frameworks, regulatory documents, and use case reference patterns** in a **CLI interface** with **≥75% faithfulness** and **≤10 second response latency**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  AGENTIFICATION VALUE SCORER             │
│                                                          │
│  ┌─────────────┐    ┌──────────────────────────────┐    │
│  │  User Input │    │       ChromaDB Vector Store   │    │
│  │  (use case  │    │  ┌──────────────────────────┐ │    │
│  │  name +     │    │  │ ARM™ framework chunks    │ │    │
│  │  description│    │  │ AVRE™ framework chunks   │ │    │
│  └──────┬──────┘    │  │ EU AI Act provisions     │ │    │
│         │           │  │ NIST AI RMF functions    │ │    │
│         ▼           │  │ Use case reference lib   │ │    │
│  ┌──────────────┐   │  └──────────────────────────┘ │    │
│  │ RAG Retrieval│◄──┤   Embedding: all-MiniLM-L6-v2 │    │
│  │ (MMR, k=5)  │   │   Retrieval: MMR (diversity)   │    │
│  └──────┬───────┘   └──────────────────────────────┘    │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │            LLM SCORING AGENT (Claude Sonnet)      │   │
│  │  System: ARM™ + AVRE™ scoring rubrics             │   │
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
│  │  Rich terminal output + JSON export               │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| RAG Framework | LangChain |
| Vector Store | ChromaDB (local persistent) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| Retrieval Strategy | MMR (Maximal Marginal Relevance), k=5 |
| Chunking | Recursive character splitter, 800 tokens, 120 overlap |
| LLM Scoring | Anthropic Claude Sonnet via API |
| Terminal UI | Rich |
| Evaluation | Custom 15-question faithfulness suite |

---

## Setup

```bash
git clone <repo>
cd agentic-value-scorer
pip install -r requirements.txt
```

No `.env` needed for RAG — embeddings run locally. The Anthropic API key is provided via the runtime environment.

---

## Usage

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

### Chunking Strategy
**Recursive character splitter** at 800 tokens with 120-token overlap. Chosen because:
- Framework documents have variable structure (tables, numbered lists, prose) requiring adaptive splitting
- 800 tokens balances semantic completeness per chunk against retrieval precision
- 120-token overlap ensures scoring rubrics (e.g., "Score 6-8: High risk...") don't get split mid-context

### Embedding Model
**all-MiniLM-L6-v2** — fast, high-quality, 384-dim, runs fully locally. No API calls for embedding = no latency, no cost, no data exfiltration risk. Appropriate for a governance intelligence tool handling sensitive enterprise use case descriptions.

### Retrieval Strategy
**MMR (Maximal Marginal Relevance)** with k=5, fetch_k=20, λ=0.6. Chosen over pure dense similarity to ensure retrieved chunks span multiple framework documents (ARM™ + AVRE™ + regulatory) rather than returning 5 near-duplicate chunks from the same section.

### "I Don't Know" Path
When retrieval does not surface relevant context (e.g., out-of-corpus questions about specific fine amounts), the LLM is prompted to flag uncertainty rather than hallucinate. The faithfulness evaluation tests this explicitly (Q14).

---

## Evaluation Results

15-question evaluation suite testing retrieval faithfulness, source accuracy, edge cases, and multi-hop reasoning. See `outputs/eval_report.json` for full results.

| Metric | Result |
|--------|--------|
| Avg faithfulness (14 scoreable) | See eval output |
| Source retrieval accuracy | See eval output |
| Chunking strategy | Recursive, 800 tokens, 120 overlap |
| Embedding model | all-MiniLM-L6-v2 |
| Retrieval | MMR, k=5 |

---

## Frameworks (Proprietary IP)

ARM™ and AVRE™ are original frameworks developed by Agni Sivaraman. They are not Cognizant IP. Use under attribution.

---

## Week 3 Extension Plan

This project serves as the **RAG backbone** for the Week 3 agentic system: an **AI Governance Intelligence Agent** with:
- `score_use_case` tool (this RAG pipeline)
- `compliance_check` tool (regulatory lookup)
- `generate_governance_brief` tool (output formatter)
- Orchestrated via LangGraph state machine with HITL gate before final report delivery
