# Week 2 Project: Agentification Value Scorer
## RAG-Powered Governance Intelligence | The Gen Academy Mastering Agentic AI Bootcamp

**Author:** Soumya V Jom  
**Date:** June 2026  
**Submission track:** Code-heavy — LangChain + custom TF-IDF vectorstore + Streamlit UI

---

## Project Overview

The **Agentification Value Scorer** is a RAG-powered governance intelligence tool that scores any enterprise business use case for AI agent readiness. Instead of building a generic Q&A bot or document retrieval system, I built a tool that directly extends my original proprietary frameworks: **ARM™ (Agentification Risk Model)** and **AVRE™ (Agentic Value Realization Engine)**.

The tool features a full **Streamlit web interface** with color-coded risk cards, animated score bars, and a Net Enterprise Value verdict — making governance assessments visual and accessible to non-technical stakeholders.

### The one-liner

> My RAG app helps **enterprise AI governance practitioners and business leaders** answer **"should we agentify this use case, and at what risk?"** from **governance frameworks, regulatory documents, and use case reference patterns** in a **Streamlit web interface** with **≥75% faithfulness** and **≤10 second response latency.**

---

## Why This Use Case

I deliberately chose NOT to use a generic HR policy bot or financial document reader. My reasoning:

1. **It's my IP.** ARM™ and AVRE™ are frameworks I developed and own. Embedding them in a RAG system means the retrieval decisions are grounded in original intellectual work — not a borrowed corpus.

2. **The governance domain is retrieval-hard.** Scoring a use case requires pulling relevant chunks across four different document types simultaneously: risk framework criteria, value lenses, regulatory provisions, and use case analogues. This is a genuinely interesting multi-document retrieval problem.

3. **It's immediately useful.** I can demo this tool to enterprise clients as a governance intake accelerator. It's not a demo for the sake of a demo.

---

## The RAG Framework (Filled Out)

| Field | Decision |
|-------|----------|
| **Use case** | Answers "what ARM™ risk score and AVRE™ Net Enterprise Value does this use case produce?" for enterprise AI governance practitioners, via Streamlit web UI and CLI |
| **Corpus** | 4 documents: ARM™ framework (original IP), AVRE™ framework (original IP), EU AI Act + NIST AI RMF + ISO 42001 (summarized governance reference), enterprise use case reference library |
| **Ingestion + cleaning** | Plain text files, UTF-8 encoded. No markup to strip. Corpus authored specifically for this RAG system to ensure terminological consistency. |
| **Ingestion + freshness** | Static for this version. Future: scheduled pull from regulatory update feeds. Corpus re-embedded when any source document changes. |
| **Chunking + embedding** | Recursive character splitter, 800 tokens, 120-token overlap, separators: `[\n\n, \n, ., space]`. Embedding: TF-IDF with bigrams (sklearn), 8,000 features, sublinear TF scaling. Local — no API calls for embedding. |
| **Retrieve** | Custom TF-IDF vectorstore (pickle-serialized). MMR (Maximal Marginal Relevance) retrieval: k=5, fetch_k=20, λ=0.6. Ensures cross-document diversity. |

---

## Build Track: LangChain + Custom Vectorstore + Streamlit

**Why not ChromaDB?** Initially planned to use ChromaDB with `all-MiniLM-L6-v2`. Hit a hard blocker: HuggingFace model downloads were unavailable in the build environment. Pivoted to a custom TF-IDF vectorstore built from scratch using scikit-learn's TfidfVectorizer with MMR-style diversity selection. This gave full control over retrieval behavior with zero external API calls for the embedding layer.

**Why TF-IDF over dense embeddings for this corpus?**

For a governance corpus with precise technical terminology (ARM™, AVRE™, EU AI Act Article references, NIST AI RMF function names), TF-IDF with bigrams is actually a strong choice:
- Exact term matching on regulatory language ("Annex III", "SR 11-7", "DPIA") is high-precision
- Bigrams capture multi-word governance concepts ("autonomy risk", "benefit realization", "capability debt")
- No hallucination risk from semantic approximation — if the term isn't in the corpus, it doesn't match

**Why Streamlit for the UI?**

The handout bonus criteria mentioned a chatbot UI. Rather than building a generic chat interface, I built a purpose-designed governance scoring UI with:
- 6 preset use case buttons for quick demos
- Color-coded ARM™ risk dimension cards (green → red)
- Animated score bars for each dimension
- A large NET ENTERPRISE VALUE verdict with color-coded background
- Expandable rationale per dimension
- RAG source tags showing which documents were retrieved

**Why Llama 3.3 70B on Nebius?**

The course requires at least one Nebius Token Factory model call. Llama 3.3 70B was selected because:
- Excellent at returning clean structured JSON — critical for the scoring pipeline
- Fast inference on Nebius infrastructure
- OpenAI-compatible API format — minimal code change from the original design

---

## Architecture

```
User Input (use case name + description)
          │
          ▼
  Streamlit Web UI (app.py)
  ┌────────────────────────────────────────┐
  │ • 6 preset use case buttons           │
  │ • Custom name + description input     │
  │ • Score button triggers pipeline      │
  └────────────────────────────────────────┘
          │
          ▼
  RAG Retrieval Layer
  ┌────────────────────────────────────────┐
  │ TF-IDF Vectorstore (sklearn)           │
  │ • 59 chunks across 4 documents        │
  │ • MMR retrieval, k=5, fetch_k=20     │
  │ • λ=0.6 (60% relevance, 40% diversity)│
  └────────────────────────────────────────┘
          │ 5 diverse chunks
          ▼
  LLM Scoring Agent (Llama 3.3 70B via Nebius)
  ┌────────────────────────────────────────┐
  │ System prompt: ARM™ + AVRE™ rubrics   │
  │ Input: use case + retrieved context   │
  │ Output: structured JSON scores        │
  └────────────────────────────────────────┘
          │
          ▼
  Scoring Engine (Python)
  ┌────────────────────────────────────────┐
  │ ARM™ composite = weighted average     │
  │ AVRE™ NEV = BR − (CDR × multiplier)  │
  │ Streamlit visual output + JSON export │
  └────────────────────────────────────────┘
```

---

## Datasets Used

All corpus documents were authored specifically for this project:

1. **arm_framework.txt** — Full ARM™ scoring rubric with five dimensions, scoring guide, mitigation requirements, risk tiers, and integration with AVRE™
2. **avre_framework.txt** — Full AVRE™ scoring rubric with ROI/ROE/ROF lenses, NEV formula, worked example, and opportunity cost dimension
3. **governance_frameworks.txt** — Summarized provisions of EU AI Act (Regulation 2024/1689), NIST AI RMF 1.0, and ISO 42001:2023
4. **use_case_reference.txt** — Six enterprise agentification use cases with typical ARM™ scores, AVRE™ ratings, key risks, and mitigation best practices

**Total corpus:** ~7,500 words → 59 chunks at 800-token chunk size with 120-token overlap

---

## Prompts Used During Development

**Core scoring system prompt:**
```
You are an expert Enterprise AI Governance analyst specializing in ARM™ 
(Agentification Risk Model) and AVRE™ (Agentic Value Realization Engine) 
frameworks. Score the use case across all ARM™ and AVRE™ dimensions. 
Return only valid JSON. Scores must be numeric (float), 0-10.
Higher ARM™ = higher risk. Higher AVRE™ = higher value.
```

**AI coding tools used:**
- Claude (claude.ai) was used extensively for code generation, debugging, architecture decisions, and iteration guidance throughout the build
- Every component was built interactively — code was generated, tested, errors were diagnosed, and fixes were applied in real time
- Key debugging sessions: ChromaDB → TF-IDF pivot, Anthropic → Nebius API switch, permissions fix on requirements.txt, OpenAI library integration

**Corpus authoring prompts used:**
- "Write a comprehensive ARM™ risk dimension scoring guide for [dimension] with specific examples at score ranges 1-2, 3-5, 6-8, and 9-10"
- "Generate a realistic use case reference entry for [use case] following the established ARM™/AVRE™ format with specific regulatory citations"
- "Create 15 RAG evaluation questions spanning easy direct framework queries, multi-document synthesis, edge cases, and out-of-corpus questions"

---

## Iterations and Learnings

**Iteration 1: ChromaDB + HuggingFace embeddings**
Initially planned to use ChromaDB with `all-MiniLM-L6-v2`. Hit a hard blocker: HuggingFace model downloads were unavailable. Forced a pivot to a custom solution.

**Iteration 2: TF-IDF vectorstore**
Rebuilt the vectorstore using sklearn's TfidfVectorizer. Key insight: for a governance corpus with precise technical terminology, TF-IDF with bigrams performs surprisingly well. The 81.2% faithfulness score validates this.

**Iteration 3: MMR retrieval over pure cosine similarity**
First pass used greedy top-k retrieval — all 5 chunks clustered around the ARM™ section, missing AVRE™ and regulatory context. Switched to MMR which penalizes near-duplicate chunk selection. Cross-document diversity improved significantly.

**Iteration 4: Chunk size tuning**
Started at 512 tokens. ARM™ scoring rubrics got split mid-criteria. Increased to 800 tokens with 120-token overlap — keeps rubrics intact while maintaining retrieval precision.

**Iteration 5: Anthropic → Nebius API**
Originally built for Anthropic API. Pivoted to Nebius Token Factory (course requirement) using OpenAI-compatible format. Switched to Llama 3.3 70B — minimal code change, same JSON output structure.

**Iteration 6: CLI → Streamlit UI**
Added Streamlit as a bonus deliverable. Designed a dark-theme enterprise UI with color-coded risk cards, animated score bars, and a prominent NEV verdict panel. Three pages: scoring, evaluation report, about.

**Key failure (Q11):** GPAI synthetic data question retrieved only 40% of expected concepts. Root cause: topic spans two distant chunks in the same document — MMR correctly avoided over-sampling but missed the second relevant chunk. Fix for v2: increase k to 7 for edge case patterns.

---

## Evaluation Results Summary

| Metric | Result |
|--------|--------|
| Total questions | 15 |
| Scoreable (excluding out-of-corpus) | 14 |
| Average faithfulness | **81.2%** |
| Source retrieval accuracy | **100%** |
| Questions passing ≥50% faithfulness | **13/14** |
| Only failure | Q11 (GPAI synthetic data, hard, 40%) |

**Retrieval strategy:** MMR, k=5, fetch_k=20, λ=0.6  
**Chunk strategy:** Recursive, 800 tokens, 120 overlap  
**Embedding:** TF-IDF + bigrams, 8,000 features, sublinear TF  
**LLM:** Llama 3.3 70B via Nebius Token Factory

---

## Week 3 Extension Plan

This project is explicitly designed as the RAG backbone for Week 3:

**Week 3 Agent: AI Governance Intelligence Agent**
- `score_use_case` tool → calls this RAG pipeline
- `compliance_check` tool → regulatory lookup for applicable laws by jurisdiction + industry
- `web_search` tool → real-time regulatory news retrieval
- `generate_governance_brief` tool → structured report output
- Orchestrated via LangGraph state machine
- HITL gate before final report delivery: human reviews risk tier before it's logged

This is not a one-off project — it's a building block in a larger platform.

---
## Submission Links
- **GitHub:** https://github.com/soumyavj/agentification-value-scorer-MAD-Week-2 
- **Video demo:**  
- **Eval report JSON:** Run `python main.py --eval` to regenerate. 
  Results summary is in this document above.
