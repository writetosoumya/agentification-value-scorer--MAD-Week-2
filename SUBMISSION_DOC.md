# Week 2 Project: Agentification Value Scorer
## RAG-Powered Governance Intelligence | The Gen Academy Mastering Agentic AI Bootcamp

**Author:** Agni Sivaraman  
**Date:** June 2026  
**Submission track:** Code-heavy — LangChain + custom TF-IDF vectorstore

---

## Project Overview

The **Agentification Value Scorer** is a RAG-powered governance intelligence tool that scores any enterprise business use case for AI agent readiness. Instead of building a generic Q&A bot or document retrieval system, I built a tool that directly extends my original proprietary frameworks: **ARM™ (Agentification Risk Model)** and **AVRE™ (Agentic Value Realization Engine)**.

### The one-liner

> My RAG app helps **enterprise AI governance practitioners and business leaders** answer **"should we agentify this use case, and at what risk?"** from **governance frameworks, regulatory documents, and use case reference patterns** in a **CLI and web interface** with **≥75% faithfulness** and **≤10 second response latency.**

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
| **Use case** | Answers "what ARM™ risk score and AVRE™ Net Enterprise Value does this use case produce?" for enterprise AI governance practitioners, via CLI and web interface |
| **Corpus** | 4 documents: ARM™ framework (original IP), AVRE™ framework (original IP), EU AI Act + NIST AI RMF + ISO 42001 (summarized governance reference), enterprise use case reference library |
| **Ingestion + cleaning** | Plain text files, UTF-8 encoded. No markup to strip. Corpus authored specifically for this RAG system to ensure terminological consistency. |
| **Ingestion + freshness** | Static for this version. Future: scheduled pull from regulatory update feeds. Corpus re-embedded when any source document changes. |
| **Chunking + embedding** | Recursive character splitter, 800 tokens, 120-token overlap, separators: `[\n\n, \n, ., space]`. Embedding: TF-IDF with bigrams (sklearn), 8,000 features, sublinear TF scaling. Local — no API calls for embedding. |
| **Retrieve** | Custom TF-IDF vectorstore (pickle-serialized). MMR (Maximal Marginal Relevance) retrieval: k=5, fetch_k=20, λ=0.6. Ensures cross-document diversity. |

---

## Build Track: LangChain + Custom Vectorstore

**Why not ChromaDB?** The deployment environment blocked HuggingFace model downloads and required a self-contained solution. Rather than using a lighter pre-built embedding, I implemented a TF-IDF vectorstore from scratch using scikit-learn's TfidfVectorizer with MMR-style diversity selection. This gave me full control over retrieval behavior and required zero external API calls for the embedding layer.

**Why TF-IDF over dense embeddings for this corpus?**

For a governance corpus with precise technical terminology (ARM™, AVRE™, EU AI Act Article references, NIST AI RMF function names), TF-IDF with bigrams is actually a strong choice:
- Exact term matching on regulatory language ("Annex III", "SR 11-7", "DPIA") is high-precision
- Bigrams capture multi-word governance concepts ("autonomy risk", "benefit realization", "capability debt")
- No hallucination risk from semantic approximation — if the term isn't in the corpus, it doesn't match

For Week 3, I plan to upgrade to a hybrid dense+sparse retrieval approach using a local embedding model.

---

## Architecture

```
User Input (use case name + description)
          │
          ▼
  RAG Retrieval Layer
  ┌────────────────────────────────────────┐
  │ TF-IDF Vectorstore (sklearn)           │
  │ • 59 chunks across 4 documents        │
  │ • MMR retrieval, k=5                  │
  │ • Cross-document diversity enforced   │
  └────────────────────────────────────────┘
          │ 5 relevant chunks
          ▼
  LLM Scoring Agent (Claude Sonnet via Anthropic API)
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
  │ Rich terminal output + JSON export    │
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

**Corpus authoring prompts used:**
- "Write a comprehensive ARM™ risk dimension scoring guide for [dimension] with specific examples at score ranges 1-2, 3-5, 6-8, and 9-10"
- "Generate a realistic use case reference entry for [use case] following the established ARM™/AVRE™ format with specific regulatory citations"
- "Create 15 RAG evaluation questions spanning easy direct framework queries, multi-document synthesis, edge cases, and out-of-corpus questions"

---

## Iterations and Learnings

**Iteration 1: ChromaDB + HuggingFace embeddings**
Initially planned to use ChromaDB with `all-MiniLM-L6-v2` — a standard, high-quality embedding model. Hit a hard blocker: HuggingFace model downloads are blocked in the deployment environment (403 Forbidden). This forced a pivot.

**Iteration 2: TF-IDF vectorstore**
Rebuilt the vectorstore using sklearn's TfidfVectorizer. Key insight: for a governance corpus with precise technical terminology and specific regulatory language, TF-IDF with bigrams performs surprisingly well. The 81.2% faithfulness score on 14 scoreable evaluation questions validates this.

**Iteration 3: MMR retrieval over pure cosine similarity**
First pass used greedy top-k retrieval. All 5 retrieved chunks clustered around the ARM™ risk dimension section, missing AVRE™ and regulatory context. Switched to MMR (Maximal Marginal Relevance) which penalizes retrieving chunks too similar to already-selected ones. Cross-document diversity improved significantly — all 15 evaluation questions retrieved chunks from at least 2 different source documents.

**Iteration 4: Chunk size tuning**
Started at 512 tokens. Found that ARM™ scoring rubrics (which have scoring guides spanning multiple tiers) got split mid-criteria, creating chunks that had the high-risk guidance but not the mitigation requirements. Increased to 800 tokens with 120-token overlap — this keeps scoring rubrics intact while maintaining retrieval precision.

**Key failure (Q11):** The question about synthetic training data and GPAI model compliance only retrieved 40% of expected concepts. Root cause: this topic sits at the intersection of the GPAI model section (governance_frameworks.txt) and the EU AI Act prohibited practices section — two chunks in the same document but far apart. MMR correctly avoided over-sampling the same document, but the specific technical concepts (copyright compliance, training data documentation for GPAI models) were in a third chunk that didn't make the top-5. Fix for v2: increase k to 7 for edge case query patterns.

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

---

## Week 3 Extension Plan

This project is explicitly designed as the RAG backbone for Week 3:

**Week 3 Agent: AI Governance Intelligence Agent**
- `score_use_case` tool → calls this RAG pipeline
- `compliance_check` tool → regulatory lookup for applicable laws by jurisdiction + industry
- `generate_governance_brief` tool → structured output in ARM™/AVRE™ report format
- Orchestrated via LangGraph state machine
- HITL gate before final report delivery: human reviews risk tier before it's logged

This is not a one-off project — it's a building block in a larger platform.

---

## Submission Links
- **GitHub:** [link]
- **Video demo:** [link]
- **Eval report JSON:** See `/outputs/eval_report.json` in repo
