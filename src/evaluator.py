"""
evaluator.py
15-question evaluation suite for the RAG-powered Agentification Value Scorer.
Tests retrieval quality, faithfulness, edge cases, and scoring consistency.
Week 2 deliverable: evaluation report generator.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich import box

from rag_engine import retrieve_context

console = Console()

# ─── Evaluation questions ─────────────────────────────────────────────────────
# Format: query, expected_sources (at least one must appear), expected_concepts

EVAL_QUESTIONS = [
    # ── Happy path: direct framework queries ──
    {
        "id": "Q01",
        "category": "ARM™ Framework",
        "query": "What is the ARM™ autonomy risk score for an agent that automatically initiates financial transfers without human approval?",
        "expected_concepts": ["autonomy risk", "critical", "9", "10", "irreversible", "human"],
        "expected_sources": ["arm_framework"],
        "difficulty": "easy",
    },
    {
        "id": "Q02",
        "category": "ARM™ Framework",
        "query": "How does the data governance risk dimension score an agent that processes HIPAA-regulated health data?",
        "expected_concepts": ["data governance", "hipaa", "regulated", "6", "7", "8", "DPIA"],
        "expected_sources": ["arm_framework"],
        "difficulty": "easy",
    },
    {
        "id": "Q03",
        "category": "AVRE™ Framework",
        "query": "What is the AVRE™ formula for Net Enterprise Value and how does ARM™ influence it?",
        "expected_concepts": ["net enterprise value", "benefit realization", "capability debt", "multiplier", "risk penalty"],
        "expected_sources": ["avre_framework"],
        "difficulty": "easy",
    },
    {
        "id": "Q04",
        "category": "AVRE™ Framework",
        "query": "Explain the Return on Freedom (ROF) lens in AVRE™ and what a score of 9 means.",
        "expected_concepts": ["optionality", "platform effect", "reusable", "competitive", "future deployments"],
        "expected_sources": ["avre_framework"],
        "difficulty": "easy",
    },
    {
        "id": "Q05",
        "category": "EU AI Act",
        "query": "Is an AI agent used for employee performance monitoring and termination decisions high-risk under the EU AI Act?",
        "expected_concepts": ["high-risk", "annex iii", "employment", "worker management", "article 6"],
        "expected_sources": ["governance_frameworks"],
        "difficulty": "medium",
    },
    # ── Cross-document synthesis ──
    {
        "id": "Q06",
        "category": "Cross-framework",
        "query": "What ARM™ score range would you expect for a resume screening agent and why?",
        "expected_concepts": ["equity", "bias", "compliance", "employment", "7", "8", "9", "critical"],
        "expected_sources": ["arm_framework", "use_case_reference"],
        "difficulty": "medium",
    },
    {
        "id": "Q07",
        "category": "Cross-framework",
        "query": "How does NIST AI RMF GOVERN function relate to setting up human-in-the-loop requirements for an agentic system?",
        "expected_concepts": ["govern", "accountability", "policies", "roles", "responsibilities", "human oversight"],
        "expected_sources": ["governance_frameworks"],
        "difficulty": "medium",
    },
    {
        "id": "Q08",
        "category": "Cross-framework",
        "query": "What mitigation is required when an ARM™ composite score exceeds 6.0?",
        "expected_concepts": ["phased", "independent review", "red", "executive", "audit"],
        "expected_sources": ["arm_framework"],
        "difficulty": "medium",
    },
    {
        "id": "Q09",
        "category": "Use Case Reference",
        "query": "What are the key risks and mitigation best practices for a fraud detection agent that can autonomously place account holds?",
        "expected_concepts": ["false positive", "SR 11-7", "DORA", "model drift", "backtesting", "threshold"],
        "expected_sources": ["use_case_reference"],
        "difficulty": "medium",
    },
    {
        "id": "Q10",
        "category": "Use Case Reference",
        "query": "Why does the Project Portfolio Management agent typically score GREEN on ARM™ while a fraud detection agent scores RED?",
        "expected_concepts": ["autonomy", "read only", "individual", "financial", "irreversible", "consequence"],
        "expected_sources": ["use_case_reference", "arm_framework"],
        "difficulty": "medium",
    },
    # ── Edge cases ──
    {
        "id": "Q11",
        "category": "Edge Case",
        "query": "An agent that generates synthetic training data — what is its compliance exposure risk under the EU AI Act GPAI model provisions?",
        "expected_concepts": ["general purpose", "GPAI", "technical documentation", "training data", "copyright"],
        "expected_sources": ["governance_frameworks"],
        "difficulty": "hard",
    },
    {
        "id": "Q12",
        "category": "Edge Case",
        "query": "What happens to AVRE™ Net Enterprise Value when capability debt is high but benefit realization is also high?",
        "expected_concepts": ["multiplier", "penalizes", "net enterprise value", "capability debt", "risk"],
        "expected_sources": ["avre_framework"],
        "difficulty": "hard",
    },
    {
        "id": "Q13",
        "category": "Edge Case",
        "query": "A voice agent for IT support that asks employees to confirm their identity — what equity and bias risks apply?",
        "expected_concepts": ["equity", "bias", "protected", "demographic", "accent", "language"],
        "expected_sources": ["arm_framework", "use_case_reference"],
        "difficulty": "hard",
    },
    # ── Out-of-corpus / graceful failure ──
    {
        "id": "Q14",
        "category": "Out of Corpus",
        "query": "What is the specific fine amount for violating Article 5 prohibited AI practices under the EU AI Act?",
        "expected_concepts": [],  # Specific fine amounts not in corpus — should surface partial or adjacent info
        "expected_sources": ["governance_frameworks"],
        "difficulty": "out-of-corpus",
        "note": "Fine amounts not in corpus. Retriever should surface EU AI Act content but specific amounts unavailable — tests graceful handling."
    },
    {
        "id": "Q15",
        "category": "Multi-hop",
        "query": "For a healthcare triage agent scoring 9 on equity and bias risk, what is the ARM™ risk tier, and what does that imply for AVRE™ Net Enterprise Value?",
        "expected_concepts": ["critical", "9", "multiplier", "1.9", "net enterprise value", "equity"],
        "expected_sources": ["arm_framework", "avre_framework"],
        "difficulty": "hard",
    },
]


@dataclass
class EvalResult:
    question_id: str
    category: str
    query: str
    difficulty: str
    retrieved_sources: List[str]
    retrieved_context_preview: str
    concept_hits: int
    concept_total: int
    source_hit: bool
    faithfulness_score: float  # 0-1: fraction of expected concepts found
    retrieval_score: float     # 1 if at least one expected source retrieved, else 0
    notes: str = ""


def evaluate_retrieval(query_dict: dict, retriever) -> EvalResult:
    """Run a single retrieval evaluation."""
    result = retrieve_context(query_dict["query"], retriever)

    context_lower = result["context"].lower()
    expected_concepts = query_dict.get("expected_concepts", [])
    expected_sources = query_dict.get("expected_sources", [])

    # Concept hit rate (proxy for faithfulness)
    hits = sum(1 for c in expected_concepts if c.lower() in context_lower)
    faithfulness = round(hits / len(expected_concepts), 2) if expected_concepts else 1.0

    # Source hit
    source_hit = any(src in result["sources"] for src in expected_sources) if expected_sources else True

    return EvalResult(
        question_id=query_dict["id"],
        category=query_dict["category"],
        query=query_dict["query"],
        difficulty=query_dict["difficulty"],
        retrieved_sources=result["sources"],
        retrieved_context_preview=result["context"][:300],
        concept_hits=hits,
        concept_total=len(expected_concepts),
        source_hit=source_hit,
        faithfulness_score=faithfulness,
        retrieval_score=1.0 if source_hit else 0.0,
        notes=query_dict.get("note", ""),
    )


def run_evaluation_suite(retriever):
    """Run all 15 evaluation questions and generate a report."""
    console.print("\n[bold cyan]═══ RAG EVALUATION SUITE — 15 Questions ═══[/bold cyan]\n")
    console.print(f"Testing retrieval quality, faithfulness, source accuracy, and edge case handling.\n")

    results: List[EvalResult] = []

    for q in EVAL_QUESTIONS:
        console.print(f"[dim]Running {q['id']}: {q['query'][:70]}...[/dim]")
        r = evaluate_retrieval(q, retriever)
        results.append(r)
        time.sleep(0.1)  # small pause for readability

    # ── Results table ──
    table = Table(
        title="\n📊 RAG Evaluation Results — Agentification Value Scorer",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("ID", style="bold", width=4)
    table.add_column("Category", width=18)
    table.add_column("Difficulty", width=12)
    table.add_column("Faithfulness", justify="center", width=12)
    table.add_column("Source Hit", justify="center", width=10)
    table.add_column("Retrieved Sources", width=30)
    table.add_column("Notes", width=25)

    def faith_color(s):
        if s >= 0.75: return "green"
        elif s >= 0.5: return "yellow"
        else: return "red"

    for r in results:
        faith_pct = f"{r.faithfulness_score*100:.0f}%  ({r.concept_hits}/{r.concept_total})"
        source_disp = "✅" if r.source_hit else "❌"
        table.add_row(
            r.question_id,
            r.category,
            r.difficulty,
            f"[{faith_color(r.faithfulness_score)}]{faith_pct}[/{faith_color(r.faithfulness_score)}]",
            source_disp,
            ", ".join(r.retrieved_sources),
            r.notes[:40] if r.notes else "—",
        )

    console.print(table)

    # ── Aggregate metrics ──
    # Exclude out-of-corpus from faithfulness aggregate
    scoreable = [r for r in results if r.difficulty != "out-of-corpus"]
    avg_faith = sum(r.faithfulness_score for r in scoreable) / len(scoreable)
    avg_source = sum(r.retrieval_score for r in results) / len(results)
    total_pass = sum(1 for r in scoreable if r.faithfulness_score >= 0.5)

    console.print(f"\n[bold]═══ Aggregate Metrics ═══[/bold]")
    console.print(f"  Questions evaluated:        {len(results)}")
    console.print(f"  Scoreable (excl. OOC):      {len(scoreable)}")
    console.print(f"  Avg faithfulness score:     [{faith_color(avg_faith)}]{avg_faith*100:.1f}%[/{faith_color(avg_faith)}]")
    console.print(f"  Source retrieval accuracy:  [{faith_color(avg_source)}]{avg_source*100:.1f}%[/{faith_color(avg_source)}]")
    console.print(f"  Questions passing (≥50%):   {total_pass}/{len(scoreable)}")

    # ── Failure analysis ──
    failures = [r for r in scoreable if r.faithfulness_score < 0.5]
    if failures:
        console.print(f"\n[bold red]Failure Analysis ({len(failures)} questions below 50% faithfulness):[/bold red]")
        for f in failures:
            console.print(f"  • {f.question_id} ({f.category}, {f.difficulty}): {f.faithfulness_score*100:.0f}%")
            console.print(f"    Query: {f.query[:80]}...")
            console.print(f"    Retrieved sources: {f.retrieved_sources}")
            console.print(f"    Root cause: Concepts may span multiple documents requiring multi-hop retrieval. "
                         f"MMR retrieval with k=5 may not surface all relevant chunks for complex queries.")

    # Save JSON report
    output = {
        "evaluation_summary": {
            "total_questions": len(results),
            "avg_faithfulness_pct": round(avg_faith * 100, 1),
            "source_retrieval_accuracy_pct": round(avg_source * 100, 1),
            "questions_passing_50pct": f"{total_pass}/{len(scoreable)}",
            "embed_model": "all-MiniLM-L6-v2",
            "retrieval_strategy": "MMR (Maximal Marginal Relevance), k=5",
            "chunk_size": 800,
            "chunk_overlap": 120,
        },
        "results": [
            {
                "id": r.question_id,
                "category": r.category,
                "query": r.query,
                "difficulty": r.difficulty,
                "faithfulness_score": r.faithfulness_score,
                "concept_hits": r.concept_hits,
                "concept_total": r.concept_total,
                "source_hit": r.source_hit,
                "retrieved_sources": r.retrieved_sources,
                "context_preview": r.retrieved_context_preview,
                "notes": r.notes,
            }
            for r in results
        ],
    }

    out_path = Path(__file__).parent.parent / "outputs" / "eval_report.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[dim]Evaluation report saved to {out_path}[/dim]")
    return output
