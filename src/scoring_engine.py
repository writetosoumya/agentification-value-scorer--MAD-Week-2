"""
scoring_engine.py
ARM™ and AVRE™ scoring logic.
Deterministic scoring rubrics + RAG-grounded LLM scoring via Anthropic API.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ARMScores:
    autonomy_risk: float = 0.0          # 0-10
    data_governance_risk: float = 0.0   # 0-10
    compliance_exposure_risk: float = 0.0  # 0-10
    capability_debt_risk: float = 0.0   # 0-10
    equity_bias_risk: float = 0.0       # 0-10

    def composite(self) -> float:
        weights = {
            "autonomy_risk": 0.25,
            "data_governance_risk": 0.20,
            "compliance_exposure_risk": 0.20,
            "capability_debt_risk": 0.20,
            "equity_bias_risk": 0.15,
        }
        return round(
            self.autonomy_risk * weights["autonomy_risk"]
            + self.data_governance_risk * weights["data_governance_risk"]
            + self.compliance_exposure_risk * weights["compliance_exposure_risk"]
            + self.capability_debt_risk * weights["capability_debt_risk"]
            + self.equity_bias_risk * weights["equity_bias_risk"],
            2,
        )

    def risk_tier(self) -> str:
        score = self.composite()
        if score <= 3.0:
            return "🟢 GREEN — Proceed with standard governance"
        elif score <= 5.5:
            return "🟡 AMBER — Proceed with enhanced controls"
        elif score <= 7.5:
            return "🔴 RED — Phased deployment only. Independent review required."
        else:
            return "⛔ CRITICAL — Do not deploy without executive sign-off and external audit"

    def risk_multiplier(self) -> float:
        return round(1 + (self.composite() / 10), 3)


@dataclass
class AVREScores:
    roi_score: float = 0.0   # 0-10
    roe_score: float = 0.0   # 0-10
    rof_score: float = 0.0   # 0-10
    opportunity_cost: float = 0.0  # 0-10

    def benefit_realization(self) -> float:
        return round(
            self.roi_score * 0.40
            + self.roe_score * 0.30
            + self.rof_score * 0.30,
            2,
        )


@dataclass
class ScoringResult:
    use_case_name: str
    use_case_description: str
    arm: ARMScores = field(default_factory=ARMScores)
    avre: AVREScores = field(default_factory=AVREScores)
    rationale: Dict[str, str] = field(default_factory=dict)
    rag_sources: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def net_enterprise_value(self) -> float:
        capability_debt = self.arm.capability_debt_risk
        multiplier = self.arm.risk_multiplier()
        br = self.avre.benefit_realization()
        raw_nev = br - (capability_debt * multiplier * 0.15)  # normalized penalty
        return round(max(0.0, min(10.0, raw_nev * 1.2)), 2)  # normalize to 0-10

    def nev_rating(self) -> str:
        nev = self.net_enterprise_value()
        if nev >= 7.0:
            return "⭐ HIGH VALUE — Prioritize for agentification"
        elif nev >= 5.0:
            return "✅ MODERATE VALUE — Proceed with phased approach"
        elif nev >= 3.0:
            return "⚠️  CONDITIONAL VALUE — Resolve blockers before committing"
        else:
            return "❌ LOW VALUE — De-prioritize. Consider alternative automation."

    def to_dict(self) -> dict:
        return {
            "use_case": self.use_case_name,
            "description": self.use_case_description,
            "arm_scores": {
                "autonomy_risk": self.arm.autonomy_risk,
                "data_governance_risk": self.arm.data_governance_risk,
                "compliance_exposure_risk": self.arm.compliance_exposure_risk,
                "capability_debt_risk": self.arm.capability_debt_risk,
                "equity_bias_risk": self.arm.equity_bias_risk,
                "composite_arm_score": self.arm.composite(),
                "risk_tier": self.arm.risk_tier(),
                "risk_multiplier": self.arm.risk_multiplier(),
            },
            "avre_scores": {
                "roi_score": self.avre.roi_score,
                "roe_score": self.avre.roe_score,
                "rof_score": self.avre.rof_score,
                "opportunity_cost": self.avre.opportunity_cost,
                "benefit_realization": self.avre.benefit_realization(),
            },
            "net_enterprise_value": self.net_enterprise_value(),
            "nev_rating": self.nev_rating(),
            "rationale": self.rationale,
            "recommendations": self.recommendations,
            "rag_sources_used": self.rag_sources,
        }

    def pretty_print(self):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()

        console.print(f"\n[bold cyan]═══ AGENTIFICATION VALUE SCORE REPORT ═══[/bold cyan]")
        console.print(f"[bold]Use Case:[/bold] {self.use_case_name}")
        console.print(f"[dim]{self.use_case_description}[/dim]\n")

        # ARM table
        arm_table = Table(title="ARM™ Risk Assessment", box=box.ROUNDED, show_header=True)
        arm_table.add_column("Dimension", style="bold")
        arm_table.add_column("Score (0-10)", justify="center")
        arm_table.add_column("Weight", justify="center")
        arm_table.add_column("Rationale", max_width=50)

        dim_map = [
            ("Autonomy Risk", self.arm.autonomy_risk, "25%", "autonomy_risk"),
            ("Data Governance Risk", self.arm.data_governance_risk, "20%", "data_governance_risk"),
            ("Compliance Exposure Risk", self.arm.compliance_exposure_risk, "20%", "compliance_exposure_risk"),
            ("Capability Debt Risk", self.arm.capability_debt_risk, "20%", "capability_debt_risk"),
            ("Equity & Bias Risk", self.arm.equity_bias_risk, "15%", "equity_bias_risk"),
        ]

        def score_color(s):
            if s <= 3: return "green"
            elif s <= 5.5: return "yellow"
            elif s <= 7.5: return "red"
            else: return "bright_red"

        for label, score, weight, key in dim_map:
            arm_table.add_row(
                label,
                f"[{score_color(score)}]{score:.1f}[/{score_color(score)}]",
                weight,
                self.rationale.get(key, "—")[:80],
            )

        arm_table.add_section()
        arm_table.add_row(
            "[bold]COMPOSITE ARM™ SCORE[/bold]",
            f"[bold]{self.arm.composite():.2f}[/bold]",
            "100%",
            self.arm.risk_tier(),
        )
        console.print(arm_table)

        # AVRE table
        avre_table = Table(title="AVRE™ Value Assessment", box=box.ROUNDED)
        avre_table.add_column("Lens", style="bold")
        avre_table.add_column("Score (0-10)", justify="center")
        avre_table.add_column("Weight", justify="center")
        avre_table.add_column("Rationale", max_width=50)

        avre_map = [
            ("ROI (Return on Investment)", self.avre.roi_score, "40%", "roi"),
            ("ROE (Return on Effort)", self.avre.roe_score, "30%", "roe"),
            ("ROF (Return on Freedom / Optionality)", self.avre.rof_score, "30%", "rof"),
            ("Opportunity Cost (not agentifying)", self.avre.opportunity_cost, "ref", "opportunity_cost"),
        ]
        for label, score, weight, key in avre_map:
            avre_table.add_row(
                label,
                f"{score:.1f}",
                weight,
                self.rationale.get(key, "—")[:80],
            )
        avre_table.add_section()
        avre_table.add_row(
            "[bold]BENEFIT REALIZATION SCORE[/bold]",
            f"[bold]{self.avre.benefit_realization():.2f}[/bold]",
            "",
            "",
        )
        console.print(avre_table)

        # NEV summary
        nev = self.net_enterprise_value()
        nev_color = "green" if nev >= 7 else "yellow" if nev >= 5 else "red"
        console.print(Panel(
            f"[bold {nev_color}]NET ENTERPRISE VALUE: {nev:.2f} / 10[/bold {nev_color}]\n"
            f"{self.nev_rating()}\n\n"
            f"ARM™ Risk Multiplier: {self.arm.risk_multiplier():.3f}\n"
            f"Sources used: {', '.join(self.rag_sources)}",
            title="[bold]AVRE™ Final Verdict[/bold]",
            border_style=nev_color,
        ))

        if self.recommendations:
            console.print("\n[bold]📋 Recommendations:[/bold]")
            for i, rec in enumerate(self.recommendations, 1):
                console.print(f"  {i}. {rec}")
        console.print()
