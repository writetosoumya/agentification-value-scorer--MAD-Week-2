"""
llm_scorer.py
LLM-powered scoring agent using Nebius API (OpenAI-compatible).
Receives RAG context + use case description → returns structured ARM™ and AVRE™ scores.
"""

import json
import re
import os
from typing import Dict, Any
from openai import OpenAI

from scoring_engine import ARMScores, AVREScores, ScoringResult

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

SYSTEM_PROMPT = """You are an expert Enterprise AI Governance analyst specializing in ARM™ (Agentification Risk Model) and AVRE™ (Agentic Value Realization Engine) frameworks.

Your job is to score a business use case for agentification readiness based on:
1. The ARM™ framework (5 risk dimensions, each 0-10)
2. The AVRE™ framework (ROI, ROE, ROF value lenses, each 0-10, plus opportunity cost 0-10)

You will receive:
- The use case name and description
- Relevant governance context retrieved from the ARM™ and AVRE™ framework documents

SCORING RULES:
- Be precise and evidence-based. Reference specific framework criteria in your rationale.
- Scores must be numeric (float), between 0 and 10.
- For ARM™: Higher score = HIGHER risk (bad)
- For AVRE™: Higher score = HIGHER value (good), EXCEPT opportunity_cost where higher = more urgency to act

OUTPUT FORMAT (strict JSON, no markdown, no preamble, no explanation outside the JSON):
{
  "arm": {
    "autonomy_risk": <float 0-10>,
    "data_governance_risk": <float 0-10>,
    "compliance_exposure_risk": <float 0-10>,
    "capability_debt_risk": <float 0-10>,
    "equity_bias_risk": <float 0-10>
  },
  "avre": {
    "roi_score": <float 0-10>,
    "roe_score": <float 0-10>,
    "rof_score": <float 0-10>,
    "opportunity_cost": <float 0-10>
  },
  "rationale": {
    "autonomy_risk": "<1-2 sentence rationale>",
    "data_governance_risk": "<1-2 sentence rationale>",
    "compliance_exposure_risk": "<1-2 sentence rationale>",
    "capability_debt_risk": "<1-2 sentence rationale>",
    "equity_bias_risk": "<1-2 sentence rationale>",
    "roi": "<1-2 sentence rationale>",
    "roe": "<1-2 sentence rationale>",
    "rof": "<1-2 sentence rationale>",
    "opportunity_cost": "<1-2 sentence rationale>"
  },
  "recommendations": [
    "<actionable recommendation 1>",
    "<actionable recommendation 2>",
    "<actionable recommendation 3>"
  ]
}"""


def call_nebius(user_message: str) -> str:
    """Make a call to the Nebius API using OpenAI-compatible client."""
    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("NEBIUS_API_KEY"),
    )
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{text[:300]}")
    return json.loads(text[start:end])


def score_use_case(
    use_case_name: str,
    use_case_description: str,
    rag_context: str,
    rag_sources: list,
) -> ScoringResult:
    """Score a use case using RAG-grounded LLM scoring."""
    user_message = f"""USE CASE TO SCORE:
Name: {use_case_name}
Description: {use_case_description}

RETRIEVED GOVERNANCE CONTEXT (from ARM™, AVRE™, EU AI Act, NIST AI RMF frameworks):
{rag_context}

Score this use case now. Return only valid JSON."""

    raw_response = call_nebius(user_message)

    try:
        parsed = extract_json(raw_response)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Failed to parse LLM scoring response: {e}\nRaw: {raw_response[:500]}")

    arm = ARMScores(
        autonomy_risk=float(parsed["arm"]["autonomy_risk"]),
        data_governance_risk=float(parsed["arm"]["data_governance_risk"]),
        compliance_exposure_risk=float(parsed["arm"]["compliance_exposure_risk"]),
        capability_debt_risk=float(parsed["arm"]["capability_debt_risk"]),
        equity_bias_risk=float(parsed["arm"]["equity_bias_risk"]),
    )

    avre = AVREScores(
        roi_score=float(parsed["avre"]["roi_score"]),
        roe_score=float(parsed["avre"]["roe_score"]),
        rof_score=float(parsed["avre"]["rof_score"]),
        opportunity_cost=float(parsed["avre"]["opportunity_cost"]),
    )

    return ScoringResult(
        use_case_name=use_case_name,
        use_case_description=use_case_description,
        arm=arm,
        avre=avre,
        rationale=parsed.get("rationale", {}),
        rag_sources=rag_sources,
        recommendations=parsed.get("recommendations", []),
    )
