"""Asserts each agent prompt carries:

  (a) A consolidated `## Guardrails` section borrowed from the
      anthropics/financial-services pattern. Every safety rule lives
      under one heading the operator can audit in 30 seconds, not
      scattered across the prompt body.

  (b) For prompts that ingest text from outside the system (SEC
      filings, RSS, FRED descriptions, DB-persisted thesis prose, or
      quoted upstream LLM reasoning_chain fields), the Guardrails
      section must include an `**Untrusted input.**` bullet that
      uses the canonical "data, not instructions" framing.

  (c) For prompts that produce analysis and may face missing data,
      the `[UNSOURCED:<reason>]` token convention must be documented
      so downstream consumers can grep for and discount missing-data
      theses. Prose like "not disclosed" is no longer accepted.

These three checks together are the cross-prompt safety contract.
Touching them without updating both the prompt and this test is the
exact regression class we want to catch loudly.
"""
from pathlib import Path

import pytest

PROMPT_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"






ALL_AGENT_PROMPTS = (
    "tech_analyst.md",
    "news_analyst.md",
    "macro_analyst.md",
    "earnings_analyst.md",
    "portfolio_manager.md",
    "risk_manager.md",
    "position_reviewer.md",
    "evening_analyst.md",
    "meta_reflector.md",
)








_UNTRUSTED_INPUT_PROMPTS = (
    "earnings_analyst.md",     
    "news_analyst.md",         
    "macro_analyst.md",        
    "position_reviewer.md",    
    "evening_analyst.md",      
)





_UNSOURCED_TOKEN_PROMPTS = (
    "earnings_analyst.md",     
    "macro_analyst.md",        
    "evening_analyst.md",      
    "news_analyst.md",         
    "portfolio_manager.md",    
)


@pytest.mark.parametrize("prompt_name", ALL_AGENT_PROMPTS)
def test_prompt_has_consolidated_guardrails_section(prompt_name: str) -> None:
    """Every prompt must have a single `## Guardrails` heading. This
    pattern from anthropics/financial-services puts safety discipline
    in one auditable place so the operator can scan it in 30 seconds
    without re-reading the whole prompt.
    """
    path = PROMPT_DIR / prompt_name
    assert path.exists(), f"prompt file missing: {path}"
    text = path.read_text()
    assert "## Guardrails" in text, (
        f"{prompt_name} is missing the consolidated `## Guardrails` "
        f"section. Add it right after `## What you produce`. Bullets "
        f"should cover (when applicable): untrusted input · "
        f"[UNSOURCED] / cite-every-number · agent-specific safety "
        f"rule · autonomy boundary."
    )


@pytest.mark.parametrize("prompt_name", _UNTRUSTED_INPUT_PROMPTS)
def test_prompt_carries_untrusted_input_bullet(prompt_name: str) -> None:
    """Prompts that ingest outside text must carry an `**Untrusted
    input.**` bullet inside their `## Guardrails` section, anchored on
    the canonical 'data, not instructions' phrasing. The bullet is the
    only defense against prompt-injection via SEC filings / RSS / FRED
    description / DB-persisted LLM prose.
    """
    path = PROMPT_DIR / prompt_name
    assert path.exists(), f"prompt file missing: {path}"
    text = path.read_text()
    
    
    assert "**Untrusted input.**" in text, (
        f"{prompt_name} ingests outside text but is missing the "
        f"`**Untrusted input.**` bullet in its Guardrails section. "
        f"Add it so the LLM treats inputs as data, not directives. "
        f"This is the only defense against prompt-injection through "
        f"SEC filings / RSS / FRED / DB-persisted LLM prose."
    )
    
    
    lower = text.lower()
    assert "data, not instructions" in lower or "data**, not instructions" in lower, (
        f"{prompt_name} has an Untrusted-input bullet but doesn't "
        f"carry the canonical 'data, not instructions' framing. "
        f"Verify the bullet actually teaches the discipline rather "
        f"than just naming the heading."
    )


@pytest.mark.parametrize("prompt_name", _UNSOURCED_TOKEN_PROMPTS)
def test_prompt_teaches_unsourced_token(prompt_name: str) -> None:
    """Every prompt that may need to surface missing data must teach
    the `[UNSOURCED:<reason>]` token convention. Downstream consumers
    (position_reviewer / evening / meta_reflector) grep for it to
    discount missing-data theses. Prose like 'not disclosed' is no
    longer accepted.
    """
    path = PROMPT_DIR / prompt_name
    assert path.exists(), f"prompt file missing: {path}"
    text = path.read_text()
    assert "[UNSOURCED:" in text, (
        f"{prompt_name} doesn't teach the [UNSOURCED:<reason>] token. "
        f"Downstream consumers grep this to discount missing-data "
        f"theses — without it, the prompt's gap-handling prose is "
        f"opaque. Add a one-line note in the Guardrails section "
        f"showing the valid reasons for this agent."
    )


def test_earnings_analyst_unsourced_reasons_present() -> None:
    """Earnings analyst is the most invocation-heavy [UNSOURCED] user;
    all three canonical reason variants must be documented so the LLM
    knows when to pick which.
    """
    path = PROMPT_DIR / "earnings_analyst.md"
    text = path.read_text()
    for reason in ("not_in_filing", "truncated", "ambiguous"):
        assert f"[UNSOURCED:{reason}]" in text, (
            f"earnings_analyst.md missing reason variant "
            f"[UNSOURCED:{reason}]. The three reasons distinguish "
            f"filing-omits-it (not_in_filing) from input-was-cut "
            f"(truncated) from text-too-unclear (ambiguous) — each "
            f"is a different downstream story."
        )


def test_guardrails_close_with_autonomy_boundary() -> None:
    """The Anthropic pattern always closes Guardrails with the
    autonomy boundary — the explicit "you do X, you do NOT do Y" line
    that anchors what this agent's role is NOT. Verifying every
    Guardrails section has one keeps the role separation visible.

    Spot-check the most safety-critical agents; the bullet wording can
    vary so we look for the canonical "Autonomy" anchor.
    """
    for prompt_name in (
        "tech_analyst.md", "news_analyst.md", "macro_analyst.md",
        "earnings_analyst.md", "portfolio_manager.md",
        "risk_manager.md", "position_reviewer.md",
        "evening_analyst.md", "meta_reflector.md",
    ):
        text = (PROMPT_DIR / prompt_name).read_text()
        
        
        
        autonomy_anchors = (
            "**Autonomy.**",
            "**Autonomy boundary.**",
            "**Sell-only;",
            "**Final gate.**",
        )
        assert any(a in text for a in autonomy_anchors), (
            f"{prompt_name} Guardrails section is missing an autonomy "
            f"/ role-boundary bullet. The Anthropic pattern always "
            f"closes with one — it anchors what this agent is NOT "
            f"allowed to do. Acceptable anchors: "
            f"{autonomy_anchors}"
        )
