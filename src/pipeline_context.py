"""Per-run context — replaces the ``self._last_*`` implicit state pattern.

Previously `TradingPipeline` stashed cross-stage data on its own instance
(``self._last_symbols_bars``, ``self._bg_threads``). That conflated per-run
state with the long-lived service container, making runs non-reentrant,
hard to test, and hard to reason about when one stage's output is
another stage's input.

`RunContext` is an explicit container created at the start of each run.
Every stage reads from it and writes to it by field name. Stages become
functions of ``(ctx, deps) -> ctx-with-fields-filled-in`` rather than
methods that rely on implicit attributes of the enclosing instance.

This module ships the dataclass only — it does not (yet) refactor the
pipeline into explicit stages. That's Phase 2 of the architecture work.
For Phase 1 the goal is just to remove implicit state and give each run
its own mutable snapshot.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.models import NewsIntelligenceReport, PortfolioDecision, Position

SessionType = Literal["morning", "midday", "evening", "intra_check"]


@dataclass
class RunContext:
    """Per-run snapshot of everything a session needs.

    Not frozen — stages populate fields as the run progresses. Discipline is
    "each field has one owning stage that writes it; other stages read only."
    """

    run_id: str
    session: SessionType
    started_at: datetime = field(default_factory=datetime.utcnow)

    
    account: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)  
    cash: float = 0.0
    total_value: float = 0.0
    last_equity: float = 0.0

    
    macro_summary: dict = field(default_factory=dict)
    macro_analysis: dict | None = None  
    news_intel: "NewsIntelligenceReport | None" = None
    analyses: list = field(default_factory=list)  
    earnings_results: list[dict] = field(default_factory=list)
    symbols_bars: dict = field(default_factory=dict)  
    valuations: dict = field(default_factory=dict)  
    data_status: dict[str, str] = field(default_factory=dict)

    
    portfolio_decision: "PortfolioDecision | None" = None
    correlation_matrix: dict = field(default_factory=dict)
    daily_pnl: float = 0.0
    macro_target_pct: float | None = None

    
    orders: list[dict] = field(default_factory=list)

    
    
    
    
    bg_threads: list[threading.Thread] = field(default_factory=list)

    @classmethod
    def start(cls, session: SessionType) -> "RunContext":
        """Build a fresh context for a new session.

        Run ID prefix matches legacy formatting so log greps like
        'run-abcd1234' and 'midday-abcd1234' keep working.
        """
        rid_prefix = "run" if session == "morning" else session
        return cls(
            run_id=f"{rid_prefix}-{uuid.uuid4().hex[:8]}",
            session=session,
        )
