from datetime import datetime, date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


def _normalize_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    return symbol


def _normalize_enum_case_fields(
    values,
    *,
    lower_fields: tuple[str, ...] = (),
    upper_fields: tuple[str, ...] = (),
):
    """Case-fold dict fields before Pydantic Literal validation.

    Pydantic ``Literal["high", "medium", "low"]`` is exact-match —
    ``"HIGH"`` or ``"Medium"`` raises ValidationError, which on the
    tech_analyst path silently drops that symbol's analysis (the
    chunk-level except catches and logs but does not surface
    upstream). Most prompts give examples in the expected case but
    LLMs occasionally drift, especially after a long CoT. Folding
    input case before the Literal check turns a cosmetic drift from
    "whole symbol lost" into a no-op.

    Only touches string values; non-string inputs (None / numbers /
    lists / dicts) pass through unchanged so Pydantic's own type
    errors still surface for genuinely malformed inputs.
    """
    if not isinstance(values, dict):
        return values
    for name in lower_fields:
        v = values.get(name)
        if isinstance(v, str):
            values[name] = v.strip().lower()
    for name in upper_fields:
        v = values.get(name)
        if isinstance(v, str):
            values[name] = v.strip().upper()
    return values


class OHLCV(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class TechnicalIndicators(BaseModel):
    symbol: str
    ma_20: float | None = None
    ma_50: float | None = None
    ma_200: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    atr_14: float | None = None
    volume_change_pct: float | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class TechReasoningChain(BaseModel):
    """5-step CoT for a single symbol — forces the LLM to show its work per
    framework step. Every field has `min_length=1` so the LLM cannot skip a
    step by sending an empty string. This matches the discipline already in
    place on the other CoT chains (Evening / Position / Meta) and closes
    the audit gap that contradicted the README's 'schema-enforced CoT,
    LLM cannot skip steps' claim.
    """
    trend: str = Field(min_length=1)                 
    momentum: str = Field(min_length=1)              
    volatility: str = Field(min_length=1)            
    volume: str = Field(min_length=1)                
    support_resistance: str = Field(min_length=1)    


class TechAnalysisResult(BaseModel):
    symbol: str
    rating: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"]
    conviction: Literal["high", "medium", "low"] = "medium"
    entry_price: float | None = None
    reference_target: float | None = None  
    stop_loss: float | None = None
    reasoning_chain: TechReasoningChain
    reasoning: str  
    
    
    
    
    thesis_invalid_if: str = ""
    
    
    
    signal_age_days: int | None = None
    
    
    
    
    
    
    
    atr_14: float | None = None

    @computed_field
    @property
    def risk_reward(self) -> float | None:
        """Reward/risk ratio from entry, stop, and reference_target.

        Computed in Python (not trusted to the LLM). For BUY we expect (target > entry > stop);
        for SELL the inequalities flip. Returns None when any price is missing, the rating
        is neutral, or the geometry is malformed (so PM / RM won't render a fake ratio).
        """
        if self.entry_price is None or self.stop_loss is None or self.reference_target is None:
            return None
        if self.rating in ("buy", "strong_buy"):
            risk = self.entry_price - self.stop_loss
            reward = self.reference_target - self.entry_price
        elif self.rating in ("sell", "strong_sell"):
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.reference_target
        else:
            return None
        if risk <= 0 or reward <= 0:
            return None
        return round(reward / risk, 2)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(
            values, lower_fields=("rating", "conviction"),
        )

    @model_validator(mode="after")
    def _validate_rating_price_consistency(self):
        """Enforce price fields match the rating's actionability.

        - Actionable (strong_buy, buy, sell, strong_sell): entry_price AND stop_loss required.
        - Stop must be on the protective side of entry (stop < entry for BUYs, stop > entry for SELLs).
        - Neutral: prices should be null; we don't hard-fail but clear them to avoid stale hints.
        """
        if self.rating == "neutral":
            
            self.__dict__["entry_price"] = None
            self.__dict__["reference_target"] = None
            self.__dict__["stop_loss"] = None
            return self

        if self.entry_price is None or self.entry_price <= 0:
            raise ValueError(
                f"{self.symbol}: rating={self.rating} requires entry_price > 0"
            )
        if self.stop_loss is None or self.stop_loss <= 0:
            raise ValueError(
                f"{self.symbol}: rating={self.rating} requires stop_loss > 0"
            )
        if self.rating in ("buy", "strong_buy"):
            if self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"{self.symbol}: BUY stop_loss {self.stop_loss} must be below entry {self.entry_price}"
                )
        else:  
            if self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"{self.symbol}: SELL stop_loss {self.stop_loss} must be above entry {self.entry_price}"
                )
        return self


class TradeDecision(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    allocation_pct: float = Field(ge=0, le=100)
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        
        return _normalize_enum_case_fields(values, upper_fields=("action",))

    @model_validator(mode="after")
    def validate_buy_prices(self):
        if self.action == "BUY":
            if self.entry_price <= 0:
                raise ValueError("BUY decisions require entry_price > 0")
            if self.stop_loss < 0:
                raise ValueError("BUY decisions require stop_loss >= 0")
            if self.take_profit <= 0:
                raise ValueError("BUY decisions require take_profit > 0")
            if self.stop_loss > 0 and self.stop_loss >= self.entry_price:
                raise ValueError(
                    "BUY decisions require stop_loss to stay below entry_price"
                )
            if self.take_profit <= self.entry_price:
                raise ValueError(
                    "BUY decisions require take_profit to stay above entry_price"
                )
        return self


class ReasoningChain(BaseModel):
    """7-step CoT for the portfolio manager — forces the audit trail on the
    central decision. Every required field has `min_length=1` so the LLM
    can't dodge a step with `""`. continuity_check AND premortem_check are
    intentionally optional (default `""`) for backward-compat with older logs
    (pre-memory-layer / pre-2026-06 respectively) but are mandatory per the
    prompt; everything else is mandatory at the schema layer too.
    """
    macro_filter: str = Field(min_length=1)
    news_check: str = Field(min_length=1)
    earnings_check: str = Field(min_length=1)
    signal_conflicts: str = Field(min_length=1)
    sizing_logic: str = Field(min_length=1)
    portfolio_balance: str = Field(min_length=1)
    cash_target: str = Field(min_length=1)
    
    
    continuity_check: str = ""
    
    
    
    
    
    premortem_check: str = ""


class TargetPosition(BaseModel):
    """PM's per-symbol intent — WHAT the book should look like, not HOW to get there.

    The PortfolioConstructor translates a list of TargetPositions + current
    holdings + market prices + TA ATR into concrete TradeDecision orders. The
    LLM no longer guesses entry prices, stops, or share counts — it only
    expresses intent.

    Semantics:
    - target_weight_pct = 0 and symbol currently held → close the position.
    - target_weight_pct > 0 on a new symbol → open.
    - target_weight_pct > current weight → add (partial BUY for the delta).
    - target_weight_pct < current weight → trim (partial SELL for the delta).
    - Held symbols NOT appearing in the target list → hold at current weight
      (no instruction = no change). PM may include them explicitly with a
      `keep` note for audit clarity, but it's not required.
    """

    model_config = ConfigDict(validate_assignment=True)

    symbol: str
    
    
    
    
    
    target_weight_pct: float = Field(ge=0.0, le=20.0)
    conviction: Literal["high", "medium", "low"] = "medium"
    thesis: str
    thesis_invalid_if: str = ""
    
    
    
    suggested_stop_price: float | None = None
    catalyst: str = ""  

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(values, lower_fields=("conviction",))


class PortfolioDecision(BaseModel):
    reasoning_chain: ReasoningChain
    
    targets: list[TargetPosition] = Field(default_factory=list)
    
    
    
    
    
    decisions: list[TradeDecision] = Field(default_factory=list)
    portfolio_view: str


class RiskModification(BaseModel):
    symbol: str
    field: str
    original_value: float
    new_value: float
    reason: str


class RiskReasoningChain(BaseModel):
    """6-step CoT for the risk manager — forces audit trail on the last gate.
    Every field has `min_length=1` so the LLM can't skip a step by sending
    `""`. Matches the discipline on the other CoT chains.
    """
    rr_audit: str = Field(min_length=1)             
    signal_fidelity: str = Field(min_length=1)      
    correlation_check: str = Field(min_length=1)    
    event_risk: str = Field(min_length=1)           
    sizing_sanity: str = Field(min_length=1)        
    overall: str = Field(min_length=1)              


class RiskVerdict(BaseModel):
    approved: bool
    reasoning_chain: RiskReasoningChain
    modifications: list[RiskModification] = []
    
    
    
    scale_all_buys: float = Field(default=1.0, ge=0.0, le=1.0)
    
    
    
    
    reason_category: Literal[
        "clean",             
        "oversized",         
        "rr_fail",           
        "concentration",     
        "correlation_risk",  
        "event_risk",        
        "macro_misalign",    
        "data_degraded",     
        "signal_fidelity",   
        "other",             
    ] = "clean"
    reasoning: str

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(values, lower_fields=("reason_category",))


class MacroObservation(BaseModel):
    indicator: str
    reading: str
    interpretation: str




_ALLOWED_SECTORS = (
    "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Industrials", "Communication Services",
    "Utilities", "Basic Materials", "Real Estate", "Broad",
)



_SECTOR_ALIASES = {
    "tech": "Technology",
    "technology": "Technology",
    "financials": "Financial Services",
    "financial": "Financial Services",
    "banks": "Financial Services",
    "consumer discretionary": "Consumer Cyclical",
    "consumer staples": "Consumer Defensive",
    "materials": "Basic Materials",
    "comm services": "Communication Services",
    "communication": "Communication Services",
    "telecom": "Communication Services",
    "reits": "Real Estate",
    "real-estate": "Real Estate",
    "index": "Broad",
    "broad market": "Broad",
    "etf": "Broad",
}


class MacroSectorGuidance(BaseModel):
    sector: Literal[
        "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
        "Consumer Defensive", "Energy", "Industrials", "Communication Services",
        "Utilities", "Basic Materials", "Real Estate", "Broad",
    ]
    stance: Literal["overweight", "neutral", "underweight"]
    reason: str

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        
        
        return _normalize_enum_case_fields(values, lower_fields=("stance",))


class MacroPositionGuidance(BaseModel):
    target_invested_pct: float = Field(ge=0, le=100)
    cash_recommendation_pct: float = Field(ge=0, le=100)
    reasoning: str


class MacroReasoningChain(BaseModel):
    """Six-step CoT, one field per step — forces the LLM to walk each stage.
    Every field has `min_length=1` so the LLM can't skip a step by sending
    `""`. Matches the discipline on the other CoT chains.
    """
    volatility_analysis: str = Field(min_length=1)        
    yield_curve_analysis: str = Field(min_length=1)       
    monetary_policy_analysis: str = Field(min_length=1)   
    inflation_labor_credit: str = Field(min_length=1)     
    cross_signal_synthesis: str = Field(min_length=1)     
    sector_implications: str = Field(min_length=1)        


class MacroAnalysis(BaseModel):
    reasoning_chain: MacroReasoningChain
    regime: Literal["risk-on", "risk-off", "neutral", "transitional"]
    confidence: Literal["high", "medium", "low"]
    equity_outlook: Literal["bullish", "bearish", "neutral"]
    regime_shift: bool = False
    shift_reason: str = ""
    key_observations: list[MacroObservation] = []
    sector_guidance: list[MacroSectorGuidance] = []
    risk_factors: list[str] = []
    position_guidance: MacroPositionGuidance
    bull_triggers: list[str] = []
    bear_triggers: list[str] = []
    alignment_with_news: str = ""
    summary: str

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        
        
        return _normalize_enum_case_fields(
            values,
            lower_fields=("regime", "confidence", "equity_outlook"),
        )

    @model_validator(mode="before")
    @classmethod
    def _sanitize_sector_guidance(cls, values):
        """Map aliases, drop unknown sectors — preserves the rest of the analysis.

        Previously a single bad sector name (e.g. "Financials" instead of
        "Financial Services") rejected the whole MacroAnalysis and left PM blind.
        """
        if not isinstance(values, dict):
            return values
        sg = values.get("sector_guidance")
        if not isinstance(sg, list):
            return values
        cleaned: list[dict] = []
        for item in sg:
            if not isinstance(item, dict):
                continue
            sec = item.get("sector")
            if not isinstance(sec, str):
                continue
            canon = _SECTOR_ALIASES.get(sec.strip().lower(), sec.strip())
            if canon in _ALLOWED_SECTORS:
                new_item = dict(item)
                new_item["sector"] = canon
                cleaned.append(new_item)
            
        values["sector_guidance"] = cleaned
        return values


class NewsEvent(BaseModel):
    headline: str
    impact: str  
    affected_sectors: list[str] = []
    affected_symbols: list[str] = []
    sentiment: str  
    explanation: str


class SectorImpact(BaseModel):
    sector: str
    sentiment: str  
    reason: str


class SymbolAlert(BaseModel):
    symbol: str
    sentiment: str  
    reason: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class NewsAnalysisResult(BaseModel):
    market_sentiment: str  
    confidence: str  
    key_events: list[NewsEvent] = []
    sector_impacts: list[SectorImpact] = []
    symbol_alerts: list[SymbolAlert] = []
    summary: str


class MacroNarrative(BaseModel):
    last_updated: str
    era_themes: list[str] = Field(min_length=1)
    current_regime: str = Field(min_length=5)
    key_state_tracker: dict[str, str] = {}

    @field_validator("last_updated")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        date.fromisoformat(v)
        return v


class StateChange(BaseModel):
    event: str
    previous_state: str
    new_state: str
    market_impact: str
    affected_symbols: list[str] = []
    conviction: Literal["high", "medium", "low"]

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(values, lower_fields=("conviction",))


class StockNewsItem(BaseModel):
    headline: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    conviction: Literal["high", "medium", "low"]
    impact_summary: str

    @field_validator("headline")
    @classmethod
    def require_headline(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("headline cannot be empty")
        return v

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(
            values, lower_fields=("sentiment", "conviction"),
        )


class NewsIntelligenceReport(BaseModel):
    macro_narrative: MacroNarrative
    state_changes: list[StateChange] = []
    stock_news: dict[str, list[StockNewsItem]] = {}
    pm_briefing: str
    market_sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: Literal["high", "medium", "low"]

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(
            values, lower_fields=("market_sentiment", "confidence"),
        )


class Position(BaseModel):
    symbol: str
    qty: float
    avg_entry: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_intraday_pnl: float = 0.0
    sector: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)


class EarningsSegment(BaseModel):
    name: str
    revenue: str
    growth: str = "not disclosed"


class EarningsRevenue(BaseModel):
    total: str
    yoy_growth: str = "not disclosed"
    segments: list[EarningsSegment] = []


class EarningsProfitability(BaseModel):
    gross_margin: str = "not disclosed"
    operating_margin: str = "not disclosed"
    net_income: str = "not disclosed"
    eps: str = "not disclosed"


class EarningsCashFlow(BaseModel):
    operating_cf: str = "not disclosed"
    free_cf: str = "not disclosed"
    capex: str = "not disclosed"


class EarningsBalanceSheet(BaseModel):
    cash_and_equivalents: str = "not disclosed"
    total_debt: str = "not disclosed"
    assessment: str = "not disclosed"


class EarningsStrategicDirection(BaseModel):
    key_initiatives: list[str] = []
    capital_allocation: str = "not disclosed"
    competitive_positioning: str = "not disclosed"


class EarningsRiskFlags(BaseModel):
    strategic_risks: list[str] = []
    operational_risks: list[str] = []


class EarningsReasoningChain(BaseModel):
    """5-step CoT for fundamental analysis — why sentiment is what it is.
    Every field has `min_length=1` so the LLM can't skip a step by sending
    `""`. Matches the discipline on the other CoT chains.
    """
    fundamental_quality: str = Field(min_length=1)       
    growth_trajectory: str = Field(min_length=1)         
    strategic_risks: str = Field(min_length=1)           
    management_execution: str = Field(min_length=1)      
    valuation_context: str = Field(min_length=1)         


class EarningsInvestmentImplications(BaseModel):
    sentiment: Literal["bullish", "bearish", "neutral"]
    conviction: Literal["high", "medium", "low"]
    reasoning_chain: EarningsReasoningChain
    key_thesis: str
    bull_case: str = "not disclosed"
    bear_case: str = "not disclosed"

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(
            values, lower_fields=("sentiment", "conviction"),
        )


class EarningsAnalysis(BaseModel):
    symbol: str
    form_type: Literal["10-Q", "10-K"]
    filing_date: str
    revenue: EarningsRevenue
    profitability: EarningsProfitability
    cash_flow: EarningsCashFlow
    balance_sheet: EarningsBalanceSheet
    management_highlights: list[str] = []
    guidance: str
    strategic_direction: EarningsStrategicDirection = EarningsStrategicDirection()
    risk_flags: EarningsRiskFlags | list[str] = EarningsRiskFlags()
    strategy_consistency: str = "No prior filing available for comparison"
    investment_implications: EarningsInvestmentImplications
    data_quality: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @field_validator("filing_date")
    @classmethod
    def validate_filing_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @field_validator("guidance", "data_quality")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("field cannot be empty")
        return text


class PositionAction(BaseModel):
    action: Literal["SELL", "REDUCE", "TRAIL_STOP", "HOLD"]
    symbol: str
    reason: str
    new_stop_price: float | None = None  

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        
        return _normalize_enum_case_fields(values, upper_fields=("action",))

    @model_validator(mode="after")
    def _trail_stop_requires_new_price(self):
        if self.action == "TRAIL_STOP" and (self.new_stop_price is None or self.new_stop_price <= 0):
            raise ValueError("TRAIL_STOP requires new_stop_price > 0")
        return self


class PositionReasoningChain(BaseModel):
    """Six-step chain the position reviewer must fill before emitting actions.

    Parallel depth to morning PM's 7-step reasoning_chain — prevents
    intraday-price knee-jerk selling and forces memory-aware, thesis-driven
    decisions. Each field is required; empty strings will fail validation
    so the agent can't skip a step by sending "".
    """
    macro_continuity_check: str = Field(min_length=1)
    """Regime + outlook today vs morning vs this week. Stable ⇒ HOLD bias."""

    thesis_progress_check: str = Field(min_length=1)
    """Per-position thesis_progress_pct / pace / distance-to-stop|target.
    Distinguishes 'fast mover' / 'on pace' / 'stalled' / 'broken'."""

    thesis_integrity_check: str = Field(min_length=1)
    """Every SELL/REDUCE must cite a specific named trigger — thesis_invalid_if
    condition, HIGH-conviction state_change reversal, bearish earnings
    analysis, or correlation breach. Intraday price alone is NOT a trigger."""

    winners_discipline_check: str = Field(min_length=1)
    """For positions with profit > 10%: is momentum fading, is it parabolic,
    has target been exceeded? If no, default is HOLD regardless of size —
    good stocks are meant to be held."""

    session_disposition_check: str = Field(min_length=1)
    """Session-aware framing: 'midday' = afternoon patience, TRAIL_STOP over
    SELL; 'close' = act-if-triggered-not-act-because-time, 17.5h no control,
    act only on clear thesis signals never on clock-driven fear."""

    execution_rationale: str = Field(min_length=1)
    """For each SELL/REDUCE action, a 'lock now' vs 'hold outcome' comparison.
    HOLD needs no comparison. TRAIL_STOP names the upside protected vs given up."""


class PositionReview(BaseModel):
    reasoning_chain: PositionReasoningChain
    actions: list[PositionAction] = []
    overall_assessment: str = Field(min_length=1)
    risk_level: Literal["low", "moderate", "elevated", "high"]

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(values, lower_fields=("risk_level",))


class EveningReasoningChain(BaseModel):
    """Seven-step chain evening analyst must fill before emitting the report.

    Depth parallel to PM's 7-step and position_reviewer's 6-step chains.
    Empty strings fail validation — the agent cannot skip a step. Gives
    evening the same thought-depth structure as other LLM agents so its
    decisions are auditable, not just narrative.

    Design note (2026-04 upgrade): the previous 6-step chain was
    structurally anchored on DAILY cycles (yesterday's outlook, today's
    tape, tomorrow's preparation). For a medium-long-term investor, the
    most important question — "how is each held thesis playing out over
    the past 6-8 weeks?" — wasn't being asked anywhere. `thesis_health_
    review` is that missing step, and it sits between the retrospective
    (what happened) and the decision-quality review (how did we react).
    """
    performance_attribution: str = Field(min_length=1)
    """What drove today's P&L? Which positions contributed + / −, which macro /
    news factors explain the moves. Concrete, not vague."""

    outlook_retrospection: str = Field(min_length=1)
    """Honest grade of yesterday's tomorrow_outlook vs today's actual. If
    yesterday said bullish and today ripped down, say so. Calibration > saving
    face. Cross-reference specific predictions to specific outcomes."""

    thesis_health_review: str = Field(min_length=1)
    """For each held position: given 6-8 weeks of fundamentals evolution
    (earnings trajectory, macro sector stance, news flow, tech rating
    history), is the ORIGINAL entry thesis strengthening, still intact,
    weakening, or broken? This is the step that makes the agent a value
    investor not a swing trader. For holdings where the thesis is
    broken — flag them for SELL consideration tomorrow even if price
    hasn't yet moved. For holdings where the thesis is strengthening
    but price hasn't caught up — flag them as add-more candidates.
    Price noise is not thesis noise; conflating them is the main way
    medium-long-term strategies go wrong."""

    decision_quality_review: str = Field(min_length=1)
    """BUY / SELL / HOLD decisions today + the last few days. Pattern check:
    are you selling winners too early? Buying near tops? Hedging at the wrong
    time? Name the pattern if one exists."""

    calibration_meta: str = Field(min_length=1)
    """Zoom out on your recent bias / conviction track record (surfaced in the
    prompt). Are you systematically too bullish? Does HIGH conviction actually
    outperform LOW? This is the meta-loop — learning from your own accuracy
    not just yesterday's single call."""

    market_regime_read: str = Field(min_length=1)
    """Where is the market now, where's it going, what's the key evidence from
    today's tape + news. This is the foundation the tomorrow_bias rests on."""

    tomorrow_preparation: str = Field(min_length=1)
    """Key events tomorrow (earnings, econ data, Fed), levels to watch, how
    today's action shapes tomorrow's posture. What PM needs to know at 09:30."""







ThesisTrajectory = Literal[
    "strengthening",   
    "intact",          
    "weakening",       
    "broken",          
                       
]


class SellGrade(BaseModel):
    """Structured grade of a single recent SELL — what evening judged right or
    wrong. PM / position reviewer can read aggregate counts to feed back into
    their SELL discretion.

    Grading is dual-axis: `grade` aggregates `price_outcome` (what the tape
    did since we sold) and `thesis_trajectory_at_sell` (whether we sold
    with thesis-justification or on nerves / noise). A defensible SELL
    is one where we exited a weakening/broken thesis, even if price
    subsequently bounced — we kept discipline. A `wrong` SELL is one
    where we exited an intact/strengthening thesis AND price ran.
    """
    symbol: str
    sell_date: str   
    sell_price: float
    current_price: float
    pct_move_since_sell: float
    grade: Literal["correct", "premature", "wrong"]
    reason: str = Field(min_length=1)
    
    
    
    thesis_trajectory_at_sell: ThesisTrajectory | None = None

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        return _normalize_symbol(v)

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(
            values, lower_fields=("grade", "thesis_trajectory_at_sell"),
        )









BuyLossRootCause = Literal[
    "greed_top_chasing",      
    "macro_warning_ignored",  
    "herd_buying",            
    "averaged_down",          
    "thesis_broken_held",     
    "concentration_blow",     
    "timing_mistake",         
    "systemic_drawdown",      
    "tail_event",             
]


class BuyGrade(BaseModel):
    """Structured grade of a recent BUY — did the entry play out?
    Mirrors SellGrade so the feedback loop is symmetric.

    Like SellGrade, grading is dual-axis. `grade` aggregates price
    action AND `thesis_trajectory` (how the underlying fundamentals /
    theme have evolved since entry). A buy can be down 8% with thesis
    strengthening — that's NOT wrong, that's value entry being tested
    by noise. A buy can be up 10% with thesis broken — that's NOT
    correct, that's momentum masking a real failure."""
    symbol: str
    buy_date: str
    buy_price: float
    current_price: float
    pct_move_since_buy: float
    grade: Literal["correct", "premature", "wrong"]
    reason: str = Field(min_length=1)
    
    
    thesis_trajectory: ThesisTrajectory | None = None
    
    
    
    
    loss_root_cause: BuyLossRootCause | None = None
    
    
    
    
    
    market_relative_move_pct: float | None = None
    
    
    
    missed_warning_ref: str | None = None

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        return _normalize_symbol(v)

    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(
            values,
            lower_fields=("grade", "thesis_trajectory", "loss_root_cause"),
        )

    @model_validator(mode="after")
    def _loss_fields_required(self) -> "BuyGrade":
        if self.grade == "wrong" and self.loss_root_cause is None:
            raise ValueError(
                "BuyGrade with grade='wrong' requires loss_root_cause so the "
                "quarterly meta-reflector can aggregate patterns"
            )
        
        
        
        
        
        
        
        if self.grade == "wrong" and self.thesis_trajectory is None:
            raise ValueError(
                "BuyGrade with grade='wrong' requires thesis_trajectory so "
                "position_reviewer can distinguish a value re-entry candidate "
                "(intact thesis) from a stay-out signal (broken thesis)"
            )
        if (self.loss_root_cause == "macro_warning_ignored"
                and not (self.missed_warning_ref or "").strip()):
            raise ValueError(
                "loss_root_cause='macro_warning_ignored' requires missed_warning_ref "
                "citing the specific signal that was ignored (agent + date + headline)"
            )
        return self


class MissedOpportunitySnapshot(BaseModel):
    """Python-computed facts for one notable mover — INPUT to the evening LLM,
    not its output. The LLM reads a list of these and writes one
    MissedOpportunity per interesting row.

    Carries enough signal-state context (prior TA rating, recent news
    headline, earnings signal, macro sector stance) that the LLM's miss
    classification has to be grounded in observable prior evidence rather
    than price retro-rationalization.

    For symbols sourced from Alpaca's top-mover screener (not in our
    trading universe), the quality fields (avg_dollar_volume_20d_m,
    volume_confirmation_ratio, single_day_concentration_pct) are the
    main filter for "worth considering adding to universe" vs "low-
    volume squeeze we should ignore". A medium-long-term investor
    doesn't chase thin moves.
    """
    symbol: str
    move_pct: float
    window_days: int
    held_during_window: bool
    had_ta_signal: bool
    had_news_signal: bool
    had_earnings_signal: bool
    source: Literal["universe", "top_mover", "both"]
    
    last_ta_rating: str | None = None          
    last_ta_date: str | None = None            
    last_news_headline: str | None = None      
    
    
    theme_tags: list[str] = []
    
    
    
    recent_earnings_signal: str | None = None
    
    
    macro_sector_tailwind: Literal["bullish", "neutral", "bearish", "unknown"] = "unknown"

    
    
    
    avg_dollar_volume_20d_m: float | None = None
    """20-day average daily dollar volume in MILLIONS of USD. Low numbers
    (< ~5M) indicate thin liquidity — easy to squeeze, dangerous for a
    medium-long-term position. Used to pre-filter very illiquid movers
    upstream; the LLM also sees it to reason about "real institutional
    interest vs low-volume drift"."""
    volume_confirmation_ratio: float | None = None
    """Today's dollar volume / 20-day avg. > ~1.5 indicates buyers
    showed up in size (real interest). < 1.0 = move happened on
    normal-or-less flow; unlikely to sustain."""
    single_day_concentration_pct: float | None = None
    """Percent of the window's total return that came from the BIGGEST
    single day. 0-100. > 70 = gap-up day (event / squeeze); < 50 =
    distributed move (trend). For a medium-long-term investor, a
    distributed trend is far more interesting than a single gap."""

    
    
    
    
    trailing_pe: float | None = None
    forward_pe: float | None = None
    ps_ratio: float | None = None
    valuation_signal: Literal["cheap", "fair", "stretched", "no_data"] = "no_data"
    """Rough forward-PE-based classifier filled by Python upstream.
    < 12 → cheap, 12-25 → fair, >= 25 → stretched, None → no_data.
    Thresholds are deliberately crude — the LLM reads raw PE numbers
    too and makes sector-adjusted judgments. `valuation_signal` is
    just a fast first cut that prevents obvious hype chasing."""

    
    
    
    value_entry_candidate: bool = False

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        return _normalize_symbol(v)


class MissedOpportunity(BaseModel):
    """Evening-analyst OUTPUT for one snapshot: classified miss + lesson +
    (for non-universe symbols) watchlist-addition recommendation.

    `miss_category` frames the miss through the three lenses the user cares
    about: catching trends, not missing themes, spotting fundamental
    mispricing. `noise_rally` and `risk_disciplined` are escape hatches so
    the LLM isn't forced to label every price move as a miss — but the
    prompt has to push back when they're overused.

    For symbols sourced from the top-mover screener (not in the trading
    universe), `universe_addition_recommendation` is the high-bar answer
    to "should we add this to the 77-symbol universe we carefully curated?"
    Default is "no" — the universe is deliberately small; thin or
    one-day-gap moves should not expand it. "add" only when volume,
    sustain, theme, and fundamentals all point in the right direction.
    """
    symbol: str
    move_pct: float
    miss_category: Literal[
        "trend_timing_miss",        
        "theme_blindspot",          
        "fundamentals_mispricing",  
        "value_entry_missed",       
                                    
        "noise_rally",              
        "risk_disciplined",         
    ]
    
    
    
    
    theme_if_any: str | None = None
    
    
    
    
    theme_durability: Literal[
        "multi_year_secular",   
                                
        "1_3_year_cycle",       
                                
        "months_fad",           
                                
        "unknown",              
    ] = "unknown"
    lesson: str = Field(min_length=1, max_length=400)
    
    
    
    
    
    universe_addition_recommendation: Literal["add", "watch", "no"] = "no"
    universe_addition_reason: str = Field(default="", max_length=400)
    """1-2 sentences citing the QUALITY metrics (volume, sustain, theme,
    fundamentals, valuation) that justify a non-'no' recommendation.
    Required when recommendation is "add" or "watch"; must stay empty
    when "no" so the reason field doesn't drift into wishful thinking."""

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        return _normalize_symbol(v)

    @model_validator(mode="after")
    def _theme_required_for_real_misses(self) -> "MissedOpportunity":
        real_miss_categories = {
            "trend_timing_miss", "theme_blindspot",
            "fundamentals_mispricing", "value_entry_missed",
        }
        if self.miss_category in real_miss_categories:
            if not (self.theme_if_any or "").strip():
                raise ValueError(
                    f"MissedOpportunity miss_category='{self.miss_category}' "
                    f"requires theme_if_any so quarterly aggregation can group by theme"
                )
        return self

    @model_validator(mode="after")
    def _theme_durability_required_when_themed(self) -> "MissedOpportunity":
        
        
        
        
        if (self.theme_if_any or "").strip():
            if self.theme_durability is None:
                raise ValueError(
                    "theme_if_any is set but theme_durability is None; pick "
                    "multi_year_secular / 1_3_year_cycle / months_fad / unknown"
                )
        return self

    @model_validator(mode="after")
    def _addition_recommendation_consistency(self) -> "MissedOpportunity":
        
        
        if self.universe_addition_recommendation in ("add", "watch"):
            if not (self.universe_addition_reason or "").strip():
                raise ValueError(
                    f"universe_addition_recommendation="
                    f"'{self.universe_addition_recommendation}' requires "
                    f"universe_addition_reason citing volume, sustain, or "
                    f"theme quality — bar is high, evidence must be concrete"
                )
        return self


class EveningReport(BaseModel):
    
    
    
    
    @model_validator(mode="before")
    @classmethod
    def _normalize_enum_case(cls, values):
        return _normalize_enum_case_fields(
            values,
            lower_fields=(
                "risk_rating", "tomorrow_bias", "tomorrow_conviction",
            ),
        )

    reasoning_chain: EveningReasoningChain
    daily_summary: str = Field(min_length=1)
    lessons: str = Field(min_length=1)
    tomorrow_outlook: str = Field(min_length=1)  
    risk_rating: Literal["low", "moderate", "elevated", "high"]
    suggested_actions: list[str] = []
    
    previous_outlook_assessment: str = ""
    
    
    
    tomorrow_bias: Literal["bullish", "neutral", "bearish"] = "neutral"
    tomorrow_conviction: Literal["high", "medium", "low"] = "medium"
    tomorrow_key_risks: list[str] = []
    
    
    sell_decisions_assessment: str = ""
    
    
    
    
    
    sell_grades: list[SellGrade] = []
    buy_grades: list[BuyGrade] = []
    
    
    
    
    missed_opportunities: list[MissedOpportunity] = []

    
    
    
    
    
    
    this_week_thesis_catalysts: list[str] = []

    
    
    
    
    
    
    
    
    
    
    
    thesis_updates: list[str] = []
    selection_rules: list[str] = []
    discipline_notes: list[str] = []


class AgentLog(BaseModel):
    agent_name: str
    run_id: str
    timestamp: datetime
    input_summary: str
    output_summary: str
    full_response: str
    model: str
    tokens_used: int











MetaReflectionAgentName = Literal[
    "tech_analyst",
    "news_analyst",
    "macro_analyst",
    "earnings_analyst",
    "portfolio_manager",
    "evening_analyst",
]


class MetaReasoningChain(BaseModel):
    """7-step chain the meta-reflector must fill before emitting the report.

    Parallel depth to morning PM's 7-step chain and position reviewer's
    6-step chain — empty strings fail validation so the LLM can't skip a
    step.

    **Ordering matters**: the LLM runs these in-order to avoid the
    trap of proposing prompt edits without first understanding (a) its
    own self-portrait across multiple axes, (b) where the self-portrait
    falls short of the ideal, (c) what the target agent's prompt ALREADY
    contains. Facts-first, synthesis-next, existing-design-audit,
    proposal-last.

    Design notes for anyone editing this chain:
      - Steps 1-3 are FACTS. They each cite numbers from a specific
        digest section. No interpretation allowed.
      - Step 4 is SYNTHESIS. It's the first step that interprets the
        facts, producing a multi-axis self-portrait. Replaces the old
        single-axis `style_bias_identification` + absorbs the old
        `agent_hit_rate_audit` (which was just another axis of self-
        portrait anyway).
      - Step 5 is DIAGNOSIS. It names 2-3 top leverage gaps between
        the self-portrait and the idealized trader profile the user
        wants the system to converge toward.
      - Step 6 is PROMPT AUDIT. For each gap named in step 5, the LLM
        consults `agent_prompts_snapshot` to understand what's already
        in the target agent's prompt — preventing duplicate / redundant
        / conflicting edits.
      - Step 7 is PROPOSAL. Grounded in both the gaps (step 5) AND the
        existing prompt state (step 6).

    The old `missed_theme_diagnosis` step was folded into
    `portrait_gap_diagnosis` — theme coverage IS one of the gap axes.
    """
    performance_vs_benchmark: str = Field(min_length=1)
    """Step 1/FACT. Where did this quarter's return land vs SPY? Alpha
    positive or negative? Drawdown profile? Be specific about numbers
    from period_performance — no "we did ok this quarter" hand-waving."""

    secular_theme_audit: str = Field(min_length=1)
    """Step 2/FACT. Enumerate this quarter's real themes (AI capex,
    nuclear/power, rare earth, reshoring, etc.). For each: did we
    participate? At what entry position relative to the breakout? For
    how long? Name themes_caught_early, themes_caught_late,
    themes_missed_entirely — mirror the structured output fields."""

    loss_autopsy_audit: str = Field(min_length=1)
    """Step 3/FACT. Enumerate the top 3-5 loss causes from
    loss_patterns.by_cause. For each: count, alpha_destruction_pct,
    which agent owns it. This feeds `loss_pattern_report`."""

    self_portrait_synthesis: str = Field(min_length=1)
    """Step 4/SYNTHESIS. **Multi-axis self-portrait**, not a single-
    line label. Synthesize facts from steps 1-3 + agent_signal_activity
    into concrete dimensions: (a) conviction_calibration — does HIGH
    conviction actually outperform LOW? (b) theme_breadth — do we
    cover only tech/AI or also energy/materials/reshoring? (c)
    loss_discipline — do we catch thesis breaks or ride losers? (d)
    execution_style — average hold days, realized vs intended
    timeframe. (e) agent_balance — any agent gone silent / any
    flooding with low-quality signals. Each dimension should be one
    sentence citing a specific digest number. This REPLACES the
    prior `style_bias_identification` + `agent_hit_rate_audit`."""

    portrait_gap_diagnosis: str = Field(min_length=1)
    """Step 5/DIAGNOSIS. For each dimension in the self-portrait, name
    the IDEAL state (what a medium-long-term value investor with broad
    theme coverage would look like) and the ACTUAL state. Pick the
    top 2-3 highest-leverage gaps — don't try to fix everything.
    Explicitly call out where failures happened: if a theme was
    missed, which agent layer (news vs macro vs tech vs PM) was
    responsible? Attribution is specific, not collective."""

    existing_prompt_audit: str = Field(min_length=1)
    """Step 6/PROMPT AUDIT. For each of the top gaps named in step 5,
    read `agent_prompts_snapshot[{target_agent}]` and enumerate: (a)
    what rules ALREADY exist that address this gap (cite the section /
    heading), (b) whether those existing rules are being followed
    (check corrigibility_trend — are the losses / misses recurring
    despite the rule?), (c) whether there's room for a new rule that
    doesn't conflict with or duplicate existing content. If the
    snapshot shows the target section is saturated with prior
    Learnings, propose a retract-or-replace rather than another
    append. **Do NOT propose a learning without citing what's already
    in the target prompt.**"""

    prompt_edit_reasoning: str = Field(min_length=1)
    """Step 7/PROPOSAL. Given the gaps (step 5) and existing-prompt
    state (step 6), why these specific `proposed_learnings` and not
    others? Corrigibility is the key check: if a cause has been
    worsening for 2 quarters AND the existing prompt has no rule for
    it → append. If a cause has been worsening AND an existing rule
    isn't being followed → DON'T append another (the issue is rule
    adherence, not rule absence); log this as a
    `persistent_blindspot` for the operator to review manually. If
    improving → don't pile on."""


class ThemeCoverage(BaseModel):
    """Quarter-level theme participation — the core "trend capture" metric.

    All four lists may be empty. The meta-reflector populates them from its
    reading of missed_themes + holdings activity during the quarter. Not
    every theme has to appear in every bucket — a theme can be both
    "caught late" and "fully exited", those nuances are in the audit text.
    """
    themes_caught_early: list[str] = []
    """Themes we bought before the move was obvious (entry < 30% of the
    quarter's total move for that theme). The system's genuine alpha."""
    themes_caught_late: list[str] = []
    """Themes we bought after the trend was already priced (entry > 50%
    of total move). Trend-follower rather than trend-identifier
    behavior — ok occasionally, systematically problematic."""
    themes_missed_entirely: list[str] = []
    """Themes that ran ≥20% in the quarter and we never held any symbol
    within. Pure coverage / blindspot failures — the highest-value
    signal for where the system needs to look."""
    emerging_themes_to_watch: list[str] = []
    """Themes forming late in the quarter that didn't run enough to
    show in the caught/missed categories yet. Prior knowledge PM
    should carry into next quarter."""
    mispricing_patterns: list[str] = []
    """Concrete examples where earnings_analyst said bullish+high but
    PM didn't buy, or where macro_analyst tagged a sector tailwind
    and we had no coverage. 1-5 entries, each specific."""




MetaLossRootCause = BuyLossRootCause


class LossPattern(BaseModel):
    """One row of loss_pattern_report.top_patterns — cause + attribution +
    proposed guard. Agent attribution drives which prompt gets the
    `proposed_guard` as a candidate learning."""
    root_cause: MetaLossRootCause
    occurrences: int = Field(ge=1)
    total_loss_pct: float
    """Signed sum of pct_move_since_buy for wrongs in this bucket — sign
    preserved so a mix of small/large isn't hidden in absolute values."""
    example_trades: list[str] = Field(min_length=1, max_length=8)
    """Concrete trades "SYMBOL YYYY-MM-DD -X%" so the prompt edit
    justification has anchors, not abstractions."""
    attributable_agent: Literal[
        "tech_analyst", "news_analyst", "macro_analyst",
        "earnings_analyst", "portfolio_manager", "evening_analyst",
        "execution", "no_agent",
    ]
    """`no_agent` when the failure is pure discipline (PM / evening's
    discipline — nothing any individual agent's prompt could have
    caught). `execution` when the issue was broker-side, not LLM."""
    proposed_guard: str = Field(min_length=1, max_length=400)
    """One-sentence candidate prompt addition that would have caught
    this pattern. Empty strings / vague hedges fail validation.
    400 cap is intentionally matched to MissedOpportunity.lesson — the
    LLM cites concrete facts (symbols, dates, pct moves) so terse caps
    force vague language, which is worse than the extra context."""


class LossPatternReport(BaseModel):
    """Quarterly loss autopsy. Parallel structure to ThemeCoverage so the
    meta-reflector's ups/downs analysis stays symmetric."""
    top_patterns: list[LossPattern] = Field(default_factory=list, max_length=5)
    systemic_vs_alpha_split: str = Field(default="")
    """Prose one-liner decomposing losses: "72% alpha-destruction (we
    under-performed the tape), 28% systemic (market also fell)"."""
    worst_single_trade: str | None = None
    """Most painful single wrong BUY this quarter + its root cause +
    whether the pattern is likely to recur. None when no wrongs."""
    corrigibility_score: Literal["improving", "stable", "degrading"] = "stable"
    """Compared to last quarter's report — are the same causes getting
    better, holding, or worse? Drives whether to add more learnings
    (degrading) or give existing ones time to work (improving)."""


class PromptLearning(BaseModel):
    """A proposed edit to one agent's prompt. Append-only for safety —
    never delete existing rules, never rewrite core sections. PR 4's
    prompt_editor enforces additional guards (length, dedup, prohibited
    words, single-quarter rate limits) on top of this schema.

    `retract` is the sole exception to append-only: used in later
    quarters to remove a learning THIS system previously added if the
    subsequent data showed it didn't help.
    """
    agent_name: MetaReflectionAgentName
    operation: Literal["append", "retract"]
    learning_text: str = Field(min_length=20, max_length=200)
    """1-2 concrete sentences. The PR 4 editor rejects entries containing
    "always"/"never"/"override"/"must always"/"must never" as these
    directly conflict with the hard-invariant wording in core prompts."""
    justification: str = Field(min_length=40)
    """Must cite specific digest facts: agent hit-rate numbers, theme
    occurrence counts, loss-cause frequencies, corrigibility deltas.
    A post-hoc model_validator enforces at least one number or '%'
    appears — no vibes-only learnings."""
    retract_target_hash: str | None = None
    """Only set when operation='retract'. Content-hash of the prior
    PromptLearning.learning_text being withdrawn. PR 4 verifies the
    hash matches an actual prior auto-append before deleting."""

    @model_validator(mode="after")
    def _justification_cites_facts(self) -> "PromptLearning":
        
        
        
        
        has_digit = any(ch.isdigit() for ch in self.justification)
        if not has_digit:
            raise ValueError(
                "PromptLearning.justification must cite at least one digest "
                "fact with a number (count, %, or quarter period). Got: "
                f"{self.justification[:80]!r}"
            )
        if self.operation == "retract" and not self.retract_target_hash:
            raise ValueError(
                "operation='retract' requires retract_target_hash pointing "
                "to the prior auto-appended learning being withdrawn"
            )
        return self


class QuarterlyMetaReflection(BaseModel):
    """Top-level meta-reflector output. Persisted to
    data/evolution/{period}/reflection.json alongside the digest."""
    period: str
    """e.g. '2026-Q1' — matches the digest's period label."""
    meta_reasoning_chain: MetaReasoningChain
    style_self_portrait: str = Field(default="", max_length=2000)
    """Multi-sentence honest self-description for ongoing audit. Optional:
    `meta_reasoning_chain.self_portrait_synthesis` carries the same
    content as part of the CoT, so some LLM outputs legitimately leave
    this top-level field empty rather than duplicating. When non-empty
    it's useful for downstream continuity rendering."""
    persistent_blindspots: list[str] = Field(default_factory=list, max_length=5)
    root_cause_hypotheses: list[str] = Field(default_factory=list, max_length=5)
    theme_coverage_report: ThemeCoverage
    loss_pattern_report: LossPatternReport
    proposed_learnings: list[PromptLearning] = Field(
        default_factory=list, max_length=3,
    )
    """System enforces max 3 agents edited per quarter AFTER schema
    validation — see PR 4's prompt_editor for the enforcement layer.
    This schema max is the upper bound the LLM sees."""
    confidence: Literal["high", "medium", "low"] = "medium"
    """Meta-confidence — with only 1-2 quarters of data the LLM should
    self-report 'low' and propose at most 1 learning. PR 4's editor
    uses this to scale down edit rates."""
