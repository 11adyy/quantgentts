"""Portfolio Constructor — turns PM target-state into concrete orders.

Phase 2 of the architecture work. Previously the LLM (Portfolio Manager)
emitted TradeDecision objects directly, including entry_price / stop_loss /
take_profit. That put the LLM dangerously close to the execution layer:
- fat-finger-protection patches
- vol-adjusted sizing patches
- stop-limit buffer patches
- sub-penny quantize patches
...were all band-aids for "LLM output an execution detail it shouldn't own."

Now PM emits TargetPosition (target_weight_pct, conviction, thesis,
invalid_if) and this module derives the actual orders from:
- Target state
- Current positions (broker truth)
- TA's ATR + suggested stop (for stop distance)
- Broker's live price (for entry price)
- Total equity + cash (for sizing)

The constructor is deterministic and unit-testable. LLM creativity is
confined to intent; math is code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models import Position, TargetPosition, TechAnalysisResult, TradeDecision

logger = logging.getLogger(__name__)


@dataclass
class ConstructorConfig:
    """Tunables for how the constructor sizes and prices orders."""
    
    risk_budget_pct: float = 0.5
    
    min_trade_weight_delta: float = 0.5
    
    default_stop_atr_multiple: float = 2.0
    
    fallback_stop_pct: float = 0.05
    
    
    
    
    
    
    
    max_new_position_pct: float = 7.5
    
    
    new_position_floor_pct: float = 2.5
    
    
    max_add_step_pct: float = 2.5
    max_add_ceiling_pct: float = 10.0


class PortfolioConstructor:
    """Stateless translator: target state → concrete orders."""

    def __init__(self, config: ConstructorConfig | None = None):
        self.cfg = config or ConstructorConfig()

    def construct_orders(
        self,
        targets: list[TargetPosition],
        positions: list[Position],
        analyses: list[TechAnalysisResult],
        total_value: float,
        price_map: dict[str, float] | None = None,
    ) -> list[TradeDecision]:
        """Produce the order list that moves the book from current → target state.

        Orders are returned in a canonical order: SELLs (partials and exits)
        first, then BUYs. Execution layer is free to re-order, but this
        matches the existing pipeline assumption (sells free up cash first).

        `price_map`: optional {symbol: live_price} — required for BUYs so
        the constructor can sanity-check TA's entry. If absent for a BUY
        symbol, we fall back to TA's entry_price.
        """
        if total_value <= 0:
            return []
        price_map = price_map or {}
        current_weights = self._current_weights(positions, total_value)
        analyses_by_sym = {a.symbol: a for a in analyses}
        positions_by_sym = {p.symbol: p for p in positions}

        sells: list[TradeDecision] = []
        buys: list[TradeDecision] = []

        for target in targets:
            sym = target.symbol
            current_pct = current_weights.get(sym, 0.0)
            target_pct = target.target_weight_pct
            delta_pct = target_pct - current_pct

            
            
            
            
            
            
            closing = (target_pct == 0 and current_pct > 0)
            if not closing and abs(delta_pct) < self.cfg.min_trade_weight_delta:
                
                
                if current_pct > 0:
                    buys.append(self._hold_decision(target))
                continue

            if delta_pct < 0:
                
                sell_decision = self._build_sell(
                    target, positions_by_sym.get(sym), current_pct, target_pct,
                )
                if sell_decision is not None:
                    sells.append(sell_decision)
            else:
                
                buy_decision = self._build_buy(
                    target,
                    analysis=analyses_by_sym.get(sym),
                    current_pct=current_pct,
                    target_pct=target_pct,
                    total_value=total_value,
                    market_price=price_map.get(sym),
                )
                if buy_decision is not None:
                    buys.append(buy_decision)
                elif current_pct > 0:
                    
                    
                    buys.append(self._hold_decision(target))

        
        
        
        
        sells.sort(key=lambda d: 0 if d.allocation_pct >= 100 else 1)
        buys.sort(key=lambda d: d.allocation_pct, reverse=True)
        return sells + buys

    @staticmethod
    def _current_weights(
        positions: list[Position], total_value: float,
    ) -> dict[str, float]:
        """Current-position weights as gross-leverage percentages.

        Uses the same `_gross_multiplier` convention as
        `RiskRuleEngine.check` (risk/rules.py:28). For inverse / leveraged
        ETFs (SH=−1x, SDS=−2x, PSQ=−1x, SQQQ=−3x) the gross multiplier
        is the unsigned magnitude — a $10K SQQQ position consumes 30%
        gross notional, not 10% raw, exactly as the risk engine
        evaluates it.

        Pre-fix this used raw `market_value / total_value`, so a PM
        target_weight_pct=20 on SQQQ (intended as the 20% single-name
        cap) computed as 20% raw in the constructor but 60% gross at
        the engine — the engine then hard-blocked every leveraged-ETF
        target at the ceiling, while the constructor's delta math saw
        no trim needed. Now constructor + engine agree on the
        semantics: target_weight_pct IS gross-leverage percentage.
        """
        if total_value <= 0:
            return {}
        
        
        from src.risk.rules import _gross_multiplier
        return {
            p.symbol: (p.market_value * _gross_multiplier(p.symbol) / total_value * 100)
            for p in positions
            if p.qty > 0
        }

    @staticmethod
    def _hold_decision(target: TargetPosition) -> TradeDecision:
        """Record PM's explicit 'keep' intent as a HOLD for audit trail."""
        return TradeDecision(
            action="HOLD",
            symbol=target.symbol,
            allocation_pct=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning=f"Hold at current weight. Thesis: {target.thesis[:200]}",
        )

    @staticmethod
    def _build_sell(
        target: TargetPosition,
        position: Position | None,
        current_pct: float,
        target_pct: float,
    ) -> TradeDecision | None:
        if position is None or position.qty <= 0:
            return None
        
        
        
        
        
        
        
        
        import math as _math
        if not _math.isfinite(current_pct) or current_pct <= 0:
            logger.warning(
                "Constructor: SELL %s skipped — current_pct=%s "
                "(market_value=%s likely NaN/zero from broker glitch)",
                target.symbol, current_pct, position.market_value,
            )
            return None
        if target_pct == 0:
            
            alloc = 100.0
        else:
            
            
            fraction = (current_pct - target_pct) / current_pct
            alloc = max(1.0, min(99.0, round(fraction * 100, 1)))
        reasoning = target.thesis
        if target.thesis_invalid_if:
            reasoning += f" (thesis_invalid_if: {target.thesis_invalid_if})"
        
        return TradeDecision(
            action="SELL",
            symbol=target.symbol,
            allocation_pct=alloc,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning=reasoning[:500],
        )

    def _build_buy(
        self,
        target: TargetPosition,
        analysis: TechAnalysisResult | None,
        current_pct: float,
        target_pct: float,
        total_value: float,
        market_price: float | None,
    ) -> TradeDecision | None:
        
        
        entry_price = 0.0
        if market_price and market_price > 0:
            entry_price = float(market_price)
        elif analysis and analysis.entry_price:
            entry_price = float(analysis.entry_price)
            logger.info(
                "Constructor: no live market_price for %s, using TA entry $%.2f",
                target.symbol, entry_price,
            )
        if entry_price <= 0:
            logger.warning(
                "Constructor: cannot construct BUY %s — no entry price available",
                target.symbol,
            )
            return None

        
        
        
        
        
        
        
        
        stop_loss = self._resolve_stop(target, analysis, entry_price)
        if stop_loss is not None:
            stop_loss = round(stop_loss, 2)
        if stop_loss is None or stop_loss <= 0 or stop_loss >= entry_price:
            logger.warning(
                "Constructor: BUY %s rejected — no valid stop below entry "
                "(entry=$%.2f, stop=%s)",
                target.symbol, entry_price, stop_loss,
            )
            return None

        
        
        if analysis and analysis.reference_target and analysis.reference_target > entry_price:
            take_profit = float(analysis.reference_target)
        else:
            stop_gap_pct = (entry_price - stop_loss) / entry_price
            take_profit = round(entry_price * (1 + 2 * stop_gap_pct), 2)

        
        
        
        
        
        
        
        
        
        
        
        
        from src.risk.rules import _gross_multiplier
        if current_pct < self.cfg.new_position_floor_pct:
            if target_pct > self.cfg.max_new_position_pct:
                logger.info(
                    "Constructor: %s NEW position target %.1f%% capped at %.1f%% "
                    "(flat entry sizing; conviction=%s, current %.2f%%)",
                    target.symbol, target_pct, self.cfg.max_new_position_pct,
                    getattr(target, "conviction", "?"), current_pct,
                )
                target_pct = self.cfg.max_new_position_pct
        else:
            add_cap = min(current_pct + self.cfg.max_add_step_pct, self.cfg.max_add_ceiling_pct)
            if target_pct > add_cap:
                logger.info(
                    "Constructor: %s ADD target %.1f%% capped at %.1f%% "
                    "(current %.1f%% + %.1fpp step, %.0f%% ceiling)",
                    target.symbol, target_pct, add_cap, current_pct,
                    self.cfg.max_add_step_pct, self.cfg.max_add_ceiling_pct,
                )
                target_pct = add_cap
        allocation_pct = (target_pct - current_pct) / _gross_multiplier(target.symbol)
        if allocation_pct <= 0:
            return None
        if (target_pct - current_pct) < self.cfg.min_trade_weight_delta:
            
            
            logger.info("Constructor: %s add of %.2fpp after caps is below the %.2fpp churn "
                        "threshold — no order", target.symbol, target_pct - current_pct,
                        self.cfg.min_trade_weight_delta)
            return None
        
        
        
        
        risk_per_share = entry_price - stop_loss
        risk_dollars_allowed = total_value * self.cfg.risk_budget_pct / 100
        
        
        
        
        if risk_per_share > 0:
            alloc_cap_by_risk = (
                risk_dollars_allowed * entry_price / risk_per_share / total_value * 100
            )
            if allocation_pct > alloc_cap_by_risk:
                logger.info(
                    "Constructor: %s alloc capped by risk budget "
                    "(delta %.2f%% → %.2f%% at %.1f%% risk budget)",
                    target.symbol, allocation_pct, alloc_cap_by_risk,
                    self.cfg.risk_budget_pct,
                )
                allocation_pct = alloc_cap_by_risk

        allocation_pct = max(0.0, round(allocation_pct, 2))
        if allocation_pct <= 0:
            return None

        reasoning = target.thesis
        if target.thesis_invalid_if:
            reasoning += f" (invalid if: {target.thesis_invalid_if})"
        if target.catalyst:
            reasoning += f" (catalyst: {target.catalyst})"

        return TradeDecision(
            action="BUY",
            symbol=target.symbol,
            allocation_pct=allocation_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,   
            take_profit=take_profit,
            reasoning=reasoning[:500],
        )

    def _resolve_stop(
        self,
        target: TargetPosition,
        analysis: TechAnalysisResult | None,
        entry_price: float,
    ) -> float | None:
        """Priority: target's suggested stop → TA's stop → ATR-based → fallback %.

        The ATR-based middle tier is meaningful for volatile small-caps:
        a hardcoded 5 % stop on a name with ATR(14) = 8 % of price gets
        thrashed by normal noise. `entry - 2 * ATR` is the standard
        volatility-aware default; matches the prompt's recommendation
        to TechAnalyst.
        """
        if target.suggested_stop_price and target.suggested_stop_price > 0:
            return float(target.suggested_stop_price)
        if analysis and analysis.stop_loss and analysis.stop_loss > 0:
            return float(analysis.stop_loss)
        
        if analysis and analysis.atr_14 and analysis.atr_14 > 0:
            atr_stop = entry_price - self.cfg.default_stop_atr_multiple * analysis.atr_14
            if atr_stop > 0:
                return round(atr_stop, 2)
            
            
            
            
            
            
            
            
            
            
            import logging
            logging.getLogger(__name__).warning(
                "ATR-based stop for entry=$%.2f with ATR=$%.4f would be "
                "non-positive (%.4f) — the symbol is too volatile for the "
                "%.1f×ATR default and no LLM-supplied stop is available. "
                "Rejecting BUY rather than falling through to naive %.0f%% "
                "stop that would be triggered on normal noise.",
                entry_price, analysis.atr_14, atr_stop,
                self.cfg.default_stop_atr_multiple,
                self.cfg.fallback_stop_pct * 100,
            )
            return None
        
        
        
        
        return round(entry_price * (1 - self.cfg.fallback_stop_pct), 2)
