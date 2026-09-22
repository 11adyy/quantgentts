"""Telegram session-status push notifications.

Disabled when TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars are
missing — callers get a no-op notifier so they don't need to branch.
HTTP failures are swallowed: a Telegram outage must never affect
trading.

Per-mode noise policy (see `format_session_result`):
  - morning / midday / close / evening: always notify on completion
  - earnings_preprocess: notify only when filings were analyzed
    (skip "nothing_new" — happens most pre-market days)
  - intra_check: notify only on emergency action (skip the 14
    silent OK ticks per trading day)
  - meta: notify on actual run; skip "not_quarter_end" (but a
    "quarter_end_check_failed" skip is a failure and notifies)
  - evening auto_meta (quarter-end piggyback): any dict result
    renders exactly one 🧪 line, including "ran but 0 proposals"
    and "skipped: calendar check failed"; only auto_meta=None
    (an ordinary evening) is silent
  - daily (P&L CSV export): the CSV itself goes out as a Telegram
    document with a self-describing caption, so the "sent" status
    text is suppressed (the document IS the confirmation); "error"
    (with the reason) and "skipped" still notify
  - Any session that raised an exception: always notify
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)







_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "quantgents.db"







_SWEEP_SYMBOLS = frozenset({"SGOV", "BIL"})


class TelegramNotifier:
    """Best-effort Telegram Bot API notifier.

    Reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from the
    environment at construction. If either is missing, `enabled`
    stays False and every `send` call is a no-op.

    `TELEGRAM_DISABLED=1` overrides the env-var path so an operator
    can mute notifications without unsetting the bot creds.
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    HTTP_TIMEOUT_S = 5.0
    
    MAX_MESSAGE_CHARS = 4000

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ):
        self.token = (token if token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        kill_switch = os.getenv("TELEGRAM_DISABLED", "").strip().lower() in ("1", "true", "yes")
        self.enabled = bool(self.token and self.chat_id) and not kill_switch
        if not self.enabled:
            if kill_switch:
                logger.info("TelegramNotifier: disabled via TELEGRAM_DISABLED env var")
            else:
                logger.info(
                    "TelegramNotifier: disabled (set TELEGRAM_BOT_TOKEN + "
                    "TELEGRAM_CHAT_ID env vars to enable)"
                )

    def send(self, text: str) -> bool:
        """Fire-and-forget send. Returns True on success.

        - No-op when not enabled (returns False).
        - Auto-truncates messages over MAX_MESSAGE_CHARS.
        - Any HTTP / network / Telegram-side error is logged and
          swallowed: trading must never fail because a notifier is
          unreachable.
        """
        if not self.enabled:
            return False
        if not text:
            return False
        if len(text) > self.MAX_MESSAGE_CHARS:
            text = text[: self.MAX_MESSAGE_CHARS - 30] + "\n[...truncated]"
        try:
            response = requests.post(
                self.API_URL.format(token=self.token),
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                timeout=self.HTTP_TIMEOUT_S,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            
            
            
            
            logger.warning("Telegram notify failed: %s", exc)
            return False

    def send_document(self, csv_bytes: bytes, filename: str, caption: str = "") -> bool:
        """Send a file (e.g. CSV) via Telegram sendDocument. Best-effort."""
        if not self.enabled:
            return False
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendDocument",
                data={"chat_id": self.chat_id, "caption": caption},
                files={"document": (filename, csv_bytes, "text/csv")},
                timeout=30.0,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Telegram send_document failed: %s", exc)
            return False







def format_session_result(
    mode: str,
    result: dict | None,
    elapsed_seconds: float,
    error: BaseException | None = None,
) -> str | None:
    """Build the human-readable message body for one completed (or
    failed) session.

    Returns None when the session shouldn't generate a notification
    per the per-mode noise policy (intra_check OK,
    earnings_preprocess nothing_new, meta skipped). Caller treats
    None as "do nothing".
    """
    from src.trading_calendar import et_now

    timestamp = et_now().strftime("%Y-%m-%d %H:%M ET")
    elapsed_str = _fmt_elapsed(elapsed_seconds)

    if error is not None:
        
        err_type = type(error).__name__
        err_msg = str(error)[:500] or "(no message)"
        return (
            f"🔴 {mode} FAILED  ({timestamp})\n"
            f"error: {err_type}: {err_msg}\n"
            f"elapsed: {elapsed_str}"
        )

    if not isinstance(result, dict):
        return (
            f"⚪ {mode} returned non-dict result ({timestamp})\n"
            f"type: {type(result).__name__}\n"
            f"elapsed: {elapsed_str}"
        )

    status = str(result.get("status", "unknown"))

    
    if mode == "intra_check" and status in ("ok", "market_holiday"):
        return None  
    if mode == "earnings_preprocess" and status in (
        "market_holiday", "nothing_new", "fetch_error",
    ):
        
        
        
        if status == "fetch_error":
            return None
        if status == "nothing_new":
            return None
        if status == "market_holiday":
            return None
    if mode == "meta" and status == "skipped":
        if result.get("reason") == "quarter_end_check_failed":
            pass  
        else:
            return None  
    if mode == "daily" and status == "sent":
        
        
        
        return None

    run_id = result.get("run_id", "?")
    emoji = _status_emoji(status)
    lines: list[str] = [
        f"{emoji} {mode}  ({timestamp})",
        f"status: {status}",
        f"run_id: {run_id}",
    ]

    
    
    
    
    
    cost_line = _session_cost_line(run_id)
    if cost_line:
        lines.append(cost_line)

    
    
    
    
    
    
    if status in ("broker_error", "fetch_error"):
        err = result.get("error")
        if err:
            lines.append(f"error: {str(err)[:300]}")

    
    if mode in ("morning", "midday", "close", "once"):
        _append_trade_session_body(lines, result)
    elif mode == "evening":
        _append_evening_body(lines, result)
    elif mode == "earnings_preprocess":
        _append_earnings_body(lines, result)
    elif mode == "intra_check":
        _append_intra_check_body(lines, result)
    elif mode == "meta":
        _append_meta_body(lines, result)
    elif mode == "daily":
        
        
        
        filename = result.get("filename", "")
        if filename:
            lines.append(f"📊 {result.get('rows', '?')} rows → {filename}")
        err = result.get("error")
        if err:
            lines.append(f"error: {err}")

    lines.append(f"elapsed: {elapsed_str}")
    return "\n".join(lines)


def _append_coverage_gap_banner(lines: list[str], result: dict) -> None:
    """Render the broker-truth stop-coverage gap banner (🔴) when the session-
    entry reconciler found held longs with less open protective-stop coverage
    than held qty — a (partially) naked position the WAL queue didn't know
    about. This is operator-actionable: a stop needs manual re-protection."""
    gaps = result.get("stop_coverage_gaps")
    if not isinstance(gaps, list) or not gaps:
        return
    parts = []
    for g in gaps[:6]:
        if not isinstance(g, dict):
            continue
        parts.append(
            f"{g.get('symbol', '?')}"
            f"({_fmt_qty(g.get('covered_qty', 0) or 0)}/{_fmt_qty(g.get('held_qty', 0) or 0)})"
        )
    lines.append(
        f"🔴 STOP-COVERAGE GAP: {len(gaps)} long(s) under-protected "
        f"(covered/held): {', '.join(parts)}"
    )


def _append_trade_session_body(lines: list[str], result: dict) -> None:
    
    
    
    
    
    
    if str(result.get("status", "")) == "analysis_error":
        lines.append(
            "🔴 PM output unparseable — no decisions were made today; "
            "this is NOT a deliberate hold (wrapper retries next 30-min tick)"
        )
        err = result.get("error")
        if err:
            lines.append(f"error: {str(err)[:300]}")

    
    _append_coverage_gap_banner(lines, result)
    orders = result.get("orders") or []

    
    
    
    
    
    
    
    
    
    
    
    
    
    forced = [
        o for o in orders
        if isinstance(o, dict) and str(o.get("action", "")).upper() in (
            "FORCE_DELEVER", "EMERGENCY_SELL", "COVER_SHORT",
        )
    ]
    if forced:
        actions = sorted({str(o.get("action", "")).upper() for o in forced})
        symbols = sorted({str(o.get("symbol", "?")) for o in forced})
        lines.append(
            f"🚨 AUTONOMOUS INTERVENTION ({', '.join(actions)}): "
            f"{len(forced)} order(s) on {', '.join(symbols)}"
        )
        covers = [o for o in forced
                  if str(o.get("action", "")).upper() == "COVER_SHORT"]
        if covers:
            cover_syms = sorted({str(o.get("symbol", "?")) for o in covers})
            lines.append(
                f"🩹 COVER_SHORT (unintended short closed): "
                f"{', '.join(cover_syms)}"
            )

    if orders:
        buys = [o for o in orders if _order_side(o) == "buy"]
        sells = [o for o in orders if _order_side(o) == "sell"]
        lines.append(f"orders: {len(orders)}  (BUY {len(buys)} / SELL {len(sells)})")
        
        
        
        
        
        for o in sells[:10]:
            
            
            action = str(o.get("action", "")).upper() if isinstance(o, dict) else ""
            label = "  SELL  "
            if action == "FORCE_DELEVER":
                label = "  🚨FORCE"
            elif action == "EMERGENCY_SELL":
                label = "  🚨EMER "
            lines.append(f"{label}{_order_summary(o)}")
        for o in buys[:10]:
            action = str(o.get("action", "")).upper() if isinstance(o, dict) else ""
            label = "  BUY   "
            if action == "COVER_SHORT":
                label = "  🩹COVER"
            lines.append(f"{label}{_order_summary(o)}")
        omitted = max(0, len(buys) - 10) + max(0, len(sells) - 10)
        if omitted:
            lines.append(f"  (+{omitted} more — see audit log)")
    else:
        lines.append("orders: 0")

    data_status = result.get("data_status") or {}
    degraded = [k for k, v in data_status.items() if v not in ("ok", "empty")]
    if degraded:
        lines.append(f"⚠️ degraded: {', '.join(sorted(degraded))}")


def _append_evening_body(lines: list[str], result: dict) -> None:
    
    analysis = result.get("analysis")

    
    
    
    
    missing = result.get("missing_sessions")
    if isinstance(missing, list) and missing:
        
        
        
        hard = [m for m in missing
                if m == "morning" or str(m).startswith("morning (")]
        for m in hard:
            detail = m if m != "morning" else (
                "morning — no agent activity logged; check the timer/scheduler"
            )
            lines.append(f"🔴 MORNING SESSION INCOMPLETE TODAY: {detail}")
        soft = [m for m in missing if m not in hard]
        if soft:
            lines.append(f"⚠️ no activity logged today for: {', '.join(soft)}")

    
    _append_coverage_gap_banner(lines, result)

    
    
    risk_for_banner = _attr_or_key(analysis, "risk_rating")
    if isinstance(risk_for_banner, str) and risk_for_banner.lower() in ("elevated", "high"):
        lines.append(f"🚨 OPERATOR ATTENTION — risk_rating={risk_for_banner}")

    
    
    
    
    
    
    
    
    
    
    
    
    esc_pnl = result.get("pnl_4pm")
    esc_close = result.get("equity_close")
    if esc_pnl is not None and isinstance(esc_close, (int, float)):
        esc_base = esc_close - esc_pnl
    else:
        esc_pnl = result.get("daily_pnl")
        esc_tv = result.get("total_value")
        esc_base = (esc_tv - esc_pnl) if (
            isinstance(esc_pnl, (int, float)) and isinstance(esc_tv, (int, float))
        ) else None
    dl_limit = result.get("max_daily_loss_pct")
    if (isinstance(esc_pnl, (int, float)) and isinstance(esc_base, (int, float))
            and isinstance(dl_limit, (int, float)) and dl_limit > 0
            and esc_pnl < 0 and esc_base > 0):
        loss_pct = abs(esc_pnl / esc_base * 100)
        if loss_pct >= 0.8 * dl_limit:
            lines.append(
                f"🚨 DETERMINISTIC ALERT — daily loss {loss_pct:.2f}% is "
                f"≥80% of the {dl_limit:.0f}% circuit-breaker limit"
            )

    
    
    
    
    
    
    
    
    
    daily_pnl = result.get("daily_pnl")
    total_value = result.get("total_value")
    pnl_4pm = result.get("pnl_4pm")
    equity_close = result.get("equity_close")

    def _fmt_pnl(v: float) -> str:
        return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"

    if pnl_4pm is not None and equity_close is not None:
        
        baseline = equity_close - pnl_4pm
        if baseline > 0:
            r = pnl_4pm / baseline * 100
            ret_str = f"+{r:.2f}%" if pnl_4pm >= 0 else f"{r:.2f}%"
        else:
            ret_str = "n/a"
        lines.append(f"💰 Daily P&L: {_fmt_pnl(pnl_4pm)} ({ret_str})  ·  4pm close")
        lines.append(f"   Equity: ${equity_close:,.2f}")
    elif daily_pnl is not None and total_value is not None:
        
        
        
        prior_equity = total_value - daily_pnl
        if prior_equity > 0:
            ret_pct = (daily_pnl / prior_equity) * 100
            ret_str = f"+{ret_pct:.2f}%" if daily_pnl >= 0 else f"{ret_pct:.2f}%"
        else:
            
            ret_str = "n/a"
        lines.append(f"💰 Daily P&L: {_fmt_pnl(daily_pnl)} ({ret_str})")
        lines.append(f"   Equity: ${total_value:,.2f}")

    
    
    
    
    
    
    
    risk_for_actions = _attr_or_key(analysis, "risk_rating")
    if isinstance(risk_for_actions, str) and risk_for_actions.lower() in ("elevated", "high"):
        actions = _attr_or_key(analysis, "suggested_actions") or []
        if isinstance(actions, list) and actions:
            lines.append("⚡ Suggested actions:")
            for act in actions[:5]:
                if not isinstance(act, str):
                    continue
                lines.append(f"   • {act[:200]}")

    
    
    
    _append_position_snapshot(lines, total_value)

    analysis = result.get("analysis")
    risk = _attr_or_key(analysis, "risk_rating")
    bias = _attr_or_key(analysis, "tomorrow_bias")
    conv = _attr_or_key(analysis, "tomorrow_conviction")
    if risk or bias or conv:
        bits = []
        if risk:
            bits.append(f"risk={risk}")
        if bias:
            bits.append(f"bias={bias}")
        if conv:
            bits.append(f"conv={conv}")
        lines.append("🔮 Tomorrow: " + "  ".join(bits))
    outlook = _attr_or_key(analysis, "tomorrow_outlook") or ""
    if outlook:
        lines.append(f"   {outlook[:280]}")

    
    
    
    
    
    
    auto_meta = result.get("auto_meta")
    if isinstance(auto_meta, dict):
        
        
        
        
        
        
        
        
        
        
        report = auto_meta.get("editor_report") or {}
        applied = len(report.get("applied") or [])
        rej_list = report.get("rejected") or []
        rejected = len(rej_list)
        staged = sum(
            1 for r in rej_list
            if isinstance(r, dict) and "dry_run" in str(r.get("reason", ""))
        )
        proposed = int(auto_meta.get("proposed_learnings_count") or 0)
        dropped = int(auto_meta.get("dropped_learnings_count") or 0)
        period = auto_meta.get("period", "?")
        status = auto_meta.get("status", "?")
        git_commit = report.get("git_commit")
        
        
        
        
        
        
        
        if status == "auto_meta_error":
            err = str(auto_meta.get("error", "?"))[:200]
            line = f"🧪 meta {period}: ERROR — {err}"
        elif status == "digest_only":
            
            
            line = (
                f"🧪 meta {period}: digest written but LLM reflection "
                f"FAILED — check logs"
            )
        elif status == "skipped":
            reason = auto_meta.get("reason", "?")
            line = f"🧪 meta {period}: skipped — {reason}"
        elif applied > 0:
            line = (
                f"🧪 meta {period}: applied {applied} learning(s); "
                f"rejected {rejected}"
            )
            if git_commit:
                line += f" commit={str(git_commit)[:7]}"
            else:
                
                
                
                
                line += (
                    " ⚠️ COMMIT FAILED — prompts edited but uncommitted; "
                    "run `git status config/prompts/`"
                )
        elif staged > 0:
            
            line = (
                f"🧪 meta {period}: {staged} proposal(s) staged "
                f"(dry-run — see data/evolution/{period}/proposed_edits.json)"
            )
        elif rejected > 0:
            
            
            line = (
                f"🧪 meta {period}: 0 applied / {rejected} rejected "
                f"(see data/evolution/edits.jsonl)"
            )
        elif proposed > 0:
            
            
            
            line = (
                f"🧪 meta {period}: {proposed} proposal(s) generated but "
                f"prompt-editor report missing — check logs"
            )
        else:
            
            
            
            line = (
                f"🧪 meta {period}: ran, 0 proposal(s) survived schema "
                f"({dropped} dropped pre-editor — see "
                f"data/evolution/{period}/reflection.json dropped_learnings)"
            )
        if dropped > 0 and "dropped pre-editor" not in line:
            line += f" ({dropped} dropped pre-editor)"
        if auto_meta.get("calendar_fallback"):
            line += (
                " (quarter-end decided by weekday fallback — Alpaca "
                "calendar failed)"
            )
        lines.append(line)


def _session_cost_line(run_id: str | None) -> str | None:
    """Return '💵 cost: $X.XX (N calls)' for a session's run_id, or
    None when the lookup can't produce a clean answer.

    Reasons for returning None (and not displaying anything):
      - No run_id (mode didn't set one — e.g. live scheduler startup ping)
      - DB file not at default path (test environments)
      - No agent_log rows for this run_id (session crashed before any
        LLM call landed — error path notification already covers this)
      - Some row has cost_usd=NULL (model missing from cost_table) —
        showing partial sum would understate; better to render nothing
        and let the operator notice the gap when they hit the
        agent_logs table directly.
    """
    if not run_id or run_id == "?":
        return None
    try:
        import sqlite3
        if not _DB_PATH.exists():
            return None
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            rows = conn.execute(
                "SELECT cost_usd FROM agent_logs WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("session cost lookup failed for %s: %s", run_id, exc)
        return None
    if not rows:
        return None
    if any(r[0] is None for r in rows):
        
        
        return f"💵 cost: $?.?? ({len(rows)} calls — see cost_table.py)"
    total = sum(float(r[0]) for r in rows)
    
    
    
    if total < 0.01:
        return f"💵 cost: ${total:.4f} ({len(rows)} calls)"
    return f"💵 cost: ${total:,.2f} ({len(rows)} calls)"


def _append_position_snapshot(lines: list[str], total_value: float | None) -> None:
    """Render top-3 winners + top-3 losers by unrealized P&L from the
    live positions table. Read-only DB hit; degrades gracefully on any
    error (the rest of the message still goes out)."""
    try:
        import sqlite3
        
        
        
        if not _DB_PATH.exists():
            return
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            rows = conn.execute(
                "SELECT symbol, qty, avg_entry, current_price, "
                "market_value, unrealized_pnl FROM positions "
                "WHERE qty > 0 ORDER BY unrealized_pnl DESC"
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("evening position snapshot failed: %s", exc)
        return
    if not rows:
        return
    
    
    
    
    
    
    parked = sum(r[4] for r in rows
                 if r[0] in _SWEEP_SYMBOLS and r[4] is not None)
    rows = [r for r in rows if r[0] not in _SWEEP_SYMBOLS]
    invested = sum(r[4] for r in rows if r[4] is not None)
    cash_pct = None
    if total_value and total_value > 0:
        cash_pct = max(0.0, (total_value - invested) / total_value * 100)
    summary = f"   Positions: {len(rows)}  invested ${invested:,.0f}"
    if cash_pct is not None:
        summary += f"  ({100 - cash_pct:.0f}% deployed / {cash_pct:.0f}% cash)"
    if parked > 0:
        summary += f"  [+${parked:,.0f} parked in T-bills]"
    lines.append(summary)
    if not rows:
        return

    def _row_line(r: tuple) -> str:
        sym, qty, avg, curr, mv, pnl = r
        pct = ((curr / avg - 1) * 100) if avg else 0
        sign = "+" if pnl >= 0 else "-"
        return f"   {sym:<6} {sign}${abs(pnl):>8,.0f}  ({pct:+.1f}%)"

    
    
    
    
    
    
    winners = [r for r in rows if r[5] is not None and r[5] > 0][:3]
    if winners:
        lines.append("📈 Top winners:")
        for r in winners:
            lines.append(_row_line(r))
    losers = [r for r in rows if r[5] is not None and r[5] < 0][-3:][::-1]
    if losers:
        lines.append("📉 Underwater:")
        for r in losers:
            lines.append(_row_line(r))


def _append_earnings_body(lines: list[str], result: dict) -> None:
    analyzed = result.get("analyzed", 0)
    confirmed = result.get("confirmed", 0)
    failed = result.get("failed", 0)
    lines.append(f"analyzed: {analyzed}  confirmed: {confirmed}  failed: {failed}")


def _append_intra_check_body(lines: list[str], result: dict) -> None:
    
    
    emergency = result.get("orders") or result.get("emergency_orders") or []
    if emergency:
        lines.append(f"⚠️ EMERGENCY orders: {len(emergency)}")
        for o in emergency[:5]:
            lines.append(f"  {_order_summary(o)}")
    reason = result.get("reason")
    if reason:
        lines.append(f"reason: {reason}")


def _append_meta_body(lines: list[str], result: dict) -> None:
    period = result.get("period")
    if period:
        lines.append(f"period: {period}")
    status = str(result.get("status", ""))
    if status == "skipped" and result.get("reason") == "quarter_end_check_failed":
        
        
        lines.append(
            "⚠️ quarter-end check FAILED (Alpaca calendar unavailable) — "
            "meta did not run; re-run with --force / --period-end once "
            "the broker is reachable"
        )
    
    
    
    
    report = result.get("editor_report") or {}
    applied = len(report.get("applied") or [])
    rej_list = report.get("rejected") or []
    rejected = len(rej_list)
    staged = sum(
        1 for r in rej_list
        if isinstance(r, dict) and "dry_run" in str(r.get("reason", ""))
    )
    dropped = int(result.get("dropped_learnings_count") or 0)
    if applied or rejected:
        lines.append(f"learnings: applied={applied} rejected={rejected}")
        if applied:
            git_commit = report.get("git_commit")
            if git_commit:
                lines.append(f"commit: {str(git_commit)[:7]}")
            else:
                lines.append(
                    "⚠️ COMMIT FAILED — prompts edited but uncommitted; "
                    "run `git status config/prompts/`"
                )
        if staged:
            lines.append(
                f"🧪 {staged} proposal(s) staged for review — "
                f"data/evolution/{period}/proposed_edits.json"
            )
    elif result.get("proposed_learnings_count"):
        lines.append(
            f"⚠️ {result['proposed_learnings_count']} proposal(s) generated "
            f"but prompt-editor report missing — check logs"
        )
    elif status in ("reflected", "applied_saved"):
        lines.append(
            f"🧪 ran, 0 proposal(s) survived schema ({dropped} dropped "
            f"pre-editor — see data/evolution/{period}/reflection.json "
            f"dropped_learnings)"
        )
    if dropped and not any("dropped pre-editor" in ln for ln in lines):
        lines.append(f"⚠️ {dropped} proposal(s) dropped pre-editor (schema)")
    if result.get("calendar_fallback"):
        lines.append(
            "⚠️ quarter-end decided by weekday fallback — Alpaca calendar failed"
        )
    reason = result.get("reason")
    if reason:
        lines.append(f"reason: {reason}")




def _status_emoji(status: str) -> str:
    if status in (
        "executed", "analyzed", "reviewed", "preprocessed", "reflected",
        "sent",
    ):
        return "🟢"
    if status in (
        "no_trades", "no_data", "nothing_new", "ok",
        "market_holiday", "early_close",
    ):
        return "⚪"
    
    
    
    
    
    if status in ("emergency_sold", "hard_risk_block", "digest_only"):
        return "🟡"
    if "error" in status or status in ("rejected", "failed"):
        return "🔴"
    return "⚪"


def _order_side(order: Any) -> str:
    """Best-effort extract of order side. Order shape varies by
    submission path: some are Alpaca SDK response dicts (have
    'side'), some are internal {'symbol','action',...} dicts."""
    if not isinstance(order, dict):
        return ""
    side = order.get("side")
    if isinstance(side, str):
        return side.lower()
    action = str(order.get("action", "")).upper()
    if any(s in action for s in (
        "SELL", "REDUCE", "TAKE_PROFIT", "EMERGENCY_SELL",
        "FORCE_DELEVER", "PARTIAL_SELL",
    )):
        return "sell"
    if action in ("BUY", "COVER_SHORT"):
        return "buy"
    return ""


def _order_summary(order: Any) -> str:
    """Render one order line like 'NVDA   qty=5  @$420.50  SL=$405.00'.

    Falls back gracefully when fields are missing (older broker
    response shapes, or close_position which only returns id/status)."""
    if not isinstance(order, dict):
        return str(order)[:60]
    sym = str(order.get("symbol", "?"))
    parts: list[str] = [f"{sym:<6}"]
    qty = order.get("qty") or order.get("filled_qty")
    if qty is not None:
        parts.append(f"qty={_fmt_qty(qty)}")
    
    
    lim = order.get("limit_price") or order.get("price")
    if lim is not None and lim > 0:
        parts.append(f"@${_fmt_price(lim)}")
    sl = order.get("stop_loss_price")
    if sl is not None and sl > 0:
        parts.append(f"SL=${_fmt_price(sl)}")
    return "  ".join(parts)


def _fmt_qty(qty: Any) -> str:
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return str(qty)
    
    
    return f"{int(q)}" if q == int(q) else f"{q:g}"


def _fmt_price(price: Any) -> str:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return str(price)
    
    return f"{p:.4f}" if p < 1.0 else f"{p:,.2f}"


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def build_daily_csv(closes: list[tuple[str, float]]) -> bytes:
    """Build a P&L history CSV from portfolio_history closes.

    Columns: Date, NAV, Daily P&L, Daily Return %, Drawdown %, SPY Close,
    SPY Return %

    SPY data is fetched via yfinance for the same date range. On any
    yfinance failure the SPY columns are left blank.
    """
    import io, csv, math
    from datetime import datetime, timedelta

    if not closes:
        return b""

    
    spy_closes: dict[str, float] = {}
    try:
        import yfinance as yf
        import pandas as pd
        earliest = closes[0][0]
        start = (datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
        end_dt = datetime.strptime(closes[-1][0], "%Y-%m-%d") + timedelta(days=2)
        end = end_dt.strftime("%Y-%m-%d")
        df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if not df.empty:
            if hasattr(df.columns, "get_level_values"):
                df.columns = df.columns.get_level_values(0)
            
            
            
            
            for dt_idx, row in df["Close"].dropna().items():
                val = float(row)
                if math.isfinite(val):
                    spy_closes[str(dt_idx.date())] = val
    except Exception as exc:
        logger.warning("build_daily_csv: SPY fetch failed: %s", exc)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "NAV", "Daily P&L", "Daily Return %", "Drawdown %", "SPY Close", "SPY Return %"])

    prev_nav: float | None = None
    prev_spy: float | None = None
    peak_nav: float | None = None
    for date, nav in closes:
        daily_pnl = nav - prev_nav if prev_nav is not None else 0.0
        daily_ret = (daily_pnl / prev_nav * 100) if prev_nav else 0.0
        peak_nav = max(peak_nav, nav) if peak_nav is not None else nav
        drawdown = (nav - peak_nav) / peak_nav * 100 if peak_nav else 0.0
        spy_close = spy_closes.get(date)
        if spy_close is not None and math.isfinite(spy_close) and prev_spy:
            spy_ret = (spy_close - prev_spy) / prev_spy * 100
        else:
            spy_ret = ""
        writer.writerow([
            date,
            f"{nav:.2f}",
            f"{daily_pnl:+.2f}",
            f"{daily_ret:+.4f}",
            f"{drawdown:+.4f}",
            f"{spy_close:.2f}" if spy_close else "",
            f"{spy_ret:+.4f}" if spy_ret != "" else "",
        ])
        prev_nav = nav
        prev_spy = spy_close if spy_close else prev_spy

    return buf.getvalue().encode("utf-8")


def _attr_or_key(obj: Any, name: str) -> Any:
    """Get `name` from either an attribute (Pydantic model) or a
    dict key (raw JSON). Returns None on miss without raising."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
