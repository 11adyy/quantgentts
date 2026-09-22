import json
import logging
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.cost_table import estimate_cost, fmt_cost

logger = logging.getLogger(__name__)


_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")









_DEEPSEEK_PREFIXES = ("deepseek-",)
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"  








_DEEPSEEK_MAX_OUTPUT = {
    "deepseek-v4-flash": 384000,
    "deepseek-v4-pro":   384000,
    "deepseek-chat":     384000,  
    "deepseek-reasoner": 384000,  
}
_DEEPSEEK_DEFAULT_CEILING = 8192  

























_DEFAULT_MAX_RETRIES = 7


def _retry_backoff_seconds(attempt: int) -> float:
    """Exponential base + full positive jitter on top.

    Returns a sleep duration in [2**attempt, 2 * 2**attempt). The
    deterministic floor preserves exponential spacing (so retries
    don't bunch right at the start), while the random ceiling
    decorrelates retries within a single sequence and across
    concurrent callers.

    Sequence for attempt 0..5 (the 6 between-attempt sleeps with N=7):
      [1, 2), [2, 4), [4, 8), [8, 16), [16, 32), [32, 64)
    """
    base = 2 ** attempt
    return base + random.uniform(0, base)

















_LLM_HTTP_TIMEOUT = 300.0



















_DEFAULT_RETRY_DEADLINE_S = 480.0


def _retry_deadline_s() -> float:
    raw = os.environ.get("QUANTGENTS_RETRY_DEADLINE_S")
    if raw is None:
        return _DEFAULT_RETRY_DEADLINE_S
    try:
        v = float(raw)
    except ValueError:
        return _DEFAULT_RETRY_DEADLINE_S
    return max(1.0, v)








_RETRY_AFTER_CAP_S = 120.0


def _retry_after_hint_seconds(exc: Exception) -> float | None:
    """Best-effort extraction of a server retry-after hint from an SDK error.

    Looks in (a) the Retry-After header of the attached httpx response
    (numeric-seconds form only — the HTTP-date form isn't worth parsing for
    a hint), (b) a retry_after field in the error body dict, (c) the message
    text (relay 524 bodies embed '"retry_after": 120'). Returns None when no
    usable hint exists; never raises.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after")
        except Exception:  # noqa: BLE001 — a weird headers object must not mask the real error
            raw = None
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        val = body.get("retry_after", body.get("retry-after"))
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return max(0.0, float(val))
    m = re.search(r'retry[_-]after["\']?\s*[:=]\s*"?(\d+(?:\.\d+)?)', str(exc), re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default














_OPENAI_MAX_CONCURRENT = _int_env("QUANTGENTS_MAX_CONCURRENT_LLM", 3)
_OPENAI_LLM_SEMAPHORE = threading.Semaphore(_OPENAI_MAX_CONCURRENT)
_ANTHROPIC_MAX_CONCURRENT = 4
_ANTHROPIC_LLM_SEMAPHORE = threading.Semaphore(_ANTHROPIC_MAX_CONCURRENT)









_TRUNCATION_FINISH_REASONS = ("max_tokens", "length", "insufficient_system_resource")


class LLMEmptyResponseError(RuntimeError):
    """HTTP 200 whose body carries no usable content (choices empty /
    content None or ""). Previously returned as a *successful* '' — which
    parses to None downstream and masquerades as a deliberate no-signal,
    consuming the agent's one shot for the session while bypassing both the
    retry budget and the Anthropic failover. Raised instead, and classified
    retryable (a degenerate 200 from a relay is transient territory)."""


class LLMStreamInterruptedError(RuntimeError):
    """A streamed response ended without a finish_reason — the connection
    was cut mid-generation (relay/proxy drop, no error frame). Partial text
    is NOT a success: a half-emitted PM decision parses like 'no trades'.
    Retryable."""


def _estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic — usage-chunk fallback only, so a relay that
    doesn't honor stream_options include_usage still yields a nonzero cost
    estimate instead of a 0/0 that run() flags as cost-unknown."""
    return max(1, len(text) // 4) if text else 0


def _max_retries() -> int:
    """Read at call time so tests can monkeypatch the env var per case
    without reloading the module."""
    raw = os.environ.get("QUANTGENTS_MAX_RETRIES")
    if raw is None:
        return _DEFAULT_MAX_RETRIES
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_MAX_RETRIES
    return max(1, n)


def _is_openai_model(model: str) -> bool:
    return any(model.startswith(p) for p in _OPENAI_PREFIXES)


def _is_deepseek_model(model: str) -> bool:
    return any(model.startswith(p) for p in _DEEPSEEK_PREFIXES)









_FALLBACK_MODEL = "claude-opus-4-7"





_RETRYABLE_EXC_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "APIConnectionTimeoutError",
    "InternalServerError", "RateLimitError", "APIError",
    "Timeout", "ConnectionError", "ConnectTimeout", "ReadTimeout",
    
    
    
    "LLMEmptyResponseError", "LLMStreamInterruptedError",
})


def _is_retryable(exc: Exception) -> bool:
    """Decide whether an LLM-call exception is worth retrying.

    The old loop retried EVERY exception identically, so a non-transient
    failure — a 401 (dead key), a 400 (bad request), a 429-vs-quota-
    exhausted, a context-length-exceeded — burned the full ~140s backoff
    budget per agent for something that can never succeed, and with 4-5
    agents/session could push the run toward the 1200s outer kill. It also
    blurred the distinction the operator most needs: 'network blipped' vs
    'your key is dead' (exactly the 2026-05-11 quota-exhaustion case).

    Retry on: transient connection/timeout classes, HTTP 429, and 5xx.
    Fast-fail on: any other 4xx (auth / bad-request / not-found / context
    length). Unknown exceptions with no status code retry conservatively
    (preserves the prior catch-all behavior for genuinely unexpected
    local/network errors).
    """
    if type(exc).__name__ in _RETRYABLE_EXC_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        
        
        
        
        if status == 402:
            return False
        return status == 429 or status >= 500
    
    
    
    return True


@dataclass
class AgentResult:
    raw_text: str
    tokens_used: int
    model: str
    user_message: str = ""
    
    
    
    
    
    
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    
    
    
    
    
    
    finish_reason: str | None = None
    truncated: bool = False

    
    
    
    
    _EXPECTED_AGENT_KEY_WEIGHTS = {
        "decisions": 50,           
        "approved": 50,            
        "actions": 50,             
        "daily_summary": 40,       
        "tomorrow_outlook": 40,    
        "regime": 40,              
        "investment_implications": 40,  
        "macro_narrative": 40,     
        "analyses": 40,            
        "portfolio_view": 20,      
        "reasoning_chain": 20,     
        "symbol": 5,               
        "rating": 5,               
    }

    @staticmethod
    def _shape_score(parsed) -> int:
        """How 'agent-output shaped' a JSON candidate looks. Higher is better."""
        
        
        
        
        
        
        
        
        
        
        if isinstance(parsed, list):
            return sum(AgentResult._shape_score(item) for item in parsed)
        if not isinstance(parsed, dict):
            return 0
        keys = set(parsed.keys())
        return sum(
            weight
            for key, weight in AgentResult._EXPECTED_AGENT_KEY_WEIGHTS.items()
            if key in keys
        )

    def parse_json(self) -> dict | list | None:
        text = self.raw_text.strip()
        try:
            parsed = json.loads(text)
            
            
            return parsed
        except json.JSONDecodeError:
            pass

        candidates: list[tuple[int, int, int, dict | list]] = []
        
        idx = 0
        
        for match in re.finditer(r"```(?:json)?\s*\n(.*?)\n```", self.raw_text, re.DOTALL):
            try:
                parsed = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
            candidates.append((self._shape_score(parsed), len(json.dumps(parsed)), idx, parsed))
            idx += 1

        decoder = json.JSONDecoder()
        for i, ch in enumerate(self.raw_text):
            if ch not in "{[":
                continue
            try:
                parsed, end = decoder.raw_decode(self.raw_text[i:])
            except json.JSONDecodeError:
                continue
            candidates.append((self._shape_score(parsed), len(json.dumps(parsed)), idx, parsed))
            idx += 1
        if candidates:
            max_shape = max(item[0] for item in candidates)
            if max_shape > 0:
                
                
                shaped = [item for item in candidates if item[0] == max_shape]
                return max(shaped, key=lambda item: (item[2], item[1]))[3]

            
            
            return max(candidates, key=lambda item: (item[1], item[2]))[3]

        logger.warning("Failed to parse agent response as JSON: %s", self.raw_text[:200])
        return None


class BaseAgent(ABC):
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096,
                 fallback_api_key: str = ""):
        self.model = model
        self.max_tokens = max_tokens
        self._use_deepseek = _is_deepseek_model(model)
        self._use_openai = _is_openai_model(model)
        
        
        
        
        
        self._fallback_api_key = (fallback_api_key or "").strip()

        
        
        
        
        
        
        if self._use_deepseek:
            
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL,
                                 timeout=_LLM_HTTP_TIMEOUT, max_retries=0)
        elif self._use_openai:
            from openai import OpenAI
            
            
            
            
            
            
            base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
            
            
            
            
            
            
            
            ca_bundle = os.environ.get("OPENAI_CA_BUNDLE", "").strip()
            if ca_bundle:
                
                
                
                try:
                    from openai import DefaultHttpxClient as _HttpClient
                except ImportError:  # pragma: no cover - very old SDKs
                    import httpx
                    _HttpClient = httpx.Client
                self.client = OpenAI(
                    api_key=api_key, base_url=base_url,
                    http_client=_HttpClient(verify=ca_bundle, timeout=_LLM_HTTP_TIMEOUT),
                    max_retries=0,
                )
            else:
                self.client = OpenAI(api_key=api_key, base_url=base_url,
                                     timeout=_LLM_HTTP_TIMEOUT, max_retries=0)
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key, timeout=_LLM_HTTP_TIMEOUT, max_retries=0)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def build_user_message(self, **kwargs) -> str:
        ...

    def run(self, **kwargs) -> AgentResult:
        user_message = self.build_user_message(**kwargs)
        return self._execute(user_message)

    def _execute(self, user_message: str) -> AgentResult:
        """The retry / cross-provider-failover / cost / parse loop, decoupled
        from build_user_message so a stored historical `input_message` can be
        replayed through the CURRENT prompt + model without rebuilding context
        (see src/replay.py / scripts/replay_decision.py). `run()` = build +
        `_execute`; behavior is identical to the pre-extraction loop."""
        logger.info("Agent %s running with model %s", self.name, self.model)
        logger.info("Agent %s input:\n%s", self.name, user_message)

        max_retries = _max_retries()
        deadline_s = _retry_deadline_s()
        loop_start = time.monotonic()
        finish_reason: str | None = None
        primary_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                if self._use_deepseek:
                    raw_text, input_tokens, output_tokens, finish_reason = self._call_deepseek(user_message)
                elif self._use_openai:
                    raw_text, input_tokens, output_tokens, finish_reason = self._call_openai(user_message)
                else:
                    raw_text, input_tokens, output_tokens, finish_reason = self._call_anthropic(user_message)
                primary_error = None
                break
            except Exception as e:
                primary_error = e
                
                
                
                if not _is_retryable(e):
                    logger.warning(
                        "Agent %s attempt %d hit a non-retryable error: %s. "
                        "No more retries.", self.name, attempt + 1, e,
                    )
                    break
                
                
                if attempt == max_retries - 1:
                    logger.warning("Agent %s attempt %d failed: %s. Primary exhausted.",
                                   self.name, attempt + 1, e)
                    break
                
                
                
                
                
                
                elapsed = time.monotonic() - loop_start
                if elapsed >= deadline_s:
                    logger.warning(
                        "Agent %s attempt %d failed: %s. Retry deadline %.0fs "
                        "exceeded (elapsed %.0fs) — abandoning primary, "
                        "proceeding to failover if configured.",
                        self.name, attempt + 1, e, deadline_s, elapsed,
                    )
                    break
                wait = _retry_backoff_seconds(attempt)
                
                
                
                hint = _retry_after_hint_seconds(e)
                if hint is not None:
                    wait = min(max(wait, hint), _RETRY_AFTER_CAP_S)
                logger.warning("Agent %s attempt %d failed: %s. Retrying in %.1fs...",
                               self.name, attempt + 1, e, wait)
                time.sleep(wait)

        
        actual_model = self.model
        if primary_error is not None:
            
            
            
            
            
            
            
            failover = None
            if (self._use_openai or self._use_deepseek) and self._fallback_api_key:
                failover = self._try_failover(user_message, primary_error)
            if failover is None:
                raise primary_error
            raw_text, input_tokens, output_tokens, finish_reason = failover
            actual_model = _FALLBACK_MODEL

        
        
        
        
        
        
        truncated = (isinstance(finish_reason, str)
                     and finish_reason.lower() in _TRUNCATION_FINISH_REASONS)
        if truncated:
            logger.warning(
                "Agent %s response was TRUNCATED (finish_reason=%s) — output is "
                "incomplete, likely hit max_tokens=%d. Treat downstream None as "
                "'cut off', not 'no signal'.",
                self.name, finish_reason, self.max_tokens,
            )

        tokens = input_tokens + output_tokens
        
        
        
        
        
        
        
        if input_tokens == 0 and output_tokens == 0:
            cost = None
            logger.warning(
                "Agent %s completed with zero tokens reported — flagging cost as unknown. "
                "Either the SDK didn't return usage data, or the call somehow consumed nothing. "
                "Check the LLM response and update _extract_*_usage if there's a new shape.",
                self.name,
            )
        else:
            cost = estimate_cost(actual_model, input_tokens, output_tokens)
        logger.info(
            "Agent %s completed | tokens in=%d out=%d total=%d | cost=%s | model=%s",
            self.name, input_tokens, output_tokens, tokens,
            fmt_cost(cost), actual_model,
        )
        logger.info("Agent %s output:\n%s", self.name, raw_text)
        return AgentResult(
            raw_text=raw_text,
            tokens_used=tokens,
            model=actual_model,
            user_message=user_message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            finish_reason=finish_reason,
            truncated=truncated,
        )

    def _anthropic_call(self, client, model: str, user_message: str) -> tuple[str, int, int, str | None]:
        """One Anthropic messages.create against an arbitrary client+model.

        Shared by the primary path (_call_anthropic) and the provider-agnostic
        failover to Anthropic (_try_failover) so both use the identical request shape +
        usage/finish-reason extraction. System prompt is sent as an ephemeral
        cache breakpoint (static per agent → cheaper + lower latency; no-op
        below the cache minimum, safe unconditionally).
        """
        with _ANTHROPIC_LLM_SEMAPHORE:
            response = client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=[{
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_message}],
            )
        in_tok, out_tok = _extract_anthropic_usage(response, self.name)
        finish_reason = getattr(response, "stop_reason", None)
        if not isinstance(finish_reason, str):
            finish_reason = None
        if not response.content or not hasattr(response.content[0], "text"):
            if finish_reason in _TRUNCATION_FINISH_REASONS:
                
                
                logger.warning("Anthropic returned empty content (stop_reason=%s)", finish_reason)
                return ("", in_tok, out_tok, finish_reason)
            raise LLMEmptyResponseError(
                f"Anthropic returned empty content (stop_reason={finish_reason})"
            )
        return (response.content[0].text, in_tok, out_tok, finish_reason)

    def _call_anthropic(self, user_message: str) -> tuple[str, int, int, str | None]:
        return self._anthropic_call(self.client, self.model, user_message)

    def _try_failover(self, user_message: str, primary_error: Exception):
        """Primary provider (OpenAI or DeepSeek) failed → attempt ONE Anthropic
        call with the fallback model. Returns the (text, in_tok, out_tok,
        finish_reason) tuple on success, or None on failure (caller re-raises
        the original primary error). Single-shot: the primary already burned its
        retry budget, so a
        second full budget here could blow the session window. Loud logging
        either way — a provider failover is an event the operator must see.
        """
        logger.error(
            "Agent %s: primary model %s failed after retries (%s) — failing over "
            "to %s on Anthropic.", self.name, self.model, primary_error, _FALLBACK_MODEL,
        )
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self._fallback_api_key,
                               timeout=_LLM_HTTP_TIMEOUT, max_retries=0)
            result = self._anthropic_call(client, _FALLBACK_MODEL, user_message)
            logger.warning(
                "Agent %s: FAILOVER to %s SUCCEEDED (in=%d out=%d) — session "
                "continues on Anthropic.", self.name, _FALLBACK_MODEL, result[1], result[2],
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Agent %s: failover to %s also FAILED: %s. Re-raising the "
                "original primary error.", self.name, _FALLBACK_MODEL, exc,
            )
            return None

    def _call_openai(self, user_message: str) -> tuple[str, int, int, str | None]:
        """OpenAI path is STREAMED on purpose. The OPENAI_BASE_URL relay sits
        behind Cloudflare, whose ~120s Proxy Read Timeout (HTTP 524) kills any
        call that sends zero bytes until the model finishes — and PM / tech /
        evening generations legitimately run 120s+, so non-streaming could
        never succeed through the relay (the 2026-06-08/09 outage: every long
        call 524'd, book froze sell-only). Streaming keeps bytes flowing so
        the proxy window never trips; _LLM_HTTP_TIMEOUT becomes a per-chunk
        read timeout, so a long *healthy* generation isn't axed either.

        The semaphore covers create + iteration: for a streamed response the
        request is in flight (and counts against the relay's per-user
        concurrency cap) until the last chunk is read.
        """
        with _OPENAI_LLM_SEMAPHORE:
            stream = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
                stream_options={"include_usage": True},
            )
            parts: list[str] = []
            finish_reason: str | None = None
            usage = None
            for chunk in stream:
                
                
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    parts.append(piece)
                fr = getattr(choice, "finish_reason", None)
                if isinstance(fr, str):
                    finish_reason = fr
        content = "".join(parts)
        if finish_reason is None:
            
            
            
            
            raise LLMStreamInterruptedError(
                f"OpenAI stream ended without finish_reason after {len(content)} "
                "chars — connection cut mid-generation; partial output discarded"
            )
        if not content and finish_reason not in _TRUNCATION_FINISH_REASONS:
            
            
            
            
            raise LLMEmptyResponseError(
                f"OpenAI returned empty content (finish_reason={finish_reason})"
            )
        if usage is not None:
            in_tok = _coerce_token_count(getattr(usage, "prompt_tokens", 0))
            out_tok = _coerce_token_count(getattr(usage, "completion_tokens", 0))
        else:
            
            
            
            in_tok = _estimate_tokens(self.system_prompt + user_message)
            out_tok = _estimate_tokens(content)
            logger.warning(
                "OpenAI stream for %s carried no usage chunk — token counts "
                "are chars/4 estimates (in≈%d out≈%d).",
                self.name, in_tok, out_tok,
            )
        return (content, in_tok, out_tok, finish_reason)

    def _deepseek_max_output(self) -> int:
        """Clamp ceiling for this DeepSeek model. DeepSeek REJECTS (does not
        clamp) a max_tokens above the model limit, so we cap client-side."""
        return _DEEPSEEK_MAX_OUTPUT.get(self.model, _DEEPSEEK_DEFAULT_CEILING)

    def _call_deepseek(self, user_message: str) -> tuple[str, int, int, str | None]:
        """DeepSeek via the OpenAI SDK (custom base_url). Three deltas vs
        _call_openai:
          1. Sends `max_tokens` (DeepSeek ignores OpenAI's `max_completion_tokens`
             → output would silently fall back to a ~4096 default and truncate).
          2. Clamps to the per-model output ceiling (DeepSeek 400s on over-ceiling
             values instead of clamping).
          3. Reads the non-standard reasoning_content defensively. We DISCARD the
             chain-of-thought (every agent parses JSON from `content`), but log its
             presence so an empty-content / full-CoT truncation is visible rather
             than looking like a clean "no signal".
        Usage is OpenAI-shaped (prompt_tokens / completion_tokens) → reuse
        _extract_openai_usage.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=min(self.max_tokens, self._deepseek_max_output()),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        if not isinstance(finish_reason, str):
            finish_reason = None
        if not content:
            reasoning = getattr(choice.message, "reasoning_content", None)
            if finish_reason not in _TRUNCATION_FINISH_REASONS:
                
                
                raise LLMEmptyResponseError(
                    f"DeepSeek returned empty content (finish_reason={finish_reason}, "
                    f"reasoning_content present={bool(reasoning)})"
                )
            
            
            logger.warning(
                "DeepSeek returned empty content (finish_reason=%s, reasoning_content present=%s)",
                finish_reason, bool(reasoning),
            )
        in_tok, out_tok = _extract_openai_usage(response, self.name)
        return (content, in_tok, out_tok, finish_reason)













def _coerce_token_count(value) -> int:
    """Return value as int iff it really IS an int (numpy.int64 subclasses
    int, so those work too). Anything else — None, MagicMock auto-attrs,
    a stray dict, a string — coerces to 0.

    This is defensive against two cases that have actually shown up:
      (1) tests using ``MagicMock`` without an explicit spec — attribute
          access auto-creates a child MagicMock whose ``__int__`` returns
          1, which would silently add +1 to every uncovered token field
          (caught by the R7 self-audit: existing tests started failing
          with 'assert 2502 == 2500' after we began summing the cache
          fields, because the cache fields weren't set in the mocks).
      (2) future SDK changes that turn a numeric field into a string
          or object — better to under-count than crash, since the
          run() layer flags 0+0 tokens as cost=unknown anyway.
    """
    
    
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _extract_anthropic_usage(response, agent_name: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.warning(
            "Anthropic response for %s missing usage object — cost will be flagged as unknown",
            agent_name,
        )
        return (0, 0)
    in_tok = _coerce_token_count(getattr(usage, "input_tokens", 0))
    cache_create = _coerce_token_count(getattr(usage, "cache_creation_input_tokens", 0))
    cache_read = _coerce_token_count(getattr(usage, "cache_read_input_tokens", 0))
    out_tok = _coerce_token_count(getattr(usage, "output_tokens", 0))
    
    
    
    return (in_tok + cache_create + cache_read, out_tok)


def _extract_openai_usage(response, agent_name: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.warning(
            "OpenAI response for %s missing usage object — cost will be flagged as unknown",
            agent_name,
        )
        return (0, 0)
    return (
        _coerce_token_count(getattr(usage, "prompt_tokens", 0)),
        _coerce_token_count(getattr(usage, "completion_tokens", 0)),
    )
