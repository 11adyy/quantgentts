import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.cost_table import estimate_cost, fmt_cost

logger = logging.getLogger(__name__)


_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")

























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


@dataclass
class AgentResult:
    raw_text: str
    tokens_used: int
    model: str
    user_message: str = ""
    
    
    
    
    
    
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    
    
    
    
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
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self._use_openai = _is_openai_model(model)

        if self._use_openai:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, timeout=_LLM_HTTP_TIMEOUT)
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key, timeout=_LLM_HTTP_TIMEOUT)

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
        logger.info("Agent %s running with model %s", self.name, self.model)
        logger.info("Agent %s input:\n%s", self.name, user_message)

        max_retries = _max_retries()
        for attempt in range(max_retries):
            try:
                if self._use_openai:
                    raw_text, input_tokens, output_tokens = self._call_openai(user_message)
                else:
                    raw_text, input_tokens, output_tokens = self._call_anthropic(user_message)
                break
            except Exception as e:
                
                
                if attempt == max_retries - 1:
                    logger.warning("Agent %s attempt %d failed: %s. Giving up.",
                                   self.name, attempt + 1, e)
                    raise
                wait = _retry_backoff_seconds(attempt)
                logger.warning("Agent %s attempt %d failed: %s. Retrying in %.1fs...",
                               self.name, attempt + 1, e, wait)
                import time
                time.sleep(wait)

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
            cost = estimate_cost(self.model, input_tokens, output_tokens)
        logger.info(
            "Agent %s completed | tokens in=%d out=%d total=%d | cost=%s | model=%s",
            self.name, input_tokens, output_tokens, tokens,
            fmt_cost(cost), self.model,
        )
        logger.info("Agent %s output:\n%s", self.name, raw_text)
        return AgentResult(
            raw_text=raw_text,
            tokens_used=tokens,
            model=self.model,
            user_message=user_message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _call_anthropic(self, user_message: str) -> tuple[str, int, int]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        in_tok, out_tok = _extract_anthropic_usage(response, self.name)
        if not response.content or not hasattr(response.content[0], "text"):
            logger.warning("Anthropic returned empty content (stop_reason=%s)", response.stop_reason)
            return ("", in_tok, out_tok)
        return (response.content[0].text, in_tok, out_tok)

    def _call_openai(self, user_message: str) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content or ""
        if not content:
            refusal = getattr(response.choices[0].message, "refusal", None)
            logger.warning("OpenAI returned empty content (refusal=%s)", refusal)
        in_tok, out_tok = _extract_openai_usage(response, self.name)
        return (content, in_tok, out_tok)













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
