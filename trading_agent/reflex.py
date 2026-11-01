"""The REFLEX: Jev (TypeSafe AI System One) judges one compact snapshot per
candle and returns typed, probability-scored answers in one call.

Wire shape (POST {JEV_API_URL}, Authorization: Bearer <key>):
  request  {"model": "jev-latest", "state": "<compact snapshot JSON>",
            "questions": {name: {"type": "choice"|"score"|"noul", "instructions": "...", "criteria": ...}}}
  response {"model": "jev-1.13.0", "answers": {name: {"type": "choice", "choice": "...", "confidence": 0.92,
            "probabilities": {...}} | {"type": "score", "score": 2, "confidence": ..., "probabilities": {...},
            "legend": {...}} | {"type": "noul", "noul": 0.93}}, "usage": {...}}

The shape above is what the community SDK (`jevper`) and the unofficial guide
document; confirm against console.typesafe.ai before Phase 5 (ship). Every
unexpected shape raises ReflexError and the loop holds: a parsing surprise can
never become an order.

Backends: `jev` (production), `claude` (Claude Haiku 5.5 structured-output
fallback), `rules` (deterministic baseline for backtests and tests; never used
live unless REFLEX_BACKEND=rules), `recorded` (replay of journaled decisions).
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Mapping, Optional, Protocol

from trading_agent.config import TraderConfig
from trading_agent.schema import (
    JEV_QUESTIONS,
    Direction,
    JevDecision,
    MarketSnapshot,
    Regime,
    RiskState,
)

log = logging.getLogger(__name__)


class ReflexError(RuntimeError):
    """The reflex layer could not produce a valid decision. The loop holds."""


class ReflexClient(Protocol):
    name: str

    def judge(self, snapshot: MarketSnapshot) -> JevDecision: ...


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------
def build_jev_request(snapshot: MarketSnapshot, model: str, questions: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    return {
        "model": model,
        "state": snapshot.compact_json(),
        "questions": dict(questions or JEV_QUESTIONS),
    }


def _answer(answers: Mapping[str, Any], name: str, expected_type: str) -> Mapping[str, Any]:
    a = answers.get(name)
    if not isinstance(a, Mapping):
        raise ReflexError(f"missing answer for {name}")
    if a.get("type", expected_type) != expected_type:
        raise ReflexError(f"answer {name} has type {a.get('type')!r}, expected {expected_type!r}")
    return a


def _probs(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[str(k)] = min(1.0, max(0.0, f))
    return out


def _confidence(a: Mapping[str, Any], chosen: str, probs: dict[str, float]) -> float:
    if chosen in probs:
        return probs[chosen]
    try:
        c = float(a.get("confidence"))
    except (TypeError, ValueError):
        raise ReflexError("answer has neither probabilities nor confidence") from None
    if not math.isfinite(c) or not (0.0 <= c <= 1.0):
        raise ReflexError(f"confidence out of range: {c}")
    return c


def parse_jev_response(data: Mapping[str, Any], *, latency_ms: float = 0.0) -> JevDecision:
    answers = data.get("answers") if isinstance(data, Mapping) else None
    if not isinstance(answers, Mapping):
        raise ReflexError("response has no `answers` object")
    try:
        a = _answer(answers, "regime", "choice")
        regime = Regime(str(a.get("choice")))
        regime_probs = _probs(a.get("probabilities"))

        a = _answer(answers, "direction", "choice")
        direction = Direction(str(a.get("choice")))
        direction_probs = _probs(a.get("probabilities"))
        confidence = _confidence(a, direction.value, direction_probs)

        a = _answer(answers, "toxic_flow", "noul")
        noul = float(a.get("noul"))
        if not math.isfinite(noul):
            raise ReflexError("noul is not finite")
        toxic = noul >= 0.5

        a = _answer(answers, "setup_quality", "score")
        score = float(a.get("score"))
        if not math.isfinite(score):
            raise ReflexError("score is not finite")
        quality = int(min(3, max(0, round(score))))
        quality_probs = _probs(a.get("probabilities"))

        a = _answer(answers, "risk_state", "choice")
        risk_state = RiskState(str(a.get("choice")))
        risk_probs = _probs(a.get("probabilities"))
    except (TypeError, ValueError) as exc:
        raise ReflexError(f"invalid answer value: {exc}") from exc

    return JevDecision(
        regime=regime,
        direction=direction,
        toxic_flow=toxic,
        setup_quality=quality,
        risk_state=risk_state,
        confidence=confidence,
        probabilities={
            "regime": regime_probs,
            "direction": direction_probs,
            "toxic_flow": {"true": min(1.0, max(0.0, noul)), "false": 1.0 - min(1.0, max(0.0, noul))},
            "setup_quality": quality_probs,
            "risk_state": risk_probs,
        },
        source="jev",
        model=str(data.get("model", "")),
        latency_ms=latency_ms,
    )


class JevClient:
    name = "jev"

    def __init__(self, api_url: str, api_key: str, model: str = "jev-latest", timeout_ms: int = 1500,
                 questions: Optional[Mapping[str, Any]] = None, session: Any = None):
        if not api_key:
            raise ValueError("JEV_API_KEY is required for the jev backend")
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_ms / 1000.0
        self.questions = dict(questions or JEV_QUESTIONS)
        if session is None:
            import requests  # lazy: keeps tests free of network libraries

            session = requests.Session()
        self.session = session
        self.last_latency_ms = 0.0

    def judge(self, snapshot: MarketSnapshot) -> JevDecision:
        body = build_jev_request(snapshot, self.model, self.questions)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        try:
            resp = self.session.post(self.api_url, json=body, headers=headers, timeout=self.timeout_s)
        except Exception as exc:  # requests.RequestException and friends
            raise ReflexError(f"jev request failed: {exc.__class__.__name__}: {exc}") from exc
        latency = (time.perf_counter() - t0) * 1000.0
        self.last_latency_ms = latency
        status = getattr(resp, "status_code", 0)
        if status != 200:
            raise ReflexError(f"jev http {status}: {str(getattr(resp, 'text', ''))[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise ReflexError("jev returned non-JSON") from exc
        return parse_jev_response(data, latency_ms=latency)


# ---------------------------------------------------------------------------
# Claude Haiku 5.5 fallback (structured output, low effort)
# ---------------------------------------------------------------------------
REFLEX_SYSTEM = (
    "You are a fast market judge. You receive one compact numeric snapshot of a crypto market and "
    "answer five typed questions. Output only the JSON object. `confidence` is your calibrated "
    "probability that `direction` is right over the next 12 candles after costs; use 0.5 when you "
    "have no edge. Never fire on thin evidence."
)


class ClaudeReflexClient:
    name = "claude"

    def __init__(self, cfg: TraderConfig):
        import anthropic

        self.cfg = cfg
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key, max_retries=1, timeout=20.0)
        self.model = cfg.reflex_claude_model

    def judge(self, snapshot: MarketSnapshot) -> JevDecision:
        prompt = (
            f"<questions>{json.dumps(JEV_QUESTIONS, separators=(',', ':'))}</questions>\n"
            f"<state>{snapshot.compact_json()}</state>"
        )
        t0 = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=REFLEX_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": JevDecision.json_schema()}},
            )
        except self.anthropic.APIStatusError as exc:
            raise ReflexError(f"claude reflex http {exc.status_code}: {exc.message}") from exc
        except self.anthropic.APIConnectionError as exc:
            raise ReflexError(f"claude reflex connection error: {exc}") from exc
        latency = (time.perf_counter() - t0) * 1000.0
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ReflexError(f"claude reflex stopped: {response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
            return JevDecision(**data, source="claude", model=response.model, latency_ms=latency)
        except (ValueError, TypeError) as exc:
            raise ReflexError(f"claude reflex invalid output: {exc}") from exc


# ---------------------------------------------------------------------------
# Deterministic rules baseline (tests, backtests): no network, no learning
# ---------------------------------------------------------------------------
class RuleReflexClient:
    """A transparent baseline. It is NOT calibrated; the policy still gates it."""

    name = "rules"

    def judge(self, snapshot: MarketSnapshot) -> JevDecision:
        s = snapshot
        atr = max(s.atr_pct, 1e-6)
        trend_strength = abs(s.ret_12_pct) / atr          # how many ATRs the last 12 candles moved
        if s.spread_bps > 30 or (s.realized_vol_pct > 200 and abs(s.ret_1_bps) > 150):
            regime = Regime.crisis
        elif s.realized_vol_pct > 90:
            regime = Regime.high_vol
        elif trend_strength > 2.0:
            regime = Regime.trending
        else:
            regime = Regime.mean_reverting

        trend_dir = Direction.long if s.ret_12_pct > 0 else Direction.short if s.ret_12_pct < 0 else Direction.neutral
        if regime == Regime.trending:
            direction = trend_dir
        elif regime == Regime.mean_reverting and s.range_pos_48 <= 0.1:
            direction = Direction.long
        elif regime == Regime.mean_reverting and s.range_pos_48 >= 0.9:
            direction = Direction.short
        else:
            direction = Direction.neutral
        if s.setup_direction != Direction.neutral and direction != s.setup_direction:
            direction = Direction.neutral

        toxic = abs(s.flow) > 0.6
        quality = 0
        if direction != Direction.neutral:
            quality += 1
            if (direction == Direction.long and s.imbalance > 0.1) or (direction == Direction.short and s.imbalance < -0.1):
                quality += 1
            if regime == Regime.trending and trend_strength > 3.0:
                quality += 1
            if (direction == Direction.long and s.range_pos_48 < 0.85) or (direction == Direction.short and s.range_pos_48 > 0.15):
                quality = min(3, quality)
            else:
                quality = max(0, quality - 1)
        if s.setup_invalidated:
            quality = 0
        confidence = 0.5 if direction == Direction.neutral else min(0.95, 0.55 + 0.08 * quality + 0.05 * min(trend_strength, 3.0))
        return JevDecision(
            regime=regime, direction=direction, toxic_flow=toxic, setup_quality=quality,
            risk_state=s.risk_state, confidence=round(confidence, 4), source="rules", model="rules-v1",
        )


class RecordedReflexClient:
    """Replays decisions journaled from a real Jev session (keyed by symbol and ts)."""

    name = "recorded"

    def __init__(self, decisions: Mapping[tuple[str, int], JevDecision]):
        self.decisions = dict(decisions)

    def judge(self, snapshot: MarketSnapshot) -> JevDecision:
        d = self.decisions.get((snapshot.symbol, snapshot.ts))
        if d is None:
            raise ReflexError(f"no recorded decision for {snapshot.symbol} at {snapshot.ts}")
        return d


class ChainedReflex:
    """Try each backend in order; a ReflexError falls through to the next."""

    def __init__(self, clients: list[Any]):
        if not clients:
            raise ValueError("at least one reflex client is required")
        self.clients = clients
        self.name = "+".join(c.name for c in clients)

    def judge(self, snapshot: MarketSnapshot) -> JevDecision:
        last: Optional[Exception] = None
        for client in self.clients:
            try:
                return client.judge(snapshot)
            except ReflexError as exc:
                log.warning("reflex backend %s failed: %s", client.name, exc)
                last = exc
        raise ReflexError(f"all reflex backends failed: {last}")


def create_reflex(cfg: TraderConfig) -> Any:
    backend = cfg.reflex_backend
    if backend == "jev":
        return JevClient(cfg.jev_api_url, cfg.jev_api_key, cfg.jev_model, cfg.jev_timeout_ms)
    if backend == "claude":
        return ClaudeReflexClient(cfg)
    if backend == "rules":
        if cfg.mode == "live":
            log.warning("REFLEX_BACKEND=rules in live mode: the rules baseline is not calibrated")
        return RuleReflexClient()
    clients: list[Any] = []
    if cfg.jev_configured:
        clients.append(JevClient(cfg.jev_api_url, cfg.jev_api_key, cfg.jev_model, cfg.jev_timeout_ms))
    if cfg.brain_configured:
        try:
            clients.append(ClaudeReflexClient(cfg))
        except ImportError:
            log.warning("anthropic package missing; no Claude reflex fallback")
    if not clients:
        if cfg.mode in ("live", "paper"):
            log.warning("no JEV_API_KEY or ANTHROPIC_API_KEY: using the uncalibrated rules baseline")
        clients.append(RuleReflexClient())
    return clients[0] if len(clients) == 1 else ChainedReflex(clients)
