"""The BRAIN: Claude Opus 5.5, slow and deep. Used only for escalations and the
overnight review. It never places orders and cannot touch hard limits.

Without ANTHROPIC_API_KEY the NullBrain answers every escalation with a
conservative pause, so the system degrades safely.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from trading_agent.config import TraderConfig
from trading_agent.schema import JevDecision, MarketSnapshot, Regime

log = logging.getLogger(__name__)

BRAIN_SYSTEM = (
    "You are the BRAIN of an autonomous crypto trading system. A fast typed model (Jev) judges one "
    "compact market snapshot per candle; deterministic code owns every threshold, position size and "
    "the kill switch. You are consulted only when confidence drops or the regime flips to crisis, and "
    "once per night to review the session. You never place orders and you cannot change hard risk "
    "limits (max drawdown, max position, max daily loss). Be conservative: when in doubt, pause new "
    "entries. Never invent metrics; reason only from the data you are given and label estimates."
)

ESCALATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["continue", "pause", "flatten"]},
        "regime": {"type": "string", "enum": [r.value for r in Regime]},
        "pause_candles": {"type": "integer", "minimum": 0, "maximum": 288},
        "rationale": {"type": "string"},
    },
    "required": ["action", "regime", "pause_candles", "rationale"],
    "additionalProperties": False,
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "calibration_assessment": {"type": "string"},
        "proposed_params": {
            "type": "object",
            "properties": {
                "min_confidence": {"type": "number", "minimum": 0.5, "maximum": 0.99},
                "min_setup_quality": {"type": "integer", "minimum": 0, "maximum": 3},
                "escalate_below_confidence": {"type": "number", "minimum": 0.0, "maximum": 0.99},
                "kelly_fraction": {"type": "number", "minimum": 0.01, "maximum": 0.25},
                "default_reward_risk": {"type": "number", "minimum": 0.5, "maximum": 5.0},
                "horizon_candles": {"type": "integer", "minimum": 1, "maximum": 288},
            },
            "required": ["min_confidence", "min_setup_quality", "escalate_below_confidence",
                         "kelly_fraction", "default_reward_risk", "horizon_candles"],
            "additionalProperties": False,
        },
        "question_rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "instructions": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["name", "instructions", "why"],
                "additionalProperties": False,
            },
        },
        "what_could_i_be_wrong_about": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "calibration_assessment", "proposed_params", "question_rewrites",
                 "what_could_i_be_wrong_about", "risk_flags"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class BrainVerdict:
    action: str                 # continue | pause | flatten
    regime: Regime
    pause_candles: int
    rationale: str
    source: str = "brain"

    @classmethod
    def conservative(cls, regime: Regime, why: str) -> "BrainVerdict":
        return cls(action="pause", regime=regime, pause_candles=12, rationale=why, source="fallback")


@dataclass
class ReviewProposal:
    summary: str
    calibration_assessment: str
    proposed_params: dict[str, Any]
    question_rewrites: list[dict[str, str]] = field(default_factory=list)
    what_could_i_be_wrong_about: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    source: str = "brain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "calibration_assessment": self.calibration_assessment,
            "proposed_params": self.proposed_params,
            "question_rewrites": self.question_rewrites,
            "what_could_i_be_wrong_about": self.what_could_i_be_wrong_about,
            "risk_flags": self.risk_flags,
            "source": self.source,
        }


class NullBrain:
    """Used when no API key is configured: always conservative."""

    name = "null"

    def deep_reread(self, snapshot: MarketSnapshot, decision: JevDecision, recent: list[dict[str, Any]], reason: str) -> BrainVerdict:
        return BrainVerdict.conservative(decision.regime, f"no brain configured; pausing after escalation: {reason}")

    def nightly_review(self, report: dict[str, Any]) -> Optional[ReviewProposal]:
        return None


class Brain:
    name = "claude"

    def __init__(self, cfg: TraderConfig):
        import anthropic  # imported lazily so the rest of the system runs without the SDK

        self.cfg = cfg
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key, max_retries=2, timeout=cfg.brain_timeout_s)

    def _json(self, prompt: str, schema: dict[str, Any], max_tokens: int = 8000) -> Optional[dict[str, Any]]:
        try:
            response = self.client.beta.messages.create(
                model=self.cfg.brain_model,
                max_tokens=max_tokens,
                system=BRAIN_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.cfg.brain_effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self.anthropic.RateLimitError as exc:
            log.warning("brain rate limited: %s", exc)
            return None
        except self.anthropic.APIStatusError as exc:
            log.warning("brain API error %s: %s", exc.status_code, exc.message)
            return None
        except self.anthropic.APIConnectionError as exc:
            log.warning("brain connection error: %s", exc)
            return None
        if response.stop_reason in ("refusal", "max_tokens"):
            log.warning("brain stopped with %s", response.stop_reason)
            return None
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("brain returned non-JSON output")
            return None

    def deep_reread(self, snapshot: MarketSnapshot, decision: JevDecision, recent: list[dict[str, Any]], reason: str) -> BrainVerdict:
        prompt = (
            "ESCALATION. Re-read the market state deeply and decide whether the live loop should continue "
            "taking new entries, pause them for N candles, or flatten the open position. You cannot change "
            "limits or sizes.\n"
            f"<reason>{reason}</reason>\n"
            f"<snapshot>{snapshot.compact_json()}</snapshot>\n"
            f"<reflex_decision>{decision.model_dump_json()}</reflex_decision>\n"
            f"<recent_decisions>{json.dumps(recent[-24:], separators=(',', ':'))}</recent_decisions>"
        )
        data = self._json(prompt, ESCALATION_SCHEMA, max_tokens=4000)
        if not data:
            return BrainVerdict.conservative(decision.regime, f"brain unavailable; pausing after escalation: {reason}")
        try:
            return BrainVerdict(
                action=data["action"], regime=Regime(data["regime"]),
                pause_candles=int(data["pause_candles"]), rationale=str(data["rationale"])[:2000],
            )
        except (KeyError, ValueError) as exc:
            return BrainVerdict.conservative(decision.regime, f"brain output invalid ({exc}); pausing")

    def nightly_review(self, report: dict[str, Any]) -> Optional[ReviewProposal]:
        prompt = (
            "NIGHTLY REVIEW. Below is the full session report: every decision, every fill, every miss "
            "(gated-out decisions whose predicted move happened), calibration by confidence bin, Brier score, "
            "hit rates by setup_quality and regime, PnL and drawdown. Assess calibration honestly. Propose "
            "parameter values for the next session (within the schema bounds; keep a value unchanged unless "
            "the evidence is clear) and rewrites of the Jev question instructions that would improve "
            "calibration. Finish with what you could be wrong about. The operator approves before anything ships.\n"
            f"<report>{json.dumps(report, separators=(',', ':'), default=str)}</report>"
        )
        data = self._json(prompt, REVIEW_SCHEMA, max_tokens=12000)
        if not data:
            return None
        try:
            return ReviewProposal(
                summary=data["summary"],
                calibration_assessment=data["calibration_assessment"],
                proposed_params=dict(data["proposed_params"]),
                question_rewrites=list(data.get("question_rewrites", [])),
                what_could_i_be_wrong_about=list(data.get("what_could_i_be_wrong_about", [])),
                risk_flags=list(data.get("risk_flags", [])),
            )
        except (KeyError, TypeError) as exc:
            log.warning("brain review output invalid: %s", exc)
            return None


def create_brain(cfg: TraderConfig):
    if not cfg.brain_configured:
        return NullBrain()
    try:
        return Brain(cfg)
    except ImportError:
        log.warning("anthropic package missing (pip install anthropic); using NullBrain")
        return NullBrain()
