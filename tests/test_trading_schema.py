import json

import pytest
from pydantic import ValidationError

from tests.trading_helpers import make_engine
from trading_agent.reflex import ReflexError, build_jev_request, parse_jev_response
from trading_agent.schema import JEV_QUESTIONS, Direction, JevDecision, Regime, RiskState, worse_risk_state


def test_decision_validation_bounds():
    with pytest.raises(ValidationError):
        JevDecision(regime="trending", direction="long", toxic_flow=False, setup_quality=4, risk_state="safe", confidence=0.5)
    with pytest.raises(ValidationError):
        JevDecision(regime="trending", direction="long", toxic_flow=False, setup_quality=2, risk_state="safe", confidence=1.2)
    with pytest.raises(ValidationError):   # extra fields are forbidden: the schema is closed
        JevDecision(regime="trending", direction="long", toxic_flow=False, setup_quality=2, risk_state="safe", confidence=0.7, size=1)


def test_json_schema_is_strict():
    schema = JevDecision.json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"regime", "direction", "toxic_flow", "setup_quality", "risk_state", "confidence"}


def test_safe_hold_never_fires():
    d = JevDecision.safe_hold()
    assert d.direction == Direction.neutral and d.confidence == 0.0 and d.setup_quality == 0


def test_worse_risk_state():
    assert worse_risk_state(RiskState.safe, RiskState.near_limit) == RiskState.near_limit
    assert worse_risk_state(RiskState.reduce, RiskState.safe) == RiskState.reduce


def test_jev_request_shape():
    eng, _ = make_engine()
    snap = eng.snapshot(eng.last_candle().ts)
    body = build_jev_request(snap, "jev-latest")
    assert body["model"] == "jev-latest"
    assert set(body["questions"]) == {"regime", "direction", "toxic_flow", "setup_quality", "risk_state"}
    assert body["questions"]["toxic_flow"]["type"] == "noul"
    assert body["questions"]["setup_quality"]["type"] == "score"
    assert isinstance(body["questions"]["regime"]["criteria"], dict)
    assert json.loads(body["state"])["sym"] == "BTC/USDT"


def test_snapshot_under_400_tokens_and_deterministic():
    eng1, _ = make_engine()
    eng2, _ = make_engine()
    ts = eng1.last_candle().ts
    s1, s2 = eng1.snapshot(ts), eng2.snapshot(ts)
    assert s1.compact_json() == s2.compact_json()
    assert s1.approx_tokens() < 400


JEV_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "regime": {"type": "choice", "choice": "trending", "confidence": 0.7,
                   "probabilities": {"trending": 0.7, "mean_reverting": 0.2, "high_vol": 0.08, "crisis": 0.02}},
        "direction": {"type": "choice", "choice": "long", "confidence": 0.84,
                      "probabilities": {"long": 0.84, "short": 0.06, "neutral": 0.10}},
        "toxic_flow": {"type": "noul", "noul": 0.23},
        "setup_quality": {"type": "score", "score": 2.4, "confidence": 0.6, "legend": {"0": "none", "1": "weak", "2": "good", "3": "excellent"},
                          "probabilities": {"0": 0.05, "1": 0.1, "2": 0.6, "3": 0.25}},
        "risk_state": {"type": "choice", "choice": "safe", "confidence": 0.9, "probabilities": {"safe": 0.9, "near_limit": 0.08, "reduce": 0.02}},
    },
    "usage": {"input_tokens": 180, "output_tokens": 12},
}


def test_parse_jev_response_documented_shape():
    d = parse_jev_response(JEV_RESPONSE, latency_ms=81.0)
    assert d.regime == Regime.trending and d.direction == Direction.long
    assert d.confidence == 0.84                      # probability of the chosen direction
    assert d.toxic_flow is False and d.setup_quality == 2 and d.risk_state == RiskState.safe
    assert d.source == "jev" and d.model == "jev-1.13.0" and d.latency_ms == 81.0
    assert d.probabilities["toxic_flow"]["true"] == 0.23


def test_parse_jev_response_noul_threshold_and_score_clamp():
    data = json.loads(json.dumps(JEV_RESPONSE))
    data["answers"]["toxic_flow"]["noul"] = 0.5
    data["answers"]["setup_quality"]["score"] = 7
    d = parse_jev_response(data)
    assert d.toxic_flow is True and d.setup_quality == 3


def test_parse_jev_response_falls_back_to_confidence():
    data = json.loads(json.dumps(JEV_RESPONSE))
    del data["answers"]["direction"]["probabilities"]
    assert parse_jev_response(data).confidence == 0.84


@pytest.mark.parametrize("mutate", [
    lambda a: a.pop("direction"),
    lambda a: a["regime"].__setitem__("choice", "sideways"),
    lambda a: a["toxic_flow"].__setitem__("noul", "nan"),
    lambda a: a["setup_quality"].__setitem__("type", "choice"),
    lambda a: a["direction"].update({"confidence": 1.7, "probabilities": {}}),
])
def test_parse_jev_response_rejects_bad_shapes(mutate):
    data = json.loads(json.dumps(JEV_RESPONSE))
    mutate(data["answers"])
    with pytest.raises(ReflexError):
        parse_jev_response(data)


def test_parse_jev_response_without_answers():
    with pytest.raises(ReflexError):
        parse_jev_response({"model": "x"})
