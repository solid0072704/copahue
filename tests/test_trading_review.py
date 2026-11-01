import json

import pytest

from tests.trading_helpers import ScriptedReflex, decision, make_config
from trading_agent.brain import NullBrain, ReviewProposal
from trading_agent.config import PolicyParams
from trading_agent.exchange import PaperExchange
from trading_agent.feed import SyntheticFeed, SyntheticRegime
from trading_agent.journal import Journal
from trading_agent.loop import LiveLoop
from trading_agent.review import NightlyReview, apply_proposal, brier_score, build_report, calibration_table
from trading_agent.risk import KillSwitch
from trading_agent.universe import load_universe, setup_from_entry
from trading_agent.schema import Direction


def test_brier_and_calibration():
    assert brier_score([]) is None
    assert brier_score([(1.0, 1), (0.0, 1)]) == pytest.approx(0.5)
    assert brier_score([(0.8, 1), (0.8, 0)]) == pytest.approx((0.04 + 0.64) / 2)
    table = calibration_table([(0.85, 1), (0.82, 0), (0.15, 0), (1.0, 1)], bins=10)
    bins = {row["bin"]: row for row in table}
    assert bins["0.8-0.9"]["n"] == 2 and bins["0.8-0.9"]["hit_rate"] == 0.5
    assert bins["0.9-1.0"]["n"] == 1


def test_apply_proposal_is_bounded():
    params, changes = apply_proposal(PolicyParams(), {"kelly_fraction": 0.9, "min_confidence": 0.5, "min_setup_quality": 0,
                                                      "escalate_below_confidence": 0.95, "horizon_candles": 24})
    assert params.kelly_fraction == 0.25 and params.kelly_cap == 0.25
    assert params.min_confidence == 0.70 and params.min_setup_quality == 1
    assert params.escalate_below_confidence <= params.min_confidence
    assert params.horizon_candles == 24
    assert any("clamped" in c for c in changes)


def test_nightly_review_writes_files(tmp_path):
    cfg = make_config(reviews_dir=str(tmp_path / "reviews"))
    fd = SyntheticFeed(["BTC/USDT"], 300, 200, seed=2, regimes=[(200, SyntheticRegime(drift_per_candle=0.0008, vol_per_candle=0.003))])
    journal = Journal(":memory:")
    loop = LiveLoop(cfg, fd, PaperExchange(cfg.starting_equity), ScriptedReflex(decision(confidence=0.9)), NullBrain(), journal, kill_switch=KillSwitch(None))
    loop.run()
    lo, hi = journal.time_bounds()
    review = NightlyReview(cfg, journal, NullBrain())
    report, proposal, markdown = review.run(date="2026-11-01", since_ts=lo - 1, until_ts=hi)
    assert proposal is None and "Nightly review 2026-11-01" in markdown
    assert (tmp_path / "reviews" / "2026-11-01.md").exists()
    saved = json.loads((tmp_path / "reviews" / "2026-11-01.json").read_text())
    assert saved["report"]["decisions"]["n"] == report["decisions"]["n"]
    assert report["calibration"]["n"] > 0
    assert journal.latest_review()["date"] == "2026-11-01"


class ProposingBrain(NullBrain):
    def nightly_review(self, report):
        return ReviewProposal(summary="s", calibration_assessment="c",
                              proposed_params={"min_confidence": 0.85, "kelly_fraction": 0.5, "min_setup_quality": 2,
                                               "escalate_below_confidence": 0.6, "default_reward_risk": 1.5, "horizon_candles": 12},
                              question_rewrites=[{"name": "direction", "instructions": "x", "why": "y"}],
                              what_could_i_be_wrong_about=["calibration sample is tiny"], risk_flags=["none"])


def test_review_with_proposal_renders_env_lines(tmp_path):
    cfg = make_config(reviews_dir=str(tmp_path / "r"))
    journal = Journal(":memory:")
    review = NightlyReview(cfg, journal, ProposingBrain())
    report, proposal, markdown = review.run(date="2026-11-02", since_ts=0, until_ts=10**13)
    assert "POLICY_MIN_CONFIDENCE=0.85" in markdown and "POLICY_KELLY_FRACTION=0.25" in markdown
    assert "WHAT COULD I BE WRONG ABOUT" in markdown


def test_universe_quality_gate(tmp_path):
    good = {"symbol": "UNI", "pair": "UNI/USDT", "direction": "long", "data_quality": "A", "thesis": "fee burn", "regime_fit": ["mean_reverting"]}
    weak = {"symbol": "TIA", "direction": "short", "data_quality": "C"}
    assert setup_from_entry(good).direction == Direction.long
    assert setup_from_entry(weak).direction == Direction.neutral and setup_from_entry(weak).symbol == "TIA/USDT"
    messy = {"symbol": "HYPE", "pair": "HYPE-USD perp / HYPE-USDC spot", "direction": "short", "data_quality": "B"}
    assert setup_from_entry(messy).symbol == "HYPE/USDT"
    assert setup_from_entry({**messy, "ccxt_symbol": "HYPE/USDC"}).symbol == "HYPE/USDC"
    assert setup_from_entry({"symbol": "ETH", "pair": "ETH-USD", "direction": "long", "data_quality": "A"}).symbol == "ETH/USD"
    path = tmp_path / "finalists.json"
    path.write_text(json.dumps([good, weak]))
    uni = load_universe(str(path))
    assert set(uni) == {"UNI/USDT", "TIA/USDT"}
    assert load_universe(str(tmp_path / "missing.json")) == {}
