"""Nightly review: read every fill and every miss, measure calibration with the
Brier score, and ask the brain for a proposal. Nothing ships without the operator.

The proposal is bounded in code (apply_proposal): it can tighten or loosen
policy thresholds only inside hard bounds and can never touch RiskLimits.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Optional

from trading_agent.brain import ReviewProposal
from trading_agent.config import PolicyParams, TraderConfig
from trading_agent.journal import Journal
from trading_agent.schema import JEV_QUESTIONS

PARAM_BOUNDS = {
    "min_confidence": (0.70, 0.95),
    "min_setup_quality": (1, 3),
    "escalate_below_confidence": (0.30, 0.80),
    "kelly_fraction": (0.01, 0.25),
    "default_reward_risk": (0.5, 5.0),
    "horizon_candles": (1, 288),
}


# -- metrics ------------------------------------------------------------------
def brier_score(pairs: list[tuple[float, int]]) -> Optional[float]:
    """Mean squared error between forecast probability p and outcome y in {0, 1}."""
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def calibration_table(pairs: list[tuple[float, int]], bins: int = 10) -> list[dict[str, Any]]:
    table = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bucket = [(p, y) for p, y in pairs if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if not bucket:
            continue
        table.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "n": len(bucket),
            "mean_confidence": round(sum(p for p, _ in bucket) / len(bucket), 4),
            "hit_rate": round(sum(y for _, y in bucket) / len(bucket), 4),
        })
    return table


def _group_hit_rates(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(int(r["win"]))
    return {k: {"n": len(v), "hit_rate": round(sum(v) / len(v), 4)} for k, v in sorted(groups.items())}


def build_report(journal: Journal, since_ts: int, until_ts: int, cfg: TraderConfig) -> dict[str, Any]:
    decisions = journal.decisions_between(since_ts, until_ts)
    resolved = [d for d in decisions if d["resolved"] and d["win"] is not None]
    pairs = [(float(d["confidence"]), int(d["win"])) for d in resolved]
    fills = journal.fills_between(since_ts, until_ts)
    orders = journal.orders_between(since_ts, until_ts)
    equity = journal.equity_between(since_ts, until_ts)
    misses = journal.misses_between(since_ts, until_ts)
    escalations = journal.escalations_between(since_ts, until_ts)
    latencies = [float(d["latency_ms"]) for d in decisions if d["latency_ms"]]
    fired = [d for d in resolved if d["outcome"] in ("order",)]
    outcomes = {}
    for d in decisions:
        outcomes[d["outcome"]] = outcomes.get(d["outcome"], 0) + 1
    eq_values = [e["equity"] for e in equity]
    peak = 0.0
    max_dd = 0.0
    for v in eq_values:
        peak = max(peak, v)
        if peak:
            max_dd = max(max_dd, 1 - v / peak)
    return {
        "window": {"since_ts": since_ts, "until_ts": until_ts},
        "decisions": {"n": len(decisions), "by_outcome": outcomes,
                      "by_source": _count(decisions, "source"), "by_regime": _count(decisions, "regime")},
        "calibration": {
            "n": len(pairs), "brier": brier_score(pairs), "table": calibration_table(pairs),
            "hit_rate_by_quality": _group_hit_rates(resolved, "setup_quality"),
            "hit_rate_by_regime": _group_hit_rates(resolved, "regime"),
            "hit_rate_by_source": _group_hit_rates(resolved, "source"),
            "fired": {"n": len(fired), "hit_rate": round(sum(int(d["win"]) for d in fired) / len(fired), 4) if fired else None,
                      "brier": brier_score([(float(d["confidence"]), int(d["win"])) for d in fired])},
        },
        "execution": {
            "orders": len(orders), "blocked": sum(1 for o in orders if not o["allowed"]),
            "clipped": sum(1 for o in orders if o["allowed"] and o["qty_allowed"] < o["qty_requested"] - 1e-12),
            "fills": len(fills), "fees": round(sum(f["fee"] for f in fills), 6),
            "avg_slippage_bps": round(statistics.fmean(f["slippage_bps"] for f in fills), 3) if fills else None,
            "block_reasons": _count(orders, "risk_reason", only=lambda o: not o["allowed"]),
        },
        "pnl": {
            "equity_start": eq_values[0] if eq_values else None, "equity_end": eq_values[-1] if eq_values else None,
            "return_pct": round((eq_values[-1] / eq_values[0] - 1) * 100, 4) if len(eq_values) > 1 and eq_values[0] else None,
            "max_drawdown_pct": round(max_dd * 100, 4),
        },
        "misses": misses[:200],
        "escalations": escalations[:100],
        "reflex_latency_ms": {"n": len(latencies), "p50": _pct(latencies, 50), "p95": _pct(latencies, 95), "max": max(latencies) if latencies else None},
        "params_in_force": dataclasses.asdict(cfg.policy),
        "risk_limits": dataclasses.asdict(cfg.risk),
        "questions_in_force": JEV_QUESTIONS,
    }


def _count(rows: list[dict[str, Any]], key: str, only=None) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        if only and not only(r):
            continue
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return dict(sorted(out.items()))


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return round(s[idx], 2)


# -- proposals ------------------------------------------------------------------
def apply_proposal(current: PolicyParams, proposed: dict[str, Any]) -> tuple[PolicyParams, list[str]]:
    """Clamp a proposal into the hard bounds and return new params plus a change log."""
    changes: list[str] = []
    values = dataclasses.asdict(current)
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key not in proposed:
            continue
        try:
            raw = proposed[key]
            new = int(round(float(raw))) if isinstance(lo, int) else float(raw)
        except (TypeError, ValueError):
            continue
        clamped = min(hi, max(lo, new))
        if clamped != values[key]:
            changes.append(f"{key}: {values[key]} -> {clamped}" + (" (clamped)" if clamped != new else ""))
            values[key] = clamped
    values["kelly_cap"] = 0.25
    if values["escalate_below_confidence"] > values["min_confidence"]:
        values["escalate_below_confidence"] = values["min_confidence"]
        changes.append("escalate_below_confidence capped at min_confidence")
    params = PolicyParams(**values)
    params.validate()
    return params, changes


def proposal_env_lines(params: PolicyParams) -> list[str]:
    return [
        f"POLICY_MIN_CONFIDENCE={params.min_confidence}",
        f"POLICY_MIN_SETUP_QUALITY={params.min_setup_quality}",
        f"POLICY_ESCALATE_BELOW={params.escalate_below_confidence}",
        f"POLICY_KELLY_FRACTION={params.kelly_fraction}",
        f"POLICY_DEFAULT_REWARD_RISK={params.default_reward_risk}",
        f"POLICY_HORIZON_CANDLES={params.horizon_candles}",
    ]


def render_markdown(date: str, report: dict[str, Any], proposal: Optional[ReviewProposal], changes: list[str], env_lines: list[str]) -> str:
    cal = report["calibration"]
    lines = [f"# Nightly review {date}", "",
             f"Decisions: {report['decisions']['n']} | resolved (long/short): {cal['n']} | Brier: {cal['brier']}",
             f"Fired: {cal['fired']['n']} (hit rate {cal['fired']['hit_rate']}, Brier {cal['fired']['brier']})",
             f"Fills: {report['execution']['fills']} | blocked: {report['execution']['blocked']} | clipped: {report['execution']['clipped']} | fees: {report['execution']['fees']}",
             f"Return: {report['pnl']['return_pct']} % | max drawdown: {report['pnl']['max_drawdown_pct']} %",
             f"Misses (gated-out winners): {len(report['misses'])} | escalations: {len(report['escalations'])}",
             f"Reflex latency ms p50/p95/max: {report['reflex_latency_ms']['p50']}/{report['reflex_latency_ms']['p95']}/{report['reflex_latency_ms']['max']}",
             "", "## Calibration", "", "| bin | n | mean confidence | hit rate |", "|---|---|---|---|"]
    for row in cal["table"]:
        lines.append(f"| {row['bin']} | {row['n']} | {row['mean_confidence']} | {row['hit_rate']} |")
    lines += ["", "## Hit rate by setup_quality", "", "```", json.dumps(cal["hit_rate_by_quality"]), "```",
              "", "## Hit rate by regime", "", "```", json.dumps(cal["hit_rate_by_regime"]), "```",
              "", "## Block reasons", "", "```", json.dumps(report["execution"]["block_reasons"]), "```"]
    if proposal:
        lines += ["", "## Brain proposal (requires operator approval)", "", proposal.summary, "",
                  "**Calibration assessment:** " + proposal.calibration_assessment, "",
                  "**Parameter changes (clamped to bounds):**", ""]
        lines += [f"- {c}" for c in changes] or ["- none"]
        lines += ["", "**Question rewrites:**", ""]
        lines += [f"- `{q.get('name')}`: {q.get('instructions')} ({q.get('why')})" for q in proposal.question_rewrites] or ["- none"]
        lines += ["", "**Risk flags:**", ""] + ([f"- {r}" for r in proposal.risk_flags] or ["- none"])
        lines += ["", "## WHAT COULD I BE WRONG ABOUT?", ""] + ([f"- {w}" for w in proposal.what_could_i_be_wrong_about] or ["- (none stated)"])
        lines += ["", "To apply the parameter proposal for the next open, set:", "", "```"] + env_lines + ["```"]
    else:
        lines += ["", "## Brain proposal", "", "No brain configured or the brain was unavailable: parameters unchanged."]
    return "\n".join(lines) + "\n"


class NightlyReview:
    def __init__(self, cfg: TraderConfig, journal: Journal, brain: Any):
        self.cfg = cfg
        self.journal = journal
        self.brain = brain

    def run(self, date: Optional[str] = None, since_ts: Optional[int] = None, until_ts: Optional[int] = None,
            write: bool = True) -> tuple[dict[str, Any], Optional[ReviewProposal], str]:
        now = int(time.time() * 1000)
        until_ts = until_ts or now
        since_ts = since_ts if since_ts is not None else until_ts - 86_400_000
        date = date or datetime.fromtimestamp(until_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        report = build_report(self.journal, since_ts, until_ts, self.cfg)
        proposal = self.brain.nightly_review(report)
        changes: list[str] = []
        env_lines: list[str] = []
        if proposal:
            params, changes = apply_proposal(self.cfg.policy, proposal.proposed_params)
            env_lines = proposal_env_lines(params)
        markdown = render_markdown(date, report, proposal, changes, env_lines)
        self.journal.save_review(date, now, report, proposal.to_dict() if proposal else None)
        if write:
            os.makedirs(self.cfg.reviews_dir, exist_ok=True)
            with open(os.path.join(self.cfg.reviews_dir, f"{date}.md"), "w", encoding="utf-8") as fh:
                fh.write(markdown)
            with open(os.path.join(self.cfg.reviews_dir, f"{date}.json"), "w", encoding="utf-8") as fh:
                json.dump({"report": report, "proposal": proposal.to_dict() if proposal else None}, fh, indent=1, default=str)
        return report, proposal, markdown
