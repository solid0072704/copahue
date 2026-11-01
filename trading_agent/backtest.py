"""Replay harness: the same LiveLoop, a paper exchange, and either the rules
baseline or recorded Jev decisions. Produces the same journal the nightly
review reads, so backtest and live are compared with one code path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from trading_agent.brain import NullBrain
from trading_agent.config import TraderConfig
from trading_agent.exchange import PaperExchange
from trading_agent.journal import Journal
from trading_agent.loop import LiveLoop, LoopStats
from trading_agent.reflex import RuleReflexClient
from trading_agent.review import build_report
from trading_agent.risk import KillSwitch
from trading_agent.schema import SetupSpec


@dataclass
class BacktestReport:
    stats: LoopStats
    metrics: dict[str, Any]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"stats": self.stats.to_dict(), "metrics": self.metrics, "report": self.report}


def run_backtest(
    cfg: TraderConfig,
    feed: Any,
    *,
    reflex: Any = None,
    brain: Any = None,
    universe: Optional[dict[str, SetupSpec]] = None,
    journal: Optional[Journal] = None,
    warmup_candles: int = 50,
    max_ticks: Optional[int] = None,
) -> BacktestReport:
    exchange = PaperExchange(cfg.starting_equity, cfg.taker_fee_bps, cfg.slippage_bps)
    journal = journal or Journal(":memory:")
    kill_switch = KillSwitch(None)   # in-memory: no trigger file in a backtest
    loop = LiveLoop(cfg, feed, exchange, reflex or RuleReflexClient(), brain or NullBrain(), journal,
                    universe=universe, kill_switch=kill_switch, warmup_candles=warmup_candles)
    stats = loop.run(max_ticks=max_ticks)
    lo, hi = journal.time_bounds()
    report = build_report(journal, lo - 1, hi, cfg)
    metrics = {
        "starting_equity": cfg.starting_equity,
        "final_equity": exchange.equity(),
        "total_return_pct": (exchange.equity() / cfg.starting_equity - 1.0) * 100.0,
        "max_drawdown_pct": stats.max_drawdown * 100.0,
        "fills": stats.fills,
        "fees_paid": exchange.fees_paid,
        "brier": report["calibration"]["brier"],
        "n_resolved": report["calibration"]["n"],
        "misses": len(report["misses"]),
        "kill_switch_triggered": kill_switch.triggered,
        "kill_switch_reason": kill_switch.reason,
    }
    return BacktestReport(stats=stats, metrics=metrics, report=report)
