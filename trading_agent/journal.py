"""SQLite journal: every snapshot, decision, order verdict, fill, equity point,
escalation and review. The nightly review reads only from here.

Outcomes are resolved in code: a decision with direction long/short "wins" when
the mid moved in that direction by more than `outcome_move_bps` at the first
candle at or after its horizon. A "miss" is a winning decision the policy or the
risk layer did not turn into an order.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Optional

from trading_agent.schema import Direction, Fill, JevDecision, MarketSnapshot, OrderIntent

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY, ts INTEGER NOT NULL, symbol TEXT NOT NULL, mid REAL NOT NULL,
  snapshot TEXT NOT NULL, decision TEXT NOT NULL,
  regime TEXT, direction TEXT, confidence REAL, setup_quality INTEGER, toxic_flow INTEGER, risk_state TEXT,
  source TEXT, latency_ms REAL, outcome TEXT NOT NULL, reason TEXT, size_frac REAL,
  horizon_ts INTEGER NOT NULL, resolved INTEGER NOT NULL DEFAULT 0, resolved_ts INTEGER, move_bps REAL, win INTEGER
);
CREATE INDEX IF NOT EXISTS ix_decisions_sym_ts ON decisions(symbol, ts);
CREATE INDEX IF NOT EXISTS ix_decisions_unresolved ON decisions(resolved, horizon_ts);
CREATE TABLE IF NOT EXISTS orders (
  intent_id TEXT PRIMARY KEY, decision_id TEXT, ts INTEGER, symbol TEXT, side TEXT,
  qty_requested REAL, qty_allowed REAL, allowed INTEGER, reduce_only INTEGER, reason TEXT, risk_reason TEXT
);
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, fee REAL,
  intent_id TEXT, order_id TEXT, slippage_bps REAL
);
CREATE TABLE IF NOT EXISTS equity (ts INTEGER, symbol TEXT, equity REAL, drawdown REAL, daily_pnl_pct REAL, position_qty REAL);
CREATE INDEX IF NOT EXISTS ix_equity_ts ON equity(ts);
CREATE TABLE IF NOT EXISTS escalations (ts INTEGER, symbol TEXT, reason TEXT, action TEXT, regime TEXT, pause_candles INTEGER, rationale TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS reviews (date TEXT PRIMARY KEY, created_ts INTEGER, report TEXT, proposal TEXT, applied INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events (ts INTEGER, level TEXT, message TEXT);
"""


class Journal:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- writes --------------------------------------------------------------
    def record_decision(self, decision_id: str, snapshot: MarketSnapshot, decision: JevDecision,
                        outcome: str, reason: str, size_frac: float, horizon_ts: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions (id, ts, symbol, mid, snapshot, decision, regime, direction, confidence, "
            "setup_quality, toxic_flow, risk_state, source, latency_ms, outcome, reason, size_frac, horizon_ts) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (decision_id, snapshot.ts, snapshot.symbol, snapshot.mid, snapshot.compact_json(), decision.model_dump_json(),
             decision.regime.value, decision.direction.value, decision.confidence, decision.setup_quality,
             int(decision.toxic_flow), decision.risk_state.value, decision.source, decision.latency_ms,
             outcome, reason, size_frac, horizon_ts),
        )
        self.conn.commit()

    def record_order(self, intent: OrderIntent, ts: int, allowed: bool, qty_allowed: float, risk_reason: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO orders (intent_id, decision_id, ts, symbol, side, qty_requested, qty_allowed, allowed, "
            "reduce_only, reason, risk_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (intent.intent_id, intent.decision_id, ts, intent.symbol, intent.side.value, intent.qty, qty_allowed,
             int(allowed), int(intent.reduce_only), intent.reason, risk_reason),
        )
        self.conn.commit()

    def record_fill(self, fill: Fill) -> None:
        self.conn.execute(
            "INSERT INTO fills (ts, symbol, side, qty, price, fee, intent_id, order_id, slippage_bps) VALUES (?,?,?,?,?,?,?,?,?)",
            (fill.ts, fill.symbol, fill.side.value, fill.qty, fill.price, fill.fee, fill.intent_id, fill.order_id, fill.slippage_bps),
        )
        self.conn.commit()

    def record_equity(self, ts: int, symbol: str, equity: float, drawdown: float, daily_pnl_pct: float, position_qty: float) -> None:
        self.conn.execute("INSERT INTO equity VALUES (?,?,?,?,?,?)", (ts, symbol, equity, drawdown, daily_pnl_pct, position_qty))
        self.conn.commit()

    def record_escalation(self, ts: int, symbol: str, reason: str, action: str, regime: str, pause_candles: int,
                          rationale: str, source: str) -> None:
        self.conn.execute("INSERT INTO escalations VALUES (?,?,?,?,?,?,?,?)",
                          (ts, symbol, reason, action, regime, pause_candles, rationale, source))
        self.conn.commit()

    def record_event(self, ts: int, level: str, message: str) -> None:
        self.conn.execute("INSERT INTO events VALUES (?,?,?)", (ts, level, message[:2000]))
        self.conn.commit()

    def save_review(self, date: str, created_ts: int, report: dict[str, Any], proposal: Optional[dict[str, Any]]) -> None:
        self.conn.execute("INSERT OR REPLACE INTO reviews (date, created_ts, report, proposal, applied) VALUES (?,?,?,?,0)",
                          (date, created_ts, json.dumps(report, default=str), json.dumps(proposal, default=str) if proposal else None))
        self.conn.commit()

    # -- outcome resolution (code, not model) --------------------------------
    def resolve_outcomes(self, symbol: str, ts: int, price: float, move_bps_threshold: float = 0.0) -> int:
        rows = self.conn.execute(
            "SELECT id, mid, direction FROM decisions WHERE resolved = 0 AND symbol = ? AND horizon_ts <= ?", (symbol, ts)
        ).fetchall()
        for row in rows:
            move = (price / row["mid"] - 1.0) * 1e4
            if row["direction"] == Direction.long.value:
                win: Optional[int] = int(move > move_bps_threshold)
            elif row["direction"] == Direction.short.value:
                win = int(-move > move_bps_threshold)
            else:
                win = None
            self.conn.execute("UPDATE decisions SET resolved = 1, resolved_ts = ?, move_bps = ?, win = ? WHERE id = ?",
                              (ts, move, win, row["id"]))
        if rows:
            self.conn.commit()
        return len(rows)

    # -- reads ---------------------------------------------------------------
    def orders_last_hour(self, symbol: str, ts: int) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM orders WHERE symbol = ? AND allowed = 1 AND ts > ? AND ts <= ?",
                                (symbol, ts - 3_600_000, ts)).fetchone()
        return int(row["n"])

    def recent_decisions(self, symbol: str, n: int = 24) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT ts, regime, direction, confidence, setup_quality, outcome, win, move_bps FROM decisions "
            "WHERE symbol = ? ORDER BY ts DESC LIMIT ?", (symbol, n)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def decisions_between(self, since_ts: int, until_ts: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, ts, symbol, mid, regime, direction, confidence, setup_quality, toxic_flow, risk_state, source, "
            "latency_ms, outcome, reason, size_frac, resolved, move_bps, win FROM decisions WHERE ts > ? AND ts <= ? ORDER BY ts",
            (since_ts, until_ts)).fetchall()
        return [dict(r) for r in rows]

    def fills_between(self, since_ts: int, until_ts: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM fills WHERE ts > ? AND ts <= ? ORDER BY ts", (since_ts, until_ts)).fetchall()
        return [dict(r) for r in rows]

    def orders_between(self, since_ts: int, until_ts: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM orders WHERE ts > ? AND ts <= ? ORDER BY ts", (since_ts, until_ts)).fetchall()
        return [dict(r) for r in rows]

    def equity_between(self, since_ts: int, until_ts: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM equity WHERE ts > ? AND ts <= ? ORDER BY ts", (since_ts, until_ts)).fetchall()
        return [dict(r) for r in rows]

    def escalations_between(self, since_ts: int, until_ts: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM escalations WHERE ts > ? AND ts <= ? ORDER BY ts", (since_ts, until_ts)).fetchall()
        return [dict(r) for r in rows]

    def misses_between(self, since_ts: int, until_ts: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, ts, symbol, direction, confidence, setup_quality, regime, outcome, reason, move_bps FROM decisions "
            "WHERE ts > ? AND ts <= ? AND resolved = 1 AND win = 1 AND outcome IN ('hold', 'escalate') ORDER BY ts",
            (since_ts, until_ts)).fetchall()
        return [dict(r) for r in rows]

    def latest_review(self) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM reviews ORDER BY date DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def time_bounds(self) -> tuple[int, int]:
        row = self.conn.execute("SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM equity").fetchone()
        return (int(row["lo"] or 0), int(row["hi"] or 0))
