"""Watchdog: if the live loop's heartbeat stops, pull the kill switch and,
when asked and allowed, close every venue position directly.

Run it from cron or a systemd timer every couple of minutes:

    python -m trading_agent.watchdog            # pulls the kill switch on a stale heartbeat
    python -m trading_agent.watchdog --flatten  # ...and closes positions on the venue (needs TRADER_LIVE=true)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

from trading_agent.config import TraderConfig
from trading_agent.risk import KillSwitch

log = logging.getLogger(__name__)


def heartbeat_age(path: str, now: Optional[float] = None) -> Optional[float]:
    """Seconds since the loop last wrote the heartbeat, or None when unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.read().split()[0]
        written = float(first)
    except (OSError, IndexError, ValueError):
        return None
    return (now if now is not None else time.time()) - written


def check(heartbeat_path: str, kill_switch_path: str, max_age_s: float, now: Optional[float] = None) -> tuple[bool, str]:
    age = heartbeat_age(heartbeat_path, now)
    if age is None:
        reason = f"watchdog: heartbeat missing or unreadable at {heartbeat_path}"
    elif age > max_age_s:
        reason = f"watchdog: heartbeat {age:.0f}s old, limit {max_age_s:.0f}s"
    else:
        return True, f"heartbeat {age:.0f}s old (limit {max_age_s:.0f}s)"
    ks = KillSwitch(kill_switch_path)
    ks.arm()
    ks.trigger(reason)
    return False, reason


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Trading agent watchdog")
    parser.add_argument("--max-age", type=float, default=None, help="seconds; default two candles + 120")
    parser.add_argument("--flatten", action="store_true", help="also close venue positions when the heartbeat is stale")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TraderConfig.from_env()
    max_age = args.max_age or (2 * cfg.candle_seconds + 120)
    ok, message = check(cfg.heartbeat_path, cfg.kill_switch_path, max_age)
    print(message)
    if ok:
        return 0
    if args.flatten:
        if not (cfg.live_confirmed and cfg.exchange_api_key):
            print("flatten skipped: TRADER_LIVE=true and exchange keys are required", file=sys.stderr)
            return 1
        from trading_agent.exchange import CcxtExchange

        venue = CcxtExchange(cfg.exchange_id, cfg.exchange_api_key, cfg.exchange_api_secret, market_type=cfg.exchange_market,
                             sandbox=cfg.exchange_sandbox, live_confirmed=True, starting_equity=cfg.starting_equity,
                             taker_fee_bps=cfg.taker_fee_bps, leverage=cfg.exchange_leverage)
        for fill in venue.flatten_all(cfg.symbols):
            print(f"flattened {fill.symbol} {fill.side.value} {fill.qty} @ {fill.price}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
