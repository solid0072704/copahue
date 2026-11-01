"""Compile the research finalists (docs/trading/research/finalists.json) into
SetupSpec objects the state engine shows to Jev.

Only setups with data_quality A or B are tradable by default; C-grade research
is shown as neutral (no thesis bias) so the system cannot fire on a thin thesis.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from trading_agent.schema import Direction, SetupSpec

log = logging.getLogger(__name__)

QUALITY_RANK = {"A": 3, "B": 2, "C": 1}


_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,12}/[A-Z0-9]{2,12}$")


def _pair(entry: dict[str, Any]) -> str:
    """The ccxt symbol to trade. `ccxt_symbol` wins; a clean `pair` is accepted; otherwise
    the research `pair` is a venue description and the default spot pair is used."""
    explicit = str(entry.get("ccxt_symbol") or "").upper().strip()
    if _SYMBOL_RE.match(explicit):
        return explicit
    pair = str(entry.get("pair") or "").upper().strip().replace("-", "/")
    if _SYMBOL_RE.match(pair):
        return pair
    symbol = str(entry.get("symbol", "")).upper().strip()
    return f"{symbol}/USDT" if symbol else pair


def _float(entry: dict[str, Any], key: str) -> Optional[float]:
    value = entry.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def setup_from_entry(entry: dict[str, Any], min_quality: str = "B") -> SetupSpec:
    pair = _pair(entry)
    quality = str(entry.get("data_quality", "C")).upper()
    direction = Direction.neutral
    if QUALITY_RANK.get(quality, 0) >= QUALITY_RANK.get(min_quality, 2):
        try:
            direction = Direction(str(entry.get("direction", "neutral")).lower())
        except ValueError:
            direction = Direction.neutral
    regimes = tuple(str(r) for r in entry.get("regime_fit", ["trending"]))
    return SetupSpec(
        symbol=pair,
        direction=direction,
        thesis=str(entry.get("thesis", ""))[:240],
        reward_risk=_float(entry, "reward_risk") or 1.5,
        regime_fit=regimes or ("trending",),
        invalidate_above=_float(entry, "invalidate_above"),
        invalidate_below=_float(entry, "invalidate_below"),
        unlock_risk_90d=str(entry.get("unlock_risk_90d", "not_sourced")),
        data_quality=quality,
    )


def load_universe(path: str, min_quality: str = "B") -> dict[str, SetupSpec]:
    if not path or not os.path.exists(path):
        log.warning("universe file %s not found; trading without setups (neutral)", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("setups", data) if isinstance(data, dict) else data
    universe: dict[str, SetupSpec] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        spec = setup_from_entry(entry, min_quality)
        universe[spec.symbol] = spec
    return universe


def setup_for(symbol: str, universe: dict[str, SetupSpec]) -> SetupSpec:
    return universe.get(symbol) or SetupSpec(symbol=symbol, direction=Direction.neutral)
