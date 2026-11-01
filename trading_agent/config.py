"""Configuration for the trading agent.

Everything is read from environment variables (or a .env file in development).
Hard risk limits live here as *defaults*; the running system never lets a model
change them. See docs/trading/ARCHITECTURE.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name, "")
    if not value:
        return default
    return value.lower() in ("1", "true", "yes", "on", "si", "sí")


def _env_list(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in _env(name, default).split(",") if x.strip()]


MODES = ("paper", "live", "backtest", "review", "once")
REFLEX_BACKENDS = ("auto", "jev", "claude", "rules")


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits. Frozen: the model layer receives a copy and cannot mutate them."""

    max_drawdown: float = 0.15          # fraction below peak equity -> kill switch fires
    max_daily_loss: float = 0.03        # fraction of day-start equity -> no new risk today
    max_position_frac: float = 0.20     # |position notional| / equity, per symbol
    max_gross_frac: float = 0.60        # sum |position notional| / equity, all symbols
    max_order_notional_frac: float = 0.10  # single order notional / equity
    max_orders_per_hour: int = 12
    near_limit_ratio: float = 0.75      # any usage >= this -> risk_state near_limit

    def validate(self) -> None:
        if not (0 < self.max_drawdown <= 0.5):
            raise ValueError("max_drawdown must be in (0, 0.5]")
        if not (0 < self.max_daily_loss <= self.max_drawdown):
            raise ValueError("max_daily_loss must be in (0, max_drawdown]")
        if not (0 < self.max_position_frac <= 1.0):
            raise ValueError("max_position_frac must be in (0, 1]")
        if not (self.max_position_frac <= self.max_gross_frac <= 3.0):
            raise ValueError("max_gross_frac must be >= max_position_frac and <= 3")
        if not (0 < self.max_order_notional_frac <= self.max_position_frac):
            raise ValueError("max_order_notional_frac must be in (0, max_position_frac]")
        if not (0.5 <= self.near_limit_ratio < 1.0):
            raise ValueError("near_limit_ratio must be in [0.5, 1)")


@dataclass(frozen=True)
class PolicyParams:
    """Deterministic gates and sizing. Only the operator changes these (review gate)."""

    min_setup_quality: int = 2          # fire only when setup_quality >= 2
    min_confidence: float = 0.80        # fire only when confidence > 0.80
    escalate_below_confidence: float = 0.60  # escalate to the brain below this
    kelly_fraction: float = 0.25        # fraction of full Kelly used for sizing
    kelly_cap: float = 0.25             # hard cap: never more than quarter Kelly
    default_reward_risk: float = 1.5    # b in Kelly when the setup does not say
    stop_atr_mult: float = 1.5          # stop distance = ATR * this
    horizon_candles: int = 12           # outcome horizon for calibration
    outcome_move_bps: float = 15.0      # a decision "wins" if the move beats this (about one round trip of fees + slippage)
    max_snapshot_tokens: int = 400

    def validate(self) -> None:
        if not (0 <= self.min_setup_quality <= 3):
            raise ValueError("min_setup_quality must be 0..3")
        if not (0.5 <= self.min_confidence < 1.0):
            raise ValueError("min_confidence must be in [0.5, 1)")
        if not (0.0 < self.escalate_below_confidence <= self.min_confidence):
            raise ValueError("escalate_below_confidence must be in (0, min_confidence]")
        if not (0 < self.kelly_fraction <= self.kelly_cap <= 0.25):
            raise ValueError("kelly_fraction must be in (0, kelly_cap] and kelly_cap <= 0.25")
        if self.default_reward_risk <= 0:
            raise ValueError("default_reward_risk must be > 0")
        if self.horizon_candles < 1:
            raise ValueError("horizon_candles must be >= 1")


@dataclass
class TraderConfig:
    mode: str = "paper"
    exchange_id: str = "binance"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    candle_seconds: int = 300
    book_depth: int = 10
    starting_equity: float = 10_000.0
    taker_fee_bps: float = 5.0
    slippage_bps: float = 2.0
    risk: RiskLimits = field(default_factory=RiskLimits)
    policy: PolicyParams = field(default_factory=PolicyParams)

    # Reflex (Jev, TypeSafe AI System One) -----------------------------------
    reflex_backend: str = "auto"         # auto | jev | claude | rules
    jev_api_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_api_key: str = ""
    jev_model: str = "jev-latest"
    jev_timeout_ms: int = 1500
    reflex_claude_model: str = "claude-haiku-5-5"

    # Brain (Claude Opus 5.5) ------------------------------------------------
    anthropic_api_key: str = ""
    brain_model: str = "claude-opus-5-5"
    brain_effort: str = "high"
    brain_timeout_s: float = 120.0

    # Exchange credentials (live only) --------------------------------------
    exchange_api_key: str = ""
    exchange_api_secret: str = ""
    exchange_sandbox: bool = True
    live_confirmed: bool = False          # TRADER_LIVE=true is required for real orders

    # Storage ---------------------------------------------------------------
    journal_path: str = "data/trader_journal.db"
    kill_switch_path: str = "data/KILL_SWITCH"
    heartbeat_path: str = "data/trader_heartbeat"
    universe_path: str = "docs/trading/research/finalists.json"
    reviews_dir: str = "docs/trading/reviews"
    timezone: str = "America/Santiago"

    @classmethod
    def from_env(cls) -> "TraderConfig":
        mode = _env("TRADER_MODE", "paper").lower()
        if mode not in MODES:
            mode = "paper"
        backend = _env("REFLEX_BACKEND", "auto").lower()
        if backend not in REFLEX_BACKENDS:
            backend = "auto"
        risk = RiskLimits(
            max_drawdown=_env_float("RISK_MAX_DRAWDOWN", 0.15),
            max_daily_loss=_env_float("RISK_MAX_DAILY_LOSS", 0.03),
            max_position_frac=_env_float("RISK_MAX_POSITION_FRAC", 0.20),
            max_gross_frac=_env_float("RISK_MAX_GROSS_FRAC", 0.60),
            max_order_notional_frac=_env_float("RISK_MAX_ORDER_FRAC", 0.10),
            max_orders_per_hour=_env_int("RISK_MAX_ORDERS_PER_HOUR", 12),
            near_limit_ratio=_env_float("RISK_NEAR_LIMIT_RATIO", 0.75),
        )
        policy = PolicyParams(
            min_setup_quality=_env_int("POLICY_MIN_SETUP_QUALITY", 2),
            min_confidence=_env_float("POLICY_MIN_CONFIDENCE", 0.80),
            escalate_below_confidence=_env_float("POLICY_ESCALATE_BELOW", 0.60),
            kelly_fraction=_env_float("POLICY_KELLY_FRACTION", 0.25),
            kelly_cap=0.25,
            default_reward_risk=_env_float("POLICY_DEFAULT_REWARD_RISK", 1.5),
            stop_atr_mult=_env_float("POLICY_STOP_ATR_MULT", 1.5),
            horizon_candles=_env_int("POLICY_HORIZON_CANDLES", 12),
            outcome_move_bps=_env_float("POLICY_OUTCOME_MOVE_BPS", 15.0),
        )
        risk.validate()
        policy.validate()
        return cls(
            mode=mode,
            exchange_id=_env("EXCHANGE_ID", "binance"),
            symbols=_env_list("TRADER_SYMBOLS", "BTC/USDT,ETH/USDT"),
            candle_seconds=_env_int("CANDLE_SECONDS", 300),
            book_depth=_env_int("BOOK_DEPTH", 10),
            starting_equity=_env_float("STARTING_EQUITY", 10_000.0),
            taker_fee_bps=_env_float("TAKER_FEE_BPS", 5.0),
            slippage_bps=_env_float("SLIPPAGE_BPS", 2.0),
            risk=risk,
            policy=policy,
            reflex_backend=backend,
            jev_api_url=_env("JEV_API_URL", "https://api.typesafe.ai/v1/systemone"),
            jev_api_key=_env("JEV_API_KEY"),
            jev_model=_env("JEV_MODEL", "jev-latest"),
            jev_timeout_ms=_env_int("JEV_TIMEOUT_MS", 1500),
            reflex_claude_model=_env("REFLEX_CLAUDE_MODEL", "claude-haiku-5-5"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            brain_model=_env("BRAIN_MODEL", "claude-opus-5-5"),
            brain_effort=_env("BRAIN_EFFORT", "high"),
            brain_timeout_s=_env_float("BRAIN_TIMEOUT_S", 120.0),
            exchange_api_key=_env("EXCHANGE_API_KEY"),
            exchange_api_secret=_env("EXCHANGE_API_SECRET"),
            exchange_sandbox=_env_bool("EXCHANGE_SANDBOX", True),
            live_confirmed=_env_bool("TRADER_LIVE", False),
            journal_path=_env("TRADER_JOURNAL", "data/trader_journal.db"),
            kill_switch_path=_env("KILL_SWITCH_PATH", "data/KILL_SWITCH"),
            heartbeat_path=_env("HEARTBEAT_PATH", "data/trader_heartbeat"),
            universe_path=_env("UNIVERSE_PATH", "docs/trading/research/finalists.json"),
            reviews_dir=_env("REVIEWS_DIR", "docs/trading/reviews"),
            timezone=_env("ZONA_HORARIA", "America/Santiago"),
        )

    @property
    def jev_configured(self) -> bool:
        return bool(self.jev_api_key)

    @property
    def brain_configured(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def candles_per_day(self) -> int:
        return max(1, 86_400 // self.candle_seconds)
