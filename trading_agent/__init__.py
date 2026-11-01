"""Autonomous crypto trading agent: two layers, code owns every threshold.

BRAIN  (slow, deep)  -> trading_agent.brain    Claude Opus 5.5: research, overnight review, escalations.
REFLEX (fast, typed) -> trading_agent.reflex   Jev (TypeSafe AI System One) judges one compact snapshot per candle.
CODE   (deterministic) -> state_engine, risk, policy, exchange, loop: thresholds, sizing, kill switch.

Nothing in the model layer can place an order: every intent passes trading_agent.risk.RiskLayer.
"""

__version__ = "0.1.0"
