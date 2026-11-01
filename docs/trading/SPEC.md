# SPEC: 24/7 autonomous crypto trading agent

Status: Phase 1 artifact. Approval gate: the operator signs off on every line of
"Hard limits" and "Out of scope" before Phase 2 starts.

## Mission

Research the market, find asymmetric setups, and execute them automatically with
production discipline. The operator only approves the gates (this spec, the
architecture, the plan, the paper-trading review, the ship decision, and every
nightly parameter proposal).

## Two layers, never blurred

| Layer | Who | Cadence | Owns |
|---|---|---|---|
| BRAIN | Claude Opus 5.5 (`claude-opus-5-5`) | research, escalations, one review per night | theses, bear cases, invalidation conditions, review proposals |
| REFLEX | Jev, TypeSafe AI System One (`jev-latest`) | one typed call per candle per symbol, ~81 ms budget | five judgements: regime, direction, toxic_flow, setup_quality, risk_state, plus calibrated probabilities |
| CODE | `trading_agent/` (deterministic Python) | every candle | every threshold, every size, every side effect, the kill switch |

Jev only judges. The brain only reasons. Code owns the world.

## Inputs

- Closed candles (default 5 minutes), the L2 order book (10 levels) at the close,
  trades during the candle, and the account state. All timestamped in epoch ms.
- The compiled research universe (`docs/trading/research/finalists.json`): per
  symbol, thesis direction, reward/risk, regime fit, invalidation levels, data grade.

## Outputs

- At most one market order per symbol per candle, after the risk layer.
- A journal row for every snapshot, decision, order verdict, fill, equity point,
  escalation and review (`data/trader_journal.db`).
- A nightly review in `docs/trading/reviews/YYYY-MM-DD.md` with a bounded proposal.

## The decision schema (what Jev answers)

| Field | Type | Values |
|---|---|---|
| regime | choice | trending, mean_reverting, high_vol, crisis |
| direction | choice | long, short, neutral |
| toxic_flow | noul | probability the flow is informed; true when >= 0.5 |
| setup_quality | score | 0, 1, 2, 3 |
| risk_state | choice | safe, near_limit, reduce |
| confidence | derived | probability of the chosen direction, 0..1 |

Full definitions: `trading_agent/schema.py` (`JEV_QUESTIONS`, `JevDecision`).

## Entry rule (code, `trading_agent/policy.py`)

Fire only when ALL hold: setup_quality >= 2, confidence > 0.80, effective
risk_state == safe (the worse of the code's and the model's), direction != neutral,
flow not toxic, regime != crisis, setup not invalidated, direction matches the
thesis direction, no open position in the symbol, order book fresher than one candle.

Size: fractional Kelly on Jev's calibrated probability p with the setup's
reward/risk b, f* = p - (1 - p) / b, times `kelly_fraction` (0.25), hard-capped at
quarter Kelly, then capped by every risk limit.

Escalate to the brain (no new risk that candle) when confidence < 0.60 or the
regime flips to crisis.

## Hard limits (code, `trading_agent/risk.py`; the model can never override)

| Limit | Default | Effect |
|---|---|---|
| max drawdown | 15 % of peak equity | kill switch fires; only reduce-only orders pass |
| max daily loss | 3 % of day-start equity | no new risk until the next UTC day |
| max position | 20 % of equity per symbol | orders clipped |
| max gross | 60 % of equity | orders clipped |
| max order | 10 % of equity | orders clipped |
| rate limit | 12 orders per hour per symbol | blocked |
| kill switch | file `data/KILL_SWITCH`, armed at start, checked before every order | triggered: flatten, then nothing |

Consequence worth stating: with these defaults a single position can lose at most
10 % of equity, and the daily-loss rule flattens long before the drawdown rule.

## Snapshot budget

The state engine emits one snapshot per candle under 400 tokens, computed only
from events with ts <= as_of. Out-of-order events raise `CausalityError`.

## Self-improvement

Every night: Brier score and calibration table over resolved decisions, fills vs
misses (gated-out winners), hit rates by setup_quality and regime, PnL and
drawdown, reflex latency. The brain proposes parameter values (bounded in code,
`review.PARAM_BOUNDS`) and question rewrites. The operator applies them by
setting `POLICY_*` variables before the next open. Risk limits are not proposable.

## Out of scope (v0.1)

- Limit orders, iceberg/TWAP execution, funding-rate arbitrage, cross-venue hedging.
- Derivatives-specific margin modelling (sizes are spot-style notional fractions).
- Automatic application of brain proposals (operator gate by design).
- Running the live loop on GitHub Actions (needs a persistent host; see BUILD_PLAN Phase 6).

## Acceptance criteria

- `pytest -q` green (55 trading tests + the 24 existing tests).
- `python run_trader.py --mode backtest --synthetic` reproduces byte-identical
  journals for the same seed.
- Paper trading for at least 14 days with zero limit breaches and a resolved
  calibration sample before `TRADER_LIVE=true` is ever set.
