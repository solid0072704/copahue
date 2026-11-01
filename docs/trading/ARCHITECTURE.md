# ARCHITECTURE

Phase 2 artifact. Approval gate: the operator confirms the layer split and the
file map before the plan is written.

## Data flow per candle

```
feed (CandleTick) ──► StateEngine.ingest_*  (causal, ts-ordered)
                      StateEngine.snapshot(as_of)  ──► MarketSnapshot (< 400 tokens)
                                                          │
                                 KillSwitch.check() ◄─────┤ (triggered → flatten only)
                                                          ▼
                                            Reflex.judge(snapshot)  Jev, one call
                                                          │ JevDecision
                                                          ▼
                                   Policy.decide(snapshot, decision)  code
                               hold │ order │ flatten │ escalate ──► Brain.deep_reread (Opus 5.5)
                                                          │ OrderIntent
                                                          ▼
                                    RiskLayer.evaluate(intent, account)  code
                                         allow / clip / block
                                                          ▼
                                            Exchange.place(final)  paper or ccxt
                                                          ▼
                                            Journal (decisions, orders, fills, equity, escalations)
```

Nightly: `NightlyReview.run` reads the journal, computes Brier/calibration/misses,
asks `Brain.nightly_review`, writes `docs/trading/reviews/<date>.md`, and prints
bounded `POLICY_*` lines the operator may apply.

## Invariants

1. **Nothing from the model layer calls the exchange.** `LiveLoop._execute` is
   the only call site of `Exchange.place`, and it runs after `RiskLayer.evaluate`.
2. **The risk layer takes no model input.** `RiskLayer.evaluate(intent, account,
   mid, ...)` has no decision or snapshot parameter (asserted by a test).
3. **The model may only tighten.** `effective_risk_state = worse(code, model)`.
4. **Causality.** Streams must arrive in order; `snapshot(as_of)` reads only
   events with ts <= as_of. Live and replay therefore produce identical snapshots
   for identical events.
5. **Failure degrades to hold.** Any `ReflexError` becomes `JevDecision.safe_hold`;
   any brain failure becomes a 12-candle pause; any exchange error is journaled and
   the candle is over.
6. **One order can never flip a position.** Orders against the position are
   clipped to the position size; re-entry goes through the normal gates later.
7. **Proposals are bounded.** `review.apply_proposal` clamps every parameter to
   `PARAM_BOUNDS`; `kelly_cap` is pinned at 0.25; `RiskLimits` are not proposable.

## File map

| Path | Role |
|---|---|
| `trading_agent/config.py` | `TraderConfig`, `RiskLimits`, `PolicyParams` from environment |
| `trading_agent/schema.py` | `JEV_QUESTIONS`, `JevDecision`, `MarketSnapshot`, events, `OrderIntent`, `Fill`, `SetupSpec` |
| `trading_agent/state_engine.py` | deterministic snapshot, causal ingestion, pure metric functions |
| `trading_agent/risk.py` | `RiskLayer`, `KillSwitch`, usage and risk-state computation |
| `trading_agent/policy.py` | gates, escalation rules, fractional Kelly sizing |
| `trading_agent/reflex.py` | `JevClient` (wire shape + tolerant parser), Claude Haiku fallback, rules baseline, recorded replay |
| `trading_agent/brain.py` | Claude Opus 5.5 escalation re-read and nightly review (structured outputs, server-side fallbacks) |
| `trading_agent/exchange.py` | `PaperExchange` (book-walking fills, fees, slippage), `CcxtExchange` |
| `trading_agent/feed.py` | `SyntheticFeed` (seeded), `ReplayFeed` (CSV), `CcxtFeed` (live polling) |
| `trading_agent/journal.py` | SQLite journal and outcome resolution |
| `trading_agent/loop.py` | `LiveLoop`: the per-candle pipeline above |
| `trading_agent/backtest.py` | same loop over a replay or synthetic feed with the paper exchange |
| `trading_agent/review.py` | Brier, calibration, report, bounded proposals, markdown |
| `trading_agent/universe.py` | finalists.json → `SetupSpec` (C-grade research becomes neutral) |
| `run_trader.py` | CLI: paper, live, backtest, review, once, kill |
| `tests/test_trading_*.py` | the Phase 4 test suite |
| `docs/trading/research/` | market regime, finalists (markdown + JSON) |
| `docs/trading/reviews/` | nightly reviews |

## External contracts

- **Jev**: `POST https://api.typesafe.ai/v1/systemone`, bearer key, body
  `{model, state, questions}`, answers keyed by question name with `choice`/`score`/
  `noul`, `confidence`, `probabilities`. Confirmed from the community SDK and guide;
  Phase 5 re-confirms against console.typesafe.ai before any live call.
- **Claude**: `claude-opus-5-5` for the brain (effort high, `output_config` JSON
  schema, `fallbacks: "default"`), `claude-haiku-5-5` for the reflex fallback.
- **Exchange**: ccxt. Market data is public; orders need keys and `TRADER_LIVE=true`.

## Deployment shape

- Live loop: one long-running process per venue (`python run_trader.py --mode paper|live`),
  on a VPS or a container with a persistent `data/` volume. Heartbeat file
  `data/trader_heartbeat` for a watchdog.
- Nightly review: cron `python run_trader.py --mode review` after 00:00 UTC.
- Kill switch: `python run_trader.py --kill "reason"` or `touch data/KILL_SWITCH`.
