# BUILD PLAN: six phases, one approval gate each

The prompt asked for this to be driven through AgenKit (agenkit.xyz). That host
did not resolve from this session's network allowlist and did not appear in web
search, so the six-phase discipline is encoded here, in the repository, with the
exact artifacts and gate checklists a harness (AgenKit, if available to you, or
plain Claude Code / Codex) needs to drive it. Each phase ends with an explicit
operator decision; nothing in a later phase starts before the gate is passed.

Legend: ✅ done in this session · ⏳ needs the operator or a network the session did not have.

## Phase 1: Spec ✅

Artifact: `docs/trading/SPEC.md`.

Gate checklist (operator):
- [ ] Hard limits confirmed (15 % max drawdown, 3 % daily loss, 20 % position, 60 % gross, 10 % order, 12 orders/hour).
- [ ] Venue, symbols, candle size and starting capital confirmed (`EXCHANGE_ID`, `TRADER_SYMBOLS`, `CANDLE_SECONDS`, `STARTING_EQUITY`).
- [ ] "Out of scope" accepted.

## Phase 2: Architecture ✅

Artifact: `docs/trading/ARCHITECTURE.md`.

Gate checklist:
- [ ] Layer split accepted: Jev judges, Opus 5.5 reasons, code owns thresholds and side effects.
- [ ] Invariants 1–7 accepted as test obligations.
- [ ] External contracts accepted (Jev wire shape, Claude models, ccxt).

## Phase 3: Plan ✅

Artifact: this file plus the research layer.

| Step | Files | Done |
|---|---|---|
| Research: regime and finalists, cited and dated | `docs/trading/research/market_regime.md`, `finalists.md`, `finalists.json` | ✅ (sources via web search snippets; direct fetches were blocked, see README) |
| Compile the universe | `trading_agent/universe.py` reads `finalists.json`, C-grade → neutral | ✅ |
| Schema | `trading_agent/schema.py` | ✅ |
| State engine | `trading_agent/state_engine.py` | ✅ |
| Risk layer + kill switch | `trading_agent/risk.py` | ✅ |
| Policy + Kelly | `trading_agent/policy.py` | ✅ |
| Reflex (Jev, Claude fallback, rules, recorded) | `trading_agent/reflex.py` | ✅ |
| Brain (Opus 5.5) | `trading_agent/brain.py` | ✅ |
| Exchange, feed, journal, loop, backtest, review, CLI | `trading_agent/exchange.py`, `feed.py`, `journal.py`, `loop.py`, `backtest.py`, `review.py`, `run_trader.py` | ✅ |

Gate checklist:
- [ ] Finalists reviewed; any setup the operator rejects is removed from `finalists.json` (or downgraded to C).
- [ ] Reward/risk per setup (`reward_risk`) and invalidation levels (`invalidate_above` / `invalidate_below`) added by the operator where the research gives levels.

## Phase 4: Test-first build ✅

Tests were written against the spec and the invariants, then the modules were made to pass them.

| Test file | Covers |
|---|---|
| `tests/test_trading_schema.py` | closed schema, strict JSON schema, Jev request/response shapes, bad-shape rejection, 400-token budget, determinism |
| `tests/test_trading_state_engine.py` | determinism, causality (out-of-order raises; snapshot never reads the future), pure metrics, risk state in code, token guard |
| `tests/test_trading_risk.py` | no model input, kill switch (not armed, operator file, drawdown breach), daily loss, order/position/gross caps, rate limit, stale price, no flip through zero |
| `tests/test_trading_policy.py` | Kelly math and quarter cap, strict thresholds, model may only tighten, escalations, sizing and stops, thesis direction, position management, pause |
| `tests/test_trading_loop.py` | end-to-end on a seeded feed, reflex failure → hold, operator kill switch → flatten and block, drawdown → kill switch, escalation → pause, backtest report, paper accounting |
| `tests/test_trading_review.py` | Brier, calibration bins, bounded proposals, nightly files, universe quality gate |

Run: `pip install -r requirements.txt pytest && pytest -q` (79 tests: 55 trading + 24 existing).

Gate checklist:
- [x] Suite green.
- [x] `python run_trader.py --mode backtest --synthetic --candles 900 --seed 7` runs and reports Brier, drawdown, fills.
- [ ] Operator reads one nightly review produced from a backtest journal.

## Phase 5: Review ⏳

Nothing here can be finished without the operator's keys and network.

1. **Jev contract check.** From a machine that can reach `console.typesafe.ai`:
   create a key, run one request with `trading_agent.reflex.build_jev_request`, and
   diff the response against `tests/test_trading_schema.py::JEV_RESPONSE`. Fix
   `parse_jev_response` if a field differs. Confirm latency on the 400-token state.
2. **Exchange reconciliation.** `CcxtExchange` tracks positions locally from fills.
   Before live: implement `sync_equity` checks against `fetch_balance` /
   `fetch_positions` for the chosen venue, and add a startup reconciliation that
   refuses to start if local and venue positions differ.
3. **Paper trading.** `TRADER_MODE=paper REFLEX_BACKEND=jev python run_trader.py`
   for at least 14 days. Nightly `--mode review`. Pass criteria: zero limit
   breaches, resolved sample >= 200 decisions, Brier on fired decisions <= 0.25
   and below the always-0.5 baseline, p95 reflex latency inside `JEV_TIMEOUT_MS`.
4. **Code review.** Security: keys only in `.env`/secrets; `TRADER_LIVE` default false;
   kill switch drill (touch the file, verify flatten on the next candle, verify no
   re-arm without `reset`).
5. **Costs and slippage.** Compare paper fills' `slippage_bps` and fees with the
   venue's real taker fee; re-run the backtest with the measured numbers.

Gate checklist:
- [ ] Jev response shape confirmed and tests updated if needed.
- [ ] Paper-trading report attached and accepted.
- [ ] Reconciliation implemented and drilled.
- [ ] Kill switch drill passed.

## Phase 6: Ship ⏳

1. Smallest capital the venue allows. `TRADER_LIVE=true`, `EXCHANGE_SANDBOX=false`,
   `REFLEX_BACKEND=jev`, `ANTHROPIC_API_KEY` set for escalations and reviews.
2. Process supervision (systemd or a container restart policy) and a watchdog on
   `data/trader_heartbeat` that pulls the kill switch if the heartbeat stalls.
3. Nightly review cron. Every parameter proposal is applied by hand by setting the
   printed `POLICY_*` lines and restarting; question rewrites go through a pull request
   that edits `JEV_QUESTIONS` with the review linked.
4. Weekly: re-run Phase 5 criteria on the live journal; any breach rolls back to paper.

Gate checklist:
- [ ] Operator approves the live start with the exact capital and symbols.
- [ ] First week of live reviews reviewed; decision to scale, hold or roll back recorded in `docs/trading/reviews/`.
