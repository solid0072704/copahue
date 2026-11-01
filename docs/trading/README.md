# Autonomous crypto trading agent: delivery report (2026-11-01)

This is the output the prompt asked for: market research and finalists with theses,
catalysts, bear cases and invalidation conditions; the compiled Jev decision schema;
the six-phase build plan with exact file paths; the state engine, policy and risk
code; the overnight review loop; and the final self-check. Everything below is
either in this repository or explicitly marked as not done.

Read in this order: [`SPEC.md`](SPEC.md) → [`ARCHITECTURE.md`](ARCHITECTURE.md) →
[`BUILD_PLAN.md`](BUILD_PLAN.md) → [`research/market_regime.md`](research/market_regime.md) →
[`research/finalists.md`](research/finalists.md).

## What was verified in this session, and what was not

| Item | Status |
|---|---|
| Code: 14 modules in `trading_agent/`, CLI, 55 trading tests + 24 existing tests | ✅ `pytest -q` passes (79) |
| Synthetic backtest reproduces byte-identical journals for the same seed | ✅ verified twice |
| Research with dated sources | ✅ written, but every number came from **web-search snippets**: direct fetches to DefiLlama, Coinbase, CoinDesk and others were blocked by the session's network allowlist |
| Jev wire shape | ⚠️ taken from the community SDK (`jevper`) and the unofficial guide; `docs.typesafe.ai` and `api.typesafe.ai` were unreachable. No live Jev call was made. |
| Exchange connectivity (Binance, Coinbase, Kraken, Bybit public APIs) | ❌ blocked from this container; `CcxtExchange` and `CcxtFeed` are untested against a venue |
| AgenKit (agenkit.xyz) | ❌ did not resolve and did not appear in web search; the six-phase discipline is implemented in `BUILD_PLAN.md` with gate checklists |
| Live trading | ❌ by design: `TRADER_LIVE=true` is refused until the Phase 5 checklist is signed |

## 1. Market regime (summary of `research/market_regime.md`)

Sources are dated Oct 20–31, 2026 and listed in the file. Arithmetic by the
researcher is tagged [ESTIMATE] there.

**BTC: trending, bearish drift, moderate conviction.** ~$77.6k on Oct 31; 1M −8.75 %,
1W −3.56 %, 90d ≈ +24 % [ESTIMATE]; below the 50d ($81.8k, lost Oct 24) and above the
200d (~$72.9k). Funding ~0.005 %/8h (neutral), OI ~$53.6B flat, DVOL 36–41 (near
multi-year lows, puts ~10 vol points over calls). October spot-ETF net ≈ −$516M with
−$643M on Oct 30. Fed hiked Sept 16 to 3.75–4.00 %, held Oct 28 with two dissents for
a further hike; 10y 4.93 %; stablecoin supply $296.7B, −2.0 % over 30d (USDC −$4.1B).
Not mean-reverting (one-directional drift, wide MA band), not high-vol (DVOL low),
not crisis (funding neutral, no venue or stablecoin break).

**ETH: trending, bearish, higher conviction.** ~$2,245; 30d −13 %, −20 % from the
September high; below the 200d (~$2,320); record 23-day ETF outflow streak (>$1.08B
in October through Oct 20); OI −10–12 % m/m; Glamsterdam testnet activation postponed
Oct 22.

**Narrative read:** a de-grossing, not a rotation. Relative holders: privacy (ZEC,
but −28 % off peak), perp-DEX products (Hyperliquid ~38 % of on-chain perp volume),
RWA infrastructure (DTCC tokenization service live Oct 28). Laggards: AI tokens,
L1 beta, DeFi governance tokens, DA.

**Flip conditions** (what changes the labels) are written per asset in the file;
for BTC: reclaim of the 50d with funding < 0.01 % → mean_reverting; DVOL > 45 with
two ±5 % days → high_vol; close below the 200d + negative funding + stablecoins
−5 %/30d → crisis.

## 2. Finalists (summary of `research/finalists.md`, machine form in `research/finalists.json`)

| # | Setup | Dir | Grade | Confirmed catalyst (dated) | Bear case, first point | Invalidation (first two) |
|---|---|---|---|---|---|---|
| 1 | HYPE (Hyperliquid) | short, tactical | B | Nov 6 contributor distribution (size disputed across Tokenomist / CoinLaunch / DefiLlama) | ~29× cap / ~108× FDV on ~$770M annualized holder revenue while volume share fell 58 % → 38 % | daily close > $98; 30-day holder revenue > $80M |
| 2 | MON (Monad) | short | B | Nov 24 cliff: 16.6–17.1B MON, ~145 % of float (~$328M) onto a $228M cap | 26 days of total volume to absorb [ESTIMATE] | team tranche re-locked; daily close > $0.033 |
| 3 | TIA (Celestia) | short | C → **neutral in the universe** | Oct 31 unlock of 175.75M TIA (~18 % of float) after a −12.8 % day | distribution already under way; no fee revenue sourced | reclaim $0.57 on > $150M/day; sourced blob-fee revenue > $100K/day |
| 4 | ZEC (Zcash) | short | B | Oct 23–30: −28 % from the ~$1,700 peak, $20.8B cap, no fee revenue | cap larger than UNI+AAVE+ENA+PUMP+TIA+MON combined [ESTIMATE] | daily close > $1,700; SEC approval of ZCSH |
| 5 | ENA (Ethena) | short | B | Nov 1 unlock of 275M ENA; fee switch gated at $7.5B USDe vs ~$4.7B | carry collapse: BTC funding ~5 % annualized vs 4.93 % 10y | 14-day avg USDe supply > $6B and rising; BTC funding > 0.01 %/8h for two weeks |
| 6 | UNI (Uniswap) | long, staged | A | fee switch live since Dec 28, 2025 (17 % of swap fees burned, ~11.4 % of supply retired) | Oct 1–29 revenue $7.58M, half of September's pace | November revenue < $6M; daily close < $6.00 |
| 7 | PUMP (Pump.fun) | long, staged | B | Hyperliquid-powered perps inside the app (Oct); Binance collateral ratio 50 % → 60 % (Oct 30) | daily revenue −16 % in a week; highest-beta activity in crypto | daily revenue < $1.0M for 14 days; buyback paused |
| 8 | AAVE (Aave) | long, staged | B | Aavenomics 3.0 (buyback, staking, burn) announced Oct 22; execution status conflicting | TVL −18.9 % in 30 days, unexplained | TVL < $13B; no on-chain buyback by 2026-12-31 |
| 9 | ETH | long, contrarian, **inactive** (regime fit: mean_reverting only) | A | record ETF outflow streak (23 days on Oct 30) | institutional distribution, not retail panic | daily close < $1,860; outflow streak > 30 days |

How this becomes code: `trading_agent/universe.py` compiles the JSON into
`SetupSpec`s. Shorts carry regime fit trending/high_vol and so can fire under the
current labels; the longs are staged behind a `mean_reverting` label; TIA is
C-grade and therefore neutral (the policy never fires a thesis it cannot source).
Reward/risk per setup defaults to 1.5 and the numeric invalidation levels
(`invalidate_above` / `invalidate_below`) are an operator input at the Phase 3 gate.

Not sourced anywhere (left as "not sourced", never filled): holder concentration
for every token; MON and TIA fees; ENA October revenue; HYPE TVL and active users;
UNI/AAVE unlock schedules; ETH implied vol. Conflicts between sources are listed
in the research file (HYPE unlock size, MON unlock size, AAVE buyback status, BTC
realized vol). One number was deliberately not used: Coinbase Institutional's
"+$2.6B October BTC ETF inflow" contradicts the Farside daily tallies.

## 3. The compiled Jev decision schema

One call per candle per symbol, five questions evaluated in parallel
(`trading_agent/schema.py::JEV_QUESTIONS`, request built by
`reflex.build_jev_request`, response parsed by `reflex.parse_jev_response`):

| Question | Jev type | Answers | Used by code as |
|---|---|---|---|
| regime | choice | trending, mean_reverting, high_vol, crisis | crisis → escalate; must be in the setup's regime fit |
| direction | choice | long, short, neutral | side of the order; must match the thesis direction |
| toxic_flow | noul (probability of yes) | true when ≥ 0.5 | no new entries when true |
| setup_quality | score 0–3 | integer | fire only when ≥ 2 |
| risk_state | choice | safe, near_limit, reduce | `worse(code, model)`; fire only when safe |
| confidence | derived | p(direction) from Jev's probabilities | fire only when > 0.80; escalate when < 0.60; Kelly input |

The state Jev sees is `MarketSnapshot.compact()`: ~185 tokens in practice, hard
guard at 400, computed only from events with ts ≤ as_of:

```json
{"c":{"atr":0.545,"n":100,"r1":16.3,"r12":2.18,"r48":1.15,"rng":0.96,"rv":130.3,"vz":-0.12},
 "cal":{"h":17,"wd":3},"f":{"age":110000,"flow":-0.177,"rate":3.8},
 "inv":{"frac":0.304,"qty":0.05,"upnl":1.01},
 "px":{"age":0,"imb":0.257,"micro":0.0,"mid":60876.03,"spr":2.0},
 "risk":{"day":-0.99,"dd":0.0476,"state":"reduce","u_day":0.33,"u_dd":0.32,"u_pos":1.52},
 "setup":{"dir":"long","fit":["trending"],"inv":false,"rr":1.5,"th":"..."},
 "sym":"BTC/USDT","t":1760030000000,"tf":300}
```

Sizing (`policy.size_fraction`): f* = p − (1 − p)/b with b the setup's reward/risk;
position fraction = min(kelly_fraction, 0.25) × f*, then every risk cap. With the
defaults a winning 0.90-confidence call on b = 1.5 sizes 20.8 % of equity before
caps and 10 % after the per-order cap.

## 4. Build plan status

[`BUILD_PLAN.md`](BUILD_PLAN.md): Phases 1–4 (spec, architecture, plan, test-first
build) are done in this repository. Phase 5 (review: Jev contract check, exchange
reconciliation, 14 days of paper trading, kill-switch drill) and Phase 6 (ship)
need the operator's keys, a host with network access, and sign-off at each gate.

## 5. Where every rule lives

| Rule from the prompt | Code | Test |
|---|---|---|
| Deterministic snapshot under 400 tokens, strict causal timestamps | `state_engine.py` (`StateEngine.snapshot`, `CausalityError`) | `test_trading_state_engine.py` |
| Mid, spread, imbalance, realized vol, inventory, drawdown computed in code | `state_engine.py` pure functions, `risk.compute_usage` | `test_pure_metrics`, `test_risk_state_is_computed_in_code` |
| Max drawdown 15 %, max position, max daily loss, armed kill switch before every order | `risk.py` (`RiskLayer.evaluate`, `KillSwitch`) | `test_trading_risk.py` |
| Fire only when quality ≥ 2, confidence > 0.80, risk_state safe | `policy.py` gates | `test_fires_only_above_thresholds` |
| Fractional Kelly capped at quarter Kelly | `policy.kelly_full`, `size_fraction`, `PolicyParams.validate` | `test_kelly_math_and_quarter_cap` |
| Escalate below 0.60 or on a crisis flip | `policy.decide` → `loop.process` → `brain.deep_reread` | `test_escalations`, `test_escalation_pauses_entries_with_null_brain` |
| Cheap fast calls by default, heavy reasoning only when it matters | Jev per candle; Opus 5.5 only on escalation and nightly | `loop.py` |
| Model can never override limits | `RiskLayer.evaluate` has no model parameter; `worse_risk_state` | `test_risk_layer_does_not_take_model_input`, `test_code_risk_state_wins_over_model` |
| Nightly review: fills, misses, Brier, schema rewrite | `review.py`, `journal.misses_between`, `brain.nightly_review` | `test_trading_review.py` |

Two rules the code adds beyond the prompt, both deliberate: no pyramiding (one
position per symbol, flatten before re-entry), and one order can never flip a
position through zero.

## 6. The overnight loop

`python run_trader.py --mode review` after the UTC day closes:

1. Resolve outcomes: each decision is scored at its horizon (12 candles); a long/short
   "wins" when the mid moved its way by more than 15 bps (about one round trip of fees
   and slippage, `POLICY_OUTCOME_MOVE_BPS`).
2. Report (`review.build_report`): Brier score and a 10-bin calibration table over
   resolved directional decisions, hit rate by setup_quality, regime and source, fired
   decisions separately, orders blocked and clipped with reasons, fills, fees, slippage,
   PnL, max drawdown, misses (gated-out winners), escalations, reflex latency p50/p95.
3. Proposal (`brain.nightly_review`, Claude Opus 5.5 with a strict JSON schema): new
   values for the six policy parameters, rewrites of the Jev question instructions,
   risk flags, and its own "what could I be wrong about".
4. Bounds (`review.apply_proposal`): every value is clamped to `PARAM_BOUNDS`;
   `kelly_cap` stays 0.25; risk limits are not proposable at all.
5. Output: `docs/trading/reviews/<date>.md` and `.json`, plus the `POLICY_*` lines to
   paste into `.env`. The operator applies them (or not) before the next open; question
   rewrites go through a pull request editing `JEV_QUESTIONS`.

## 7. Running it

```bash
pip install -r requirements.txt pytest
pytest -q                                                   # 79 tests
python run_trader.py --mode backtest --synthetic --candles 900 --seed 7
python run_trader.py --mode backtest --replay BTC/USDT=data/btc_5m.csv   # ts,open,high,low,close,volume
cp .env.example .env                                        # fill JEV_API_KEY, ANTHROPIC_API_KEY, exchange keys
python run_trader.py --mode paper                           # live data via ccxt, paper fills, journal in data/
python run_trader.py --mode review --date 2026-11-02
python run_trader.py --kill "manual stop"                   # or: touch data/KILL_SWITCH
TRADER_LIVE=true python run_trader.py --mode live           # only after the Phase 5 gate
```

Synthetic backtest on seed 7 (900 candles, rules baseline, no model): 4 fills,
63 escalations, max drawdown 2.5 %, Brier 0.277 on 71 resolved decisions. That
Brier is **worse than the 0.25 of always answering 0.5**: the rules baseline is a
pipeline test, not an edge, and the report says so. The number that matters is
Jev's Brier on real paper-trading data in Phase 5.

## 8. Final check

- **Is the edge organic or incentive-driven?** The five shorts are supply-schedule
  and valuation theses (unlocks, no fee revenue, a buyback that cannot switch on);
  that is organic but crowded: unlock dates are public and front-run. The three longs
  are buyback-backed cash-flow theses; the buyback is itself an incentive and shrinks
  with revenue, which is the bear case the research states.
- **Is the catalyst already priced in?** Probably partly, for every dated unlock. The
  system does not trade the date; it trades Jev's per-candle judgement under the
  regime label, and the research's invalidation levels tell the operator when the
  thesis is wrong.
- **Does value accrue to the token?** Yes with evidence for UNI (burn), PUMP (buyback),
  HYPE (buyback, which is why the short is tactical); announced but unconfirmed for
  AAVE; no for MON, TIA, ZEC, ENA (until $7.5B USDe).
- **Will the strategy survive costs and slippage?** Only if Jev's 0.80+ calls are right
  well over half the time after 15 bps round trips; the outcome threshold and the
  paper exchange's fees and slippage are built in so the review measures exactly that.
- **Is any hard limit delegated to the model?** No. The risk layer has no model
  parameter; the model's risk_state can only tighten; proposals cannot touch limits.

## WHAT COULD I BE WRONG ABOUT?

1. **The research numbers.** Every figure came from search snippets because the
   primary pages were unreachable. Snippets truncate, mix dates, and quote secondary
   outlets. Macro facts (the September hike, the October hold, the dissents) were
   not read from a Fed page. Treat the regime labels as a hypothesis to re-check from
   primary sources before Phase 5.
2. **Jev's wire shape and behaviour.** The request/response format comes from a
   community SDK and an unofficial guide; a renamed field would make every call a
   `ReflexError` (safe: the loop holds) but would also make the system do nothing.
   The 81 ms latency in the prompt is a vendor-style figure I could not measure.
3. **Calibration is assumed, then verified, not guaranteed.** Kelly sizing on a
   miscalibrated probability over-bets. Quarter Kelly plus the order and position
   caps limit the damage to 10 % of equity per position, but the whole edge rests on
   Jev's probabilities being honest on this snapshot distribution, which only the
   Brier gate in Phase 5 can show.
4. **The snapshot is thin.** 400 tokens of one symbol's book, candles and flow carry
   no funding, OI, cross-asset, ETF-flow or on-chain data. The regime a single symbol's
   candles imply can disagree with the regime the research describes. Jev is judging
   a narrow state.
5. **Shorts on spot do not exist.** Five of nine setups are shorts; the exchange
   adapters size spot-style notional and do not model margin, funding payments or
   liquidation. Executing the shorts needs a perp adapter with margin modelling that
   is not written.
6. **The synthetic feed lies kindly.** Gaussian returns, no fat tails, no liquidity
   holes, order books that always refill. Paper fills are therefore optimistic. The
   backtest proves plumbing and determinism, not profitability.
7. **Outcome labelling is a choice.** A 12-candle horizon and a 15 bps threshold
   define "win"; different horizons would produce different Brier scores and a
   different list of "misses". Misses are gated-out decisions whose direction happened
   to pay, which includes noise.
8. **Daily and drawdown accounting.** The daily-loss window resets at UTC midnight and
   the equity peak never decays, so after a 12 % drawdown the system sits near its
   limit until a new high: safe, but it can stall for weeks.
9. **Position truth lives locally.** `CcxtExchange` tracks positions from its own fills;
   partial fills, exchange-side cancels or manual trades desynchronise it. The Phase 5
   reconciliation is a requirement, not a nice-to-have.
10. **Operational state is not persisted.** A restart forgets paused entries and the
    previous regime; the kill switch is a file and the heartbeat is a file. There is no
    clock-drift handling and no multi-venue failover.
11. **Nothing here is legal, tax or venue-compliance advice**, and the prompt's
    AgenKit harness could not be verified to exist.
