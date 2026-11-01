# Finalists — asymmetric setups, as of 2026-11-01

**Method note.** Every figure comes from a web-search result captured 2026-10-31/11-01; direct fetches of defillama.com, coinbase.com, coindesk.com and cryptoslate.com were blocked by the session network allowlist, so their numbers are quoted via search snippets or secondary outlets. Arithmetic is tagged [ESTIMATE]; opinion is tagged [SPECULATION]; "not sourced" means no result stated it. Regime context (see market_regime.md): BTC *trending* (bearish drift), ETH *trending* (bearish). Shorts below are active-regime trades; longs are staged for a *mean_reverting* flip and should not be sized until the flip conditions in market_regime.md trigger.

| # | Ticker | Direction | Regime fit | 90d unlock risk | Value accrual | Data |
|---|---|---|---|---|---|---|
| 1 | HYPE | short (tactical) | trending, high_vol | medium (conflicting) | buyback | B |
| 2 | MON | short | trending, high_vol | high | not sourced | B |
| 3 | TIA | short | trending, high_vol | high (just released) | none | C |
| 4 | ZEC | short | high_vol, trending | none | none | B |
| 5 | ENA | short | trending, mean_reverting | medium | none (gated) | B |
| 6 | UNI | long (staged) | mean_reverting, trending-up | not sourced | buyback/burn | A |
| 7 | PUMP | long (staged) | mean_reverting, trending-up | low | buyback/burn | B |
| 8 | AAVE | long (staged) | mean_reverting, trending-up | not sourced | buyback (paused) | B |
| 9 | ETH | long (contrarian, inactive) | mean_reverting | none | staking real yield | A |

---

## 1. HYPE — Hyperliquid — SHORT (tactical, into the November distribution)

**Where it trades.** Hyperliquid L1 (HyperCore). Native spot and HYPE-USD perp on Hyperliquid; CEX perp listings not re-verified this session.
**Price / size.** $87.12 (Coinbase, Oct 31; CMC $88.25), ATH $97.99; market cap $22.1B; 24h volume $1.85B. Circulating 253.5M (CMC) vs 240M (CoinGecko) — provider conflict; max supply 950.4M → FDV ≈ $83B [ESTIMATE].
**Supply / unlocks.** ~26% of max supply circulates. 45.7% of total supply staked across 34 validators; five Hyper Foundation validators hold 48.7% of stake (Aug 30). Unlocks conflict: Tokenomist shows a $923.63M HYPE allocation on Nov 6 (≈10.6M HYPE, ≈4% of float [ESTIMATE]); CoinLaunch shows 6.43M HYPE on Nov 1 monthly; tokenomics.com shows 14.18M on Oct 29; DefiLlama's tracker (observed transfers) shows none scheduled. Multiple sources agree contributor distributions land on the 6th of each month; size is unresolved.
**Revenue / fees.** Holder revenue $63.94M over 30d to Oct 28 (≈$770M annualized [ESTIMATE]); September revenue ≈$105M, flat YoY, with core perp fees down to ~64% of the mix (secondary). Assistance Fund has removed 49.87M HYPE = 4.99% of max supply (Oct 29). Market share Oct 26: 38% of $15.0B/day on-chain perp volume, 56% of OI ($7.55B), down from a 58% volume share reported in August. TVL / active users: not sourced this session.
**Holders.** Top-10 share not sourced. Corporate treasuries hold 33.79M HYPE (Hyperliquid Strategies 29.3M).
**Value accrual.** Yes — ~97–99% of fees route to the Assistance Fund buyback; the strongest programmatic bid in crypto.
**Confirmed catalysts.** Nov 6 monthly contributor distribution (date confirmed; size disputed — see above). Pump.fun perps routed through Hyperliquid builder codes, live October 2026 (new fee line).
**Speculative catalysts.** HYPE ETF (AMINA research note); CFTC "onshoring" narrative; HIP-3 OI growth.
**Bear case (supports the short).** (1) $22B cap / $83B FDV on ≈$770M annualized holder revenue is ~29x / ~108x while volume share erodes 58% → 38%. (2) 74% of supply is non-circulating; monthly contributor tranches plus 33.8M treasury HYPE are latent sellers into a thinning tape. (3) The buyback is pro-cyclical: a bearish regime cuts perp volume → fees → bid at exactly the moment supply arrives. (4) Relative-strength unwind: HYPE is −11% from ATH while BTC −8.75% / ETH −13% on the month and altcoin OI dominance is >1.0 — the crowded winner is sold last and hardest in a de-gross. (5) ~49% of stake controlled by Foundation validators — centralization/governance tail.
**Risks to the short (bull case).** Daily buybacks absorb supply; builder-code and HIP-3 revenue lines grow; ETF filing headlines; low float squeezes.
**Invalidation.** Daily close > $98 (new ATH); 30d holder revenue > $80M (volume re-accelerating); DefiLlama confirms no November contributor transfer while treasuries add; BTC regime flips to trending-up.
**Regime fit:** trending (down), high_vol. **Data quality: B.**

## 2. MON — Monad — SHORT (into the Nov 24 cliff)

**Where it trades.** Monad L1. Perps: Binance MONUSDT, Bybit MONUSDT, OKX MON-USDT-SWAP, Hyperliquid MON-USD (listed as a pre-launch hyperp Oct 8, 2025).
**Price / size.** $0.019 (DefiLlama / CoinGecko $0.01933, Oct 31); market cap $227.7–228.7M; 24h volume $12.7M (CoinGecko). Circulating ≈11.8B [derived]; total 100B → FDV ≈ $1.9B [ESTIMATE]. Mid-October price was capped below $0.033 → ≈ −42% in two weeks [ESTIMATE].
**Unlock.** Nov 24, 2026: DefiLlama 17.105B MON = $328.11M = **144.65% of circulating supply**; tokenomics.com 16.61B (16.6% of total); CryptoTicker/Pluang 10.7B team tokens at the one-year lock-up expiry. Every version exceeds 90% of float.
**Fundamentals.** TVL $888.81M (DefiLlama Oct 31) of which $632.71M stablecoins — TVL ≈ 3.9x market cap, consistent with incentive-driven parking rather than token demand [SPECULATION]. Fees/revenue, active addresses, holder concentration: not sourced.
**Value accrual.** Not sourced (gas/staking token; no buyback or fee share found).
**Confirmed catalysts.** Nov 24 unlock (CoinMarketCal via TradingView; DefiLlama; tokenomics.com).
**Speculative catalysts.** "NYSE partnership" (single blog, unverified); exchange incentive campaigns.
**Bear case.** (1) $328M of new float vs $12.7M/day volume = 26 days of total turnover [ESTIMATE]. (2) Team/insider tokens unlock after a ~99% collapse from the $14–15B valuation hyperps implied pre-launch — cost basis is irrelevant, liquidity-seeking behaviour dominates. (3) No sourced revenue; TVL is stablecoin parking. (4) Price is already trending down into a fully telegraphed event — no front-running bid has appeared.
**Risks to the short.** "Sell-the-rumour" exhaustion and a post-Nov 24 relief rally; a team re-lock/stake commitment; thin liquidity and negative funding make the short expensive (borrow/funding not sourced).
**Invalidation.** Official announcement that the team tranche is re-locked or staked; daily close > $0.033; funding below −0.05%/8h (crowded short); unlock passes with < 5% drawdown.
**Regime fit:** trending, high_vol. **Data quality: B.**

## 3. TIA — Celestia — SHORT (post-cliff overhang, no fee economy)

**Where it trades.** Celestia (Cosmos SDK). Perps: Binance TIAUSDT / TIAUSDC (OI ≈ $11.9M, 29% share), Bybit TIAUSDT, OKX, Hyperliquid TIA-USD (OI ≈ $2.8M, 5x max). Spot on Coinbase.
**Price / size.** $0.460 (CMC, Oct 30, −12.84% in 24h); $0.5741 with $527M cap on Oct 22 (Coinbase); ≈$0.66 around Oct 17 (Bitget) → ≈ −30% in two weeks [ESTIMATE]. Market cap ≈ $450M at $0.46 × 979M [ESTIMATE]. 24h volume $50–91M (Oct 9–18 snapshots).
**Supply / unlocks.** Circulating 979.2M (Oct 9); total 1.179B. 175.75M TIA released Oct 31 (Cointelegraph / DropsTab) ≈ 18% of circulating [ESTIMATE]; DropsTab also shows ~345K/day drip. Issuance cut to ~2.5%/yr, declining to 1.5% (CIP-41, Nov 2025).
**Revenue / fees.** Blob fees are the only revenue; no source gives a material figure (one aggregator shows < $100/day, unverified). TVL n/a. Users and holder concentration: not sourced.
**Value accrual.** None — staking yield is inflation, not fee-backed.
**Confirmed catalysts.** Oct 31 cliff executed on schedule (DropsTab).
**Speculative catalysts.** Large rollup commitments; none dated.
**Bear case.** (1) 18% of float released into a $50–90M/day market as price falls 12.8% in one day — distribution has started. (2) ≈$450M cap with negligible blob-fee revenue: valuation rests entirely on the DA narrative while competing DA (Ethereum blobs, alt-DA) is free or cheaper [SPECULATION — competition data not sourced]. (3) −30% in two weeks and −93% from the $6.51 at which this unlock was originally valued. (4) Aggregate OI only ~$15–40M — the market is spot-driven, so recipients' selling hits price directly.
**Risks to the short.** Already-sold unlock → sharp relief rally; tiny OI → squeezes; an ecosystem announcement.
**Invalidation.** Reclaim $0.57 on > $150M/day volume; sourced blob-fee revenue > $100K/day; a foundation buyback/burn; BTC regime flip.
**Regime fit:** trending, high_vol. **Data quality: C** (price snapshots conflict; revenue not sourced).

## 4. ZEC — Zcash — SHORT (blow-off reversal)

**Where it trades.** Zcash (PoW). Perps: Binance ZECUSDT / ZECUSDC (75x), Bybit, OKX, Hyperliquid ZEC-USD. Spot on Coinbase.
**Price / size.** $1,222 (Coinbase, ~Oct 30–31; −6% d/d, flat w/w); market cap $20.76B; 24h volume $1.387B. Path: mid-$500s (mid-Aug) → ≈$1,700 (early Oct) → $1,221 (Oct 23) = −28% from peak; +178.6% YTD (Crypto.com). ATH $5,941.80 was Oct 29, 2016 — not 2026.
**Supply.** ≈16.7M circulating of 21M max; ≈30% (4.9–5.15M ZEC) in shielded pools. No vesting unlocks (PoW issuance only). Holders: not sourced (shielded pool is opaque by design).
**Derivatives.** Aggregate OI ≈ $524M (Coinalyze, undated); Hyperliquid ZEC OI $565.6M (Buildix snapshot at mark $1,211) vs Binance $258–429M (Sharpe, two snapshots) — conflicting. Perp OI ≈ 2.5–5% of market cap [ESTIMATE].
**Revenue / value accrual.** No fee revenue of note; no mechanism returns value to ZEC. None.
**Confirmed catalysts.** Grayscale filing to convert the Zcash Trust into "The Zcash ETF" (ZCSH, NYSE Arca) — filed; no decision date.
**Speculative catalysts.** SEC approval; further privacy-coin ETF filings; regulatory shifts.
**Bear case.** (1) $20.8B for a chain with no fee revenue — larger than UNI + AAVE + ENA + PUMP + TIA + MON combined (≈$10B) [ESTIMATE]; a pure narrative premium. (2) Shielded-pool growth is mostly partially-shielded flows, not fully private transfers (Coin Metrics), and shielded ≠ illiquid. (3) Textbook parabola: +200% in seven weeks, −28% off the top, momentum broken while $0.5B+ of OI keeps leverage in the trade. (4) Privacy-coin delisting/regulatory tail [SPECULATION — no 2026 action sourced]. (5) The largest OI venue is a DEX with a public liquidation engine — cascades are reflexive.
**Risks to the short.** ETF approval headline; the privacy narrative has been 2026's strongest; a thin tradable float squeezes.
**Invalidation.** Daily close > $1,700 (new high); SEC approval of ZCSH; shielded share > 35% with price stable (adoption outrunning speculation).
**Regime fit:** high_vol, trending. **Data quality: B.**

## 5. ENA — Ethena — SHORT (buyback unreachable; carry collapse)

**Where it trades.** Ethereum. Perps: Binance ENAUSDT, Bybit ENAUSDT (OI ≈ $149M snapshot), OKX, Hyperliquid ENA-USD (10x; OI $20–103M across snapshots).
**Price / size.** $0.18–0.214 (Oct 29–31; CoinPaprika $0.191 on Oct 31); market cap $2.39–2.56B; FDV $2.67B; 24h volume $634M (Oct 29). Circulating 13.41B / max 15B.
**Supply / unlocks.** Nov 1, 2026: 275.0M ENA (1.83% of total, ≈2.1% of float [ESTIMATE]); the 1.41B investor cliff (14.3% of float) landed Oct 5; monthly cadence continues.
**Fundamentals.** USDe supply ≈ $4.66B (Oct 21) / $4.98B (Oct 7), down from ≈$15B at the Oct 2025 peak; sUSDe yield averaged ~5% in Q3; > $800M cumulative rewards paid. October fees/revenue and holder concentration: not sourced.
**Value accrual.** Fee switch approved September 2026 (95% of Foundation net revenue → ENA buybacks) but gated on a 14-day average USDe supply ≥ $7.5B → currently **none**.
**Confirmed catalysts.** Nov 1 unlock (CryptoTicker / Tokenomist).
**Speculative catalysts.** White-label USDe growth (ether.fi USD); discretionary foundation buybacks; sell-side price targets (Standard Chartered $1) [opinion].
**Bear case.** (1) The carry is gone: BTC funding ≈ 0.005%/8h ≈ 5% annualized vs a 4.93% 10-year — USDe has no yield edge in a hiking cycle, and supply is −69% from peak. (2) Buybacks need +60% USDe growth to switch on — unreachable in this regime. (3) ≈275M ENA/month (≈$52M at $0.19 [ESTIMATE]) of new supply with no programmatic bid. (4) Market cap ≈ 0.5x USDe supply looks cheap, but revenue is roughly linear in funding, so low funding → revenue toward zero (revenue not sourced). (5) Risk-off redemptions are reflexive.
**Risks to the short.** A regime flip spikes funding → USDe mints → fee switch activates (a reflexive squeeze); new exchange-collateral integrations; treasury buybacks.
**Invalidation.** 14-day average USDe > $6B and rising; BTC funding > 0.01%/8h for two weeks; ENA daily close > $0.25.
**Regime fit:** trending, mean_reverting. **Data quality: B.**

## 6. UNI — Uniswap — LONG (burn-backed value; staged for regime flip)

**Where it trades.** Ethereum + Unichain / L2s / Robinhood Chain. Perps: Binance UNIUSDT and UNI/USDC (75x), Bybit, OKX, Hyperliquid UNI-USD (OI $75.9M). Spot on Coinbase.
**Price / size.** $6.86–6.89 (Oct 30; MetaMask, Etherscan); market cap $4.23–4.30B; circulating 624.17M; total 886.4M; max 1B → FDV ≈ $6.9B [ESTIMATE]. ATH $44.92. −29% in four weeks. 24h volume $0.7–1.5B (providers conflict).
**Supply / unlocks.** UNI did not appear on any November unlock calendar surfaced (Tokenomist, DefiLlama 30d list), but no source explicitly states "no unlocks" → not sourced.
**Revenue / fees.** Fee switch live since Dec 28, 2025: 17% of swap fees buy and burn UNI; V4 enabled Jul 27, 2026 (DefiLlama). September 2026 revenue $15.49M (DefiLlama) / $14.81M (Token Terminal), a record, with Robinhood Chain contributing $7.87M; **Oct 1–29 only $7.58M** (≈ −51% MoM pace). Burned: 113.62M UNI total including the 100M treasury burn ≈ 11.4% of total supply [ESTIMATE]; 13.68M YTD; 778K in Oct 1–25. Annualized burn ≈ $95M at October pace (≈2.2% of cap) or ≈$186M at September pace (≈4.3%) [ESTIMATE]. Trader fees Jan–Jul 2026 $297.9M (Token Terminal via aggregator). Liquidity ≈ $4B (Coinbase news headline; TVL not verified on DefiLlama). Users and holder concentration: not sourced.
**Value accrual.** Yes — programmatic buy-and-burn.
**Confirmed catalysts.** Multi-chain fee-switch rollout (V4, Jul 27, 2026; Optimism/Arbitrum/Base earlier in 2026 per DefiLlama).
**Speculative catalysts.** Unichain / Robinhood Chain volume recovery; a higher protocol take rate.
**Bear case.** (1) October revenue halved vs September — a 2% burn yield is not a floor. (2) 53% of September revenue came from one partner chain (Robinhood Chain) — concentration, not diversification. (3) UNI fell 29% in four weeks *with* burns running — the burn does not set price in a de-gross. (4) A 17% take rate makes Uniswap LPs less competitive vs zero-fee venues [SPECULATION]. (5) DEX volume is beta to the regime: a bearish month means the smallest burn exactly when support is needed.
**Invalidation.** November revenue < $6M; daily close < $6.00; governance reduces or reverses the fee switch; Robinhood Chain volume −50%.
**Regime fit:** mean_reverting; trending-up after a regime flip. **Data quality: A.**

## 7. PUMP — Pump.fun — LONG (≈15% buyback yield; staged)

**Where it trades.** Solana. Binance (PUMP collateral ratio raised 50% → 60% on cross-margin/Unified Account, effective Oct 30), Bybit, OKX (price page dated Oct 31), Hyperliquid PUMP/USDC (OI ≈ $97M, 38% volume share, CoinGecko snapshot).
**Price / size.** $0.004102 (OKX, Oct 31); market cap $1.89B; circulating 460.17B; total ≈826B; max 1T → FDV ≈ $4.1B on max supply [ESTIMATE]. ATH ≈ $0.009 (Sept 2025). 24h volume: not sourced precisely.
**Supply / unlocks.** Nov 14: 6.9B PUMP ≈ $27.8M ≈ 1.5% of float (Tokenomist via ETHNews; not confirmed by an official post). Reporting conflicts on insiders: one outlet says a 82.5B team+investor cliff (29% of float) unlocked July 12, 2026; Lookonchain relays the project's claim that no team/investor tokens unlock before the 12-month cliff. Team allocation ≈50B per the former.
**Revenue / fees.** Daily revenue $1.59M (Oct 9, −16% from $1.89M on Oct 5); ≈$18.6M/week in early October; top Solana app by revenue (DefiLlama, Oct 20); launchpads earn > 1/3 of Solana's $158M 30-day app revenue (to Oct 24). Buyback: 50% of net revenue bought and burned on-market, program runs to April 2027; cumulative ≈$475.93M / 170.11B PUMP (17.0% of supply) burned as of early October (secondary). At $1.59M/day: ≈$580M annualized revenue → cap/revenue ≈ 3.3x; ≈$290M/yr of buybacks ≈ 15% of market cap [ESTIMATE]. Users and holder concentration: not sourced.
**Value accrual.** Yes — programmatic buy-and-burn.
**Confirmed catalysts.** Hyperliquid-powered perps live in the Pump.fun app (October 2026; 40x BTC, 25x ETH, 20x single stocks); Binance collateral-ratio increase (Oct 30); buyback program through April 2027.
**Speculative catalysts.** Routing perps fees into the buyback (no published document links them); further CEX listings.
**Bear case.** (1) Revenue fell 16% in a week; memecoin launches are the highest-beta activity in crypto — a 50% revenue drop halves the bid overnight. (2) "50% of net revenue" is a company-defined denominator; perps fees are excluded today. (3) −55% from ATH with 17% of supply already burned — supply reduction has not overcome sentiment. (4) Insider-unlock reporting is contradictory → data/trust risk. (5) Launchpad competition on Solana [SPECULATION; share not sourced].
**Invalidation.** Daily revenue < $1.0M for 14 days; buyback paused or redefined; daily close < $0.0035; Solana 30-day app revenue < $100M.
**Regime fit:** mean_reverting; trending-up after a flip. **Data quality: B.**

## 8. AAVE — Aave — LONG (TVL dislocation + buyback restart; staged)

**Where it trades.** Ethereum + multichain. Perps: Binance AAVEUSDT and AAVE/USDC (75x), Bybit, OKX, Hyperliquid AAVE-USD (OI $58–96M across snapshots). Spot on Coinbase.
**Price / size.** $126.05 (Oct 30), from $164.16 on Oct 1 (−23%); an overnight Oct 29–30 move from ≈$137 to ≈$118. Market cap ≈ $2.0B at ≈16M supply [ESTIMATE]; $2.63B on Oct 1. 24h volume: not sourced.
**Supply / unlocks.** ≈16M AAVE (derived); no November unlock listed anywhere surfaced → not sourced.
**Fundamentals.** TVL $15.458B (DefiLlama Oct 30), **−18.9% over 30 days** from $19.17B on Oct 1; DefiLlama's daily table shows $17.65B (Oct 29) → $15.91B (Oct 30), a ≈ −10% day with no confirmed exploit or bad debt — cause unexplained [SPECULATION: large deleveraging or a methodology change]. Revenue ≈ $402M annualized (The Defiant citing DefiLlama 7-day trailing, Oct 22) → cap/revenue ≈ 5x [ESTIMATE]; 2025 revenue ≈ $142M (DL News). Users and holder concentration: not sourced.
**Value accrual.** Buyback — $42.9M / 205K AAVE (1.28% of supply) bought Apr 2025–Apr 2026; **paused since Apr 19, 2026** after the rsETH incident (TokenLogic ARFC; last recorded buy Jun 24 per tokenbuybacks.app). "Aavenomics 3.0" (buyback + staking + burn) announced by Stani Kulechov on the Oct 22, 2026 livestream; "live" status conflicts (The Defiant: live and automated; BloFin Oct 29 and tokenbuybacks.app: still paused). Treat as announced, not executing.
**Confirmed catalysts.** Aavenomics 3.0 announcement (Oct 22); V4 institutional markets for RLUSD, PYUSD, OUSD (Oct 22); Coinbase tokenized stocks live on V4 (TechFlow).
**Speculative catalysts.** On-chain buyback execution restarting; budget sizing (the March ARFC had cut the annual budget from ≈$50M to ≈$30M).
**Bear case.** (1) TVL −19% in 30 days including a −10% day: interest revenue falls with deposits and nobody has explained the move. (2) Buybacks are "restarting" for the third time this year (March cut, April pause, June last buy, October announcement) — credibility is spent. (3) The DAO balance sheet is funding rsETH restitution — contingent liabilities first, buybacks second. (4) DeFi governance tokens are being sold indiscriminately (UNI −29%); AAVE lost $140 then $129.50 support. (5) A prediction market lists "Aave TVL < $10B in 2026" — tail risk is being priced.
**Invalidation.** TVL < $13B; no on-chain buyback executed by Dec 31, 2026; daily close < $118; any bad-debt or exploit disclosure.
**Regime fit:** mean_reverting; trending-up after a flip. **Data quality: B.**

## 9. ETH — Ethereum — LONG (contrarian; inactive under the current regime)

**Where it trades.** All major venues; spot ETFs.
**Price / size.** $2,245 (Oct 31); 30d −13%; −20% from the September high; just below the 200d (SMA ≈ $2,320 / EMA $2,316). Jul 31 ≈ $1,877 → 90d ≈ +20% [ESTIMATE]. 12-month −38.4% vs BTC −29.7%.
**Supply.** ≈122M ETH; 44.13M (36.1%) staked (Oct 19); exit queue spiked to ≈851K ETH on Oct 2 (MetaMask validator exits), 786K on Oct 5; entry queue 1.16M ETH / 20-day wait on Oct 20. No unlocks.
**Flows / derivatives.** Spot ETH ETFs > −$1.08B in October through Oct 20 (largest monthly outflow of 2026), a record 23-day outflow streak (Glassnode) still running Oct 30 (−$155.85M); ETHA produced 93% of Oct 12–16 outflows; cumulative net inflows down to $12.29B; 62% of 2026 inflows returned by Oct 19. OI $23.88B (−10–12% m/m); funding 0.0042%.
**Fees / value accrual.** Chain-level fees across all Ethereum protocols $58.07M/24h on Oct 30 (DefiLlama; includes builders) — protocol-only burn/priority-fee figures not sourced. Accrual: staking real yield (priority fees + EIP-1559 burn), magnitude not sourced.
**Confirmed catalysts.** Glamsterdam Hoodi testnet activation postponed (Oct 22); no mainnet date — a *negative* confirmed catalyst.
**Speculative catalysts.** Glamsterdam mainnet in 2027; ETF flow reversal; staking inside ETFs.
**Bear case.** (1) A record outflow streak led by the largest, stickiest holder base (ETHA) is institutional distribution, not retail panic. (2) The roadmap slipped: no Q4 upgrade catalyst. (3) ETH trails BTC by 9 points over 12 months and lost its 200d while BTC holds its own — a relative-weakness regime. (4) MetaMask exits and restaking unbonding push liquid supply up. (5) The highest-beta major in a Fed hiking cycle with the 10-year at 4.93%.
**Invalidation (for the long).** Daily close < $1,860 (Jul 31 level); outflow streak > 30 days; BTC closes below its 200d ($72.9k).
**Regime fit:** mean_reverting only. **Data quality: A.**

---

## Sources

Shared macro/regime sources are listed in market_regime.md. Setup-specific:

**HYPE** — Coinbase HYPE (Oct 31, 2026) https://www.coinbase.com/price/hyperliquid · CoinMarketCap HYPE https://coinmarketcap.com/currencies/hyperliquid/ · CoinGecko HYPE https://www.coingecko.com/en/coins/hyperliquid · AltcoinBuzz (Oct 28, 2026) $63.94M holder revenue https://www.altcoinbuzz.io/hyperliquid-hype-holder-revenue-buybacks · ETHNews (Oct 2026) Tokenomist $1.72B Nov unlocks https://ethnews.com/1-72b-in-november-token-unlocks-puts-hyperliquid-in-focus/ · DefiLlama HYPE unlocks https://defillama.com/unlocks/hyperliquid · CoinLaunch vesting https://coinlaunch.space/projects/hyperliquid/vesting/ · tokenomics.com HYPE unlocks https://app.tokenomics.com/tokenomics/hyperliquid/unlocks · ad-hoc-news/ForkLog (Oct 29, 2026) 49.87M burned https://www.ad-hoc-news.de/boerse/news/marktberichte/hyperliquid-price-jumps-as-burn-meets-november-unlock/70432878 · The Traders Spread (Aug 30, 2026) staking/Foundation share https://thetradersspread.com/crypto/hyperliquid-hype-climbs-52-percent-30-days-cftc-onshoring-push · Castle Labs (Oct 26, 2026) https://castlelabs.io/research/perp-dex-rankings · Datawallet statistics (2026) https://www.datawallet.com/crypto/hyperliquid-statistics · AlphaWire treasuries https://alphawire.xyz/deepdive/crypto-ecosystem/hyperliquid-hype-financialization-treasury-unlocks/ · AMINA ETF note (2026) https://aminagroup.com/research/hyperliquid-hype-etf-buyback-staking-yield-institutional-access-2026/ · Crypto Briefing (Aug 2026) 58% share https://cryptobriefing.com/hyperliquid-perp-dex-market-share/

**MON** — DefiLlama Monad unlocks (Oct 31, 2026) https://defillama.com/unlocks/monad · DefiLlama Monad chain (Oct 31, 2026) https://defillama.com/chain/monad · CoinGecko MON https://www.coingecko.com/en/coins/monad · TradingView/CoinMarketCal Nov 24 unlock https://www.tradingview.com/news/coinmarketcal:91bdeaee6094b:0-monad-mon-16-8b-token-unlock-24-nov-2026/ · tokenomics.com MON https://app.tokenomics.com/tokenomics/monad/unlocks · CryptoTicker 10.7B unlock https://cryptoticker.io/en/monad-mon-unlock-lock-up-expiry/ · Pluang $280M team unlock https://pluang.com/en/news-feed/monad-buka-kunci-nov-24-10-7-miliar-mon-terbebas · CMC AI MON updates (mid-Oct 2026) https://coinmarketcap.com/cmc-ai/monad/latest-updates/ · Sharpe MON funding/venues https://www.sharpe.ai/funding-rates/monad · CoinCentral (Oct 8, 2025) Hyperliquid MON hyperp https://coincentral.com/hyperliquid-lists-3x-leveraged-mon-perpetuals-before-token-launch/

**TIA** — Cointelegraph 175.75M TIA unlock https://cointelegraph.com/news/tia-token-release-october-crypto-unlocks-2-billion · CMC AI TIA (Oct 30, 2026) https://coinmarketcap.com/cmc-ai/celestia/price-analysis/ · Coinbase TIA (Oct 22, 2026) https://www.coinbase.com/price/celestia · CoinStats TIA news (Oct 9, 2026) https://coinstats.app/ai/a/latest-news-for-celestia · Bitget TIA (~Oct 17, 2026) https://www.bitget.com/price/celestia · DropsTab TIA https://dropstab.com/coins/celestia · Sharpe TIA futures https://www.sharpe.ai/futures/tia · Buildix Hyperliquid TIA https://www.buildix.trade/hyperliquid/TIA · KuCoin Celestia issuance (CIP-29/41) https://www.kucoin.com/blog/celestia-tia-price-prediction-2026-2030-modular-future · CoinGecko TIA (fees, unverified) https://www.coingecko.com/en/coins/celestia

**ZEC** — Coinbase ZEC (Oct 30–31, 2026) https://www.coinbase.com/price/zcash · Crypto Times (Oct 23, 2026) https://www.cryptotimes.io/2026/10/23/zcash-zec-price-holds-above-1200-with-growing-etf-attention/ · Coin Metrics SOTN #338 https://coinmetrics.substack.com/p/state-of-the-network-issue-338 · crypto.news shielded pool https://crypto.news/why-30-of-zcash-supply-is-now-in-the-shielded-pool/ · Coinalyze ZEC OI https://coinalyze.net/zcash/open-interest/ · Buildix Hyperliquid ZEC https://www.buildix.trade/hyperliquid/ZEC · Sharpe ZEC futures https://www.sharpe.ai/futures/zec · Yahoo/Grayscale ETF push https://finance.yahoo.com/markets/crypto/articles/zcash-hits-eight-high-grayscale-013700467.html · Crypto.com (Oct 2026) +178.6% YTD https://crypto.com/us/market-updates/best-crypto-to-watch-october-2026

**ENA** — CoinPaprika ENA (Oct 31, 2026) https://coinpaprika.com/coin/ena-ethena · DefiLlama Ethena https://defillama.com/protocol/ethena · CryptoTicker Nov 1 unlock / $7.5B gate https://cryptoticker.io/en/ena-unlock-november-1-buyback-threshold/ · CryptoTicker fee switch https://cryptoticker.io/en/ethena-fee-switch-ena-buyback/ · CryptoTicker Oct 5 unlock https://cryptoticker.io/en/ethena-ena-investor-unlock/ · Crypto Briefing (Oct 29, 2026) $800M rewards, Q3 yield https://cryptobriefing.com/ethena-800m-usde-rewards-fee-switch/ · CoinStats ENA news (Oct 2026) https://coinstats.app/ai/a/latest-news-for-ethena · Sharpe ENA futures https://www.sharpe.ai/futures/ena · Buildix Hyperliquid ENA https://www.buildix.trade/hyperliquid/ENA

**UNI** — DefiLlama Uniswap (Oct 29–31, 2026) https://defillama.com/protocol/uniswap · Etherscan UNI (Oct 30, 2026) https://etherscan.io/token/0x1f9840a85d5af5bf1d1762f925bdaddc4201f984 · MetaMask UNI (Oct 30, 2026) https://metamask.com/price/uniswap · Coinbase UNI https://www.coinbase.com/price/uniswap · Bybit UNI supply https://www.bybit.com/en/price/uniswap/ · CryptoTicker Firepit burn (Oct 2026) https://cryptoticker.io/en/uniswap-firepit-burn-uni-price/ · CoinPerps Uniswap statistics https://www.coinperps.com/learn/uniswap-statistics · DL News fee switch https://www.dlnews.com/articles/defi/uniswap-dao-to-activate-fee-switch-and-burn-100m-uni-tokens/ · ETHNews Sept $14.81M https://ethnews.com/uniswap-revenue-reaches-14-81m-as-fee-switch-reshapes-defi/ · Buildix Hyperliquid UNI https://www.buildix.trade/hyperliquid/UNI · Sharpe UNI OI https://www.sharpe.ai/futures/uniswap/open-interest

**PUMP** — OKX PUMP (Oct 31, 2026) https://www.okx.com/en-us/price/pump-fun-pump · Coinbase PUMP https://www.coinbase.com/price/pump-fun-solana · pump.fun token page https://pump.fun/pump-token · CryptoTicker revenue −16% https://cryptoticker.io/en/pump-fun-revenue-buyback/ · Pluang revenue/buyback https://pluang.com/en/news-feed/pump-fun-pendapatan-turun-16-persen-harga-rendah · CMC AI PUMP updates (Oct 2026; Binance collateral ratio) https://coinmarketcap.com/cmc-ai/pump-fun/latest-updates/ · ETHNews Nov unlocks (PUMP Nov 14) https://ethnews.com/1-72b-in-november-token-unlocks-puts-hyperliquid-in-focus/ · Bankless perps https://www.bankless.com/read/news/pump-fun-adds-leveraged-perps-via-hyperliquid · Lookonchain perps/unlock claim https://www.lookonchain.com/feeds/79199 · crypto.news July unlock https://crypto.news/pump-fun-127m-unlock-fair-launch-stress-test/ · CoinGecko PUMP (Hyperliquid OI) https://www.coingecko.com/en/coins/pump-fun · FinanceFeeds Solana app revenue https://financefeeds.com/solana-dominates-as-onchain-app-revenue/ · news.bitcoin.com buyback model https://news.bitcoin.com/pump-fun-burns-370-million-pump-tokens-revenue-buyback/

**AAVE** — DefiLlama Aave (Oct 30, 2026) https://defillama.com/protocol/aave · KuCoin Aavenomics 3.0 (Oct 22, 2026) https://www.kucoin.com/news/flash/aave-founder-announces-aavenomics-3-0-to-include-aave-buybacks-staking-and-burning · TokenLogic pause thread https://x.com/Token_Logic/status/2048761794001617286 · Aave governance ARFC https://governance.aave.com/t/arfc-pause-aave-buybacks/24686 · tokenbuybacks.app AAVE https://www.tokenbuybacks.app/token/aave/ · The Defiant Aavenomics 3.0 https://thedefiant.io/news/defi/aave-confirms-aavenomics-3-0-live-buybacks-dao-spending-cut · BloFin (Oct 29, 2026) https://blofin.com/en/academy/education/aave/aave-buyback-and-aavenomics · Gate (Oct 24, 2026) https://www.gate.com/news/detail/AAVE/aave-drops-6-below-140-support-despite-aave-v4-lending-expansion-25178383 · Coingabbar (Oct 30, 2026) $126.05 https://www.coingabbar.com/en/aave-price-prediction-can-aave-clear-129-50-after-monad-tvl-news · TechFlow livestream https://www.techflowpost.com/en-US/newsletter/141045 · DL News buyback programme https://www.dlnews.com/articles/defi/aave-eyes-permanent-50m-buyback-programme/ · Sharpe AAVE futures https://www.sharpe.ai/futures/aave · Lines prediction market https://www.lines.com/prediction-markets/crypto/will-the-aave-tvl-go-below-10b-in-2026

**ETH** — Benzinga (Oct 31, 2026) https://www.benzinga.com/crypto/26/10/62763699/ethereum-flashes-a-risky-pattern-as-eth-etfs-post-their-worst-monthly-outflows-on-record · Market Posture Daily https://marketpd.com/asset-eth · Yahoo (Jul 31, 2026) https://finance.yahoo.com/personal-finance/investing/article/bitcoin-and-ethereum-prices-today-friday-july-31-2026-crypto-prices-back-off-this-morning-130657761.html · TokenPost (Oct 31, 2026) https://www.tokenpost.com/news/investing/45341 · Crypto Briefing streak https://cryptobriefing.com/us-spot-ethereum-etfs-record-outflow-streak/ · Cointelegraph https://cointelegraph.com/markets/ether-etf-outflows-near-1-1-billion-october · 24/7 Wall St (Oct 19, 2026) https://247wallst.com/investing/cryptocurrency/2026/10/19/ether-etfs-have-given-back-62-of-this-years-inflows-is-wall-street-giving-up-on-ethereum/ · Stakefish (Oct 19, 2026) https://blog.stake.fish/who-stakes-eth-2026/ · CoinDesk (Oct 5, 2026) https://www.coindesk.com/tech/2026/10/05/ethereum-has-a-25-day-wait-to-start-staking-with-nearly-1-5-million-eth-in-line · CryptoTicker (Oct 20, 2026) https://cryptoticker.io/en/ethereum-price-prediction-staking-queue-20-days/ · KuCoin Glamsterdam https://www.kucoin.com/news/flash/ethereum-s-glamsterdam-upgrade-activated-on-hoodi-testnet-mainnet-date-undetermined · LiveBitcoinNews ETH OI/funding https://www.livebitcoinnews.com/ether-funding-rates-slip-under-0005/ · DefiLlama Ethereum fees (Oct 30, 2026) https://defillama.com/fees/chain/ethereum
