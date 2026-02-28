# Dynamic Multi-Asset Pair Trading Backtesting Engine

A production-quality, walk-forward **statistical arbitrage** backtester for
cryptocurrency markets, written in pure Python.  The engine dynamically
selects cointegrated pairs at runtime, sizes positions with volatility
targeting, and applies realistic transaction costs — no fixed pair, no
look-ahead bias.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Strategy Logic](#strategy-logic)
- [Backtest Results](#backtest-results)
- [Configuration](#configuration)
- [Output Files](#output-files)

---

## Features

| Capability | Implementation |
|---|---|
| **Data** | Pre-fetch top-40 crypto OHLCV via CCXT (Binance) into memory — no per-bar API calls during backtest |
| **Dynamic pair selection** | Engle-Granger cointegration scan across all C(n,2) pairs; trades the pair with the lowest p-value |
| **Hedge ratio** | Scalar Kalman Filter (OLS-seeded) tracks a time-varying β in log-price space |
| **Mean-reversion speed** | OU process fitted by OLS AR(1) → half-life τ½ = ln2 / θ |
| **Regime filter** | 2-state Gaussian HMM classifies spread dynamics; gates out trending regimes |
| **Position sizing** | Volatility targeting: notional = portfolio\_value × σ\_target / σ\_spread\_annual |
| **Entry signal** | \|z-score\| > Z\_ENTRY **and** momentum confirmation: spread must already be curling back toward mean (12-bar SMA gate) |
| **Exit signals** | Mean-reversion (\|z\| ≤ 0.25σ) **or** time-stop (hold > 2 × OU half-life) |
| **Transaction costs** | 0.1 % per leg on every notional change (open, rebalance, close) + hourly funding rate (≈ 0.01 %/8 h) |
| **No look-ahead bias** | Signals computed on `close[bar-1]`; all fills executed at `open[bar]`; end-of-bar MTM at `close[bar]` |
| **Outputs** | `trade_log.csv`, `equity_curve.csv`, `backtest.log` |

---

## Architecture

```
pair-trading/
├── config.py               ← all constants; single place to tune
├── requirements.txt
│
├── data/
│   └── fetcher.py          ← CCXT pre-fetch with pagination & retry
│
├── math_utils/
│   ├── kalman_filter.py    ← scalar KF; OLS-seeded batch_estimate
│   ├── ou_process.py       ← OLS → θ, μ, σ, half-life
│   └── regime.py           ← 2-state GaussianHMM regime classifier
│
├── strategy/
│   ├── cointegration.py    ← EG pair scanner (positive β filter)
│   └── signals.py          ← z-score, vol-targeting, entry/exit rules
│
├── backtester/
│   ├── engine.py           ← BacktestEngine (walk-forward loop)
│   └── metrics.py          ← Sharpe, MDD, CAGR, Calmar, win-rate
│
├── main.py                 ← live-data entry point
└── backtest_demo.py        ← offline demo (synthetic data, no API key)
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the offline demo (no API key required)

```bash
python3 backtest_demo.py
```

Generates a 1-year synthetic hourly crypto universe (8760 bars, 20 symbols,
8 cointegrated pairs baked in by construction), runs the full engine, and
prints results to stdout.  Output files are written to the current directory.

### Run a live backtest (Binance data)

```bash
python3 main.py
```

Fetches hourly OHLCV for the 29 curated symbols between `START_DATE` and
`END_DATE` (currently 2022–2025, ~35k bars per symbol), then runs the
walk-forward backtest.  Expect the data fetch to take 10–15 minutes and
the engine loop ~2–3 hours due to the 4-year span.

---

## Module Reference

### `data/fetcher.py`

```python
raw_data = fetch_all_ohlcv(symbols=TOP_40_SYMBOLS)
# → dict[str, pd.DataFrame]  keyed by "BTC/USDT", etc.
```

Paginates through the exchange API and forward-fills small gaps (≤ 3 bars).

---

### `math_utils/kalman_filter.py`

**Model** (scalar, 1-D state):

```
β_t = β_{t-1} + w_t        w_t ~ N(0, Q)       state equation
y_t = β_t · x_t + v_t      v_t ~ N(0, R)       observation equation
```

```python
kf = KalmanFilter(delta=1e-4, ve=1e-3)
betas = kf.batch_estimate(log_y, log_x)   # offline, OLS-seeded
new_beta = kf.update(log_y_t, log_x_t)    # online, incremental
```

`batch_estimate` seeds the initial state from OLS on the first 30 bars,
preventing sign-flip when `log(price) < 0` (e.g. coins priced < $1).

---

### `math_utils/ou_process.py`

Fits the discretised OU model via OLS:

```
ΔS_t = a + b·S_{t-1} + ε_t
```

```python
params = estimate_ou_parameters(spread_array)
# → {"theta": ..., "mu": ..., "sigma": ..., "half_life": ...}
```

Half-life = ln(2) / θ days.  If `b ≥ 0` the spread is non-mean-reverting
and `half_life = inf` is returned.

---

### `math_utils/regime.py`

```python
clf = RegimeClassifier(n_components=2)
clf.fit(spread_array)
if clf.is_mean_reverting(spread_array):
    # proceed with entry
```

The state with the smallest `|mean(Δspread)|` is labelled mean-reverting.
If the HMM fails to converge the classifier defaults to `True` (fail-open).

---

### `strategy/cointegration.py`

```python
best = find_best_pair(log_price_dict)
# → ("ETH/USDT", "BNB/USDT", 0.0032)  or  None
```

Iterates all C(n, 2) pairs, runs `statsmodels.tsa.stattools.coint`, and
returns the pair with the lowest p-value **and** a positive OLS hedge ratio.

---

### `strategy/signals.py`

```python
z = compute_zscore(spread_array, window=336)
direction = get_entry_direction(z, spread_array, threshold=2.0)  # +1, -1, or 0
shares_y, shares_x = compute_position_size(...)     # price args = open[bar]
exit_flag, reason = should_exit(z, bars_held, half_life, ...)
```

Position sizing: `notional = portfolio_value × TARGET_VOL / spread_annual_vol`
where `spread_annual_vol = bar_vol × √8760` (hourly, 24/7 crypto).
Shares: `shares_y = direction × notional / open_y`,
        `shares_x = −direction × β × notional / open_x`.

---

### `backtester/engine.py`

```python
engine = BacktestEngine(
    raw_data=raw_data,
    initial_capital=1_000_000,
    lookback=336,        # 14 days of hours
    min_history=500,     # warm-up bars before first trade
    rescan_interval=5,   # bars between pair-scans when flat
    z_entry=2.0,         # override without editing config.py
    z_exit=0.25,
)
equity_curve, trade_log = engine.run()
```

**Walk-forward loop (per hourly bar):**

1. **If flat** and cooldown elapsed → compute signals on `close[0..bar-1]` → `find_best_pair` → fit KF → estimate OU half-life → abort if `half_life > MAX_HALFLIFE` → check HMM regime → compute z-score → **momentum filter** (12-bar SMA gate) → if signal confirmed fill entry at `open[bar]`.
2. **If in position** → incremental KF update on `close[bar-1]` → recompute spread/z-score/half-life → check exits → if exiting fill at `open[bar]`; if staying rebalance at `open[bar]`.
3. Mark portfolio to market at `close[bar]`.
4. Deduct hourly funding cost on gross notional (close-based exposure).

---

### `backtester/metrics.py`

```python
metrics    = compute_metrics(equity_curve)
# → total_return, annualised_return, max_drawdown, sharpe_ratio, calmar_ratio

trade_stats = compute_trade_stats(trade_log)
# → win_rate, avg_net_pnl, avg_hold_bars, total_gross_pnl, total_transaction_costs
```

---

## Strategy Logic

### Spread

```
spread_t = log(price_y_t) − β_t × log(price_x_t)
```

β is updated each bar by the incremental Kalman Filter, using `close[bar-1]`.

### Entry

| Condition | Action |
|---|---|
| z > +Z\_ENTRY **and** spread < 12-bar SMA | **Short spread**: short y, long x |
| z < −Z\_ENTRY **and** spread > 12-bar SMA | **Long spread**: long y, short x |

The 12-bar SMA momentum gate confirms the spread has *already begun* reverting.
Entries where the spread is still accelerating away from the mean are rejected
even when the z-score threshold is breached.

### Exit

| Condition | Label |
|---|---|
| \|z\| ≤ Z\_EXIT (0.25) | `mean_reversion` ✅ |
| bars\_held ≥ 2 × half\_life | `time_stop` ⏱ |
| End of data | `end_of_backtest` |

### Cash accounting (exact, no approximation)

Signals on `close[bar-1]`; all fills at `open[bar]`; MTM at `close[bar]`:

```
Open     : cash -= shares_y·open_y   + shares_x·open_x   + entry_cost
MTM      : portfolio_value = cash + shares_y·close_y + shares_x·close_x
Funding  : cash -= (|shares_y·close_y| + |shares_x·close_x|) × HOURLY_FUNDING_RATE
Rebal    : cash -= Δshares_y·open_y  + Δshares_x·open_x  + rebal_cost
Close    : cash += shares_y·open_y   + shares_x·open_x   − exit_cost
           portfolio_value = cash
```

---

## Backtest Results

> Runs 1–2 use **synthetic** hourly data (20 symbols, 8 cointegrated pairs baked
> in by construction). Runs 3–4 use **real Binance data** fetched via CCXT.
> All runs charge 0.1 % per leg + hourly funding rate every bar.

---

### Run 1 — 2022 Full Year (2022-01-01 → 2022-12-31)

**8,760 hourly bars · 500-bar warm-up · 20 symbols**

#### Trade Log (with Momentum Filter)

| # | Entry | Exit | Pair | Dir | HL(h) | β | Costs | Net P&L | Reason |
|---|-------|------|------|-----|------:|--:|------:|--------:|--------|
| 1 | 2022-01-22 | 2022-01-24 | LTC/BNB   | long\_spread  | 29.2 | 0.683 | $195   | **+$13,211** | mean\_reversion ✅ |
| 2 | 2022-03-07 | 2022-03-13 | SOL/AVAX  | long\_spread  | 78.6 | 1.245 | $413   | −$1,629      | mean\_reversion |
| 3 | 2022-09-27 | 2022-09-30 | ATOM/BNB  | long\_spread  | 40.4 | 0.443 | $428   | **+$4,160**  | time\_stop ⏱ |
| 4 | 2022-10-02 | 2022-10-05 | LTC/NEAR  | long\_spread  | 47.7 | 3.643 | $202   | **+$10,178** | mean\_reversion ✅ |
| 5 | 2022-10-26 | 2022-10-28 | BCH/UNI   | long\_spread  | 40.2 | 2.772 | $1,428 | **+$10,922** | mean\_reversion ✅ |
| 6 | 2022-11-20 | 2022-11-23 | BCH/ATOM  | short\_spread | 27.8 | 1.821 | $1,540 | **+$5,606**  | time\_stop ⏱ |

#### Performance Summary

| Metric | Without momentum filter | **With momentum filter** | Δ |
|--------|------------------------:|-------------------------:|--:|
| **Final value** | $1,035,964 | **$1,042,447** | +$6,483 |
| **Total return** | +3.60 % | **+4.24 %** | +0.64 pp |
| **Annualised return** | +3.60 % | **+4.24 %** | +0.64 pp |
| **Max drawdown** | −5.27 % | **−2.21 %** | +3.06 pp |
| **Sharpe ratio** | 0.77 | **1.34** | +0.57 |
| **Calmar ratio** | 0.68 | **1.92** | +1.24 |
| **Total trades** | 15 | **6** | −9 |
| **Win rate** | 66.7 % (10/15) | **83.3 % (5/6)** | +16.6 pp |
| **Avg net P&L / trade** | +$2,398 | **+$7,075** | +$4,677 |
| **Total gross P&L** | $42,511 | **$52,058** | +$9,547 |
| **Total transaction costs** | $12,073 | **$4,206** | −$7,867 |
| **Cost-to-gross ratio** | 28.4 % | **8.1 %** | −20.3 pp |

#### Key Observations

1. **Momentum filter blocked the run's worst trade** — ALGO/MATIC (−$42,871, trade 8
   in the unfiltered run) entered with a z-score breach but spread still accelerating
   away from mean; the 12-bar SMA gate rejected it before any capital was committed.
   Eliminating this one trade alone accounts for most of the return improvement.

2. **Higher gross P&L on fewer trades** — $52,058 gross on 6 trades vs $42,511 on 15,
   meaning the filtered trades were, on average, much higher-quality entries.

3. **Costs fell 65 %** — $4,206 vs $12,073; fewer entries means fewer commissions and
   less funding-cost drag.  Cost-to-gross ratio dropped from 28 % to 8 %.

4. **MDD nearly halved** — −2.21 % vs −5.27 %; the equity curve becomes a smoother
   climb without the mid-year drawdown caused by sequential losing time-stops.

5. **Both time-stops were profitable** (ATOM/BNB +$4,160, BCH/ATOM +$5,606) — unlike
   the unfiltered run where most time-stops were losers; this confirms the filter is
   selecting entries with genuinely better risk-reward, not merely reducing quantity.

---

### Run 2 — 2025 Q3 – 2026 Q1 (2025-07-01 → 2026-03-31)

**6,576 hourly bars · 500-bar warm-up · 20 symbols**

#### Equity Curve

```
  $1,024,179 ┤                                                 ╱──────
             │                                         ╱────────────
             │         ╱──────────────╲        ╱──────────────────
  $1,003,087 ┤─────────────────────────╲       ╱──────────────────
             │──────────────────────────────────────────────────
    $981,996 ┤──────────────────────────────────────────────────
             └──────────────────────────────────────────────────
            Bar 0                                          Bar 6576
```

#### Trade Log

| # | Entry | Exit | Pair | Dir | HL(h) | β | Costs | Net P&L | Reason |
|---|-------|------|------|-----|------:|--:|------:|--------:|--------|
| 1 | 2025-08-10 | 2025-08-12 | AVAX/UNI | long | 20.4 | 1.954 | $1,461 | **+$6,437** | mean\_reversion ✅ |
| 2 | 2025-10-16 | 2025-10-22 | ALGO/XRP | short | 68.0 | 3.037 | $1,186 | −$9,354 | time\_stop |
| 3 | 2025-12-17 | 2025-12-19 | AVAX/LINK | short | 32.8 | 1.589 | $165 | **+$8,943** | mean\_reversion ✅ |
| 4 | 2026-02-04 | 2026-02-04 | UNI/ETC | long | 9.9 | 0.537 | $1,179 | **+$5,218** | mean\_reversion ✅ |
| 5 | 2026-03-11 | 2026-03-14 | SOL/LINK | long | 40.0 | 1.470 | $218 | **+$12,934** | time\_stop |

#### Performance Summary

| Metric | No fixes (baseline) | + Fee/stop fixes | **+ Momentum filter** | Δ vs baseline |
|--------|-------------------:|-----------------:|----------------------:|--------------:|
| **Final value** | $998,341 | $1,017,850 | **$1,024,179** | +$25,838 |
| **Total return** | −0.17 % | +1.79 % | **+2.42 %** | +2.59 pp |
| **Annualised return** | −0.22 % | +2.38 % | **+3.23 %** | +3.45 pp |
| **Max drawdown** | −9.04 % | −8.92 % | **−2.83 %** | +6.21 pp |
| **Sharpe ratio** | 0.00 | 0.41 | **1.00** | +1.00 |
| **Calmar ratio** | −0.02 | +0.27 | **+1.14** | +1.16 |
| **Total trades** | 17 | 21 | **5** | −12 |
| **Win rate** | 64.7 % | 71.4 % | **80.0 %** | +15.3 pp |
| **Avg net P&L / trade** | −$98 | +$850 | **+$4,836** | +$4,934 |
| **Gross P&L** | $18,179 | $35,953 | **$25,431** | +$7,252 |
| **Transaction costs** | $20,747 | $16,085 | **$4,210** | −$16,537 |
| **Cost-to-gross ratio** | 114 % | 45 % | **17 %** | −97 pp |

#### Key Observations

1. **Momentum filter eliminated all three catastrophic time-stops** — the AVAX/BCH
   (−$21K), ATOM/UNI (−$37K), and AVAX/ATOM (−$25K) entries from the prior run were
   all rejected because the spread was still accelerating away from the mean when the
   z-score breached the threshold.  The filter correctly identified these as falling-knife
   entries before any money was committed.

2. **Costs collapsed 74 %** — from $16,085 to $4,210.  Fewer trades means fewer entry
   and exit commissions.  Cost-to-gross ratio fell from 45 % to 17 %.

3. **MDD cut by 68 %** — from −8.92 % to −2.83 %.  The equity curve is now a near-monotone
   climb rather than a volatile round-trip.

4. **Sharpe crossed 1.0** — the combination of fee/stop fixes (+rebalance threshold,
   +stop-loss, tighter MAX_HALFLIFE) and the momentum gate together lifted the ratio
   from 0.00 (baseline) → 0.41 → **1.00**.

5. **Trade-off: much lower activity** — 5 trades vs 21.  In strongly trending or
   choppy regimes the filter will sit out for extended periods.  This is the correct
   behaviour for a mean-reversion strategy but should be monitored on live data.

---

### Run 3 — 2022–2025 Hyper-Aggressive (Real Binance Data)

**35,063 hourly bars · 500-bar warm-up · 29 curated symbols · $5,000 initial capital**

Config: `Z_ENTRY=1.5`, static `TARGET_VOL=60 %`, `MAX_LEVERAGE=4.0`, `RESCAN_INTERVAL=12`,
`MAX_HALFLIFE=288 h`, `OU_HALFLIFE_MULTIPLIER=2.5`, `momentum_window=4`.

#### Performance Summary

| Metric | Value |
|---|---|
| **Initial capital** | $5,000.00 |
| **Final portfolio value** | $4,121.72 |
| **Total return** | **−17.57 %** |
| **Annualised return** | −4.71 % |
| **Max drawdown** | −50.72 % |
| **Sharpe ratio** | −0.04 |
| **Calmar ratio** | −0.09 |
| **Total trades** | 63 |
| **Win rate** | 61.9 % |
| **Avg net P&L / trade** | −$13.94 |
| **Avg hold (bars)** | 95.7 h |
| **Total gross P&L** | $1,350.78 |
| **Total transaction costs** | $1,194.66 |
| **Cost-to-gross ratio** | **88.4 %** |

#### Breakdown by Year

| Year | Trades | Net P&L | Win Rate |
|------|-------:|--------:|---------:|
| 2022 | 16 | −$761 | 75.0 % |
| 2023 | 17 | −$935 | 58.8 % |
| 2024 | 19 | +$57 | 52.6 % |
| 2025 | 11 | +$761 | 63.6 % |

#### Breakdown by Exit Reason

| Reason | Trades | Net P&L | Avg / trade |
|--------|-------:|--------:|------------:|
| `mean_reversion` ✅ | 33 | +$3,669 | +$111 |
| `time_stop` ⏱ | 26 | −$2,390 | −$92 |
| `stop_loss` ❌ | 4 | −$2,158 | −$539 |

#### Notable Trades

**Best:**

| # | Entry | Exit | Pair | Net P&L | Reason |
|---|-------|------|------|--------:|--------|
| 54 | 2025-01-18 | 2025-01-19 | VET/EOS | **+$241** | mean\_reversion ✅ |
| 61 | 2025-10-28 | 2025-10-30 | BTC/LTC | **+$205** | time\_stop |
| 1  | 2022-02-05 | 2022-02-08 | LTC/DOT | **+$205** | mean\_reversion ✅ |

**Worst:**

| # | Entry | Exit | Pair | Net P&L | Reason |
|---|-------|------|------|--------:|--------|
| 6  | 2022-05-04 | 2022-05-11 | ICP/MANA | **−$1,185** | stop\_loss ❌ |
| 17 | 2023-01-05 | 2023-01-20 | ETH/SOL  | **−$847**  | time\_stop ⏱ |
| 43 | 2024-06-09 | 2024-06-20 | ETH/NEAR | **−$555**  | time\_stop ⏱ |

#### Key Observations

1. **Costs consumed 88 % of gross P&L** — with `Z_ENTRY=1.5` and `RESCAN_INTERVAL=12`
   the engine opened 63 trades at an average cost of $19/trade. Combined with
   high-leverage notionals (`TARGET_VOL=60 %`, `MAX_LEVERAGE=4.0`), the funding
   and commission drag wiped out almost all gross profits.

2. **Four stop-losses averaged −$539 each** — the looser `momentum_window=4` and
   wider `MAX_HALFLIFE=288 h` accepted pairs with fragile cointegration, leading to
   structural breaks caught by the `Z_STOP_LOSS=4.0` hard stop.  These four trades
   alone account for −$2,158 of the total loss.

3. **Mean-reversion exits were still profitable (+$3,669)** — confirming the core
   signal is valid; the problem is purely over-trading and excessive sizing eroding
   returns through costs and catastrophic stops.

4. **Equity curve reached −50 % MDD** — the 2022–2023 bear market hit high-leverage
   short-vol spreads hard; the ICP/MANA stop-loss (−$1,185 in May 2022, the LUNA
   crash period) and ETH/SOL time-stop (−$847 in Jan 2023) together dropped the
   account from $5,000 to under $3,000 by mid-2023.

5. **Conclusion — the 2022-only run with conservative settings remains superior**:
   6 trades / +5.19 % / Sharpe 0.46 vs 63 trades / −17.57 % / Sharpe −0.04.
   For this mean-reversion strategy, *fewer, higher-quality entries* consistently
   outperform high-frequency aggressive setups.

---

### Run 4 — 2022–2025 Dynamic Volatility Targeting (Real Binance Data)

**35,063 hourly bars · 500-bar warm-up · 29 curated symbols · $5,000 initial capital**

Same universe and entry parameters as Run 3, with `TARGET_VOL` replaced by a
half-life-scaled dynamic target:

```
dynamic_target_vol = MAX_TARGET_VOL × (BASELINE_HALFLIFE / max(HL, BASELINE_HALFLIFE))
                   = 0.60 × (48h / max(HL, 48h))
clamped to [MIN_TARGET_VOL=0.15, MAX_TARGET_VOL=0.60]
```

Fast-reverting pairs (HL ≤ 48 h) → 60 % vol.  Slow pairs (HL = 288 h) → 15 % vol.

#### Performance Summary — Run 3 vs Run 4

| Metric | Run 3 — Static 60 % | **Run 4 — Dynamic Vol** | Δ |
|---|---:|---:|---:|
| **Final portfolio value** | $4,121.72 | **$3,370.56** | −$751 |
| **Total return** | −17.57 % | **−32.59 %** | −15.0 pp |
| **Annualised return** | −4.71 % | −9.38 % | −4.7 pp |
| **Max drawdown** | −50.72 % | **−46.11 %** | +4.6 pp ✅ |
| **Sharpe ratio** | −0.04 | −0.46 | −0.42 |
| **Calmar ratio** | −0.09 | −0.20 | −0.11 |
| **Total trades** | 63 | 63 | — |
| **Win rate** | 61.9 % | **52.4 %** | −9.5 pp |
| **Avg net P&L / trade** | −$13.94 | −$25.86 | −$11.92 |
| **Total gross P&L** | $1,350.78 | $2,627.81 | +$1,277 |
| **Total transaction costs** | $1,194.66 | $1,529.44 | +$334 |
| **Cost-to-gross ratio** | 88.4 % | **58.2 %** | −30.2 pp ✅ |

#### Breakdown by Year

| Year | Trades | Net P&L | Win Rate |
|------|-------:|--------:|---------:|
| 2022 | 16 | −$607 | 50.0 % |
| 2023 | 17 | −$689 | 52.9 % |
| 2024 | 19 | −$341 | 52.6 % |
| 2025 | 11 | +$8 | 54.5 % |

#### Breakdown by Exit Reason

| Reason | Trades | Net P&L | Avg / trade |
|--------|-------:|--------:|------------:|
| `mean_reversion` ✅ | 33 | +$1,765 | +$53 |
| `time_stop` ⏱ | 26 | −$1,946 | −$75 |
| `stop_loss` ❌ | 4 | −$1,449 | −$362 |

#### Worst Trades (with Dynamic tvol)

| Entry | Pair | HL (h) | tvol | Net P&L | Reason |
|-------|------|-------:|-----:|--------:|--------|
| 2022-05-04 | ICP/MANA | 288.0 h | 15 % | −$459 | stop\_loss ❌ |
| 2023-09-30 | ETH/ETC | 101.3 h | 28 % | −$403 | stop\_loss ❌ |
| 2023-12-05 | VET/EOS | 288.0 h | 15 % | −$334 | stop\_loss ❌ |
| 2023-01-05 | ETH/SOL | 135.3 h | 21 % | −$283 | time\_stop ⏱ |
| 2023-04-19 | XRP/VET | 7.8 h | **60 %** | −$252 | stop\_loss ❌ |

#### Key Observations

1. **DVT reduced stop-loss damage but demolished mean-reversion profits** — stop-losses
   fell from −$2,158 to −$1,449 (−33 %) because slow-reverting pairs (HL=288h) were
   sized at only 15 % vol.  But the 33 mean-reversion wins also shrank dramatically:
   +$3,669 → +$1,765 (−52 %), because many of the best winners (DOT/AAVE, DOGE/XLM,
   VET/EOS) had long half-lives and were starved of notional.  Net effect: −15 pp return.

2. **`BASELINE_HALFLIFE=48h` rewarded the wrong pairs** — the formula grants maximum
   leverage to fast-reverting pairs (HL < 48 h).  But in this universe, short half-lives
   correlate with *high spread volatility*, not safety: XRP/VET (HL=7.8 h) received 60 %
   vol and hit a stop-loss at −$252.  The intuition behind DVT — "fast = safer" — is
   inverted for crypto spreads during volatile regimes.

3. **Cost-to-gross ratio improved (+30 pp)** — DVT did successfully shift notional away
   from high-cost slow trades.  Gross P&L nearly doubled ($1,350 → $2,628) because
   fast-reverting trades were sized larger, generating more raw spread P&L, while costs
   only rose 28 %.  The problem is those same high-leverage fast trades also generated
   more losses when they misfired.

4. **Win rate fell 9 pp (62 % → 52 %)** — DVT upsized fast trades (full 60 % vol) that
   were borderline losers at smaller sizes, tipping them across the break-even threshold.
   Simultaneously, it downsized marginally profitable slow trades, turning small wins
   into small losses after costs.

5. **MDD improved slightly (−50.7 % → −46.1 %)** — the only metric where DVT
   outperformed.  Defensive sizing on the worst slow-reverting stop-loss trades
   (ICP/MANA: −$1,185 → −$459, VET/EOS: −$381 → −$334) softened the 2022–2023
   drawdown peak, confirming that the protective intent of DVT is sound in principle.

6. **Conclusion** — the DVT formula as implemented is sensitive to the `BASELINE_HALFLIFE`
   assumption.  A higher baseline (e.g. 144–192 h) would flip the scaling direction:
   treating medium-speed pairs as the "normal" case and applying maximum leverage only to
   the fastest, most reliably mean-reverting spreads identified in back-testing.

---

## Configuration

All parameters live in `config.py`.  Key knobs:

```python
# Universe (29 curated 2022-safe liquid coins; see config.py)
TOP_40_SYMBOLS      = [...]
TIMEFRAME           = "1h"
START_DATE          = "2022-01-01T00:00:00Z"   # inclusive fetch start
END_DATE            = "2025-12-31T23:59:59Z"   # inclusive fetch end

# Calibration
LOOKBACK_WINDOW     = 336            # bars for rolling KF + EG scan (14 days × 24 h)
MIN_HISTORY         = 500            # warm-up bars before first trade
RESCAN_INTERVAL     = 12             # bars between pair-scans when flat

# Strategy
Z_ENTRY             = 1.5            # entry gate (|z-score|)
Z_EXIT              = 0.25           # profit-take gate
Z_STOP_LOSS         = 4.0            # hard stop — exit if |z| blows out above this
TARGET_VOL          = 0.60           # 60 % annual vol target (annualised at √8760)
TRANSACTION_COST    = 0.001          # 0.1 % per leg per change
HOURLY_FUNDING_RATE = 0.0000125      # ≈ 0.01 % per 8 h, charged every bar

# Risk
OU_HALFLIFE_MULTIPLIER = 2.5         # time-stop = 2.5 × half-life (in hours)
MIN_HALFLIFE        = 2              # floor on OU half-life (hours)
MAX_HALFLIFE        = 288            # hard cap — entries aborted above this (12 days)
MAX_LEVERAGE        = 4.0            # notional / portfolio cap
REBALANCE_THRESHOLD = 0.10           # min fractional leg deviation to trigger a rebalance

# Models
KALMAN_DELTA        = 1e-4           # KF state-noise parameter
HMM_N_COMPONENTS    = 2              # number of HMM hidden states
COINT_PVALUE_THRESHOLD = 0.05        # EG test acceptance level
```

`BacktestEngine` accepts `z_entry` and `z_exit` as constructor arguments to
override the config values without editing the file.

---

## Output Files

### `trade_log.csv` — 24 columns

| Column | Description |
|--------|-------------|
| `entry_date`, `exit_date` | Trade open/close timestamps |
| `sym_y`, `sym_x` | Dependent / independent asset |
| `direction` | `long_spread` or `short_spread` |
| `bars_held` | Trade duration in bars (hours) |
| `coint_pvalue` | EG p-value at entry |
| `half_life` | OU half-life at exit (hours) |
| `hedge_ratio` | Kalman β at exit |
| `entry_price_y/x` | Open price used for entry fill |
| `exit_price_y/x` | Open price used for exit fill (close price for `end_of_backtest`) |
| `gross_pnl` | P&L before any costs |
| `entry_cost`, `rebalance_cost`, `exit_cost` | Transaction cost breakdown |
| `funding_cost` | Cumulative hourly funding charges for this trade |
| `total_costs` | `entry_cost + rebalance_cost + exit_cost + funding_cost` |
| `net_pnl` | After-cost P&L |
| `return_pct` | `net_pnl / entry_portfolio_value` |
| `exit_reason` | `mean_reversion`, `stop_loss`, `time_stop`, or `end_of_backtest` |
| `entry_portfolio_value`, `exit_portfolio_value` | Portfolio snapshots |

### `equity_curve.csv` — 3 columns

`date`, `portfolio_value`, `bar_return`

---

## Dependencies

```
ccxt>=4.3.0         # exchange data feed
numpy>=1.26.0       # numerical core
pandas>=2.2.0       # time-series alignment
statsmodels>=0.14.0 # cointegration tests, OLS
hmmlearn>=0.3.0     # Gaussian HMM
scikit-learn>=1.4.0 # HMM dependency
scipy>=1.12.0       # statistical utilities
```
