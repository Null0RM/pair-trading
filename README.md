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

Fetches ~6 months of hourly OHLCV (1h bars, `SINCE_DAYS=180`) for the
top-40 crypto assets, then runs the walk-forward backtest.  Expect the data
fetch to take 5–10 minutes due to exchange rate limits and bar count.

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

> All runs use **synthetic** hourly data (20 symbols, 8 cointegrated pairs baked
> in by construction), `z_entry=1.5`, `z_exit=0.25`, `rescan_interval=24` (scan
> once per day), `TARGET_VOL=15 %`, `MAX_HALFLIFE=96 h`, `Z_STOP_LOSS=4.0`,
> `REBALANCE_THRESHOLD=0.10`, `momentum_window=12`, funding rate charged every bar.
> Run `python3 backtest_demo.py` to regenerate.

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

## Configuration

All parameters live in `config.py`.  Key knobs:

```python
# Universe
TOP_40_SYMBOLS      = [...]          # 40 Binance spot pairs
TIMEFRAME           = "1h"           # hourly bars
SINCE_DAYS          = 180            # calendar days of history to fetch

# Calibration
LOOKBACK_WINDOW     = 336            # bars for rolling KF + EG scan (14 days × 24 h)
MIN_HISTORY         = 500            # warm-up bars before first trade
RESCAN_INTERVAL     = 5              # bars between pair-scans when flat

# Strategy
Z_ENTRY             = 2.0            # entry gate (|z-score|)
Z_EXIT              = 0.25           # profit-take gate
Z_STOP_LOSS         = 4.0            # hard stop — exit if |z| blows out above this
TARGET_VOL          = 0.15           # 15 % annual vol target (annualised at √8760)
TRANSACTION_COST    = 0.001          # 0.1 % per leg per change
HOURLY_FUNDING_RATE = 0.0000125      # ≈ 0.01 % per 8 h, charged every bar

# Risk
OU_HALFLIFE_MULTIPLIER = 2.0         # time-stop = 2 × half-life (in hours)
MIN_HALFLIFE        = 2              # floor on OU half-life (hours)
MAX_HALFLIFE        = 96             # hard cap — entries aborted above this (4 days)
MAX_LEVERAGE        = 2.0            # notional / portfolio cap
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
