# CLAUDE.md — Project Context for Claude Code

This file records the full development history, architectural decisions, known bugs
fixed, and running conventions for the **Dynamic Multi-Asset Pair Trading
Backtesting Engine**.  Read this before touching any file.

---

## Project Goal

Build a walk-forward crypto pair-trading backtester in pure Python that:

* Pre-fetches **hourly** OHLCV data for the top-40 crypto assets via CCXT (Binance spot).
* Dynamically selects the active pair at runtime (lowest Engle-Granger p-value).
* Uses a **Kalman Filter** for the hedge ratio, **OU process** for the half-life,
  and a **GaussianHMM** for regime gating.
* Sizes positions via **volatility targeting** (15 % annual).
* Charges **0.1 % per leg** on every notional change plus an **hourly funding rate** (≈ 0.01 %/8 h).
* Exports `trade_log.csv` and `equity_curve.csv`.

---

## Module Map

```
pair-trading/
├── config.py               All constants — edit here first
├── requirements.txt
├── data/
│   └── fetcher.py          CCXT OHLCV pre-fetch (paginated, retry)
├── math_utils/
│   ├── kalman_filter.py    Scalar 1-D Kalman Filter; OLS-seeded batch_estimate
│   ├── ou_process.py       OLS AR(1) → θ, μ, σ, half-life
│   └── regime.py           GaussianHMM (2-state); mean-reverting state = min |mean|
├── strategy/
│   ├── cointegration.py    EG scan: lowest p-value, positive OLS β required
│   └── signals.py          Z-score, vol-targeting, entry/exit logic
├── backtester/
│   ├── engine.py           BacktestEngine walk-forward loop
│   └── metrics.py          Sharpe, MDD, CAGR, Calmar, win-rate
├── main.py                 Entry point (live data via CCXT)
└── backtest_demo.py        Self-contained demo using synthetic data (no API key)
```

---

## How to Run

### Live backtest (requires Binance internet access)
```bash
pip install -r requirements.txt
python3 main.py
```

### Offline demo (synthetic data, no API key needed)
```bash
python3 backtest_demo.py
```

Outputs written to the working directory:
* `trade_log.csv`    — one row per completed round-trip trade
* `equity_curve.csv` — hourly portfolio value + hourly return
* `backtest.log`     — full INFO-level execution log

---

## Key Config Parameters (`config.py`)

| Parameter | Default | Notes |
|---|---|---|
| `TOP_40_SYMBOLS` | 40 pairs | Edit to reduce universe for faster runs |
| `TIMEFRAME` | `"1h"` | Hourly bars; fetcher paginates accordingly |
| `SINCE_DAYS` | 180 days | Calendar days of history to pre-fetch |
| `LOOKBACK_WINDOW` | 336 hours | Rolling calibration window (14 days × 24 h) |
| `MIN_HISTORY` | 500 bars | Warm-up before first trade |
| `RESCAN_INTERVAL` | 5 bars | Bars between pair-scans when flat |
| `Z_ENTRY` | 2.0 | Entry z-score threshold |
| `Z_EXIT` | 0.25 | Mean-reversion exit threshold |
| `Z_STOP_LOSS` | 4.0 | Hard stop-loss — exit immediately if `\|z\|` ≥ this |
| `TARGET_VOL` | 0.15 | 15 % annual vol target |
| `TRANSACTION_COST` | 0.001 | 0.1 % per leg |
| `HOURLY_FUNDING_RATE` | 0.0000125 | ≈ 0.01 %/8 h funding fee, charged every bar |
| `REBALANCE_THRESHOLD` | 0.10 | Min fractional leg deviation required to trigger a rebalance |
| `OU_HALFLIFE_MULTIPLIER` | 2.0 | Time-stop = 2× OU half-life |
| `MIN_HALFLIFE` | 2 hours | OU half-life floor |
| `MAX_HALFLIFE` | 96 hours | OU half-life hard cap (4 days); entries aborted above this |
| `KALMAN_DELTA` | 1e-4 | KF state-noise parameter |
| `COINT_PVALUE_THRESHOLD` | 0.05 | EG test acceptance cut-off |

`BacktestEngine` also accepts `z_entry` and `z_exit` as constructor overrides
so the demo can use looser thresholds without touching `config.py`.

---

## Architecture Decisions

### Spread definition
```
spread_t = log(price_y_t) − β_t · log(price_x_t)
```
Log-price space is used throughout (KF, OU, Z-score).  PnL is computed
in raw-price space via share counts.

### Kalman Filter seeding
`batch_estimate` runs OLS on the first 30 bars to initialise β before
the filter starts.  This prevents sign-flip when `log(price) < 0`
(coins priced below $1 like XRP, ADA, ALGO).

### Positive β requirement
`cointegration.find_best_pair` rejects any pair whose OLS hedge ratio ≤ 0.
Both legs must move in the same direction for the trade to make sense as
a statistical-arbitrage position.

### Cash accounting (exact)

Prices used at each step eliminate look-ahead bias:

```
Signal   : computed on close[0 .. bar_idx-1]        ← prev-bar close only
Open     : cash -= shares_y·open_y + shares_x·open_x + entry_cost
                                                     ← filled at open[bar_idx]
MTM      : portfolio_value = cash + shares_y·close_y + shares_x·close_x
                                                     ← valued at close[bar_idx]
Funding  : gross_exposure = |shares_y·close_y| + |shares_x·close_x|
           cash           -= gross_exposure × HOURLY_FUNDING_RATE
           portfolio_value -= gross_exposure × HOURLY_FUNDING_RATE
Close    : cash += shares_y·open_y + shares_x·open_x − exit_cost
                                                     ← filled at open[bar_idx]
           portfolio_value = cash
Rebal    : cash -= Δshares_y·open_y + Δshares_x·open_x + rebal_cost
                                                     ← filled at open[bar_idx]
```

`total_costs` on exit = entry_cost + rebalance_cost + exit_cost + cumulative_funding_cost.

End-of-backtest forced close uses `close[last_bar]` (no next open available).

### HMM regime gating
`RegimeClassifier` is fit on `[Δspread, |Δspread|]` features.  The state
with the smallest `|mean(Δspread)|` is labelled *mean-reverting*.  When
the HMM does not converge (common on 336-bar windows) `is_mean_reverting`
returns `True` (fail-open), preventing the filter from silently blocking all trades.
HMM convergence warnings are suppressed via `logging.getLogger('hmmlearn')`.

### Half-life hard gate
`_try_open_position` checks `ou["half_life"] > MAX_HALFLIFE` on the **raw**
(unclipped) OU estimate before opening.  If the condition is true, the entry
is aborted and the scan is throttled by `RESCAN_INTERVAL`.  This replaces the
old silent clip to `MAX_HALFLIFE` which would allow entering slowly-reverting pairs.

### Z-score stop-loss
`should_exit` evaluates three conditions in strict priority order:
1. **Stop-loss** — `|z| ≥ Z_STOP_LOSS` (4.0): exits immediately, labeled `"stop_loss"`.
2. **Mean-reversion** — `|z| ≤ Z_EXIT` (0.25): normal profit-take.
3. **Time-stop** — `bars_held ≥ halflife_multiplier × half_life`.

The stop-loss caps damage from structural cointegration breaks before the time-stop would fire.
`stop_loss_threshold` defaults to `float("inf")` so callers that don't pass it retain the old behaviour.

### Rebalance threshold gate
Every bar, the Kalman Filter produces a new β and new target share counts.
Without a threshold, even tiny β drift triggers full rebalances at 0.1 % cost per leg — "fee bleed".
`_handle_active` in `engine.py` computes fractional deviation for each leg:
```
dev = abs(new_shares - pos.shares) / abs(pos.shares)
```
The rebalance (and its cost) is **skipped** if both `dev_y ≤ REBALANCE_THRESHOLD` AND
`dev_x ≤ REBALANCE_THRESHOLD`.  The hedge ratio `pos.hedge_ratio` is still updated
unconditionally so the next z-score uses the fresh β.  Division-by-zero is guarded
with `abs(pos.shares) > 1e-10`.

### Annualization (hourly data)
All annualization factors use **8 760 hours/year** (crypto trades 24/7):
* `signals.py` — `annual_vol = hourly_vol × √8760`
* `metrics.py` — CAGR denominator `len(equity) / 8760`; Sharpe `× √8760`

OU parameters (`theta`, `half_life`) from `ou_process.py` are naturally
in **hours** when the input spread series is hourly.  No code change was
needed there, but all downstream thresholds (`MIN_HALFLIFE`, `MAX_HALFLIFE`,
`bars_held` time-stop) are now interpreted in hours.

---

## Bugs Fixed During Development

| Bug | Root cause | Fix |
|---|---|---|
| `%,.2f` crash in logger | Python `%`-style formatting doesn't support `,` separator | Pre-format number as f-string, pass as `%s` arg |
| β = 0.0001 (near-zero) | KF initialised as `y/x` which flips sign for negative log-prices (price < $1) | OLS seed on first 30 bars in `batch_estimate` |
| β clipped to 1e-4 | Hard `max(β, 1e-4)` hid invalid estimates | Replaced with validity gate: reject if `abs(β) < 0.01` |
| Spurious pairs selected | EG test found false positives; negative OLS β accepted | Added `ols_beta > 0` guard in `find_best_pair` |
| HMM warning spam | `hmmlearn` logs to Python logging, not `warnings` module | `logging.getLogger('hmmlearn').setLevel(logging.ERROR)` |
| Too few trades in demo | Synthetic spread σ too small; Z_ENTRY=2.0 rarely breached | Added `z_entry` constructor param; demo uses 1.5 |
| Slow-reverting pairs entered | `MAX_HALFLIFE` clip silently allowed bad entries | Hard abort in `_try_open_position` if raw `half_life > MAX_HALFLIFE` |
| Funding cost not modelled | Perpetual swap funding charges were ignored | `HOURLY_FUNDING_RATE` deducted from cash + portfolio_value each bar; accumulated in `Position.cumulative_funding_cost`; included in `total_costs` on close |
| Wrong annualization | `√252` (daily) used on hourly data | Changed to `√8760` in `signals.py` and `metrics.py` |
| Look-ahead bias | Signals computed on `close[bar_idx]`; trades filled at same close | `hist` window ends at `prev_idx = bar_idx - 1`; all fills use `open[bar_idx]`; MTM uses `close[bar_idx]` after execution; KF incremental update feeds `close[prev_idx]` |
| Stale naming (`days_held`, `daily_vol`) | Variables carried daily-timeframe semantics on hourly data | Renamed: `days_held` → `bars_held` (Position, TradeRecord, signals.py, metrics.py); `daily_vol` → `bar_vol` (signals.py); `daily_returns` → `bar_returns` (metrics.py); `avg_hold_days` → `avg_hold_bars` (metrics.py, main.py) |
| Single price matrix | `_build_price_matrix` only extracted close prices | Replaced by `_build_price_matrices` returning `(close_prices, open_prices)`; open matrix falls back to close when open data is missing |
| Fee bleed (costs > gross P&L) | Kalman β drift every hour triggered full rebalances at 0.1 % cost per leg, ballooning costs to 114 % of gross P&L | Added `REBALANCE_THRESHOLD=0.10`; rebalance skipped unless either leg deviates > 10 % from current shares |
| No hard stop-loss | Cointegration breaks accumulated losses until the time-stop fired; large time-stop losses dominated P&L | Added `Z_STOP_LOSS=4.0` checked first in `should_exit`; exits immediately with `"stop_loss"` before mean-reversion or time-stop |
| Slow-reverting pairs still entered at cap boundary | `MAX_HALFLIFE=168h` allowed HL~120–144h pairs that consistently ended as losers | Tightened to `MAX_HALFLIFE=96h`; half-lives above 4 days are now rejected at entry |

---

## Synthetic Demo Data (`backtest_demo.py`)

20 symbols × **8760 hourly bars** (2022-01-01 → 2022-12-31, `freq="h"`).

**8 explicitly cointegrated pairs** (OU spread baked in, half-lives in hours):
```
BTC/ETH   β=1.15  HL= 96h  σ=0.040
SOL/AVAX  β=1.05  HL=120h  σ=0.038
LTC/BCH   β=0.95  HL= 72h  σ=0.042
DOT/ATOM  β=0.90  HL=144h  σ=0.036
LINK/UNI  β=0.85  HL=100h  σ=0.044
BNB/ADA   β=0.70  HL=108h  σ=0.040
XMR/ETC   β=1.10  HL= 60h  σ=0.045
NEAR/ALGO β=0.80  HL=132h  σ=0.038
```

With `MAX_HALFLIFE=96h` (4 days), only **3 of 8** baked-in pairs have HL ≤ 96h and
pass the hard entry gate during the 2022 demo:

| Pair | HL | Gate |
|---|---|---|
| BTC/ETH | 96h | ✅ passes (exactly at cap) |
| LTC/BCH | 72h | ✅ passes |
| XMR/ETC | 60h | ✅ passes |
| SOL/AVAX | 120h | ❌ blocked |
| LINK/UNI | 100h | ❌ blocked |
| BNB/ADA | 108h | ❌ blocked |
| DOT/ATOM | 144h | ❌ blocked |
| NEAR/ALGO | 132h | ❌ blocked |

The engine falls back to the EG scan and may trade cross-pairs (e.g. ALGO/MATIC from
standalone symbols) when no baked-in pair qualifies.

**4 standalone symbols** (market + idio, no forced cointegration):
`XRP, DOGE, TRX, MATIC`

Common market factor is kept deliberately weak (σ_mkt ≈ 0.00204/hour, ≈ 1 %/day)
so spurious cross-pair cointegrations are minimised and the explicitly
constructed pairs dominate the EG scan.

---

## Backtest Run Results

Two runs are maintained in `README.md` (full trade logs + before/after comparison):

| Period | Bars | Trades | Win % | Net Return | Sharpe | MDD | Cost/Gross |
|---|---|---|---|---|---|---|---|
| 2022 (synthetic) | 8 760 | 15 | 73.3 % | +3.60 % | 0.77 | −5.27 % | 33 % |
| 2025 Q3–2026 Q1 (synthetic) | 6 576 | 21 | 71.4 % | +1.79 % | 0.41 | −8.92 % | 45 % |

`backtest_demo.py` is currently configured for the **2025 Q3–2026 Q1** period.
To re-run 2022: change `start=datetime(2025,7,1), n_bars=6576` → `start=datetime(2022,1,1), n_bars=8760`.

Key engine parameters for both runs: `seed=42`, `z_entry=1.5`, `z_exit=0.25`,
`rescan_interval=24`, `Z_STOP_LOSS=4.0`, `REBALANCE_THRESHOLD=0.10`, `MAX_HALFLIFE=96h`.

> **Legacy note:** Pre-hourly results (700 daily bars, annualised at √252,
> 8 trades, −3.42 % return) are preserved in README.md under
> "Legacy Results (Daily Demo)" for historical reference only.

---

## Python Interpreter

Use `python3` (not `python`) on this machine (Ubuntu 24, Python 3.12).
Install dependencies with `--break-system-packages` if using system pip:

```bash
pip install -r requirements.txt --break-system-packages
```

---

## File Outputs

| File | Contents |
|---|---|
| `trade_log.csv` | One row per closed trade; 24 columns including gross/net P&L, entry/rebalance/exit/funding cost breakdown, exit reason |
| `equity_curve.csv` | Timestamp, portfolio_value, hourly_return (one row per bar) |
| `backtest.log` | Full INFO log of the run |
