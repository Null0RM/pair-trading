"""
backtest_demo.py
----------------
Runs the full engine on SYNTHETIC hourly data so no API key / internet is needed.

Universe  : 20 crypto-like symbols over 8760 hourly bars (≈ 1 year)
Pairs baked in (strong OU spread, half-life in hours, all ≤ MAX_HALFLIFE=168h):
    BTC/USDT  ↔  ETH/USDT   (β=1.15, HL= 96h, σ=0.040)
    SOL/USDT  ↔  AVAX/USDT  (β=1.05, HL=120h, σ=0.038)
    LTC/USDT  ↔  BCH/USDT   (β=0.95, HL= 72h, σ=0.042)
    DOT/USDT  ↔  ATOM/USDT  (β=0.90, HL=144h, σ=0.036)
    LINK/USDT ↔  UNI/USDT   (β=0.85, HL=100h, σ=0.044)
    BNB/USDT  ↔  ADA/USDT   (β=0.70, HL=108h, σ=0.040)
    XMR/USDT  ↔  ETC/USDT   (β=1.10, HL= 60h, σ=0.045)
    NEAR/USDT ↔  ALGO/USDT  (β=0.80, HL=132h, σ=0.038)

Standalone (market + idio, NOT cointegrated with each other):
    XRP, DOGE, TRX, MATIC
"""

from __future__ import annotations

import logging
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, ".")

# ── silence hmmlearn's convergence warnings ──────────────────────────────────
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
# ─────────────────────────────────────────────────────────────────────────────

from backtester.engine import BacktestEngine
from backtester.metrics import compute_metrics, compute_trade_stats
from main import export_equity_curve, export_trade_log, print_summary
from config import INITIAL_CAPITAL


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def _ou_path(n: int, half_life: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Discrete-time zero-mean OU (AR-1) process.  half_life is in bars (hours)."""
    phi = np.exp(-np.log(2.0) / half_life)
    path = np.zeros(n)
    for t in range(1, n):
        path[t] = phi * path[t - 1] + rng.normal(0.0, sigma)
    return path


def _make_ohlcv(
    close: np.ndarray, dates: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    n = len(close)
    hi = close * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    lo = close * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    return pd.DataFrame(
        {
            "open":   close * rng.uniform(0.9990, 1.0010, n),
            "high":   hi,
            "low":    lo,
            "close":  close,
            "volume": rng.uniform(1e4, 1e6, n),
        },
        index=dates,
    )


def generate_universe(n_bars: int = 8760, seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Build a synthetic hourly crypto universe with 8 strongly cointegrated pairs
    and 4 standalone symbols.

    Key tuning
    ----------
    * Common market factor has *low* hourly vol (≈ 0.002/bar, ~1 %/day) so it
      does not dominate the spread and generate spurious cointegrations.
    * OU half-lives are in hours and are all ≤ MAX_HALFLIFE (168 h = 7 days).
    * OU sigma is large enough that the rolling z-score crosses ±1.5 several
      times per pair over the backtest period.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(
        start=datetime(2025, 7, 1, tzinfo=timezone.utc),
        periods=n_bars,
        freq="h",
    )

    # ── Common market factor (weak hourly drift + vol) ───────────────────────
    # ~0.00125 %/h drift, ~0.2 %/h vol  →  ~1 %/day vol
    mkt = np.cumsum(rng.normal(0.0000125, 0.00204, n_bars))

    # ── Cointegrated pairs ───────────────────────────────────────────────────
    # (sym_y, sym_x, start_y, start_x, beta, half_life_hours, ou_sigma)
    coint_spec = [
        ("BTC/USDT",  "ETH/USDT",   30_000,  2_000,  1.15,  96, 0.040),
        ("SOL/USDT",  "AVAX/USDT",  50,      20,     1.05, 120, 0.038),
        ("LTC/USDT",  "BCH/USDT",   80,      200,    0.95,  72, 0.042),
        ("DOT/USDT",  "ATOM/USDT",  7,       10,     0.90, 144, 0.036),
        ("LINK/USDT", "UNI/USDT",   8,       5,      0.85, 100, 0.044),
        ("BNB/USDT",  "ADA/USDT",   300,     0.30,   0.70, 108, 0.040),
        ("XMR/USDT",  "ETC/USDT",   150,     20,     1.10,  60, 0.045),
        ("NEAR/USDT", "ALGO/USDT",  2.0,     0.15,   0.80, 132, 0.038),
    ]

    data: dict[str, pd.DataFrame] = {}

    for sym_y, sym_x, s_y, s_x, beta, hl, sigma in coint_spec:
        # x-asset: market + moderate idiosyncratic hourly walk
        bx = rng.uniform(0.7, 1.3)
        ix = np.cumsum(rng.normal(0.0, 0.00367, n_bars))   # ~1.8 %/day idio vol
        log_x = np.log(s_x) + bx * mkt + ix

        # y-asset: cointegrated with x via OU spread
        ou = _ou_path(n_bars, hl, sigma, rng)
        log_y = np.log(s_y) + beta * (log_x - np.log(s_x)) + ou

        for sym, lp in [(sym_y, log_y), (sym_x, log_x)]:
            if sym not in data:
                data[sym] = _make_ohlcv(np.exp(lp), dates, rng)

    # ── Standalone symbols (pure market + idio, no explicit cointegration) ──
    standalone = [
        ("XRP/USDT",   0.50,  0.90, 0.00612),
        ("DOGE/USDT",  0.08,  1.20, 0.00918),
        ("TRX/USDT",   0.07,  0.80, 0.00653),
        ("MATIC/USDT", 0.80,  1.00, 0.00775),
    ]
    for sym, start, bm, vol in standalone:
        if sym in data:
            continue
        idio = np.cumsum(rng.normal(0.0, vol, n_bars))
        log_p = np.log(start) + bm * mkt + idio
        data[sym] = _make_ohlcv(np.exp(log_p), dates, rng)

    print(f"  Synthetic universe: {len(data)} symbols × {n_bars} hourly bars")
    return data


# ---------------------------------------------------------------------------
# ASCII equity chart
# ---------------------------------------------------------------------------


def ascii_equity_chart(equity: list[float], width: int = 62, height: int = 14) -> None:
    eq = equity
    lo, hi = min(eq), max(eq)
    span = hi - lo or 1.0

    rows: list[str] = []
    step = max(1, len(eq) // width)
    for row in range(height):
        threshold = hi - (row / (height - 1)) * span
        line = ""
        for col in range(width):
            idx = min(col * step, len(eq) - 1)
            nxt = min((col + 1) * step, len(eq) - 1)
            v, vn = eq[idx], eq[nxt]
            if (v >= threshold > vn) or (vn >= threshold > v):
                line += "╱" if vn > v else "╲"
            elif v >= threshold and vn >= threshold:
                line += "─"
            else:
                line += " "

        if row == 0:
            label = f"  ${hi:>13,.0f}"
        elif row == height // 2:
            label = f"  ${(hi + lo) / 2:>13,.0f}"
        elif row == height - 1:
            label = f"  ${lo:>13,.0f}"
        else:
            label = ""
        rows.append(line + label)

    print("  ┌" + "─" * width + "┐")
    for r in rows:
        print(f"  │{r:<{width}}│")
    print("  └" + "─" * width + "┘")
    print(f"  {'Bar 0':<{width // 2}}{'Bar ' + str(len(eq)):>{width // 2}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Dynamic Multi-Asset Pair Trading  —  Backtest Demo      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Generate data
    print("\n[1/3]  Generating synthetic hourly universe …")
    raw_data = generate_universe(n_bars=6576, seed=42)

    # 2. Run engine (scan every flat bar — maximise opportunity capture)
    print("[2/3]  Running walk-forward backtest …\n")
    engine = BacktestEngine(
        raw_data=raw_data,
        initial_capital=INITIAL_CAPITAL,
        rescan_interval=24,     # scan once per day (24 bars) instead of every bar
        z_entry=1.5,            # widen entry gate (production default = 2.0)
        z_exit=0.25,
    )
    equity_curve, trade_log = engine.run()

    # 3. Metrics & output
    print("\n[3/3]  Computing metrics …\n")
    metrics = compute_metrics(equity_curve)
    trade_stats = compute_trade_stats(trade_log)

    export_trade_log(trade_log, "trade_log.csv")
    export_equity_curve(equity_curve, engine.dates, "equity_curve.csv")

    # ── Equity chart ──────────────────────────────────────────────────
    print("\n  Portfolio Equity Curve")
    ascii_equity_chart(equity_curve)

    # ── Trade-by-trade table ──────────────────────────────────────────
    if trade_log:
        hdr = (
            f"  {'#':>3}  {'Entry':>10}  {'Exit':>10}  "
            f"{'Pair':<24}  {'Dir':<12}  {'HL(h)':>6}  {'β':>7}  "
            f"{'Costs':>8}  {'Net P&L':>11}  Reason"
        )
        print(f"\n{hdr}")
        print("  " + "─" * (len(hdr) - 2))
        for i, t in enumerate(trade_log, 1):
            pair = f"{t.sym_y.split('/')[0]}/{t.sym_x.split('/')[0]}"
            print(
                f"  {i:>3}  {t.entry_date:>10}  {t.exit_date:>10}  "
                f"{pair:<24}  {t.direction:<12}  {t.half_life:>6.1f}  "
                f"{t.hedge_ratio:>7.4f}  "
                f"${t.total_costs:>6,.0f}  "
                f"${t.net_pnl:>+10,.2f}  {t.exit_reason}"
            )

    # ── Final metrics panel ───────────────────────────────────────────
    print_summary(
        metrics=metrics,
        trade_stats=trade_stats,
        n_trades=len(trade_log),
        initial_capital=INITIAL_CAPITAL,
        final_value=engine.portfolio_value,
    )
