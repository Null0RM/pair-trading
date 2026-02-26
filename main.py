"""
main.py
-------
Entry point for the Dynamic Multi-Asset Pair Trading Backtesting Engine.

Pipeline
--------
1.  Pre-fetch OHLCV data for the top-40 crypto universe via CCXT.
2.  Run the walk-forward backtesting engine.
3.  Compute and print final performance metrics.
4.  Export:
    - ``trade_log.csv``   – every trade with full attribution
    - ``equity_curve.csv``– daily portfolio values

Usage
-----
    python main.py

Optional env / config overrides: edit ``config.py`` directly.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import sys
from pathlib import Path

import pandas as pd

from backtester.engine import BacktestEngine, TradeRecord
from backtester.metrics import compute_metrics, compute_trade_stats
from config import INITIAL_CAPITAL, TOP_40_SYMBOLS
from data.fetcher import fetch_all_ohlcv

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def export_trade_log(
    trade_log: list[TradeRecord],
    path: str = "trade_log.csv",
) -> None:
    """Write every ``TradeRecord`` to a CSV file."""
    if not trade_log:
        logger.warning("No trades to export – trade_log.csv not written.")
        return

    records = [dataclasses.asdict(t) for t in trade_log]
    fieldnames = list(records[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info("Trade log  → %s  (%d trades)", path, len(records))


def export_equity_curve(
    equity_curve: list[float],
    dates: list,
    path: str = "equity_curve.csv",
) -> None:
    """Write the daily equity curve to a CSV file."""
    # The equity curve starts at MIN_HISTORY so we align dates accordingly
    offset = len(dates) - len(equity_curve)
    aligned_dates = [str(d.date()) for d in dates[offset:]]

    df = pd.DataFrame(
        {
            "date": aligned_dates,
            "portfolio_value": equity_curve,
        }
    )
    df["daily_return"] = df["portfolio_value"].pct_change().round(6)
    df.to_csv(path, index=False)
    logger.info("Equity curve → %s  (%d rows)", path, len(df))


def print_summary(
    metrics: dict,
    trade_stats: dict,
    n_trades: int,
    initial_capital: float,
    final_value: float,
) -> None:
    """Pretty-print the final performance summary to stdout."""
    sep = "=" * 56
    print(f"\n{sep}")
    print("  BACKTEST RESULTS — Dynamic Pair Trading Engine")
    print(sep)
    print(f"  Initial capital        : ${initial_capital:>14,.2f}")
    print(f"  Final portfolio value  : ${final_value:>14,.2f}")
    print(f"  Total return           : {metrics['total_return']*100:>+13.2f} %")
    print(f"  Annualised return      : {metrics['annualised_return']*100:>+13.2f} %")
    print(f"  Max drawdown           : {metrics['max_drawdown']*100:>+13.2f} %")
    print(f"  Sharpe ratio           : {metrics['sharpe_ratio']:>14.4f}")
    print(f"  Calmar ratio           : {metrics['calmar_ratio']:>14.4f}")
    print(sep)
    print(f"  Total trades           : {n_trades:>14d}")
    if n_trades:
        print(f"  Win rate               : {trade_stats['win_rate']*100:>13.1f} %")
        print(f"  Avg net P&L / trade    : ${trade_stats['avg_net_pnl']:>13,.2f}")
        print(f"  Avg hold (bars)        : {trade_stats['avg_hold_bars']:>14.1f}")
        print(f"  Total gross P&L        : ${trade_stats['total_gross_pnl']:>13,.2f}")
        print(f"  Total transaction cost : ${trade_stats['total_transaction_costs']:>13,.2f}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 56)
    logger.info("Dynamic Multi-Asset Pair Trading Backtesting Engine")
    logger.info("=" * 56)

    # ------------------------------------------------------------------
    # 1. Pre-fetch data
    # ------------------------------------------------------------------
    logger.info("Step 1/3 – Pre-fetching OHLCV data …")
    raw_data = fetch_all_ohlcv(symbols=TOP_40_SYMBOLS)

    if len(raw_data) < 2:
        logger.error(
            "Fewer than 2 symbols loaded (%d).  "
            "Check connectivity and symbol names.  Exiting.",
            len(raw_data),
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Run backtest
    # ------------------------------------------------------------------
    logger.info("Step 2/3 – Running backtest engine …")
    engine = BacktestEngine(raw_data=raw_data, initial_capital=INITIAL_CAPITAL)
    equity_curve, trade_log = engine.run()

    # ------------------------------------------------------------------
    # 3. Compute metrics and export
    # ------------------------------------------------------------------
    logger.info("Step 3/3 – Computing metrics and exporting results …")
    metrics = compute_metrics(equity_curve)
    trade_stats = compute_trade_stats(trade_log)

    export_trade_log(trade_log)
    export_equity_curve(equity_curve, engine.dates)

    print_summary(
        metrics=metrics,
        trade_stats=trade_stats,
        n_trades=len(trade_log),
        initial_capital=INITIAL_CAPITAL,
        final_value=engine.portfolio_value,
    )


if __name__ == "__main__":
    main()
