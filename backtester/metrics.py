"""
backtester/metrics.py
---------------------
Portfolio performance metrics computed from a daily equity curve.

Metrics
-------
total_return      – cumulative return over the full period
annualised_return – CAGR (compound annual growth rate)
max_drawdown      – peak-to-trough maximum drawdown (negative number)
sharpe_ratio      – annualised Sharpe ratio (risk-free rate = 0)
calmar_ratio      – annualised return / |max_drawdown|
win_rate          – fraction of trades with positive net P&L
avg_trade_pnl     – average net P&L per trade (USD)
avg_hold_bars     – average number of bars a trade is open
"""

from __future__ import annotations

import numpy as np


def compute_metrics(equity_curve: list[float] | np.ndarray) -> dict[str, float]:
    """
    Compute key risk/return metrics from a daily equity curve.

    Parameters
    ----------
    equity_curve:
        Daily portfolio values, one element per trading day.
        The first element is assumed to be the starting capital.

    Returns
    -------
    dict
        All values are plain Python floats rounded for readability.
        Non-computable metrics are returned as ``0.0`` or ``np.nan``.
    """
    equity = np.asarray(equity_curve, dtype=float)

    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "annualised_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
        }

    # --- Per-bar returns ---
    bar_returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], np.nan)
    bar_returns = bar_returns[np.isfinite(bar_returns)]

    # --- Total return ---
    total_return: float = float(equity[-1] / equity[0] - 1.0)

    # --- CAGR ---
    n_years: float = len(equity) / 8760.0
    if n_years > 0 and equity[0] > 0 and equity[-1] > 0:
        annualised_return: float = float((equity[-1] / equity[0]) ** (1.0 / n_years) - 1.0)
    else:
        annualised_return = 0.0

    # --- Max drawdown ---
    running_max = np.maximum.accumulate(equity)
    drawdowns = np.where(running_max > 0, (equity - running_max) / running_max, 0.0)
    max_drawdown: float = float(np.min(drawdowns))

    # --- Sharpe ratio (annualised, risk-free = 0) ---
    if len(bar_returns) > 1 and np.std(bar_returns, ddof=1) > 1e-10:
        sharpe: float = float(
            np.mean(bar_returns) / np.std(bar_returns, ddof=1) * np.sqrt(8760)
        )
    else:
        sharpe = 0.0

    # --- Calmar ratio ---
    calmar: float = (
        float(annualised_return / abs(max_drawdown))
        if max_drawdown < -1e-10
        else 0.0
    )

    return {
        "total_return": round(total_return, 6),
        "annualised_return": round(annualised_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "sharpe_ratio": round(sharpe, 4),
        "calmar_ratio": round(calmar, 4),
    }


def compute_trade_stats(trade_records: list) -> dict[str, float]:
    """
    Aggregate statistics computed from the list of ``TradeRecord`` objects
    produced by the engine.

    Parameters
    ----------
    trade_records:
        List of ``TradeRecord`` dataclass instances (or any objects with
        ``net_pnl`` and ``bars_held`` attributes).

    Returns
    -------
    dict
        ``win_rate``, ``avg_net_pnl``, ``avg_hold_bars``,
        ``total_gross_pnl``, ``total_transaction_costs``.
    """
    if not trade_records:
        return {
            "win_rate": 0.0,
            "avg_net_pnl": 0.0,
            "avg_hold_bars": 0.0,
            "total_gross_pnl": 0.0,
            "total_transaction_costs": 0.0,
        }

    net_pnls = np.array([t.net_pnl for t in trade_records], dtype=float)
    hold_bars = np.array([t.bars_held for t in trade_records], dtype=float)
    gross_pnls = np.array([t.gross_pnl for t in trade_records], dtype=float)
    costs = np.array([t.total_costs for t in trade_records], dtype=float)

    win_rate: float = float(np.mean(net_pnls > 0)) if len(net_pnls) > 0 else 0.0

    return {
        "win_rate": round(win_rate, 4),
        "avg_net_pnl": round(float(np.mean(net_pnls)), 2),
        "avg_hold_bars": round(float(np.mean(hold_bars)), 1),
        "total_gross_pnl": round(float(np.sum(gross_pnls)), 2),
        "total_transaction_costs": round(float(np.sum(costs)), 2),
    }
