"""
backtester/engine.py
---------------------
Dynamic Multi-Asset Pair Trading Backtesting Engine.

Walk-forward simulation overview
---------------------------------
1.  Pre-build aligned close-price and open-price matrices and their log-prices.
2.  For every bar (starting at MIN_HISTORY):

    A.  **Portfolio flat?**
        *   Throttle scanning with ``RESCAN_INTERVAL`` to avoid re-scanning
            every single flat bar (reduces compute with 780+ pairs).
        *   Compute signals (Kalman, OU, HMM, Z-score) using close prices
            strictly up to bar_idx - 1 (no look-ahead).
        *   If a qualifying pair is found and |z| > Z_ENTRY:
            -   Execute the entry at the **open price** of bar_idx.
            -   Deduct entry transaction costs.

    B.  **Position open?**
        *   Increment ``bars_held``.
        *   Compute signals (incremental Kalman update, spread, Z-score,
            OU half-life) using close prices up to bar_idx - 1.
        *   Check exit conditions (mean-reversion or time-stop).
            -   If exiting: close position at **open price** of bar_idx.
            -   If staying: rebalance at **open price** of bar_idx.

    C.  Mark portfolio to market at bar_idx close (end-of-bar valuation).
    D.  Deduct hourly funding cost on gross notional (close-based exposure).

3.  At end-of-data: close any open position at the last bar's close price.

No look-ahead bias
------------------
Indicators are computed on close[0 .. bar_idx-1].
All fills (entry, exit, rebalance) use open[bar_idx].
End-of-bar portfolio valuation uses close[bar_idx].

Cash / position accounting
---------------------------
    portfolio_value = cash
                     + shares_y  × price_y
                     + shares_x  × price_x    (shares_x < 0 for short)

*Opening* a position (at open[bar_idx]):
    cash  -= shares_y × open_y + shares_x × open_x + entry_cost

*Rebalancing* (at open[bar_idx]):
    Δcash  = -(Δshares_y × open_y + Δshares_x × open_x) - rebalance_cost

*Closing* a position (at open[bar_idx]):
    cash  += shares_y × open_y + shares_x × open_x - exit_cost
    portfolio_value = cash   (position fully converted to cash)

*Funding* (at close[bar_idx]):
    cash           -= gross_exposure × HOURLY_FUNDING_RATE
    portfolio_value -= gross_exposure × HOURLY_FUNDING_RATE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    HOURLY_FUNDING_RATE,
    INITIAL_CAPITAL,
    KALMAN_DELTA,
    KALMAN_VE,
    LOOKBACK_WINDOW,
    MAX_HALFLIFE,
    MIN_HALFLIFE,
    MIN_HISTORY,
    OU_HALFLIFE_MULTIPLIER,
    REBALANCE_THRESHOLD,
    RESCAN_INTERVAL,
    TRANSACTION_COST,
    Z_ENTRY,
    Z_EXIT,
    Z_STOP_LOSS,
)
from math_utils.kalman_filter import KalmanFilter
from math_utils.ou_process import estimate_ou_parameters
from math_utils.regime import RegimeClassifier
from strategy.cointegration import find_best_pair
from strategy.signals import (
    compute_position_size,
    compute_spread,
    compute_zscore,
    get_entry_direction,
    should_exit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """State of a single open pair trade."""

    sym_y: str
    sym_x: str
    direction: int              # +1 long spread / -1 short spread
    shares_y: float
    shares_x: float
    entry_date: pd.Timestamp
    entry_price_y: float
    entry_price_x: float
    hedge_ratio: float
    half_life: float
    coint_pvalue: float
    entry_portfolio_value: float    # portfolio_value *before* entry costs
    entry_cost: float               # transaction cost paid at entry
    bars_held: int = 0
    cumulative_rebalance_cost: float = 0.0
    cumulative_funding_cost: float = 0.0


@dataclass
class TradeRecord:
    """Immutable record of one completed round-trip trade."""

    entry_date: str
    exit_date: str
    sym_y: str
    sym_x: str
    direction: str              # "long_spread" | "short_spread"
    bars_held: int
    coint_pvalue: float
    half_life: float            # OU half-life at entry (hours)
    hedge_ratio: float          # Kalman β at entry
    entry_price_y: float
    entry_price_x: float
    exit_price_y: float
    exit_price_x: float
    gross_pnl: float            # P&L before any transaction costs
    entry_cost: float
    rebalance_cost: float
    exit_cost: float
    funding_cost: float
    total_costs: float
    net_pnl: float
    return_pct: float           # net_pnl / entry_portfolio_value
    exit_reason: str            # "mean_reversion" | "time_stop" | "end_of_backtest"
    entry_portfolio_value: float
    exit_portfolio_value: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """
    Dynamic multi-asset pair trading backtesting engine.

    Parameters
    ----------
    raw_data:
        Pre-fetched OHLCV DataFrames keyed by symbol (output of ``fetch_all_ohlcv``).
    initial_capital:
        Starting portfolio value in USD.
    lookback:
        Rolling calibration window in bars (hours for 1h data).
    min_history:
        Number of bars to wait before the first trade is allowed.
    rescan_interval:
        Minimum bars between pair-scan attempts while the portfolio is flat.
    """

    def __init__(
        self,
        raw_data: dict[str, pd.DataFrame],
        initial_capital: float = INITIAL_CAPITAL,
        lookback: int = LOOKBACK_WINDOW,
        min_history: int = MIN_HISTORY,
        rescan_interval: int = RESCAN_INTERVAL,
        z_entry: float = Z_ENTRY,
        z_exit: float = Z_EXIT,
    ) -> None:
        self.lookback = lookback
        self.min_history = min_history
        self._z_entry = z_entry
        self._z_exit = z_exit
        self.rescan_interval = rescan_interval

        # Build aligned close and open price matrices
        self.close_prices, self.open_prices = self._build_price_matrices(raw_data)
        self.log_prices: pd.DataFrame = np.log(self.close_prices)
        self.dates: list[pd.Timestamp] = list(self.close_prices.index)
        self.symbols: list[str] = list(self.close_prices.columns)

        logger.info(
            "Price matrix: %d symbols × %d bars  (%s → %s)",
            len(self.symbols),
            len(self.dates),
            self.dates[0],
            self.dates[-1],
        )

        # Portfolio state
        self.portfolio_value: float = initial_capital
        self.cash: float = initial_capital
        self.position: Optional[Position] = None

        # Outputs
        self.equity_curve: list[float] = []
        self.trade_log: list[TradeRecord] = []

        # Per-pair Kalman filters (persisted across scans)
        self._kalman_filters: dict[tuple[str, str], KalmanFilter] = {}

        # Regime classifier (shared; re-fitted per pair at entry)
        self._regime: RegimeClassifier = RegimeClassifier()

        # Throttle: next bar index at which a re-scan is allowed
        self._next_scan_idx: int = min_history

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[float], list[TradeRecord]]:
        """
        Execute the full walk-forward backtest.

        Returns
        -------
        (equity_curve, trade_log)
            ``equity_curve`` has one float per bar in ``self.dates``.
            ``trade_log``    has one ``TradeRecord`` per closed trade.
        """
        logger.info(
            "Backtest started  capital=%.0f  lookback=%d  min_history=%d",
            self.cash,
            self.lookback,
            self.min_history,
        )

        for bar_idx, current_date in enumerate(self.dates):

            if bar_idx % 1000 == 0:
                logger.info("Processing bar %d / %d...", bar_idx, len(self.dates))

            # Skip warm-up period
            if bar_idx < self.min_history:
                self.equity_curve.append(self.portfolio_value)
                continue

            # prev_idx is the last completed bar.  All signals are computed
            # from close[0..prev_idx] to guarantee zero look-ahead bias.
            prev_idx = bar_idx - 1

            # 1. Execute decisions: signals from prev-bar close, fills at cur-bar open
            if self.position is None:
                self._handle_flat(bar_idx, prev_idx, current_date)
            else:
                self._handle_active(bar_idx, prev_idx, current_date)

            # 2. Mark portfolio to market at current bar's close
            self._mark_to_market(bar_idx)

            # 3. Deduct funding cost on gross notional held through this bar
            if self.position is not None:
                pos = self.position
                close_y = self._get_close_price(bar_idx, pos.sym_y)
                close_x = self._get_close_price(bar_idx, pos.sym_x)
                if close_y is not None and close_x is not None:
                    gross_exposure = (
                        abs(pos.shares_y * close_y) + abs(pos.shares_x * close_x)
                    )
                    funding_cost = gross_exposure * HOURLY_FUNDING_RATE
                    self.cash -= funding_cost
                    self.portfolio_value -= funding_cost
                    pos.cumulative_funding_cost += funding_cost

            self.equity_curve.append(self.portfolio_value)

        # Close any remaining open position at the last bar's close price
        if self.position is not None:
            last_idx = len(self.dates) - 1
            self._close_position(
                last_idx, self.dates[last_idx], "end_of_backtest", use_close_price=True
            )

        logger.info(
            "Backtest finished  final_value=%.2f  trades=%d",
            self.portfolio_value,
            len(self.trade_log),
        )
        return self.equity_curve, self.trade_log

    # ------------------------------------------------------------------
    # Private – bar dispatch
    # ------------------------------------------------------------------

    def _mark_to_market(self, bar_idx: int) -> None:
        """Recompute ``portfolio_value`` using the close price of bar_idx."""
        if self.position is None:
            return

        pos = self.position
        price_y = self._get_close_price(bar_idx, pos.sym_y)
        price_x = self._get_close_price(bar_idx, pos.sym_x)

        if price_y is None or price_x is None:
            return

        self.portfolio_value = (
            self.cash
            + pos.shares_y * price_y
            + pos.shares_x * price_x
        )

    def _handle_flat(
        self, bar_idx: int, prev_idx: int, current_date: pd.Timestamp
    ) -> None:
        """Scan for a new pair and open a position when the portfolio is flat."""
        if bar_idx < self._next_scan_idx:
            return

        # Signal window: close[window_start .. prev_idx] (does NOT include bar_idx)
        window_start = max(0, prev_idx - self.lookback + 1)
        hist = slice(window_start, prev_idx + 1)

        # Build log-price dict with enough history
        min_bars = self.lookback // 2
        log_window: dict[str, pd.Series] = {
            sym: self.log_prices[sym].iloc[hist].dropna()
            for sym in self.symbols
            if self.log_prices[sym].iloc[hist].notna().sum() >= min_bars
        }

        if len(log_window) < 2:
            self._next_scan_idx = bar_idx + self.rescan_interval
            return

        best = find_best_pair(log_window)

        if best is None:
            self._next_scan_idx = bar_idx + self.rescan_interval
            return

        sym_y, sym_x, pvalue = best
        self._try_open_position(
            bar_idx, prev_idx, current_date, sym_y, sym_x, pvalue, hist
        )

    def _handle_active(
        self, bar_idx: int, prev_idx: int, current_date: pd.Timestamp
    ) -> None:
        """Manage an open position: check exits, then rebalance if staying."""
        pos = self.position
        pos.bars_held += 1

        # Execution prices for this bar: fills at open (no look-ahead)
        exec_y = self._get_open_price(bar_idx, pos.sym_y)
        exec_x = self._get_open_price(bar_idx, pos.sym_x)

        if exec_y is None or exec_x is None:
            return  # missing data – skip this bar

        # Incremental Kalman update using prev bar's close (already known)
        prev_close_y = self._get_close_price(prev_idx, pos.sym_y)
        prev_close_x = self._get_close_price(prev_idx, pos.sym_x)

        if prev_close_y is None or prev_close_x is None:
            return

        kf = self._get_kf(pos.sym_y, pos.sym_x)
        new_beta: float = float(
            kf.update(np.log(prev_close_y), np.log(prev_close_x))
        )
        new_beta = max(new_beta, 1e-4)

        # Signal window: close[window_start .. prev_idx]
        window_start = max(0, prev_idx - self.lookback + 1)
        hist = slice(window_start, prev_idx + 1)

        log_y_arr = self.log_prices[pos.sym_y].iloc[hist].values
        log_x_arr = self.log_prices[pos.sym_x].iloc[hist].values
        spread_arr = compute_spread(log_y_arr, log_x_arr, new_beta)

        zscore = compute_zscore(spread_arr, window=self.lookback)

        # OU half-life refresh (from prev-close window)
        ou = estimate_ou_parameters(spread_arr)
        new_hl = float(np.clip(ou["half_life"], MIN_HALFLIFE, MAX_HALFLIFE))
        pos.half_life = new_hl

        # Check exit signal (stop-loss evaluated first)
        exit_flag, exit_reason = should_exit(
            zscore=zscore,
            bars_held=pos.bars_held,
            half_life=pos.half_life,
            halflife_multiplier=OU_HALFLIFE_MULTIPLIER,
            exit_threshold=self._z_exit,
            stop_loss_threshold=Z_STOP_LOSS,
        )

        if exit_flag:
            self._close_position(bar_idx, current_date, exit_reason)
            return

        # Rebalance: recalculate target sizes; execute at open prices
        new_shares_y, new_shares_x = compute_position_size(
            spread_series=spread_arr,
            portfolio_value=self.portfolio_value,
            price_y=exec_y,
            price_x=exec_x,
            hedge_ratio=new_beta,
            direction=pos.direction,
        )

        # Always update the KF-derived hedge ratio even if no rebalance executes
        pos.hedge_ratio = new_beta

        if new_shares_y == 0.0 or new_shares_x == 0.0:
            return

        # Only rebalance when at least one leg deviates by more than the threshold.
        # This prevents fee bleed from tiny hourly β drift.
        dev_y = (
            abs(new_shares_y - pos.shares_y) / abs(pos.shares_y)
            if abs(pos.shares_y) > 1e-10
            else 0.0
        )
        dev_x = (
            abs(new_shares_x - pos.shares_x) / abs(pos.shares_x)
            if abs(pos.shares_x) > 1e-10
            else 0.0
        )

        if dev_y <= REBALANCE_THRESHOLD and dev_x <= REBALANCE_THRESHOLD:
            return  # deviation too small — skip rebalance, incur no cost

        delta_y = new_shares_y - pos.shares_y
        delta_x = new_shares_x - pos.shares_x
        rebal_cost = (
            TRANSACTION_COST * abs(delta_y * exec_y)
            + TRANSACTION_COST * abs(delta_x * exec_x)
        )

        self.cash -= delta_y * exec_y + delta_x * exec_x + rebal_cost
        pos.cumulative_rebalance_cost += rebal_cost
        pos.shares_y = new_shares_y
        pos.shares_x = new_shares_x

    # ------------------------------------------------------------------
    # Private – open / close helpers
    # ------------------------------------------------------------------

    def _try_open_position(
        self,
        bar_idx: int,
        prev_idx: int,
        current_date: pd.Timestamp,
        sym_y: str,
        sym_x: str,
        pvalue: float,
        hist: slice,
    ) -> None:
        """Calibrate on prev-close data; open position at current-bar open."""
        # Execution prices (fills at open of current bar)
        exec_y = self._get_open_price(bar_idx, sym_y)
        exec_x = self._get_open_price(bar_idx, sym_x)

        if exec_y is None or exec_x is None:
            return

        # Kalman filter: batch calibration on prev-close lookback window
        kf = self._get_kf(sym_y, sym_x)
        log_y_arr = self.log_prices[sym_y].iloc[hist].values
        log_x_arr = self.log_prices[sym_x].iloc[hist].values
        hedge_ratios = kf.batch_estimate(log_y_arr, log_x_arr)
        current_beta = float(hedge_ratios[-1])
        if not np.isfinite(current_beta) or abs(current_beta) < 0.01:
            self._next_scan_idx = bar_idx + self.rescan_interval
            return

        # Spread and OU parameters (prev-close window only)
        spread_arr = compute_spread(log_y_arr, log_x_arr, current_beta)
        ou = estimate_ou_parameters(spread_arr)

        # Strictly reject pairs whose mean-reversion is too slow
        if ou["half_life"] > MAX_HALFLIFE:
            logger.debug(
                "%s  half_life=%.1f exceeds MAX_HALFLIFE=%.0f for %s/%s – skip.",
                current_date,
                ou["half_life"],
                MAX_HALFLIFE,
                sym_y,
                sym_x,
            )
            self._next_scan_idx = bar_idx + self.rescan_interval
            return

        half_life = float(np.clip(ou["half_life"], MIN_HALFLIFE, MAX_HALFLIFE))

        # Regime filter (fitted on prev-close spread)
        self._regime.fit(spread_arr)
        if not self._regime.is_mean_reverting(spread_arr):
            logger.debug(
                "%s  Regime not mean-reverting for %s/%s – skip.",
                current_date,
                sym_y,
                sym_x,
            )
            self._next_scan_idx = bar_idx + self.rescan_interval
            return

        # Z-score and entry direction (prev-close spread)
        # Momentum filter inside get_entry_direction requires spread_arr
        zscore = compute_zscore(spread_arr, window=self.lookback)
        direction = get_entry_direction(
            zscore=zscore,
            spread_series=spread_arr,
            threshold=self._z_entry,
        )

        if direction == 0:
            self._next_scan_idx = bar_idx + self.rescan_interval
            return

        # Volatility-targeted sizing; share counts based on open execution prices
        shares_y, shares_x = compute_position_size(
            spread_series=spread_arr,
            portfolio_value=self.portfolio_value,
            price_y=exec_y,
            price_x=exec_x,
            hedge_ratio=current_beta,
            direction=direction,
        )

        if shares_y == 0.0 or shares_x == 0.0:
            return

        # Entry cost (0.1 % per leg on open-price notional)
        entry_cost = (
            TRANSACTION_COST * abs(shares_y * exec_y)
            + TRANSACTION_COST * abs(shares_x * exec_x)
        )

        entry_portfolio_value = self.portfolio_value

        # Fill both legs at open price, deduct costs
        self.cash -= shares_y * exec_y + shares_x * exec_x + entry_cost
        self.portfolio_value -= entry_cost

        self.position = Position(
            sym_y=sym_y,
            sym_x=sym_x,
            direction=direction,
            shares_y=shares_y,
            shares_x=shares_x,
            entry_date=current_date,
            entry_price_y=exec_y,
            entry_price_x=exec_x,
            hedge_ratio=current_beta,
            half_life=half_life,
            coint_pvalue=pvalue,
            entry_portfolio_value=entry_portfolio_value,
            entry_cost=entry_cost,
        )

        logger.info(
            "%s  OPEN  %s/%s  dir=%+d  z=%.2f  hl=%.1fh  β=%.4f  "
            "p=%.4f  shares_y=%.4f  shares_x=%.4f",
            current_date,
            sym_y,
            sym_x,
            direction,
            zscore,
            half_life,
            current_beta,
            pvalue,
            shares_y,
            shares_x,
        )

    def _close_position(
        self,
        bar_idx: int,
        current_date: pd.Timestamp,
        reason: str,
        use_close_price: bool = False,
    ) -> None:
        """
        Unwind the current position, compute P&L, log the trade, reset state.

        Fills are executed at the **open price** of bar_idx by default.
        Pass ``use_close_price=True`` for the forced end-of-backtest close,
        which uses the close price of the last bar instead.
        """
        if self.position is None:
            return

        pos = self.position

        if use_close_price:
            exit_y = self._get_close_price(bar_idx, pos.sym_y) or pos.entry_price_y
            exit_x = self._get_close_price(bar_idx, pos.sym_x) or pos.entry_price_x
        else:
            exit_y = self._get_open_price(bar_idx, pos.sym_y) or pos.entry_price_y
            exit_x = self._get_open_price(bar_idx, pos.sym_x) or pos.entry_price_x

        # Exit transaction costs
        exit_cost = (
            TRANSACTION_COST * abs(pos.shares_y * exit_y)
            + TRANSACTION_COST * abs(pos.shares_x * exit_x)
        )

        # Unwind: convert position to cash at exit prices
        self.cash += pos.shares_y * exit_y + pos.shares_x * exit_x - exit_cost
        self.portfolio_value = self.cash  # position fully liquidated

        # --- P&L attribution ---
        gross_pnl = (
            pos.shares_y * (exit_y - pos.entry_price_y)
            + pos.shares_x * (exit_x - pos.entry_price_x)
        )
        total_costs = (
            pos.entry_cost
            + pos.cumulative_rebalance_cost
            + exit_cost
            + pos.cumulative_funding_cost
        )
        net_pnl = self.portfolio_value - pos.entry_portfolio_value
        return_pct = (
            net_pnl / pos.entry_portfolio_value
            if pos.entry_portfolio_value > 0
            else 0.0
        )

        record = TradeRecord(
            entry_date=str(pos.entry_date.date()),
            exit_date=str(current_date.date()),
            sym_y=pos.sym_y,
            sym_x=pos.sym_x,
            direction="long_spread" if pos.direction == 1 else "short_spread",
            bars_held=pos.bars_held,
            coint_pvalue=round(pos.coint_pvalue, 6),
            half_life=round(pos.half_life, 2),
            hedge_ratio=round(pos.hedge_ratio, 6),
            entry_price_y=round(pos.entry_price_y, 6),
            entry_price_x=round(pos.entry_price_x, 6),
            exit_price_y=round(exit_y, 6),
            exit_price_x=round(exit_x, 6),
            gross_pnl=round(gross_pnl, 4),
            entry_cost=round(pos.entry_cost, 4),
            rebalance_cost=round(pos.cumulative_rebalance_cost, 4),
            exit_cost=round(exit_cost, 4),
            funding_cost=round(pos.cumulative_funding_cost, 4),
            total_costs=round(total_costs, 4),
            net_pnl=round(net_pnl, 4),
            return_pct=round(return_pct, 6),
            exit_reason=reason,
            entry_portfolio_value=round(pos.entry_portfolio_value, 2),
            exit_portfolio_value=round(self.portfolio_value, 2),
        )
        self.trade_log.append(record)

        logger.info(
            "%s  CLOSE %s/%s  reason=%-16s  net_pnl=%s  pv=%s",
            current_date,
            pos.sym_y,
            pos.sym_x,
            reason,
            f"{net_pnl:+,.2f}",
            f"{self.portfolio_value:,.2f}",
        )

        self.position = None
        # Allow immediate re-scan after closing a trade
        self._next_scan_idx = 0

    # ------------------------------------------------------------------
    # Private – utility
    # ------------------------------------------------------------------

    def _build_price_matrices(
        self, raw_data: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align all symbol close and open price series into two DataFrames.

        Both matrices share the same index and column set.  Only bars where
        at least ``len(symbols) // 2`` assets have data are kept.  Isolated
        NaN gaps of up to 3 bars are forward-filled.
        """
        close_dict = {
            sym: df["close"]
            for sym, df in raw_data.items()
            if "close" in df.columns and not df["close"].empty
        }
        open_dict = {
            sym: df["open"]
            for sym, df in raw_data.items()
            if "open" in df.columns and not df["open"].empty
        }

        if not close_dict:
            raise ValueError("raw_data contains no usable close-price series.")

        close_matrix = pd.DataFrame(close_dict)

        # Drop rows with too many missing values (governed by close data)
        min_valid = max(1, len(close_matrix.columns) // 2)
        close_matrix = close_matrix.dropna(thresh=min_valid)
        close_matrix = close_matrix.ffill(limit=3)
        close_matrix = close_matrix.dropna(axis=1, how="all")
        close_matrix = close_matrix.sort_index()

        # Build open matrix aligned to the same index / columns
        valid_symbols = list(close_matrix.columns)
        open_matrix = pd.DataFrame(
            {sym: open_dict[sym] for sym in valid_symbols if sym in open_dict}
        )
        open_matrix = open_matrix.reindex(index=close_matrix.index)
        open_matrix = open_matrix.ffill(limit=3)
        # Fall back to close price where open is missing
        open_matrix = open_matrix.combine_first(close_matrix[valid_symbols])
        open_matrix = open_matrix[valid_symbols]

        logger.info(
            "Price matrices built: %d symbols, %d bars",
            close_matrix.shape[1],
            close_matrix.shape[0],
        )
        return close_matrix, open_matrix

    def _get_close_price(self, bar_idx: int, symbol: str) -> float | None:
        """Return the close price of bar_idx for symbol, or None if missing."""
        if symbol not in self.close_prices.columns:
            return None
        val = self.close_prices.iloc[bar_idx][symbol]
        return float(val) if np.isfinite(val) and val > 0 else None

    def _get_open_price(self, bar_idx: int, symbol: str) -> float | None:
        """Return the open price of bar_idx for symbol, or None if missing."""
        if symbol not in self.open_prices.columns:
            return None
        val = self.open_prices.iloc[bar_idx][symbol]
        return float(val) if np.isfinite(val) and val > 0 else None

    def _get_kf(self, sym_y: str, sym_x: str) -> KalmanFilter:
        """
        Return the Kalman Filter for the (sym_y, sym_x) pair.
        Creates a fresh filter if this pair has never been seen before.
        """
        key = (sym_y, sym_x)
        if key not in self._kalman_filters:
            self._kalman_filters[key] = KalmanFilter(
                delta=KALMAN_DELTA, ve=KALMAN_VE
            )
        return self._kalman_filters[key]
