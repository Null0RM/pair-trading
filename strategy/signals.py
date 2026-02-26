"""
strategy/signals.py
--------------------
Signal generation and volatility-targeted position sizing.

Spread definition
-----------------
    spread_t = log_y_t  -  β_t · log_x_t

where β_t is the Kalman-filtered hedge ratio.

Z-score
-------
    z_t = (spread_t - μ_w) / σ_w

where μ_w and σ_w are the rolling mean and std over the last *window* bars
of the spread series.

Volatility targeting
--------------------
We want the annualised P&L volatility of the position to equal
``target_vol × portfolio_value``.

    hourly spread return = Δspread_t = spread_t - spread_{t-1}

    spread_hourly_vol = std(Δspread)  [hourly]

    notional (USD) = portfolio_value × target_vol / spread_annual_vol
                   = portfolio_value × target_vol / (spread_hourly_vol × √8760)

Then:
    shares_y =  direction × notional / price_y
    shares_x = -direction × β × notional / price_x

    (for long spread:  direction=+1  →  long y, short x)
    (for short spread: direction=-1  →  short y, long x)

Entry / exit rules
------------------
Entry  : |z| > Z_ENTRY   →  short spread if z > 0,  long if z < 0
         AND momentum filter: spread must have already begun reverting
           Long  entry (z < 0): spread[-1] > SMA(spread, momentum_window)
           Short entry (z > 0): spread[-1] < SMA(spread, momentum_window)
Exit   : |z| ≤ Z_EXIT    (mean-reversion target reached)
         OR  bars_held ≥ halflife_multiplier × half_life  (time-stop)
"""

from __future__ import annotations

import numpy as np

from config import (
    LOOKBACK_WINDOW,
    MAX_LEVERAGE,
    TARGET_VOL,
    Z_ENTRY,
    Z_EXIT,
)


# ---------------------------------------------------------------------------
# Spread helpers
# ---------------------------------------------------------------------------


def compute_spread(
    log_y: np.ndarray,
    log_x: np.ndarray,
    hedge_ratio: float,
) -> np.ndarray:
    """
    Compute the spread series  S_t = log_y_t - β · log_x_t.

    Parameters
    ----------
    log_y, log_x:
        Log close-price arrays of equal length.
    hedge_ratio:
        Scalar β (Kalman-filtered estimate at the current time step).

    Returns
    -------
    np.ndarray
        Spread array of the same length as the inputs.
    """
    return np.asarray(log_y, dtype=float) - hedge_ratio * np.asarray(log_x, dtype=float)


def compute_zscore(
    spread: np.ndarray,
    window: int = LOOKBACK_WINDOW,
) -> float:
    """
    Rolling z-score of the *last* element of *spread*.

    Uses the most recent *window* observations (or all available if fewer).

    Returns 0.0 when the standard deviation is negligibly small.
    """
    spread = np.asarray(spread, dtype=float)
    recent = spread[-window:] if len(spread) >= window else spread

    mu: float = float(np.mean(recent))
    sigma: float = float(np.std(recent, ddof=1))

    if sigma < 1e-10:
        return 0.0

    return float((spread[-1] - mu) / sigma)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def compute_position_size(
    spread_series: np.ndarray,
    portfolio_value: float,
    price_y: float,
    price_x: float,
    hedge_ratio: float,
    direction: int,
    target_vol: float = TARGET_VOL,
    max_leverage: float = MAX_LEVERAGE,
    window: int = LOOKBACK_WINDOW,
) -> tuple[float, float]:
    """
    Compute volatility-targeted share counts for both legs.

    Parameters
    ----------
    spread_series:
        Historical spread values up to and including the current bar.
    portfolio_value:
        Current total portfolio value in USD.
    price_y, price_x:
        Current close prices (not log) of the two assets.
    hedge_ratio:
        Kalman-filtered β at the current time step.
    direction:
        +1 → long spread (long y, short x)
        -1 → short spread (short y, long x)
    target_vol:
        Target annualised volatility as a decimal (e.g. 0.15 = 15 %).
    max_leverage:
        Hard cap on ``notional / portfolio_value``.
    window:
        Lookback window for vol estimation.

    Returns
    -------
    (shares_y, shares_x)
        Positive shares_y with long spread; negative shares_x with long spread.
        Returns (0.0, 0.0) if sizing cannot be computed.
    """
    if portfolio_value <= 0 or price_y <= 0 or price_x <= 0 or hedge_ratio <= 0:
        return 0.0, 0.0

    spread_series = np.asarray(spread_series, dtype=float)
    recent = spread_series[-window:] if len(spread_series) >= window else spread_series

    if len(recent) < 3:
        return 0.0, 0.0

    spread_returns = np.diff(recent)
    spread_returns = spread_returns[np.isfinite(spread_returns)]

    if len(spread_returns) < 2:
        return 0.0, 0.0

    bar_vol: float = float(np.std(spread_returns, ddof=1))
    annual_vol: float = bar_vol * np.sqrt(8760)

    if annual_vol < 1e-8:
        return 0.0, 0.0

    # Dollar notional to commit to this spread
    notional: float = portfolio_value * target_vol / annual_vol

    # Hard leverage cap
    notional = min(notional, max_leverage * portfolio_value)

    # Convert to share counts
    shares_y: float = direction * notional / price_y
    shares_x: float = -direction * hedge_ratio * notional / price_x

    return shares_y, shares_x


# ---------------------------------------------------------------------------
# Entry / exit signals
# ---------------------------------------------------------------------------


def get_entry_direction(
    zscore: float,
    spread_series: np.ndarray,
    threshold: float = Z_ENTRY,
    momentum_window: int = 6,
) -> int:
    """
    Determine trade direction from the z-score, gated by a momentum filter.

    The momentum filter requires evidence that the spread has already begun
    reverting toward its mean before an entry is taken.  This avoids
    "catching falling knives" when the spread is still moving away from mean.

    Momentum condition
    ------------------
    Short entry (z > +threshold): spread[-1] must be BELOW the short-term SMA
        → spread is curling *down* toward mean.
    Long  entry (z < -threshold): spread[-1] must be ABOVE the short-term SMA
        → spread is curling *up* toward mean.

    Parameters
    ----------
    zscore:
        Current z-score of the spread.
    spread_series:
        Historical spread values up to and including the current bar.
    threshold:
        |z-score| required to consider an entry.
    momentum_window:
        Number of recent bars used to compute the short-term SMA (default 6).

    Returns
    -------
    +1  long spread  (z < -threshold AND spread > short-term SMA)
    -1  short spread (z > +threshold AND spread < short-term SMA)
     0  no trade (z-score insufficient OR momentum condition not met)
    """
    spread_series = np.asarray(spread_series, dtype=float)
    if len(spread_series) < 2:
        return 0

    current_spread: float = float(spread_series[-1])
    window = min(momentum_window, len(spread_series))
    sma: float = float(np.mean(spread_series[-window:]))

    if zscore > threshold:
        # Short spread: spread must be curling DOWN → current value below SMA
        return -1 if current_spread < sma else 0

    if zscore < -threshold:
        # Long spread: spread must be curling UP → current value above SMA
        return 1 if current_spread > sma else 0

    return 0


def should_exit(
    zscore: float,
    bars_held: int,
    half_life: float,
    halflife_multiplier: float,
    exit_threshold: float = Z_EXIT,
    stop_loss_threshold: float = float("inf"),
) -> tuple[bool, str]:
    """
    Check whether the current position should be closed.

    Three exit conditions are evaluated in priority order:
    1.  **Stop-loss**           – |z| has blown out beyond *stop_loss_threshold*,
                                   indicating a structural cointegration break.
    2.  **Mean-reversion exit** – |z| has fallen below *exit_threshold*.
    3.  **Time-stop**           – position held longer than
                                   *halflife_multiplier × half_life* bars.

    Returns
    -------
    (exit_flag, reason_string)
        ``exit_flag`` is ``True`` if the position should be closed.
        ``reason_string`` is one of ``"stop_loss"``, ``"mean_reversion"``,
        ``"time_stop"``, or ``""``.
    """
    if abs(zscore) >= stop_loss_threshold:
        return True, "stop_loss"

    if abs(zscore) <= exit_threshold:
        return True, "mean_reversion"

    time_limit: float = halflife_multiplier * max(half_life, 1.0)
    if bars_held >= time_limit:
        return True, "time_stop"

    return False, ""
