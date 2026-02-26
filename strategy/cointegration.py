"""
strategy/cointegration.py
--------------------------
Engle-Granger cointegration screening for dynamic pair selection.

The engine calls ``find_best_pair`` every time the portfolio is flat and
a fresh pair is needed.  The function:

1.  Iterates over all C(n, 2) combinations of the available symbols.
2.  Drops non-finite observations via a numpy boolean mask (no pd.concat).
3.  Runs the Engle-Granger two-step cointegration test
    (``statsmodels.tsa.stattools.coint``).
4.  Returns the pair with the *lowest* p-value that is also below
    ``pvalue_threshold``.  If no pair qualifies, ``None`` is returned.

Performance note
----------------
For a 29-symbol universe there are 406 pairs.  The hot path extracts
``.values`` once per symbol before the loop, then uses
``np.isfinite(y) & np.isfinite(x)`` as a boolean mask to align the two
arrays — eliminating the ``pd.concat`` + ``join="inner"`` + ``.dropna()``
call that previously dominated the scan time (~100× faster per pair).
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import coint

from config import COINT_PVALUE_THRESHOLD

logger = logging.getLogger(__name__)


def find_best_pair(
    log_price_data: dict[str, pd.Series],
    pvalue_threshold: float = COINT_PVALUE_THRESHOLD,
    min_observations: int = 30,
) -> tuple[str, str, float] | None:
    """
    Scan all pairs and return the one with the lowest cointegration p-value.

    Parameters
    ----------
    log_price_data:
        ``{symbol: pd.Series}`` mapping of *log* close-price series.
        Series are expected to share the same index (as produced by the
        engine's price-matrix slice); gaps are handled via a per-pair
        ``np.isfinite`` mask rather than a pandas inner join.
    pvalue_threshold:
        Only pairs with ``p_value < threshold`` are considered.  If no pair
        qualifies the function returns ``None``.
    min_observations:
        Minimum number of aligned finite observations required to run the test.

    Returns
    -------
    (sym_y, sym_x, p_value)  or  None
        ``sym_y`` is the *dependent* asset (y-axis of the regression),
        ``sym_x`` is the *independent* asset.
    """
    symbols = list(log_price_data.keys())

    if len(symbols) < 2:
        logger.debug("Need at least 2 symbols for pair scanning; got %d.", len(symbols))
        return None

    # Extract numpy arrays once — avoids per-pair pandas overhead in the loop
    arrays: dict[str, np.ndarray] = {
        sym: log_price_data[sym].values for sym in symbols
    }

    best_pvalue: float = 1.0
    best_pair: tuple[str, str, float] | None = None

    for sym_y, sym_x in combinations(symbols, 2):
        y_raw = arrays[sym_y]
        x_raw = arrays[sym_x]

        # Boolean mask: retain only rows where both values are finite
        mask = np.isfinite(y_raw) & np.isfinite(x_raw)
        y = y_raw[mask]
        x = x_raw[mask]

        if len(y) < min_observations:
            continue

        # Skip constant series (cannot cointegrate)
        if np.std(y) < 1e-10 or np.std(x) < 1e-10:
            continue

        try:
            _, pvalue, _ = coint(y, x)
        except Exception as exc:
            logger.debug("coint() raised for %s/%s: %s", sym_y, sym_x, exc)
            continue

        # Require a positive OLS hedge ratio: both assets should move together.
        try:
            ols_beta = float(OLS(y, add_constant(x)).fit().params[1])
        except Exception:
            ols_beta = 0.0

        if ols_beta <= 0:
            continue  # skip pairs where the linear relationship inverts

        if pvalue < best_pvalue:
            best_pvalue = pvalue
            best_pair = (sym_y, sym_x, float(pvalue))

    if best_pair is None or best_pvalue >= pvalue_threshold:
        logger.debug(
            "No cointegrated pair found below threshold %.3f "
            "(best p=%.4f).",
            pvalue_threshold,
            best_pvalue,
        )
        return None

    logger.info(
        "Best cointegrated pair: %s / %s  p=%.4f",
        best_pair[0],
        best_pair[1],
        best_pair[2],
    )
    return best_pair
