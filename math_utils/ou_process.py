"""
math_utils/ou_process.py
------------------------
Ornstein-Uhlenbeck (OU) process parameter estimation.

Model
-----
The continuous-time OU process:

    dS_t = θ (μ - S_t) dt + σ dW_t

is discretised (Euler-Maruyama, dt = 1 day) to:

    ΔS_t  =  a  +  b · S_{t-1}  +  ε_t

    where  a = θμ,  b = -θ   (so b < 0 for mean-reversion)

An OLS regression of ``ΔS`` on ``S_{t-1}`` (plus intercept) yields
estimates of *a* and *b*, from which we recover:

    θ  = -b  (mean-reversion speed, must be > 0)
    μ  =  a / θ
    σ  =  std(residuals)

Half-life of mean reversion:
    τ½ = ln(2) / θ  days
"""

from __future__ import annotations

import logging

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

logger = logging.getLogger(__name__)


def estimate_ou_parameters(
    spread: np.ndarray | list,
    dt: float = 1.0,
) -> dict[str, float]:
    """
    Estimate OU parameters (θ, μ, σ) and the implied mean-reversion half-life.

    Parameters
    ----------
    spread:
        1-D array of spread values (e.g. log-price difference).
    dt:
        Time step between observations (default 1 day).

    Returns
    -------
    dict with keys
        ``theta``     – mean-reversion speed (annualised when dt = 1/252)
        ``mu``        – long-run mean
        ``sigma``     – diffusion coefficient (daily std of residuals)
        ``half_life`` – half-life in units of *dt* (days when dt=1)
    """
    spread = np.asarray(spread, dtype=float)

    _FALLBACK = {"theta": 1e-6, "mu": 0.0, "sigma": 0.0, "half_life": np.inf}

    if len(spread) < 5:
        return _FALLBACK

    delta_s = np.diff(spread)           # ΔS_t  (length n-1)
    lagged_s = spread[:-1]              # S_{t-1}

    # Remove any NaN pairs
    mask = np.isfinite(delta_s) & np.isfinite(lagged_s)
    if mask.sum() < 4:
        return _FALLBACK

    X = add_constant(lagged_s[mask])    # [1, S_{t-1}]
    y = delta_s[mask]

    try:
        result = OLS(y, X).fit()
    except Exception as exc:
        logger.debug("OLS failed in OU estimation: %s", exc)
        return _FALLBACK

    a: float = float(result.params[0])   # intercept  → θμ
    b: float = float(result.params[1])   # slope      → -θ

    # b must be negative for a mean-reverting process
    if b >= 0.0:
        # Non-mean-reverting; return very slow reversion
        return {"theta": 1e-6, "mu": float(np.mean(spread)), "sigma": float(np.std(spread)), "half_life": np.inf}

    theta: float = -b / dt                                  # mean-reversion speed
    mu: float = a / (theta * dt) if theta > 0 else float(np.mean(spread))
    sigma: float = float(np.std(result.resid, ddof=1))
    half_life: float = float(np.log(2.0) / theta)

    return {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life,
    }
