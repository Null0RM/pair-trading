"""
math_utils/kalman_filter.py
---------------------------
One-dimensional Kalman Filter for dynamic hedge-ratio estimation.

Model
-----
State equation (random walk for the hedge ratio β):
    β_t = β_{t-1} + w_t,    w_t ~ N(0, Q)

Observation equation:
    y_t = β_t · x_t + v_t,  v_t ~ N(0, R)

where
    y_t  = log-price of the *dependent*   asset at time t
    x_t  = log-price of the *independent* asset at time t
    β_t  = latent hedge ratio (state variable)
    Q    = state-noise  variance  (controls how fast β can drift)
    R    = observation-noise variance (adapted on-line via an EMA)

Implementation notes
--------------------
* The filter is deliberately scalar (1-D) to keep it fast.
* ``delta`` is the state-noise-to-signal ratio used in the original
  Pairs Trading Kalman Filter formulation (Pole et al. 1994 / Vidyamurthy 2004).
  A smaller delta means the hedge ratio adapts more slowly.
* ``ve`` is the EMA learning rate for the adaptive observation noise.
* Call ``batch_estimate`` for offline calibration and ``update`` for the
  live (online) tick-by-tick update.
"""

from __future__ import annotations

import numpy as np

from config import KALMAN_DELTA, KALMAN_VE


class KalmanFilter:
    """
    Scalar Kalman Filter for estimating a dynamic hedge ratio.

    Parameters
    ----------
    delta:
        Controls how rapidly the hedge ratio can change between steps.
        Corresponds to the state noise variance  Q = delta / (1 - delta).
    ve:
        EMA learning rate for the adaptive observation noise variance R.
    """

    def __init__(
        self,
        delta: float = KALMAN_DELTA,
        ve: float = KALMAN_VE,
    ) -> None:
        self.delta = delta
        self.ve = ve

        # Process noise covariance (constant, derived from delta)
        self._Q: float = delta / (1.0 - delta)

        # State estimate and its variance (error covariance)
        self._beta: float | None = None
        self._P: float = 0.0

        # Adaptive observation noise variance
        self._R: float | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, y: float, x: float) -> float:
        """
        Incorporate one new (y, x) observation and return the updated β.

        Parameters
        ----------
        y:
            Log-price of the dependent asset.
        x:
            Log-price of the independent asset.

        Returns
        -------
        float
            Updated hedge ratio estimate β_t.
        """
        if self._beta is None:
            # --- Initialise on first call ---
            # Fall back to 1.0; batch_estimate pre-seeds via OLS before calling here.
            self._beta = 1.0
            self._R = self.ve
            return self._beta

        # --- Predict step ---
        P_pred: float = self._P + self._Q

        # --- Innovation (residual) ---
        y_hat: float = self._beta * x
        innovation: float = y - y_hat

        # --- Innovation variance ---
        S: float = x * P_pred * x + self._R

        # --- Kalman gain ---
        K: float = P_pred * x / S if abs(S) > 1e-20 else 0.0

        # --- Update step ---
        self._beta += K * innovation
        self._P = (1.0 - K * x) * P_pred

        # --- Adaptive observation noise (EMA of squared innovation) ---
        self._R = (1.0 - self.ve) * self._R + self.ve * (innovation ** 2)

        return self._beta

    def batch_estimate(
        self,
        y_series: np.ndarray,
        x_series: np.ndarray,
    ) -> np.ndarray:
        """
        Run the filter over an entire price series (offline / calibration).

        The filter is reset and pre-seeded from OLS on the first 30 bars
        so the initial hedge-ratio estimate is stable regardless of whether
        log-prices are positive or negative (e.g. for coins priced < $1).

        Parameters
        ----------
        y_series:
            1-D array of log-prices for the dependent asset.
        x_series:
            1-D array of log-prices for the independent asset.

        Returns
        -------
        np.ndarray
            Array of hedge-ratio estimates, one per observation.
        """
        self.reset()
        n = len(y_series)
        betas = np.empty(n, dtype=float)

        # ---- OLS seed (avoids sign-flip when log-prices are negative) ----
        init_n = min(n, 30)
        x_init = x_series[:init_n]
        y_init = y_series[:init_n]
        if np.std(x_init) > 1e-10:
            # OLS with intercept: y = β·x + α
            X = np.column_stack([x_init, np.ones(init_n)])
            try:
                coef, _, _, _ = np.linalg.lstsq(X, y_init, rcond=None)
                beta_ols = float(coef[0])
                if np.isfinite(beta_ols) and abs(beta_ols) > 1e-6:
                    resid = y_init - coef[0] * x_init - coef[1]
                    self._beta = beta_ols
                    self._R = max(float(np.var(resid)), self.ve)
            except np.linalg.LinAlgError:
                pass  # keep default beta = 1.0
        # -------------------------------------------------------------------

        for i in range(n):
            betas[i] = self.update(float(y_series[i]), float(x_series[i]))
        return betas

    def reset(self) -> None:
        """Reset the filter to an un-initialised state."""
        self._beta = None
        self._P = 0.0
        self._R = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hedge_ratio(self) -> float:
        """Current hedge-ratio estimate (1.0 before first ``update`` call)."""
        return self._beta if self._beta is not None else 1.0

    @property
    def error_covariance(self) -> float:
        """Current state error covariance P_t."""
        return self._P
