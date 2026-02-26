"""
math_utils/regime.py
--------------------
Gaussian Hidden Markov Model (GaussianHMM) for market-regime classification.

The HMM operates on two features extracted from the spread time series:
    1. spread return    (first difference of the spread)
    2. |spread return|  (proxy for realised volatility)

Hidden states represent latent market regimes; we label the state with the
lowest absolute mean spread-return as the **mean-reverting** regime — the one
most conducive to statistical arbitrage.

When the current regime is NOT mean-reverting the engine is instructed to stay
flat (no new entries), reducing exposure during trending / noisy periods.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM

from config import HMM_COVARIANCE_TYPE, HMM_N_COMPONENTS, HMM_N_ITER

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Two-state (or *n*-state) Gaussian HMM regime detector.

    Usage
    -----
    >>> clf = RegimeClassifier()
    >>> clf.fit(spread_array)
    >>> regime = clf.predict_current_regime(spread_array)
    >>> if clf.is_mean_reverting(spread_array):
    ...     # proceed with trade entry
    """

    def __init__(
        self,
        n_components: int = HMM_N_COMPONENTS,
        covariance_type: str = HMM_COVARIANCE_TYPE,
        n_iter: int = HMM_N_ITER,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter

        self._model: GaussianHMM | None = None
        self._mean_reverting_state: int = 0     # default state index

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, spread: np.ndarray) -> "RegimeClassifier":
        """
        Fit the HMM on the provided spread series.

        Silently resets and returns ``self`` so the call can be chained.
        If fitting fails the object falls back to always returning state 0
        (treated as mean-reverting).

        Parameters
        ----------
        spread:
            1-D array of raw spread values.
        """
        features = self._build_features(spread)
        if features is None or len(features) < self.n_components * 10:
            self._model = None
            return self

        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
            tol=1e-4,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features)
            self._model = model
            self._identify_mean_reverting_state()
        except Exception as exc:
            logger.debug("HMM fit failed: %s", exc)
            self._model = None
        return self

    def predict_current_regime(self, spread: np.ndarray) -> int:
        """
        Return the regime index for the **last** observation in *spread*.

        Falls back to ``0`` if the model has not been fitted.
        """
        if self._model is None:
            return 0
        features = self._build_features(spread)
        if features is None or len(features) == 0:
            return 0
        try:
            states = self._model.predict(features)
            return int(states[-1])
        except Exception as exc:
            logger.debug("HMM predict failed: %s", exc)
            return 0

    def is_mean_reverting(self, spread: np.ndarray) -> bool:
        """
        Return ``True`` when the latest regime is the mean-reverting state.

        If the model has not been successfully fitted we default to ``True``
        (allow trading) so a failed HMM fit does not silently block all trades.
        """
        if self._model is None:
            return True
        return self.predict_current_regime(spread) == self._mean_reverting_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(self, spread: np.ndarray) -> np.ndarray | None:
        """
        Construct the (n-1, 2) feature matrix from the spread series.

        Features:
            col 0 – first-difference of spread (return)
            col 1 – absolute first-difference   (volatility proxy)
        """
        spread = np.asarray(spread, dtype=float)
        if len(spread) < 2:
            return None
        returns = np.diff(spread)
        valid = np.isfinite(returns)
        if valid.sum() < 2:
            return None
        r = returns[valid]
        return np.column_stack([r, np.abs(r)])

    def _identify_mean_reverting_state(self) -> None:
        """
        Label the HMM state whose mean spread-return is closest to zero as
        the *mean-reverting* state.  A purely mean-reverting spread has
        E[ΔS] ≈ 0, whereas a trending one has |E[ΔS]| > 0.
        """
        if self._model is None:
            return
        try:
            # means_ shape: (n_components, n_features);  col 0 is spread return
            abs_mean_returns = np.abs(self._model.means_[:, 0])
            self._mean_reverting_state = int(np.argmin(abs_mean_returns))
        except Exception:
            self._mean_reverting_state = 0
