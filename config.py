"""
Global configuration for the Dynamic Multi-Asset Pair Trading Backtest Engine.
All constants are centralised here to avoid magic numbers throughout the code.
"""

# ---------------------------------------------------------------------------
# Universe — curated for 2022: liquid coins that existed all year and are
# still traded today under the same ticker.
# Explicitly excluded: LUNA (collapsed May 2022), FTT (collapsed Nov 2022),
# MATIC (rebranded to POL), and any coin launched after 2022-01-01.
# ---------------------------------------------------------------------------
TOP_40_SYMBOLS: list[str] = [
    "BTC/USDT",  "ETH/USDT",  "BNB/USDT",  "SOL/USDT",  "XRP/USDT",
    "ADA/USDT",  "DOGE/USDT", "TRX/USDT",  "LTC/USDT",  "AVAX/USDT",
    "DOT/USDT",  "LINK/USDT", "ATOM/USDT", "XMR/USDT",  "UNI/USDT",
    "ETC/USDT",  "BCH/USDT",  "NEAR/USDT", "ALGO/USDT", "VET/USDT",
    "ICP/USDT",  "GRT/USDT",  "XLM/USDT",  "SAND/USDT", "MANA/USDT",
    "AAVE/USDT", "EOS/USDT",  "THETA/USDT","XTZ/USDT",
]

# ---------------------------------------------------------------------------
# Exchange / data
# ---------------------------------------------------------------------------
EXCHANGE_ID: str = "binance"
TIMEFRAME: str = "1h"
START_DATE: str = "2022-01-01T00:00:00Z"   # Inclusive start for data fetch
END_DATE: str   = "2022-12-31T23:59:59Z"   # Inclusive end for data fetch

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 5_000.0           # USD — retail aggressive account
LOOKBACK_WINDOW: int = 336                 # Hours used for rolling calibration (14 days)
MIN_HISTORY: int = 500                     # Bars before the engine starts trading
RESCAN_INTERVAL: int = 5                   # Bars between pair-scan attempts when flat

# ---------------------------------------------------------------------------
# Strategy / signals
# ---------------------------------------------------------------------------
Z_ENTRY: float = 2.0                       # |z-score| required to open a trade
Z_EXIT: float = 0.25                       # |z-score| at which we take profit
Z_STOP_LOSS: float = 4.0                   # |z-score| hard stop-loss (spread blow-up)
TARGET_VOL: float = 0.45                   # Target annualised portfolio volatility (aggressive)
MAX_LEVERAGE: float = 3.0                  # Hard cap on notional / portfolio_value

# ---------------------------------------------------------------------------
# Risk / costs
# ---------------------------------------------------------------------------
TRANSACTION_COST: float = 0.001            # 0.10 % per leg, applied on notional change
HOURLY_FUNDING_RATE: float = 0.0000125     # Approx 0.01 % per 8 h funding fee per hour
REBALANCE_THRESHOLD: float = 0.10          # Min fractional share deviation to trigger rebalance
OU_HALFLIFE_MULTIPLIER: float = 2.0        # Time-stop multiplier on OU half-life
MIN_HALFLIFE: float = 2.0                  # Minimum half-life accepted (hours)
MAX_HALFLIFE: float = 96.0                 # Maximum half-life cap (4 days in hours)

# ---------------------------------------------------------------------------
# Cointegration filter
# ---------------------------------------------------------------------------
COINT_PVALUE_THRESHOLD: float = 0.05       # Pairs with p >= threshold are rejected

# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------
KALMAN_DELTA: float = 1e-4                 # State-noise / (1-state-noise), controls drift
KALMAN_VE: float = 1e-3                    # Observation-noise adaptive learning rate

# ---------------------------------------------------------------------------
# Hidden Markov Model
# ---------------------------------------------------------------------------
HMM_N_COMPONENTS: int = 2
HMM_COVARIANCE_TYPE: str = "full"
HMM_N_ITER: int = 100
