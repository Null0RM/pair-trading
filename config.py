"""
Global configuration for the Dynamic Multi-Asset Pair Trading Backtest Engine.
All constants are centralised here to avoid magic numbers throughout the code.
"""

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
TOP_40_SYMBOLS: list[str] = [
    "BTC/USDT",  "ETH/USDT",  "BNB/USDT",  "XRP/USDT",  "ADA/USDT",
    "SOL/USDT",  "DOGE/USDT", "TRX/USDT",  "LTC/USDT",  "AVAX/USDT",
    "DOT/USDT",  "MATIC/USDT","LINK/USDT", "ATOM/USDT", "XMR/USDT",
    "UNI/USDT",  "ETC/USDT",  "BCH/USDT",  "APT/USDT",  "FIL/USDT",
    "ARB/USDT",  "NEAR/USDT", "ALGO/USDT", "VET/USDT",  "ICP/USDT",
    "HBAR/USDT", "GRT/USDT",  "XLM/USDT",  "SAND/USDT", "MANA/USDT",
    "AXS/USDT",  "THETA/USDT","EOS/USDT",  "AAVE/USDT", "XTZ/USDT",
    "FTM/USDT",  "CAKE/USDT", "EGLD/USDT", "CHZ/USDT",  "ONE/USDT",
]

# ---------------------------------------------------------------------------
# Exchange / data
# ---------------------------------------------------------------------------
EXCHANGE_ID: str = "binance"
TIMEFRAME: str = "1h"
SINCE_DAYS: int = 180           # How many calendar days of history to fetch

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 1_000_000.0   # USD
LOOKBACK_WINDOW: int = 336              # Hours used for rolling calibration (14 days)
MIN_HISTORY: int = 500                  # Bars before the engine starts trading
RESCAN_INTERVAL: int = 5               # Days between pair-scan attempts when flat

# ---------------------------------------------------------------------------
# Strategy / signals
# ---------------------------------------------------------------------------
Z_ENTRY: float = 2.0                    # |z-score| required to open a trade
Z_EXIT: float = 0.25                    # |z-score| at which we take profit
Z_STOP_LOSS: float = 4.0               # |z-score| hard stop-loss (spread blow-up)
TARGET_VOL: float = 0.15               # Target annualised portfolio volatility
MAX_LEVERAGE: float = 2.0              # Hard cap on notional / portfolio_value

# ---------------------------------------------------------------------------
# Risk / costs
# ---------------------------------------------------------------------------
TRANSACTION_COST: float = 0.001        # 0.10 % per leg, applied on notional change
HOURLY_FUNDING_RATE: float = 0.0000125 # Approx 0.01 % per 8 h funding fee per hour
REBALANCE_THRESHOLD: float = 0.10      # Min fractional share deviation to trigger rebalance
OU_HALFLIFE_MULTIPLIER: float = 2.0    # Time-stop multiplier on OU half-life
MIN_HALFLIFE: float = 2.0              # Minimum half-life accepted (hours)
MAX_HALFLIFE: float = 96.0             # Maximum half-life cap (4 days in hours)

# ---------------------------------------------------------------------------
# Cointegration filter
# ---------------------------------------------------------------------------
COINT_PVALUE_THRESHOLD: float = 0.05   # Pairs with p >= threshold are rejected

# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------
KALMAN_DELTA: float = 1e-4             # State-noise / (1-state-noise), controls drift
KALMAN_VE: float = 1e-3               # Observation-noise adaptive learning rate

# ---------------------------------------------------------------------------
# Hidden Markov Model
# ---------------------------------------------------------------------------
HMM_N_COMPONENTS: int = 2
HMM_COVARIANCE_TYPE: str = "full"
HMM_N_ITER: int = 100
