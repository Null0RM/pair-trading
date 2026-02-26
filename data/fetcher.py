"""
data/fetcher.py
---------------
Pre-fetch OHLCV data for the full symbol universe into memory so the
backtesting loop never hits the exchange API.

Design notes
------------
* Pagination is handled automatically: CCXT limits one request to ~1 000 bars,
  so we loop until no new bars arrive.
* Symbols that error out (delisted, not listed on the exchange, etc.) are
  silently skipped – a warning is logged.
* All timestamps are stored in UTC.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from config import EXCHANGE_ID, SINCE_DAYS, TIMEFRAME, TOP_40_SYMBOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_all_ohlcv(
    symbols: list[str] | None = None,
    since_days: int = SINCE_DAYS,
    exchange_id: str = EXCHANGE_ID,
    timeframe: str = TIMEFRAME,
) -> dict[str, pd.DataFrame]:
    """
    Pre-fetch OHLCV data for every symbol in *symbols* and return a mapping
    ``{symbol: DataFrame}``.

    The DataFrame columns are ``["open", "high", "low", "close", "volume"]``
    with a UTC-aware DatetimeIndex (daily frequency after forward-filling).

    Parameters
    ----------
    symbols:
        List of CCXT-style trading pairs, e.g. ``["BTC/USDT", "ETH/USDT"]``.
        Defaults to ``TOP_40_SYMBOLS`` from *config*.
    since_days:
        How many calendar days of history to retrieve.
    exchange_id:
        CCXT exchange identifier (must support ``fetch_ohlcv``).
    timeframe:
        CCXT timeframe string, e.g. ``"1d"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Only symbols for which at least one valid bar was returned.
    """
    if symbols is None:
        symbols = TOP_40_SYMBOLS

    exchange: ccxt.Exchange = _build_exchange(exchange_id)

    since_ms: int = _since_ms(since_days)

    data: dict[str, pd.DataFrame] = {}
    total = len(symbols)

    for idx, symbol in enumerate(symbols, start=1):
        logger.info("[%d/%d] Fetching %s ...", idx, total, symbol)
        try:
            raw = _paginate_ohlcv(exchange, symbol, timeframe, since_ms)
            if raw:
                df = _to_dataframe(raw)
                if not df.empty:
                    data[symbol] = df
                    logger.info("  -> %d bars loaded.", len(df))
                else:
                    logger.warning("  -> Empty DataFrame for %s. Skipping.", symbol)
            else:
                logger.warning("  -> No data returned for %s. Skipping.", symbol)
        except (ccxt.BadSymbol, ccxt.ExchangeError) as exc:
            logger.warning("  -> Exchange error for %s: %s", symbol, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  -> Unexpected error for %s: %s", symbol, exc)

        # Respect the exchange rate-limit (milliseconds → seconds)
        time.sleep(exchange.rateLimit / 1_000.0)

    logger.info(
        "Data pre-fetch complete: %d / %d symbols loaded.", len(data), total
    )
    return data


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_exchange(exchange_id: str) -> ccxt.Exchange:
    """Instantiate and load markets for the requested exchange."""
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown CCXT exchange: '{exchange_id}'")
    exchange: ccxt.Exchange = exchange_class(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )
    try:
        exchange.load_markets()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load markets: %s", exc)
    return exchange


def _since_ms(since_days: int) -> int:
    """Return a UTC millisecond timestamp for *since_days* ago."""
    cutoff: datetime = datetime.now(timezone.utc) - timedelta(days=since_days)
    return int(cutoff.timestamp() * 1_000)


def _paginate_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    page_limit: int = 1_000,
) -> list[list]:
    """
    Collect all OHLCV bars for *symbol* starting from *since_ms* by issuing
    successive requests until the exchange returns fewer bars than *page_limit*
    (indicating there is no more data).

    Parameters
    ----------
    page_limit:
        Max bars per request.  Most exchanges cap at 500–1 000.

    Returns
    -------
    list[list]
        Concatenated raw OHLCV lists: ``[timestamp_ms, O, H, L, C, V]``.
    """
    all_bars: list[list] = []
    cursor_ms: int = since_ms
    max_retries: int = 3

    while True:
        for attempt in range(max_retries):
            try:
                batch: list[list] = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=cursor_ms,
                    limit=page_limit,
                )
                break
            except ccxt.NetworkError as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Network error (%s) for %s – retrying in %ds ...", exc, symbol, wait
                )
                time.sleep(wait)
        else:
            logger.error("Max retries exceeded for %s. Partial data returned.", symbol)
            break

        if not batch:
            break

        all_bars.extend(batch)

        if len(batch) < page_limit:
            break  # last page

        # Advance cursor past the last bar to avoid duplicates
        cursor_ms = batch[-1][0] + 1

    return all_bars


def _to_dataframe(raw: list[list]) -> pd.DataFrame:
    """
    Convert raw OHLCV list to a clean, indexed DataFrame.

    * Timestamps converted to UTC DatetimeIndex.
    * Duplicate indices removed (keep last).
    * Up to 3 consecutive NaN values forward-filled.
    * Rows where *close* is NaN are dropped.
    """
    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.ffill(limit=3)
    df = df.dropna(subset=["close"])
    return df
