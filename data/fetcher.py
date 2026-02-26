"""
data/fetcher.py
---------------
Pre-fetch OHLCV data for the full symbol universe into memory so the
backtesting loop never hits the exchange API.

Design notes
------------
* Pagination is handled automatically: CCXT limits one request to ~1 000 bars,
  so we loop until no new bars arrive or we exceed END_DATE.
* Symbols that error out (delisted, not listed on the exchange, etc.) are
  silently skipped – a warning is logged.
* All timestamps are stored in UTC.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from config import END_DATE, EXCHANGE_ID, START_DATE, TIMEFRAME, TOP_40_SYMBOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_all_ohlcv(
    symbols: list[str] | None = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    exchange_id: str = EXCHANGE_ID,
    timeframe: str = TIMEFRAME,
) -> dict[str, pd.DataFrame]:
    """
    Pre-fetch OHLCV data for every symbol in *symbols* between *start_date*
    and *end_date* (both inclusive) and return a mapping ``{symbol: DataFrame}``.

    The DataFrame columns are ``["open", "high", "low", "close", "volume"]``
    with a UTC-aware DatetimeIndex.

    Parameters
    ----------
    symbols:
        List of CCXT-style trading pairs, e.g. ``["BTC/USDT", "ETH/USDT"]``.
        Defaults to ``TOP_40_SYMBOLS`` from *config*.
    start_date:
        ISO 8601 UTC string for the start of the fetch window, e.g.
        ``"2022-01-01T00:00:00Z"``.
    end_date:
        ISO 8601 UTC string for the end of the fetch window (inclusive), e.g.
        ``"2022-12-31T23:59:59Z"``.
    exchange_id:
        CCXT exchange identifier (must support ``fetch_ohlcv``).
    timeframe:
        CCXT timeframe string, e.g. ``"1h"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Only symbols for which at least one valid bar was returned.
    """
    if symbols is None:
        symbols = TOP_40_SYMBOLS

    exchange: ccxt.Exchange = _build_exchange(exchange_id)

    since_ms: int = _parse_dt_ms(start_date)
    end_ms: int = _parse_dt_ms(end_date)

    data: dict[str, pd.DataFrame] = {}
    total = len(symbols)

    for idx, symbol in enumerate(symbols, start=1):
        logger.info("[%d/%d] Fetching %s ...", idx, total, symbol)
        try:
            raw = _paginate_ohlcv(exchange, symbol, timeframe, since_ms, end_ms)
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


def _parse_dt_ms(dt_str: str) -> int:
    """Parse an ISO 8601 UTC datetime string to a millisecond timestamp."""
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1_000)


def _paginate_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: int,
    page_limit: int = 1_000,
) -> list[list]:
    """
    Collect all OHLCV bars for *symbol* in the window [since_ms, end_ms] by
    issuing successive requests until the exchange returns fewer bars than
    *page_limit* or a fetched batch extends past *end_ms*.

    Parameters
    ----------
    since_ms:
        Start of the window (milliseconds UTC, inclusive).
    end_ms:
        End of the window (milliseconds UTC, inclusive).  Bars beyond this
        timestamp are clipped before being added to the result.
    page_limit:
        Max bars per request.  Most exchanges cap at 500–1 000.

    Returns
    -------
    list[list]
        Concatenated raw OHLCV lists: ``[timestamp_ms, O, H, L, C, V]``,
        all with ``timestamp_ms <= end_ms``.
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

        # Clip bars that fall beyond the requested end date
        clipped = [bar for bar in batch if bar[0] <= end_ms]
        all_bars.extend(clipped)

        # Stop if this was the last exchange page or we've passed end_ms
        if len(batch) < page_limit or len(clipped) < len(batch):
            break

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
