"""Market data fetcher module for NexusQuant.

Provides the MarketDataFetcher class, which wraps the ccxt library to fetch
historical OHLCV data from Binance (or any ccxt-compatible exchange) and
returns it as a clean, typed pandas DataFrame.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OHLCV_COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_EXCHANGE: str = "binance"
DEFAULT_LIMIT: int = 500
MAX_RETRIES: int = 3
RETRY_BACKOFF_SECONDS: float = 2.0


class MarketDataFetcher:
    """Fetches historical OHLCV market data from a ccxt-compatible exchange.

    This class operates in read-only mode using only public API endpoints,
    meaning no API credentials are required for OHLCV data retrieval.

    Attributes:
        exchange_id: The ccxt exchange identifier (e.g. ``"binance"``).
        exchange: The initialised ccxt exchange instance.

    Example:
        >>> fetcher = MarketDataFetcher()
        >>> df = fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=100)
        >>> print(df.head())
    """

    def __init__(
        self,
        exchange_id: str = DEFAULT_EXCHANGE,
        timeout_ms: int = 30_000,
        rate_limit: bool = True,
    ) -> None:
        """Initialises the MarketDataFetcher and connects to the exchange.

        Args:
            exchange_id: A valid ccxt exchange id string. Defaults to
                ``"binance"``.
            timeout_ms: HTTP request timeout in milliseconds. Defaults to
                ``30000`` (30 seconds).
            rate_limit: Whether to enable ccxt's built-in rate limiter.
                Strongly recommended to keep as ``True`` to avoid bans.

        Raises:
            ccxt.ExchangeNotAvailable: If the specified exchange cannot be
                reached during initialisation.
            AttributeError: If ``exchange_id`` is not a valid ccxt exchange.
        """
        self.exchange_id = exchange_id
        logger.info("Initialising MarketDataFetcher for exchange: %s", exchange_id)

        try:
            exchange_class = getattr(ccxt, exchange_id)
        except AttributeError as exc:
            logger.error("Unknown ccxt exchange id: '%s'", exchange_id)
            raise AttributeError(
                f"'{exchange_id}' is not a recognised ccxt exchange id."
            ) from exc

        self.exchange: ccxt.Exchange = exchange_class(
            {
                "timeout": timeout_ms,
                "enableRateLimit": rate_limit,
            }
        )
        logger.info(
            "Exchange '%s' initialised successfully (rate_limit=%s, timeout=%dms).",
            exchange_id,
            rate_limit,
            timeout_ms,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetches historical OHLCV candle data for a given trading pair.

        Retries up to ``MAX_RETRIES`` times with exponential back-off on
        transient network errors before propagating the exception.

        Args:
            symbol: The trading pair symbol in ccxt format (e.g.
                ``"BTC/USDT"``).
            timeframe: The candle interval string recognised by the exchange
                (e.g. ``"1m"``, ``"15m"``, ``"1h"``, ``"1d"``).
            limit: Maximum number of candles to return. Defaults to
                ``500``. Exchange-specific maximums may apply.
            since: Optional Unix timestamp in **milliseconds** marking the
                start of the requested range. When ``None`` the exchange
                returns the most recent candles.

        Returns:
            A :class:`pandas.DataFrame` with a timezone-aware UTC
            :class:`pandas.DatetimeIndex` and the following columns::

                open  |  high  |  low  |  close  |  volume

        Raises:
            ccxt.BadSymbol: If ``symbol`` is not listed on the exchange.
            ccxt.NetworkError: If all retry attempts are exhausted due to
                network issues.
            ccxt.ExchangeError: On any unrecoverable exchange-side error.
        """
        logger.info(
            "Fetching OHLCV — symbol=%s  timeframe=%s  limit=%d",
            symbol,
            timeframe,
            limit,
        )

        self._validate_symbol(symbol)
        self._validate_timeframe(timeframe)

        raw_data = self._fetch_with_retry(symbol, timeframe, limit, since)

        if not raw_data:
            logger.warning(
                "Exchange returned an empty OHLCV response for %s / %s.",
                symbol,
                timeframe,
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS).set_index("timestamp")

        df = self._build_dataframe(raw_data)
        logger.info(
            "Successfully fetched %d candles for %s [%s → %s].",
            len(df),
            symbol,
            df.index[0],
            df.index[-1],
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_symbol(self, symbol: str) -> None:
        """Checks that the symbol exists on the exchange market list.

        Args:
            symbol: Trading pair in ccxt format (e.g. ``"BTC/USDT"``).

        Raises:
            ccxt.BadSymbol: If the symbol is not found on the exchange.
        """
        try:
            markets = self.exchange.load_markets()
        except ccxt.NetworkError as exc:
            logger.error("Network error while loading markets: %s", exc)
            raise

        if symbol not in markets:
            logger.error(
                "Symbol '%s' not found on %s. Available pairs: %d",
                symbol,
                self.exchange_id,
                len(markets),
            )
            raise ccxt.BadSymbol(
                f"Symbol '{symbol}' is not listed on {self.exchange_id}."
            )

    def _validate_timeframe(self, timeframe: str) -> None:
        """Checks that the timeframe is supported by the exchange.

        Args:
            timeframe: Candle interval string (e.g. ``"1h"``).

        Raises:
            ValueError: If the timeframe is not in the exchange's timeframe
                list.
        """
        supported = self.exchange.timeframes or {}
        if timeframe not in supported:
            logger.error(
                "Timeframe '%s' is not supported by %s. Supported: %s",
                timeframe,
                self.exchange_id,
                list(supported.keys()),
            )
            raise ValueError(
                f"Timeframe '{timeframe}' is not supported by {self.exchange_id}. "
                f"Supported timeframes: {list(supported.keys())}"
            )

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[int],
    ) -> list[list]:
        """Wraps the ccxt OHLCV call with retry logic and exponential back-off.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle interval string.
            limit: Max number of candles.
            since: Optional start timestamp in milliseconds.

        Returns:
            Raw list of ``[timestamp, open, high, low, close, volume]`` lists
            returned by ccxt.

        Raises:
            ccxt.NetworkError: After all retries are exhausted.
            ccxt.ExchangeError: On a non-retriable exchange error.
        """
        last_exception: Exception = RuntimeError("Unknown error before first attempt.")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                    since=since,
                )
            except ccxt.NetworkError as exc:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Network error on attempt %d/%d for %s. Retrying in %.1fs — %s",
                    attempt,
                    MAX_RETRIES,
                    symbol,
                    wait,
                    exc,
                )
                last_exception = exc
                time.sleep(wait)
            except ccxt.ExchangeError as exc:
                logger.error(
                    "Exchange error fetching OHLCV for %s: %s", symbol, exc
                )
                raise

        logger.error(
            "All %d retry attempts exhausted for %s / %s.",
            MAX_RETRIES,
            symbol,
            timeframe,
        )
        raise last_exception

    @staticmethod
    def _build_dataframe(raw_data: list[list]) -> pd.DataFrame:
        """Converts raw ccxt OHLCV lists into a clean, typed DataFrame.

        Args:
            raw_data: A list of ``[timestamp_ms, open, high, low, close, volume]``
                rows as returned by ccxt.

        Returns:
            A :class:`pandas.DataFrame` with a UTC-aware
            :class:`pandas.DatetimeIndex` named ``"timestamp"`` and float
            columns ``open``, ``high``, ``low``, ``close``, ``volume``.
        """
        df = pd.DataFrame(raw_data, columns=OHLCV_COLUMNS)

        # Convert millisecond epoch to UTC-aware datetime and use as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Ensure numeric types (guard against exchange returning strings)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop any rows where all price columns are NaN (malformed candles)
        df.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)

        return df
