"""Feature engineering module for NexusQuant.

Transforms raw OHLCV DataFrames into enriched, model-ready DataFrames by
appending a standardised set of technical indicators via the pandas-ta library.
"""

from __future__ import annotations

import logging

import importlib.metadata  # noqa: F401 — must precede pandas_ta on Python 3.10
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
EMA_SHORT: int = 20
EMA_LONG: int = 50
ATR_PERIOD: int = 14


class FeatureEngineer:
    """Enriches raw OHLCV DataFrames with technical indicators.

    All indicators are calculated using the ``pandas-ta`` library and appended
    as new columns to a *copy* of the input DataFrame so the original is never
    mutated.

    Indicators added by :meth:`add_technical_indicators`:

    * **RSI_14** — Relative Strength Index (momentum).
    * **MACD_12_26_9** — MACD line (trend / momentum).
    * **MACDh_12_26_9** — MACD histogram.
    * **MACDs_12_26_9** — MACD signal line.
    * **EMA_20** — Short-term exponential moving average (trend).
    * **EMA_50** — Long-term exponential moving average (trend).
    * **ATRr_14** — Average True Range (volatility).

    Example:
        >>> engineer = FeatureEngineer()
        >>> enriched_df = engineer.add_technical_indicators(raw_df)
        >>> print(enriched_df.columns.tolist())
    """

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appends technical indicators to a raw OHLCV DataFrame.

        Operates on a defensive copy so the caller's original DataFrame is
        never modified.  Rows that contain ``NaN`` values introduced by
        rolling-window warm-up periods are dropped before returning.

        Args:
            df: A :class:`pandas.DataFrame` with at minimum the columns
                ``open``, ``high``, ``low``, ``close``, and ``volume``.
                A UTC-aware :class:`pandas.DatetimeIndex` is expected (as
                produced by :class:`~src.data.fetcher.MarketDataFetcher`).

        Returns:
            A new :class:`pandas.DataFrame` containing all original columns
            plus the computed indicator columns, with any ``NaN``-containing
            rows removed.

        Raises:
            ValueError: If any of the required OHLCV columns are missing from
                ``df``.
        """
        self._validate_columns(df)

        logger.info(
            "Starting feature engineering on DataFrame with %d rows.", len(df)
        )

        enriched = df.copy()

        # ── RSI ──────────────────────────────────────────────────────────────
        enriched.ta.rsi(length=RSI_PERIOD, append=True)
        logger.debug("RSI_%d calculated.", RSI_PERIOD)

        # ── MACD ─────────────────────────────────────────────────────────────
        enriched.ta.macd(
            fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True
        )
        logger.debug(
            "MACD_%d_%d_%d calculated.", MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )

        # ── EMA (short & long) ───────────────────────────────────────────────
        enriched.ta.ema(length=EMA_SHORT, append=True)
        enriched.ta.ema(length=EMA_LONG, append=True)
        logger.debug("EMA_%d and EMA_%d calculated.", EMA_SHORT, EMA_LONG)

        # ── ATR ──────────────────────────────────────────────────────────────
        enriched.ta.atr(length=ATR_PERIOD, append=True)
        logger.debug("ATR_%d calculated.", ATR_PERIOD)

        rows_before = len(enriched)
        enriched.dropna(inplace=True)
        rows_dropped = rows_before - len(enriched)

        logger.info(
            "Feature engineering complete. %d rows produced (%d NaN warm-up rows dropped).",
            len(enriched),
            rows_dropped,
        )
        return enriched

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Ensures the required OHLCV columns are present in the DataFrame.

        Args:
            df: The DataFrame to validate.

        Raises:
            ValueError: If one or more required columns are absent.
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required OHLCV columns: {missing}"
            )
