"""End-to-end pipeline verification script for NexusQuant Phase 2.

Wires together:
    1. MarketDataFetcher  — fetches 100 candles of BTC/USDT 1h from Binance.
    2. FeatureEngineer    — enriches the raw OHLCV data with technical indicators.
    3. DimmerForceAgent   — generates a trading signal from the enriched data.

Run from the repository root with:
    conda run -n trading_bot python run_pipeline_test.py
"""

from __future__ import annotations

import logging
import sys

# ---------------------------------------------------------------------------
# Logging — configure before importing project modules so all loggers inherit
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("pipeline_test")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.agents.dimmer_force import DimmerForceAgent
from src.data.features import FeatureEngineer
from src.data.fetcher import MarketDataFetcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL: str = "BTC/USDT"
TIMEFRAME: str = "1h"
LIMIT: int = 100
DISPLAY_TAIL: int = 3


def main() -> None:
    """Runs the full Phase 2 pipeline verification."""
    logger.info("=" * 60)
    logger.info("NexusQuant — Phase 2 Pipeline Verification")
    logger.info("=" * 60)

    # ── Step 1: Fetch raw OHLCV data ─────────────────────────────────────────
    logger.info("Step 1 — Fetching %d candles of %s [%s]", LIMIT, SYMBOL, TIMEFRAME)
    fetcher = MarketDataFetcher()
    raw_df = fetcher.fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    logger.info("Raw DataFrame shape: %s", raw_df.shape)

    # ── Step 2: Feature engineering ──────────────────────────────────────────
    logger.info("Step 2 — Running FeatureEngineer")
    engineer = FeatureEngineer()
    enriched_df = engineer.add_technical_indicators(raw_df)
    logger.info("Enriched DataFrame shape: %s", enriched_df.shape)
    logger.info("Columns: %s", enriched_df.columns.tolist())

    # ── Step 3: Generate agent signal ────────────────────────────────────────
    logger.info("Step 3 — Running DimmerForceAgent")
    agent = DimmerForceAgent()
    signal = agent.generate_signal(enriched_df)

    # ── Results ──────────────────────────────────────────────────────────────
    separator = "-" * 60
    print(f"\n{separator}")
    print("  TRADING SIGNAL")
    print(separator)
    for key, value in signal.items():
        print(f"  {key:<20}: {value}")

    print(f"\n{separator}")
    print(f"  LAST {DISPLAY_TAIL} ENRICHED CANDLES")
    print(separator)
    print(enriched_df.tail(DISPLAY_TAIL).to_string())
    print(separator)

    logger.info("Pipeline verification completed successfully.")


if __name__ == "__main__":
    main()
