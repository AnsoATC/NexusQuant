"""End-to-end pipeline verification script for NexusQuant Phase 4.

Wires together the full stack:
    1. MarketDataFetcher  — fetches 100 candles of BTC/USDT 1h from Binance.
    2. FeatureEngineer    — enriches raw OHLCV data with technical indicators.
    3. DimmerForceAgent   — queries Gemma 4 via Ollama for a trading signal.
    4. RiskManager        — computes ATR-based position sizing.
    5. PaperTrader        — simulates order execution and portfolio tracking.

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
from src.execution.paper_trader import PaperTrader
from src.execution.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL: str = "BTC/USDT"
TIMEFRAME: str = "1h"
LIMIT: int = 100
DISPLAY_TAIL: int = 3
STARTING_CAPITAL: float = 100.0


def _print_section(title: str, data: dict, width: int = 60) -> None:
    """Prints a labelled key-value section to stdout.

    Args:
        title: Section header string.
        data: Dictionary of field → value pairs to display.
        width: Total width of the separator line.
    """
    sep = "-" * width
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    for key, value in data.items():
        if isinstance(value, float):
            print(f"  {key:<26}: {value:.6g}")
        else:
            print(f"  {key:<26}: {value}")
    print(sep)


def main() -> None:
    """Runs the full Phase 4 pipeline verification."""
    logger.info("=" * 60)
    logger.info("NexusQuant — Phase 4 Pipeline Verification")
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

    # ── Step 3: Generate agent signal (Gemma 4 via Ollama) ───────────────────
    logger.info("Step 3 — Running DimmerForceAgent (Gemma 4)")
    agent = DimmerForceAgent()
    signal = agent.generate_signal(enriched_df)

    # ── Step 4: Extract latest price and ATR for risk sizing ─────────────────
    latest_candle = enriched_df.iloc[-1]
    current_price: float = float(latest_candle["close"])
    latest_atr: float = float(latest_candle["ATRr_14"])
    logger.info(
        "Step 4 — Extracted latest close=%.4f USDT  ATR=%.4f",
        current_price,
        latest_atr,
    )

    # ── Step 5: Risk Manager — calculate position size ────────────────────────
    logger.info("Step 5 — Running RiskManager")
    risk_manager = RiskManager()
    risk_metrics = risk_manager.calculate_position_size(
        signal_action=signal["action"],
        current_price=current_price,
        atr=latest_atr,
        available_capital=STARTING_CAPITAL,
    )

    # ── Step 6: Paper Trader — execute simulated order ────────────────────────
    logger.info("Step 6 — Running PaperTrader (starting balance=%.2f USDT)", STARTING_CAPITAL)
    trader = PaperTrader(starting_balance_usdt=STARTING_CAPITAL)
    trader.process_signal(signal, current_price=current_price, risk_metrics=risk_metrics, symbol=SYMBOL)

    # Simulate SL/TP check at the same price (no movement since we just opened)
    trader.update_positions(current_price=current_price, symbol=SYMBOL)

    # ── Print results ─────────────────────────────────────────────────────────
    _print_section("TRADING SIGNAL (Gemma 4)", signal)
    _print_section("RISK METRICS", risk_metrics)
    _print_section("PORTFOLIO SUMMARY", trader.portfolio_summary)

    print(f"\n{'-' * 60}")
    print(f"  LAST {DISPLAY_TAIL} ENRICHED CANDLES")
    print("-" * 60)
    display_cols = ["close", "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "EMA_20", "EMA_50", "ATRr_14"]
    available_cols = [c for c in display_cols if c in enriched_df.columns]
    print(enriched_df[available_cols].tail(DISPLAY_TAIL).round(4).to_string())
    print("-" * 60)

    logger.info("Pipeline verification completed successfully.")
    logger.info(
        "Final account balance: %.4f USDT (started with %.2f USDT)",
        trader.balance_usdt,
        STARTING_CAPITAL,
    )


if __name__ == "__main__":
    main()
