"""Zenith agent — Mean Reversion / Grid Trading Persona.

ZenithAgent operates on a tight trading grid or strict mean reversion signals
using Bollinger Bands, RSI oversold/overbought boundaries, and multi-level grid intervals.
It places limit orders to scale into/out of positions.
"""

from __future__ import annotations

import logging
import pandas as pd

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ZenithAgent(BaseAgent):
    """Mean-reversion / Grid-trading programmatic agent.

    **Framework:**
      - Center Line (Mean): 20-period SMA.
      - Grid spacing: 0.6% intervals above and below the center line.
      - Buy dips: Places buy limit orders in micro-positions (allocation_fraction=0.33)
        when the price falls below the lower grid lines (or RSI <= 35).
      - Sell bounces: Places sell limit orders to scale out and lock in profits
        when the price rises above the upper grid lines (or RSI >= 65).
    """

    @property
    def name(self) -> str:
        """Returns ``"Zenith"``."""
        return "Zenith"

    def generate_signal(
        self,
        market_data: pd.DataFrame,
        current_position_size: float = 0.0,
    ) -> dict:
        """Generates a Grid Trading / Mean Reversion signal with limit price and allocation fraction.

        Args:
            market_data: Enriched OHLCV DataFrame with indicators.
            current_position_size: Current open position size in the base asset.

        Returns:
            Signal dict: ``{action, confidence, reason, price, allocation_fraction, roc_5}``.
        """
        if market_data.empty or len(market_data) < 20:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "Insufficient market data to compute grid indicators.",
            }

        latest = market_data.iloc[-1]
        current_price = float(latest["close"])

        # Defensively check/calculate Bollinger Bands
        if "BBM_20_2.0" in market_data.columns:
            bbm = float(latest["BBM_20_2.0"])
            bbl = float(latest["BBL_20_2.0"])
            bbu = float(latest["BBU_20_2.0"])
        else:
            # Fallback calculation
            ma = market_data["close"].rolling(window=20).mean()
            std = market_data["close"].rolling(window=20).std()
            bbm = float(ma.iloc[-1])
            bbl = float(ma.iloc[-1] - 2 * std.iloc[-1])
            bbu = float(ma.iloc[-1] + 2 * std.iloc[-1])

        # Defensively check/calculate RSI
        if "RSI_14" in market_data.columns:
            rsi = float(latest["RSI_14"])
        else:
            # Simple fallback RSI check
            rsi = 50.0

        # Define 3 levels of buy grid and 3 levels of sell grid
        grid_spacing = 0.006  # 0.6% spacing

        buy_grid_1 = bbm * (1.0 - 1 * grid_spacing)
        buy_grid_2 = bbm * (1.0 - 2 * grid_spacing)
        buy_grid_3 = bbm * (1.0 - 3 * grid_spacing)

        sell_grid_1 = bbm * (1.0 + 1 * grid_spacing)
        sell_grid_2 = bbm * (1.0 + 2 * grid_spacing)
        sell_grid_3 = bbm * (1.0 + 3 * grid_spacing)

        action = "HOLD"
        confidence = 0.0
        reason = f"Price (${current_price:.2f}) inside neutral channel. Mean=${bbm:.2f}, RSI={rsi:.1f}."
        price = None
        allocation_fraction = 1.0

        # Mean Reversion / Grid logic
        if current_price <= buy_grid_1 or current_price <= bbl or rsi <= 35:
            # DIP DETECTED -> BUY
            action = "BUY"
            if current_price <= buy_grid_3:
                price = buy_grid_3
                confidence = 0.95
                reason = f"Strong dip: Close ({current_price:.2f}) <= Buy Grid 3 ({buy_grid_3:.2f}). RSI={rsi:.1f}."
                allocation_fraction = 0.33
            elif current_price <= buy_grid_2:
                price = buy_grid_2
                confidence = 0.85
                reason = f"Moderate dip: Close ({current_price:.2f}) <= Buy Grid 2 ({buy_grid_2:.2f}). RSI={rsi:.1f}."
                allocation_fraction = 0.33
            else:
                price = buy_grid_1
                confidence = 0.75
                reason = f"Light dip: Close ({current_price:.2f}) <= Buy Grid 1 ({buy_grid_1:.2f}) or BBL ({bbl:.2f}). RSI={rsi:.1f}."
                allocation_fraction = 0.33
        elif current_price >= sell_grid_1 or current_price >= bbu or rsi >= 65:
            # BOUNCE DETECTED -> SELL (if holding position)
            if current_position_size > 0.0:
                action = "SELL"
                if current_price >= sell_grid_3:
                    price = sell_grid_3
                    confidence = 0.95
                    reason = f"Strong bounce: Close ({current_price:.2f}) >= Sell Grid 3 ({sell_grid_3:.2f}). RSI={rsi:.1f}."
                    allocation_fraction = 1.0  # Take full profit
                elif current_price >= sell_grid_2:
                    price = sell_grid_2
                    confidence = 0.85
                    reason = f"Moderate bounce: Close ({current_price:.2f}) >= Sell Grid 2 ({sell_grid_2:.2f}). RSI={rsi:.1f}."
                    allocation_fraction = 0.5  # Take partial profit
                else:
                    price = sell_grid_1
                    confidence = 0.75
                    reason = f"Light bounce: Close ({current_price:.2f}) >= Sell Grid 1 ({sell_grid_1:.2f}) or BBU ({bbu:.2f}). RSI={rsi:.1f}."
                    allocation_fraction = 0.33  # Scale out micro-position
            else:
                action = "HOLD"
                reason = f"Bounce detected but no position to sell. Close={current_price:.2f}, RSI={rsi:.1f}."

        signal = {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "price": round(price, 4) if price is not None else None,
            "allocation_fraction": allocation_fraction,
            "roc_5": round(float(latest.get("ROC_5", 0.0)), 2),
        }

        self.validate_signal(signal)
        logger.info("[%s] Programmatic Grid Signal: %s", self.name, signal)
        return signal
