"""Savinov statistical trading agent for NexusQuant.

SavinovAgent implements a momentum/mean-reversion statistical strategy.
The current implementation is a structured stub that returns deterministic
mock signals based on the latest RSI and EMA relationship so the multi-agent
dashboard has meaningful variety to display.

The full statistical model (z-score regime detection, Kalman filter,
co-integration pair selection) will be wired in a future phase.
"""

from __future__ import annotations

import logging
import random

import pandas as pd

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Statistical thresholds (stub constants — will become configurable)
# ---------------------------------------------------------------------------
RSI_OVERSOLD_THRESHOLD: float = 35.0
RSI_OVERBOUGHT_THRESHOLD: float = 65.0
RANDOM_SEED_OFFSET: int = 42   # keeps results reproducible within a session


class SavinovAgent(BaseAgent):
    """Statistical momentum/mean-reversion trading agent.

    **Current Phase — Structured Stub:** Uses simple RSI thresholds combined
    with a seeded pseudo-random confidence value to produce realistic-looking
    mock signals.  This allows the multi-agent dashboard to display varied
    outputs while the full Kalman-filter / z-score model is developed.

    Strategy concept (to be fully implemented):
        * Regime detection via rolling z-score of close prices.
        * Mean-reversion entry when z-score > ±2 σ.
        * Momentum entry when price is above both EMAs with rising MACD.
        * Position sizing via Kelly Criterion fraction.

    Attributes:
        rsi_oversold: RSI level below which a BUY bias is generated.
        rsi_overbought: RSI level above which a SELL bias is generated.

    Example:
        >>> agent = SavinovAgent()
        >>> signal = agent.generate_signal(enriched_df)
        >>> print(signal["action"], signal["confidence"])
        BUY 0.74
    """

    def __init__(
        self,
        rsi_oversold: float = RSI_OVERSOLD_THRESHOLD,
        rsi_overbought: float = RSI_OVERBOUGHT_THRESHOLD,
    ) -> None:
        """Initialises SavinovAgent with configurable RSI thresholds.

        Args:
            rsi_oversold: RSI level treated as oversold (BUY bias). Defaults
                to ``35.0``.
            rsi_overbought: RSI level treated as overbought (SELL bias).
                Defaults to ``65.0``.
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        logger.info(
            "SavinovAgent initialised — RSI thresholds: oversold=%.1f  overbought=%.1f",
            self.rsi_oversold,
            self.rsi_overbought,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Returns the agent identifier.

        Returns:
            ``"Savinov"``
        """
        return "Savinov"

    def generate_signal(self, market_data: pd.DataFrame) -> dict:
        """Generates a mock statistical signal from the latest indicator values.

        **Stub logic:**
            * RSI < ``rsi_oversold``  → BUY bias.
            * RSI > ``rsi_overbought`` → SELL bias.
            * Otherwise               → HOLD.
        Confidence is a seeded pseudo-random float in [0.55, 0.90] so the
        dashboard shows realistic variation without being fully random on each
        call.

        Args:
            market_data: Enriched OHLCV DataFrame from
                :class:`~src.data.features.FeatureEngineer`.

        Returns:
            Signal dict conforming to the :class:`~src.agents.base.BaseAgent`
            schema with an additional ``"strategy"`` metadata key.
        """
        latest = market_data.iloc[-1]
        rsi = float(latest.get("RSI_14", 50.0))
        close = float(latest.get("close", 0.0))
        ema_20 = float(latest.get("EMA_20", close))
        ema_50 = float(latest.get("EMA_50", close))

        # Seed from latest close price integer part → reproducible within a candle
        rng = random.Random(int(close) + RANDOM_SEED_OFFSET)
        confidence = round(rng.uniform(0.55, 0.90), 2)

        if rsi < self.rsi_oversold:
            action = "BUY"
            reason = (
                f"RSI ({rsi:.1f}) below oversold threshold ({self.rsi_oversold}) "
                "signals mean-reversion long opportunity."
            )
        elif rsi > self.rsi_overbought:
            action = "SELL"
            reason = (
                f"RSI ({rsi:.1f}) above overbought threshold ({self.rsi_overbought}) "
                "signals mean-reversion short opportunity."
            )
        elif close > ema_20 > ema_50:
            action = "BUY"
            reason = "Bullish EMA alignment: price > EMA_20 > EMA_50 confirms uptrend momentum."
        elif close < ema_20 < ema_50:
            action = "SELL"
            reason = "Bearish EMA alignment: price < EMA_20 < EMA_50 confirms downtrend momentum."
        else:
            action = "HOLD"
            reason = "No clear statistical edge detected — remaining flat."
            confidence = round(rng.uniform(0.40, 0.60), 2)

        signal = {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "agent": self.name,
            "strategy": "statistical_stub",
            "latest_rsi": round(rsi, 2),
        }

        self.validate_signal(signal)
        logger.info(
            "[%s] Signal: action=%s  confidence=%.2f  RSI=%.2f",
            self.name, action, confidence, rsi,
        )
        return signal
