"""DeepAlpha neural network trading agent for NexusQuant.

DeepAlphaAgent will ultimately be powered by a PyTorch LSTM / Transformer
model trained on historical OHLCV + indicator data.  The current phase
implements a structured stub that produces indicator-informed mock signals,
giving the multi-agent dashboard meaningful output while the model training
pipeline is developed.
"""

from __future__ import annotations

import logging
import random

import pandas as pd

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulated model confidence band
# ---------------------------------------------------------------------------
NN_HIGH_CONFIDENCE_BAND: tuple[float, float] = (0.70, 0.95)
NN_LOW_CONFIDENCE_BAND: tuple[float, float] = (0.45, 0.65)


class DeepAlphaAgent(BaseAgent):
    """Neural-network-powered trading agent (stub implementation).

    **Current Phase — Structured Stub:** Derives a directional bias from MACD
    histogram momentum and ATR-normalised volatility, then returns a
    pseudo-random confidence from a model-appropriate band.  This mimics the
    expected output distribution of a real NN model without requiring a trained
    checkpoint.

    Planned architecture (to be implemented):
        * Feature vector: 60-candle rolling window of [OHLCV + 7 indicators].
        * Model: Bidirectional LSTM → Attention → Dense(3) with softmax.
        * Output: Probability distribution over {BUY, HOLD, SELL}.
        * Confidence: ``max(softmax_probabilities)``.
        * Training: Supervised on labelled historical BTC/USDT data with
          walkforward validation.

    Attributes:
        model_version: Identifier of the model checkpoint in use.  Set to
            ``"stub_v0"`` until a real checkpoint is loaded.

    Example:
        >>> agent = DeepAlphaAgent()
        >>> signal = agent.generate_signal(enriched_df)
        >>> print(signal["action"], signal["confidence"])
        SELL 0.83
    """

    def __init__(self, model_version: str = "stub_v0") -> None:
        """Initialises DeepAlphaAgent.

        Args:
            model_version: Identifier for the neural network checkpoint.
                Defaults to ``"stub_v0"`` (no real checkpoint loaded).
        """
        self.model_version = model_version
        logger.info(
            "DeepAlphaAgent initialised — model_version=%s", self.model_version
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Returns the agent identifier.

        Returns:
            ``"DeepAlpha"``
        """
        return "DeepAlpha"

    def generate_signal(self, market_data: pd.DataFrame) -> dict:
        """Generates a mock neural-network signal from indicator momentum.

        **Stub logic:**
            * Reads MACD histogram (``MACDh_12_26_9``) as the primary momentum
              proxy — positive histogram → BUY, negative → SELL.
            * Uses ATR relative to close price to modulate confidence: high
              volatility → lower confidence band (less certain predictions).
            * Seeds the RNG from the latest MACD histogram value so results are
              reproducible within the same candle cycle.

        Args:
            market_data: Enriched OHLCV DataFrame from
                :class:`~src.data.features.FeatureEngineer`.

        Returns:
            Signal dict conforming to the :class:`~src.agents.base.BaseAgent`
            schema with additional ``"model_version"`` and ``"volatility_regime"``
            metadata keys.
        """
        latest = market_data.iloc[-1]
        macd_hist = float(latest.get("MACDh_12_26_9", 0.0))
        atr = float(latest.get("ATRr_14", 0.0))
        close = float(latest.get("close", 1.0))
        rsi = float(latest.get("RSI_14", 50.0))

        # Volatility regime: ATR as % of price
        atr_pct = (atr / close) * 100.0 if close > 0 else 0.0
        high_volatility = atr_pct > 0.5   # >0.5% of price = elevated volatility

        # Seed RNG from MACD histogram (scaled to int) for reproducibility
        rng = random.Random(int(abs(macd_hist) * 100))

        confidence_band = (
            NN_LOW_CONFIDENCE_BAND if high_volatility else NN_HIGH_CONFIDENCE_BAND
        )
        confidence = round(rng.uniform(*confidence_band), 2)
        volatility_regime = "HIGH" if high_volatility else "NORMAL"

        # Primary directional signal from MACD histogram
        if macd_hist > 0 and rsi < 70:
            action = "BUY"
            reason = (
                f"NN detects positive MACD momentum (hist={macd_hist:.2f}) "
                f"with RSI={rsi:.1f} not yet overbought."
            )
        elif macd_hist < 0 and rsi > 30:
            action = "SELL"
            reason = (
                f"NN detects negative MACD momentum (hist={macd_hist:.2f}) "
                f"with RSI={rsi:.1f} not yet oversold."
            )
        else:
            action = "HOLD"
            reason = "NN model detects conflicting signals — insufficient edge to act."
            confidence = round(rng.uniform(0.40, 0.55), 2)

        signal = {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "agent": self.name,
            "model_version": self.model_version,
            "volatility_regime": volatility_regime,
            "atr_pct": round(atr_pct, 4),
        }

        self.validate_signal(signal)
        logger.info(
            "[%s] Signal: action=%s  confidence=%.2f  MACDh=%.2f  vol_regime=%s",
            self.name, action, confidence, macd_hist, volatility_regime,
        )
        return signal
