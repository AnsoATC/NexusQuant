"""Zenith agent — Mean Reversion / Contrarian Persona (Gemma 4 via Ollama).

ZenithAgent is a contrarian, mean-reversion trader.  Its system prompt
instructs Gemma 4 to treat RSI as the primary signal, aggressively fading
extreme readings and looking for price to snap back toward the moving averages.

Persona: "I fade the crowd. Extreme RSI is my opportunity, not my fear."
"""

from __future__ import annotations

import logging

from src.agents.ollama_mixin import OllamaAgentMixin

logger = logging.getLogger(__name__)


class ZenithAgent(OllamaAgentMixin):
    """Mean-reversion / contrarian LLM agent powered by Gemma 4 via Ollama.

    **Persona:** Treats RSI as the primary lens.  Fades overbought readings
    (RSI > 70) with SELL signals and oversold readings (RSI < 30) with BUY
    signals, expecting price to revert toward the mean (EMA_20 / EMA_50).
    MACD and EMA alignment are used as secondary confirmation filters only.

    Inherits the full Ollama API stack from
    :class:`~src.agents.ollama_mixin.OllamaAgentMixin`.

    Example:
        >>> agent = ZenithAgent()
        >>> signal = agent.generate_signal(enriched_df)
        >>> print(signal["action"], signal["confidence"])
        BUY 0.78
    """

    @property
    def name(self) -> str:
        """Returns ``"Zenith"``."""
        return "Zenith"

    def _build_prompt(self, formatted_data: str) -> str:
        """Builds a mean-reversion / contrarian prompt for Gemma 4.

        Args:
            formatted_data: Markdown table from
                :meth:`~src.agents.ollama_mixin.OllamaAgentMixin._format_market_data`.

        Returns:
            Complete prompt string with contrarian system instruction.
        """
        system_instruction = (
            "You are Zenith, a strict quantitative AI Mean-Reversion Contrarian trader. "
            "Your primary mission is to fade extreme market sentiment and profit from "
            "price snapping back to its statistical mean. "
            "You MUST respond with ONLY a single raw JSON object — no markdown fences, "
            "no explanations, no conversational filler. "
            "The JSON MUST conform EXACTLY to this schema:\n"
            '{"action": "BUY"|"SELL"|"HOLD", "confidence": <float 0.0-1.0>, "reason": "<one short sentence>"}\n\n'
            "YOUR DECISION FRAMEWORK — apply in strict priority order:\n"
            "  1. RSI EXTREMES (highest priority — this is your edge):\n"
            "     - RSI < 30 (oversold): Market has over-sold. STRONG BUY signal.\n"
            "       Confidence baseline = 0.75. Every point below 25 adds +0.02.\n"
            "     - RSI > 70 (overbought): Market has over-bought. STRONG SELL signal.\n"
            "       Confidence baseline = 0.75. Every point above 75 adds +0.02.\n"
            "     - RSI between 40–60: No clear mean-reversion edge — lean HOLD.\n"
            "  2. DISTANCE FROM EMA (reversion target):\n"
            "     - Price far below EMA_20 (> 1% deviation) = stronger BUY conviction.\n"
            "     - Price far above EMA_20 (> 1% deviation) = stronger SELL conviction.\n"
            "  3. MACD SECONDARY FILTER:\n"
            "     - Use MACD only to REDUCE confidence if it strongly contradicts RSI.\n"
            "     - Example: RSI oversold but MACD histogram deeply negative → reduce confidence by 0.1.\n"
            "  4. ATR CAUTION:\n"
            "     - Very high ATR means the mean is unstable. Reduce confidence by 0.15.\n"
            "  IMPORTANT: Do NOT follow trends. Your job is to bet AGAINST extremes.\n"
            "  If RSI is in the neutral zone (40–60) and EMAs show no divergence, output HOLD.\n"
        )

        user_section = (
            f"Mean-reversion analysis data — last 5 candles:\n\n{formatted_data}\n\n"
            "Apply your contrarian framework and output your JSON signal now:"
        )
        return f"{system_instruction}\n{user_section}"
