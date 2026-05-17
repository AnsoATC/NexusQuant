"""DimmerForce agent — Trend-Following Persona (Gemma 4 via Ollama).

DimmerForceAgent is a momentum/trend-following trader.  Its system prompt
instructs Gemma 4 to prioritise MACD histogram direction and EMA alignment
as the primary decision signals, while using RSI and ATR as secondary filters.

Persona: "I follow the trend.  The MACD and EMAs are my primary lens."
"""

from __future__ import annotations

import logging

from src.agents.ollama_mixin import OllamaAgentMixin

logger = logging.getLogger(__name__)


class DimmerForceAgent(OllamaAgentMixin):
    """Trend-following LLM agent powered by Gemma 4 via Ollama.

    **Persona:** Focuses on MACD histogram momentum and EMA price structure.
    Enters long (BUY) when MACD turns positive and price is above both EMAs;
    exits (SELL) when MACD histogram turns negative or price breaks below EMAs.

    Inherits the full Ollama API stack from
    :class:`~src.agents.ollama_mixin.OllamaAgentMixin`. Only the prompt
    system instruction is persona-specific.

    Example:
        >>> agent = DimmerForceAgent()
        >>> signal = agent.generate_signal(enriched_df)
        >>> print(signal["action"], signal["confidence"])
        BUY 0.82
    """

    @property
    def name(self) -> str:
        """Returns ``"DimmerForce"``."""
        return "DimmerForce"

    def _build_prompt(self, formatted_data: str) -> str:
        """Builds a trend-following prompt for Gemma 4.

        Args:
            formatted_data: Markdown table from
                :meth:`~src.agents.ollama_mixin.OllamaAgentMixin._format_market_data`.

        Returns:
            Complete prompt string with trend-following system instruction.
        """
        system_instruction = (
            "You are DimmerForce, a strict quantitative AI Trend-Following trader. "
            "Your primary mission is to identify and ride price momentum. "
            "You MUST respond with ONLY a single raw JSON object — no markdown fences, "
            "no explanations, no conversational filler. "
            "The JSON MUST conform EXACTLY to this schema:\n"
            '{"action": "BUY"|"SELL"|"HOLD", "confidence": <float 0.0-1.0>, "reason": "<one short sentence>"}\n\n'
            "YOUR DECISION FRAMEWORK — apply in strict priority order:\n"
            "  1. MACD HISTOGRAM (highest priority):\n"
            "     - MACDh crossing above 0 and rising = strong BUY signal.\n"
            "     - MACDh crossing below 0 and falling = strong SELL signal.\n"
            "     - MACDh flat or near 0 = lean toward HOLD.\n"
            "  2. EMA STRUCTURE (second priority):\n"
            "     - Price > EMA_20 > EMA_50 = bullish structure, supports BUY.\n"
            "     - Price < EMA_20 < EMA_50 = bearish structure, supports SELL.\n"
            "     - Mixed alignment = reduce confidence, consider HOLD.\n"
            "  3. RSI FILTER:\n"
            "     - Avoid BUY when RSI > 75 (overbought — trend exhaustion risk).\n"
            "     - Avoid SELL when RSI < 25 (oversold — bounce risk).\n"
            "  4. ATR VOLATILITY:\n"
            "     - High ATR (> 0.5% of price) = lower your confidence by 0.1.\n"
            "     - Trend must be confirmed by ALL top-priority signals to exceed 0.8 confidence.\n"
        )

        user_section = (
            f"Trend analysis data — last 5 candles:\n\n{formatted_data}\n\n"
            "Apply your trend-following framework and output your JSON signal now:"
        )
        return f"{system_instruction}\n{user_section}"
