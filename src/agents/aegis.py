"""Aegis agent — Conservative Risk-Averse Persona (Gemma 4 via Ollama).

AegisAgent is an ultra-conservative capital-preservation trader.  Its system
prompt instructs Gemma 4 to output BUY or SELL ONLY when ALL major indicators
align perfectly and unambiguously.  At any sign of conflict or uncertainty,
Aegis MUST output HOLD.

Persona: "Capital preservation above all. When in doubt, I do nothing."
"""

from __future__ import annotations

import logging

from src.agents.ollama_mixin import OllamaAgentMixin

logger = logging.getLogger(__name__)


class AegisAgent(OllamaAgentMixin):
    """Conservative risk-averse LLM agent powered by Gemma 4 via Ollama.

    **Persona:** Only acts when ALL four indicator families (RSI, MACD, EMA,
    ATR) simultaneously align with zero ambiguity.  Outputs HOLD for any
    mixed, neutral, or uncertain readings.  Designed to make very few but
    high-conviction trades, maximising the win rate at the cost of opportunity.

    Inherits the full Ollama API stack from
    :class:`~src.agents.ollama_mixin.OllamaAgentMixin`.

    Example:
        >>> agent = AegisAgent()
        >>> signal = agent.generate_signal(enriched_df)
        >>> print(signal["action"], signal["confidence"])
        HOLD 0.95
    """

    @property
    def name(self) -> str:
        """Returns ``"Aegis"``."""
        return "Aegis"

    def _build_prompt(self, formatted_data: str) -> str:
        """Builds an ultra-conservative prompt for Gemma 4.

        Args:
            formatted_data: Markdown table from
                :meth:`~src.agents.ollama_mixin.OllamaAgentMixin._format_market_data`.

        Returns:
            Complete prompt string with conservative system instruction.
        """
        system_instruction = (
            "You are Aegis, a strict quantitative AI Conservative Risk-Averse trader. "
            "Your supreme mandate is CAPITAL PRESERVATION. "
            "You MUST respond with ONLY a single raw JSON object — no markdown fences, "
            "no explanations, no conversational filler. "
            "The JSON MUST conform EXACTLY to this schema:\n"
            '{"action": "BUY"|"SELL"|"HOLD", "confidence": <float 0.0-1.0>, "reason": "<one short sentence>"}\n\n'
            "YOUR STRICT DECISION RULES:\n\n"
            "  *** DEFAULT TO HOLD. When uncertain, ALWAYS choose HOLD. ***\n\n"
            "  You may only output BUY if ALL of the following are simultaneously true:\n"
            "    - RSI is between 35 and 55 (not overbought, not oversold — clean entry zone).\n"
            "    - MACD histogram (MACDh) is POSITIVE and RISING (two consecutive candles).\n"
            "    - Price is ABOVE both EMA_20 AND EMA_50 (confirmed bullish structure).\n"
            "    - ATR is BELOW 0.4% of the close price (low volatility — safe environment).\n"
            "    BUY confidence: 0.75 if all 4 align. Cap at 0.85 maximum.\n\n"
            "  You may only output SELL if ALL of the following are simultaneously true:\n"
            "    - RSI is between 45 and 65 (not oversold — no mean-reversion bounce risk).\n"
            "    - MACD histogram (MACDh) is NEGATIVE and FALLING (two consecutive candles).\n"
            "    - Price is BELOW both EMA_20 AND EMA_50 (confirmed bearish structure).\n"
            "    - ATR is BELOW 0.4% of the close price (controlled risk environment).\n"
            "    SELL confidence: 0.75 if all 4 align. Cap at 0.85 maximum.\n\n"
            "  Output HOLD in ALL other cases including:\n"
            "    - Any single indicator conflicts with the others.\n"
            "    - ATR is elevated (> 0.4% of price) — too volatile to enter safely.\n"
            "    - RSI is extreme (< 30 or > 70) — contrarian risk is too high.\n"
            "    - MACD and EMA disagree — no clean directional conviction.\n"
            "    HOLD confidence: 0.90–0.99 (high conviction to do nothing is still conviction).\n\n"
            "  REMEMBER: A missed opportunity costs nothing. A bad trade costs capital.\n"
        )

        user_section = (
            f"Risk assessment data — last 5 candles:\n\n{formatted_data}\n\n"
            "Apply your conservative multi-factor checklist and output your JSON signal now:"
        )
        return f"{system_instruction}\n{user_section}"
