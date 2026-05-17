"""DimmerForce agent module for NexusQuant.

Implements the first concrete trading agent, ``DimmerForceAgent``, which will
be powered by a local Gemma 4 LLM via Ollama.  In this phase the
``generate_signal`` method returns a structured dummy signal so that the full
pipeline can be wired and tested end-to-end before the LLM layer is added.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class DimmerForceAgent(BaseAgent):
    """LLM-powered trading agent driven by a local Gemma 4 model via Ollama.

    **Current Phase:** Stub implementation — ``generate_signal`` returns a
    deterministic dummy signal.  The Ollama API integration will be wired in
    the next development phase, replacing the stub body while keeping the
    public interface identical.

    Architecture note:
        ``DimmerForceAgent`` follows the Strategy Pattern defined by
        :class:`~src.agents.base.BaseAgent`.  The execution layer interacts
        exclusively with the ``BaseAgent`` interface, so swapping this agent
        for any other concrete implementation requires zero changes outside
        the agent itself.

    Attributes:
        model: The Ollama model identifier that will be used for LLM inference.
        ollama_url: Base URL of the running Ollama instance.

    Example:
        >>> agent = DimmerForceAgent()
        >>> signal = agent.generate_signal(enriched_df)
        >>> print(signal)
        {'action': 'HOLD', 'confidence': 1.0, 'reason': '...'}
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        """Initialises the DimmerForceAgent.

        Args:
            model: The Ollama model tag to use for LLM inference.  Defaults to
                ``"gemma3:4b"``.
            ollama_url: Base URL of the local Ollama server.  Defaults to
                ``"http://localhost:11434"``.
        """
        self.model = model
        self.ollama_url = ollama_url
        logger.info(
            "DimmerForceAgent initialised — model=%s  ollama_url=%s",
            self.model,
            self.ollama_url,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Returns the agent's identifier.

        Returns:
            ``"DimmerForce"``
        """
        return "DimmerForce"

    def generate_signal(self, market_data: pd.DataFrame) -> dict:
        """Generates a trading signal from enriched market data.

        **Phase 2 stub:** Logs key statistics from the latest candle and
        returns a deterministic HOLD signal.  In Phase 3 this method will
        build a structured prompt from ``market_data``, send it to the local
        Gemma 4 model via the Ollama REST API, parse the JSON response, and
        return the real LLM-derived signal.

        Args:
            market_data: Enriched :class:`pandas.DataFrame` produced by
                :class:`~src.data.features.FeatureEngineer`.  Must contain at
                least ``close``, ``RSI_14``, ``EMA_20``, and ``EMA_50`` columns.

        Returns:
            A signal ``dict`` conforming to the :class:`~src.agents.base.BaseAgent`
            schema::

                {
                    "action":     "HOLD",
                    "confidence": 1.0,
                    "reason":     "Awaiting LLM integration (Phase 3).",
                    "agent":      "DimmerForce",
                    "model":      "<ollama model tag>",
                    "latest_close": <float>,
                }
        """
        latest = market_data.iloc[-1]

        logger.info(
            "[%s] Analysing %d enriched candles. Latest close=%.4f  RSI=%.2f  "
            "EMA_20=%.4f  EMA_50=%.4f",
            self.name,
            len(market_data),
            latest.get("close", float("nan")),
            latest.get("RSI_14", float("nan")),
            latest.get("EMA_20", float("nan")),
            latest.get("EMA_50", float("nan")),
        )

        signal = {
            "action": "HOLD",
            "confidence": 1.0,
            "reason": "Awaiting LLM integration (Phase 3).",
            "agent": self.name,
            "model": self.model,
            "latest_close": float(latest.get("close", float("nan"))),
        }

        self.validate_signal(signal)
        logger.info("[%s] Signal generated: %s", self.name, signal)
        return signal
