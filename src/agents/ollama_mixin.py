"""Shared Ollama REST API mixin for NexusQuant LLM-powered agents.

All three Gemma 4 agents (DimmerForce, Zenith, Aegis) share identical
infrastructure for calling the Ollama API, parsing the response, formatting
market data as a Markdown table, and returning a safe fallback on error.

This mixin centralises that logic so subclasses only need to implement:
    * ``name`` property
    * ``_build_prompt()`` — the persona-specific system instruction
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from typing import Any

import pandas as pd
import requests

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants — all Ollama agents inherit these defaults
# ---------------------------------------------------------------------------
DEFAULT_API_URL: str = "http://localhost:11434/api/generate"
DEFAULT_MODEL: str = "gemma4:latest"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_REQUEST_TIMEOUT: int = 120  # seconds
CONTEXT_ROWS: int = 5               # candles sent to the LLM per call

FALLBACK_SIGNAL: dict[str, Any] = {
    "action": "HOLD",
    "confidence": 0.0,
    "reason": "JSON parsing error — safe fallback applied.",
}


class OllamaAgentMixin(BaseAgent):
    """Mixin that provides the full Ollama API integration layer.

    Subclasses inherit:
        * ``__init__`` with ``api_url``, ``model_name``, ``temperature``,
          ``request_timeout`` parameters.
        * ``generate_signal()`` — the end-to-end signal generation pipeline.
        * ``_format_market_data()`` — Markdown table serialiser.
        * ``_call_ollama()`` — HTTP POST with retry-safe error handling.
        * ``_parse_llm_response()`` — JSON parser + fence stripper.
        * ``_safe_fallback()`` — HOLD/0.0 fallback dict.

    Subclasses MUST implement:
        * ``name`` → ``str`` property.
        * ``_build_prompt(formatted_data: str)`` → ``str``.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        """Initialises the shared Ollama connection parameters.

        Args:
            api_url: Full URL of the Ollama ``/api/generate`` endpoint.
            model_name: Ollama model tag (e.g. ``"gemma4:latest"``).
            temperature: Sampling temperature (0.0–1.0). Lower = more
                deterministic. Defaults to ``0.1``.
            request_timeout: HTTP timeout in seconds. Defaults to ``120``.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout

        logger.info(
            "%s initialised — model=%s  temperature=%.2f",
            self.name,
            self.model_name,
            self.temperature,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface — generate_signal is shared across all agents
    # ------------------------------------------------------------------

    def generate_signal(self, market_data: pd.DataFrame) -> dict:
        """Queries Gemma 4 via Ollama and returns a validated trading signal.

        Workflow:
            1. Format the last :data:`CONTEXT_ROWS` candles as a Markdown table.
            2. Call :meth:`_build_prompt` (persona-specific — implemented by
               each subclass).
            3. POST to the Ollama ``/api/generate`` endpoint.
            4. Parse and validate the JSON response.
            5. Return validated signal, or safe HOLD fallback on any error.

        Args:
            market_data: Enriched OHLCV DataFrame from
                :class:`~src.data.features.FeatureEngineer`.

        Returns:
            Signal dict: ``{action, confidence, reason, agent, model}``.
        """
        latest = market_data.iloc[-1]
        logger.info(
            "[%s] Analysing %d candles — close=%.4f  RSI=%.2f  EMA_20=%.4f  EMA_50=%.4f",
            self.name,
            len(market_data),
            latest.get("close", float("nan")),
            latest.get("RSI_14", float("nan")),
            latest.get("EMA_20", float("nan")),
            latest.get("EMA_50", float("nan")),
        )

        formatted_data = self._format_market_data(market_data, rows=CONTEXT_ROWS)
        prompt = self._build_prompt(formatted_data)

        raw_response = self._call_ollama(prompt)
        if raw_response is None:
            return self._safe_fallback("Ollama API call failed.")

        signal = self._parse_llm_response(raw_response)
        signal["agent"] = self.name
        signal["model"] = self.model_name

        try:
            self.validate_signal(signal)
        except (KeyError, ValueError) as exc:
            logger.error("[%s] Signal validation failed: %s", self.name, exc)
            return self._safe_fallback(str(exc))

        logger.info("[%s] Signal: %s", self.name, signal)
        return signal

    # ------------------------------------------------------------------
    # Abstract — each agent persona overrides only this
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_prompt(self, formatted_data: str) -> str:
        """Constructs the persona-specific prompt sent to Gemma 4.

        Args:
            formatted_data: Markdown table from :meth:`_format_market_data`.

        Returns:
            The complete prompt string (system instruction + market data).
        """

    # ------------------------------------------------------------------
    # Shared infrastructure — identical across all agents
    # ------------------------------------------------------------------

    def _format_market_data(self, df: pd.DataFrame, rows: int = CONTEXT_ROWS) -> str:
        """Serialises the last ``rows`` candles into a Markdown table.

        Args:
            df: Enriched OHLCV DataFrame with indicator columns.
            rows: Number of most-recent rows to include.

        Returns:
            Multi-line Markdown table string ready for embedding in a prompt.
        """
        columns_of_interest = [
            "close", "volume",
            "RSI_14",
            "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
            "EMA_20", "EMA_50",
            "ATRr_14",
        ]
        available = [c for c in columns_of_interest if c in df.columns]
        subset = df[available].tail(rows).copy().round(4)

        header = "| Timestamp | " + " | ".join(available) + " |"
        separator = "|---" * (len(available) + 1) + "|"
        rows_md = []
        for ts, row in subset.iterrows():
            ts_str = str(ts)[:19]
            values = " | ".join(str(row[c]) for c in available)
            rows_md.append(f"| {ts_str} | {values} |")

        return "\n".join([header, separator] + rows_md)

    def _call_ollama(self, prompt: str) -> str | None:
        """POSTs the prompt to Ollama and returns the raw response text.

        Uses ``stream=False`` and ``format="json"`` to enforce single-shot
        JSON output at the engine level.

        Args:
            prompt: The fully constructed prompt string.

        Returns:
            Raw ``response`` string from Ollama, or ``None`` on any error.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": 256,
            },
        }

        logger.debug("[%s] Calling Ollama model=%s", self.name, self.model_name)

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "")
            logger.debug("[%s] Raw response: %s", self.name, raw_text)
            return raw_text

        except requests.exceptions.Timeout:
            logger.error("[%s] Ollama timed out after %ds.", self.name, self.request_timeout)
        except requests.exceptions.ConnectionError:
            logger.error("[%s] Cannot reach Ollama at %s.", self.name, self.api_url)
        except requests.exceptions.HTTPError as exc:
            logger.error("[%s] Ollama HTTP error: %s", self.name, exc)
        except (KeyError, ValueError) as exc:
            logger.error("[%s] Unexpected Ollama response format: %s", self.name, exc)

        return None

    def _parse_llm_response(self, raw_response: str) -> dict:
        """Parses raw LLM text into a Python dict, handling fence artifacts.

        Args:
            raw_response: Raw text string from the Ollama API.

        Returns:
            Parsed dict on success; :data:`FALLBACK_SIGNAL` on
            :exc:`json.JSONDecodeError`.
        """
        cleaned = (
            raw_response.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            parsed = json.loads(cleaned)
            if "action" in parsed:
                parsed["action"] = str(parsed["action"]).upper()
            return parsed
        except json.JSONDecodeError as exc:
            logger.error(
                "[%s] JSON parse failed. Raw: %r  Error: %s",
                self.name, raw_response, exc,
            )
            return dict(FALLBACK_SIGNAL)

    def _safe_fallback(self, reason: str) -> dict:
        """Returns a safe HOLD signal with agent metadata attached.

        Args:
            reason: Short description of why the fallback was triggered.

        Returns:
            ``{"action": "HOLD", "confidence": 0.0, ...}`` dict.
        """
        signal = dict(FALLBACK_SIGNAL)
        signal["reason"] = reason
        signal["agent"] = self.name
        signal["model"] = self.model_name
        logger.warning("[%s] Returning safe fallback. Reason: %s", self.name, reason)
        return signal
