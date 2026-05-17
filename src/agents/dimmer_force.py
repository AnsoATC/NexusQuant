"""DimmerForce agent module for NexusQuant.

Implements ``DimmerForceAgent``, a quantitative trading agent powered by a
local Gemma 4 model served via the Ollama REST API.

Phase 3 changes vs Phase 2 stub:
    * ``_format_market_data()`` serialises the last N enriched candles into a
      Markdown table the LLM can interpret with zero ambiguity.
    * ``_build_prompt()`` constructs a tightly constrained prompt that forces
      the model to output only a raw JSON object.
    * ``generate_signal()`` sends the prompt to Ollama, parses the JSON
      response, validates it against the ``BaseAgent`` schema, and returns a
      safe fallback on any parsing failure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import requests

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_API_URL: str = "http://localhost:11434/api/generate"
DEFAULT_MODEL: str = "gemma4:latest"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_REQUEST_TIMEOUT: int = 120   # seconds
CONTEXT_ROWS: int = 5                # number of candles sent to the LLM

FALLBACK_SIGNAL: dict[str, Any] = {
    "action": "HOLD",
    "confidence": 0.0,
    "reason": "JSON parsing error — safe fallback applied.",
}


class DimmerForceAgent(BaseAgent):
    """LLM-powered quantitative trading agent driven by Gemma 4 via Ollama.

    Sends enriched OHLCV + indicator data to a locally hosted Gemma 4 model
    and parses its JSON response into a standardised trading signal.

    Architecture note:
        Follows the Strategy Pattern defined by
        :class:`~src.agents.base.BaseAgent`. The execution layer interacts
        exclusively with the ``BaseAgent`` interface.

    Attributes:
        api_url: Full URL of the Ollama ``/api/generate`` endpoint.
        model_name: Ollama model tag (e.g. ``"gemma4:latest"``).
        temperature: Sampling temperature passed to the model. Low values
            produce deterministic, analytical outputs.
        request_timeout: HTTP timeout in seconds for Ollama requests.

    Example:
        >>> agent = DimmerForceAgent()
        >>> signal = agent.generate_signal(enriched_df)
        >>> print(signal["action"], signal["confidence"])
        BUY 0.82
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        """Initialises the DimmerForceAgent.

        Args:
            api_url: Full URL of the Ollama generate endpoint. Defaults to
                ``"http://localhost:11434/api/generate"``.
            model_name: Ollama model tag to use for inference. Defaults to
                ``"gemma4:latest"``.
            temperature: Sampling temperature (0.0–1.0). Lower values produce
                more deterministic outputs. Defaults to ``0.1``.
            request_timeout: Maximum seconds to wait for an Ollama response.
                Defaults to ``120``.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.request_timeout = request_timeout

        logger.info(
            "DimmerForceAgent initialised — model=%s  api_url=%s  temperature=%.2f",
            self.model_name,
            self.api_url,
            self.temperature,
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
        """Queries Gemma 4 with enriched market data and returns a trading signal.

        Workflow:
            1. Format the last :data:`CONTEXT_ROWS` candles into a Markdown
               table via :meth:`_format_market_data`.
            2. Build a tightly constrained prompt via :meth:`_build_prompt`.
            3. POST the prompt to the Ollama ``/api/generate`` endpoint.
            4. Parse and validate the JSON response.
            5. Return the validated signal, or a safe HOLD fallback on error.

        Args:
            market_data: Enriched :class:`pandas.DataFrame` produced by
                :class:`~src.data.features.FeatureEngineer`.  Expected columns
                include ``close``, ``RSI_14``, ``MACD_12_26_9``, ``EMA_20``,
                ``EMA_50``, and ``ATRr_14``.

        Returns:
            A signal ``dict`` conforming to the :class:`~src.agents.base.BaseAgent`
            schema::

                {
                    "action":     "BUY" | "SELL" | "HOLD",
                    "confidence": float,   # 0.0 – 1.0
                    "reason":     str,     # one short sentence
                    "agent":      "DimmerForce",
                    "model":      str,     # model tag used
                }

            On any LLM or parsing failure a safe fallback is returned:
            ``{"action": "HOLD", "confidence": 0.0, ...}``.
        """
        latest = market_data.iloc[-1]
        logger.info(
            "[%s] Analysing %d enriched candles — latest close=%.4f  RSI=%.2f  "
            "EMA_20=%.4f  EMA_50=%.4f",
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

        logger.info("[%s] Signal generated: %s", self.name, signal)
        return signal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_market_data(self, df: pd.DataFrame, rows: int = CONTEXT_ROWS) -> str:
        """Serialises the last ``rows`` candles into a Markdown table string.

        Selects a curated subset of columns so the prompt remains concise and
        the LLM is not overwhelmed with raw floats.  Values are rounded to 4
        decimal places for readability.

        Args:
            df: Enriched OHLCV DataFrame with indicator columns.
            rows: Number of most-recent rows to include. Defaults to
                :data:`CONTEXT_ROWS`.

        Returns:
            A multi-line string containing a Markdown table with one row per
            candle, suitable for embedding directly in the LLM prompt.
        """
        columns_of_interest = [
            "close", "volume",
            "RSI_14",
            "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
            "EMA_20", "EMA_50",
            "ATRr_14",
        ]
        # Keep only columns that actually exist (guards against schema changes)
        available = [c for c in columns_of_interest if c in df.columns]
        subset = df[available].tail(rows).copy()

        # Round for readability
        subset = subset.round(4)

        # Build Markdown table manually so it is readable without extra deps
        header = "| Timestamp | " + " | ".join(available) + " |"
        separator = "|---" * (len(available) + 1) + "|"
        rows_md = []
        for ts, row in subset.iterrows():
            ts_str = str(ts)[:19]  # e.g. "2026-05-17 13:00:00"
            values = " | ".join(str(row[c]) for c in available)
            rows_md.append(f"| {ts_str} | {values} |")

        return "\n".join([header, separator] + rows_md)

    def _build_prompt(self, formatted_data: str) -> str:
        """Constructs the full prompt sent to the Gemma 4 model.

        The prompt has two logical sections:
        1. **System instruction** — role, strict JSON-only constraint, output
           schema, and field definitions.
        2. **User data** — the Markdown table of recent candles.

        Args:
            formatted_data: Markdown table string from
                :meth:`_format_market_data`.

        Returns:
            A complete, ready-to-send prompt string.
        """
        system_instruction = (
            "You are a strict quantitative AI trading agent. "
            "Your sole task is to analyse the provided market data and output a trading decision. "
            "You MUST respond with ONLY a single raw JSON object — no markdown fences, "
            "no explanations, no conversational filler. "
            "The JSON object must conform EXACTLY to this schema:\n"
            '{"action": "BUY"|"SELL"|"HOLD", "confidence": <float 0.0-1.0>, "reason": "<one short sentence>"}\n\n'
            "Field definitions:\n"
            "  action     — Your trading decision: BUY (long bias), SELL (short/exit bias), or HOLD (neutral).\n"
            "  confidence — Your certainty in the decision as a decimal between 0.0 (no conviction) and 1.0 (maximum conviction).\n"
            "  reason     — A single concise sentence explaining the primary technical reason for the decision.\n\n"
            "Analysis rules:\n"
            "  - RSI > 70 indicates overbought; RSI < 30 indicates oversold.\n"
            "  - MACD histogram (MACDh) turning positive signals bullish momentum; negative signals bearish.\n"
            "  - Price above EMA_20 and EMA_50 is a bullish structure; below is bearish.\n"
            "  - ATR measures volatility; high ATR increases risk and should lower confidence.\n"
        )

        user_section = (
            f"Here is the most recent enriched market data for analysis:\n\n"
            f"{formatted_data}\n\n"
            "Based solely on the technical indicators above, provide your JSON trading signal now:"
        )

        return f"{system_instruction}\n{user_section}"

    def _call_ollama(self, prompt: str) -> str | None:
        """Sends the prompt to the Ollama REST API and returns the raw text response.

        Uses ``"stream": False`` for a single-shot response and
        ``"format": "json"`` to instruct the Ollama engine to enforce JSON
        output at the sampling level.

        Args:
            prompt: The fully constructed prompt string.

        Returns:
            The raw ``response`` string from the Ollama API, or ``None`` if the
            request fails.
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

        logger.debug("[%s] Sending request to Ollama — model=%s", self.name, self.model_name)

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "")
            logger.debug("[%s] Raw LLM response: %s", self.name, raw_text)
            return raw_text

        except requests.exceptions.Timeout:
            logger.error(
                "[%s] Ollama request timed out after %ds.",
                self.name,
                self.request_timeout,
            )
        except requests.exceptions.ConnectionError:
            logger.error(
                "[%s] Cannot reach Ollama at %s. Is the server running?",
                self.name,
                self.api_url,
            )
        except requests.exceptions.HTTPError as exc:
            logger.error("[%s] Ollama returned HTTP error: %s", self.name, exc)
        except (KeyError, ValueError) as exc:
            logger.error("[%s] Unexpected Ollama response format: %s", self.name, exc)

        return None

    def _parse_llm_response(self, raw_response: str) -> dict:
        """Parses the raw LLM string into a Python dict.

        Handles two common failure modes:
        * The model wraps its response in markdown fences (stripped before
          parsing).
        * The model produces malformed JSON (falls back to safe HOLD signal).

        Args:
            raw_response: The raw text string returned by the Ollama API.

        Returns:
            A parsed ``dict`` if successful, otherwise the
            :data:`FALLBACK_SIGNAL` dict.
        """
        # Strip accidental markdown fences the model might still produce
        cleaned = raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(cleaned)
            # Normalise action to uppercase in case the model uses lowercase
            if "action" in parsed:
                parsed["action"] = str(parsed["action"]).upper()
            return parsed
        except json.JSONDecodeError as exc:
            logger.error(
                "[%s] JSON parsing failed. Raw response: %r. Error: %s",
                self.name,
                raw_response,
                exc,
            )
            return dict(FALLBACK_SIGNAL)

    def _safe_fallback(self, reason: str) -> dict:
        """Returns a safe HOLD signal enriched with agent metadata.

        Args:
            reason: Short description of why the fallback was triggered.

        Returns:
            A minimal, validated signal dict with ``action="HOLD"`` and
            ``confidence=0.0``.
        """
        signal = dict(FALLBACK_SIGNAL)
        signal["reason"] = reason
        signal["agent"] = self.name
        signal["model"] = self.model_name
        logger.warning("[%s] Returning safe fallback signal. Reason: %s", self.name, reason)
        return signal
