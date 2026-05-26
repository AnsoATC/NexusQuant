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

    def generate_signal(
        self,
        market_data: pd.DataFrame,
        current_position_size: float = 0.0,
    ) -> dict:
        """Queries Gemma 4 via Ollama and returns a validated trading signal.

        Workflow:
            1. Hard trailing-stop check (EMA_20 cross) — if triggered, return
               immediate SELL without calling the LLM.
            2. Compute volume_ratio = current_volume / VMA_20 for conviction gating.
            3. Build position-context and Market Conviction instruction blocks.
            4. Call :meth:`_build_prompt` (persona-specific) and prepend both blocks.
            5. POST to the Ollama ``/api/generate`` endpoint.
            6. Parse, validate, and apply hard action guard.
            7. Return final signal, or safe HOLD fallback on any error.

        Args:
            market_data: Enriched OHLCV DataFrame from
                :class:`~src.data.features.FeatureEngineer`.
            current_position_size: Current open position size in the base asset.
                ``0.0`` means no open position.

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

        # ── Task 3: Hard trailing stop (EMA_20 cross) ─────────────────────────
        # This executes BEFORE the LLM is called and takes absolute priority.
        # If the current close has dropped below EMA_20 while holding a position,
        # we force an immediate SELL to protect capital.
        trailing_stop_signal = self.check_trailing_stop(market_data, current_position_size)
        if trailing_stop_signal is not None:
            trailing_stop_signal["agent"] = self.name
            trailing_stop_signal["model"] = self.model_name
            logger.info("[%s] Trailing stop fired — bypassing LLM.", self.name)
            return trailing_stop_signal

        # ── Rate-of-Change (ROC) Momentum Analysis ───────────────────────────
        # Retrieve pre-computed ROC_5 (5-period Price Rate-of-Change) from the latest candle.
        # This tells us if price accelerated upward (>0.5%) or dropped (< -0.5%).
        latest_roc: float = 0.0
        if "ROC_5" in market_data.columns:
            latest_roc = float(latest["ROC_5"])
        else:
            # Fallback calculation if the column is somehow missing
            ROC_WINDOW: int = 5
            if len(market_data) >= (ROC_WINDOW + 1):
                close_now = float(market_data["close"].iloc[-1])
                close_prev = float(market_data["close"].iloc[-(ROC_WINDOW + 1)])
                if close_prev > 0:
                    latest_roc = ((close_now - close_prev) / close_prev) * 100

        logger.info(
            "[%s] ROC Momentum analysis: ROC_5=%.2f%%",
            self.name, latest_roc,
        )

        roc_momentum_block = (
            f"CRITICAL \u2014 MOMENTUM ANALYSIS:\n"
            f"  Current 5-period Price Rate-of-Change (ROC) = {latest_roc:.2f}%\n"
            f"  You are a Momentum Trader. Evaluate the current 5-period Price Rate-of-Change (ROC). "
            f"If the price is accelerating (ROC > 0.5%), issue a BUY signal immediately. "
            f"If the price is decelerating (ROC < -0.5%), issue a SELL signal immediately. "
            f"Trade the price action itself, do not wait for volume confirmation."
        )

        # ── Position-context injection ─────────────────────────────────────────
        # Tells the LLM what actions are valid based on the current position.
        if current_position_size <= 0.0:
            position_context = (
                "CRITICAL PORTFOLIO STATUS: You currently have NO OPEN POSITION. "
                "Your ONLY valid actions are BUY (to enter a long trade) or HOLD "
                "(to wait for a better entry). You MUST NOT output SELL."
            )
        else:
            position_context = (
                f"CRITICAL PORTFOLIO STATUS: You currently have an OPEN LONG POSITION "
                f"of {current_position_size:.8f} units. You CANNOT BUY more. "
                f"Your ONLY valid actions are SELL (to close the position and realise "
                f"profit or cut losses) or HOLD (to stay in the trade). "
                f"You MUST NOT output BUY."
            )

        logger.info(
            "[%s] Position context: position_size=%.8f — injecting constraint into prompt.",
            self.name, current_position_size,
        )

        formatted_data = self._format_market_data(market_data, rows=CONTEXT_ROWS)
        base_prompt = self._build_prompt(formatted_data)

        # Stack: momentum block → position context → persona prompt.
        # Order ensures momentum instructions are the first thing the model reads.
        prompt = f"{roc_momentum_block}\n\n{position_context}\n\n{base_prompt}"

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

        # ── Aggressive Momentum Strategy Override ───────────────────────────
        # Ensure we enforce the mathematical ROC trigger on the final signal.
        # If ROC > 0.5% and position size is flat: force BUY.
        # If ROC < -0.5% and position size is long: force SELL.
        action = signal.get("action", "HOLD")
        if latest_roc > 0.5 and current_position_size <= 0.0 and action != "BUY":
            logger.warning(
                "[%s] ROC is %.2f%% > 0.5%% but LLM returned %s — overriding to BUY.",
                self.name, latest_roc, action,
            )
            signal["action"] = "BUY"
            signal["confidence"] = 1.0
            signal["reason"] = f"[ROC Momentum Override] ROC ({latest_roc:.2f}%) > 0.5%. Forced BUY."
        elif latest_roc < -0.5 and current_position_size > 0.0 and action != "SELL":
            logger.warning(
                "[%s] ROC is %.2f%% < -0.5%% but LLM returned %s — overriding to SELL.",
                self.name, latest_roc, action,
            )
            signal["action"] = "SELL"
            signal["confidence"] = 1.0
            signal["reason"] = f"[ROC Momentum Override] ROC ({latest_roc:.2f}%) < -0.5%. Forced SELL."

        # ── Hard action guard ──────────────────────────────────────────────────
        # Safety net: even if the LLM ignores the position-context instruction,
        # enforce the constraint so invalid actions never reach execution.
        action = signal.get("action", "HOLD")
        if current_position_size <= 0.0 and action == "SELL":
            logger.warning(
                "[%s] LLM returned SELL with no open position — overriding to HOLD.",
                self.name,
            )
            signal["action"] = "HOLD"
            signal["reason"] = (
                f"[Overridden] No open position; SELL is invalid. Original reason: "
                f"{signal.get('reason', '')}"
            )
        elif current_position_size > 0.0 and action == "BUY":
            logger.warning(
                "[%s] LLM returned BUY with existing position — overriding to HOLD.",
                self.name,
            )
            signal["action"] = "HOLD"
            signal["reason"] = (
                f"[Overridden] Open position exists; BUY is invalid. Original reason: "
                f"{signal.get('reason', '')}"
            )

        # Attach ROC momentum metadata for dashboard display.
        signal["roc_5"] = round(latest_roc, 2)

        logger.info("[%s] Final signal: %s", self.name, signal)
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
            "close", "ROC_5", "volume",
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
