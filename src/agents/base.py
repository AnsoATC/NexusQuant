"""Abstract base class for all NexusQuant trading agents.

Defines the Strategy Pattern contract that every concrete agent must fulfil,
ensuring a consistent, plug-and-play interface across LLM-based, statistical,
and rule-based agent implementations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal schema constants — shared across all agents
# ---------------------------------------------------------------------------
VALID_ACTIONS: frozenset[str] = frozenset({"BUY", "SELL", "HOLD"})


class BaseAgent(ABC):
    """Abstract base class that every NexusQuant trading agent must inherit.

    Implements the **Strategy Pattern**: each concrete subclass encapsulates
    a different decision-making algorithm (LLM reasoning, statistical model,
    rule-based logic, etc.) while exposing a single, uniform interface to the
    rest of the system.

    The contract enforced by this class:

    * :meth:`generate_signal` **must** be implemented by every subclass.
    * The returned ``dict`` **must** conform to the signal schema described
      in :meth:`generate_signal`.

    Subclassing example::

        class MyAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "MyAgent"

            def generate_signal(self, market_data: pd.DataFrame) -> dict:
                return {"action": "BUY", "confidence": 0.9, "reason": "..."}
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this agent.

        Returns:
            A short, unique name string (e.g. ``"DimmerForce"``).
        """

    @abstractmethod
    def generate_signal(self, market_data: pd.DataFrame) -> dict:
        """Analyses enriched market data and returns a standardised trading signal.

        This is the **core method** every agent must implement.  The
        system calls this method on each decision cycle and routes the
        returned signal to the execution layer.

        Args:
            market_data: A :class:`pandas.DataFrame` produced by
                :class:`~src.data.features.FeatureEngineer`, containing OHLCV
                columns plus all computed technical indicator columns.  The
                index is a UTC-aware :class:`pandas.DatetimeIndex`.

        Returns:
            A ``dict`` conforming to the following schema:

            .. code-block:: python

                {
                    "action":     str,    # One of: "BUY" | "SELL" | "HOLD"
                    "confidence": float,  # Range [0.0, 1.0]
                    "reason":     str,    # Human-readable rationale
                    # Any additional agent-specific keys are allowed.
                }

        Raises:
            NotImplementedError: If the subclass does not override this method
                (enforced by :mod:`abc`).
        """

    # ------------------------------------------------------------------
    # Shared utility — available to all subclasses
    # ------------------------------------------------------------------

    def validate_signal(self, signal: dict) -> None:
        """Validates that a signal dict conforms to the required schema.

        Subclasses should call this before returning a signal to catch
        schema violations early rather than propagating bad data downstream.

        Args:
            signal: The signal dictionary to validate.

        Raises:
            ValueError: If ``action`` is not one of the allowed values, or if
                ``confidence`` is outside the ``[0.0, 1.0]`` range.
            KeyError: If any required key is absent from ``signal``.
        """
        required_keys = {"action", "confidence", "reason"}
        missing_keys = required_keys - signal.keys()
        if missing_keys:
            raise KeyError(
                f"[{self.name}] Signal is missing required keys: {missing_keys}"
            )

        action = signal["action"]
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"[{self.name}] Invalid action '{action}'. "
                f"Must be one of {sorted(VALID_ACTIONS)}."
            )

        confidence = signal["confidence"]
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"[{self.name}] Confidence {confidence} is out of range [0.0, 1.0]."
            )

        logger.debug(
            "[%s] Signal validated — action=%s  confidence=%.2f",
            self.name,
            action,
            confidence,
        )
