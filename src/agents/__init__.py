"""Agent package exports for NexusQuant.

Three LLM personas — all powered by Gemma 4 via Ollama, differentiated
exclusively by their system-prompt instructions:

* :class:`DimmerForceAgent` — Trend-Following (MACD + EMA primary).
* :class:`ZenithAgent`      — Mean-Reversion / Contrarian (RSI primary).
* :class:`AegisAgent`       — Conservative Risk-Averse (all-or-nothing HOLD).

Usage::

    from src.agents import DimmerForceAgent, ZenithAgent, AegisAgent
"""

from src.agents.aegis import AegisAgent
from src.agents.dimmer_force import DimmerForceAgent
from src.agents.zenith import ZenithAgent

__all__ = [
    "DimmerForceAgent",
    "ZenithAgent",
    "AegisAgent",
]
