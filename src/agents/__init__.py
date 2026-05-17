"""Agent package exports for NexusQuant.

Exposes all concrete trading agents so they can be imported directly from
``src.agents`` without navigating to individual sub-modules.

Usage::

    from src.agents import DimmerForceAgent, SavinovAgent, DeepAlphaAgent
"""

from src.agents.deepalpha_bot import DeepAlphaAgent
from src.agents.dimmer_force import DimmerForceAgent
from src.agents.savinov_bot import SavinovAgent

__all__ = [
    "DimmerForceAgent",
    "SavinovAgent",
    "DeepAlphaAgent",
]
