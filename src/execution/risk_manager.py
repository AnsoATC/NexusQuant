"""Risk management module for NexusQuant.

Provides the RiskManager class, which translates a raw trading signal into
concrete position sizing metrics using an ATR-based stop-loss model.

All risk parameters are driven by ``config/settings.yaml`` so the strategy can
be tuned without touching source code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — used when config keys are absent
# ---------------------------------------------------------------------------
DEFAULT_RISK_PER_TRADE_PERCENT: float = 2.0
DEFAULT_ATR_STOP_LOSS_MULTIPLIER: float = 1.5
DEFAULT_RISK_REWARD_RATIO: float = 2.0
DEFAULT_MAX_POSITION_SIZE_USDT: float = 100.0

CONFIG_PATH: Path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class RiskManager:
    """Calculates ATR-based position sizes and risk/reward levels.

    All parameters are loaded from ``config/settings.yaml`` at initialisation.
    If the config file is missing or a key is absent, safe defaults are applied
    and a warning is logged so the system keeps running.

    Attributes:
        risk_per_trade_pct: Fraction of available capital to risk per trade
            (e.g. ``2.0`` means 2 %).
        atr_multiplier: Multiplier applied to ATR to compute the stop-loss
            distance (e.g. ``1.5`` → SL = price − 1.5 × ATR).
        risk_reward_ratio: Take-profit distance expressed as a multiple of the
            stop-loss distance (e.g. ``2.0`` → TP = price + 2 × SL_distance).
        max_position_size_usdt: Hard cap on total position cost in USDT.

    Example:
        >>> rm = RiskManager()
        >>> metrics = rm.calculate_position_size("BUY", 78000.0, 250.0, 100.0)
        >>> print(metrics)
        {'units': 0.001..., 'cost_usdt': ..., 'stop_loss_price': ..., 'take_profit_price': ...}
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        """Loads risk parameters from the YAML config file.

        Args:
            config_path: Absolute path to ``settings.yaml``. Defaults to the
                canonical project location resolved relative to this file.
        """
        risk_cfg = self._load_risk_config(config_path)

        self.risk_per_trade_pct: float = float(
            risk_cfg.get("risk_per_trade_percent", DEFAULT_RISK_PER_TRADE_PERCENT)
        )
        self.atr_multiplier: float = float(
            risk_cfg.get("atr_stop_loss_multiplier", DEFAULT_ATR_STOP_LOSS_MULTIPLIER)
        )
        self.risk_reward_ratio: float = float(
            risk_cfg.get("risk_reward_ratio", DEFAULT_RISK_REWARD_RATIO)
        )
        self.max_position_size_usdt: float = float(
            risk_cfg.get("max_position_size_usdt", DEFAULT_MAX_POSITION_SIZE_USDT)
        )

        logger.info(
            "RiskManager initialised — risk_per_trade=%.1f%%  ATR_mult=%.2f  "
            "R:R=1:%.1f  max_position=%.2f USDT",
            self.risk_per_trade_pct,
            self.atr_multiplier,
            self.risk_reward_ratio,
            self.max_position_size_usdt,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        signal_action: str,
        current_price: float,
        atr: float,
        available_capital: float,
    ) -> dict[str, Any]:
        """Computes ATR-based position sizing for a given signal.

        For a **BUY** signal:

        1. Stop-loss distance  = ``atr_multiplier`` × ATR
        2. Stop-loss price     = ``current_price`` − SL distance
        3. Risk amount (USDT)  = ``available_capital`` × ``risk_per_trade_pct`` / 100
        4. Units to buy        = risk amount / SL distance  (risk per unit)
        5. Cost (USDT)         = units × ``current_price``
        6. Clamp cost to ``min(cost, available_capital, max_position_size_usdt)``
           and recalculate units accordingly.
        7. Take-profit price   = ``current_price`` + SL distance × ``risk_reward_ratio``

        For **SELL** or **HOLD** signals, all sizing values are zero and no
        levels are computed.

        Args:
            signal_action: One of ``"BUY"``, ``"SELL"``, or ``"HOLD"``
                (case-insensitive).
            current_price: Latest close price in USDT.
            atr: Latest ATR value from the indicator DataFrame.
            available_capital: Current available USDT balance in the portfolio.

        Returns:
            A ``dict`` with the following keys:

            * ``units``            — Number of asset units to purchase (float).
            * ``cost_usdt``        — Total cost of the position in USDT (float).
            * ``stop_loss_price``  — ATR-derived stop-loss level (float or None).
            * ``take_profit_price``— R:R-derived take-profit level (float or None).
            * ``risk_amount_usdt`` — USDT amount at risk if SL is hit (float).
            * ``sl_distance``      — Raw stop-loss distance in USDT (float or None).

        Raises:
            ValueError: If ``current_price`` or ``atr`` is non-positive.
        """
        action = signal_action.upper()

        if action in ("HOLD", "SELL"):
            logger.info(
                "RiskManager: action=%s — position size is 0 (no entry).", action
            )
            return self._zero_metrics(action)

        # ── Input guards ──────────────────────────────────────────────────────
        if current_price <= 0:
            raise ValueError(f"current_price must be positive, got {current_price}.")
        if atr <= 0:
            raise ValueError(f"atr must be positive, got {atr}.")
        if available_capital <= 0:
            logger.warning("RiskManager: available_capital=%.4f — insufficient funds.", available_capital)
            return self._zero_metrics(action)

        # ── Core ATR sizing math ──────────────────────────────────────────────
        sl_distance: float = self.atr_multiplier * atr
        stop_loss_price: float = current_price - sl_distance
        take_profit_price: float = current_price + (sl_distance * self.risk_reward_ratio)

        risk_amount_usdt: float = available_capital * (self.risk_per_trade_pct / 100.0)

        # Risk per unit = how many USDT we lose if price drops by sl_distance
        units: float = risk_amount_usdt / sl_distance
        cost_usdt: float = units * current_price

        # ── Clamp to hard limits ──────────────────────────────────────────────
        max_affordable: float = min(available_capital, self.max_position_size_usdt)
        if cost_usdt > max_affordable:
            logger.warning(
                "RiskManager: calculated cost %.4f USDT exceeds cap %.4f USDT — clamping.",
                cost_usdt,
                max_affordable,
            )
            cost_usdt = max_affordable
            units = cost_usdt / current_price

        metrics: dict[str, Any] = {
            "action": action,
            "units": round(units, 8),
            "cost_usdt": round(cost_usdt, 4),
            "stop_loss_price": round(stop_loss_price, 4),
            "take_profit_price": round(take_profit_price, 4),
            "risk_amount_usdt": round(risk_amount_usdt, 4),
            "sl_distance": round(sl_distance, 4),
        }

        logger.info(
            "RiskManager [BUY]: units=%.8f  cost=%.4f USDT  SL=%.4f  TP=%.4f  "
            "risk=%.4f USDT",
            metrics["units"],
            metrics["cost_usdt"],
            metrics["stop_loss_price"],
            metrics["take_profit_price"],
            metrics["risk_amount_usdt"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_risk_config(config_path: Path) -> dict:
        """Reads the ``risk`` section from the YAML settings file.

        Args:
            config_path: Path to the settings YAML file.

        Returns:
            The ``risk`` sub-dictionary from the config, or an empty ``dict``
            if the file is missing or malformed.
        """
        if not config_path.exists():
            logger.warning(
                "Config file not found at %s — using default risk parameters.",
                config_path,
            )
            return {}
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
            risk_section = config.get("risk", {})
            logger.debug("Loaded risk config: %s", risk_section)
            return risk_section
        except yaml.YAMLError as exc:
            logger.error("Failed to parse %s: %s — using defaults.", config_path, exc)
            return {}

    @staticmethod
    def _zero_metrics(action: str) -> dict[str, Any]:
        """Returns a zeroed metrics dict for non-BUY signals.

        Args:
            action: The original signal action string.

        Returns:
            A metrics dict with all numeric fields set to zero or None.
        """
        return {
            "action": action,
            "units": 0.0,
            "cost_usdt": 0.0,
            "stop_loss_price": None,
            "take_profit_price": None,
            "risk_amount_usdt": 0.0,
            "sl_distance": None,
        }
