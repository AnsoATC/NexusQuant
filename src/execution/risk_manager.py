"""Risk management module for NexusQuant.

Provides the RiskManager class which:
  * Translates raw trading signals into concrete position sizing metrics
    using a Fixed Capital Allocation model.
  * Detects high-conviction price breakouts using a 20-candle range filter
    combined with a volume-ratio threshold.

All risk parameters are driven by ``config/settings.yaml`` so the strategy
can be tuned without touching source code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — used when config keys are absent
# ---------------------------------------------------------------------------
DEFAULT_RISK_PER_TRADE_PERCENT: float = 2.0
DEFAULT_ATR_STOP_LOSS_MULTIPLIER: float = 1.5
DEFAULT_RISK_REWARD_RATIO: float = 2.0
DEFAULT_MAX_POSITION_SIZE_USDT: float = 100.0
# Binance BTC/USDT lot size defaults — overridden by config when available
DEFAULT_MIN_LOT_SIZE: float = 0.00001    # Minimum order qty in base asset
DEFAULT_LOT_STEP_SIZE: float = 0.00001   # Precision step for order qty rounding

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
        # Lot-size precision — prevents Binance -1013 / precision errors
        self.min_lot_size: float = float(
            risk_cfg.get("min_lot_size", DEFAULT_MIN_LOT_SIZE)
        )
        self.lot_step_size: float = float(
            risk_cfg.get("lot_step_size", DEFAULT_LOT_STEP_SIZE)
        )

        logger.info(
            "RiskManager initialised — risk_per_trade=%.1f%%  ATR_mult=%.2f  "
            "R:R=1:%.1f  max_position=%.2f USDT  min_lot=%.5f  step=%.5f",
            self.risk_per_trade_pct,
            self.atr_multiplier,
            self.risk_reward_ratio,
            self.max_position_size_usdt,
            self.min_lot_size,
            self.lot_step_size,
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
        """Computes a fixed-allocation position size for a given signal.

        Sizing model: **Fixed Capital Allocation**.
        The trade spend is a fixed percentage of available capital, avoiding
        the ATR-unit model that produces sub-minimum lot sizes on small accounts.

        For a **BUY** signal:

        1. Allowed spend  = ``available_capital`` × ``risk_per_trade_pct`` / 100
        2. Cap spend to   ``min(allowed_spend, max_position_size_usdt, available_capital)``
        3. Raw units      = spend / ``current_price``
        4. Truncate units to ``lot_step_size`` precision.
        5. If truncated units < ``min_lot_size`` and capital covers it:
               force units = ``min_lot_size`` (guarantees execution on small accounts).
        6. Theoretical SL / TP are derived from ATR for tracking only —
           they do **not** gate the trade size.

        For **SELL** or **HOLD** signals, all sizing values are zero and no
        levels are computed.

        Args:
            signal_action: One of ``"BUY"``, ``"SELL"``, or ``"HOLD"``
                (case-insensitive).
            current_price: Latest close price in USDT.
            atr: ATR value used only for theoretical SL/TP levels.
            available_capital: Current available USDT balance.

        Returns:
            A ``dict`` with keys: ``units``, ``cost_usdt``,
            ``stop_loss_price``, ``take_profit_price``,
            ``risk_amount_usdt``, ``sl_distance``.

        Raises:
            ValueError: If ``current_price`` is non-positive.
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
        if available_capital <= 0:
            logger.warning(
                "RiskManager: available_capital=%.4f — insufficient funds.",
                available_capital,
            )
            return self._zero_metrics(action)

        # ── Theoretical SL / TP levels (ATR-based, for tracking only) ─────────
        # atr may be 0 on early candles; guard so we never divide by zero.
        sl_distance: float | None       = None
        stop_loss_price: float | None   = None
        take_profit_price: float | None = None
        if atr > 0:
            sl_distance       = self.atr_multiplier * atr
            stop_loss_price   = current_price - sl_distance
            take_profit_price = current_price + (sl_distance * self.risk_reward_ratio)

        # ── Fixed Capital Allocation sizing ───────────────────────────────────
        # Step 1: compute allowed spend for this trade.
        risk_amount_usdt: float = available_capital * (self.risk_per_trade_pct / 100.0)

        # Step 2: apply hard caps.
        max_spend: float = min(risk_amount_usdt, self.max_position_size_usdt, available_capital)

        # Step 3: convert spend to asset units.
        raw_units: float = max_spend / current_price

        # Step 4: truncate to exchange lot-step precision.
        units: float = self._truncate_to_step(raw_units, self.lot_step_size)

        # Step 5: enforce exchange minimum lot size.
        if units < self.min_lot_size:
            min_lot_cost = self.min_lot_size * current_price
            if min_lot_cost <= available_capital:
                # Force up to minimum — a marginally higher spend % is
                # preferable to skipping the trade on a small account.
                logger.warning(
                    "RiskManager: truncated units %.8f < min_lot_size %.5f — "
                    "forcing to min_lot_size (cost=%.4f USDT, capital=%.4f USDT).",
                    units, self.min_lot_size, min_lot_cost, available_capital,
                )
                units = self.min_lot_size
            else:
                # Even the minimum lot exceeds available capital.
                logger.warning(
                    "RiskManager: min_lot_size cost %.4f USDT > capital %.4f USDT "
                    "— returning zero position.",
                    min_lot_cost, available_capital,
                )
                return self._zero_metrics(action)

        # Final truncation — removes any floating-point creep from the force-up.
        units     = self._truncate_to_step(units, self.lot_step_size)
        cost_usdt = units * current_price

        metrics: dict[str, Any] = {
            "action":            action,
            "units":             units,
            "cost_usdt":         round(cost_usdt, 4),
            "stop_loss_price":   round(stop_loss_price, 4) if stop_loss_price is not None else None,
            "take_profit_price": round(take_profit_price, 4) if take_profit_price is not None else None,
            "risk_amount_usdt":  round(risk_amount_usdt, 4),
            "sl_distance":       round(sl_distance, 4) if sl_distance is not None else None,
        }

        logger.info(
            "RiskManager [BUY — Fixed Alloc]: units=%.8f  cost=%.4f USDT  "
            "spend_pct=%.1f%%  SL=%s  TP=%s",
            metrics["units"],
            metrics["cost_usdt"],
            self.risk_per_trade_pct,
            metrics["stop_loss_price"],
            metrics["take_profit_price"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Breakout detection
    # ------------------------------------------------------------------

    def detect_breakout(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        volume_ratio: float,
        lookback: int = 20,
        conviction_threshold: float = 1.5,
    ) -> str:
        """Identifies a volume-confirmed price breakout over the last ``lookback`` candles.

        Algorithm:
            1. Calculate the highest high and lowest low of the last ``lookback``
               candles (excluding the current / most-recent candle).
            2. If ``current_price > highest_high`` **and** ``volume_ratio >= conviction_threshold``
               → **BUY_BREAKOUT**  (upside breakout with conviction).
            3. If ``current_price < lowest_low``  **and** ``volume_ratio >= conviction_threshold``
               → **SELL_BREAKOUT** (downside breakout with conviction).
            4. Otherwise → **NONE** (no breakout or insufficient volume).

        Args:
            market_data: Enriched OHLCV DataFrame with at least ``high`` and
                ``low`` columns.
            current_price: Latest close price in USDT.
            volume_ratio: Current volume divided by VMA_20 (computed in the
                agent layer and passed in to avoid recalculation).
            lookback: Number of prior candles used to establish the range.
                Defaults to ``20``.
            conviction_threshold: Minimum volume_ratio required to confirm a
                breakout.  Defaults to ``1.5``.

        Returns:
            One of:
            * ``"BUY_BREAKOUT"``  — price broke above the 20-candle high on
              high volume.
            * ``"SELL_BREAKOUT"`` — price broke below the 20-candle low on
              high volume.
            * ``"NONE"``          — no confirmed breakout.
        """
        required_cols = {"high", "low"}
        if not required_cols.issubset(market_data.columns):
            logger.warning(
                "RiskManager.detect_breakout: 'high' or 'low' column missing — "
                "returning NONE."
            )
            return "NONE"

        if len(market_data) < lookback + 1:
            logger.debug(
                "RiskManager.detect_breakout: insufficient rows (%d < %d) — "
                "returning NONE.",
                len(market_data), lookback + 1,
            )
            return "NONE"

        # Exclude the current (last) candle to form a clean prior range.
        prior = market_data.iloc[-(lookback + 1):-1]
        highest_high = float(prior["high"].max())
        lowest_low   = float(prior["low"].min())

        logger.info(
            "RiskManager.detect_breakout: price=%.4f  high_20=%.4f  low_20=%.4f  "
            "vol_ratio=%.2f  threshold=%.1f",
            current_price, highest_high, lowest_low, volume_ratio, conviction_threshold,
        )

        if current_price > highest_high and volume_ratio >= conviction_threshold:
            logger.info(
                "RiskManager.detect_breakout: BUY_BREAKOUT confirmed — "
                "price %.4f > high_20 %.4f on volume_ratio %.2f.",
                current_price, highest_high, volume_ratio,
            )
            return "BUY_BREAKOUT"

        if current_price < lowest_low and volume_ratio >= conviction_threshold:
            logger.info(
                "RiskManager.detect_breakout: SELL_BREAKOUT confirmed — "
                "price %.4f < low_20 %.4f on volume_ratio %.2f.",
                current_price, lowest_low, volume_ratio,
            )
            return "SELL_BREAKOUT"

        return "NONE"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_to_step(value: float, step: float) -> float:
        """Truncates ``value`` to the nearest multiple of ``step`` (floor).

        Uses integer arithmetic to avoid floating-point precision errors that
        arise from naive ``round(value / step) * step`` approaches.

        Args:
            value: The raw quantity to truncate.
            step:  The exchange lot step size (e.g. ``0.00001`` for BTC/USDT).

        Returns:
            The largest multiple of ``step`` that does not exceed ``value``.
        """
        if step <= 0:
            return value
        # Scale to integer space, floor-divide, scale back.
        precision = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        factor = 10 ** precision
        return int(value * factor) / factor

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
