"""Paper trading simulation module for NexusQuant.

Provides the PaperTrader class, which simulates order execution and portfolio
management without touching any real exchange API.  All state is held in memory
and can be inspected at any point via the ``portfolio_summary`` property.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents a single open paper-trade position.

    Attributes:
        symbol: Trading pair (e.g. ``"BTC/USDT"``).
        units: Number of asset units held.
        entry_price: Price at which the position was opened (USDT).
        cost_usdt: Total USDT spent to open the position.
        stop_loss_price: ATR-derived stop-loss level (USDT).
        take_profit_price: R:R-derived take-profit level (USDT).
        opened_at: UTC timestamp when the position was opened.
    """
    symbol: str
    units: float
    entry_price: float
    cost_usdt: float
    stop_loss_price: float
    take_profit_price: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TradeRecord:
    """Immutable record of a completed (closed) trade.

    Attributes:
        symbol: Trading pair.
        units: Units traded.
        entry_price: Open price.
        exit_price: Close price.
        pnl_usdt: Realised profit or loss in USDT.
        close_reason: Why the trade was closed (``"SIGNAL"``, ``"STOP_LOSS"``,
            ``"TAKE_PROFIT"``).
        opened_at: UTC open timestamp.
        closed_at: UTC close timestamp.
    """
    symbol: str
    units: float
    entry_price: float
    exit_price: float
    pnl_usdt: float
    close_reason: str
    opened_at: datetime
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PaperTrader:
    """Simulates a local paper-trading account with position and PnL tracking.

    Maintains a USDT balance and a dictionary of open positions keyed by
    symbol.  Positions are automatically closed when the current price crosses
    their Stop-Loss or Take-Profit threshold via :meth:`update_positions`.

    Attributes:
        balance_usdt: Current available USDT balance.
        open_positions: Active positions keyed by trading pair symbol.
        trade_history: Chronological list of all closed :class:`TradeRecord`s.
        starting_balance: The initial balance passed at construction (immutable
            reference for drawdown / PnL calculations).

    Example:
        >>> trader = PaperTrader(starting_balance_usdt=100.0)
        >>> trader.process_signal(signal, current_price=78000.0, risk_metrics=metrics)
        >>> trader.update_positions(current_price=78500.0)
        >>> print(trader.portfolio_summary)
    """

    def __init__(self, starting_balance_usdt: float = 100.0) -> None:
        """Initialises the PaperTrader with a clean account.

        Args:
            starting_balance_usdt: Starting USDT balance. Defaults to
                ``100.0``.
        """
        self.starting_balance: float = starting_balance_usdt
        self.balance_usdt: float = starting_balance_usdt
        self.open_positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []

        logger.info(
            "PaperTrader initialised — starting balance: %.4f USDT",
            self.starting_balance,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_signal(
        self,
        signal: dict[str, Any],
        current_price: float,
        risk_metrics: dict[str, Any],
        symbol: str = "BTC/USDT",
    ) -> None:
        """Processes a trading signal and executes the appropriate simulated action.

        Routing logic:

        * ``BUY``  + no open position  → open a new position.
        * ``BUY``  + position already open → log skip (no pyramiding).
        * ``SELL`` + open position      → close position at ``current_price``.
        * ``SELL`` + no open position   → log skip (nothing to close).
        * ``HOLD`` → no action taken.

        Args:
            signal: Signal dict from :class:`~src.agents.base.BaseAgent`,
                must contain an ``"action"`` key.
            current_price: Latest asset price in USDT.
            risk_metrics: Sizing dict from
                :class:`~src.execution.risk_manager.RiskManager`.
            symbol: Trading pair identifier. Defaults to ``"BTC/USDT"``.
        """
        action = signal.get("action", "HOLD").upper()

        if action == "BUY":
            self._open_position(symbol, current_price, risk_metrics)
        elif action == "SELL":
            self._close_position(symbol, current_price, close_reason="SIGNAL")
        else:
            logger.info("PaperTrader: HOLD — no action taken.")

    def update_positions(self, current_price: float, symbol: str = "BTC/USDT") -> None:
        """Checks open positions against SL/TP thresholds and auto-closes if triggered.

        This method should be called on each new candle close to simulate
        intra-candle SL/TP monitoring.

        Args:
            current_price: Latest close price in USDT.
            symbol: Trading pair to check. Defaults to ``"BTC/USDT"``.
        """
        position = self.open_positions.get(symbol)
        if position is None:
            return

        if current_price <= position.stop_loss_price:
            logger.warning(
                "PaperTrader: STOP-LOSS triggered for %s at %.4f (SL=%.4f).",
                symbol,
                current_price,
                position.stop_loss_price,
            )
            self._close_position(symbol, current_price, close_reason="STOP_LOSS")

        elif current_price >= position.take_profit_price:
            logger.info(
                "PaperTrader: TAKE-PROFIT triggered for %s at %.4f (TP=%.4f).",
                symbol,
                current_price,
                position.take_profit_price,
            )
            self._close_position(symbol, current_price, close_reason="TAKE_PROFIT")

    @property
    def portfolio_summary(self) -> dict[str, Any]:
        """Returns a snapshot of the current portfolio state.

        Returns:
            A dict containing:

            * ``balance_usdt``        — Current free USDT balance.
            * ``starting_balance``    — Original starting USDT balance.
            * ``total_pnl_usdt``      — Sum of all closed-trade PnL.
            * ``total_pnl_percent``   — Total PnL as a percentage of starting balance.
            * ``open_positions_count``— Number of currently open positions.
            * ``total_trades``        — Number of completed trades.
            * ``winning_trades``      — Number of profitable completed trades.
            * ``win_rate_percent``    — Win rate as a percentage (0.0 if no trades).
        """
        total_pnl = sum(t.pnl_usdt for t in self.trade_history)
        winning = [t for t in self.trade_history if t.pnl_usdt > 0]
        total_trades = len(self.trade_history)
        win_rate = (len(winning) / total_trades * 100.0) if total_trades > 0 else 0.0

        return {
            "balance_usdt": round(self.balance_usdt, 4),
            "starting_balance": round(self.starting_balance, 4),
            "total_pnl_usdt": round(total_pnl, 4),
            "total_pnl_percent": round((total_pnl / self.starting_balance) * 100.0, 2),
            "open_positions_count": len(self.open_positions),
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "win_rate_percent": round(win_rate, 1),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_position(
        self,
        symbol: str,
        current_price: float,
        risk_metrics: dict[str, Any],
    ) -> None:
        """Opens a new simulated long position.

        Silently skips if a position for ``symbol`` is already open (no
        pyramiding) or if ``risk_metrics`` indicates zero cost (e.g. the risk
        manager returned a non-BUY result or insufficient capital).

        Args:
            symbol: Trading pair identifier.
            current_price: Entry price in USDT.
            risk_metrics: Sizing dict from RiskManager.
        """
        if symbol in self.open_positions:
            logger.warning(
                "PaperTrader: BUY signal skipped — position for %s already open.", symbol
            )
            return

        cost = risk_metrics.get("cost_usdt", 0.0)
        units = risk_metrics.get("units", 0.0)
        sl = risk_metrics.get("stop_loss_price")
        tp = risk_metrics.get("take_profit_price")

        if cost <= 0 or units <= 0:
            logger.warning(
                "PaperTrader: BUY signal skipped — RiskManager returned zero sizing "
                "(cost=%.4f, units=%.8f).",
                cost,
                units,
            )
            return

        if cost > self.balance_usdt:
            logger.warning(
                "PaperTrader: BUY signal skipped — insufficient balance "
                "(needed=%.4f USDT, available=%.4f USDT).",
                cost,
                self.balance_usdt,
            )
            return

        self.balance_usdt -= cost
        self.open_positions[symbol] = Position(
            symbol=symbol,
            units=units,
            entry_price=current_price,
            cost_usdt=cost,
            stop_loss_price=sl,
            take_profit_price=tp,
        )

        logger.info(
            "PaperTrader [OPEN]: %s | units=%.8f | entry=%.4f | cost=%.4f USDT | "
            "SL=%.4f | TP=%.4f | balance_after=%.4f USDT",
            symbol,
            units,
            current_price,
            cost,
            sl,
            tp,
            self.balance_usdt,
        )

    def _close_position(
        self,
        symbol: str,
        current_price: float,
        close_reason: str,
    ) -> None:
        """Closes an existing simulated position and records the trade.

        Args:
            symbol: Trading pair identifier.
            current_price: Exit price in USDT.
            close_reason: One of ``"SIGNAL"``, ``"STOP_LOSS"``,
                ``"TAKE_PROFIT"``.
        """
        position = self.open_positions.get(symbol)
        if position is None:
            logger.warning(
                "PaperTrader: SELL/close signal skipped — no open position for %s.",
                symbol,
            )
            return

        proceeds = position.units * current_price
        pnl = proceeds - position.cost_usdt
        self.balance_usdt += proceeds

        record = TradeRecord(
            symbol=symbol,
            units=position.units,
            entry_price=position.entry_price,
            exit_price=current_price,
            pnl_usdt=pnl,
            close_reason=close_reason,
            opened_at=position.opened_at,
        )
        self.trade_history.append(record)
        del self.open_positions[symbol]

        pnl_sign = "+" if pnl >= 0 else ""
        logger.info(
            "PaperTrader [CLOSE %s]: %s | exit=%.4f | PnL=%s%.4f USDT | "
            "balance_after=%.4f USDT",
            close_reason,
            symbol,
            current_price,
            pnl_sign,
            pnl,
            self.balance_usdt,
        )
