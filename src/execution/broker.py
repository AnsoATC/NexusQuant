"""Binance broker module for NexusQuant.

Translates validated trading signals into real exchange orders via ccxt,
routed exclusively to the Binance Spot Testnet (sandbox mode).

Security contract:
    * API keys are loaded exclusively from environment variables — never
      hardcoded, never logged.
    * Sandbox mode is enforced in __init__ and cannot be overridden by callers.
    * All order-related errors are caught, logged at ERROR level, and re-raised
      as domain-specific exceptions so the execution layer can respond safely.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import ccxt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TESTNET_API_KEY_ENV: str = "BINANCE_TESTNET_API_KEY"
TESTNET_SECRET_KEY_ENV: str = "BINANCE_TESTNET_SECRET_KEY"
DEFAULT_RECV_WINDOW: int = 10_000   # milliseconds — wider window for testnet latency


class BrokerError(Exception):
    """Base exception for all BinanceBroker errors.

    Wraps ccxt exceptions so callers do not need to import ccxt directly.
    """


class BinanceBroker:
    """Executes trades on the Binance Spot Testnet via ccxt.

    All orders are routed to ``https://testnet.binance.vision`` through
    ccxt's built-in sandbox mode.  This class MUST NOT be used with mainnet
    keys — sandbox mode is enforced unconditionally in ``__init__``.

    Attributes:
        exchange: The ccxt ``binance`` exchange instance in sandbox mode.
        sandbox_mode: Always ``True`` — read-only guard for callers to inspect.

    Example:
        >>> broker = BinanceBroker()
        >>> balance = broker.get_free_balance("USDT")
        >>> print(f"Available USDT: {balance:.2f}")
    """

    def __init__(self) -> None:
        """Loads testnet credentials from environment and enables sandbox mode.

        Raises:
            BrokerError: If either ``BINANCE_TESTNET_API_KEY`` or
                ``BINANCE_TESTNET_SECRET_KEY`` environment variables are
                missing or empty.
            BrokerError: If the ccxt exchange object cannot be initialised
                (e.g. network unavailable at startup).
        """
        api_key    = os.getenv(TESTNET_API_KEY_ENV, "").strip()
        secret_key = os.getenv(TESTNET_SECRET_KEY_ENV, "").strip()

        if not api_key or not secret_key:
            missing = []
            if not api_key:
                missing.append(TESTNET_API_KEY_ENV)
            if not secret_key:
                missing.append(TESTNET_SECRET_KEY_ENV)
            raise BrokerError(
                f"Missing required environment variables: {missing}. "
                f"Copy .env.example → .env and fill in your Binance Testnet keys."
            )

        # Keys are intentionally never logged — only their presence is confirmed.
        logger.info(
            "BinanceBroker: Testnet credentials loaded from environment "
            "(key length=%d, secret length=%d).",
            len(api_key),
            len(secret_key),
        )

        try:
            self.exchange: ccxt.binance = ccxt.binance({
                "apiKey":      api_key,
                "secret":      secret_key,
                "enableRateLimit": True,
                "options": {
                    "recvWindow": DEFAULT_RECV_WINDOW,
                    "defaultType": "spot",
                },
            })

            # ── CRITICAL: enforce sandbox mode unconditionally ────────────────
            self.exchange.set_sandbox_mode(True)
            self.sandbox_mode: bool = True

            logger.info(
                "BinanceBroker initialised — exchange=%s  sandbox=True  "
                "endpoint=%s",
                self.exchange.id,
                self.exchange.urls.get("test", {}).get("api", "unknown"),
            )

        except ccxt.BaseError as exc:
            raise BrokerError(f"Failed to initialise ccxt exchange: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_free_balance(self, ticker: str = "USDT") -> float:
        """Returns the free (available) balance for a given currency.

        Args:
            ticker: Currency ticker to query (e.g. ``"USDT"``, ``"BTC"``).
                Defaults to ``"USDT"``.

        Returns:
            Free balance as a ``float``.  Returns ``0.0`` if the currency
            is not present in the account.

        Raises:
            BrokerError: On authentication failure, network error, or any
                ccxt exception.
        """
        logger.info("BinanceBroker: Fetching balance for %s (testnet).", ticker)
        try:
            balance_data = self.exchange.fetch_balance()
            free = float(balance_data.get("free", {}).get(ticker, 0.0))
            logger.info("BinanceBroker: Free %s balance = %.6f", ticker, free)
            return free

        except ccxt.AuthenticationError as exc:
            msg = (
                f"Authentication failed for Binance Testnet. "
                f"Check your keys in .env are valid testnet keys "
                f"(from https://testnet.binance.vision/). Error: {exc}"
            )
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.NetworkError as exc:
            msg = f"Network error fetching balance: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.BaseError as exc:
            msg = f"Unexpected ccxt error fetching balance: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> dict[str, Any]:
        """Places a spot order on the Binance Testnet.

        Creates a **limit order** when ``price`` is provided, or a **market
        order** when ``price`` is ``None``.

        Args:
            symbol: Trading pair in ccxt format (e.g. ``"BTC/USDT"``).
            side: Order direction — ``"buy"`` or ``"sell"`` (case-insensitive).
            amount: Quantity of the base asset to trade (e.g. BTC units).
            price: Limit price in quote currency (USDT).  ``None`` triggers
                a market order.

        Returns:
            The full ccxt order response dict containing at minimum:
            ``id``, ``status``, ``symbol``, ``side``, ``type``,
            ``amount``, ``price``, ``timestamp``.

        Raises:
            BrokerError: On insufficient funds, invalid order parameters,
                network errors, or any other ccxt exception.
            ValueError: If ``side`` is not ``"buy"`` or ``"sell"``.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side '{side}'. Must be 'buy' or 'sell'.")

        if amount <= 0:
            raise ValueError(f"Order amount must be positive, got {amount}.")

        order_type = "limit" if price is not None else "market"

        logger.info(
            "BinanceBroker [TESTNET]: Placing %s %s order — "
            "symbol=%s  amount=%.8f  price=%s",
            order_type.upper(),
            side.upper(),
            symbol,
            amount,
            f"{price:.4f}" if price is not None else "MARKET",
        )

        try:
            if order_type == "limit":
                order = self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=price,
                )
            else:
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                )

            logger.info(
                "BinanceBroker [TESTNET]: Order placed — id=%s  status=%s  "
                "type=%s  side=%s  amount=%.8f  price=%s",
                order.get("id"),
                order.get("status"),
                order.get("type"),
                order.get("side"),
                float(order.get("amount", 0)),
                order.get("price"),
            )
            return order

        except ccxt.InsufficientFunds as exc:
            msg = (
                f"Insufficient testnet funds for {side.upper()} {amount} {symbol}. "
                f"Top up at https://testnet.binance.vision/. Error: {exc}"
            )
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.InvalidOrder as exc:
            msg = f"Invalid order parameters (symbol={symbol}, side={side}, amount={amount}, price={price}): {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.NetworkError as exc:
            msg = f"Network error placing order on testnet: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.AuthenticationError as exc:
            msg = f"Authentication error on testnet — check .env keys: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.BaseError as exc:
            msg = f"Unexpected ccxt error executing order: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc
