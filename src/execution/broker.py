"""Binance broker module for NexusQuant.

Translates validated trading signals into real exchange orders via CCXT,
routing to either the Binance Spot Testnet (sandbox) or the live Binance Mainnet
depending on the BINANCE_TESTNET environment variable.

Security contract:
    * API keys are loaded exclusively from environment variables — never
      hardcoded, never logged.
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
DEFAULT_RECV_WINDOW: int = 10_000   # milliseconds — robust against latency


class BrokerError(Exception):
    """Base exception for all BinanceBroker errors.

    Wraps CCXT exceptions so callers do not need to import CCXT directly.
    """


class BinanceBroker:
    """Executes trades on the Binance Spot Exchange via CCXT.

    Supports both Binance Spot Testnet (sandbox) and live Mainnet based on
    the `BINANCE_TESTNET` environment variable.

    Attributes:
        exchange: The CCXT ``binance`` exchange instance.
        sandbox_mode: True if connected to Binance Testnet, False if Mainnet.
    """

    def __init__(self) -> None:
        """Loads credentials from environment and configures sandbox mode.

        Raises:
            BrokerError: If required credentials are missing.
            BrokerError: If the CCXT exchange object cannot be initialised.
        """
        is_testnet = os.getenv("BINANCE_TESTNET", "True").strip().lower() != "false"

        if is_testnet:
            # Fallback chain for testnet credentials
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "").strip() or os.getenv("BINANCE_API_KEY", "").strip()
            secret_key = os.getenv("BINANCE_TESTNET_SECRET_KEY", "").strip() or os.getenv("BINANCE_SECRET_KEY", "").strip()
            env_vars_source = "BINANCE_TESTNET_API_KEY / BINANCE_API_KEY"
        else:
            # Production Mainnet credentials only
            api_key = os.getenv("BINANCE_API_KEY", "").strip()
            secret_key = os.getenv("BINANCE_SECRET_KEY", "").strip()
            env_vars_source = "BINANCE_API_KEY"

        if not api_key or not secret_key:
            raise BrokerError(
                f"Missing credentials from environment ({env_vars_source}). "
                f"Please ensure your keys are configured correctly in the .env file."
            )

        mode_str = "TESTNET (Sandbox)" if is_testnet else "MAINNET (Live Production)"
        logger.info(
            "BinanceBroker: Initialising in %s mode (key length=%d, secret length=%d).",
            mode_str,
            len(api_key),
            len(secret_key),
        )

        try:
            self.exchange: ccxt.binance = ccxt.binance({
                "apiKey":          api_key,
                "secret":          secret_key,
                "enableRateLimit": True,
                "options": {
                    "recvWindow":              DEFAULT_RECV_WINDOW,
                    "defaultType":             "spot",
                    # Automatically measure and compensate for clock drift.
                    "adjustForTimeDifference": True,
                },
            })

            # Configure sandbox / testnet mode
            self.exchange.set_sandbox_mode(is_testnet)
            self.sandbox_mode: bool = is_testnet

            endpoint = (
                self.exchange.urls.get("test", {}).get("api", "testnet endpoint")
                if is_testnet
                else "https://api.binance.com"
            )

            logger.info(
                "BinanceBroker successfully initialised — exchange=%s, sandbox=%s, endpoint=%s",
                self.exchange.id,
                self.sandbox_mode,
                endpoint,
            )

        except ccxt.BaseError as exc:
            raise BrokerError(f"Failed to initialise CCXT exchange: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_free_balance(self, ticker: str = "USDT") -> float:
        """Returns the free (available) balance for a given currency.

        Args:
            ticker: Currency ticker to query (e.g. ``"USDT"``, ``"BTC"``).
                Defaults to ``"USDT"``.

        Returns:
            Free balance as a ``float``. Returns ``0.0`` if the currency
            is not present in the account.

        Raises:
            BrokerError: On authentication failure, network error, or any
                CCXT exception.
        """
        mode_str = "testnet" if self.sandbox_mode else "mainnet"
        logger.info("BinanceBroker: Fetching balance for %s on %s.", ticker, mode_str)
        try:
            balance_data = self.exchange.fetch_balance()
            free = float(balance_data.get("free", {}).get(ticker, 0.0))
            logger.info("BinanceBroker: Free %s balance = %.6f", ticker, free)
            return free

        except ccxt.AuthenticationError as exc:
            msg = (
                f"Authentication failed for Binance {mode_str.upper()}. "
                f"Please check that your API keys are valid. Error: {exc}"
            )
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.NetworkError as exc:
            msg = f"Network error fetching balance from Binance {mode_str.upper()}: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.BaseError as exc:
            msg = f"Unexpected CCXT error fetching balance: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> dict[str, Any]:
        """Places a spot order on the configured Binance network.

        Creates a **limit order** when ``price`` is provided, or a **market
        order** when ``price`` is ``None``.

        Args:
            symbol: Trading pair in CCXT format (e.g. ``"BTC/USDT"``).
            side: Order direction — ``"buy"`` or ``"sell"`` (case-insensitive).
            amount: Quantity of the base asset to trade (e.g. BTC units).
            price: Limit price in quote currency (USDT). ``None`` triggers
                a market order.

        Returns:
            The full CCXT order response dict containing at minimum:
            ``id``, ``status``, ``symbol``, ``side``, ``type``,
            ``amount``, ``price``, ``timestamp``.

        Raises:
            BrokerError: On insufficient funds, invalid order parameters,
                network errors, or any other CCXT exception.
            ValueError: If ``side`` is not ``"buy"`` or ``"sell"``.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side '{side}'. Must be 'buy' or 'sell'.")

        if amount <= 0:
            raise ValueError(f"Order amount must be positive, got {amount}.")

        order_type = "limit" if price is not None else "market"
        mode_str = "TESTNET" if self.sandbox_mode else "MAINNET"

        logger.info(
            "BinanceBroker [%s]: Placing %s %s order — "
            "symbol=%s  amount=%.8f  price=%s",
            mode_str,
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
                "BinanceBroker [%s]: Order placed — id=%s  status=%s  "
                "type=%s  side=%s  amount=%.8f  price=%s",
                mode_str,
                order.get("id"),
                order.get("status"),
                order.get("type"),
                order.get("side"),
                float(order.get("amount", 0)),
                order.get("price"),
            )
            return order

        except ccxt.InsufficientFunds as exc:
            msg = f"Insufficient funds on {mode_str} for {side.upper()} {amount} {symbol}. Error: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.InvalidOrder as exc:
            msg = f"Invalid order parameters on {mode_str} (symbol={symbol}, side={side}, amount={amount}, price={price}): {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.NetworkError as exc:
            msg = f"Network error placing order on {mode_str}: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.AuthenticationError as exc:
            msg = f"Authentication error on {mode_str} — check configured credentials: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc

        except ccxt.BaseError as exc:
            msg = f"Unexpected CCXT error executing order: {exc}"
            logger.error("BinanceBroker: %s", msg)
            raise BrokerError(msg) from exc
