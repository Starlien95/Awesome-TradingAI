from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from typing import Any, ClassVar

from quant_bench.runtime.core.execution_result import TradeExecutionResult
from quant_bench.trading import (
    AccountSnapshot,
    ExecutionReport,
    MarketType,
    OrderAction,
    OrderRequest,
    OrderType,
)

try:
    import okx.Account as account
    import okx.MarketData as MarketData
    import okx.Trade as trade

    HAS_OKX_LIB = True
except ImportError:
    HAS_OKX_LIB = False

logger = logging.getLogger(__name__)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return default if math.isnan(out) or math.isinf(out) else out
    except (TypeError, ValueError):
        return default


def _format_order_size(value: float, decimals: int = 8) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _first_okx_data(res: dict[str, Any]) -> dict[str, Any]:
    data = res.get("data") if isinstance(res, dict) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


class TradeExecutor:
    """
    Thin OKX wrapper used by ML/RL runners and FinGPT brokers.

    `execute_trade()` returns a `TradeExecutionResult`. The result implements
    `__bool__`, so old callers that only check success continue to work, while
    new audit logging can persist order ids, fills, fees, slippage, and raw
    exchange responses.
    """

    DEFAULT_FLAG = "1"
    DEFAULT_SWAP_LEVERAGE = 1
    TRANSIENT_OKX_CODES: ClassVar[set[str]] = {"429", "50004", "50011", "50013"}
    TRANSIENT_OKX_MARKERS: ClassVar[tuple[str, ...]] = (
        "connection aborted",
        "connection reset",
        "gateway timeout",
        "read timed out",
        "remote end closed",
        "server error",
        "temporarily unavailable",
        "too many",
        "timed out",
        "rate limit",
        "systems are busy",
        "request timeout",
        "please try again later",
    )
    DUPLICATE_CLIENT_ORDER_MARKERS = (
        "client order id already exists",
        "clordid already exists",
        "duplicate client order",
        "duplicate order",
    )
    REQUEST_RETRIES = 3
    REQUEST_RETRY_BASE_SLEEP_SEC = 0.5
    REQUEST_RETRY_MAX_SLEEP_SEC = 3.0
    REQUEST_MAX_ELAPSED_SEC = 8.0
    PRICE_CACHE_TTL_SEC = 3.0
    ACCOUNT_CACHE_TTL_SEC = 2.0
    ORDER_SUBMIT_MAX_ATTEMPTS = 5
    ORDER_SUBMIT_RETRY_DEADLINE_SEC = 12.0
    ORDER_SUBMIT_RETRY_BASE_SLEEP_SEC = 0.75
    ORDER_SUBMIT_RETRY_MAX_SLEEP_SEC = 3.0
    ORDER_RETRY_MAX_PRICE_DRIFT_BPS = 50.0
    ORDER_SETTLE_TIMEOUT_SEC = 8.0
    ORDER_SETTLE_POLL_INTERVAL_SEC = 1.0
    DEFAULT_MIN_ORDER_NOTIONAL_USDT = 0.0

    def __init__(
        self,
        identifier: str = "default_model",
        api_key: str = "",
        secret_key: str = "",
        passphrase: str = "",
        flag: str = "",
        trade_mode: str = "spot",
        allow_env_credentials: bool = True,
    ):
        self.identifier = identifier
        if allow_env_credentials:
            self.api_key = api_key or os.getenv("OKX_API_KEY_SIMU") or os.getenv("OKX_API_KEY") or ""
            self.secret_key = secret_key or os.getenv("OKX_SECRET_KEY_SIMU") or os.getenv("OKX_SECRET_KEY") or ""
            self.passphrase = passphrase or os.getenv("OKX_PASSPHRASE") or ""
        else:
            self.api_key = api_key or ""
            self.secret_key = secret_key or ""
            self.passphrase = passphrase or ""
        self._okx_flag = (flag or os.getenv("OKX_FLAG") or self.DEFAULT_FLAG).strip() or self.DEFAULT_FLAG
        self.trade_mode = trade_mode
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._account_balance_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._positions_cache: dict[str, tuple[float, dict[str, Any]]] = {}

        if HAS_OKX_LIB:
            target_domain = "https://www.okx.com"
            api_params = {}
            if self.api_key and self.secret_key and self.passphrase:
                api_params = {
                    "api_key": self.api_key,
                    "api_secret_key": self.secret_key,
                    "passphrase": self.passphrase,
                }

            self.market_api = MarketData.MarketAPI(use_server_time=False, flag=self._okx_flag, domain=target_domain)
            self.account_api = account.AccountAPI(use_server_time=False, flag=self._okx_flag, domain=target_domain, **api_params)
            self.trade_api = trade.TradeAPI(use_server_time=False, flag=self._okx_flag, domain=target_domain, **api_params)
            logger.info(
                "[%s] OKX %s API ready (mode=%s)",
                self.identifier,
                "demo" if self._okx_flag == "1" else "live",
                self.trade_mode,
            )
        else:
            self.market_api = None
            self.account_api = None
            self.trade_api = None
            logger.warning("[%s] python-okx is not installed", self.identifier)

    @property
    def account_mode(self) -> str:
        return "demo" if self._okx_flag == "1" else "live"

    @property
    def backend_id(self) -> str:
        return f"okx-sdk:{self.identifier}"

    @property
    def execution_mode(self) -> str:
        return "simulated" if self._okx_flag == "1" else "live"

    def account_snapshot(self, request: OrderRequest) -> AccountSnapshot:
        raw = self.get_account_snapshot(
            [request.instrument_id],
            {request.instrument_id: float(request.reference_price or 0.0)},
            force_refresh=True,
        )
        return AccountSnapshot.from_mapping(
            raw,
            backend_id=self.backend_id,
            execution_mode=self.execution_mode,
            instrument_id=request.instrument_id,
            reference_price=request.reference_price,
        )

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        expected_market = MarketType.SWAP if self.trade_mode == "swap" else MarketType.SPOT
        if request.market_type is not expected_market:
            return ExecutionReport.failed(
                request,
                backend_id=self.backend_id,
                execution_mode=self.execution_mode,
                error=ValueError(
                    f"order market {request.market_type.value} does not match executor market "
                    f"{expected_market.value}"
                ),
            )
        if request.order_type is not OrderType.MARKET:
            return ExecutionReport.failed(
                request,
                backend_id=self.backend_id,
                execution_mode=self.execution_mode,
                error=ValueError("TradeExecutor currently supports market orders only"),
            )
        if request.action is OrderAction.HOLD:
            return ExecutionReport.from_mapping(
                request,
                {"status": "skipped", "submitted": False, "executed_action": "hold"},
                backend_id=self.backend_id,
                execution_mode=self.execution_mode,
            )
        if request.action is OrderAction.CLOSE:
            close_marker = self.clear_positions(
                request.instrument_id,
                float(request.reference_price or 0.0),
            )
            return ExecutionReport.from_mapping(
                request,
                {
                    "status": "submitted" if close_marker > 0 else "skipped",
                    "submitted": close_marker > 0,
                    "executed_action": "close" if close_marker > 0 else "hold",
                    "expected_price": request.reference_price,
                    "estimated_price": request.reference_price,
                },
                backend_id=self.backend_id,
                execution_mode=self.execution_mode,
            )

        reference_price = float(request.reference_price or 0.0)
        if request.notional_usdt is not None:
            amount = float(request.notional_usdt)
        elif request.quantity is not None:
            amount = (
                float(request.quantity)
                if request.market_type is MarketType.SWAP
                else float(request.quantity) * reference_price
            )
        else:
            return ExecutionReport.failed(
                request,
                backend_id=self.backend_id,
                execution_mode=self.execution_mode,
                error=ValueError("TradeExecutor cannot infer an order size"),
            )
        result = self.execute_trade(
            request.instrument_id,
            request.action.value,
            amount,
            reference_price,
            client_order_id=request.client_order_id,
            min_notional_usdt=float(request.metadata.get("min_notional_usdt", 0.0) or 0.0),
            max_sell_qty=(
                request.metadata.get("max_sell_qty", request.quantity)
                if request.market_type is MarketType.SPOT and request.action is OrderAction.SELL
                else None
            ),
        )
        payload = result.to_order_row()
        payload.update(
            {
                "executed_action": request.action.value if result.success else "hold",
                "actual_avg_fill_price": result.avg_fill_price or None,
                "raw_order_response": result.raw_response,
                "raw_fill_response": result.fills_response,
            }
        )
        return ExecutionReport.from_mapping(
            request,
            payload,
            backend_id=self.backend_id,
            execution_mode=self.execution_mode,
        )

    def _retry_sleep_sec(self, base_sleep_sec: float, attempt: int, max_sleep_sec: float | None = None) -> float:
        max_sleep = self.REQUEST_RETRY_MAX_SLEEP_SEC if max_sleep_sec is None else max_sleep_sec
        return min(max_sleep, max(0.0, base_sleep_sec) * (2 ** max(0, attempt)))

    def _sleep_before_retry(self, sleep_sec: float, deadline_ts: float | None = None) -> bool:
        if deadline_ts is not None:
            remaining = deadline_ts - time.time()
            if remaining <= 0:
                return False
            sleep_sec = min(max(0.0, sleep_sec), remaining)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        return True

    def _response_codes(self, res: Any) -> tuple[str, str, str, str]:
        first = _first_okx_data(res) if isinstance(res, dict) else {}
        return (
            str(res.get("code", "") if isinstance(res, dict) else ""),
            str(first.get("sCode", "") or ""),
            str(res.get("msg", "") if isinstance(res, dict) else ""),
            str(first.get("sMsg", "") or ""),
        )

    def _okx_response_accepted(self, res: Any) -> bool:
        okx_code, okx_s_code, _, _ = self._response_codes(res)
        return okx_code == "0" and okx_s_code in {"", "0"}

    def _is_duplicate_client_order_response(self, res: Any) -> bool:
        okx_code, okx_s_code, okx_msg, okx_s_msg = self._response_codes(res)
        text = f"{okx_code} {okx_s_code} {okx_msg} {okx_s_msg}".lower()
        return any(marker in text for marker in self.DUPLICATE_CLIENT_ORDER_MARKERS)

    def _call_okx_request(
        self,
        label: str,
        call: Callable[[], Any],
        *,
        retries: int | None = None,
        base_sleep_sec: float | None = None,
        max_sleep_sec: float | None = None,
        max_elapsed_sec: float | None = None,
    ) -> Any:
        attempts = max(1, int(retries or self.REQUEST_RETRIES))
        base_sleep = self.REQUEST_RETRY_BASE_SLEEP_SEC if base_sleep_sec is None else base_sleep_sec
        max_sleep = self.REQUEST_RETRY_MAX_SLEEP_SEC if max_sleep_sec is None else max_sleep_sec
        max_elapsed = self.REQUEST_MAX_ELAPSED_SEC if max_elapsed_sec is None else max_elapsed_sec
        deadline_ts = time.time() + max_elapsed if max_elapsed and max_elapsed > 0 else None
        last_res: Any = {}
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                res = call()
                if not isinstance(res, dict):
                    return res
                okx_code, okx_s_code, okx_msg, okx_s_msg = self._response_codes(res)
                if not self._is_transient_okx_error(okx_code, okx_s_code, okx_msg, okx_s_msg):
                    return res
                last_res = res
                logger.warning(
                    "[%s] transient OKX %s response attempt %s/%s: code=%s sCode=%s msg=%s sMsg=%s",
                    self.identifier,
                    label,
                    attempt + 1,
                    attempts,
                    okx_code,
                    okx_s_code,
                    okx_msg,
                    okx_s_msg,
                )
            except Exception as exc:
                last_exc = exc
                if not self._is_transient_okx_error(okx_msg=str(exc)):
                    raise
                logger.warning(
                    "[%s] transient OKX %s exception attempt %s/%s: %s",
                    self.identifier,
                    label,
                    attempt + 1,
                    attempts,
                    exc,
                )

            if attempt >= attempts - 1:
                break
            sleep_sec = self._retry_sleep_sec(base_sleep, attempt, max_sleep)
            if not self._sleep_before_retry(sleep_sec, deadline_ts):
                break

        if last_res:
            return last_res
        if last_exc is not None:
            raise last_exc
        return {}

    def _cached_account_balance_response(self, ccy: str | None = None) -> dict[str, Any] | None:
        key = ccy or "__all__"
        cached = self._account_balance_cache.get(key)
        if not cached:
            return None
        ts, res = cached
        if time.time() - ts <= self.ACCOUNT_CACHE_TTL_SEC:
            return res
        return None

    def _get_account_balance_response(self, ccy: str | None = None, force_refresh: bool = False) -> dict[str, Any]:
        if not self.account_api:
            return {}
        key = ccy or "__all__"
        if not force_refresh:
            cached = self._cached_account_balance_response(ccy)
            if cached is not None:
                return cached

        def _call() -> Any:
            if ccy:
                return self.account_api.get_account_balance(ccy=ccy)
            return self.account_api.get_account_balance()

        res = self._call_okx_request(f"account_balance {ccy or 'all'}", _call)
        if isinstance(res, dict) and res.get("code") == "0":
            self._account_balance_cache[key] = (time.time(), res)
        return res if isinstance(res, dict) else {}

    def _get_positions_response(self, inst_type: str = "SWAP", force_refresh: bool = False) -> dict[str, Any]:
        if not self.account_api:
            return {}
        key = inst_type
        cached = self._positions_cache.get(key)
        if not force_refresh and cached and time.time() - cached[0] <= self.ACCOUNT_CACHE_TTL_SEC:
            return cached[1]
        res = self._call_okx_request(
            f"positions {inst_type}",
            lambda: self.account_api.get_positions(instType=inst_type),
        )
        if isinstance(res, dict) and res.get("code") == "0":
            self._positions_cache[key] = (time.time(), res)
        return res if isinstance(res, dict) else {}

    def _invalidate_account_cache(self) -> None:
        self._account_balance_cache.clear()
        self._positions_cache.clear()

    def get_coin_kline(self, instId: str, bar: str = "15m", limit: int = 100) -> list:
        if bar.endswith("h"):
            bar = bar.replace("h", "H")
        elif bar.endswith("d"):
            bar = bar.replace("d", "D")
        elif bar.endswith("w"):
            bar = bar.replace("w", "W")

        if HAS_OKX_LIB and self.market_api:
            try:
                res = self._call_okx_request(
                    f"candlesticks {instId}",
                    lambda: self.market_api.get_candlesticks(instId=instId, bar=bar, limit=str(limit)),
                    retries=2,
                    base_sleep_sec=0.8,
                    max_elapsed_sec=4.0,
                )
                if res.get("code") == "0":
                    return res.get("data", [])
                logger.warning("[%s] kline API error %s: %s", self.identifier, res.get("code"), res.get("msg"))
            except Exception as exc:
                logger.warning("[%s] get_coin_kline %s failed: %s", self.identifier, instId, exc)
        return []

    def get_current_price(
        self,
        instId: str,
        retries: int = 3,
        retry_sleep_sec: float = 0.5,
        use_cache: bool = True,
    ) -> float:
        if not HAS_OKX_LIB or not self.market_api:
            return 0.0
        cached = self._price_cache.get(instId)
        if use_cache and cached and time.time() - cached[0] <= self.PRICE_CACHE_TTL_SEC:
            return cached[1]
        last_note = ""
        try:
            res = self._call_okx_request(
                f"ticker {instId}",
                lambda: self.market_api.get_ticker(instId=instId),
                retries=retries,
                base_sleep_sec=retry_sleep_sec,
                max_elapsed_sec=max(1.0, retries * max(retry_sleep_sec, 0.1) + 1.0),
            )
            if res.get("code") == "0" and res.get("data"):
                price = float(res["data"][0]["last"])
                if price > 0:
                    self._price_cache[instId] = (time.time(), price)
                return price
            last_note = f"code={res.get('code')} msg={res.get('msg')}"
        except Exception as exc:
            last_note = str(exc)
            logger.warning("[%s] get_current_price %s failed: %s", self.identifier, instId, exc)
        if last_note:
            logger.warning("[%s] get_current_price %s gave up: %s", self.identifier, instId, last_note)
        return 0.0

    def get_holdings(self, force_refresh: bool = False) -> dict[str, float]:
        if not HAS_OKX_LIB or not self.account_api:
            return {}
        try:
            inst_type = "SWAP" if self.trade_mode == "swap" else "SPOT"
            if inst_type == "SWAP":
                res = self._get_positions_response("SWAP", force_refresh=force_refresh)
                if res.get("code") != "0":
                    logger.warning("[%s] get_holdings positions error %s: %s", self.identifier, res.get("code"), res.get("msg"))
                    return {}
                out: dict[str, float] = {}
                for item in res.get("data", []):
                    qty = _to_float(item.get("pos"), 0.0)
                    if qty == 0:
                        continue
                    pos_side = item.get("posSide", "long")
                    out[item.get("instId", "")] = qty if pos_side == "long" else -qty
                return out

            res = self._get_account_balance_response(force_refresh=force_refresh)
            if res.get("code") != "0":
                logger.warning("[%s] get_holdings balance error %s: %s", self.identifier, res.get("code"), res.get("msg"))
                return {}
            out: dict[str, float] = {}
            details = res.get("data", [{}])[0].get("details", [])
            for detail in details:
                qty = _to_float(detail.get("availBal"), 0.0)
                ccy = detail.get("ccy")
                if qty > 0 and ccy and ccy != "USDT":
                    out[f"{ccy}-USDT"] = qty
            return out
        except Exception as exc:
            logger.warning("[%s] get_holdings failed: %s", self.identifier, exc)
            return {}

    def get_usdt_balance(self, force_refresh: bool = False) -> float:
        if not HAS_OKX_LIB or not self.account_api:
            return 0.0

        def _extract_usdt_from_balance_response(res: dict[str, Any]) -> float:
            if res.get("code") != "0" or not res.get("data"):
                return 0.0
            first = res["data"][0] if res["data"] else {}
            details = first.get("details", []) or []
            for detail in details:
                if detail.get("ccy") != "USDT":
                    continue
                for key in ("availEq", "eq", "cashBal", "availBal", "frozenBal"):
                    value = _to_float(detail.get(key), 0.0)
                    if value > 0:
                        return value
            for key in ("totalEq", "adjEq", "isoEq"):
                value = _to_float(first.get(key), 0.0)
                if value > 0:
                    return value
            return 0.0

        try:
            cached_all = None if force_refresh else self._cached_account_balance_response()
            if cached_all is not None:
                bal = _extract_usdt_from_balance_response(cached_all)
                if bal > 0:
                    return bal
            bal = _extract_usdt_from_balance_response(
                self._get_account_balance_response("USDT", force_refresh=force_refresh)
            )
            if bal > 0:
                return bal
            return _extract_usdt_from_balance_response(
                self._get_account_balance_response(force_refresh=force_refresh)
            )
        except Exception as exc:
            logger.warning("[%s] get_usdt_balance failed: %s", self.identifier, exc)
            return 0.0

    def get_account_snapshot(
        self,
        coins: list[str] | None = None,
        prices: dict[str, float] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        holdings = self.get_holdings(force_refresh=force_refresh)
        prices = prices or {}
        requested_coins = set(coins or [])
        coins_to_value = sorted(set(holdings) | requested_coins)
        holdings_value = 0.0
        for coin in coins_to_value:
            qty = float(holdings.get(coin, 0.0) or 0.0)
            price = float(prices.get(coin, 0.0) or 0.0)
            if qty > 0 and price <= 0:
                price = self.get_current_price(coin)
            if qty > 0 and price > 0:
                holdings_value += qty * price
        usdt = self.get_usdt_balance(force_refresh=force_refresh)
        return {
            "account_mode": self.account_mode,
            "trade_mode": self.trade_mode,
            "usdt_available": usdt,
            "holdings_value_usdt": holdings_value,
            "estimated_total_equity_usdt": usdt + holdings_value,
            "holdings_json": holdings,
        }

    def set_leverage(self, instId: str, lever: int | None = None, mgnMode: str = "cross") -> None:
        if self.trade_mode != "swap" or not HAS_OKX_LIB or not self.account_api:
            return
        lv = self.DEFAULT_SWAP_LEVERAGE if lever is None else int(lever)
        try:
            res = self._call_okx_request(
                f"set_leverage {instId}",
                lambda: self.account_api.set_leverage(instId=instId, lever=str(lv), mgnMode=mgnMode),
            )
            if res.get("code") == "0":
                logger.info("[%s] set leverage ok: %s %sx", self.identifier, instId, lv)
            else:
                logger.debug("[%s] set leverage response: %s", self.identifier, res.get("msg"))
        except Exception as exc:
            logger.warning("[%s] set_leverage failed: %s", self.identifier, exc)

    def _client_order_id(self, coin: str, side: str) -> str:
        safe_identifier = "".join(ch for ch in self.identifier if ch.isalnum())[-10:] or "qtf"
        safe_coin = "".join(ch for ch in coin if ch.isalnum())[:10]
        return f"QTF{safe_identifier}{safe_coin}{side[:1]}{int(time.time() * 1000) % 10_000_000_000}"

    def _is_transient_okx_error(
        self,
        okx_code: str = "",
        okx_s_code: str = "",
        okx_msg: str = "",
        okx_s_msg: str = "",
    ) -> bool:
        codes = {str(okx_code or ""), str(okx_s_code or "")}
        if any(code in self.TRANSIENT_OKX_CODES for code in codes):
            return True
        text = f"{okx_msg or ''} {okx_s_msg or ''}".lower()
        return any(marker in text for marker in self.TRANSIENT_OKX_MARKERS)

    def _query_order(
        self,
        coin: str,
        ord_id: str = "",
        cl_ord_id: str = "",
        retries: int = 3,
        retry_sleep_sec: float = 1.0,
    ) -> dict[str, Any]:
        if (not ord_id and not cl_ord_id) or not self.trade_api:
            return {}
        method = getattr(self.trade_api, "get_order", None)
        if not method:
            return {}
        candidates: list[dict[str, str]] = []
        if ord_id and cl_ord_id:
            candidates.append({"instId": coin, "ordId": ord_id, "clOrdId": cl_ord_id})
        if ord_id:
            candidates.append({"instId": coin, "ordId": ord_id})
        if cl_ord_id:
            candidates.append({"instId": coin, "clOrdId": cl_ord_id})
        last: dict[str, Any] = {}
        for attempt in range(max(1, retries)):
            for kwargs in candidates:
                try:
                    last = method(**kwargs)
                    if not self._is_transient_okx_error(
                        okx_code=str(last.get("code", "") if isinstance(last, dict) else ""),
                        okx_msg=str(last.get("msg", "") if isinstance(last, dict) else ""),
                    ):
                        return last
                except TypeError:
                    continue
                except Exception as exc:
                    logger.debug(
                        "[%s] get_order %s/%s/%s failed: %s",
                        self.identifier,
                        coin,
                        ord_id,
                        cl_ord_id,
                        exc,
                    )
                    last = {}
                    break
            if attempt < retries - 1:
                time.sleep(retry_sleep_sec * (attempt + 1))
        return last

    def _order_response_has_order(self, order_response: dict[str, Any]) -> bool:
        if not isinstance(order_response, dict) or order_response.get("code") != "0":
            return False
        data = order_response.get("data") or []
        return bool(isinstance(data, list) and data and isinstance(data[0], dict))

    def _query_fills(self, coin: str, ord_id: str, retries: int = 3, retry_sleep_sec: float = 1.0) -> dict[str, Any]:
        if not ord_id or not self.trade_api:
            return {}
        method = getattr(self.trade_api, "get_fills", None)
        if not method:
            return {}
        last: dict[str, Any] = {}
        for attempt in range(max(1, retries)):
            for kwargs in (
                {"instId": coin, "ordId": ord_id},
                {"ordId": ord_id},
                {"instType": "SPOT" if self.trade_mode == "spot" else "SWAP", "instId": coin, "ordId": ord_id},
            ):
                try:
                    last = method(**kwargs)
                    if not self._is_transient_okx_error(
                        okx_code=str(last.get("code", "") if isinstance(last, dict) else ""),
                        okx_msg=str(last.get("msg", "") if isinstance(last, dict) else ""),
                    ):
                        return last
                except TypeError:
                    continue
                except Exception as exc:
                    logger.debug("[%s] get_fills %s/%s failed: %s", self.identifier, coin, ord_id, exc)
                    last = {}
                    break
            if attempt < retries - 1:
                time.sleep(retry_sleep_sec * (attempt + 1))
        return {}

    def _make_trade_result(
        self,
        *,
        response: Any,
        coin: str,
        side: str,
        action: str,
        amount_usdt: float,
        requested_qty: float,
        submitted_sz: str,
        price: float,
        submit_ts: float,
        ack_ts: float,
        td_mode: str,
        tgt_ccy: str,
        cl_ord_id: str,
        fallback_status: str = "rejected",
        error_type: str = "exchange_rejected",
    ) -> TradeExecutionResult:
        first = _first_okx_data(response)
        okx_code, okx_s_code, okx_msg, okx_s_msg = self._response_codes(response)
        success = okx_code == "0" and okx_s_code in {"", "0"}
        result = TradeExecutionResult(
            success=success,
            status="accepted" if success else fallback_status,
            account_mode=self.account_mode,
            trade_mode=self.trade_mode,
            inst_id=coin,
            side=side,
            action=action,
            ord_type="market",
            td_mode=td_mode,
            tgt_ccy=tgt_ccy,
            requested_notional_usdt=float(amount_usdt or 0.0),
            requested_qty=float(requested_qty or 0.0),
            submitted_sz=str(submitted_sz),
            reference_price=float(price or 0.0),
            submit_ts=submit_ts,
            ack_ts=ack_ts,
            ord_id=str(first.get("ordId", "") or ""),
            cl_ord_id=str(first.get("clOrdId", cl_ord_id) or cl_ord_id),
            okx_code=okx_code,
            okx_msg=okx_msg,
            okx_s_code=okx_s_code,
            okx_s_msg=okx_s_msg,
            raw_response=response if isinstance(response, dict) else {"response": response},
            error_type="" if success else error_type,
            error_message="" if success else (okx_s_msg or okx_msg or str(response)),
        )
        return result

    def _result_from_existing_order(
        self,
        *,
        order_response: dict[str, Any],
        coin: str,
        side: str,
        action: str,
        amount_usdt: float,
        requested_qty: float,
        submitted_sz: str,
        price: float,
        submit_ts: float,
        ack_ts: float,
        td_mode: str,
        tgt_ccy: str,
        cl_ord_id: str,
        order_poll_delay_sec: float,
    ) -> TradeExecutionResult:
        result = self._make_trade_result(
            response=order_response,
            coin=coin,
            side=side,
            action=action,
            amount_usdt=amount_usdt,
            requested_qty=requested_qty,
            submitted_sz=submitted_sz,
            price=price,
            submit_ts=submit_ts,
            ack_ts=ack_ts,
            td_mode=td_mode,
            tgt_ccy=tgt_ccy,
            cl_ord_id=cl_ord_id,
            fallback_status="accepted",
        )
        result.update_from_order_response(order_response)
        logger.info(
            "[%s] order found by client id after retry uncertainty: %s %s %s",
            self.identifier,
            result.cl_ord_id,
            result.ord_id,
            result.status,
        )
        return self._finalize_execution_result(result, order_poll_delay_sec)

    def _find_existing_order_by_client_id(
        self,
        *,
        coin: str,
        side: str,
        action: str,
        amount_usdt: float,
        requested_qty: float,
        submitted_sz: str,
        price: float,
        submit_ts: float,
        ack_ts: float,
        td_mode: str,
        tgt_ccy: str,
        cl_ord_id: str,
        order_poll_delay_sec: float = 0.0,
        retries: int = 1,
    ) -> TradeExecutionResult | None:
        if not cl_ord_id:
            return None
        order_response = self._query_order(
            coin,
            cl_ord_id=cl_ord_id,
            retries=max(1, retries),
            retry_sleep_sec=0.5,
        )
        if not self._order_response_has_order(order_response):
            return None
        return self._result_from_existing_order(
            order_response=order_response,
            coin=coin,
            side=side,
            action=action,
            amount_usdt=amount_usdt,
            requested_qty=requested_qty,
            submitted_sz=submitted_sz,
            price=price,
            submit_ts=submit_ts,
            ack_ts=ack_ts,
            td_mode=td_mode,
            tgt_ccy=tgt_ccy,
            cl_ord_id=cl_ord_id,
            order_poll_delay_sec=order_poll_delay_sec,
        )

    def _retry_price_is_fresh(self, coin: str, reference_price: float, max_drift_bps: float) -> tuple[bool, float, float]:
        if reference_price <= 0 or max_drift_bps <= 0:
            return True, 0.0, 0.0
        current_price = self.get_current_price(coin, retries=1, retry_sleep_sec=0.0, use_cache=False)
        if current_price <= 0:
            return True, 0.0, 0.0
        drift_bps = abs(current_price - reference_price) / reference_price * 10_000.0
        return drift_bps <= max_drift_bps, current_price, drift_bps

    def _finalize_execution_result(self, result: TradeExecutionResult, order_poll_delay_sec: float) -> TradeExecutionResult:
        if result.success and result.ord_id:
            if order_poll_delay_sec > 0:
                time.sleep(order_poll_delay_sec)
            deadline_ts = time.time() + max(0.0, self.ORDER_SETTLE_TIMEOUT_SEC)
            terminal_states = {"filled", "canceled", "cancelled", "partially_filled"}
            while True:
                order_response = self._query_order(result.inst_id, result.ord_id, result.cl_ord_id)
                if order_response:
                    result.update_from_order_response(order_response)
                fills_response = self._query_fills(result.inst_id, result.ord_id)
                if fills_response:
                    result.update_from_fills_response(fills_response)
                if result.fill_verified or result.status in terminal_states or time.time() >= deadline_ts:
                    break
                self._sleep_before_retry(self.ORDER_SETTLE_POLL_INTERVAL_SEC, deadline_ts)
        result.settle_ts = result.settle_ts or time.time()
        result.recompute_derived()
        if result.ord_id:
            self._invalidate_account_cache()
        if result.status in {"canceled", "cancelled"} and float(result.filled_qty or 0.0) <= 0.0:
            result.success = False
            result.error_type = result.error_type or "order_canceled"
            result.error_message = result.error_message or "order canceled without fill"
        if (
            result.success
            and result.status in {"accepted", "live", "submitted", "submitted_unverified"}
            and float(result.filled_qty or 0.0) <= 0.0
        ):
            result.success = False
            result.error_type = result.error_type or "fill_unverified"
            result.error_message = result.error_message or "order accepted but no fill confirmed"
        if result.success and not result.status:
            result.status = "filled" if result.fill_verified else "submitted_unverified"
        return result

    def execute_trade(
        self,
        coin: str,
        action: str,
        amount_usdt: float,
        price: float,
        client_order_id: str | None = None,
        order_poll_delay_sec: float = 1.0,
        min_notional_usdt: float = 0.0,
        max_sell_qty: float | None = None,
        max_submit_attempts: int | None = None,
        submit_retry_deadline_sec: float | None = None,
        max_retry_price_drift_bps: float | None = None,
    ) -> TradeExecutionResult:
        side = "buy" if action == "buy" else "sell"
        min_order_notional = max(
            0.0,
            float(min_notional_usdt or self.DEFAULT_MIN_ORDER_NOTIONAL_USDT or 0.0),
        )
        if price <= 0 or amount_usdt <= 0 or not HAS_OKX_LIB or not self.trade_api:
            return TradeExecutionResult.failure(
                inst_id=coin,
                side=side,
                action=action,
                requested_notional_usdt=amount_usdt,
                reference_price=price,
                trade_mode=self.trade_mode,
                error_type="precheck_failed",
                error_message="invalid price/amount or OKX trade API unavailable",
            )
        if min_order_notional > 0 and amount_usdt < min_order_notional:
            skipped = TradeExecutionResult.failure(
                inst_id=coin,
                side=side,
                action=action,
                requested_notional_usdt=amount_usdt,
                reference_price=price,
                trade_mode=self.trade_mode,
                error_type="order_notional_too_small",
                error_message=f"requested notional {amount_usdt:.4f} < min_notional {min_order_notional:.4f}",
            )
            skipped.status = "skipped"
            return skipped

        try:
            requested_qty = 0.0
            if self.trade_mode == "swap":
                self.set_leverage(coin, mgnMode="cross")
                td_mode = "cross"
                sz = str(round(amount_usdt, 2))
                tgt_ccy = "quote_ccy"
            else:
                td_mode = "cash"
                if side == "buy":
                    sz = str(round(amount_usdt, 2))
                    tgt_ccy = "quote_ccy"
                else:
                    holdings = self.get_holdings(force_refresh=True)
                    max_qty = max(0.0, holdings.get(coin, 0.0))
                    target_qty = min(max_qty, amount_usdt / price)
                    if max_sell_qty is not None:
                        target_qty = min(target_qty, max(0.0, float(max_sell_qty or 0.0)))
                    target_notional = target_qty * price
                    if target_qty <= 0:
                        return TradeExecutionResult.failure(
                            inst_id=coin,
                            side=side,
                            action=action,
                            requested_notional_usdt=amount_usdt,
                            requested_qty=target_qty,
                            reference_price=price,
                            trade_mode=self.trade_mode,
                            td_mode=td_mode,
                            tgt_ccy="base_ccy",
                            error_type="no_sellable_position",
                            error_message="no available spot holding to sell",
                        )
                    if min_order_notional > 0 and target_notional < min_order_notional:
                        skipped = TradeExecutionResult.failure(
                            inst_id=coin,
                            side=side,
                            action=action,
                            requested_notional_usdt=amount_usdt,
                            requested_qty=target_qty,
                            reference_price=price,
                            trade_mode=self.trade_mode,
                            td_mode=td_mode,
                            tgt_ccy="base_ccy",
                            submitted_sz=_format_order_size(target_qty),
                            error_type="order_notional_too_small",
                            error_message=(
                                f"sellable notional {target_notional:.4f} < "
                                f"min_notional {min_order_notional:.4f}"
                            ),
                        )
                        skipped.status = "skipped"
                        return skipped
                    requested_qty = target_qty
                    sz = _format_order_size(target_qty)
                    tgt_ccy = "base_ccy"

            cl_ord_id = client_order_id or self._client_order_id(coin, side)
            submit_ts = time.time()
            ack_ts = submit_ts
            order_kwargs = {
                "instId": coin,
                "tdMode": td_mode,
                "side": side,
                "ordType": "market",
                "sz": sz,
                "tgtCcy": tgt_ccy,
                "clOrdId": cl_ord_id,
            }
            attempts = max(1, int(max_submit_attempts or self.ORDER_SUBMIT_MAX_ATTEMPTS))
            retry_deadline_sec = max(
                0.0,
                float(submit_retry_deadline_sec or self.ORDER_SUBMIT_RETRY_DEADLINE_SEC),
            )
            deadline_ts = submit_ts + retry_deadline_sec if retry_deadline_sec > 0 else None
            max_drift_bps = float(
                self.ORDER_RETRY_MAX_PRICE_DRIFT_BPS
                if max_retry_price_drift_bps is None
                else max_retry_price_drift_bps
            )
            res: dict[str, Any] | Any = {}
            client_order_supported = True
            last_error = ""

            def _exchange_cl_ord_id() -> str:
                return cl_ord_id if client_order_supported else ""

            def _submission_failure(error_type: str, message: str, raw_response: Any = None) -> TradeExecutionResult:
                failure = TradeExecutionResult.failure(
                    inst_id=coin,
                    side=side,
                    action=action,
                    requested_notional_usdt=amount_usdt,
                    requested_qty=requested_qty,
                    submitted_sz=str(sz),
                    reference_price=price,
                    trade_mode=self.trade_mode,
                    td_mode=td_mode,
                    tgt_ccy=tgt_ccy,
                    error_type=error_type,
                    error_message=message,
                    raw_response=raw_response if isinstance(raw_response, dict) else {"response": raw_response} if raw_response else {},
                    submit_ts=submit_ts,
                    ack_ts=ack_ts,
                )
                failure.cl_ord_id = _exchange_cl_ord_id()
                return failure

            def _place_order_once() -> Any:
                nonlocal client_order_supported
                try:
                    return self.trade_api.place_order(**order_kwargs)
                except TypeError as exc:
                    if "clOrdId" not in order_kwargs or "clordid" not in str(exc).lower():
                        raise
                    order_kwargs.pop("clOrdId", None)
                    client_order_supported = False
                    logger.warning(
                        "[%s] OKX client library rejected clOrdId, submitting without client id: %s",
                        self.identifier,
                        exc,
                    )
                    return self.trade_api.place_order(**order_kwargs)

            for attempt in range(attempts):
                if deadline_ts is not None and time.time() >= deadline_ts:
                    existing = self._find_existing_order_by_client_id(
                        coin=coin,
                        side=side,
                        action=action,
                        amount_usdt=amount_usdt,
                        requested_qty=requested_qty,
                        submitted_sz=str(sz),
                        price=price,
                        submit_ts=submit_ts,
                        ack_ts=ack_ts,
                        td_mode=td_mode,
                        tgt_ccy=tgt_ccy,
                        cl_ord_id=_exchange_cl_ord_id(),
                        order_poll_delay_sec=0.0,
                        retries=2,
                    )
                    if existing:
                        return existing
                    return _submission_failure(
                        "submit_retry_expired",
                        f"order submit retry deadline exceeded after {time.time() - submit_ts:.2f}s",
                        res,
                    )

                if attempt > 0 or client_order_id:
                    existing = self._find_existing_order_by_client_id(
                        coin=coin,
                        side=side,
                        action=action,
                        amount_usdt=amount_usdt,
                        requested_qty=requested_qty,
                        submitted_sz=str(sz),
                        price=price,
                        submit_ts=submit_ts,
                        ack_ts=ack_ts,
                        td_mode=td_mode,
                        tgt_ccy=tgt_ccy,
                        cl_ord_id=_exchange_cl_ord_id(),
                        order_poll_delay_sec=0.0,
                        retries=1,
                    )
                    if existing:
                        return existing

                try:
                    res = _place_order_once()
                except Exception as exc:
                    ack_ts = time.time()
                    last_error = str(exc)
                    if not self._is_transient_okx_error(okx_msg=str(exc)):
                        raise
                    existing = self._find_existing_order_by_client_id(
                        coin=coin,
                        side=side,
                        action=action,
                        amount_usdt=amount_usdt,
                        requested_qty=requested_qty,
                        submitted_sz=str(sz),
                        price=price,
                        submit_ts=submit_ts,
                        ack_ts=ack_ts,
                        td_mode=td_mode,
                        tgt_ccy=tgt_ccy,
                        cl_ord_id=_exchange_cl_ord_id(),
                        order_poll_delay_sec=0.0,
                        retries=2,
                    )
                    if existing:
                        return existing
                    if not client_order_supported:
                        return _submission_failure(
                            "submit_uncertain_without_client_order_id",
                            f"order submit timed out without exchange client id; not retrying to avoid duplicate: {exc}",
                        )
                    if attempt >= attempts - 1:
                        return _submission_failure("submit_retry_exhausted", str(exc), res)
                    fresh, current_price, drift_bps = self._retry_price_is_fresh(coin, price, max_drift_bps)
                    if not fresh:
                        return _submission_failure(
                            "stale_price",
                            (
                                f"retry stopped because {coin} drifted {drift_bps:.2f} bps "
                                f"from {price:.8g} to {current_price:.8g}"
                            ),
                            res,
                        )
                    sleep_sec = self._retry_sleep_sec(
                        self.ORDER_SUBMIT_RETRY_BASE_SLEEP_SEC,
                        attempt,
                        self.ORDER_SUBMIT_RETRY_MAX_SLEEP_SEC,
                    )
                    logger.warning(
                        "[%s] transient OKX order exception, retrying in %.1fs (%s/%s): %s",
                        self.identifier,
                        sleep_sec,
                        attempt + 1,
                        attempts,
                        exc,
                    )
                    if not self._sleep_before_retry(sleep_sec, deadline_ts):
                        return _submission_failure("submit_retry_expired", str(exc), res)
                    continue
                ack_ts = time.time()
                first_attempt = _first_okx_data(res)
                attempt_okx_s_code = str(first_attempt.get("sCode", "") or "")
                attempt_okx_s_msg = str(first_attempt.get("sMsg", "") or "")
                attempt_okx_code = str(res.get("code", "") if isinstance(res, dict) else "")
                attempt_okx_msg = str(res.get("msg", "") if isinstance(res, dict) else "")
                accepted = attempt_okx_code == "0" and attempt_okx_s_code in {"", "0"}
                if accepted:
                    result = self._make_trade_result(
                        response=res,
                        coin=coin,
                        side=side,
                        action=action,
                        amount_usdt=amount_usdt,
                        requested_qty=requested_qty,
                        submitted_sz=str(sz),
                        price=price,
                        submit_ts=submit_ts,
                        ack_ts=ack_ts,
                        td_mode=td_mode,
                        tgt_ccy=tgt_ccy,
                        cl_ord_id=_exchange_cl_ord_id(),
                    )
                    logger.info("[%s] order accepted: %s %s (~%.2f USDT)", self.identifier, side, coin, amount_usdt)
                    return self._finalize_execution_result(result, order_poll_delay_sec)

                if self._is_duplicate_client_order_response(res):
                    existing = self._find_existing_order_by_client_id(
                        coin=coin,
                        side=side,
                        action=action,
                        amount_usdt=amount_usdt,
                        requested_qty=requested_qty,
                        submitted_sz=str(sz),
                        price=price,
                        submit_ts=submit_ts,
                        ack_ts=ack_ts,
                        td_mode=td_mode,
                        tgt_ccy=tgt_ccy,
                        cl_ord_id=_exchange_cl_ord_id(),
                        order_poll_delay_sec=order_poll_delay_sec,
                        retries=3,
                    )
                    if existing:
                        return existing
                    return _submission_failure(
                        "duplicate_client_order_unresolved",
                        attempt_okx_s_msg or attempt_okx_msg or str(res),
                        res,
                    )

                transient_response = self._is_transient_okx_error(
                    attempt_okx_code,
                    attempt_okx_s_code,
                    attempt_okx_msg,
                    attempt_okx_s_msg,
                )
                if not transient_response:
                    result = self._make_trade_result(
                        response=res,
                        coin=coin,
                        side=side,
                        action=action,
                        amount_usdt=amount_usdt,
                        requested_qty=requested_qty,
                        submitted_sz=str(sz),
                        price=price,
                        submit_ts=submit_ts,
                        ack_ts=ack_ts,
                        td_mode=td_mode,
                        tgt_ccy=tgt_ccy,
                        cl_ord_id=_exchange_cl_ord_id(),
                    )
                    logger.warning("[%s] order rejected: %s %s %s", self.identifier, attempt_okx_msg, attempt_okx_s_code, attempt_okx_s_msg)
                    return result

                existing = self._find_existing_order_by_client_id(
                    coin=coin,
                    side=side,
                    action=action,
                    amount_usdt=amount_usdt,
                    requested_qty=requested_qty,
                    submitted_sz=str(sz),
                    price=price,
                    submit_ts=submit_ts,
                    ack_ts=ack_ts,
                    td_mode=td_mode,
                    tgt_ccy=tgt_ccy,
                    cl_ord_id=_exchange_cl_ord_id(),
                    order_poll_delay_sec=0.0,
                    retries=2,
                )
                if existing:
                    return existing
                if attempt >= attempts - 1:
                    return _submission_failure(
                        "submit_retry_exhausted",
                        attempt_okx_s_msg or attempt_okx_msg or str(res),
                        res,
                    )
                fresh, current_price, drift_bps = self._retry_price_is_fresh(coin, price, max_drift_bps)
                if not fresh:
                    return _submission_failure(
                        "stale_price",
                        (
                            f"retry stopped because {coin} drifted {drift_bps:.2f} bps "
                            f"from {price:.8g} to {current_price:.8g}"
                        ),
                        res,
                    )
                sleep_sec = self._retry_sleep_sec(
                    self.ORDER_SUBMIT_RETRY_BASE_SLEEP_SEC,
                    attempt,
                    self.ORDER_SUBMIT_RETRY_MAX_SLEEP_SEC,
                )
                logger.warning(
                    "[%s] transient OKX order error, retrying in %.1fs (%s/%s %s %s %s %s)",
                    self.identifier,
                    sleep_sec,
                    attempt + 1,
                    attempts,
                    attempt_okx_code,
                    attempt_okx_msg,
                    attempt_okx_s_code,
                    attempt_okx_s_msg,
                )
                if not self._sleep_before_retry(sleep_sec, deadline_ts):
                    return _submission_failure(
                        "submit_retry_expired",
                        attempt_okx_s_msg or attempt_okx_msg or last_error or str(res),
                        res,
                    )

            return _submission_failure(
                "submit_retry_exhausted",
                last_error or "order submit retry exhausted",
                res,
            )
        except Exception as exc:
            logger.warning("[%s] execute_trade failed: %s", self.identifier, exc)
            return TradeExecutionResult.failure(
                inst_id=coin,
                side=side,
                action=action,
                requested_notional_usdt=amount_usdt,
                reference_price=price,
                trade_mode=self.trade_mode,
                error_type="exception",
                error_message=str(exc),
            )

    def clear_positions(self, coin: str, price: float) -> float:
        if not HAS_OKX_LIB or not self.trade_api:
            return 0.0
        try:
            if self.trade_mode == "swap":
                res = self.trade_api.close_positions(instId=coin, mgnMode="cross")
            else:
                qty = self.get_holdings(force_refresh=True).get(coin, 0.0)
                if qty <= 0:
                    return 0.0
                res = self.trade_api.place_order(
                    instId=coin,
                    tdMode="cash",
                    side="sell",
                    ordType="market",
                    sz=_format_order_size(qty),
                    tgtCcy="base_ccy",
                    clOrdId=self._client_order_id(coin, "sell"),
                )
            if res.get("code") == "0":
                self._invalidate_account_cache()
                logger.info("[%s] cleared position: %s", self.identifier, coin)
                return float(price)
        except Exception as exc:
            logger.warning("[%s] clear_positions failed: %s", self.identifier, exc)
        return 0.0
