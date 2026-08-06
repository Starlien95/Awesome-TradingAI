class InvestorActionAdapter:
    """
    将 INVESTOR-BENCH 的 position/signal 转换成 okx_trade.execute_decision() 可以执行的格式。

    支持两种模式：

    1. spot：
       buy  -> 买入现货
       sell -> close，现货清仓，不做空
       hold -> 不操作

    2. swap：
       position > 0 / buy  -> SWAP 开多
       position < 0 / sell -> SWAP 开空
       position = 0 / hold -> hold，最终是否 close 由 run_one_live_cycle 的仓位控制决定

    阶段B真实做空主要由 run_one_live_cycle.py 根据当前本地状态生成最终 buy/sell/close/reverse 决策；
    本 adapter 保持轻量，只做基础方向兼容。
    """

    def __init__(
        self,
        trade_amount_usdt=500,
        min_confidence=None,
        allow_short=False,
        trade_mode="spot",
        td_mode="cash",
        leverage=1,
    ):
        self.trade_amount_usdt = trade_amount_usdt
        self.min_confidence = min_confidence
        self.allow_short = allow_short
        self.trade_mode = str(trade_mode).lower().strip()
        self.td_mode = td_mode
        self.leverage = leverage

    @staticmethod
    def normalize_symbol(symbol):
        symbol = str(symbol).upper().strip()
        if symbol.endswith("-USDT-SWAP"):
            return symbol.split("-")[0]
        if symbol.endswith("-USDT"):
            return symbol.split("-")[0]
        return symbol

    @staticmethod
    def normalize_signal(signal):
        signal = str(signal).lower().strip()
        if signal in {"buy", "long", "open_long"}:
            return "buy"
        if signal in {"sell", "short", "open_short"}:
            return "sell"
        if signal in {"hold", "neutral", "wait", "flat"}:
            return "hold"
        return "hold"

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            if isinstance(value, str) and value.strip() == "":
                return default
            return float(value)
        except Exception:
            return default

    def _extract_position(self, action):
        raw_action = action.get("raw_action", {}) or {}
        for key in ["target_position", "position"]:
            if key in action:
                return self._safe_float(action.get(key), 0.0)
        if "position" in raw_action:
            return self._safe_float(raw_action.get("position"), 0.0)
        return None

    def convert_single_action(self, action):
        symbol = self.normalize_symbol(action.get("symbol", "BTC"))
        signal = self.normalize_signal(action.get("signal", "hold"))
        position = self._extract_position(action)

        if position is not None:
            if position > 0:
                signal = "buy"
            elif position < 0:
                signal = "sell"
            else:
                signal = "hold"

        confidence = action.get("confidence", None)
        if self.min_confidence is not None and confidence is not None:
            if float(confidence) < self.min_confidence:
                signal = "hold"

        if self.trade_mode == "swap":
            inst_id = f"{symbol}-USDT-SWAP"
            base = {
                "trade_mode": "swap",
                "instId": inst_id,
                "tdMode": self.td_mode,
                "leverage": str(self.leverage),
                "notional_usdt": str(self.trade_amount_usdt),
                "quantity": str(self.trade_amount_usdt),
                "coin": symbol,
            }

            if signal == "buy":
                return {symbol: {**base, "signal": "buy", "target_state": "long", "target_sign": 1}}

            if signal == "sell":
                if self.allow_short:
                    return {symbol: {**base, "signal": "sell", "target_state": "short", "target_sign": -1}}
                return {symbol: {**base, "signal": "close", "target_state": "flat", "target_sign": 0}}

            return {symbol: {**base, "signal": "hold", "quantity": "0", "notional_usdt": "0", "target_state": "flat", "target_sign": 0}}

        # spot compatibility
        if signal == "buy":
            return {
                symbol: {
                    "signal": "buy",
                    "quantity": str(self.trade_amount_usdt),
                    "tgtCcy": "quote_ccy",
                    "coin": symbol,
                }
            }

        if signal == "sell":
            if self.allow_short:
                return {
                    symbol: {
                        "signal": "sell",
                        "quantity": str(self.trade_amount_usdt),
                        "tgtCcy": "quote_ccy",
                        "coin": symbol,
                    }
                }

            return {
                symbol: {
                    "signal": "close",
                    "quantity": "AUTO",
                    "tgtCcy": "base_ccy",
                    "coin": symbol,
                }
            }

        return {
            symbol: {
                "signal": "hold",
                "quantity": "0",
                "coin": symbol,
            }
        }
