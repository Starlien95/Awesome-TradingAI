import os
import time
import okx.Account as account
import okx.Trade as trade
from quant_bench.methods.finmem.live.env import get_env_bool, load_project_env
import okx.PublicData as PublicData
import okx.MarketData as MarketData

load_project_env()


class okxbot:
    def __init__(self, is_simu):
        if is_simu:
            self.api_key = os.getenv("OKX_API_KEY_SIMU")
            self.secret_key = os.getenv("OKX_SECRET_KEY_SIMU")
            self.passphrase = os.getenv("OKX_PASSPHRASE")
            self.flag = "1"
        else:
            self.api_key = os.getenv("OKX_API_KEY")
            self.secret_key = os.getenv("OKX_SECRET_KEY")
            self.passphrase = os.getenv("OKX_PASSPHRASE")
            self.flag = "0"
        self.orders_enabled = get_env_bool("FINMEM_EXECUTE_ORDER", False)
        self.order_confirmation = os.getenv("FINMEM_ORDER_CONFIRM", "")
        self.account = account.AccountAPI(self.api_key, self.secret_key, self.passphrase, False, self.flag)
        self.tradeapi = trade.TradeAPI(self.api_key, self.secret_key, self.passphrase, False, self.flag)
        self.publicDataAPI = PublicData.PublicAPI(flag=self.flag)
        self.marketDataAPI = MarketData.MarketAPI(flag=self.flag)

    def _require_order_permission(self):
        expected = "DEMO_ORDERS" if self.flag == "1" else "LIVE_ORDERS"
        if not self.orders_enabled or self.order_confirmation != expected:
            raise PermissionError(
                f"FinMem order submission is disabled; set FINMEM_EXECUTE_ORDER=1 "
                f"and FINMEM_ORDER_CONFIRM={expected}"
            )

    def normalize_inst_id(self, coin):
        coin = str(coin).upper().strip()

        if coin.endswith("-USDT-SWAP"):
            return coin

        if coin.endswith("-USDT"):
            return coin

        return f"{coin}-USDT"

    def normalize_swap_inst_id(self, coin):
        coin = str(coin).upper().strip()

        if coin.endswith("-USDT-SWAP"):
            return coin

        if coin.endswith("-USDT"):
            return f"{coin}-SWAP"

        return f"{coin}-USDT-SWAP"

    def close_spot_position(self, instId):
        """
        现货平仓：查询当前币种余额，然后市价卖出。
        注意：现货不能用 close_positions，那主要是合约/保证金仓位接口。
        """
        self._require_order_permission()
        symbol = instId.split("-")[0]

        balances = self.get_coin_num()
        quantity = float(balances.get(symbol, 0))

        if quantity <= 0:
            return {
                "status": "no_position",
                "instId": instId,
                "symbol": symbol,
                "quantity": 0.0,
            }

        return self.trade(
            instId=instId,
            sz=str(quantity),
            side="sell",
            ordType="market",
            tdMode="cash",
            tgtCcy="base_ccy",
        )

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            if isinstance(value, str):
                value = value.strip()
                if value == "" or value.upper() == "AUTO":
                    return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _floor_to_step(value, step):
        value = float(value)
        step = float(step)
        if step <= 0:
            return value
        return int(value / step) * step

    @staticmethod
    def _fmt_size(value):
        # OKX 接受字符串，去掉多余 0，避免科学计数法。
        s = f"{float(value):.12f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def get_swap_instrument(self, instId):
        """
        查询 SWAP 合约规格。用于把 USDT notional 转成 OKX 下单所需的合约张数 sz。
        """
        res = self.publicDataAPI.get_instruments(instType="SWAP", instId=instId)
        if str(res.get("code")) != "0" or not res.get("data"):
            raise RuntimeError(f"get_instruments failed for {instId}: {res}")
        return res["data"][0]

    def get_last_price_float(self, instId):
        ticker = self.get_price(instId)
        for key in ["last", "markPx", "idxPx", "askPx", "bidPx"]:
            if key in ticker:
                px = self._safe_float(ticker.get(key), 0.0)
                if px > 0:
                    return px
        raise RuntimeError(f"Cannot parse price from ticker: {ticker}")


    def get_balance_detail(self, ccy="USDT"):
        """
        Return raw account balance detail for one currency plus totalEq.
        This is used by the live logger to record real OKX simulated-account changes.
        """
        result = self.account.get_account_balance()
        out = {
            "ok": False,
            "raw": result,
            "ccy": ccy,
            "total_eq": None,
            "avail_bal": None,
            "frozen_bal": None,
            "cash_bal": None,
        }
        if str(result.get("code")) != "0" or not result.get("data"):
            out["error"] = result.get("msg", "get_account_balance failed")
            return out

        data0 = result["data"][0]
        out["ok"] = True
        out["total_eq"] = data0.get("totalEq")

        for detail in data0.get("details", []):
            if str(detail.get("ccy", "")).upper() == str(ccy).upper():
                out["avail_bal"] = detail.get("availBal")
                out["frozen_bal"] = detail.get("frozenBal", "0")
                out["cash_bal"] = detail.get("cashBal", detail.get("availBal"))
                out["detail"] = detail
                break
        return out

    def get_swap_position_raw(self, instId):
        """
        Return raw active SWAP position for instId. In net mode, short position has pos < 0.
        """
        result = self.account.get_positions(instType="SWAP", instId=instId)
        out = {"ok": False, "instId": instId, "raw": result, "position": None}
        if str(result.get("code")) != "0":
            out["error"] = result.get("msg", "get_positions failed")
            return out

        for pos in result.get("data", []):
            if str(pos.get("instId", "")).upper() != str(instId).upper():
                continue
            if abs(self._safe_float(pos.get("pos"), 0.0)) > 0:
                out["ok"] = True
                out["position"] = pos
                return out

        out["ok"] = True
        out["position"] = None
        return out

    def get_order_detail_safe(self, instId, ordId=None, clOrdId=None):
        """
        Best-effort order detail query. Different okx SDK versions expose slightly
        different method names, so this function never raises.
        """
        if not ordId and not clOrdId:
            return {"ok": False, "reason": "missing ordId/clOrdId"}

        params = {"instId": instId}
        if ordId:
            params["ordId"] = str(ordId)
        if clOrdId:
            params["clOrdId"] = str(clOrdId)

        for method_name in ["get_order", "get_order_details", "get_orders"]:
            fn = getattr(self.tradeapi, method_name, None)
            if fn is None:
                continue
            try:
                res = fn(**params)
                return {"ok": str(res.get("code")) == "0", "method": method_name, "raw": res}
            except Exception as e:
                last_error = repr(e)

        return {"ok": False, "reason": "no compatible order detail method", "last_error": locals().get("last_error", "")}

    def get_account_snapshot(self, instId=None, ccy="USDT"):
        """
        Snapshot real OKX simulated-account state. This is intentionally separate
        from the local strategy ledger.
        """
        snapshot = {
            "timestamp_ms": int(time.time() * 1000),
            "ccy": ccy,
            "balance": self.get_balance_detail(ccy=ccy),
        }
        if instId:
            try:
                snapshot["ticker"] = self.get_price(instId)
            except Exception as e:
                snapshot["ticker_error"] = repr(e)
            snapshot["position"] = self.get_swap_position_raw(instId)
        return snapshot

    def _extract_first_order_id(self, res):
        if not isinstance(res, dict):
            return ""
        data = res.get("data", [])
        if isinstance(data, list) and data:
            return str(data[0].get("ordId") or "")
        return ""

    def enrich_execution_result(self, res, instId, symbol, pre_snapshot=None, size_info=None, signal=None):
        """
        Attach real account/position snapshots to an OKX order response.
        This makes the cycle_record able to audit real execution changes.
        """
        if not isinstance(res, dict):
            res = {"status": "non_dict_response", "raw_response": res}

        post_snapshot = self.get_account_snapshot(instId=instId, ccy="USDT")
        ord_id = self._extract_first_order_id(res)
        order_detail = self.get_order_detail_safe(instId=instId, ordId=ord_id) if ord_id else {"ok": False, "reason": "no ordId"}

        position = (post_snapshot.get("position") or {}).get("position") or {}
        pos = self._safe_float(position.get("pos"), 0.0) if isinstance(position, dict) else 0.0
        avg_px = self._safe_float(position.get("avgPx"), 0.0) if isinstance(position, dict) else 0.0
        mark_px = self._safe_float(position.get("markPx", position.get("last")), 0.0) if isinstance(position, dict) else 0.0
        notional_usd = self._safe_float(position.get("notionalUsd"), 0.0) if isinstance(position, dict) else 0.0
        upl = self._safe_float(position.get("upl"), 0.0) if isinstance(position, dict) else 0.0
        realized_pnl = self._safe_float(position.get("realizedPnl"), 0.0) if isinstance(position, dict) else 0.0
        fee = self._safe_float(position.get("fee"), 0.0) if isinstance(position, dict) else 0.0

        res.setdefault("trade_mode", "swap")
        res.setdefault("instId", instId)
        res.setdefault("symbol", symbol)
        if size_info is not None:
            res.setdefault("size_info", size_info)
        if signal is not None:
            res.setdefault("signal", signal)

        res["execution_audit"] = {
            "pre_snapshot": pre_snapshot or {},
            "post_snapshot": post_snapshot,
            "order_detail": order_detail,
            "actual_position": position,
            "actual_pos": pos,
            "actual_avg_px": avg_px,
            "actual_mark_px": mark_px,
            "actual_notional_usd": notional_usd,
            "actual_upl": upl,
            "actual_realized_pnl": realized_pnl,
            "actual_fee": fee,
        }
        return res

    def calc_swap_size_from_notional(self, instId, notional_usdt, price=None):
        """
        将 USDT 名义本金转换成 OKX SWAP 市价单 sz。

        对 USDT 本位线性合约，OKX 的 sz 通常是合约张数，
        合约面值 ctVal 的币种由 ctValCcy 决定，例如 BTC-USDT-SWAP 常见 ctValCcy=BTC。
        """
        notional = self._safe_float(notional_usdt, 0.0)
        if notional <= 0:
            raise ValueError(f"notional_usdt must be positive, got {notional_usdt}")

        if price is None or self._safe_float(price, 0.0) <= 0:
            price = self.get_last_price_float(instId)
        price = self._safe_float(price, 0.0)

        inst = self.get_swap_instrument(instId)
        ct_val = self._safe_float(inst.get("ctVal"), 0.0)
        ct_mult = self._safe_float(inst.get("ctMult"), 1.0) or 1.0
        lot_sz = self._safe_float(inst.get("lotSz"), 1.0) or 1.0
        min_sz = self._safe_float(inst.get("minSz"), lot_sz) or lot_sz
        ct_val_ccy = str(inst.get("ctValCcy", "")).upper()
        base_ccy = str(inst.get("baseCcy", instId.split("-")[0])).upper()
        quote_ccy = str(inst.get("quoteCcy", "USDT")).upper()

        # 常见 USDT 永续：ctValCcy=BTC/ETH，1 张合约对应 ctVal 个基础币。
        if ct_val_ccy == base_ccy:
            contract_notional_usdt = ct_val * ct_mult * price
        # 如果合约面值已经是 USDT/USD，则不再乘价格。
        elif ct_val_ccy in {quote_ccy, "USDT", "USD"}:
            contract_notional_usdt = ct_val * ct_mult
        else:
            # 保守 fallback：按基础币面值处理。
            contract_notional_usdt = ct_val * ct_mult * price

        if contract_notional_usdt <= 0:
            raise RuntimeError(
                f"Invalid contract_notional_usdt={contract_notional_usdt}, inst={inst}, price={price}"
            )

        raw_sz = notional / contract_notional_usdt
        sz = self._floor_to_step(raw_sz, lot_sz)
        if sz < min_sz:
            sz = min_sz

        return self._fmt_size(sz), {
            "instId": instId,
            "price": price,
            "notional_usdt": notional,
            "ctVal": ct_val,
            "ctMult": ct_mult,
            "ctValCcy": ct_val_ccy,
            "lotSz": lot_sz,
            "minSz": min_sz,
            "contract_notional_usdt": contract_notional_usdt,
            "raw_sz": raw_sz,
            "sz": sz,
        }

    def close_swap_position(self, instId, mgnMode="cross", posSide=None):
        """
        合约/永续平仓。适用于阶段B真实 OKX simulated SWAP 做空/做多。
        """
        self._require_order_permission()
        params = {
            "instId": instId,
            "mgnMode": mgnMode,
        }
        if posSide:
            params["posSide"] = posSide

        res = self.tradeapi.close_positions(**params)
        print(res)
        return res

    def execute_swap_trade(self, symbol, instId, trade_info):
        """
        执行 SWAP 市价交易，并记录真实 OKX 模拟盘账户变化。

        返回结果会包含 execution_audit:
          - pre_snapshot: 下单前账户/持仓/ticker
          - post_snapshot: 下单后账户/持仓/ticker
          - order_detail: best-effort 订单详情
          - actual_position / actual_avg_px / actual_notional_usd / actual_fee 等
        """
        signal = str(trade_info.get("signal", "hold")).lower().strip()
        td_mode = str(trade_info.get("tdMode", trade_info.get("td_mode", "cross"))).lower().strip()
        pos_side = trade_info.get("posSide") or trade_info.get("pos_side")
        leverage = trade_info.get("leverage") or trade_info.get("lever")
        reduce_only = trade_info.get("reduceOnly", trade_info.get("reduce_only", None))
        price_hint = self._safe_float(trade_info.get("price"), 0.0)

        pre_snapshot = self.get_account_snapshot(instId=instId, ccy="USDT")

        if leverage:
            try:
                self.set_leverage(instId=instId, lever=str(leverage), mgnMode=td_mode, posSide=pos_side)
            except Exception as e:
                print(f"[WARNING] set_leverage failed but continue: {e}")

        if signal in {"hold", "wait", "neutral"}:
            res = {
                "status": "waiting",
                "instId": instId,
                "symbol": symbol,
                "trade_mode": "swap",
                "signal": signal,
            }
            return self.enrich_execution_result(res, instId, symbol, pre_snapshot=pre_snapshot, signal=signal)

        if signal in {"close", "close_long", "close_short"}:
            res = self.close_swap_position(instId=instId, mgnMode=td_mode, posSide=pos_side)
            return self.enrich_execution_result(res, instId, symbol, pre_snapshot=pre_snapshot, signal=signal)

        if signal in {"reverse_to_long", "reverse_to_short"}:
            close_res = self.close_swap_position(instId=instId, mgnMode=td_mode, posSide=pos_side)
            # 等待平仓状态同步，避免立即查询还是旧仓位。
            time.sleep(0.8)
            mid_snapshot = self.get_account_snapshot(instId=instId, ccy="USDT")

            open_side = "buy" if signal == "reverse_to_long" else "sell"
            open_pos_side = pos_side
            if pos_side in {"long", "short"}:
                open_pos_side = "long" if open_side == "buy" else "short"

            notional = (
                trade_info.get("open_notional_usdt")
                or trade_info.get("notional_usdt")
                or trade_info.get("quantity")
            )
            sz = trade_info.get("sz")
            size_info = {}
            if not sz:
                sz, size_info = self.calc_swap_size_from_notional(instId, notional, price=price_hint)

            open_res = self.trade(
                instId=instId,
                sz=str(sz),
                side=open_side,
                ordType="market",
                tdMode=td_mode,
                posSide=open_pos_side,
                reduceOnly=False,
            )

            ok = str(close_res.get("code", "")) == "0" and str(open_res.get("code", "")) == "0"
            res = {
                "code": "0" if ok else "1",
                "msg": "reverse executed" if ok else "reverse partially failed",
                "data": [{
                    "ordId": self._extract_first_order_id(open_res) or f"reverse_{instId}_{signal}",
                    "sCode": "0" if ok else "1",
                    "sMsg": "close then open",
                }],
                "instId": instId,
                "symbol": symbol,
                "trade_mode": "swap",
                "signal": signal,
                "close_result": close_res,
                "open_result": open_res,
                "mid_snapshot_after_close": mid_snapshot,
                "size_info": size_info,
            }
            return self.enrich_execution_result(res, instId, symbol, pre_snapshot=pre_snapshot, size_info=size_info, signal=signal)

        if signal in {"buy", "open_long", "long"}:
            side = "buy"
            if pos_side in {"long", "short"}:
                pos_side = "long"
        elif signal in {"sell", "open_short", "short"}:
            side = "sell"
            if pos_side in {"long", "short"}:
                pos_side = "short"
        else:
            res = {
                "status": "invalid_signal",
                "instId": instId,
                "symbol": symbol,
                "signal": signal,
                "trade_mode": "swap",
            }
            return self.enrich_execution_result(res, instId, symbol, pre_snapshot=pre_snapshot, signal=signal)

        notional = trade_info.get("notional_usdt") or trade_info.get("quantity")
        sz = trade_info.get("sz")
        size_info = {}
        if not sz:
            sz, size_info = self.calc_swap_size_from_notional(instId, notional, price=price_hint)

        res = self.trade(
            instId=instId,
            sz=str(sz),
            side=side,
            ordType="market",
            tdMode=td_mode,
            posSide=pos_side,
            reduceOnly=reduce_only,
        )

        return self.enrich_execution_result(res, instId, symbol, pre_snapshot=pre_snapshot, size_info=size_info, signal=signal)

    def execute_decision(self, decision):
        """
        执行交易决策。

        现货格式保持兼容：
          buy / sell / close / hold

        阶段B SWAP 格式：
        {
            "BTC": {
                "trade_mode": "swap",
                "instId": "BTC-USDT-SWAP",
                "signal": "sell",              # 开空
                "notional_usdt": "10000",
                "tdMode": "cross",
                "leverage": "1"
            }
        }
        """

        print(decision)

        if not isinstance(decision, dict):
            return {
                "status": "invalid_decision",
                "reason": "decision must be a dict",
                "raw_decision": decision,
            }

        results = {}

        for coin, trade_info in decision.items():
            if not isinstance(trade_info, dict):
                results[coin] = {
                    "status": "invalid_trade_info",
                    "raw_trade_info": trade_info,
                }
                continue

            trade_mode = str(trade_info.get("trade_mode", "spot")).lower().strip()
            raw_inst_id = trade_info.get("instId") or trade_info.get("inst_id")

            if trade_mode == "swap" or (raw_inst_id and str(raw_inst_id).upper().endswith("-SWAP")):
                instId = str(raw_inst_id).upper().strip() if raw_inst_id else self.normalize_swap_inst_id(coin)
            else:
                instId = str(raw_inst_id).upper().strip() if raw_inst_id else self.normalize_inst_id(coin)

            symbol = instId.split("-")[0]
            signal = str(trade_info.get("signal", "hold")).lower().strip()

            if signal == "hole":
                signal = "hold"

            try:
                if trade_mode == "swap" or instId.endswith("-SWAP"):
                    results[symbol] = self.execute_swap_trade(symbol, instId, trade_info)
                    continue

                if signal == "buy":
                    quantity = trade_info.get("quantity", "0")
                    tgtCcy = trade_info.get("tgtCcy", "quote_ccy")

                    quantity_float = float(quantity)
                    if quantity_float <= 0:
                        results[symbol] = {
                            "status": "skip",
                            "reason": "buy quantity <= 0",
                            "instId": instId,
                            "quantity": quantity,
                        }
                        continue

                    results[symbol] = self.trade(
                        instId=instId,
                        sz=str(quantity),
                        side="buy",
                        ordType="market",
                        tdMode="cash",
                        tgtCcy=tgtCcy,
                    )

                elif signal == "sell":
                    quantity = trade_info.get("quantity", "AUTO")

                    if str(quantity).upper() == "AUTO":
                        results[symbol] = self.close_spot_position(instId)
                        continue

                    quantity_float = float(quantity)
                    if quantity_float <= 0:
                        results[symbol] = {
                            "status": "skip",
                            "reason": "sell quantity <= 0",
                            "instId": instId,
                            "quantity": quantity,
                        }
                        continue

                    results[symbol] = self.trade(
                        instId=instId,
                        sz=str(quantity),
                        side="sell",
                        ordType="market",
                        tdMode="cash",
                        tgtCcy="base_ccy",
                    )

                elif signal == "close":
                    results[symbol] = self.close_spot_position(instId)

                elif signal == "hold":
                    results[symbol] = {
                        "status": "waiting",
                        "instId": instId,
                        "symbol": symbol,
                    }

                else:
                    results[symbol] = {
                        "status": "invalid_signal",
                        "instId": instId,
                        "symbol": symbol,
                        "signal": signal,
                    }

            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "instId": instId,
                    "symbol": symbol,
                    "signal": signal,
                    "trade_mode": trade_mode,
                    "error": str(e),
                }

        return results

    def get_price(self, instId):
        result = self.marketDataAPI.get_ticker(
            instId=instId
        )["data"][0]
        return result

    def get_balance(self, ccy="USDT"):
        """
        获取账户余额
        Args:
            ccy: 币种，默认USDT
        Returns:
            dict: 余额信息
        """
        result = self.account.get_account_balance()

        if result['code'] == '0':
            balance_data = result['data'][0]

            print(f"总权益: {balance_data['totalEq']} USDT")

            if ccy:
                for detail in balance_data['details']:
                    if detail['ccy'] == ccy:
                        print(f"{ccy}余额: 可用={detail['availBal']}, 冻结={detail.get('frozenBal', '0')}")
                        return {
                            'total_eq': balance_data['totalEq'],
                            'available': detail['availBal'],
                            'frozen': detail.get('frozenBal', '0')
                        }
            return {'total_eq': balance_data['totalEq']}
        else:
            print(f"获取余额失败: {result['msg']}")
            return None

    def get_position(self, inst_type="SWAP", inst_id=None):
        """
        获取持仓信息
        Args:
            inst_type: 产品类型
            inst_id: 指定交易对
        Returns:
            list: 持仓列表
        """
        result = self.account.get_positions(instType=inst_type)

        if result['code'] == '0':
            positions = result['data']
            active_positions = []

            for pos in positions:
                print(pos)
                pos_qty = float(pos.get('pos', 0))
                if pos_qty != 0:
                    position_info = {
                        'instId': pos['instId'],
                        'pos': pos['pos'],
                        'posSide': pos['posSide'],
                        'lever': pos['lever'],
                        'avgPx': pos['avgPx'],
                        'markPx': pos.get('markPx', pos.get('last', 'N/A')),
                        'upl': pos['upl'],
                        'uplRatio': pos.get('uplRatio', '0')
                    }
                    active_positions.append(position_info)

                    # 简洁输出
                    print(
                        f"{pos['instId']} {pos['posSide']} {pos['pos']} 杠杆{pos['lever']}x 盈亏{float(pos['upl']):.2f}USDT")

            if not active_positions:
                print("无持仓")

            return active_positions
        else:
            print(f"获取持仓失败: {result['msg']}")
            return None

    def trade(self, instId, sz, side, px=None, ordType="market", tdMode="cash", tgtCcy=None, posSide=None, reduceOnly=None):
        """
        通用下单封装。

        spot:
          tdMode="cash", tgtCcy="quote_ccy" 或 "base_ccy"

        swap:
          tdMode="cross" / "isolated", 可选 posSide / reduceOnly
        """
        self._require_order_permission()
        if ordType == "limit" and px is None:
            return "选了限价单必须填写价格"
        if side != "buy" and side != "sell":
            return "side参数错误，必须是买或者卖"

        params = {
            "instId": instId,
            "side": side,
            "tdMode": tdMode,
            "sz": str(sz),
            "ordType": ordType,
        }

        if px is not None:
            params["px"] = px
        if tgtCcy is not None:
            params["tgtCcy"] = tgtCcy
        if posSide is not None:
            params["posSide"] = posSide
        if reduceOnly is not None:
            params["reduceOnly"] = str(reduceOnly).lower() if isinstance(reduceOnly, bool) else str(reduceOnly)

        res = self.tradeapi.place_order(**params)
        print(res)
        return res

    def set_leverage(self, instId, lever, mgnMode, posSide=None):
        """
        设置杠杆倍数
        Args:
            instId: 交易对，如 "BTC-USDT-SWAP"
            lever: 杠杆倍数，如 "10"
            mgnMode: 保证金模式 "cross"全仓 / "isolated"逐仓
            posSide: 持仓方向 (仅在开平仓模式下需要)
                    "long" 多 / "short" 空 / "net" 净持仓
        """
        self._require_order_permission()
        params = {
            "instId": instId,
            "lever": lever,
            "mgnMode": mgnMode
        }

        # 只有特定模式才需要posSide
        if posSide:
            params["posSide"] = posSide

        result = self.account.set_leverage(**params)

        if result['code'] == '0':
            print(f"设置杠杆成功: {instId} {lever}倍 {mgnMode}模式")
            return result
        else:
            print(f"设置杠杆失败: {result['msg']}")
            return result

    def get_coin_kline(self,instId,bar,limit=100):
        """
        获取K线数据
        Args:
            instId: 交易对
            bar: K线粒度
            limit: K线数量
        Returns:
            list: K线数据列表
        """
        return self.marketDataAPI.get_candlesticks(instId=instId,bar=bar,limit=limit)

    def get_history_kline(self, instId, bar, limit=100, after=None, before=None):
        params = {
            "instId": instId,
            "bar": bar,
            "limit": str(limit),
        }

        if after is not None:
            params["after"] = str(after)

        if before is not None:
            params["before"] = str(before)

        return self.marketDataAPI.get_history_candlesticks(**params)

    def get_coin_num(self):
        """
        获取所有币种的数量
        Returns:
            dict: {币种: 数量}
        """
        result = self.account.get_account_balance()

        if result['code'] != '0':
            print(f"获取余额失败: {result['msg']}")
            return {}

        balance_data = result['data'][0]
        coin_dict = {}

        for detail in balance_data['details']:
            ccy = detail['ccy']
            avail_bal = float(detail['availBal'])
            frozen_bal = float(detail.get('frozenBal', 0))
            total_bal = avail_bal + frozen_bal

            # 只保留有余额的币种
            if total_bal > 0:
                coin_dict[ccy] = total_bal

        # 简洁输出
        print("币种余额:")
        for ccy, amount in coin_dict.items():
            print(f"  {ccy}: {amount}")

        return coin_dict
