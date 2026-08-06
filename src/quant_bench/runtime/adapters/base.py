import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quant_bench.runtime.core.config import ConfigManager
from quant_bench.strategies import select_target_weights

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """
    Base adapter for K-line driven strategies.

    The default signal generation uses the shared
    `ThresholdTopkDropoutStrategy` target-selection semantics, then converts target
    weights to executable USDT trading signals. By default, execution is less
    active than a strict target-weight rebalance: retained target holdings are
    not adjusted solely to restore equal weights.
    """

    @abstractmethod
    def load_model(self, model_path: str):
        """Load model weights."""
        pass

    @abstractmethod
    def predict(
        self, shared_data: dict[str, pd.DataFrame], config: ConfigManager
    ) -> list[dict[str, Any]]:
        """Return per-symbol model scores."""
        pass

    @staticmethod
    def _coerce_scores(predictions: list[dict[str, Any]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for pred in predictions:
            coin = pred.get("coin")
            if not coin:
                continue
            try:
                scores[str(coin)] = float(pred.get("score", float("nan")))
            except (TypeError, ValueError):
                continue
        return {coin: score for coin, score in scores.items() if pd.notna(score)}

    @staticmethod
    def _holding_values(
        current_positions: dict[str, float],
        current_prices: dict[str, float],
        min_value_usdt: float,
        allowed_coins: set[str] | None = None,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for coin, qty in current_positions.items():
            if allowed_coins is not None and coin not in allowed_coins:
                continue
            price = float(current_prices.get(coin, 0.0) or 0.0)
            value = float(qty) * price
            if qty > 0 and price > 0 and value >= min_value_usdt:
                values[coin] = value
        return values

    def _custom_strategy_target_weights(
        self,
        scores: dict[str, float],
        current_weight_dict: dict[str, float],
        config: ConfigManager,
    ) -> dict[str, float]:
        """
        Apply the canonical top-k/dropout target-weight selection.

        Differences from qlib are intentional and execution-related only:
        - short target weights are returned but the spot executor later skips
          actual short opening unless a subclass/executor supports it.
        - missing scores for current holdings are treated as 0.0, matching
          pandas Series.get(inst, 0) usage in the custom strategy.
        """
        top_k = int(config.top_k)
        if top_k <= 0 or not scores:
            return {}
        max_dropout = config.max_dropout
        max_dropout_value = len(current_weight_dict) if max_dropout is None else max(0, int(max_dropout))
        return select_target_weights(
            pd.Series(scores, dtype=float),
            current_weight_dict,
            top_k=top_k,
            threshold=float(config.threshold),
            max_dropout=max_dropout_value,
            enable_short=bool(getattr(config, "enable_short", False)),
        )

    @staticmethod
    def _effective_target_values(
        target_weights: dict[str, float],
        current_values: dict[str, float],
        investable_equity: float,
        rebalance_policy: str,
        abs_tolerance: float,
    ) -> dict[str, float]:
        target_values = {
            coin: max(0.0, weight) * investable_equity
            for coin, weight in target_weights.items()
            if weight > 0
        }
        if rebalance_policy == "target_weight":
            return target_values

        # qlib built-in TopkDropoutStrategy only trades the dropout/new names
        # instead of refreshing retained positions every cycle. Keep existing
        # target holdings unchanged, then distribute the remaining risk budget
        # across newly selected names. This caps new buying without actively
        # trimming retained winners just because their equal weight drifted.
        retained = {coin: current_values[coin] for coin in target_values if coin in current_values}
        new_targets = [coin for coin in target_values if coin not in retained]
        out: dict[str, float] = dict(retained)
        if not new_targets:
            return out

        remaining_budget = max(0.0, investable_equity - sum(retained.values()))
        if remaining_budget < abs_tolerance:
            return out

        weight_sum = sum(target_weights.get(coin, 0.0) for coin in new_targets if target_weights.get(coin, 0.0) > 0)
        for coin in new_targets:
            weight = target_weights.get(coin, 0.0)
            if weight <= 0:
                continue
            out[coin] = remaining_budget * (weight / weight_sum) if weight_sum > 0 else 0.0
        return out

    @staticmethod
    def _sort_by_score(coins: set[str], scores: dict[str, float], reverse: bool = False) -> list[str]:
        return sorted(coins, key=lambda c: (scores.get(c, 0.0), c), reverse=reverse)

    def generate_signals(
        self,
        predictions: list[dict[str, Any]],
        current_positions: dict[str, float],
        current_prices: dict[str, float],
        config: ConfigManager,
        budget_snapshot: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate executable trading actions.

        Selection is aligned with qlib custom `ThresholdTopkDropoutStrategy`:
        score filter -> topk target -> max_dropout sell candidates -> retained
        holdings -> topk cap -> equal target weights.

        To reduce meaningless churn in live/paper trading, the execution layer
        defaults to `passive_topk_dropout`: keep retained target holdings
        unchanged, sell dropout positions, and buy newly selected targets. It
        also filters deltas below:
        `max(min_position_value_usdt, rebalance_tolerance_usdt,
        strategy_equity * rebalance_tolerance_pct)`.
        """
        scores = self._coerce_scores(predictions)
        min_val = float(config.min_position_value_usdt)
        risk_degree = float(getattr(config, "risk_degree", 0.95))
        risk_degree = min(max(risk_degree, 0.0), 1.0)

        if budget_snapshot:
            strategy_equity = max(0.0, float(budget_snapshot.get("strategy_equity", 0.0)))
            cash_available = max(
                0.0,
                float(budget_snapshot.get("cash_available_for_strategy", 0.0)),
            )
        else:
            strategy_equity = max(0.0, float(config.trade_amount_usdt) * max(1, int(config.top_k)))
            cash_available = strategy_equity

        trade_coins = set(getattr(config, "coins", []) or [])
        if bool(getattr(config, "enforce_trade_coins_whitelist", False)):
            # Dedicated strategy account mode: only assets in config.coins are
            # controlled by this strategy; any whitelisted holding without a
            # fresh score is assigned score 0.0 below.
            allowed_coins: set[str] | None = trade_coins or set(scores.keys())
        else:
            # Shared/manual account safe mode: only holdings that received a
            # model score this cycle can be modified.
            allowed_coins = set(scores.keys())

        current_values = self._holding_values(
            current_positions,
            current_prices,
            min_val,
            allowed_coins=allowed_coins,
        )
        current_stock_value = sum(current_values.values())
        total_for_weight = max(strategy_equity, current_stock_value)
        current_weights = (
            {coin: value / total_for_weight for coin, value in current_values.items()}
            if total_for_weight > 0
            else {}
        )

        for coin in current_weights:
            scores.setdefault(coin, 0.0)

        abs_tolerance = max(
            min_val,
            float(getattr(config, "rebalance_tolerance_usdt", min_val)),
            strategy_equity * float(getattr(config, "rebalance_tolerance_pct", 0.0)),
        )
        target_weights = self._custom_strategy_target_weights(scores, current_weights, config)
        investable_equity = strategy_equity * risk_degree
        rebalance_policy = getattr(config, "rebalance_policy", "passive_topk_dropout")
        target_values = self._effective_target_values(
            target_weights=target_weights,
            current_values=current_values,
            investable_equity=investable_equity,
            rebalance_policy=rebalance_policy,
            abs_tolerance=abs_tolerance,
        )

        signals: list[dict[str, Any]] = []
        sell_total = 0.0
        trade_universe = set(current_values) | set(target_values)

        for coin in self._sort_by_score(trade_universe, scores, reverse=False):
            current_value = current_values.get(coin, 0.0)
            target_value = target_values.get(coin, 0.0)
            delta = current_value - target_value
            if delta >= abs_tolerance:
                reason = "dropout_or_rank_trim" if target_value <= 0 else "target_weight_rebalance"
                price = float(current_prices.get(coin, 0.0) or 0.0)
                max_sell_qty = delta / price if price > 0 else 0.0
                signals.append(
                    {
                        "coin": coin,
                        "side": "sell",
                        "amount_usdt": delta,
                        "max_sell_qty": max_sell_qty,
                        "reason": reason,
                        "target_value_usdt": target_value,
                        "target_weight": target_weights.get(coin, 0.0),
                    }
                )
                sell_total += delta

        buy_budget = cash_available + sell_total
        for coin in self._sort_by_score(set(target_values), scores, reverse=True):
            current_value = current_values.get(coin, 0.0)
            target_value = target_values[coin]
            need = target_value - current_value
            if need < abs_tolerance:
                continue
            amount = min(need, buy_budget)
            if amount < min_val:
                continue
            signals.append(
                {
                    "coin": coin,
                    "side": "buy",
                    "amount_usdt": amount,
                    "reason": "target_weight_rebalance",
                    "target_value_usdt": target_value,
                    "target_weight": target_weights.get(coin, 0.0),
                }
            )
            buy_budget -= amount
            if buy_budget < min_val:
                break

        logger.info(
            f"[{self.__class__.__name__}] target_weights={target_weights}, "
            f"equity={strategy_equity:.2f}U, risk_degree={risk_degree:.2f}, "
            f"rebalance_policy={rebalance_policy}, tolerance={abs_tolerance:.2f}U, "
            f"signals={len(signals)}"
        )
        return signals
