import json
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    配置管理器：负责加载和解析 YAML/JSON 格式的模型及策略配置。
    支持运行时热加载（reload）。
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def reload(self):
        """运行时重新从磁盘加载配置文件，用于热更新交易参数"""
        try:
            new_config = self._load_config()
            self._log_config_diff(self.config, new_config)
            self.config = new_config
        except Exception:
            logger.exception("配置热更新失败，保持旧配置")

    def _log_config_diff(self, old: dict, new: dict, prefix: str = ""):
        """递归比较新旧配置，输出变更项"""
        all_keys = set(list(old.keys()) + list(new.keys()))
        for key in sorted(all_keys):
            full_key = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                self._log_config_diff(old_val, new_val, full_key)
            elif old_val != new_val:
                logger.info(f"🔄 配置变更: {full_key}: {old_val} → {new_val}")

    def _load_config(self) -> dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"⚠️ 找不到配置文件: {self.config_path}")

        config_path_lower = self.config_path.lower()
        with open(self.config_path, encoding='utf-8') as f:
            if config_path_lower.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(f) or {}
            elif config_path_lower.endswith('.json'):
                data = json.load(f) or {}
            else:
                raise ValueError("⚠️ 仅支持 .yaml, .yml 或 .json 格式的配置文件")

        if not isinstance(data, dict):
            raise ValueError(f"⚠️ 配置文件顶层必须是对象/字典: {self.config_path}")
        return data

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def model_path(self) -> str:
        return self.get("model.path", "")

    @property
    def coins(self) -> list:
        return self.get("trade.coins", [])

    @property
    def trade_mode(self) -> str:
        return self.get("trade.mode", "spot")

    @property
    def top_k(self) -> int:
        return self.get("strategy.top_k", 2)

    @property
    def threshold(self) -> float:
        return self.get("strategy.threshold", 0.1)

    @property
    def stop_loss(self) -> float:
        return self.get("strategy.stop_loss", 0.05)

    @property
    def trade_amount_usdt(self) -> float:
        return self.get("strategy.trade_amount_usdt", 500.0)

    @property
    def initial_capital_usdt(self) -> float | None:
        """
        策略启动资金池。

        仅这部分本金及其后续盈亏参与收益率统计和下单额度约束。
        未配置时保持旧行为：以启动时账户总权益作为统计本金。
        """
        val = self.get(
            "strategy.initial_capital_usdt",
            self.get("strategy.startup_capital_usdt")
        )
        if val in (None, ""):
            return None
        try:
            capital = float(val)
        except (TypeError, ValueError):
            logger.warning(f"initial_capital_usdt 配置无效: {val}, 将按账户权益初始化")
            return None
        return capital if capital > 0 else None

    @property
    def risk_degree(self) -> float:
        """策略最多使用资金池权益的比例，1.0 表示满仓使用该资金池。"""
        try:
            risk = float(self.get("strategy.risk_degree", 0.95))
        except (TypeError, ValueError):
            return 0.95
        return min(max(risk, 0.0), 1.0)

    @property
    def capital_allocation_mode(self) -> str:
        """
        资金分配模式。

        目标总仓位始终受 risk_degree 约束，保留该字段是为了兼容旧配置。
        """
        mode = str(
            self.get("strategy.capital_allocation_mode", "qlib_risk_degree")
        ).strip().lower()
        if mode in {"full", "full_investment", "fully_invested"}:
            return "full_investment"
        return "qlib_risk_degree"

    @property
    def rebalance_policy(self) -> str:
        """
        live/paper 执行层的再平衡方式。

        passive_topk_dropout: 默认。选币集合对齐 custom_strategy，但仓位执行靠近
        qlib 内置 TopkDropout：卖出 dropout，买入新增目标，保留仓位不主动调到等权。
        target_weight: 严格按 custom_strategy 目标权重调仓，适合需要完全等权复现时使用。
        """
        value = str(
            self.get("strategy.rebalance_policy", "passive_topk_dropout")
        ).strip().lower()
        aliases = {
            "passive": "passive_topk_dropout",
            "dropout_only": "passive_topk_dropout",
            "topk_dropout": "passive_topk_dropout",
            "qlib_builtin_like": "passive_topk_dropout",
            "target": "target_weight",
            "equal_weight": "target_weight",
            "strict": "target_weight",
        }
        return aliases.get(value, value if value in {"passive_topk_dropout", "target_weight"} else "passive_topk_dropout")

    # 最大持仓个数，超过则本周期不再开新仓；默认 None 表示不限制（仅受 top_k 约束）
    @property
    def max_position_count(self) -> int | None:
        return self.get("strategy.max_position_count")

    # 每周期最多卖出几个持仓；默认 None 表示不限制（当前行为）
    @property
    def max_sell_per_cycle(self) -> int | None:
        return self.get("strategy.max_sell_per_cycle")

    @property
    def max_dropout(self) -> int | None:
        """
        对齐 qlib ThresholdTopkDropoutStrategy 的 max_dropout。
        未单独配置时兼容沿用 max_sell_per_cycle。
        """
        value = self.get("strategy.max_dropout", self.max_sell_per_cycle)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(f"max_dropout 配置无效: {value}, 将不限制 dropout")
            return None

    # 持仓市值低于此视为“粉尘”，不占持仓数、可被卖出；买入时仅当 held_usdt < 此值才视为无仓
    @property
    def min_position_value_usdt(self) -> float:
        return self.get("strategy.min_position_value_usdt", 2.0)

    @property
    def rebalance_tolerance_usdt(self) -> float:
        """目标仓位和实际仓位差额小于该值时不触发调仓，减少粉尘订单。"""
        fallback = max(float(self.min_position_value_usdt), 5.0)
        try:
            return float(self.get("strategy.rebalance_tolerance_usdt", fallback))
        except (TypeError, ValueError):
            return fallback

    @property
    def rebalance_tolerance_pct(self) -> float:
        """目标仓位偏差低于策略资金池该比例时不触发调仓，默认 0.1%。"""
        try:
            value = float(self.get("strategy.rebalance_tolerance_pct", 0.001))
        except (TypeError, ValueError):
            return 0.001
        return max(0.0, value)

    @property
    def enable_short(self) -> bool:
        """是否启用自定义 qlib 策略里的空头目标。现货执行器默认不支持做空。"""
        return bool(self.get("strategy.enable_short", False))

    @property
    def enforce_trade_coins_whitelist(self) -> bool:
        """
        是否把不在本策略 coins/predictions 中的持仓视为需要清理。
        默认 False，避免误卖同一账户里的人工或其它策略持仓。
        """
        return bool(self.get("strategy.enforce_trade_coins_whitelist", False))

    @property
    def account_position_reconcile_mode(self) -> str:
        """
        本地策略 ledger 和交易账户实际现货不一致时的自动修复模式。

        full_account: 默认，采用账户现货余额作为策略 ledger，包括接管 ledger
        未记录但账户实际持有的本策略交易币种。新增接管的实际持仓会在
        ledger 中同步扣减现金，避免凭空增加策略权益。

        remove_unbacked: 仅移除本地 ledger 声称持有但账户没有足额
        可卖余额的仓位，并清零现货不应出现的负仓。适合账户里仍有外部
        demo 余额或历史资产，但每个策略只使用 configured budget 的场景。

        off: 关闭自动对账。
        """
        raw = str(
            self.get("strategy.account_position_reconcile_mode", "full_account")
        ).strip().lower()
        aliases = {
            "true": "remove_unbacked",
            "on": "remove_unbacked",
            "enabled": "remove_unbacked",
            "safe": "remove_unbacked",
            "account": "full_account",
            "account_authoritative": "full_account",
            "full": "full_account",
            "false": "off",
            "disabled": "off",
            "none": "off",
        }
        mode = aliases.get(raw, raw)
        if mode not in {"remove_unbacked", "full_account", "off"}:
            logger.warning(
                f"account_position_reconcile_mode 配置无效: {raw}, 将使用 full_account"
            )
            return "full_account"
        return mode

    @property
    def scheduler_close_delay_sec(self) -> float:
        """K线理论收盘后额外等待几秒再拉取，给交易所 confirm 字段留缓冲。"""
        try:
            return float(self.get("scheduler.close_delay_sec", 8.0))
        except (TypeError, ValueError):
            return 8.0

    @property
    def api_key(self) -> str:
        env_name = self.get("api.api_key_env", "OKX_API_KEY_SIMU" if self.is_simulated else "OKX_API_KEY")
        return os.getenv(env_name, self.get("api.api_key", "") or "")

    @property
    def secret_key(self) -> str:
        env_name = self.get("api.secret_key_env", "OKX_SECRET_KEY_SIMU" if self.is_simulated else "OKX_SECRET_KEY")
        return os.getenv(env_name, self.get("api.secret_key", "") or "")

    @property
    def passphrase(self) -> str:
        env_name = self.get("api.passphrase_env", "OKX_PASSPHRASE")
        return os.getenv(env_name, self.get("api.passphrase", "") or "")

    @property
    def is_simulated(self) -> bool:
        return self.get("api.is_simulated", True)

    @property
    def data_limit(self) -> int:
        """模型预测所需的最小 K 线根数，默认为 300 (覆盖 288 窗口)"""
        return self.get("strategy.data_limit", 300)
