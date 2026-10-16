"""
交易策略模块：
- prob_threshold: 基于概率阈值生成做多/做空信号（默认）
- trend_yesno: 仅按趋势方向(0/1/2)跟随：上涨做多、下跌做空、无趋势空仓
"""

from .prob_threshold import ProbThresholdStrategy
from .trend_yesno import TrendYesNoStrategy
from .trend_profit_guard import TrendProfitGuardStrategy


def get_strategy(name: str):
    name = (name or "").strip().lower()
    if name in ("", "prob", "prob_threshold", "threshold"):
        return ProbThresholdStrategy()
    if name in ("trend", "trend_yesno", "yesno", "trend_follow"):
        return TrendYesNoStrategy()
    if name in ("trend_profit_guard", "profit_guard", "guard", "trend_guard"):
        return TrendProfitGuardStrategy()
    raise ValueError(f"未知策略: {name}")

