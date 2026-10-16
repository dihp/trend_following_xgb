from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Strategy(Protocol):
    """
    策略接口：
    - prepare_signals: 为每个symbol生成前一日信号列（避免未来信息）
    - should_exit_long/short: 决定开盘是否平仓
    """

    def prepare_signals(self, df_all: pd.DataFrame, allow_long: bool, allow_short: bool) -> pd.DataFrame: ...
    def should_exit_long(self, row: pd.Series, pos: dict, use_tp: bool, use_sl: bool, tp_pct: float, sl_pct: float) -> bool: ...
    def should_exit_short(self, row: pd.Series, pos: dict, use_tp: bool, use_sl: bool, tp_pct: float, sl_pct: float) -> bool: ...


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str = ""


def _to_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)

