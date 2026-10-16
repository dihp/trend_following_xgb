from __future__ import annotations

import numpy as np
import pandas as pd

import config
from .base import _to_float


class ProbThresholdStrategy:
    """
    基于概率阈值的策略（当前默认策略）：
    - 开仓：使用前一日的 (mdl_p_*) 严格条件生成 long/short 信号
    - 平仓：
        * 做多：open < MA20 或 止盈/止损
        * 做空：前一日 pred 为 no_trend(0) 或 止盈/止损
    """

    def prepare_signals(self, df_all: pd.DataFrame, allow_long: bool, allow_short: bool) -> pd.DataFrame:
        df_all = df_all.copy()
        # 基本清洗
        for col in ["mdl_p_up", "mdl_p_no_trend", "mdl_p_down"]:
            if col not in df_all.columns:
                df_all[col] = 0.0
        if "mdl_pred" not in df_all.columns:
            df_all["mdl_pred"] = 0

        def _prep_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date").reset_index(drop=True)
            p_up = pd.to_numeric(g.get("mdl_p_up", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            p_no = pd.to_numeric(g.get("mdl_p_no_trend", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            p_dn = pd.to_numeric(g.get("mdl_p_down", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pred_series = pd.to_numeric(g.get("mdl_pred", 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

            # 阈值配置
            long_conf_thr = float(getattr(config, "STRAT_LONG_CONFIDENCE_THRESHOLD", 0.90))
            short_conf_thr = float(getattr(config, "STRAT_SHORT_CONFIDENCE_THRESHOLD", 0.90))
            long_other_thr = float(getattr(config, "STRAT_LONG_OTHER_THRESHOLD", 0.2))
            short_other_thr = float(getattr(config, "STRAT_SHORT_OTHER_THRESHOLD", 0.12))
            long_sum_thr = float(getattr(config, "STRAT_LONG_SUM_THRESHOLD", 0.3))
            short_sum_thr = float(getattr(config, "STRAT_SHORT_SUM_THRESHOLD", 0.2))

            # 前一日预测/概率（用于开仓/出场记录）
            g["pred_prev"] = pred_series.shift(1).fillna(0).astype(int)
            g["pred_curr"] = pred_series
            g["p_up_prev"] = p_up.shift(1).fillna(0.0)
            g["p_no_prev"] = p_no.shift(1).fillna(0.0)
            g["p_dn_prev"] = p_dn.shift(1).fillna(0.0)

            # 入场信号：用“当日是否满足”再shift到前一日
            if allow_long:
                cond_buy = (p_up >= long_conf_thr) & (p_no <= long_other_thr) & (p_dn <= long_other_thr) & ((p_no + p_dn) <= long_sum_thr)
                g["long_entry_prev"] = cond_buy.shift(1).fillna(False).astype(bool)
            else:
                g["long_entry_prev"] = False

            if allow_short:
                cond_short = (p_dn >= short_conf_thr) & (p_no <= short_other_thr) & (p_up <= short_other_thr) & ((p_no + p_up) <= short_sum_thr)
                g["short_entry_prev"] = cond_short.shift(1).fillna(False).astype(bool)
            else:
                g["short_entry_prev"] = False

            # MA20（用于做多出场）
            ma20 = pd.to_numeric(g.get("MA20", np.nan), errors="coerce").replace([np.inf, -np.inf], np.nan)
            g["ma20_curr"] = ma20
            return g

        return df_all.groupby("symbol", group_keys=False).apply(_prep_group)

    def should_exit_long(self, row: pd.Series, pos: dict, use_tp: bool, use_sl: bool, tp_pct: float, sl_pct: float) -> bool:
        exit_px = _to_float(row.get("open", np.nan))
        ma20 = _to_float(row.get("ma20_curr", row.get("MA20", np.nan)))
        entry_px = _to_float(pos.get("entry_px", np.nan))
        if not np.isfinite(exit_px) or not np.isfinite(entry_px):
            return False
        float_ret = (exit_px - entry_px) / max(entry_px, 1e-12)
        is_tp = bool(use_tp) and (float_ret >= tp_pct)
        is_sl = bool(use_sl) and (float_ret <= -sl_pct)
        is_break_ma20 = (np.isfinite(ma20) and (exit_px < ma20))
        return bool(is_break_ma20 or is_tp or is_sl)

    def should_exit_short(self, row: pd.Series, pos: dict, use_tp: bool, use_sl: bool, tp_pct: float, sl_pct: float) -> bool:
        prev_pred = int(row.get("pred_prev", 0))
        is_no_trend = (prev_pred == 0)
        exit_px = _to_float(row.get("open", np.nan))
        entry_px = _to_float(pos.get("entry_px", np.nan))
        if not np.isfinite(exit_px) or not np.isfinite(entry_px):
            return bool(is_no_trend)
        float_ret = (entry_px - exit_px) / max(entry_px, 1e-12)
        is_tp = bool(use_tp) and (float_ret >= tp_pct)
        is_sl = bool(use_sl) and (float_ret <= -sl_pct)
        return bool(is_no_trend or is_tp or is_sl)

