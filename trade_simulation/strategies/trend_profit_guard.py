from __future__ import annotations

import numpy as np
import pandas as pd

import config
from .base import _to_float


class TrendProfitGuardStrategy:
    """
    趋势入场 + 利润保护/快速止损策略：
    - 入场（开盘）：若前一日存在趋势（mdl_pred!=0）且对应方向置信度 >= 阈值，则入场。
      * 上涨趋势(1) => 做多；下跌趋势(2) => 做空。
      * 若已持仓，由引擎跳过，不会重复入场。
    - 出场（开盘）触发条件（做多/做空对称）：
      * 若“已达峰值利润” < 10%，当前收益为负 => 立即出场。
      * 若“已达峰值利润” >= 10%，当前收益跌破峰值利润的 50% => 出场（50%回撤止盈）。
      * 若连续 3 天趋势为 0（无趋势） => 出场。
      * 若趋势从 +1 变为 -1 或 -1 变为 +1（方向反转） => 出场。
    """

    def prepare_signals(self, df_all: pd.DataFrame, allow_long: bool, allow_short: bool) -> pd.DataFrame:
        df_all = df_all.copy()
        if "mdl_pred" not in df_all.columns:
            df_all["mdl_pred"] = 0
        for col in ["mdl_p_up", "mdl_p_no_trend", "mdl_p_down"]:
            if col not in df_all.columns:
                df_all[col] = 0.0

        # 阈值配置（允许复用现有阈值；也可单独覆盖）
        long_thr = float(getattr(config, "STRAT_TREND_GUARD_LONG_CONFIDENCE_THRESHOLD",
                                 getattr(config, "STRAT_LONG_CONFIDENCE_THRESHOLD", 0.90)))
        short_thr = float(getattr(config, "STRAT_TREND_GUARD_SHORT_CONFIDENCE_THRESHOLD",
                                  getattr(config, "STRAT_SHORT_CONFIDENCE_THRESHOLD", 0.90)))

        def _prep_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date").reset_index(drop=True)
            pred = pd.to_numeric(g.get("mdl_pred", 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

            p_up = pd.to_numeric(g.get("mdl_p_up", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            p_no = pd.to_numeric(g.get("mdl_p_no_trend", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            p_dn = pd.to_numeric(g.get("mdl_p_down", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # 给引擎/记录使用的前一日预测/概率
            g["pred_prev"] = pred.shift(1).fillna(0).astype(int)
            g["pred_curr"] = pred
            g["p_up_prev"] = p_up.shift(1).fillna(0.0)
            g["p_no_prev"] = p_no.shift(1).fillna(0.0)
            g["p_dn_prev"] = p_dn.shift(1).fillna(0.0)

            # 趋势方向 sign：+1(上涨), -1(下跌), 0(无趋势)
            sign = np.where(pred == 1, 1, np.where(pred == 2, -1, 0)).astype(int)
            sign_prev = pd.Series(sign, index=g.index).shift(1).fillna(0).astype(int)
            sign_prev2 = pd.Series(sign, index=g.index).shift(2).fillna(0).astype(int)
            g["trend_flip_prev"] = ((sign_prev != 0) & (sign_prev2 != 0) & (sign_prev != sign_prev2)).astype(bool)

            # 连续无趋势天数（以“当前日 pred”为准），再 shift(1) 得到“昨日为止”的连续无趋势天数
            is_no = (pred == 0).astype(int)
            grp = (is_no == 0).cumsum()
            no_streak = is_no.groupby(grp).cumsum()
            g["no_trend_streak_prev"] = no_streak.shift(1).fillna(0).astype(int)

            # 入场信号：用“当日是否满足”再 shift 到前一日，避免未来信息
            cond_long = (pred == 1) & (p_up >= long_thr)
            cond_short = (pred == 2) & (p_dn >= short_thr)
            if allow_long:
                g["long_entry_prev"] = cond_long.shift(1).fillna(False).astype(bool)
            else:
                g["long_entry_prev"] = False
            if allow_short:
                g["short_entry_prev"] = cond_short.shift(1).fillna(False).astype(bool)
            else:
                g["short_entry_prev"] = False
            return g

        return df_all.groupby("symbol", group_keys=False).apply(_prep_group)

    @staticmethod
    def _update_peak_profit_pct(pos: dict, profit_pct: float) -> float:
        peak = pos.get("peak_profit_pct", np.nan)
        try:
            peak = float(peak)
        except Exception:
            peak = float("nan")
        if not np.isfinite(peak):
            peak = float(profit_pct) if np.isfinite(profit_pct) else float("nan")
        if np.isfinite(profit_pct) and (not np.isfinite(peak) or profit_pct > peak):
            peak = float(profit_pct)
        pos["peak_profit_pct"] = float(peak) if np.isfinite(peak) else float("nan")
        return float(peak)

    def _should_exit_common(self, row: pd.Series, pos: dict, direction: int) -> bool:
        """
        direction: 1=做多, -1=做空
        """
        # 1) 趋势连续 0
        zero_exit_days = int(getattr(config, "STRAT_TREND_GUARD_ZERO_EXIT_DAYS", 3))
        if int(row.get("no_trend_streak_prev", 0)) >= max(1, zero_exit_days):
            return True

        # 2) 趋势方向反转
        if bool(row.get("trend_flip_prev", False)):
            return True

        # 3) 利润规则（以开盘价评估）
        px = _to_float(row.get("open", np.nan))
        entry_px = _to_float(pos.get("entry_px", np.nan))
        if not np.isfinite(px) or not np.isfinite(entry_px):
            return False

        if direction == 1:
            profit_pct = (px - entry_px) / max(entry_px, 1e-12) * 100.0
        else:
            profit_pct = (entry_px - px) / max(entry_px, 1e-12) * 100.0

        peak_profit = self._update_peak_profit_pct(pos, profit_pct)

        arm_profit = float(getattr(config, "STRAT_TREND_GUARD_ARM_PROFIT_PCT", 10.0))
        keep_ratio = float(getattr(config, "STRAT_TREND_GUARD_TRAIL_KEEP_RATIO", 0.50))

        # 未达到 10% 峰值利润前，出现亏损则立刻出场
        if np.isfinite(peak_profit) and (peak_profit < arm_profit) and (profit_pct < 0.0):
            return True

        # 达到 10% 峰值利润后，跌破峰值利润的 50% 则出场
        if np.isfinite(peak_profit) and (peak_profit >= arm_profit):
            if profit_pct <= peak_profit * keep_ratio:
                return True

        return False

    def should_exit_long(self, row: pd.Series, pos: dict, use_tp: bool, use_sl: bool, tp_pct: float, sl_pct: float) -> bool:
        return bool(self._should_exit_common(row=row, pos=pos, direction=1))

    def should_exit_short(self, row: pd.Series, pos: dict, use_tp: bool, use_sl: bool, tp_pct: float, sl_pct: float) -> bool:
        return bool(self._should_exit_common(row=row, pos=pos, direction=-1))

