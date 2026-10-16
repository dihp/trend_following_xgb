from __future__ import annotations

import numpy as np
import pandas as pd


class TrendYesNoStrategy:
    """
    纯“趋势(yes/no) + 方向”跟随策略：
    - 只依赖 mdl_pred（0/1/2）
        * 1 => 上涨：做多
        * 2 => 下跌：做空
        * 0 => 无趋势：空仓
    - 开仓：使用前一日 pred_prev
    - 平仓：
        * 做多：pred_prev != 1 则出场
        * 做空：pred_prev != 2 则出场
    - 下注额度由引擎统一控制（固定 alloc，不随置信度变化）
    """

    def prepare_signals(self, df_all: pd.DataFrame, allow_long: bool, allow_short: bool) -> pd.DataFrame:
        df_all = df_all.copy()
        if "mdl_pred" not in df_all.columns:
            df_all["mdl_pred"] = 0
        # 给交易记录用的概率列（缺失填0）
        for col in ["mdl_p_up", "mdl_p_no_trend", "mdl_p_down"]:
            if col not in df_all.columns:
                df_all[col] = 0.0

        def _prep_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date").reset_index(drop=True)
            pred_series = pd.to_numeric(g.get("mdl_pred", 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
            g["pred_prev"] = pred_series.shift(1).fillna(0).astype(int)
            g["pred_curr"] = pred_series
            # 概率列：仅用于记录（引擎会写 entry_p_* / exit_p_*）
            p_up = pd.to_numeric(g.get("mdl_p_up", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            p_no = pd.to_numeric(g.get("mdl_p_no_trend", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            p_dn = pd.to_numeric(g.get("mdl_p_down", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            g["p_up_prev"] = p_up.shift(1).fillna(0.0)
            g["p_no_prev"] = p_no.shift(1).fillna(0.0)
            g["p_dn_prev"] = p_dn.shift(1).fillna(0.0)

            if allow_long:
                g["long_entry_prev"] = (g["pred_prev"] == 1)
            else:
                g["long_entry_prev"] = False
            if allow_short:
                g["short_entry_prev"] = (g["pred_prev"] == 2)
            else:
                g["short_entry_prev"] = False
            return g

        return df_all.groupby("symbol", group_keys=False).apply(_prep_group)

    def should_exit_long(self, row: pd.Series, pos: dict, use_tp: bool, use_sl: bool, tp_pct: float, sl_pct: float) -> bool:
        # 纯跟随：不再上涨就退出（包括无趋势/转下跌）
        prev_pred = int(row.get("pred_prev", 0))
        return bool(prev_pred != 1)

    def should_exit_short(self, row: pd.Series, pos: dict, use_tp: bool, use_sl: bool, tp_pct: float, sl_pct: float) -> bool:
        # 纯跟随：不再下跌就退出（包括无趋势/转上涨）
        prev_pred = int(row.get("pred_prev", 0))
        return bool(prev_pred != 2)

