"""
trade_simulation.py
最简“实战模拟”：按时间推进，使用 data/with_prediction/all_with_prediction.csv（如不存在则合并目录下各标的）
- 初始资金：config.STRAT_INITIAL_CAPITAL（默认25000）
- 入场规则（仅多头现货）：
  - 使用前一交易日的上升趋势信号：signal_up = (mdl_pred==1 且 mdl_p_up>=阈值)
  - 在当日开盘执行买入，投入资金比例与昨日 p_up 线性挂钩：
      alloc = min(PER_SYMBOL_CAP, max(0, (p_up_prev - up_thr)/(1 - up_thr)))，再受总资金上限 ALLOC_CAP 约束
  - 同日多标的存在时，按各自 alloc 比例归一化分配，不超总上限
- 卖出规则：
  - 当前持仓对应标的的昨日信号不再为上涨（即 signal_up_prev=False）时，当日开盘全部卖出
- 成交价：开盘价；费用：开/平各一次（fee_bps + slippage_bps）
- 输出：models/result/trading_result/trade_sim_trades_{ts}.csv 与 trade_sim_summary_{ts}.csv
"""
import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# 确保可导入项目根目录下的 config.py（支持直接运行本文件）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from trade_simulation.strategies import get_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def bps_to_frac(bps: float) -> float:
    return float(bps) / 10000.0


def load_all_with_prediction(project_root: str, symbols=None) -> pd.DataFrame:
    all_path = os.path.join(project_root, "data", "with_prediction", "all_with_prediction.csv")
    if os.path.exists(all_path):
        df = pd.read_csv(all_path)
    else:
        # 合并生成
        base = os.path.join(project_root, "data", "with_prediction")
        files = [f for f in os.listdir(base) if f.endswith("_with_prediction.csv")]
        dfs = []
        for f in files:
            sym = f.replace("_with_prediction.csv", "")
            if symbols and sym not in symbols:
                continue
            p = os.path.join(base, f)
            try:
                dfi = pd.read_csv(p)
                if "symbol" not in dfi.columns:
                    dfi["symbol"] = sym
                dfs.append(dfi)
            except Exception as e:
                logger.warning(f"{p} 读取失败：{e}")
        if len(dfs) == 0:
            raise FileNotFoundError("未找到 with_prediction 文件，请先运行 annotate_predictions.py")
        df = pd.concat(dfs, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        df.to_csv(all_path, index=False, encoding="utf-8-sig")
        logger.info(f"已生成合并表：{all_path}")
    # 基本列检查
    if "date" not in df.columns or "open" not in df.columns:
        raise ValueError("缺少必要列 date/open")
    df["date"] = pd.to_datetime(df["date"])
    if "symbol" not in df.columns:
        raise ValueError("缺少 symbol 列")
    # 概率缺失填0
    for col in ["mdl_p_up", "mdl_p_no_trend", "mdl_p_down"]:
        if col not in df.columns:
            df[col] = 0.0
    if "mdl_pred" not in df.columns:
        df["mdl_pred"] = 0
    return df


def simulate(df_all: pd.DataFrame,
             up_thr: float,
             alloc_cap: float,
             per_symbol_cap: float,
             fee_bps: float,
             slippage_bps: float,
             initial_capital: float,
             start_date: str = "",
             end_date: str = "",
             allow_long: bool = None,
             allow_short: bool = None):
    # 从config读取做多/做空开关（如果未传入）
    if allow_long is None:
        allow_long = bool(getattr(config, 'STRAT_ALLOW_LONG', True))
    if allow_short is None:
        allow_short = bool(getattr(config, 'STRAT_ALLOW_SHORT', True))
    
    # 输出做多/做空开关状态
    logger.info("=" * 80)
    logger.info(f"交易策略配置：做多功能 {'已启用' if allow_long else '已禁用'}")
    logger.info(f"交易策略配置：做空功能 {'已启用' if allow_short else '已禁用'}")
    logger.info("=" * 80)
    
    if start_date:
        df_all = df_all[df_all["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df_all = df_all[df_all["date"] <= pd.to_datetime(end_date)]
    df_all = df_all.sort_values(["date", "symbol"]).reset_index(drop=True)
    fee = bps_to_frac(fee_bps)
    slp = bps_to_frac(slippage_bps)
    tcost = fee + slp
    # 止盈 / 止损参数（使用收益率小数，例：0.10=10%）
    use_take_profit = bool(getattr(config, "STRAT_USE_TAKE_PROFIT", False))
    use_stop_loss = bool(getattr(config, "STRAT_USE_STOP_LOSS", False))
    take_profit_pct = float(getattr(config, "STRAT_TAKE_PROFIT_PCT", 0.10))
    stop_loss_pct = float(getattr(config, "STRAT_STOP_LOSS_PCT", 0.08))
    # ===== 策略模块化：按config选择策略并生成“前一日入场信号” =====
    strategy_name = str(getattr(config, "SIM_STRATEGY_NAME", "prob_threshold"))
    strategy = get_strategy(strategy_name)
    df_all = strategy.prepare_signals(df_all=df_all, allow_long=allow_long, allow_short=allow_short)
    all_dates = sorted(df_all["date"].unique())
    symbols = sorted(df_all["symbol"].unique())
    idx_map = {sym: df_all[df_all["symbol"] == sym].set_index("date") for sym in symbols}
    # 账户状态
    cash = float(initial_capital)
    # positions: shares > 0 表示做多，shares < 0 表示做空，direction: 1=做多, -1=做空
    positions = {sym: {"shares": 0, "entry_px": None, "entry_date": None, "entry_pred": None, "direction": 0} for sym in symbols}
    trades = []
    daily_balances = []  # 记录每日余额
    daily_positions = []  # 记录每日持仓明细
    stopped_early = False

    def _force_sell_longs_to_raise_cash(date_d, cash_needed, exclude_symbol=None):
        """
        为了避免平空导致现金<0：在同一天开盘价强制卖出部分/全部多头持仓来补足现金。
        返回更新后的cash以及本次强平的卖出笔数。
        """
        nonlocal cash
        forced_closes = 0
        if cash_needed <= 0:
            return cash, forced_closes
        # 优先卖出市值较大的多头（更快补现金）
        long_syms = []
        for s, p in positions.items():
            if exclude_symbol is not None and s == exclude_symbol:
                continue
            if p.get("direction", 0) == 1 and int(p.get("shares", 0)) > 0:
                try:
                    s_imap = idx_map.get(s)
                    if s_imap is None or date_d not in s_imap.index:
                        continue
                    s_row = s_imap.loc[date_d]
                    if isinstance(s_row, pd.DataFrame):
                        s_row = s_row.iloc[-1]
                    px0 = float(s_row["open"])
                    mv = int(p["shares"]) * px0
                    long_syms.append((mv, s, px0))
                except Exception:
                    continue
        long_syms.sort(reverse=True, key=lambda x: x[0])
        for _, s, px0 in long_syms:
            if cash >= cash_needed:
                break
            p = positions[s]
            shares0 = int(p["shares"])
            if shares0 <= 0:
                continue
            # 全部卖出该多头（开盘价），补现金
            entry_px0 = float(p["entry_px"])
            entry_date0 = p.get("entry_date")
            entry_pred0 = p.get("entry_pred", 1)
            gross_pnl0 = shares0 * (px0 - entry_px0)
            exit_cost0 = shares0 * px0 * tcost
            entry_cost0 = shares0 * entry_px0 * tcost
            cash += shares0 * px0 * (1.0 - tcost)
            net_pnl0 = gross_pnl0 - entry_cost0 - exit_cost0
            profit_pct0 = (net_pnl0 / max(shares0 * entry_px0, 1e-12)) * 100.0
            trades.append({
                "symbol": s,
                "entry_date": pd.to_datetime(entry_date0) if entry_date0 is not None else pd.to_datetime(date_d),
                "exit_date": pd.to_datetime(date_d),
                "direction": 1,
                "shares": shares0,
                "entry_price": entry_px0,
                "exit_price": px0,
                "price_diff": px0 - entry_px0,
                "gross_pnl": gross_pnl0,
                "net_pnl": net_pnl0,
                "profit_pct": profit_pct0,
                "open_cash_flow": float(p.get("entry_cash_flow", -shares0 * entry_px0 * (1.0 + tcost))),
                "close_cash_flow": float(shares0 * px0 * (1.0 - tcost)),
                "equity_before_open": float(p.get("equity_before_open", np.nan)),
                "equity_after_close": cash,
                "balance_after": cash,
                "entry_p_no": float(p.get("entry_p_no", 0.0)),
                "entry_p_up": float(p.get("entry_p_up", 0.0)),
                "entry_p_down": float(p.get("entry_p_down", 0.0)),
                "exit_p_no": np.nan,
                "exit_p_up": np.nan,
                "exit_p_down": np.nan,
                "reason": "force_sell_long_to_cover_short",
            })
            positions[s] = {"shares": 0, "entry_px": None, "entry_date": None, "entry_pred": None, "direction": 0}
            forced_closes += 1
        return cash, forced_closes
    for d in all_dates:
        # 检查余额，小于0则停止
        if cash < 0:
            logger.warning(f"余额小于0 ({cash:.2f})，停止交易。日期：{d}")
            stopped_early = True
            break
        # 调试：记录每日交易活动
        daily_trades_count = 0
        daily_opens_count = 0
        daily_closes_count = 0
        daily_signals_long = 0
        daily_signals_short = 0
        # 卖出/平仓（开盘）
        for sym in symbols:
            imap = idx_map[sym]
            if d not in imap.index:
                continue
            row = imap.loc[d]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            pos = positions[sym]
            # 使用前一天的预测和概率（避免未来信息泄漏）
            prev_pred = int(row.get("pred_prev", 0))
            p_dn_prev = float(row.get("p_dn_prev", 0.0))
            
            # 处理做多平仓（如果允许做多）
            if allow_long and pos["shares"] > 0 and pos["direction"] == 1:
                exit_px = float(row["open"])
                entry_px = float(pos["entry_px"])
                shares = int(pos["shares"])
                # 如果满足策略出场条件，则卖出
                if strategy.should_exit_long(
                    row=row,
                    pos=pos,
                    use_tp=use_take_profit,
                    use_sl=use_stop_loss,
                    tp_pct=take_profit_pct,
                    sl_pct=stop_loss_pct,
                ):
                    shares = int(pos["shares"])
                    gross_pnl = shares * (exit_px - entry_px)
                    exit_cost = shares * exit_px * tcost
                    entry_cost = shares * entry_px * tcost
                    cash += shares * exit_px * (1.0 - tcost)
                    net_pnl = gross_pnl - entry_cost - exit_cost
                    # 计算当前总资产（现金+持仓市值），不计入已平仓标的
                    total_equity = cash
                    for s, p in positions.items():
                        if s == sym:
                            continue
                        if p["shares"] != 0 and s in idx_map:
                            try:
                                s_imap = idx_map[s]
                                if d in s_imap.index:
                                    s_row = s_imap.loc[d]
                                    if isinstance(s_row, pd.DataFrame):
                                        s_row = s_row.iloc[-1]
                                    total_equity += p["shares"] * float(s_row["open"])
                            except:
                                pass
                    close_cash_flow = shares * exit_px * (1.0 - tcost)
                    price_diff = exit_px - entry_px
                    profit_pct = (net_pnl / max(shares * entry_px, 1e-12)) * 100.0
                    trades.append({
                        "symbol": sym,
                        "entry_date": pd.to_datetime(pos["entry_date"]),
                        "exit_date": pd.to_datetime(d),
                        "direction": 1,
                        "shares": shares,
                        "entry_price": entry_px,
                        "exit_price": exit_px,
                        "price_diff": price_diff,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "profit_pct": profit_pct,
                        "open_cash_flow": float(pos.get("entry_cash_flow", -shares * entry_px * (1.0 + tcost))),
                        "close_cash_flow": close_cash_flow,
                        "equity_before_open": float(pos.get("equity_before_open", np.nan)),
                        "equity_after_close": total_equity,
                        "balance_after": total_equity,
                        "entry_p_no": float(pos.get("entry_p_no", 0.0)),
                        "entry_p_up": float(pos.get("entry_p_up", 0.0)),
                        "entry_p_down": float(pos.get("entry_p_down", 0.0)),
                        "exit_p_no": float(row.get("p_no_prev", 0.0)),
                        "exit_p_up": float(row.get("p_up_prev", 0.0)),
                        "exit_p_down": float(row.get("p_dn_prev", 0.0)),
                    })
                    positions[sym] = {"shares": 0, "entry_px": None, "entry_date": None, "entry_pred": None, "direction": 0}
            
            # 处理做空平仓（如果允许做空）
            elif allow_short and pos["shares"] < 0 and pos["direction"] == -1:
                exit_px = float(row["open"])
                entry_px = float(pos["entry_px"])
                shares = abs(int(pos["shares"]))  # 做空时shares为负，取绝对值
                # 平仓条件：由策略决定（可叠加止盈/止损）
                if strategy.should_exit_short(
                    row=row,
                    pos=pos,
                    use_tp=use_take_profit,
                    use_sl=use_stop_loss,
                    tp_pct=take_profit_pct,
                    sl_pct=stop_loss_pct,
                ):
                    # 做空：先卖后买，盈利 = 卖出价 - 买入价（平仓价）
                    gross_pnl = shares * (entry_px - exit_px)  # 做空盈利计算
                    exit_cost = shares * exit_px * tcost
                    entry_cost = shares * entry_px * tcost
                    # 平仓时：买入股票还券，支付买入价（加上费用）
                    # 注意：开仓时已经收到现金，这里只需要支付平仓买入价
                    required_cash = shares * exit_px * (1.0 + tcost)
                    if required_cash > cash:
                        # 现金不足：先强制卖出其它多头补足现金，避免 cash<0
                        _force_sell_longs_to_raise_cash(d, required_cash, exclude_symbol=sym)
                    if required_cash > cash:
                        # 仍不足：只能部分平仓（能买回多少买回多少），剩余继续持有，避免现金为负
                        affordable = int(np.floor(cash / max(exit_px * (1.0 + tcost), 1e-12)))
                        if affordable <= 0:
                            # 无法平仓，跳过（保持仓位不变）
                            logger.warning(f"{sym} 做空平仓资金不足（需 {required_cash:.2f}，仅有 {cash:.2f}），且无法部分平仓，跳过。日期：{d}")
                            continue
                        cover_shares = min(shares, affordable)
                        cash -= cover_shares * exit_px * (1.0 + tcost)
                        # 记录部分平仓交易
                        gross_pnl_part = cover_shares * (entry_px - exit_px)
                        exit_cost_part = cover_shares * exit_px * tcost
                        entry_cost_part = cover_shares * entry_px * tcost
                        net_pnl_part = gross_pnl_part - entry_cost_part - exit_cost_part
                        profit_pct_part = (net_pnl_part / max(cover_shares * entry_px, 1e-12)) * 100.0
                        trades.append({
                            "symbol": sym,
                            "entry_date": pd.to_datetime(pos["entry_date"]),
                            "exit_date": pd.to_datetime(d),
                            "direction": -1,
                            "shares": cover_shares,
                            "entry_price": entry_px,
                            "exit_price": exit_px,
                            "price_diff": exit_px - entry_px,
                            "gross_pnl": gross_pnl_part,
                            "net_pnl": net_pnl_part,
                            "profit_pct": profit_pct_part,
                            "open_cash_flow": float(cover_shares * entry_px * (1.0 - tcost)),
                            "close_cash_flow": float(-cover_shares * exit_px * (1.0 + tcost)),
                            "equity_before_open": float(pos.get("equity_before_open", np.nan)),
                            "equity_after_close": cash,
                            "balance_after": cash,
                            "entry_p_no": float(pos.get("entry_p_no", 0.0)),
                            "entry_p_up": float(pos.get("entry_p_up", 0.0)),
                            "entry_p_down": float(pos.get("entry_p_down", 0.0)),
                            "exit_p_no": float(row.get("p_no_prev", 0.0)),
                            "exit_p_up": float(row.get("p_up_prev", 0.0)),
                            "exit_p_down": float(row.get("p_dn_prev", 0.0)),
                            "reason": "partial_cover_due_to_cash_constraint",
                        })
                        remaining = shares - cover_shares
                        if remaining <= 0:
                            positions[sym] = {"shares": 0, "entry_px": None, "entry_date": None, "entry_pred": None, "direction": 0}
                            daily_closes_count += 1
                        else:
                            positions[sym]["shares"] = -int(remaining)
                        daily_trades_count += 1
                        continue
                    # 现金充足：正常全额平仓
                    cash -= required_cash  # 支付平仓买入价
                    net_pnl = gross_pnl - entry_cost - exit_cost
                    # 计算当前总资产（不计入已平仓标的）
                    total_equity = cash
                    for s, p in positions.items():
                        if s == sym:
                            continue
                        if p["shares"] != 0 and s in idx_map:
                            try:
                                s_imap = idx_map[s]
                                if d in s_imap.index:
                                    s_row = s_imap.loc[d]
                                    if isinstance(s_row, pd.DataFrame):
                                        s_row = s_row.iloc[-1]
                                    total_equity += p["shares"] * float(s_row["open"])
                            except:
                                pass
                    close_cash_flow = -shares * exit_px * (1.0 + tcost)
                    price_diff = exit_px - entry_px
                    profit_pct = (net_pnl / max(shares * entry_px, 1e-12)) * 100.0
                    trades.append({
                        "symbol": sym,
                        "entry_date": pd.to_datetime(pos["entry_date"]),
                        "exit_date": pd.to_datetime(d),
                        "direction": -1,
                        "shares": shares,
                        "entry_price": entry_px,
                        "exit_price": exit_px,
                        "price_diff": price_diff,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "profit_pct": profit_pct,
                        "open_cash_flow": float(pos.get("entry_cash_flow", shares * entry_px * (1.0 - tcost))),
                        "close_cash_flow": close_cash_flow,
                        "equity_before_open": float(pos.get("equity_before_open", np.nan)),
                        "equity_after_close": total_equity,
                        "balance_after": total_equity,
                        "entry_p_no": float(pos.get("entry_p_no", 0.0)),
                        "entry_p_up": float(pos.get("entry_p_up", 0.0)),
                        "entry_p_down": float(pos.get("entry_p_down", 0.0)),
                        "exit_p_no": float(row.get("p_no_prev", 0.0)),
                        "exit_p_up": float(row.get("p_up_prev", 0.0)),
                        "exit_p_down": float(row.get("p_dn_prev", 0.0)),
                    })
                    positions[sym] = {"shares": 0, "entry_px": None, "entry_date": None, "entry_pred": None, "direction": 0}
                    daily_closes_count += 1
                    daily_trades_count += 1
        # 买入/做空（开盘）：收集候选
        cands_long = []  # 做多候选
        weights_long = []
        cands_short = []  # 做空候选
        weights_short = []
        for sym in symbols:
            imap = idx_map[sym]
            if d not in imap.index:
                continue
            row = imap.loc[d]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            # 如果已有持仓，跳过
            if positions[sym]["shares"] != 0:
                continue
            
            # 做多候选（如果允许做多）
            if allow_long and bool(row.get("long_entry_prev", False)):
                daily_signals_long += 1
                # 如果满足买入信号，则买入
                # 固定下注比例（不随置信度变化）
                alloc = float(getattr(config, "STRAT_FIXED_ALLOC", 0.10))
                alloc = min(alloc, per_symbol_cap)  # 仍受单标的上限约束
                if alloc > 0:
                    cands_long.append((sym, row))
                    weights_long.append(alloc)
            
            # 做空候选（如果允许做空）
            if allow_short and bool(row.get("short_entry_prev", False)):
                daily_signals_short += 1
                # 如果满足做空信号，则做空
                # 固定下注比例（不随置信度变化）
                alloc = float(getattr(config, "STRAT_FIXED_ALLOC", 0.10))
                alloc = min(alloc, per_symbol_cap)
                if alloc > 0:
                    cands_short.append((sym, row))
                    weights_short.append(alloc)
        
        # 处理做多（如果允许做多）
        if allow_long:
            cands = cands_long
            weights = weights_long
            if len(cands) > 0 and sum(weights) > 0 and cash > 0:
                # 计算当前总市值（现金+持仓市值，包括做多和做空）
                total_equity = cash
                for s, p in positions.items():
                    if p["shares"] != 0 and s in idx_map:
                        try:
                            s_imap = idx_map[s]
                            if d in s_imap.index:
                                s_row = s_imap.loc[d]
                                if isinstance(s_row, pd.DataFrame):
                                    s_row = s_row.iloc[-1]
                                if p["shares"] > 0:
                                    total_equity += p["shares"] * float(s_row["open"])
                                else:  # 做空
                                    total_equity += p["shares"] * float(s_row["open"])  # 负值
                        except:
                            pass
                weights = np.array(weights, dtype=float)
                # 归一化到 alloc_cap 上限
                tot = weights.sum()
                weights = weights / tot * min(alloc_cap, tot)
                # 逐个下单，可能受现金限制
                for (sym, row), w in zip(cands, weights):
                    # 再次检查余额
                    if cash <= 0:
                        break
                    px = float(row["open"])
                    # 每次下注10%的市值，向下取整
                    desired_cash = total_equity * float(w)  # 使用总市值
                    est_cost_per_share = px * (1.0 + tcost)
                    shares = int(np.floor(desired_cash / max(est_cost_per_share, 1e-12)))
                    if shares <= 0:
                        continue
                    spend = shares * px * (1.0 + tcost)
                    if spend > cash:
                        shares = int(np.floor(cash / max(est_cost_per_share, 1e-12)))
                        if shares <= 0:
                            continue
                        spend = shares * px * (1.0 + tcost)
                    # 确保不会导致余额小于0
                    if spend > cash:
                        continue
                    cash -= spend
                    # 记录买入时的预测值和三项置信度
                    entry_pred = int(row.get("pred_prev", 1))
                    positions[sym] = {
                        "shares": shares,
                        "entry_px": px,
                        "entry_date": d,
                        "entry_pred": entry_pred,
                        "direction": 1,
                        "equity_before_open": float(total_equity),
                        "entry_cash_flow": -float(spend),
                        "entry_p_no": float(row.get("p_no_prev", 0.0)),
                        "entry_p_up": float(row.get("p_up_prev", 0.0)),
                        "entry_p_down": float(row.get("p_dn_prev", 0.0)),
                    }
                    daily_opens_count += 1
                    daily_trades_count += 1
        
        # 处理做空（如果允许做空）
        if allow_short and len(cands_short) > 0 and sum(weights_short) > 0 and cash > 0:
            # 计算当前总市值（现金+持仓市值）
            total_equity = cash
            for s, p in positions.items():
                if p["shares"] != 0 and s in idx_map:
                    try:
                        s_imap = idx_map[s]
                        if d in s_imap.index:
                            s_row = s_imap.loc[d]
                            if isinstance(s_row, pd.DataFrame):
                                s_row = s_row.iloc[-1]
                            if p["shares"] > 0:
                                total_equity += p["shares"] * float(s_row["open"])
                            else:  # 做空
                                total_equity += p["shares"] * float(s_row["open"])  # 负值
                    except:
                        pass
            weights_short = np.array(weights_short, dtype=float)
            # 归一化到 alloc_cap 上限
            tot = weights_short.sum()
            weights_short = weights_short / tot * min(alloc_cap, tot)
            # 逐个下单，可能受现金限制
            for (sym, row), w in zip(cands_short, weights_short):
                # 再次检查余额
                if cash <= 0:
                    break
                px = float(row["open"])
                # 每次下注10%的市值，向下取整
                desired_cash = total_equity * float(w)  # 使用总市值
                est_cost_per_share = px * (1.0 + tcost)
                shares = int(np.floor(desired_cash / max(est_cost_per_share, 1e-12)))
                if shares <= 0:
                    continue
                # 做空：卖出股票，收到现金（扣除费用）
                # 做空时，我们收到卖出价，但需要预留保证金（通常需要更多现金，这里简化处理）
                # 做空成本：卖出时的费用
                short_cost = shares * px * tcost
                if short_cost > cash:
                    shares = int(np.floor(cash / max(px * tcost, 1e-12)))
                    if shares <= 0:
                        continue
                    short_cost = shares * px * tcost
                # 确保不会导致余额小于0
                if short_cost > cash:
                    continue
                # 做空：卖出股票，收到现金（扣除费用）
                cash += shares * px * (1.0 - tcost)  # 收到卖出价，扣除费用
                # 记录做空时的预测值和三项置信度
                entry_pred = int(row.get("pred_prev", 2))
                positions[sym] = {
                    "shares": -shares,
                    "entry_px": px,
                    "entry_date": d,
                    "entry_pred": entry_pred,
                    "direction": -1,
                    "equity_before_open": float(total_equity),
                    "entry_cash_flow": float(shares * px * (1.0 - tcost)),
                    "entry_p_no": float(row.get("p_no_prev", 0.0)),
                    "entry_p_up": float(row.get("p_up_prev", 0.0)),
                    "entry_p_down": float(row.get("p_dn_prev", 0.0)),
                }
                daily_opens_count += 1
                daily_trades_count += 1
        # 记录每日余额和持仓明细（现金+持仓市值）
        total_equity = cash
        total_market_value = 0.0
        for sym, pos in positions.items():
            if pos["shares"] != 0 and sym in idx_map:
                try:
                    imap = idx_map[sym]
                    if d in imap.index:
                        row = imap.loc[d]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[-1]
                        current_price = float(row["open"])
                        shares = int(pos["shares"])
                        if shares > 0:  # 做多
                            market_value = shares * current_price
                            total_market_value += market_value
                            total_equity += market_value
                            # 记录该股票的持仓明细
                            daily_positions.append({
                                "date": pd.to_datetime(d),
                                "symbol": sym,
                                "direction": 1,
                                "shares": shares,
                                "price": current_price,
                                "market_value": market_value,
                            })
                        else:  # 做空
                            shares_abs = abs(shares)
                            # 做空市值：负值，表示负债
                            market_value = shares * current_price  # 负值
                            total_market_value += market_value  # 负值
                            total_equity += market_value  # 负值，减少总资产
                            # 记录该股票的持仓明细
                            daily_positions.append({
                                "date": pd.to_datetime(d),
                                "symbol": sym,
                                "direction": -1,
                                "shares": shares_abs,
                                "price": current_price,
                                "market_value": market_value,  # 负值
                            })
                except:
                    pass
        daily_balances.append({
            "date": pd.to_datetime(d),
            "balance": total_equity,
            "cash": cash,
            "total_market_value": total_market_value,
        })
        # 调试：每月第一天或每周一输出交易活动摘要
        if d.day == 1 or (d.weekday() == 0) or daily_trades_count > 0:
            logger.info(f"日期: {d.strftime('%Y-%m-%d')}, 现金: {cash:.2f}, 总资产: {total_equity:.2f}, 做多信号: {daily_signals_long}, 做空信号: {daily_signals_short}, 开仓: {daily_opens_count}, 平仓: {daily_closes_count}, 持仓数: {sum(1 for p in positions.values() if p['shares'] != 0)}")
    # 收官：最后一日强平所有持仓
    if len(all_dates) > 0:
        last_d = all_dates[-1]
        for sym in symbols:
            pos = positions[sym]
            if pos["shares"] == 0 or pos["entry_date"] is None:
                continue
            imap = idx_map[sym]
            # 用该标的最后可见日的开盘价强平
            last_row = imap.iloc[-1]
            exit_px = float(last_row["open"])
            entry_px = float(pos["entry_px"])
            direction = pos.get("direction", 1)
            
            if allow_long and direction == 1:  # 做多平仓（如果允许做多）
                shares = int(pos["shares"])
                gross_pnl = shares * (exit_px - entry_px)
                exit_cost = shares * exit_px * tcost
                entry_cost = shares * entry_px * tcost
                cash += shares * exit_px * (1.0 - tcost)
                net_pnl = gross_pnl - entry_cost - exit_cost
                price_diff = exit_px - entry_px
                profit_pct = (net_pnl / max(shares * entry_px, 1e-12)) * 100.0
                # 获取最后一日的前一日概率（如果存在）
                exit_p_no = float(last_row.get("p_no_prev", last_row.get("p_no_curr", 0.0)) if hasattr(last_row, "get") else 0.0)
                exit_p_up = float(last_row.get("p_up_prev", last_row.get("p_up_curr", 0.0)) if hasattr(last_row, "get") else 0.0)
                exit_p_down = float(last_row.get("p_dn_prev", last_row.get("p_dn_curr", 0.0)) if hasattr(last_row, "get") else 0.0)
                trades.append({
                    "symbol": sym,
                    "entry_date": pd.to_datetime(pos["entry_date"]),
                    "exit_date": pd.to_datetime(last_row.name if isinstance(last_row.name, pd.Timestamp) else last_d),
                    "direction": 1,
                    "shares": shares,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "price_diff": price_diff,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "profit_pct": profit_pct,
                    "open_cash_flow": float(pos.get("entry_cash_flow", -shares * entry_px * (1.0 + tcost))),
                    "close_cash_flow": float(shares * exit_px * (1.0 - tcost)),
                    "equity_before_open": float(pos.get("equity_before_open", np.nan)),
                    "equity_after_close": cash,
                    "balance_after": cash,  # 收官后只有现金
                    "entry_p_no": float(pos.get("entry_p_no", 0.0)),
                    "entry_p_up": float(pos.get("entry_p_up", 0.0)),
                    "entry_p_down": float(pos.get("entry_p_down", 0.0)),
                    "exit_p_no": exit_p_no,
                    "exit_p_up": exit_p_up,
                    "exit_p_down": exit_p_down,
                })
            elif allow_short and direction == -1:  # 做空平仓（如果允许做空）
                shares = abs(int(pos["shares"]))
                gross_pnl = shares * (entry_px - exit_px)  # 做空盈利计算
                exit_cost = shares * exit_px * tcost
                entry_cost = shares * entry_px * tcost
                # 平仓时：买入股票还券，支付买入价（加上费用）
                # 注意：开仓时已经收到现金，这里只需要支付平仓买入价
                required_cash = shares * exit_px * (1.0 + tcost)
                if required_cash > cash:
                    _force_sell_longs_to_raise_cash(last_d, required_cash, exclude_symbol=sym)
                if required_cash > cash:
                    affordable = int(np.floor(cash / max(exit_px * (1.0 + tcost), 1e-12)))
                    if affordable <= 0:
                        logger.warning(f"[收官] {sym} 做空强平资金不足（需 {required_cash:.2f}，仅有 {cash:.2f}），无法强平，跳过（保留未平仓亏损）。")
                        continue
                    # 部分强平
                    cover_shares = min(shares, affordable)
                    cash -= cover_shares * exit_px * (1.0 + tcost)
                    gross_pnl_part = cover_shares * (entry_px - exit_px)
                    exit_cost_part = cover_shares * exit_px * tcost
                    entry_cost_part = cover_shares * entry_px * tcost
                    net_pnl_part = gross_pnl_part - entry_cost_part - exit_cost_part
                    profit_pct_part = (net_pnl_part / max(cover_shares * entry_px, 1e-12)) * 100.0
                    trades.append({
                        "symbol": sym,
                        "entry_date": pd.to_datetime(pos["entry_date"]),
                        "exit_date": pd.to_datetime(last_row.name if isinstance(last_row.name, pd.Timestamp) else last_d),
                        "direction": -1,
                        "shares": cover_shares,
                        "entry_price": entry_px,
                        "exit_price": exit_px,
                        "price_diff": exit_px - entry_px,
                        "gross_pnl": gross_pnl_part,
                        "net_pnl": net_pnl_part,
                        "profit_pct": profit_pct_part,
                        "open_cash_flow": float(cover_shares * entry_px * (1.0 - tcost)),
                        "close_cash_flow": float(-cover_shares * exit_px * (1.0 + tcost)),
                        "equity_before_open": float(pos.get("equity_before_open", np.nan)),
                        "equity_after_close": cash,
                        "balance_after": cash,  # 收官后只有现金
                        "entry_p_no": float(pos.get("entry_p_no", 0.0)),
                        "entry_p_up": float(pos.get("entry_p_up", 0.0)),
                        "entry_p_down": float(pos.get("entry_p_down", 0.0)),
                        "exit_p_no": float(last_row.get("p_no_prev", last_row.get("p_no_curr", 0.0)) if hasattr(last_row, "get") else 0.0),
                        "exit_p_up": float(last_row.get("p_up_prev", last_row.get("p_up_curr", 0.0)) if hasattr(last_row, "get") else 0.0),
                        "exit_p_down": float(last_row.get("p_dn_prev", last_row.get("p_dn_curr", 0.0)) if hasattr(last_row, "get") else 0.0),
                        "reason": "final_partial_cover_due_to_cash_constraint",
                    })
                    remaining = shares - cover_shares
                    if remaining > 0:
                        positions[sym]["shares"] = -int(remaining)
                    else:
                        positions[sym] = {"shares": 0, "entry_px": None, "entry_date": None, "entry_pred": None, "direction": 0}
                    continue
                cash -= required_cash  # 支付平仓买入价
                net_pnl = gross_pnl - entry_cost - exit_cost
                price_diff = exit_px - entry_px
                profit_pct = (net_pnl / max(shares * entry_px, 1e-12)) * 100.0
                # 获取最后一日的前一日概率（如果存在）
                exit_p_no = float(last_row.get("p_no_prev", last_row.get("p_no_curr", 0.0)) if hasattr(last_row, "get") else 0.0)
                exit_p_up = float(last_row.get("p_up_prev", last_row.get("p_up_curr", 0.0)) if hasattr(last_row, "get") else 0.0)
                exit_p_down = float(last_row.get("p_dn_prev", last_row.get("p_dn_curr", 0.0)) if hasattr(last_row, "get") else 0.0)
                trades.append({
                    "symbol": sym,
                    "entry_date": pd.to_datetime(pos["entry_date"]),
                    "exit_date": pd.to_datetime(last_row.name if isinstance(last_row.name, pd.Timestamp) else last_d),
                    "direction": -1,
                    "shares": shares,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "price_diff": price_diff,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "profit_pct": profit_pct,
                    "open_cash_flow": float(pos.get("entry_cash_flow", shares * entry_px * (1.0 - tcost))),
                    "close_cash_flow": float(-shares * exit_px * (1.0 + tcost)),
                    "equity_before_open": float(pos.get("equity_before_open", np.nan)),
                    "equity_after_close": cash,
                    "balance_after": cash,  # 收官后只有现金
                    "entry_p_no": float(pos.get("entry_p_no", 0.0)),
                    "entry_p_up": float(pos.get("entry_p_up", 0.0)),
                    "entry_p_down": float(pos.get("entry_p_down", 0.0)),
                    "exit_p_no": exit_p_no,
                    "exit_p_up": exit_p_up,
                    "exit_p_down": exit_p_down,
                })
    # 创建交易DataFrame，如果为空则创建包含必要列的空的DataFrame
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
    else:
        # 如果没有交易，创建空的DataFrame，包含所有必要的列
        trades_df = pd.DataFrame(columns=[
            "symbol", "entry_date", "exit_date", "direction", "shares",
            "entry_price", "exit_price", "price_diff", "gross_pnl", "net_pnl",
            "profit_pct", "open_cash_flow", "close_cash_flow",
            "equity_before_open", "equity_after_close", "balance_after",
            "entry_p_no", "entry_p_up", "entry_p_down",
            "exit_p_no", "exit_p_up", "exit_p_down",
            "reason",
        ])
    # 将每日余额合并到交易记录中（按日期匹配）
    if len(daily_balances) > 0:
        daily_df = pd.DataFrame(daily_balances)
        # 为每笔交易添加当日余额（使用exit_date匹配）
        if len(trades_df) > 0:
            trades_df = trades_df.merge(
                daily_df[["date", "balance"]].rename(columns={"date": "exit_date", "balance": "daily_balance"}),
                on="exit_date",
                how="left"
            )
            # 如果合并失败，使用balance_after
            trades_df["daily_balance"] = trades_df["daily_balance"].fillna(trades_df.get("balance_after", cash))
        else:
            # 如果没有交易，直接输出每日余额
            trades_df = daily_df.rename(columns={"balance": "daily_balance"})
    else:
        # 如果没有每日余额记录，使用balance_after
        if len(trades_df) > 0:
            trades_df["daily_balance"] = trades_df.get("balance_after", cash)
    # 以开仓时间排序
    if len(trades_df) > 0 and "entry_date" in trades_df.columns:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    summaries = []
    if len(trades_df) > 0 and "symbol" in trades_df.columns:
        for sym, g in trades_df.groupby("symbol"):
            # 分别计算做多和做空的PnL
            long_trades = g[g["direction"] == 1]
            short_trades = g[g["direction"] == -1]
            long_pnl = float(long_trades["net_pnl"].sum()) if len(long_trades) > 0 else 0.0
            short_pnl = float(short_trades["net_pnl"].sum()) if len(short_trades) > 0 else 0.0
            total_pnl = float(g["net_pnl"].sum())
            
            summaries.append({
                "symbol": sym,
                "num_trades": int(len(g)),
                "num_long": int(len(long_trades)),
                "num_short": int(len(short_trades)),
                "long_pnl": long_pnl,
                "short_pnl": short_pnl,
                "total_net_pnl": total_pnl,
                "win_rate": float((g["net_pnl"] > 0).mean()),
                "avg_net_pnl": float(g["net_pnl"].mean()),
            })
    # 如果没有交易，创建空的DataFrame，包含所有列
    if len(summaries) == 0:
        summary_df = pd.DataFrame(columns=["symbol", "num_trades", "num_long", "num_short", "long_pnl", "short_pnl", "total_net_pnl", "win_rate", "avg_net_pnl"])
    else:
        summary_df = pd.DataFrame(summaries).sort_values("total_net_pnl", ascending=False)
    daily_positions_df = pd.DataFrame(daily_positions) if len(daily_positions) > 0 else pd.DataFrame(columns=["date", "symbol", "direction", "shares", "price", "market_value"])
    daily_balances_df = pd.DataFrame(daily_balances) if len(daily_balances) > 0 else pd.DataFrame(columns=["date", "balance", "cash", "total_market_value"])
    
    # 计算夏普比
    sharpe_ratio = None
    if len(daily_balances_df) > 1:
        try:
            daily_balances_df = daily_balances_df.sort_values("date")
            daily_returns = daily_balances_df["balance"].pct_change().dropna()
            if len(daily_returns) > 0:
                # 年化收益率和波动率（假设252个交易日）
                annual_return = daily_returns.mean() * 252
                annual_vol = daily_returns.std() * np.sqrt(252)
                if annual_vol > 0:
                    sharpe_ratio = annual_return / annual_vol
                else:
                    sharpe_ratio = 0.0
        except Exception as e:
            logger.warning(f"计算夏普比失败: {e}")
    
    return trades_df, summary_df, cash, stopped_early, daily_positions_df, daily_balances_df, sharpe_ratio


def main():
    parser = argparse.ArgumentParser(description="简易实战模拟（现货，只做多，按置信度分配资金比例）")
    parser.add_argument("--symbols", type=str, default="", help="标的逗号分隔；为空使用全部")
    parser.add_argument("--start_date", type=str, default="", help="开始日期YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default="", help="结束日期YYYY-MM-DD")
    parser.add_argument("--up_thr", type=float, default=None, help="上涨趋势阈值，默认取config或0.55")
    parser.add_argument("--alloc_cap", type=float, default=None, help="当日总资金使用上限，默认取config或1.0")
    parser.add_argument("--per_symbol_cap", type=float, default=None, help="单标的资金上限，默认取config或0.25")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = None
    up_thr = args.up_thr if args.up_thr is not None else float(getattr(config, "CLASS_PROB_THRESHOLD_UP_TREND", 0.55))
    alloc_cap = args.alloc_cap if args.alloc_cap is not None else float(getattr(config, "STRAT_ALLOC_CAP", 1.0))
    per_symbol_cap = args.per_symbol_cap if args.per_symbol_cap is not None else float(getattr(config, "STRAT_PER_SYMBOL_CAP", 0.25))

    fee_bps = float(getattr(config, "STRAT_FEE_BPS", 2.0))
    slippage_bps = float(getattr(config, "STRAT_SLIPPAGE_BPS", 1.0))
    initial_capital = float(getattr(config, "STRAT_INITIAL_CAPITAL", 100000.0))  # 100k

    df_all = load_all_with_prediction(project_root, symbols=symbols)
    allow_long = bool(getattr(config, 'STRAT_ALLOW_LONG', True))
    allow_short = bool(getattr(config, 'STRAT_ALLOW_SHORT', True))
    trades_df, summary_df, final_cash, stopped_early, daily_positions_df, daily_balances_df, sharpe_ratio = simulate(
        df_all=df_all,
        up_thr=up_thr,
        alloc_cap=alloc_cap,
        per_symbol_cap=per_symbol_cap,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        start_date=args.start_date,
        end_date=args.end_date,
        allow_long=allow_long,
        allow_short=allow_short,
    )
    out_dir = os.path.join(project_root, "models", "result", "trading_result")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = os.path.join(out_dir, f"trade_sim_trades_{ts}.csv")
    try:
        if 'entry_date' in trades_df.columns:
            trades_df = trades_df.sort_values('entry_date').reset_index(drop=True)
    except Exception:
        pass
    desired = [
        "entry_date", "exit_date", "symbol", "entry_price", "exit_price",
        "price_diff", "shares", "open_cash_flow", "close_cash_flow",
        "net_pnl", "profit_pct", "equity_before_open", "equity_after_close",
        "entry_p_no", "entry_p_up", "entry_p_down",
        "exit_p_no", "exit_p_up", "exit_p_down"
    ]
    cols = [c for c in desired if c in trades_df.columns] + [c for c in trades_df.columns if c not in desired]
    trades_df = trades_df[cols]
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    sum_path = os.path.join(out_dir, f"trade_sim_summary_{ts}.csv")
    summary_df.to_csv(sum_path, index=False, encoding="utf-8-sig")
    # 输出每日持仓明细和余额
    positions_path = os.path.join(out_dir, f"trade_sim_daily_positions_{ts}.csv")
    daily_positions_df.to_csv(positions_path, index=False, encoding="utf-8-sig")
    balances_path = os.path.join(out_dir, f"trade_sim_daily_balances_{ts}.csv")
    daily_balances_df.to_csv(balances_path, index=False, encoding="utf-8-sig")
    status_msg = "（提前停止：余额<0）" if stopped_early else ""
    total_return = (final_cash - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0.0
    logger.info(f"完成：final_cash={final_cash:.2f}, trades={len(trades_df)} {status_msg}")
    logger.info(f"总收益率：{total_return:.2f}%")
    if sharpe_ratio is not None:
        logger.info(f"夏普比：{sharpe_ratio:.4f}")
    logger.info(f"交易明细：{trades_path}")
    logger.info(f"汇总：{sum_path}")
    logger.info(f"每日持仓明细：{positions_path}")
    logger.info(f"每日余额：{balances_path}")


if __name__ == "__main__":
    main()


