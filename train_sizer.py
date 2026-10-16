"""
下注尺寸模型训练（简版，回归）
- 训练目标：预测次日开盘收益 r_{t+1} = open_{t+1}/open_t - 1
- 特征：优先使用 mdl_* 与少量已存在的技术指标（缺失填0）
- 模型：XGBoost 回归器
- 输出：models/sizer/sizer.pkl（包含 model 与 feature_names）
"""
import os
import sys
import argparse
import logging
import pickle
from typing import List

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


DEFAULT_FEATURES = [
    "mdl_p_no_trend", "mdl_p_up", "mdl_p_down",
    "mdl_margin", "mdl_signal_strength",
    "trend_position_weight",
    "R2_5D", "R2_10D",
    "ReturnAccel10_5", "MA5Slope", "MA20Slope",
]


def load_with_pred(project_root: str, symbol: str) -> pd.DataFrame:
    path = os.path.join(project_root, "data", "with_prediction", f"{symbol}_with_prediction.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "date" not in df.columns or "open" not in df.columns:
        raise ValueError(f"{symbol} 缺少必要列(date/open)")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_dataset(dfs: List[pd.DataFrame], horizon: int, feature_names: List[str]) -> (pd.DataFrame, pd.Series):
    X_list, y_list = [], []
    for df in dfs:
        if len(df) <= horizon:
            continue
        df_feat = df.copy()
        df_feat["r_next"] = (df_feat["open"].shift(-horizon) / df_feat["open"] - 1.0)
        # 丢尾部NaN
        df_feat = df_feat.iloc[:-horizon]
        X = df_feat[feature_names].copy()
        X = X.fillna(0.0)
        y = df_feat["r_next"].astype(float)
        X_list.append(X)
        y_list.append(y)
    if len(X_list) == 0:
        raise ValueError("没有足够数据用于训练下注尺寸模型")
    X_all = pd.concat(X_list, ignore_index=True)
    y_all = pd.concat(y_list, ignore_index=True)
    return X_all, y_all


def train_sizer(symbols: List[str], feature_names: List[str], horizon: int):
    project_root = os.path.dirname(os.path.abspath(__file__))
    dfs = []
    for sym in symbols:
        try:
            df = load_with_pred(project_root, sym)
            # 确保特征列存在
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            dfs.append(df)
        except Exception as e:
            logger.warning(f"{sym} 载入失败，跳过：{e}")
    if len(dfs) == 0:
        raise ValueError("未载入到任何标的数据")
    X, y = build_dataset(dfs, horizon=horizon, feature_names=feature_names)
    logger.info(f"训练集样本数: {len(X)}，特征数: {X.shape[1]}")
    model = XGBRegressor(
        max_depth=3,
        n_estimators=600,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=2025,
        n_jobs=4,
    )
    model.fit(X, y)
    # 简单评估
    pred = model.predict(X)
    mse = float(np.mean((pred - y.values) ** 2))
    mae = float(np.mean(np.abs(pred - y.values)))
    logger.info(f"训练结束：MSE={mse:.6e}, MAE={mae:.6e}")
    # 保存
    out_dir = os.path.join(project_root, "models", "sizer")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sizer.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    logger.info(f"已保存下注尺寸模型: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="下注尺寸模型训练（回归，预测次日开盘收益）")
    parser.add_argument("--symbols", type=str, default="", help="标的逗号分隔；为空使用config.SYMBOLS")
    parser.add_argument("--features", type=str, default="", help="自定义特征列逗号分隔；默认使用内置集合")
    parser.add_argument("--horizon", type=int, default=None, help="预测未来天数，默认取config或1")
    args = parser.parse_args()

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = [item["symbol"] for item in getattr(config, "SYMBOLS", [])]
    if not symbols:
        raise ValueError("未指定标的")
    feat_names = [c.strip() for c in args.features.split(",") if c.strip()] if args.features.strip() else DEFAULT_FEATURES
    horizon = int(args.horizon if args.horizon is not None else getattr(config, "STRAT_SIZER_TARGET_HORIZON", 1))
    logger.info(f"训练标的: {symbols}")
    logger.info(f"特征列: {feat_names}")
    logger.info(f"horizon: {horizon}")
    train_sizer(symbols, feat_names, horizon)


if __name__ == "__main__":
    main()


