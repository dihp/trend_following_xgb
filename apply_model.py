"""
应用阶段：从 apply 目录加载已训练模型，对指定标的使用 processed 数据进行预测
输出预测到 models/result 目录
"""

import os
import sys
import glob
import json
import pickle
import shutil
from datetime import datetime
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# 与 data_collector 保持一致的 EventKit 修复
try:
    import eventkit_fix  # noqa: F401
except Exception:
    pass

# 可选：IBKR持仓读取（若未安装或未连接会自动跳过）
try:
    from ib_insync import IB
except Exception:
    IB = None

import config
from trend_model.train_xgboost import TrendXGBoostTrainer

# 日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_latest_model_from_apply(apply_dir):
    """
    从 apply/model 目录加载最新的 .pkl 模型（包含feature_names与encoders）
    如果模型名称包含 long_only 或 short_only，自动寻找配对模型
    返回: dict(model, feature_names, encoders) 或包含两个模型的 bundle
    """
    # 优先从 apply/model 目录读取，如果不存在则回退到 apply 目录
    apply_model_dir = os.path.join(apply_dir, 'model')
    if not os.path.exists(apply_model_dir):
        apply_model_dir = apply_dir
        logger.info(f"apply/model 目录不存在，使用 {apply_dir} 目录")
    else:
        logger.info(f"从 apply/model 目录加载模型")
    
    if not os.path.exists(apply_model_dir):
        raise FileNotFoundError(f"未找到apply目录: {apply_model_dir}")
    
    candidates = sorted(glob.glob(os.path.join(apply_model_dir, "*.pkl")), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"apply目录中未找到.pkl模型文件: {apply_model_dir}")
    
    model_path = candidates[0]
    model_name = os.path.basename(model_path)
    
    # 兼容旧包：Stage2OVR 在不同模块名下保存导致无法反序列化
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'Stage2OVR':
                try:
                    from trend_model.train_xgboost import Stage2OVR as C
                    return C
                except Exception:
                    pass
            return super().find_class(module, name)
    
    def _load_model_file(path):
        """加载单个模型文件"""
        with open(path, 'rb') as f:
            try:
                data = _CompatUnpickler(f).load()
            except Exception:
                f.seek(0)
                data = pickle.load(f)
        if 'feature_names' not in data:
            raise ValueError(f"模型文件不包含必要字段(feature_names): {path}")
        return data

    # 加载主模型
    data = _load_model_file(model_path)
    logger.info(f"加载主模型: {model_path}")
    
    # 检测模型类型（兼容 long_only / longonly 命名）
    name_lower = model_name.lower()
    is_long_only = ('long_only' in name_lower) or ('longonly' in name_lower)
    is_short_only = ('short_only' in name_lower) or ('shortonly' in name_lower)
    
    # 尝试寻找配对模型
    paired_model_path = None
    paired_data = None
    if is_long_only:
        # 寻找 short_only / shortonly 模型
        short_candidates = [
            p for p in candidates
            if ('short_only' in os.path.basename(p).lower()) or ('shortonly' in os.path.basename(p).lower())
        ]
        if short_candidates:
            paired_model_path = short_candidates[0]
            paired_data = _load_model_file(paired_model_path)
            logger.info(f"找到配对模型（short_only）: {paired_model_path}")
        else:
            logger.info("未找到 short_only 配对模型，将仅使用 long_only 模型")
    elif is_short_only:
        # 寻找 long_only / longonly 模型
        long_candidates = [
            p for p in candidates
            if ('long_only' in os.path.basename(p).lower()) or ('longonly' in os.path.basename(p).lower())
        ]
        if long_candidates:
            paired_model_path = long_candidates[0]
            paired_data = _load_model_file(paired_model_path)
            logger.info(f"找到配对模型（long_only）: {paired_model_path}")
        else:
            logger.info("未找到 long_only 配对模型，将仅使用 short_only 模型")
    
    # 构建 bundle
    # 优先使用文件名检测结果，如果模型数据中也有标志则使用数据中的（更可靠）
    # 但如果文件名明确标识了类型，即使数据中没有，也应该使用文件名检测的结果
    data_long_only = data.get('long_only', False)
    data_short_only = data.get('short_only', False)
    # 如果文件名检测到类型，但数据中没有，使用文件名检测的结果
    if is_long_only and not data_long_only and not data_short_only:
        data_long_only = True
    if is_short_only and not data_short_only and not data_long_only:
        data_short_only = True
    
    bundle = {
        'model_path': model_path,
        'feature_names': data['feature_names'],
        'encoders': data.get('encoders', None),
        'calibration': data.get('calibration', None),
        'timestamp': data.get('timestamp', 'unknown'),
        'two_stage': data.get('two_stage', False),
        'long_only': data_long_only,
        'short_only': data_short_only,
        'has_paired_model': paired_data is not None,
    }
    
    if bundle['two_stage']:
        bundle['model_stage1'] = data.get('model_stage1')
        if bundle['long_only']:
            bundle['model_stage2_up'] = data.get('model_stage2_up')
            if bundle['model_stage1'] is None or bundle['model_stage2_up'] is None:
                raise ValueError("长多两阶段模型缺少stage1或stage2_up")
        elif bundle['short_only']:
            bundle['model_stage2_down'] = data.get('model_stage2_down')
            if bundle['model_stage1'] is None or bundle['model_stage2_down'] is None:
                raise ValueError("短空两阶段模型缺少stage1或stage2_down")
        else:
            bundle['model_stage2'] = data.get('model_stage2')
            if bundle['model_stage1'] is None or bundle['model_stage2'] is None:
                raise ValueError("两阶段模型缺少stage1或stage2")
    else:
        bundle['model'] = data['model']
    
    # 如果有配对模型，也加载它
    if paired_data is not None:
        bundle['paired_model_path'] = paired_model_path
        bundle['paired_long_only'] = paired_data.get('long_only', False)
        bundle['paired_short_only'] = paired_data.get('short_only', False)
        if paired_data.get('two_stage', False):
            bundle['paired_model_stage1'] = paired_data.get('model_stage1')
            if bundle['paired_long_only']:
                bundle['paired_model_stage2_up'] = paired_data.get('model_stage2_up')
            elif bundle['paired_short_only']:
                bundle['paired_model_stage2_down'] = paired_data.get('model_stage2_down')
            else:
                bundle['paired_model_stage2'] = paired_data.get('model_stage2')
        else:
            bundle['paired_model'] = paired_data.get('model')
        logger.info(f"已加载配对模型，将合并两个模型的输出")
    
    logger.info(f"模型两阶段: {bundle['two_stage']}, 长多: {bundle['long_only']}, 短空: {bundle['short_only']}, 有配对: {bundle['has_paired_model']}")
    return bundle


def attach_encoders_to_trainer(trainer: TrendXGBoostTrainer, encoders):
    """
    将保存的LabelEncoder附加回训练器，用于一致的类别编码。
    兼容 None 与缺失情形。
    """
    if encoders is None:
        logger.warning("模型中未保存encoders，将使用默认编码逻辑（可能与训练时有偏差）")
        return
    try:
        sector_enc = encoders.get('sector_encoder', None)
        industry_enc = encoders.get('industry_encoder', None)
        sub_industry_enc = encoders.get('sub_industry_encoder', None)
        if sector_enc is not None:
            trainer.sector_encoder = sector_enc
        if industry_enc is not None:
            trainer.industry_encoder = industry_enc
        if sub_industry_enc is not None:
            trainer.sub_industry_encoder = sub_industry_enc
        logger.info("已加载并附加LabelEncoders（sector/industry/sub_industry）")
    except Exception as e:
        logger.warning(f"附加encoders失败，将继续但编码可能不一致: {e}")


def align_features(X: pd.DataFrame, feature_names):
    """
    将X的列对齐到feature_names：
    - 缺失列补0
    - 多余列删除
    - 按顺序重排
    """
    X = X.copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    return X


def build_inference_features(df: pd.DataFrame, trainer: TrendXGBoostTrainer) -> pd.DataFrame:
    """
    仅用于推理的特征构建（不依赖label，不过滤样本）
    与训练阶段保持同样的工程步骤：行业编码 + 时间特征 + 列排除
    """
    df = df.copy()
    # 行业特征
    if getattr(config, 'USE_INDUSTRY_FEATURES', True):
        df = trainer.add_industry_features(df)
        df = trainer.encode_categorical_features(df, fit_encoders=False)
    # 时间特征
    if getattr(config, 'USE_TIME_FEATURES', True) and 'date' in df.columns:
        df = trainer.add_time_features(df)
    # 排除列
    exclude_columns = ['date', 'symbol', 'label', 'label_old', 'processed_label', 'trend_position_weight']
    exclude_columns += ['NormalizedReturn', 'Slope', 'R2', 'TNR']
    if not getattr(config, 'USE_RAW_OHLCV', True):
        exclude_columns += ['open', 'high', 'low', 'close', 'volume']
    # 组装特征
    feature_columns = [c for c in df.columns if c not in exclude_columns]
    X = df[feature_columns].copy()
    return X


def load_processed_df(project_root, symbol):
    """
    读取 processed 数据: data/processed/{SYMBOL}_processed.csv
    """
    processed_path = os.path.join(project_root, 'data', 'processed', f"{symbol}_processed.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"未找到processed数据: {processed_path}")
    df = pd.read_csv(processed_path)
    # 若缺少symbol列，填充为当前symbol，便于行业特征与输出
    if 'symbol' not in df.columns:
        df['symbol'] = symbol
        logger.warning("数据中缺少'symbol'列，已自动填充为当前标的")
    logger.info(f"加载processed数据: {processed_path}, 行数={len(df)}")
    return df


def predict_for_symbol(model_bundle, trainer, df_raw, feature_names, use_3_class=True, symbol_for_fallback=None):
    """
    对单个标的的processed数据进行预测，返回包含日期/预测/概率的DataFrame
    """
    # 构建推理特征（不依赖标签，不过滤样本）
    X = build_inference_features(df_raw, trainer)
    # 对齐列
    X = align_features(X, feature_names)
    # 加载阈值（支持两阶段与三分类）
    project_root = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(project_root, 'models', 'result')
    thr = None
    try:
        import glob
        cands = []
        cands += glob.glob(os.path.join(result_dir, "class_thresholds_two_stage_*.json"))
        cands += glob.glob(os.path.join(result_dir, "class_thresholds_*.json"))
        cands = sorted(cands, key=os.path.getmtime, reverse=True)
        if cands:
            with open(cands[0], 'r', encoding='utf-8') as f:
                thr = json.load(f)
    except Exception:
        thr = None
    
    two_stage = bool(model_bundle.get('two_stage', False))
    has_paired = bool(model_bundle.get('has_paired_model', False))
    
    # 预测概率
    if two_stage:
        # 检查是否有配对模型（long_only + short_only）
        if has_paired:
            # 使用两个模型的输出合并
            model_stage1 = model_bundle['model_stage1']
            paired_model_stage1 = model_bundle.get('paired_model_stage1')
            
            # Stage1: 使用主模型的 Stage1（或取平均）
            proba1 = model_stage1.predict_proba(X)  # [:,1] = P(trend)
            p_trend = proba1[:, 1]
            if paired_model_stage1 is not None:
                # 取两个 Stage1 的平均
                proba1_paired = paired_model_stage1.predict_proba(X)
                p_trend = (p_trend + proba1_paired[:, 1]) / 2.0
                logger.info("使用两个模型的 Stage1 平均作为 P(trend)")
            
            # Stage2: 根据模型类型获取对应的 Stage2
            if model_bundle.get('long_only', False):
                # 主模型是 long_only，配对模型是 short_only
                model_stage2_up = model_bundle.get('model_stage2_up')
                paired_model_stage2_down = model_bundle.get('paired_model_stage2_down')
                if model_stage2_up is None or paired_model_stage2_down is None:
                    raise ValueError("配对模型缺少必要的 Stage2 模型")
                proba2_up = model_stage2_up.predict_proba(X)  # [:,1] = P(up | trend)
                proba2_down = paired_model_stage2_down.predict_proba(X)  # [:,1] = P(down | trend)
                p_up_cond = proba2_up[:, 1]
                p_down_cond = proba2_down[:, 1]
                logger.info("合并 long_only 和 short_only 模型的输出")
            elif model_bundle.get('short_only', False):
                # 主模型是 short_only，配对模型是 long_only
                model_stage2_down = model_bundle.get('model_stage2_down')
                paired_model_stage2_up = model_bundle.get('paired_model_stage2_up')
                if model_stage2_down is None or paired_model_stage2_up is None:
                    raise ValueError("配对模型缺少必要的 Stage2 模型")
                proba2_down = model_stage2_down.predict_proba(X)  # [:,1] = P(down | trend)
                proba2_up = paired_model_stage2_up.predict_proba(X)  # [:,1] = P(up | trend)
                p_down_cond = proba2_down[:, 1]
                p_up_cond = proba2_up[:, 1]
                logger.info("合并 short_only 和 long_only 模型的输出")
            else:
                raise ValueError("配对模型但无法确定模型类型")
            
            # 计算无条件概率
            p_up = p_trend * p_up_cond
            p_down = p_trend * p_down_cond
            p_no = 1.0 - p_trend
        else:
            # 单个模型（可能是 long_only 或 short_only 或标准两阶段）
            model_stage1 = model_bundle['model_stage1']
            proba1 = model_stage1.predict_proba(X)  # [:,1] = P(trend)
            p_trend = proba1[:, 1]
            
            if model_bundle.get('long_only', False):
                # 长多模型：只有 Stage2_up
                model_stage2_up = model_bundle.get('model_stage2_up')
                if model_stage2_up is None:
                    raise ValueError("长多模型缺少 model_stage2_up")
                proba2_up = model_stage2_up.predict_proba(X)  # [:,1] = P(up | trend)
                p_up_cond = proba2_up[:, 1]
                p_up = p_trend * p_up_cond
                p_down = p_trend * (1.0 - p_up_cond)
                p_no = 1.0 - p_trend
            elif model_bundle.get('short_only', False):
                # 短空模型：只有 Stage2_down
                model_stage2_down = model_bundle.get('model_stage2_down')
                if model_stage2_down is None:
                    raise ValueError("短空模型缺少 model_stage2_down")
                proba2_down = model_stage2_down.predict_proba(X)  # [:,1] = P(down | trend)
                p_down_cond = proba2_down[:, 1]
                p_down = p_trend * p_down_cond
                p_up = p_trend * (1.0 - p_down_cond)
                p_no = 1.0 - p_trend
            else:
                # 标准两阶段模型
                model_stage2 = model_bundle.get('model_stage2')
                if model_stage2 is None:
                    raise ValueError("两阶段模型缺少 model_stage2")
                proba2 = model_stage2.predict_proba(X)  # [:,1] = P(down)
                p_down_cond = proba2[:, 1]
                p_up_cond = 1.0 - p_down_cond
                p_up = p_trend * p_up_cond
                p_down = p_trend * p_down_cond
                p_no = 1.0 - p_trend
        # 阈值（两阶段结构）
        stage1_thr_global = float(getattr(config, 'PREDICTION_CONFIDENCE_THRESHOLD', 0.6))
        stage2_thr_up_global = float(getattr(config, 'CLASS_PROB_THRESHOLD_UP_TREND', 0.55))
        stage2_thr_down_global = float(getattr(config, 'CLASS_PROB_THRESHOLD_DOWN_TREND', 0.60))
        per_symbol_s1 = {}
        per_symbol_s2 = {}
        if isinstance(thr, dict) and thr.get('two_stage', False):
            s1 = thr.get('stage1', {})
            s2 = thr.get('stage2', {})
            stage1_thr_global = s1.get('global', {}).get('trend', stage1_thr_global)
            stage2_thr_up_global = s2.get('global', {}).get('up_trend', stage2_thr_up_global)
            stage2_thr_down_global = s2.get('global', {}).get('down_trend', stage2_thr_down_global)
            per_symbol_s1 = s1.get('per_symbol', {}) or {}
            per_symbol_s2 = s2.get('per_symbol', {}) or {}
        # 组合预测
        preds = []
        syms = df_raw['symbol'].values if 'symbol' in df_raw.columns else np.array([symbol_for_fallback] * len(X))
        for i in range(len(X)):
            sym = str(syms[i])
            t1 = float(per_symbol_s1.get(sym, stage1_thr_global))
            s2_local = per_symbol_s2.get(sym, {})
            tu = float(s2_local.get('up_trend', stage2_thr_up_global))
            td = float(s2_local.get('down_trend', stage2_thr_down_global))
            if p_trend[i] < t1:
                preds.append(0)
            else:
                candidates = []
                if p_up[i] >= tu:
                    candidates.append((1, p_up[i]))
                if p_down[i] >= td:
                    candidates.append((2, p_down[i]))
                if len(candidates) == 0:
                    preds.append(1 if p_up[i] >= p_down[i] else 2)
                else:
                    preds.append(max(candidates, key=lambda t: t[1])[0])
        y_pred = np.array(preds, dtype=int)
        proba = np.vstack([p_no, p_up, p_down]).T  # 便于输出
    else:
        # 单模型（三分类或二分类）
        model = model_bundle['model']
        calibration = model_bundle.get('calibration')
        # 预测概率（支持温度缩放校准）
        use_calib = getattr(config, 'USE_PROB_CALIBRATION', False) and isinstance(calibration, dict) and calibration.get('method') == 'temperature'
        if use_calib:
            T = float(calibration.get('T', 1.0))
            try:
                raw = model.predict(X, output_margin=True)
                if use_3_class:
                    z = np.asarray(raw, dtype=float)
                    z_max = np.max(z, axis=1, keepdims=True)
                    exp_z = np.exp((z - z_max) / max(T, 1e-6))
                    proba = exp_z / np.clip(np.sum(exp_z, axis=1, keepdims=True), 1e-12, None)
                else:
                    logit = np.asarray(raw, dtype=float)
                    p1 = 1.0 / (1.0 + np.exp(-logit / max(T, 1e-6)))
                    p0 = 1.0 - p1
                    proba = np.vstack([p0, p1]).T
            except Exception:
                proba = model.predict_proba(X)
        else:
            proba = model.predict_proba(X)
        # 时间平滑（单标的顺序EMA）
        if getattr(config, 'APPLY_SMOOTHING_IN_INFERENCE', True) and getattr(config, 'USE_TEMPORAL_SMOOTHING', False):
            alpha = float(getattr(config, 'SMOOTHING_EMA_ALPHA', 0.3))
            proba_sm = np.empty_like(proba, dtype=float)
            proba_sm[0] = proba[0]
            for i in range(1, len(proba)):
                proba_sm[i] = alpha * proba[i] + (1 - alpha) * proba_sm[i - 1]
            proba = proba_sm
        # 应用阈值（保持原三分类逻辑）
        use_threshold = getattr(config, 'USE_PROBABILITY_THRESHOLD', True)
        thr_global = getattr(config, 'PREDICTION_CONFIDENCE_THRESHOLD', 0.6)
        use_class_thresh = getattr(config, 'USE_CLASS_SPECIFIC_THRESHOLDS', False) or (isinstance(thr, dict) and not thr.get('two_stage', False))
        if use_3_class and use_threshold:
            if use_class_thresh:
                if isinstance(thr, dict) and not thr.get('two_stage', False):
                    thr0 = float(thr.get('no_trend', thr_global))
                    thr1 = float(thr.get('up_trend', thr_global))
                    thr2 = float(thr.get('down_trend', thr_global))
                else:
                    thr0 = getattr(config, 'CLASS_PROB_THRESHOLD_NO_TREND', thr_global)
                    thr1 = getattr(config, 'CLASS_PROB_THRESHOLD_UP_TREND', thr_global)
                    thr2 = getattr(config, 'CLASS_PROB_THRESHOLD_DOWN_TREND', thr_global)
                preds = []
                for p0, p1, p2 in proba:
                    candidates = []
                    if p0 >= thr0:
                        candidates.append((0, p0))
                    if p1 >= thr1:
                        candidates.append((1, p1))
                    if p2 >= thr2:
                        candidates.append((2, p2))
                    if len(candidates) == 0:
                        preds.append(0)
                    else:
                        preds.append(max(candidates, key=lambda t: t[1])[0])
                y_pred = np.array(preds, dtype=int)
            else:
                max_probs = np.max(proba, axis=1)
                max_classes = np.argmax(proba, axis=1)
                y_pred = np.where(max_probs >= thr_global, max_classes, 0)
        else:
            y_pred = model.predict(X)
    # 组装输出
    # 对齐原始索引，确保长度一致
    idx = X.index
    if 'date' in df_raw.columns:
        dates = pd.to_datetime(df_raw.loc[idx, 'date'])
    else:
        dates = pd.Series([pd.NaT] * len(idx), index=idx)
    if 'symbol' in df_raw.columns:
        symbols_series = df_raw.loc[idx, 'symbol']
    else:
        fallback_symbol = symbol_for_fallback or ''
        symbols_series = pd.Series([fallback_symbol] * len(idx), index=idx)
    out = pd.DataFrame({
        'date': dates.values,
        'symbol': symbols_series.values,
        'pred': y_pred
    })
    # 加入概率列
    if proba.shape[1] == 3:
        out['p_no_trend'] = proba[:, 0]
        out['p_up'] = proba[:, 1]
        out['p_down'] = proba[:, 2]
    else:
        out['p_no_trend'] = proba[:, 0]
        out['p_trend'] = proba[:, 1]
    return out


def main():
    parser = argparse.ArgumentParser(description="应用模型进行预测")
    parser.add_argument("--symbols", type=str, default="",
                        help="要预测的标的，逗号分隔。为空表示使用config.SYMBOLS。例：--symbols AAPL,AMD")
    parser.add_argument("--apply_dir", type=str, default="apply",
                        help="存放待应用模型的目录（默认: 项目根目录/apply）")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    apply_dir = os.path.join(project_root, args.apply_dir) if not os.path.isabs(args.apply_dir) else args.apply_dir
    
    def _extract_trend_confidence(row, use_3_class=True):
        """返回三项概率字典以及当前预测类别的置信度"""
        pred = row.get('pred', None)
        if use_3_class:
            p_no = float(row.get('p_no_trend', np.nan))
            p_up = float(row.get('p_up', np.nan))
            p_down = float(row.get('p_down', np.nan))
            if pred == 1:
                return p_up, {'p_no_trend': p_no, 'p_up': p_up, 'p_down': p_down}
            if pred == 2:
                return p_down, {'p_no_trend': p_no, 'p_up': p_up, 'p_down': p_down}
            return p_no, {'p_no_trend': p_no, 'p_up': p_up, 'p_down': p_down}
        else:
            p_no = float(row.get('p_no_trend', np.nan))
            p_trend = float(row.get('p_trend', np.nan))
            if pred == 1:
                return p_trend, {'p_no_trend': p_no, 'p_trend': p_trend}
            return p_no, {'p_no_trend': p_no, 'p_trend': p_trend}

    # 加载模型
    model_bundle = load_latest_model_from_apply(apply_dir)
    feature_names = model_bundle['feature_names']
    encoders = model_bundle.get('encoders')
    
    # 检测三分类开关（两阶段视为三分类输出）
    use_3_class = True
    if not model_bundle.get('two_stage', False):
        try:
            model = model_bundle['model']
            n_classes = getattr(model, 'n_classes_', 3)
            use_3_class = (n_classes == 3)
        except Exception:
            pass
    logger.info(f"推理模式: {'两阶段' if model_bundle.get('two_stage', False) else ('三分类' if use_3_class else '二分类')}")
    
    # 初始化特征准备器，并附加编码器
    trainer = TrendXGBoostTrainer()
    attach_encoders_to_trainer(trainer, encoders)
    
    # 确定标的列表
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        # 使用config中的标的
        symbols = [item['symbol'] for item in getattr(config, 'SYMBOLS', [])]
    if not symbols:
        raise ValueError("未指定任何标的，也未能从config.SYMBOLS获取到标的列表")
    logger.info(f"将对以下标的进行预测: {symbols}")
    
    # 结果输出目录
    result_dir = os.path.join(project_root, 'models', 'result')
    predictions_dir = os.path.join(result_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    high_confidence_records = []
    all_trend_records = []  # 所有有趋势的标的（不管置信度高低）
    trend_conf_thr = float(getattr(config, 'APPLY_TREND_CONFIDENCE_THRESHOLD', 0.85))
    # 当前最新一日的全量预测（即便不满足高置信度也记录，用于反转/取消对比）
    current_latest_map = {}
    # 不再对比上一版高置信度列表（去掉反转/取消提醒）
    
    # 加载测试集每标的准确度（用于在结果中显示）
    # 1) 通用三分类/二分类模型的 per_symbol_accuracy_test_*.csv
    test_accuracy_map = {}
    test_accuracy_source_file = None  # 记录通用accuracy来源文件
    # 2) 长多/短空专用模型的 per_symbol_accuracy_test_longonly_*.csv / per_symbol_accuracy_test_shortonly_*.csv
    longonly_accuracy_map = {}
    longonly_accuracy_source_file = None
    shortonly_accuracy_map = {}
    shortonly_accuracy_source_file = None
    # 3) Stage1模型的 per_symbol_accuracy（分别加载 long_only 和 short_only 的 stage1 accuracy）
    stage1_longonly_accuracy_map = {}
    stage1_longonly_accuracy_source_file = None
    stage1_shortonly_accuracy_map = {}
    stage1_shortonly_accuracy_source_file = None
    try:
        # 查找最新的测试集准确度文件（通用）
        test_acc_files = sorted(
            glob.glob(os.path.join(result_dir, 'per_symbol_accuracy_test_*.csv')),
            key=os.path.getmtime,
            reverse=True
        )
        if test_acc_files:
            # 最新的accuracy来源文件
            test_accuracy_source_file = test_acc_files[0]
            test_acc_df = pd.read_csv(test_accuracy_source_file)
            if 'symbol' in test_acc_df.columns and 'accuracy' in test_acc_df.columns:
                for _, row in test_acc_df.iterrows():
                    sym = str(row['symbol']).upper()
                    acc = float(row.get('accuracy', np.nan)) if pd.notna(row.get('accuracy')) else np.nan
                    samples = int(row.get('samples', 0)) if pd.notna(row.get('samples')) else 0
                    test_accuracy_map[sym] = {'accuracy': acc, 'samples': samples}
                logger.info(f"已加载测试集准确度数据: {len(test_accuracy_map)} 个标的，来源文件: {test_accuracy_source_file}")
            else:
                logger.warning(f"测试集准确度文件格式不正确: {test_acc_files[0]}")
        else:
            logger.info("未找到测试集准确度文件，将不显示准确度信息")
    except Exception as e:
        logger.warning(f"加载测试集准确度失败: {e}")

    # 额外加载 long_only / short_only 各自的 per_symbol_accuracy
    try:
        # Long-Only: per_symbol_accuracy_test_longonly_*.csv
        longonly_files = sorted(
            glob.glob(os.path.join(result_dir, 'per_symbol_accuracy_test_longonly_*.csv')),
            key=os.path.getmtime,
            reverse=True
        )
        if longonly_files:
            longonly_accuracy_source_file = longonly_files[0]
            df_long = pd.read_csv(longonly_accuracy_source_file)
            if 'symbol' in df_long.columns and 'accuracy' in df_long.columns:
                for _, row in df_long.iterrows():
                    sym = str(row['symbol']).upper()
                    acc = float(row.get('accuracy', np.nan)) if pd.notna(row.get('accuracy')) else np.nan
                    samples = int(row.get('samples', 0)) if pd.notna(row.get('samples')) else 0
                    longonly_accuracy_map[sym] = {'accuracy': acc, 'samples': samples}
                logger.info(f"已加载Long-Only测试集准确度: {len(longonly_accuracy_map)} 个标的，来源文件: {longonly_accuracy_source_file}")
            else:
                logger.warning(f"Long-Only测试集准确度文件格式不正确: {longonly_files[0]}")
        else:
            logger.info("未找到 Long-Only 的 per_symbol_accuracy_test_longonly_*.csv")
    except Exception as e:
        logger.warning(f"加载 Long-Only 测试集准确度失败: {e}")

    try:
        # Short-Only: per_symbol_accuracy_test_shortonly_*.csv
        shortonly_files = sorted(
            glob.glob(os.path.join(result_dir, 'per_symbol_accuracy_test_shortonly_*.csv')),
            key=os.path.getmtime,
            reverse=True
        )
        if shortonly_files:
            shortonly_accuracy_source_file = shortonly_files[0]
            df_short = pd.read_csv(shortonly_accuracy_source_file)
            if 'symbol' in df_short.columns and 'accuracy' in df_short.columns:
                for _, row in df_short.iterrows():
                    sym = str(row['symbol']).upper()
                    acc = float(row.get('accuracy', np.nan)) if pd.notna(row.get('accuracy')) else np.nan
                    samples = int(row.get('samples', 0)) if pd.notna(row.get('samples')) else 0
                    shortonly_accuracy_map[sym] = {'accuracy': acc, 'samples': samples}
                logger.info(f"已加载Short-Only测试集准确度: {len(shortonly_accuracy_map)} 个标的，来源文件: {shortonly_accuracy_source_file}")
            else:
                logger.warning(f"Short-Only测试集准确度文件格式不正确: {shortonly_files[0]}")
        else:
            logger.info("未找到 Short-Only 的 per_symbol_accuracy_test_shortonly_*.csv")
    except Exception as e:
        logger.warning(f"加载 Short-Only 测试集准确度失败: {e}")

    # 加载 Stage1 准确度（分别加载 long_only 和 short_only 的 stage1 accuracy）
    try:
        # Long-Only Stage1: per_symbol_accuracy_test_stage1_longonly_*.csv
        stage1_longonly_files = sorted(
            glob.glob(os.path.join(result_dir, 'per_symbol_accuracy_test_stage1_longonly_*.csv')),
            key=os.path.getmtime,
            reverse=True
        )
        if stage1_longonly_files:
            stage1_longonly_accuracy_source_file = stage1_longonly_files[0]
            df_stage1_long = pd.read_csv(stage1_longonly_accuracy_source_file)
            if 'symbol' in df_stage1_long.columns and 'accuracy' in df_stage1_long.columns:
                for _, row in df_stage1_long.iterrows():
                    sym = str(row['symbol']).upper()
                    acc = float(row.get('accuracy', np.nan)) if pd.notna(row.get('accuracy')) else np.nan
                    samples = int(row.get('samples', 0)) if pd.notna(row.get('samples')) else 0
                    stage1_longonly_accuracy_map[sym] = {'accuracy': acc, 'samples': samples}
                logger.info(f"已加载Long-Only Stage1测试集准确度: {len(stage1_longonly_accuracy_map)} 个标的，来源文件: {stage1_longonly_accuracy_source_file}")
            else:
                logger.warning(f"Long-Only Stage1测试集准确度文件格式不正确: {stage1_longonly_files[0]}")
        else:
            logger.info("未找到 Long-Only Stage1 的 per_symbol_accuracy_test_stage1_longonly_*.csv")
    except Exception as e:
        logger.warning(f"加载 Long-Only Stage1 测试集准确度失败: {e}")

    try:
        # Short-Only Stage1: per_symbol_accuracy_test_stage1_shortonly_*.csv
        stage1_shortonly_files = sorted(
            glob.glob(os.path.join(result_dir, 'per_symbol_accuracy_test_stage1_shortonly_*.csv')),
            key=os.path.getmtime,
            reverse=True
        )
        if stage1_shortonly_files:
            stage1_shortonly_accuracy_source_file = stage1_shortonly_files[0]
            df_stage1_short = pd.read_csv(stage1_shortonly_accuracy_source_file)
            if 'symbol' in df_stage1_short.columns and 'accuracy' in df_stage1_short.columns:
                for _, row in df_stage1_short.iterrows():
                    sym = str(row['symbol']).upper()
                    acc = float(row.get('accuracy', np.nan)) if pd.notna(row.get('accuracy')) else np.nan
                    samples = int(row.get('samples', 0)) if pd.notna(row.get('samples')) else 0
                    stage1_shortonly_accuracy_map[sym] = {'accuracy': acc, 'samples': samples}
                logger.info(f"已加载Short-Only Stage1测试集准确度: {len(stage1_shortonly_accuracy_map)} 个标的，来源文件: {stage1_shortonly_accuracy_source_file}")
            else:
                logger.warning(f"Short-Only Stage1测试集准确度文件格式不正确: {stage1_shortonly_files[0]}")
        else:
            logger.info("未找到 Short-Only Stage1 的 per_symbol_accuracy_test_stage1_shortonly_*.csv")
    except Exception as e:
        logger.warning(f"加载 Short-Only Stage1 测试集准确度失败: {e}")
    
    # 逐标的预测并保存
    for symbol in symbols:
        try:
            df_raw = load_processed_df(project_root, symbol)
            preds_df = predict_for_symbol(model_bundle, trainer, df_raw, feature_names, use_3_class=use_3_class, symbol_for_fallback=symbol)
            if 'date' in preds_df.columns:
                preds_df['date'] = pd.to_datetime(preds_df['date'])
            out_file = os.path.join(predictions_dir, f"applied_predictions_{symbol}.csv")
            preds_df.to_csv(out_file, index=False, encoding='utf-8-sig')
            logger.info(f"{symbol} 预测结果已保存: {out_file}")
            # 记录当前最新一日的预测（用于反转/取消判断）
            if 'date' in preds_df.columns and len(preds_df) > 0:
                latest_date = preds_df['date'].max()
                latest_rows = preds_df[preds_df['date'] == latest_date]
                if not latest_rows.empty:
                    latest_row = latest_rows.iloc[-1]
                    pred_class = int(latest_row.get('pred', 0))
                    # 方向与概率
                    if use_3_class:
                        p_no = float(latest_row.get('p_no_trend', np.nan))
                        p_up = float(latest_row.get('p_up', np.nan))
                        p_down = float(latest_row.get('p_down', np.nan))
                        direction = 'no_trend'
                        if pred_class == 1:
                            direction = 'up'
                        elif pred_class == 2:
                            direction = 'down'
                        curr_conf, prob_dict = _extract_trend_confidence(latest_row, use_3_class=True)
                        current_latest_map[symbol] = {
                            'date': latest_date, 'pred_class': pred_class, 'direction': direction,
                            'p_no_trend': p_no, 'p_up': p_up, 'p_down': p_down,
                            'confidence': curr_conf
                        }
                        # 获取测试集准确度和样本数（通用 + long_only + short_only + stage1）
                        sym_key = symbol.upper()
                        test_acc_info = test_accuracy_map.get(sym_key, {})
                        test_samples = test_acc_info.get('samples', 0)
                        long_acc_info = longonly_accuracy_map.get(sym_key, {})
                        long_accuracy = long_acc_info.get('accuracy', np.nan)
                        long_samples = long_acc_info.get('samples', 0)
                        short_acc_info = shortonly_accuracy_map.get(sym_key, {})
                        short_accuracy = short_acc_info.get('accuracy', np.nan)
                        short_samples = short_acc_info.get('samples', 0)
                        # 根据模型类型选择对应的 stage1 accuracy
                        is_long_only = model_bundle.get('long_only', False)
                        is_short_only = model_bundle.get('short_only', False)
                        has_paired = model_bundle.get('has_paired_model', False)
                        # 如果有配对模型，优先使用对应的 stage1 accuracy；否则根据主模型类型选择
                        if has_paired:
                            # 配对模型：同时显示两个 stage1 accuracy（如果有的话）
                            stage1_long_acc_info = stage1_longonly_accuracy_map.get(sym_key, {})
                            stage1_long_accuracy = stage1_long_acc_info.get('accuracy', np.nan)
                            stage1_short_acc_info = stage1_shortonly_accuracy_map.get(sym_key, {})
                            stage1_short_accuracy = stage1_short_acc_info.get('accuracy', np.nan)
                            # 对于配对模型，使用两个 stage1 accuracy 的平均值，或者分别显示
                            # 这里先使用平均值，后续可以根据需要调整
                            if pd.notna(stage1_long_accuracy) and pd.notna(stage1_short_accuracy):
                                stage1_accuracy = (stage1_long_accuracy + stage1_short_accuracy) / 2.0
                            elif pd.notna(stage1_long_accuracy):
                                stage1_accuracy = stage1_long_accuracy
                            elif pd.notna(stage1_short_accuracy):
                                stage1_accuracy = stage1_short_accuracy
                            else:
                                stage1_accuracy = np.nan
                        elif is_long_only:
                            stage1_acc_info = stage1_longonly_accuracy_map.get(sym_key, {})
                            stage1_accuracy = stage1_acc_info.get('accuracy', np.nan)
                            stage1_samples = stage1_acc_info.get('samples', 0)
                        elif is_short_only:
                            stage1_acc_info = stage1_shortonly_accuracy_map.get(sym_key, {})
                            stage1_accuracy = stage1_acc_info.get('accuracy', np.nan)
                            stage1_samples = stage1_acc_info.get('samples', 0)
                        else:
                            # 标准两阶段模型：尝试使用通用的 stage1 accuracy（如果存在）
                            # 这里可以保留原有的 stage1_accuracy_map 逻辑，或者设为 NaN
                            stage1_accuracy = np.nan
                            stage1_samples = 0
                        
                        # 如果有趋势，加入所有趋势列表（不管置信度高低）
                        if direction in ['up', 'down']:
                            # 计算 Stage2 置信度：上涨用 p_up，下跌用 p_down
                            stage2_confidence = p_up if direction == 'up' else p_down
                            trend_record = {
                                'date': latest_date.strftime('%Y-%m-%d'),
                                'symbol': symbol,
                                'direction': direction,
                                'stage2_confidence': round(float(stage2_confidence), 6) if pd.notna(stage2_confidence) and np.isfinite(stage2_confidence) else np.nan,
                                'overall_confidence': round(float(curr_conf), 6) if curr_conf is not None and np.isfinite(curr_conf) else np.nan,
                                'p_no_trend': round(p_no, 6),
                                'p_up': round(p_up, 6),
                                'p_down': round(p_down, 6),
                                'test_accuracy_stage1': round(stage1_accuracy, 4) if pd.notna(stage1_accuracy) else np.nan,
                                'test_accuracy_long_only': round(long_accuracy, 4) if pd.notna(long_accuracy) else np.nan,
                                'test_accuracy_short_only': round(short_accuracy, 4) if pd.notna(short_accuracy) else np.nan,
                                'test_samples': test_samples,
                            }
                            all_trend_records.append(trend_record)
                            
                            # 高置信度趋势则加入高置信度列表
                            if curr_conf is not None and np.isfinite(curr_conf) and curr_conf >= trend_conf_thr:
                                record = {
                                    'date': latest_date.strftime('%Y-%m-%d'),
                                    'symbol': symbol,
                                    'direction': direction,
                                    'confidence': round(float(curr_conf), 6),
                                    'generated_at': timestamp,
                                    'status': 'new_high_conf',
                                    'struck': 0,
                                    'p_no_trend': round(p_no, 6),
                                    'p_up': round(p_up, 6),
                                    'p_down': round(p_down, 6),
                                    'test_accuracy_stage1': round(stage1_accuracy, 4) if pd.notna(stage1_accuracy) else np.nan,
                                    'test_accuracy_long_only': round(long_accuracy, 4) if pd.notna(long_accuracy) else np.nan,
                                    'test_accuracy_short_only': round(short_accuracy, 4) if pd.notna(short_accuracy) else np.nan,
                                    'test_samples': test_samples,
                                }
                                high_confidence_records.append(record)
                    else:
                        p_no = float(latest_row.get('p_no_trend', np.nan))
                        p_trend = float(latest_row.get('p_trend', np.nan))
                        direction = 'trend' if pred_class == 1 else 'no_trend'
                        curr_conf, prob_dict = _extract_trend_confidence(latest_row, use_3_class=False)
                        current_latest_map[symbol] = {
                            'date': latest_date, 'pred_class': pred_class, 'direction': direction,
                            'p_no_trend': p_no, 'p_trend': p_trend,
                            'confidence': curr_conf
                        }
                        # 获取测试集准确度和样本数（通用 + long_only + short_only + stage1）
                        sym_key = symbol.upper()
                        test_acc_info = test_accuracy_map.get(sym_key, {})
                        test_samples = test_acc_info.get('samples', 0)
                        long_acc_info = longonly_accuracy_map.get(sym_key, {})
                        long_accuracy = long_acc_info.get('accuracy', np.nan)
                        long_samples = long_acc_info.get('samples', 0)
                        short_acc_info = shortonly_accuracy_map.get(sym_key, {})
                        short_accuracy = short_acc_info.get('accuracy', np.nan)
                        short_samples = short_acc_info.get('samples', 0)
                        # 根据模型类型选择对应的 stage1 accuracy
                        is_long_only = model_bundle.get('long_only', False)
                        is_short_only = model_bundle.get('short_only', False)
                        has_paired = model_bundle.get('has_paired_model', False)
                        # 如果有配对模型，优先使用对应的 stage1 accuracy；否则根据主模型类型选择
                        if has_paired:
                            # 配对模型：同时显示两个 stage1 accuracy（如果有的话）
                            stage1_long_acc_info = stage1_longonly_accuracy_map.get(sym_key, {})
                            stage1_long_accuracy = stage1_long_acc_info.get('accuracy', np.nan)
                            stage1_short_acc_info = stage1_shortonly_accuracy_map.get(sym_key, {})
                            stage1_short_accuracy = stage1_short_acc_info.get('accuracy', np.nan)
                            # 对于配对模型，使用两个 stage1 accuracy 的平均值，或者分别显示
                            # 这里先使用平均值，后续可以根据需要调整
                            if pd.notna(stage1_long_accuracy) and pd.notna(stage1_short_accuracy):
                                stage1_accuracy = (stage1_long_accuracy + stage1_short_accuracy) / 2.0
                            elif pd.notna(stage1_long_accuracy):
                                stage1_accuracy = stage1_long_accuracy
                            elif pd.notna(stage1_short_accuracy):
                                stage1_accuracy = stage1_short_accuracy
                            else:
                                stage1_accuracy = np.nan
                        elif is_long_only:
                            stage1_acc_info = stage1_longonly_accuracy_map.get(sym_key, {})
                            stage1_accuracy = stage1_acc_info.get('accuracy', np.nan)
                            stage1_samples = stage1_acc_info.get('samples', 0)
                        elif is_short_only:
                            stage1_acc_info = stage1_shortonly_accuracy_map.get(sym_key, {})
                            stage1_accuracy = stage1_acc_info.get('accuracy', np.nan)
                            stage1_samples = stage1_acc_info.get('samples', 0)
                        else:
                            # 标准两阶段模型：尝试使用通用的 stage1 accuracy（如果存在）
                            # 这里可以保留原有的 stage1_accuracy_map 逻辑，或者设为 NaN
                            stage1_accuracy = np.nan
                            stage1_samples = 0
                        
                        # 如果有趋势，加入所有趋势列表（不管置信度高低）
                        if direction == 'trend':
                            # 对于二分类模型，Stage2 置信度就是 p_trend
                            trend_record = {
                                'date': latest_date.strftime('%Y-%m-%d'),
                                'symbol': symbol,
                                'direction': direction,
                                'stage2_confidence': round(float(p_trend), 6) if pd.notna(p_trend) and np.isfinite(p_trend) else np.nan,
                                'overall_confidence': round(float(curr_conf), 6) if curr_conf is not None and np.isfinite(curr_conf) else np.nan,
                                'p_no_trend': round(p_no, 6),
                                'p_trend': round(p_trend, 6),
                                'test_accuracy_stage1': round(stage1_accuracy, 4) if pd.notna(stage1_accuracy) else np.nan,
                                'test_accuracy_long_only': round(long_accuracy, 4) if pd.notna(long_accuracy) else np.nan,
                                'test_accuracy_short_only': round(short_accuracy, 4) if pd.notna(short_accuracy) else np.nan,
                                'test_samples': test_samples,
                            }
                            all_trend_records.append(trend_record)
                            
                            # 高置信度趋势则加入高置信度列表
                            if curr_conf is not None and np.isfinite(curr_conf) and curr_conf >= trend_conf_thr:
                                record = {
                                    'date': latest_date.strftime('%Y-%m-%d'),
                                    'symbol': symbol,
                                    'direction': direction,
                                    'confidence': round(float(curr_conf), 6),
                                    'generated_at': timestamp,
                                    'status': 'new_high_conf',
                                    'struck': 0,
                                    'p_no_trend': round(p_no, 6),
                                    'p_trend': round(p_trend, 6),
                                    'test_accuracy_stage1': round(stage1_accuracy, 4) if pd.notna(stage1_accuracy) else np.nan,
                                    'test_accuracy_long_only': round(long_accuracy, 4) if pd.notna(long_accuracy) else np.nan,
                                    'test_accuracy_short_only': round(short_accuracy, 4) if pd.notna(short_accuracy) else np.nan,
                                    'test_samples': test_samples,
                                }
                                high_confidence_records.append(record)
        except Exception as e:
            logger.warning(f"{symbol} 预测失败: {e}")
    
    # 仅输出本轮高置信度记录（去除反转/取消对比逻辑）
    output_records = list(high_confidence_records)
    # 输出高置信度/变化清单
    if output_records:
        high_conf_df = pd.DataFrame(output_records)
        # 按最高置信度降序排序；同置信度按日期降序、struck升序、symbol升序
        try:
            # 置信度排序键
            high_conf_df['confidence'] = pd.to_numeric(high_conf_df.get('confidence'), errors='coerce')
            high_conf_df['sort_conf'] = high_conf_df['confidence'].fillna(-1.0)
            # 日期排序键
            high_conf_df['date'] = pd.to_datetime(high_conf_df['date'], errors='coerce')
            high_conf_df = high_conf_df.sort_values(
                ['sort_conf', 'date', 'struck', 'symbol'],
                ascending=[False, False, True, True]
            ).drop(columns=['sort_conf'])
            # 写回日期字符串
            high_conf_df['date'] = high_conf_df['date'].dt.strftime('%Y-%m-%d')
        except Exception:
            pass
        os.makedirs(apply_dir, exist_ok=True)
        high_conf_file = os.path.join(apply_dir, "high_confidence_signals.csv")
        
        # 如果原文件存在，先创建带时间戳的备份
        if os.path.exists(high_conf_file):
            backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(apply_dir, f"high_confidence_signals_{backup_timestamp}.csv")
            try:
                shutil.copy2(high_conf_file, backup_file)
                logger.info(f"已备份原文件到: {backup_file}")
            except Exception as e:
                logger.warning(f"备份原文件失败: {e}")
        
        high_conf_df.to_csv(high_conf_file, index=False, encoding='utf-8-sig')
        logger.info(f"高置信度趋势/变化信号已保存: {high_conf_file}")
    else:
        logger.info(f"未发现置信度≥{trend_conf_thr:.2f}的最新趋势信号；也无反转/取消变更")
    
    # 输出所有有趋势的标的，按 Stage2 置信度从高到低排列
    if all_trend_records:
        all_trend_df = pd.DataFrame(all_trend_records)
        # 按 Stage2 置信度降序排序；同置信度按日期降序、symbol升序
        try:
            # Stage2 置信度排序键
            all_trend_df['stage2_confidence'] = pd.to_numeric(all_trend_df.get('stage2_confidence'), errors='coerce')
            all_trend_df['sort_stage2'] = all_trend_df['stage2_confidence'].fillna(-1.0)
            # 日期排序键
            all_trend_df['date'] = pd.to_datetime(all_trend_df['date'], errors='coerce')
            all_trend_df = all_trend_df.sort_values(
                ['sort_stage2', 'date', 'symbol'],
                ascending=[False, False, True]
            ).drop(columns=['sort_stage2'])
            # 写回日期字符串
            all_trend_df['date'] = all_trend_df['date'].dt.strftime('%Y-%m-%d')
        except Exception:
            pass
        os.makedirs(apply_dir, exist_ok=True)
        all_trend_file = os.path.join(apply_dir, "all_trend_signals.csv")
        
        # 如果原文件存在，先创建带时间戳的备份
        if os.path.exists(all_trend_file):
            backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(apply_dir, f"all_trend_signals_{backup_timestamp}.csv")
            try:
                shutil.copy2(all_trend_file, backup_file)
                logger.info(f"已备份原文件到: {backup_file}")
            except Exception as e:
                logger.warning(f"备份原文件失败: {e}")
        
        all_trend_df.to_csv(all_trend_file, index=False, encoding='utf-8-sig')
        logger.info(f"所有趋势信号已保存: {all_trend_file} (共 {len(all_trend_df)} 个标的，按 Stage2 置信度降序排列)")
    else:
        logger.info("未发现任何趋势信号")
    
    # 额外输出当前持仓信号（若IB Gateway在线）
    try:
        def _gather_current_positions():
            if IB is None:
                logger.info("ib_insync 未安装，跳过读取IBKR持仓")
                return []
            host = getattr(config, 'IBKR_HOST', '127.0.0.1')
            port = int(getattr(config, 'IBKR_PORT', 4001))
            # 使用与 data_collector 相同的 clientId
            client_id = 1
            ib = IB()
            try:
                ib.connect(host, port, clientId=client_id)
            except Exception as e:
                logger.info(f"未连接到IBKR({host}:{port})，跳过持仓读取：{e}")
                return []
            if not ib.isConnected():
                logger.info("IBKR未连接，跳过持仓读取")
                return []
            pos_list = []
            try:
                positions = ib.positions()
                for p in positions:
                    c = p.contract
                    secType = getattr(c, 'secType', '')
                    sym = str(getattr(c, 'symbol', '')).upper()
                    right = getattr(c, 'right', None)
                    size = float(getattr(p, 'position', 0.0))
                    if abs(size) <= 0 or not sym:
                        continue
                    if secType == 'STK':
                        pos_list.append({'symbol': sym, 'secType': 'STK', 'right': None, 'position': size})
                    elif secType == 'OPT':
                        # 期权：使用其正股symbol（contract.symbol为正股），保留权利别C/P
                        pos_list.append({'symbol': sym, 'secType': 'OPT', 'right': str(right) if right else None, 'position': size})
                return pos_list
            except Exception as e:
                logger.info(f"读取IBKR持仓失败：{e}")
                return []
            finally:
                try:
                    ib.disconnect()
                except Exception:
                    pass
        positions = _gather_current_positions()
        if positions:
            logger.info(f"检测到IBKR持仓 {len(positions)} 条，生成 current_position_signals")
            pos_df = pd.DataFrame(positions)
            # 聚合到(symbol, secType, right)
            pos_agg = pos_df.groupby(['symbol', 'secType', 'right'], as_index=False)['position'].sum()
            cps = []
            for _, r in pos_agg.iterrows():
                sym = str(r['symbol'])
                secType = str(r['secType'])
                right = (None if (r['right'] is None or str(r['right']).strip() == '' or str(r['right']).upper() == 'None') else str(r['right']).upper())
                net_pos = float(r['position'])
                latest = current_latest_map.get(sym)
                if latest is None:
                    continue
                # 计算following：若预测方向对该持仓理论上有利则 True，否则 False；无趋势标 NaN
                direction = latest['direction']
                following_val = None
                if direction in ['up', 'down']:
                    if secType == 'STK':
                        # 简化：持仓为多头时，上涨为绿、下跌为红；若净空头则相反
                        if net_pos >= 0:
                            following_val = True if direction == 'up' else False
                        else:
                            following_val = True if direction == 'down' else False
                    elif secType == 'OPT':
                        # 期权：CALL 对上涨有利，PUT 对下跌有利；若净空头则方向相反
                        is_call = (right == 'C')
                        is_put = (right == 'P')
                        if net_pos >= 0:
                            if (direction == 'up' and is_call) or (direction == 'down' and is_put):
                                following_val = True
                            else:
                                following_val = False
                        else:
                            # 空头期权：与多头相反
                            if (direction == 'up' and is_call) or (direction == 'down' and is_put):
                                following_val = False
                            else:
                                following_val = True
                # 获取测试集准确度和样本数（通用 + long_only + short_only + stage1）
                sym_key = sym.upper()
                test_acc_info = test_accuracy_map.get(sym_key, {})
                test_samples = test_acc_info.get('samples', 0)
                long_acc_info = longonly_accuracy_map.get(sym_key, {})
                long_accuracy = long_acc_info.get('accuracy', np.nan)
                long_samples = long_acc_info.get('samples', 0)
                short_acc_info = shortonly_accuracy_map.get(sym_key, {})
                short_accuracy = short_acc_info.get('accuracy', np.nan)
                short_samples = short_acc_info.get('samples', 0)
                # 根据模型类型选择对应的 stage1 accuracy
                is_long_only = model_bundle.get('long_only', False)
                is_short_only = model_bundle.get('short_only', False)
                has_paired = model_bundle.get('has_paired_model', False)
                # 如果有配对模型，优先使用对应的 stage1 accuracy；否则根据主模型类型选择
                if has_paired:
                    # 配对模型：同时显示两个 stage1 accuracy（如果有的话）
                    stage1_long_acc_info = stage1_longonly_accuracy_map.get(sym_key, {})
                    stage1_long_accuracy = stage1_long_acc_info.get('accuracy', np.nan)
                    stage1_short_acc_info = stage1_shortonly_accuracy_map.get(sym_key, {})
                    stage1_short_accuracy = stage1_short_acc_info.get('accuracy', np.nan)
                    # 对于配对模型，使用两个 stage1 accuracy 的平均值，或者分别显示
                    # 这里先使用平均值，后续可以根据需要调整
                    if pd.notna(stage1_long_accuracy) and pd.notna(stage1_short_accuracy):
                        stage1_accuracy = (stage1_long_accuracy + stage1_short_accuracy) / 2.0
                    elif pd.notna(stage1_long_accuracy):
                        stage1_accuracy = stage1_long_accuracy
                    elif pd.notna(stage1_short_accuracy):
                        stage1_accuracy = stage1_short_accuracy
                    else:
                        stage1_accuracy = np.nan
                elif is_long_only:
                    stage1_acc_info = stage1_longonly_accuracy_map.get(sym_key, {})
                    stage1_accuracy = stage1_acc_info.get('accuracy', np.nan)
                    stage1_samples = stage1_acc_info.get('samples', 0)
                elif is_short_only:
                    stage1_acc_info = stage1_shortonly_accuracy_map.get(sym_key, {})
                    stage1_accuracy = stage1_acc_info.get('accuracy', np.nan)
                    stage1_samples = stage1_acc_info.get('samples', 0)
                else:
                    # 标准两阶段模型：尝试使用通用的 stage1 accuracy（如果存在）
                    # 这里可以保留原有的 stage1_accuracy_map 逻辑，或者设为 NaN
                    stage1_accuracy = np.nan
                    stage1_samples = 0
                
                if use_3_class:
                    cps.append({
                        'date': latest['date'].strftime('%Y-%m-%d') if isinstance(latest['date'], pd.Timestamp) else str(latest['date']),
                        'symbol': sym,
                        'secType': secType,
                        'right': right if right is not None else '',
                        'direction': latest['direction'],
                        'confidence': round(float(latest.get('confidence', np.nan)), 6) if latest.get('confidence', None) is not None else np.nan,
                        'generated_at': timestamp,
                        'position': net_pos,
                        'p_no_trend': round(float(latest.get('p_no_trend', np.nan)), 6) if latest.get('p_no_trend', None) is not None else np.nan,
                        'p_up': round(float(latest.get('p_up', np.nan)), 6) if latest.get('p_up', None) is not None else np.nan,
                        'p_down': round(float(latest.get('p_down', np.nan)), 6) if latest.get('p_down', None) is not None else np.nan,
                        'following': following_val,
                        'test_accuracy_stage1': round(stage1_accuracy, 4) if pd.notna(stage1_accuracy) else np.nan,
                        'test_accuracy_long_only': round(long_accuracy, 4) if pd.notna(long_accuracy) else np.nan,
                        'test_accuracy_short_only': round(short_accuracy, 4) if pd.notna(short_accuracy) else np.nan,
                        'test_samples': test_samples,
                    })
                else:
                    cps.append({
                        'date': latest['date'].strftime('%Y-%m-%d') if isinstance(latest['date'], pd.Timestamp) else str(latest['date']),
                        'symbol': sym,
                        'secType': secType,
                        'right': right if right is not None else '',
                        'direction': latest['direction'],
                        'confidence': round(float(latest.get('confidence', np.nan)), 6) if latest.get('confidence', None) is not None else np.nan,
                        'generated_at': timestamp,
                        'position': net_pos,
                        'p_no_trend': round(float(latest.get('p_no_trend', np.nan)), 6) if latest.get('p_no_trend', None) is not None else np.nan,
                        'p_trend': round(float(latest.get('p_trend', np.nan)), 6) if latest.get('p_trend', None) is not None else np.nan,
                        'following': following_val,
                        'test_accuracy_stage1': round(stage1_accuracy, 4) if pd.notna(stage1_accuracy) else np.nan,
                        'test_accuracy_long_only': round(long_accuracy, 4) if pd.notna(long_accuracy) else np.nan,
                        'test_accuracy_short_only': round(short_accuracy, 4) if pd.notna(short_accuracy) else np.nan,
                        'test_samples': test_samples,
                    })
            if cps:
                cps_df = pd.DataFrame(cps)
                try:
                    cps_df['confidence'] = pd.to_numeric(cps_df.get('confidence'), errors='coerce')
                    cps_df['date'] = pd.to_datetime(cps_df['date'], errors='coerce')
                    cps_df = cps_df.sort_values(['confidence', 'date', 'symbol'], ascending=[False, False, True])
                except Exception:
                    pass
                cps_dir = apply_dir
                os.makedirs(cps_dir, exist_ok=True)
                cps_file = os.path.join(cps_dir, "current_position_signals.csv")
                # 备份旧文件
                if os.path.exists(cps_file):
                    backup_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_file = os.path.join(cps_dir, f"current_position_signals_{backup_ts}.csv")
                    try:
                        shutil.copy2(cps_file, backup_file)
                        logger.info(f"已备份原持仓信号到: {backup_file}")
                    except Exception as e:
                        logger.warning(f"备份原持仓信号失败: {e}")
                cps_df.to_csv(cps_file, index=False, encoding='utf-8-sig')
                logger.info(f"current_position_signals 已保存: {cps_file}")
            else:
                logger.info("当前持仓标的暂无可用最新预测，跳过生成 current_position_signals")
        else:
            logger.info("未检测到IBKR持仓或无法读取，跳过 current_position_signals 生成")
    except Exception as e:
        logger.warning(f"生成 current_position_signals 失败：{e}")

    # 在脚本结尾输出accuracy来源信息，便于确认/排查
    if test_accuracy_source_file:
        logger.info(f"本次运行中信号里展示的通用 test_accuracy 来源文件: {test_accuracy_source_file}")
    else:
        logger.info("本次运行未使用任何通用 per_symbol_accuracy_test_*.csv 文件作为accuracy来源")

    if longonly_accuracy_source_file:
        logger.info(f"Long-Only per_symbol_accuracy 来源文件: {longonly_accuracy_source_file}")
    else:
        logger.info("本次运行未使用任何 per_symbol_accuracy_test_longonly_*.csv 作为 Long-Only accuracy 来源")

    if shortonly_accuracy_source_file:
        logger.info(f"Short-Only per_symbol_accuracy 来源文件: {shortonly_accuracy_source_file}")
    else:
        logger.info("本次运行未使用任何 per_symbol_accuracy_test_shortonly_*.csv 作为 Short-Only accuracy 来源")

    if stage1_longonly_accuracy_source_file:
        logger.info(f"Long-Only Stage1 per_symbol_accuracy 来源文件: {stage1_longonly_accuracy_source_file}")
    else:
        logger.info("本次运行未使用任何 per_symbol_accuracy_test_stage1_longonly_*.csv 作为 Long-Only Stage1 accuracy 来源")

    if stage1_shortonly_accuracy_source_file:
        logger.info(f"Short-Only Stage1 per_symbol_accuracy 来源文件: {stage1_shortonly_accuracy_source_file}")
    else:
        logger.info("本次运行未使用任何 per_symbol_accuracy_test_stage1_shortonly_*.csv 作为 Short-Only Stage1 accuracy 来源")

    logger.info("全部预测完成")


if __name__ == "__main__":
    main()


