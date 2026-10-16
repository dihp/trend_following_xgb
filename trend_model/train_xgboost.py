"""
XGBoost趋势预测模型训练
"""

import os
import sys
# 添加项目根目录到路径，以便导入根目录的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import config
import logging
import pickle
import json
import shutil
from datetime import datetime
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Stage2OVR:
    """
    Stage2 OVR 包装器：
    - model_up: 训练于 (上涨 vs 其余)
    - model_down: 训练于 (下跌 vs 其余)
    predict_proba(X) 返回条件化方向概率 [p_up|trend, p_down|trend]，
    以便与现有两阶段组合逻辑无缝对接。
    """
    def __init__(self, model_up, model_down):
        self.model_up = model_up
        self.model_down = model_down
        self.is_ovr = True

    def predict_proba(self, X):
        # OVR概率（正类为1）：q_up = P(up vs rest), q_down = P(down vs rest)
        q_up = self.model_up.predict_proba(X)[:, 1]
        q_down = self.model_down.predict_proba(X)[:, 1]
        # 条件化方向概率（归一化到 up/down 之和为1）
        denom = np.clip(q_up + q_down, 1e-12, None)
        p_up_cond = q_up / denom
        p_down_cond = q_down / denom
        return np.vstack([p_up_cond, p_down_cond]).T


class TrendXGBoostTrainer:
    """XGBoost趋势预测模型训练器"""
    
    def __init__(self):
        """初始化训练器"""
        # 获取项目根目录（上级目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, config.DATA_DIR)
        self.xgb_dir = os.path.join(self.data_dir, 'xgb')
        self.train_dir = os.path.join(self.xgb_dir, 'train')
        self.val_dir = os.path.join(self.xgb_dir, 'validation')
        self.test_dir = os.path.join(self.xgb_dir, 'test')
        self.test_early_dir = os.path.join(self.xgb_dir, 'test_early')
        self.test_late_dir = os.path.join(self.xgb_dir, 'test_late')
        # XGBoost模型保存目录
        self.model_dir = os.path.join(project_root, 'models', 'xgboost')
        
        # 创建模型目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"创建XGBoost模型目录: {self.model_dir}")
        else:
            logger.info(f"XGBoost模型目录已存在: {self.model_dir}")
        
        # 加载元数据（行业信息）
        self.metadata_file = os.path.join(self.data_dir, 'symbols_metadata.json')
        self.metadata = self.load_metadata()
        
        # 用于编码分类特征的LabelEncoder
        self.sector_encoder = LabelEncoder()
        self.industry_encoder = LabelEncoder()
        self.sub_industry_encoder = LabelEncoder()
        # 概率校准参数
        self.calibration = None
    
    def load_metadata(self):
        """加载标的元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"成功加载元数据，包含 {len(metadata)} 个标的")
                return metadata
            except Exception as e:
                logger.warning(f"加载元数据失败: {e}")
                return {}
        else:
            logger.warning(f"元数据文件不存在: {self.metadata_file}")
            return {}
    
    def add_industry_features(self, df):
        """
        添加行业特征到数据中
        
        Args:
            df: DataFrame，需要包含'symbol'列
        
        Returns:
            DataFrame: 添加了行业特征的DataFrame
        """
        if 'symbol' not in df.columns:
            logger.warning("数据中缺少'symbol'列，无法添加行业特征")
            return df
        
        if not self.metadata:
            logger.warning("元数据为空，无法添加行业特征")
            return df
        
        df = df.copy()
        
        # 添加行业信息列
        df['sector'] = df['symbol'].map(lambda x: self.metadata.get(x, {}).get('sector', 'Unknown'))
        df['industry'] = df['symbol'].map(lambda x: self.metadata.get(x, {}).get('industry', 'Unknown'))
        df['sub_industry'] = df['symbol'].map(lambda x: self.metadata.get(x, {}).get('sub_industry', 'Unknown'))
        
        # 统计信息
        sector_counts = df['sector'].value_counts()
        industry_counts = df['industry'].value_counts()
        logger.info(f"添加行业特征:")
        logger.info(f"  Sector分布: {dict(sector_counts)}")
        logger.info(f"  Industry分布: {dict(industry_counts)}")
        
        return df
    
    def encode_categorical_features(self, df, fit_encoders=False):
        """
        将分类特征编码为数值特征
        
        Args:
            df: DataFrame，包含分类特征
            fit_encoders: 是否拟合编码器（通常在训练集上fit，其他集上transform）
        
        Returns:
            DataFrame: 编码后的DataFrame
        """
        df = df.copy()
        
        # 对sector、industry、sub_industry进行label encoding
        for col, encoder in [('sector', self.sector_encoder), 
                            ('industry', self.industry_encoder),
                            ('sub_industry', self.sub_industry_encoder)]:
            if col in df.columns:
                series_str = df[col].astype(str)
                if fit_encoders:
                    df[col] = encoder.fit_transform(series_str)
                    logger.info(f"拟合{col}编码器，类别数: {len(encoder.classes_)}")
                else:
                    # 推理阶段：优先用已拟合编码器；若未拟合则回退拟合元数据全集，再transform
                    try:
                        df[col] = encoder.transform(series_str)
                    except Exception:
                        # 未拟合或类别不完整，尝试用元数据全集拟合后再transform
                        try:
                            categories = []
                            if hasattr(self, 'metadata') and isinstance(self.metadata, dict):
                                key = col  # 'sector' / 'industry' / 'sub_industry'
                                for sym, meta in self.metadata.items():
                                    categories.append(str(meta.get(key, 'Unknown')))
                            # 至少包含现有数据中的类别与'Unknown'
                            categories = list(set(categories + series_str.tolist() + ['Unknown']))
                            encoder.fit(categories)
                            df[col] = encoder.transform(series_str)
                            logger.warning(f"{col}编码器在推理阶段回退拟合（可能与训练映射不一致），类别数: {len(encoder.classes_)}")
                        except Exception:
                            # 最终回退：不可用时将该列置为-1
                            df[col] = -1
                            logger.warning(f"{col}编码失败，已回退为-1")
        
        return df
    
    def load_data(self, data_path, use_all_symbols=True, dataset_name=''):
        """
        加载数据，确保按时间顺序排列
        
        Args:
            data_path: 数据文件路径
            use_all_symbols: 是否使用合并后的all_symbols.csv文件
            dataset_name: 数据集名称（用于日志）
        
        Returns:
            DataFrame: 按时间排序后的DataFrame
        """
        if use_all_symbols:
            file_path = os.path.join(data_path, 'all_symbols.csv')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据文件不存在: {file_path}")
            
            df = pd.read_csv(file_path)
        else:
            # 如果all_symbols.csv不存在，加载所有单个文件
            csv_files = [f for f in os.listdir(data_path) 
                        if f.endswith('.csv') and f != 'all_symbols.csv']
            if not csv_files:
                raise FileNotFoundError(f"在 {data_path} 中未找到数据文件")
            
            all_data = []
            for csv_file in csv_files:
                file_path = os.path.join(data_path, csv_file)
                df = pd.read_csv(file_path)
                symbol = csv_file.replace('.csv', '')
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                all_data.append(df)
            
            df = pd.concat(all_data, ignore_index=True)
        
        # 确保数据按时间顺序排列
        if 'date' not in df.columns:
            raise ValueError(f"{dataset_name}数据缺少'date'列")
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 按时间排序（如果有symbol列，也按symbol排序以保持一致性）
        if 'symbol' in df.columns:
            df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        else:
            df = df.sort_values('date').reset_index(drop=True)
        
        # 验证时间顺序
        date_diff = df['date'].diff()
        if (date_diff < pd.Timedelta(0)).any():
            logger.warning(f"{dataset_name}数据中存在日期倒退，已重新排序")
            if 'symbol' in df.columns:
                df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"成功加载{dataset_name}数据: {file_path if use_all_symbols else '多个文件'}，共 {len(df)} 行")
        if len(df) > 0:
            logger.info(f"  时间范围: {df['date'].min()} 到 {df['date'].max()}")
        
        return df
    
    def add_time_features(self, df):
        """
        从日期列提取时间特征
        
        Args:
            df: DataFrame，需要包含'date'列
        
        Returns:
            DataFrame: 添加了时间特征的DataFrame
        """
        if 'date' not in df.columns:
            return df
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # 提取时间特征
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=周一, 6=周日
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        return df
    
    def prepare_features(self, df, use_time_features=True, use_raw_ohlcv=False, 
                        use_industry_features=True, fit_encoders=False):
        """
        准备特征和标签
        
        Args:
            df: 原始DataFrame
            use_time_features: 是否使用时间特征（从date提取）
            use_raw_ohlcv: 是否使用原始K线数据（open, high, low, close, volume）作为特征
            use_industry_features: 是否使用行业特征（sector, industry, sub_industry）
            fit_encoders: 是否拟合编码器（通常在训练集上fit=True，其他集上fit=False）
        
        Returns:
            X: 特征DataFrame
            y: 标签Series（processed_label，已转换为0和1）
            feature_columns: 特征列名列表
        """
        df = df.copy()
        
        # 添加行业特征（如果启用）
        if use_industry_features:
            df = self.add_industry_features(df)
            # 编码分类特征
            df = self.encode_categorical_features(df, fit_encoders=fit_encoders)
            logger.info("已添加行业特征: sector, industry, sub_industry（已编码为数值）")
        
        # 添加时间特征（如果启用）
        if use_time_features and 'date' in df.columns:
            df = self.add_time_features(df)
            logger.info("已添加时间特征: day_of_week, day_of_month, month, quarter, is_month_start, is_month_end, is_quarter_start, is_quarter_end")
        
        # 基础排除列（必须排除的）
        exclude_columns = ['date', 'symbol', 'label', 'label_old', 'processed_label', 'trend_position_weight']
        
        # 排除基于未来数据的特征（这些特征和标签基于相同的数据计算，会导致数据泄露）
        # 即使滞后了future_days+1天，这些特征仍然与标签高度相关，因为它们本质上就是标签的组成部分
        exclude_columns.extend(['NormalizedReturn', 'Slope', 'R2', 'TNR'])
        
        # 注意：label、processed_label、trend_position_weight 都被排除，不会作为特征
        # processed_label 将作为预测目标
        
        # 根据配置决定是否排除原始K线数据
        if not use_raw_ohlcv:
            exclude_columns.extend(['open', 'high', 'low', 'close', 'volume'])
            logger.info("排除原始K线数据（open, high, low, close, volume），使用技术指标代替")
        else:
            logger.info("包含原始K线数据作为特征")
        
        logger.info("排除NormalizedReturn, Slope, R2, TNR特征（这些特征基于未来数据计算，与标签高度相关，会导致数据泄露）")
        logger.info("排除label、processed_label、trend_position_weight（这些列不会作为特征，processed_label将作为预测目标）")
        
        # 获取所有特征列（排除上述列后的所有列）
        # 允许从config注入额外排除列（用于AB实验/快速禁用一组新特征）
        try:
            extra_excludes = getattr(config, 'EXTRA_EXCLUDE_COLUMNS', []) or []
            if isinstance(extra_excludes, (list, tuple, set)):
                exclude_columns.extend(list(extra_excludes))
        except Exception:
            pass
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # 检查是否使用三分类模型
        use_3_class = getattr(config, 'USE_3_CLASS_MODEL', True)
        
        # 根据配置选择目标变量
        if use_3_class:
            # 三分类：使用label列（0=无趋势, 1=上涨, -1=下跌）
            if 'label' not in df.columns:
                raise ValueError("数据中缺少'label'列，无法进行三分类训练")
            target_column = 'label'
            logger.info("使用三分类模型：预测无趋势(0)、上涨趋势(1)、下跌趋势(-1)")
        else:
            # 二分类：使用processed_label列（0=无趋势, 1=有趋势）
            if 'processed_label' not in df.columns:
                raise ValueError("数据中缺少'processed_label'列，请先运行label_processor.py生成processed_label")
            target_column = 'processed_label'
            logger.info("使用二分类模型：预测无趋势(0)、有趋势(1)")
        
        # 显示列信息
        logger.info(f"\n数据列信息:")
        logger.info(f"  总列数: {len(df.columns)}")
        logger.info(f"  排除的列（非特征）: {exclude_columns}")
        logger.info(f"  特征列数: {len(feature_columns)}")
        logger.info(f"  特征列: {feature_columns}")
        if target_column in df.columns:
            logger.info(f"  预测目标列: {target_column}（位置: {df.columns.get_loc(target_column) + 1}/{len(df.columns)}）")
        
        # 提取特征和标签
        X = df[feature_columns].copy()
        y = df[target_column].copy()  # 使用选定的目标列
        
        # 提取趋势位置权重（如果存在）
        trend_position_weight = None
        if 'trend_position_weight' in df.columns:
            trend_position_weight = df['trend_position_weight'].copy()
            logger.info("检测到趋势位置权重列，将用于样本权重计算")
        else:
            logger.info("未检测到趋势位置权重列，将使用默认权重1.0")
        
        # 保留symbol列用于后续按标的评估
        symbols_series = None
        if 'symbol' in df.columns:
            symbols_series = df['symbol'].copy()
        
        # 时间衰减权重（仅用于训练集，后续在训练函数中应用）
        time_weights = None
        try:
            if getattr(config, 'USE_TIME_DECAY_WEIGHT', False) and 'date' in df.columns:
                dates = pd.to_datetime(df['date'])
                max_date = dates.max()
                age_days = (max_date - dates).dt.days.astype(float).clip(lower=0.0)
                half_life = float(getattr(config, 'TIME_DECAY_HALF_LIFE_DAYS', 180))
                # 指数衰减：w = exp(-ln(2) * age/half_life)
                lam = np.log(2.0) / half_life if half_life > 0 else 0.0
                w = np.exp(-lam * age_days)
                w = w.clip(lower=float(getattr(config, 'TIME_DECAY_MIN', 0.6)),
                           upper=float(getattr(config, 'TIME_DECAY_MAX', 1.0)))
                time_weights = w
        except Exception as e:
            logger.warning(f"计算时间衰减权重失败，将跳过：{e}")
            time_weights = None
        
        # 质量权重（Quality Weight）：基于R²、段长、振幅（仅用于训练样本权重，不作为特征）
        quality_weights = None
        try:
            use_quality = getattr(config, 'USE_QUALITY_WEIGHT', True)
            if use_quality and ('label' in df.columns) and ('symbol' in df.columns) and ('date' in df.columns):
                df_q = df[['symbol', 'date', 'label']].copy()
                # 需要close用于振幅；若不存在则仅使用R²与段长
                has_close = 'close' in df.columns
                if has_close:
                    df_q['close'] = df['close'].astype(float)
                # R²列
                r2_col = getattr(config, 'QUALITY_R2_COLUMN', 'R2_10D')
                if r2_col not in df.columns:
                    r2_col = 'R2_5D' if 'R2_5D' in df.columns else (r2_col if r2_col in df.columns else None)
                r2_weight = getattr(config, 'QUALITY_R2_WEIGHT', 0.5)
                len_weight = getattr(config, 'QUALITY_LENGTH_WEIGHT', 0.25)
                amp_weight = getattr(config, 'QUALITY_AMPLITUDE_WEIGHT', 0.25)
                len_cap = float(getattr(config, 'QUALITY_LENGTH_CAP', 30))
                amp_cap = float(getattr(config, 'QUALITY_AMPLITUDE_CAP', 0.20))
                q_min = float(getattr(config, 'QUALITY_MIN', 0.30))
                q_max = float(getattr(config, 'QUALITY_MAX', 1.20))
                q0 = float(getattr(config, 'QUALITY_WEIGHT_FOR_NO_TREND', 1.0))
                
                # 基于symbol/date排序，便于连续段计算；保留原索引以便回写
                df_q = df_q.sort_values(['symbol', 'date'])
                # 非零标签为趋势
                trend_mask = df_q['label'] != 0
                # 计算组id：符号变化、标签变化、或symbol边界触发新组；对非趋势行后续设为NaN
                symbol_change = (df_q['symbol'] != df_q['symbol'].shift()).fillna(True)
                label_series = df_q['label'].where(trend_mask, np.nan)
                trend_break = symbol_change | (label_series != label_series.shift()) | label_series.isna()
                group_id = trend_break.cumsum()
                # 准备容器
                df_q['group_id'] = group_id
                # 仅在趋势段计算长度与振幅
                run_len = pd.Series(0, index=df_q.index, dtype=float)
                if trend_mask.any():
                    run_len.loc[trend_mask] = df_q.loc[trend_mask].groupby('group_id')['group_id'].transform('size').astype(float)
                # 段长归一化
                len_norm = (run_len / len_cap).clip(0.0, 1.0)
                # 振幅：需要close且为趋势行
                amp_norm = pd.Series(0.0, index=df_q.index, dtype=float)
                if has_close and trend_mask.any():
                    start_close = df_q.loc[trend_mask].groupby('group_id')['close'].transform('first')
                    # 对齐索引
                    start_close_full = pd.Series(np.nan, index=df_q.index)
                    start_close_full.loc[trend_mask] = start_close
                    # 上涨与下跌分别计算
                    up_mask = trend_mask & (df_q['label'] == 1)
                    down_mask = trend_mask & (df_q['label'] == -1)
                    amp_raw = pd.Series(0.0, index=df_q.index, dtype=float)
                    if up_mask.any():
                        amp_raw.loc[up_mask] = (df_q.loc[up_mask, 'close'] - start_close_full.loc[up_mask]) / start_close_full.loc[up_mask]
                    if down_mask.any():
                        amp_raw.loc[down_mask] = (start_close_full.loc[down_mask] - df_q.loc[down_mask, 'close']) / start_close_full.loc[down_mask]
                    amp_norm = (amp_raw / amp_cap).clip(0.0, 1.0)
                # R²归一化
                if r2_col is not None and r2_col in df.columns:
                    r2_norm = df[r2_col].astype(float).clip(0.0, 1.0).fillna(0.5)
                else:
                    r2_norm = pd.Series(0.5, index=df.index, dtype=float)
                # 组合质量权重
                q = (r2_weight * r2_norm.reindex(df_q.index, fill_value=0.5) +
                     len_weight * len_norm +
                     amp_weight * amp_norm)
                # 无趋势样本使用固定权重
                q.loc[~trend_mask] = q0
                q = q.clip(q_min, q_max)
                # 回到原始顺序（按原索引）
                q = q.reindex(df.index)
                quality_weights = q
                logger.info(f"已计算质量权重（基于R²/段长/振幅），范围: [{quality_weights.min():.2f}, {quality_weights.max():.2f}]")
        except Exception as e:
            logger.warning(f"质量权重计算失败，将跳过（不影响训练）：{e}")
            quality_weights = None
        
        # 移除标签为NaN的行
        valid_mask = y.notna()
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        if trend_position_weight is not None:
            trend_position_weight = trend_position_weight[valid_mask].copy()
        if quality_weights is not None:
            quality_weights = quality_weights[valid_mask].copy()
        if symbols_series is not None:
            symbols_series = symbols_series[valid_mask].copy()
        if time_weights is not None:
            time_weights = time_weights[valid_mask].copy()
        
        # 方案B：数据去噪过滤
        # 1) 基于R2/TNR的质量过滤（对趋势样本生效，无趋势样本不过滤）
        try:
            if getattr(config, 'USE_R2_TNR_FILTER', False):
                r2_cols = [col for col in ['R2_10D', 'R2_5D'] if col in df.columns]
                r2_series = None
                if len(r2_cols) > 0:
                    r2_series = df.loc[valid_mask, r2_cols[0]].astype(float)
                r2_thr = float(getattr(config, 'FILTER_R2_THRESHOLD', 0.5))
                tnr_thr = float(getattr(config, 'FILTER_TNR_THRESHOLD', 1.0))
                # 默认通过
                mask_r2 = pd.Series(True, index=y.index)
                mask_tnr = pd.Series(True, index=y.index)
                if r2_series is not None:
                    # 趋势样本要求R2>=阈值；无趋势样本保留
                    mask_r2 = ((y == 0) | (r2_series >= r2_thr))
                if 'TNR' in df.columns:
                    tnr_series = df.loc[valid_mask, 'TNR'].astype(float)
                    mask_tnr = ((y == 0) | (tnr_series >= tnr_thr))
                mask_quality = mask_r2 & mask_tnr
                before = len(y)
                X = X[mask_quality].copy()
                y = y[mask_quality].copy()
                if trend_position_weight is not None:
                    trend_position_weight = trend_position_weight[mask_quality].copy()
                if quality_weights is not None:
                    quality_weights = quality_weights[mask_quality].copy()
                if symbols_series is not None:
                    symbols_series = symbols_series[mask_quality].copy()
                if time_weights is not None:
                    time_weights = time_weights[mask_quality].copy()
                after = len(y)
                removed = before - after
                if removed > 0:
                    logger.info(f"基于R2/TNR过滤去噪：移除 {removed} / {before} 样本（仅对趋势样本生效）")
        except Exception as e:
            logger.warning(f"R2/TNR过滤失败（忽略，不影响训练）：{e}")
        
        # 2) 模糊样本过滤（近似）：对无趋势且R2位于[amb_min, amb_max]区间的样本移除
        try:
            if getattr(config, 'FILTER_AMBIGUOUS_SAMPLES', False):
                r2_cols = [col for col in ['R2_10D', 'R2_5D'] if col in df.columns]
                if len(r2_cols) > 0:
                    r2_series2 = df.loc[y.index, r2_cols[0]].astype(float)
                    amb_min = float(getattr(config, 'LABEL_AMBIGUOUS_MIN_RATIO', 0.45))
                    amb_max = float(getattr(config, 'LABEL_AMBIGUOUS_MAX_RATIO', 0.55))
                    amb_mask = ~((y == 0) & (r2_series2.between(amb_min, amb_max, inclusive='both')))
                    before = len(y)
                    X = X[amb_mask].copy()
                    y = y[amb_mask].copy()
                    if trend_position_weight is not None:
                        trend_position_weight = trend_position_weight[amb_mask].copy()
                    if quality_weights is not None:
                        quality_weights = quality_weights[amb_mask].copy()
                    if symbols_series is not None:
                        symbols_series = symbols_series[amb_mask].copy()
                    if time_weights is not None:
                        time_weights = time_weights[amb_mask].copy()
                    after = len(y)
                    removed = before - after
                    if removed > 0:
                        logger.info(f"模糊样本过滤（无趋势且R2∈[{amb_min},{amb_max}]）：移除 {removed} / {before} 样本")
        except Exception as e:
            logger.warning(f"模糊样本过滤失败（忽略，不影响训练）：{e}")
        
        # 根据模型类型处理标签
        if use_3_class:
            # 三分类：保留原始标签（0=无趋势, 1=上涨, -1=下跌）
            # 但XGBoost需要从0开始的连续整数，所以需要转换：-1 -> 2
            y = y.astype(int)
            # 将-1转换为2（XGBoost要求类别从0开始：0=无趋势, 1=上涨, 2=下跌）
            y = y.replace(-1, 2)
            
            # 检查标签值
            unique_labels = y.unique()
            if not all(label in [0, 1, 2] for label in unique_labels):
                unexpected_labels = [label for label in unique_labels if label not in [0, 1, 2]]
                logger.warning(f"⚠️ 发现意外的标签值: {unexpected_labels}，这些标签将被忽略")
            
            # 三分类过滤
            filter_class_0 = getattr(config, 'FILTER_LABEL_CLASS_0', False)
            filter_class_1 = getattr(config, 'FILTER_LABEL_CLASS_1', False)
            filter_class_neg1 = getattr(config, 'FILTER_LABEL_CLASS_NEG1', False)
            
            if filter_class_0 or filter_class_1 or filter_class_neg1:
                original_count = len(y)
                filter_mask = pd.Series(True, index=y.index)
                
                if filter_class_0:
                    filter_mask = filter_mask & (y != 0)
                    logger.info(f"过滤label=0（无趋势）的样本")
                
                if filter_class_1:
                    filter_mask = filter_mask & (y != 1)
                    logger.info(f"过滤label=1（上涨趋势）的样本")
                
                if filter_class_neg1:
                    filter_mask = filter_mask & (y != 2)  # 注意：-1已经转换为2
                    logger.info(f"过滤label=-1（下跌趋势）的样本")
                
                # 应用过滤
                X = X[filter_mask].copy()
                y = y[filter_mask].copy()
                if trend_position_weight is not None:
                    trend_position_weight = trend_position_weight[filter_mask].copy()
                if quality_weights is not None:
                    quality_weights = quality_weights[filter_mask].copy()
                if symbols_series is not None:
                    symbols_series = symbols_series[filter_mask].copy()
                if time_weights is not None:
                    time_weights = time_weights[filter_mask].copy()
                
                filtered_count = original_count - len(y)
                if filtered_count > 0:
                    logger.info(f"已过滤 {filtered_count} 个样本（原始: {original_count}, 剩余: {len(y)}）")
        else:
            # 二分类：将标签转换为0（无趋势）和1（有趋势）
            y = y.astype(int)
            # 将-1转换为1（下跌趋势和上涨趋势都算作有趋势）
            y = y.replace(-1, 1)
            
            # 二分类过滤
            filter_class_0 = getattr(config, 'FILTER_PROCESSED_LABEL_CLASS_0', False)
            filter_class_1 = getattr(config, 'FILTER_PROCESSED_LABEL_CLASS_1', False)
            
            if filter_class_0 or filter_class_1:
                original_count = len(y)
                filter_mask = pd.Series(True, index=y.index)
                
                if filter_class_0:
                    filter_mask = filter_mask & (y != 0)
                    logger.info(f"过滤processed_label=0的样本")
                
                if filter_class_1:
                    filter_mask = filter_mask & (y != 1)
                    logger.info(f"过滤processed_label=1的样本")
                
                # 应用过滤
                X = X[filter_mask].copy()
                y = y[filter_mask].copy()
                if trend_position_weight is not None:
                    trend_position_weight = trend_position_weight[filter_mask].copy()
                if quality_weights is not None:
                    quality_weights = quality_weights[filter_mask].copy()
                if symbols_series is not None:
                    symbols_series = symbols_series[filter_mask].copy()
                
                filtered_count = original_count - len(y)
                if filtered_count > 0:
                    logger.info(f"已过滤 {filtered_count} 个样本（原始: {original_count}, 剩余: {len(y)}）")
        
        # 处理特征中的NaN和inf值
        # 先替换inf为NaN，然后统一填充为0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 检查并处理过大的值（超过XGBoost能处理的范围）
        # XGBoost通常能处理float32范围的值，但为了安全，我们将超过1e10的值截断
        max_value = 1e10
        min_value = -1e10
        X = X.clip(lower=min_value, upper=max_value)
        
        # 记录处理情况
        inf_count = (X == np.inf).sum().sum() + (X == -np.inf).sum().sum()
        if inf_count > 0:
            logger.warning(f"⚠️ 发现 {inf_count} 个inf值，已替换为0")
        
        large_value_count = ((X > max_value) | (X < min_value)).sum().sum()
        if large_value_count > 0:
            logger.warning(f"⚠️ 发现 {large_value_count} 个过大值，已截断到[{min_value}, {max_value}]")
        
        logger.info(f"\n数据统计:")
        logger.info(f"  有效样本数: {len(X)}")
        if use_3_class:
            logger.info(f"  预测目标分布（label，0=无趋势, 1=上涨, 2=下跌）:\n{y.value_counts().sort_index()}")
        else:
            logger.info(f"  预测目标分布（processed_label，0=无趋势, 1=有趋势）:\n{y.value_counts().sort_index()}")
        
        # 检查是否有意外的标签值
        if use_3_class:
            expected_labels = [0, 1, 2]
            label_names = {0: '无趋势', 1: '上涨趋势', 2: '下跌趋势'}
        else:
            expected_labels = [0, 1]
            label_names = {0: '无趋势', 1: '有趋势'}
        
        unique_labels = y.unique()
        if not all(label in expected_labels for label in unique_labels):
            unexpected_labels = [label for label in unique_labels if label not in expected_labels]
            logger.warning(f"⚠️ 发现意外的标签值: {unexpected_labels}，这些标签将被忽略或可能导致错误")
        
        # 提取趋势位置权重（如果存在）
        pos_weights = None
        if trend_position_weight is not None:
            pos_weights = trend_position_weight.copy()
            logger.info(f"  趋势位置权重范围: [{pos_weights.min():.2f}, {pos_weights.max():.2f}]")
            logger.info(f"  趋势位置权重统计: 均值={pos_weights.mean():.2f}, 中位数={pos_weights.median():.2f}")
        if quality_weights is not None:
            logger.info(f"  质量权重范围: [{quality_weights.min():.2f}, {quality_weights.max():.2f}]")
        
        return X, y, feature_columns, pos_weights, quality_weights, time_weights, symbols_series, use_3_class
    
    def upsample_down_trend(self, X_train, y_train, pos_weights=None, quality_weights=None, time_weights=None, symbols_train=None, use_smote=None):
        """
        上采样下跌趋势样本，使上涨和下跌样本数量接近
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签（0=无趋势, 1=上涨, 2=下跌）
            pos_weights: 趋势位置权重
            quality_weights: 质量权重
            time_weights: 时间权重
            symbols_train: 标的序列
            use_smote: 是否使用SMOTE（None时从config读取，如果SMOTE不可用则使用随机重复）
        
        Returns:
            X_train_upsampled, y_train_upsampled, pos_weights_upsampled, quality_weights_upsampled, 
            time_weights_upsampled, symbols_train_upsampled
        """
        if use_smote is None:
            use_smote = getattr(config, 'USE_SMOTE_UPSAMPLING', False) and SMOTE_AVAILABLE
        
        y_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        # 统计各类别数量
        up_mask = (y_array == 1)
        down_mask = (y_array == 2)
        up_count = up_mask.sum()
        down_count = down_mask.sum()
        
        if down_count == 0:
            logger.warning("下跌趋势样本数为0，跳过上采样")
            return X_train, y_train, pos_weights, quality_weights, time_weights, symbols_train
        
        if up_count == 0:
            logger.warning("上涨趋势样本数为0，跳过上采样")
            return X_train, y_train, pos_weights, quality_weights, time_weights, symbols_train
        
        logger.info(f"上采样前：上涨={up_count}, 下跌={down_count}, 比例={down_count/up_count:.2f}")
        
        # 计算需要上采样的数量
        target_down_count = up_count  # 目标：下跌样本数 = 上涨样本数
        n_upsample = target_down_count - down_count
        
        if n_upsample <= 0:
            logger.info(f"下跌样本数({down_count})已>=上涨样本数({up_count})，无需上采样")
            return X_train, y_train, pos_weights, quality_weights, time_weights, symbols_train
        
        logger.info(f"需要上采样 {n_upsample} 个下跌趋势样本")
        
        # 获取下跌趋势样本的索引
        down_indices = np.where(down_mask)[0]
        
        if use_smote:
            # 使用SMOTE上采样
            try:
                logger.info("使用SMOTE进行上采样...")
                # 只对趋势样本（上涨+下跌）进行SMOTE，保留无趋势样本
                trend_mask = (y_array == 1) | (y_array == 2)
                X_trend = X_train[trend_mask].values
                y_trend = y_array[trend_mask]
                
                # SMOTE需要将标签转换为0和1（0=上涨，1=下跌）
                y_trend_binary = (y_trend == 2).astype(int)
                
                # 计算需要上采样的数量（SMOTE的sampling_strategy使用比例或绝对数量）
                smote = SMOTE(sampling_strategy={1: up_count}, random_state=42)  # 使下跌样本数=上涨样本数
                X_trend_upsampled, y_trend_binary_upsampled = smote.fit_resample(X_trend, y_trend_binary)
                
                # 转换回原始标签（1=上涨，2=下跌）
                y_trend_upsampled = np.where(y_trend_binary_upsampled == 1, 2, 1)
                
                # 合并无趋势样本
                no_trend_mask = (y_array == 0)
                X_no_trend = X_train[no_trend_mask].values
                y_no_trend = y_array[no_trend_mask]
                
                X_upsampled = np.vstack([X_trend_upsampled, X_no_trend])
                y_upsampled = np.hstack([y_trend_upsampled, y_no_trend])
                
                # 转换为DataFrame/Series
                X_train_upsampled = pd.DataFrame(X_upsampled, columns=X_train.columns, index=range(len(X_upsampled)))
                y_train_upsampled = pd.Series(y_upsampled) if isinstance(y_train, pd.Series) else y_upsampled
                
                logger.info(f"SMOTE上采样完成：上涨={(y_upsampled==1).sum()}, 下跌={(y_upsampled==2).sum()}, 无趋势={(y_upsampled==0).sum()}")
                
                # 权重需要重新计算（简化处理：新样本使用平均值）
                if pos_weights is not None:
                    pos_weights_avg = pos_weights[down_mask].mean()
                    pos_weights_new = np.full(n_upsample, pos_weights_avg)
                    pos_weights_upsampled = np.hstack([pos_weights.values, pos_weights_new]) if isinstance(pos_weights, pd.Series) else np.hstack([pos_weights, pos_weights_new])
                    pos_weights_upsampled = pd.Series(pos_weights_upsampled) if isinstance(pos_weights, pd.Series) else pos_weights_upsampled
                else:
                    pos_weights_upsampled = None
                
                if quality_weights is not None:
                    quality_weights_avg = quality_weights[down_mask].mean()
                    quality_weights_new = np.full(n_upsample, quality_weights_avg)
                    quality_weights_upsampled = np.hstack([quality_weights.values, quality_weights_new]) if isinstance(quality_weights, pd.Series) else np.hstack([quality_weights, quality_weights_new])
                    quality_weights_upsampled = pd.Series(quality_weights_upsampled) if isinstance(quality_weights, pd.Series) else quality_weights_upsampled
                else:
                    quality_weights_upsampled = None
                
                if time_weights is not None:
                    time_weights_avg = time_weights[down_mask].mean()
                    time_weights_new = np.full(n_upsample, time_weights_avg)
                    time_weights_upsampled = np.hstack([time_weights.values, time_weights_new]) if isinstance(time_weights, pd.Series) else np.hstack([time_weights, time_weights_new])
                    time_weights_upsampled = pd.Series(time_weights_upsampled) if isinstance(time_weights, pd.Series) else time_weights_upsampled
                else:
                    time_weights_upsampled = None
                
                if symbols_train is not None:
                    symbols_avg = symbols_train[down_mask].mode()[0] if len(symbols_train[down_mask].mode()) > 0 else symbols_train[down_mask].iloc[0]
                    symbols_new = pd.Series([symbols_avg] * n_upsample)
                    symbols_train_upsampled = pd.concat([symbols_train, symbols_new], ignore_index=True)
                else:
                    symbols_train_upsampled = None
                
                return X_train_upsampled, y_train_upsampled, pos_weights_upsampled, quality_weights_upsampled, time_weights_upsampled, symbols_train_upsampled
            except Exception as e:
                logger.warning(f"SMOTE上采样失败，回退到随机重复: {e}")
                use_smote = False
        
        # 使用随机重复上采样
        if not use_smote:
            logger.info("使用随机重复进行上采样...")
            # 随机选择需要重复的样本
            np.random.seed(42)
            selected_indices = np.random.choice(down_indices, size=n_upsample, replace=True)
            
            # 重复选中的样本
            X_down_selected = X_train.iloc[selected_indices]
            y_down_selected = y_array[selected_indices]
            
            # 合并
            X_train_upsampled = pd.concat([X_train, X_down_selected], ignore_index=True)
            if isinstance(y_train, pd.Series):
                y_train_upsampled = pd.concat([y_train, pd.Series(y_down_selected)], ignore_index=True)
            else:
                y_train_upsampled = np.hstack([y_array, y_down_selected])
            
            logger.info(f"随机重复上采样完成：上涨={(y_train_upsampled==1).sum()}, 下跌={(y_train_upsampled==2).sum()}, 无趋势={(y_train_upsampled==0).sum()}")
            
            # 处理权重
            if pos_weights is not None:
                pos_weights_selected = pos_weights.iloc[selected_indices] if isinstance(pos_weights, pd.Series) else pos_weights[selected_indices]
                pos_weights_upsampled = pd.concat([pos_weights, pos_weights_selected], ignore_index=True) if isinstance(pos_weights, pd.Series) else np.hstack([pos_weights, pos_weights_selected])
            else:
                pos_weights_upsampled = None
            
            if quality_weights is not None:
                quality_weights_selected = quality_weights.iloc[selected_indices] if isinstance(quality_weights, pd.Series) else quality_weights[selected_indices]
                quality_weights_upsampled = pd.concat([quality_weights, quality_weights_selected], ignore_index=True) if isinstance(quality_weights, pd.Series) else np.hstack([quality_weights, quality_weights_selected])
            else:
                quality_weights_upsampled = None
            
            if time_weights is not None:
                time_weights_selected = time_weights.iloc[selected_indices] if isinstance(time_weights, pd.Series) else time_weights[selected_indices]
                time_weights_upsampled = pd.concat([time_weights, time_weights_selected], ignore_index=True) if isinstance(time_weights, pd.Series) else np.hstack([time_weights, time_weights_selected])
            else:
                time_weights_upsampled = None
            
            if symbols_train is not None:
                symbols_selected = symbols_train.iloc[selected_indices]
                symbols_train_upsampled = pd.concat([symbols_train, symbols_selected], ignore_index=True)
            else:
                symbols_train_upsampled = None
            
            return X_train_upsampled, y_train_upsampled, pos_weights_upsampled, quality_weights_upsampled, time_weights_upsampled, symbols_train_upsampled
    
    def calculate_class_weights(self, y_train, use_3_class=True):
        """
        计算类别权重
        权重倍数从config中读取
        
        Args:
            y_train: 训练集标签
                - 三分类：0=无趋势, 1=上涨, 2=下跌
                - 二分类：0=无趋势, 1=有趋势
            use_3_class: 是否使用三分类
        
        Returns:
            dict: 类别权重字典
        """
        # 统计各类别数量
        if isinstance(y_train, pd.Series):
            y_series = y_train.copy()
        else:
            y_series = pd.Series(y_train)
        
        class_counts = y_series.value_counts().sort_index()
        total = len(y_series)
        
        if use_3_class:
            # 三分类：从config读取权重倍数
            weight_no_trend = getattr(config, 'CLASS_WEIGHT_NO_TREND_3', 1.0)
            weight_up_trend = getattr(config, 'CLASS_WEIGHT_UP_TREND_3', 1.0)
            weight_down_trend = getattr(config, 'CLASS_WEIGHT_DOWN_TREND_3', 2.0)
            
            # 检查是否有意外的标签值
            unique_labels = y_series.unique()
            if not all(label in [0, 1, 2] for label in unique_labels):
                unexpected_labels = [label for label in unique_labels if label not in [0, 1, 2]]
                logger.warning(f"⚠️ calculate_class_weights: 发现意外的标签值: {unexpected_labels}，这些标签将被忽略")
            
            # 计算基础权重（逆频率）
            # 权重 = 总样本数 / (类别数 * 该类别样本数)
            weights = {}
            weight_configs = {
                0: (weight_no_trend, '无趋势'),
                1: (weight_up_trend, '上涨趋势'),
                2: (weight_down_trend, '下跌趋势')
            }
            
            for class_idx in [0, 1, 2]:
                count = class_counts.get(class_idx, 0)
                weight_mult, class_name = weight_configs[class_idx]
                if count > 0:
                    # 基础权重：逆频率
                    base_weight = total / (3 * count)
                    # 应用配置的权重倍数
                    weights[class_idx] = base_weight * weight_mult
                else:
                    weights[class_idx] = 1.0
            
            logger.info(f"类别权重计算（三分类，从config读取权重倍数）:")
            for class_idx in [0, 1, 2]:
                weight_mult, class_name = weight_configs[class_idx]
                logger.info(f"  类别{class_idx}（{class_name}）: {weights[class_idx]:.4f}, 样本数: {class_counts.get(class_idx, 0)}, 权重倍数: {weight_mult}")
        else:
            # 二分类：从config读取权重倍数
            # 如果是Stage2（涨/跌二分类），使用专门的权重配置
            # 如果是Stage1（是否趋势二分类），使用通用权重配置
            weight_no_trend = getattr(config, 'CLASS_WEIGHT_NO_TREND', 1.0)
            weight_trend = getattr(config, 'CLASS_WEIGHT_TREND', 2.0)
            
            # 检查是否是Stage2（涨/跌二分类），如果是则使用专门的权重
            # Stage2标签：0=上涨, 1=下跌
            if len(y_series.unique()) == 2 and all(label in [0, 1] for label in y_series.unique()):
                # 可能是Stage2，检查是否有专门的权重配置
                use_detailed = getattr(config, 'USE_STAGE2_DETAILED_WEIGHTS', False)
                weight_up = getattr(config, 'CLASS_WEIGHT_UP_TREND', None)
                weight_down = getattr(config, 'CLASS_WEIGHT_DOWN_TREND', None)
                
                if use_detailed:
                    # 使用细分权重配置
                    weight_down_tp = getattr(config, 'STAGE2_WEIGHT_DOWN_TP', weight_down if weight_down is not None else 1.0)
                    weight_up_tp = getattr(config, 'STAGE2_WEIGHT_UP_TP', weight_up if weight_up is not None else 1.0)
                    
                    class_counts = y_series.value_counts().sort_index()
                    total = len(y_series)
                    weights = {}
                    for class_idx in [0, 1]:
                        count = class_counts.get(class_idx, 0)
                        if count > 0:
                            base_weight = total / (2 * count)
                            if class_idx == 0:  # 上涨
                                weights[class_idx] = base_weight * weight_up_tp
                            else:  # 下跌
                                weights[class_idx] = base_weight * weight_down_tp
                        else:
                            weights[class_idx] = 1.0
                    
                    logger.info(f"类别权重计算（Stage2涨/跌二分类 - 细分权重模式）:")
                    logger.info(f"  类别0（上涨，TP权重）: {weights[0]:.4f}, 样本数: {class_counts.get(0, 0)}, 权重倍数: {weight_up_tp}")
                    logger.info(f"  类别1（下跌，TP权重）: {weights[1]:.4f}, 样本数: {class_counts.get(1, 0)}, 权重倍数: {weight_down_tp}")
                    logger.info(f"  细分权重配置:")
                    logger.info(f"    下跌TP（正确预测下跌）: {weight_down_tp}")
                    logger.info(f"    下跌FN（漏报下跌）: {getattr(config, 'STAGE2_WEIGHT_DOWN_FN', 1.0)} (用于损失函数)")
                    logger.info(f"    下跌FP（误报下跌）: {getattr(config, 'STAGE2_WEIGHT_DOWN_FP', 1.0)} (用于损失函数)")
                    logger.info(f"    上涨TP（正确预测上涨）: {weight_up_tp}")
                    logger.info(f"    上涨FN（漏报上涨）: {getattr(config, 'STAGE2_WEIGHT_UP_FN', 1.0)} (用于损失函数)")
                    logger.info(f"    上涨FP（误报上涨）: {getattr(config, 'STAGE2_WEIGHT_UP_FP', 1.0)} (用于损失函数)")
                    return weights
                elif weight_up is not None and weight_down is not None:
                    # 使用传统权重配置
                    class_counts = y_series.value_counts().sort_index()
                    total = len(y_series)
                    weights = {}
                    for class_idx in [0, 1]:
                        count = class_counts.get(class_idx, 0)
                        if count > 0:
                            base_weight = total / (2 * count)
                            if class_idx == 0:  # 上涨
                                weights[class_idx] = base_weight * weight_up
                            else:  # 下跌
                                weights[class_idx] = base_weight * weight_down
                        else:
                            weights[class_idx] = 1.0
                    logger.info(f"类别权重计算（Stage2涨/跌二分类 - 传统权重模式）:")
                    logger.info(f"  类别0（上涨）: {weights[0]:.4f}, 样本数: {class_counts.get(0, 0)}, 权重倍数: {weight_up}")
                    logger.info(f"  类别1（下跌）: {weights[1]:.4f}, 样本数: {class_counts.get(1, 0)}, 权重倍数: {weight_down}")
                    return weights
            
            # 如果标签中有-1（下跌趋势），将其转换为1（有趋势）
            y_series = y_series.replace(-1, 1)
            class_counts = y_series.value_counts().sort_index()
            
            # 检查是否有意外的标签值
            unique_labels = y_series.unique()
            if not all(label in [0, 1] for label in unique_labels):
                unexpected_labels = [label for label in unique_labels if label not in [0, 1]]
                logger.warning(f"⚠️ calculate_class_weights: 发现意外的标签值: {unexpected_labels}，这些标签将被忽略")
            
            # 计算基础权重（逆频率）
            weights = {}
            for class_idx in [0, 1]:
                count = class_counts.get(class_idx, 0)
                if count > 0:
                    # 基础权重：逆频率
                    base_weight = total / (2 * count)
                    # 应用配置的权重倍数
                    if class_idx == 0:  # 无趋势
                        weights[class_idx] = base_weight * weight_no_trend
                    else:  # 1 (有趋势)
                        weights[class_idx] = base_weight * weight_trend
                else:
                    weights[class_idx] = 1.0
            
            logger.info(f"类别权重计算（二分类，从config读取权重倍数）:")
            logger.info(f"  类别0（无趋势）: {weights[0]:.4f}, 样本数: {class_counts.get(0, 0)}, 权重倍数: {weight_no_trend}")
            logger.info(f"  类别1（有趋势）: {weights[1]:.4f}, 样本数: {class_counts.get(1, 0)}, 权重倍数: {weight_trend}")
        
        return weights
    
    def train_model(self, X_train, y_train, X_val, y_val, params=None, use_class_weights=True, pos_weights=None, quality_weights=None, time_weights=None, use_3_class=True):
        """
        训练XGBoost模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
                - 三分类：0=无趋势, 1=上涨, 2=下跌
                - 二分类：0=无趋势, 1=有趋势
            X_val: 验证集特征
            y_val: 验证集标签
            params: XGBoost参数，如果为None则使用默认参数
            use_class_weights: 是否使用类别权重
            pos_weights: 趋势位置权重
            use_3_class: 是否使用三分类
        
        Returns:
            训练好的模型
        """
        if params is None:
            if use_3_class:
                # 从config读取超参数（如果存在），否则使用默认值
                params = {
                    'objective': 'multi:softprob',  # 三分类问题
                    'num_class': 3,  # 类别数：0=无趋势, 1=上涨, 2=下跌
                    'eval_metric': 'mlogloss',  # 多分类对数损失
                    'max_depth': getattr(config, 'XGB_MAX_DEPTH', 3),
                    'learning_rate': getattr(config, 'XGB_LEARNING_RATE', 0.05),
                    'min_child_weight': getattr(config, 'XGB_MIN_CHILD_WEIGHT', 3),
                    'gamma': getattr(config, 'XGB_GAMMA', 0.3),
                    'subsample': getattr(config, 'XGB_SUBSAMPLE', 0.7),
                    'colsample_bytree': getattr(config, 'XGB_COLSAMPLE_BYTREE', 0.7),
                    'reg_lambda': getattr(config, 'XGB_REG_LAMBDA', 2.0),
                    'reg_alpha': getattr(config, 'XGB_REG_ALPHA', 0.0),
                    'random_state': 42,
                    'n_jobs': -1
                }
            else:
                params = {
                    'objective': 'binary:logistic',  # 二分类问题
                    'eval_metric': 'logloss',
                    'max_depth': 3,  # 降低深度，防止过拟合
                    'learning_rate': 0.05,  # 降低学习率
                    'min_child_weight': 3,  # 提高，防止过拟合
                    'gamma': 0.3,  # 添加正则化
                    'subsample': 0.7,  # 降低采样比例
                    'colsample_bytree': 0.7,  # 降低特征采样比例
                    'reg_lambda': 2.0,  # L2正则化
                    'reg_alpha': 0.0,
                    'random_state': 42,
                    'n_jobs': -1
                }
        
        logger.info("开始训练XGBoost模型...")
        logger.info(f"模型类型: {'三分类' if use_3_class else '二分类'}")
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"验证集大小: {len(X_val)}")
        logger.info(f"模型参数: {params}")
        
        # 确保标签格式正确
        if isinstance(y_train, pd.Series):
            y_train = y_train.copy()
        else:
            y_train = pd.Series(y_train)
        
        if isinstance(y_val, pd.Series):
            y_val = y_val.copy()
        else:
            y_val = pd.Series(y_val)
        
        if not use_3_class:
            # 二分类：将-1转换为1
            y_train = y_train.replace(-1, 1)
            y_val = y_val.replace(-1, 1)
        
        # 转换为numpy数组
        y_train = y_train.values
        y_val = y_val.values
        
        # 检查标签分布
        train_class_counts = pd.Series(y_train).value_counts().sort_index()
        if use_3_class:
            class_names = {0: '无趋势', 1: '上涨趋势', 2: '下跌趋势'}
            expected_classes = [0, 1, 2]
        else:
            class_names = {0: '无趋势', 1: '有趋势'}
            expected_classes = [0, 1]
        
        logger.info(f"训练集标签分布:")
        for class_idx in expected_classes:
            count = train_class_counts.get(class_idx, 0)
            pct = count / len(y_train) * 100 if len(y_train) > 0 else 0
            logger.info(f"  类别{class_idx} ({class_names[class_idx]}): {count} ({pct:.2f}%)")
        
        # 检查验证集标签分布
        val_class_counts = pd.Series(y_val).value_counts().sort_index()
        logger.info(f"验证集标签分布:")
        for class_idx in expected_classes:
            count = val_class_counts.get(class_idx, 0)
            pct = count / len(y_val) * 100 if len(y_val) > 0 else 0
            logger.info(f"  类别{class_idx} ({class_names[class_idx]}): {count} ({pct:.2f}%)")
        
        # 检查是否有足够的类别
        if len(train_class_counts) < len(expected_classes):
            missing_classes = [c for c in expected_classes if c not in train_class_counts.index]
            logger.warning(f"⚠️ 训练集缺少类别 {missing_classes}！只有类别 {train_class_counts.index.tolist()}")
            logger.warning(f"这会导致模型无法学习{'三分类' if use_3_class else '二分类'}任务")
        
        # 检查并处理特征中的inf和过大值
        # 训练集
        X_train_inf = (X_train == np.inf).sum().sum() + (X_train == -np.inf).sum().sum()
        if X_train_inf > 0:
            logger.warning(f"⚠️ 训练集中发现 {X_train_inf} 个inf值，正在处理...")
            X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 检查过大的值
        max_value = 1e10
        min_value = -1e10
        X_train_large = ((X_train > max_value) | (X_train < min_value)).sum().sum()
        if X_train_large > 0:
            logger.warning(f"⚠️ 训练集中发现 {X_train_large} 个过大值，正在截断...")
            X_train = X_train.clip(lower=min_value, upper=max_value)
        
        # 验证集
        X_val_inf = (X_val == np.inf).sum().sum() + (X_val == -np.inf).sum().sum()
        if X_val_inf > 0:
            logger.warning(f"⚠️ 验证集中发现 {X_val_inf} 个inf值，正在处理...")
            X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        X_val_large = ((X_val > max_value) | (X_val < min_value)).sum().sum()
        if X_val_large > 0:
            logger.warning(f"⚠️ 验证集中发现 {X_val_large} 个过大值，正在截断...")
            X_val = X_val.clip(lower=min_value, upper=max_value)
        
        # 检查训练集是否为空
        if len(X_train) == 0 or len(y_train) == 0:
            logger.error("❌ 训练集为空，无法训练模型！")
            logger.error("可能原因：过滤后没有剩余样本，请检查过滤配置")
            return None
        
        # 计算并应用类别权重
        sample_weights = None
        if use_class_weights:
            class_weights = self.calculate_class_weights(y_train, use_3_class=use_3_class)
            # 为每个样本分配对应的类别权重
            class_sample_weights = np.array([class_weights[label] for label in y_train])
            
            # 检查数组是否为空
            if len(class_sample_weights) == 0:
                logger.error("❌ 类别权重数组为空，无法继续训练！")
                return None
            
            # 结合趋势位置权重与质量权重
            if pos_weights is not None:
                # 确保长度一致
                if len(pos_weights) == len(class_sample_weights):
                    sample_weights = class_sample_weights * pos_weights.values
                    logger.info(f"使用类别权重 × 趋势位置权重")
                    if len(class_sample_weights) > 0:
                        logger.info(f"  类别权重范围: [{class_sample_weights.min():.4f}, {class_sample_weights.max():.4f}]")
                    if len(pos_weights) > 0:
                        logger.info(f"  趋势位置权重范围: [{pos_weights.min():.2f}, {pos_weights.max():.2f}]")
                else:
                    logger.warning(f"趋势位置权重长度({len(pos_weights)})与样本数({len(class_sample_weights)})不匹配，仅使用类别权重")
                    sample_weights = class_sample_weights
            else:
                sample_weights = class_sample_weights
                logger.info(f"使用类别权重（无趋势位置权重）")
                if len(sample_weights) > 0:
                    logger.info(f"  样本权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
            # 叠加质量权重
            if (quality_weights is not None) and (len(quality_weights) == len(class_sample_weights)):
                sample_weights = sample_weights * quality_weights.values
                logger.info(f"叠加质量权重")
                logger.info(f"  质量权重范围: [{quality_weights.min():.2f}, {quality_weights.max():.2f}]")
            # 叠加时间衰减权重
            if (time_weights is not None) and (len(time_weights) == len(sample_weights)):
                sample_weights = sample_weights * time_weights.values
                logger.info(f"叠加时间衰减权重")
                logger.info(f"  时间权重范围: [{time_weights.min():.2f}, {time_weights.max():.2f}]")
            # 最终范围
            if sample_weights is not None and len(sample_weights) > 0:
                logger.info(f"  最终样本权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
        else:
            # 即使不使用类别权重，也可以使用趋势位置权重
            if pos_weights is not None:
                sample_weights = pos_weights.values
                logger.info(f"仅使用趋势位置权重（未使用类别权重）")
                if len(sample_weights) > 0:
                    logger.info(f"  样本权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
            else:
                logger.info("不使用任何样本权重")
            # 叠加质量权重
            if (quality_weights is not None) and (sample_weights is not None) and (len(quality_weights) == len(sample_weights)):
                sample_weights = sample_weights * quality_weights.values
                logger.info(f"叠加质量权重")
                logger.info(f"  质量权重范围: [{quality_weights.min():.2f}, {quality_weights.max():.2f}]")
            if (time_weights is not None) and (sample_weights is not None) and (len(time_weights) == len(sample_weights)):
                sample_weights = sample_weights * time_weights.values
                logger.info(f"叠加时间衰减权重")
                logger.info(f"  时间权重范围: [{time_weights.min():.2f}, {time_weights.max():.2f}]")
                if len(sample_weights) > 0:
                    logger.info(f"  最终样本权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
        
        # 使用XGBClassifier API，支持early stopping
        # early_stopping_rounds 应该作为初始化参数，而不是 fit() 的参数
        n_estimators = getattr(config, 'XGB_N_ESTIMATORS', 500)
        early_stopping_rounds = getattr(config, 'XGB_EARLY_STOPPING_ROUNDS', 50)
        early_stopping_metric = getattr(config, 'XGB_EARLY_STOPPING_METRIC', 'trend_f1')  # 'trend_f1' 或 'accuracy'
        
        # 检查是否使用自定义early stopping（基于trend_f1或accuracy）
        use_custom_early_stopping = early_stopping_metric in ['trend_f1', 'accuracy']
        
        if use_custom_early_stopping:
            # 使用自定义early stopping：基于验证集的trend_f1或accuracy
            logger.info(f"使用自定义early stopping，基于验证集的 {early_stopping_metric}")
            
            from sklearn.metrics import classification_report, accuracy_score
            
            # 使用XGBoost的train API进行增量训练
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # 训练参数（移除XGBClassifier特有的参数）
            xgb_params = params.copy()
            xgb_params.pop('eval_metric', None)  # 移除eval_metric，我们手动评估
            
            # 分阶段训练：每次训练一定轮数，评估后决定是否继续
            best_score = -1.0
            best_iteration = 0
            no_improve_count = 0
            best_booster = None
            current_iteration = 0
            step_size = 10  # 每次训练的轮数
            
            while current_iteration < n_estimators:
                # 计算本次训练的轮数
                remaining = n_estimators - current_iteration
                n_rounds = min(step_size, remaining)
                
                # 增量训练
                booster = xgb.train(
                    xgb_params,
                    dtrain,
                    num_boost_round=n_rounds,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    xgb_model=best_booster,  # 继续训练
                    verbose_eval=False
                )
                
                current_iteration += n_rounds
                
                # 在验证集上预测
                y_val_pred_proba = booster.predict(dval)
                if use_3_class:
                    y_val_pred = np.argmax(y_val_pred_proba.reshape(-1, 3), axis=1)
                else:
                    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
                
                # 计算指标
                if early_stopping_metric == 'trend_f1':
                    # 计算trend_f1（上涨和下跌的平均F1）
                    try:
                        class_report = classification_report(
                            y_val, y_val_pred,
                            output_dict=True,
                            zero_division=0
                        )
                        up_f1 = class_report.get('1', {}).get('f1-score', 0.0)
                        down_f1 = class_report.get('2', {}).get('f1-score', 0.0)
                        current_score = (up_f1 + down_f1) / 2.0
                    except Exception:
                        current_score = 0.0
                else:  # accuracy
                    current_score = accuracy_score(y_val, y_val_pred)
                
                # 检查是否有改善
                if current_score > best_score:
                    best_score = current_score
                    best_iteration = current_iteration
                    no_improve_count = 0
                    best_booster = booster
                    logger.info(f"  轮数 {current_iteration}/{n_estimators}, {early_stopping_metric}: {current_score:.4f} ⬆️ (新最佳)")
                else:
                    no_improve_count += 1
                    logger.info(f"  轮数 {current_iteration}/{n_estimators}, {early_stopping_metric}: {current_score:.4f} (最佳: {best_score:.4f}, 连续{no_improve_count}轮未改善)")
                
                # Early stopping
                if no_improve_count >= early_stopping_rounds:
                    logger.info(f"Early stopping触发：验证集{early_stopping_metric}连续{early_stopping_rounds}轮未改善")
                    logger.info(f"最佳{early_stopping_metric}: {best_score:.4f} (轮数 {best_iteration})")
                    booster = best_booster
                    break
            
            if no_improve_count < early_stopping_rounds:
                logger.info(f"训练完成：达到最大轮数 {n_estimators}")
                logger.info(f"最佳{early_stopping_metric}: {best_score:.4f} (轮数 {best_iteration})")
                booster = best_booster if best_booster is not None else booster
            
            # 将XGBoost Booster转换为XGBClassifier
            # 直接创建XGBClassifier并设置必要的属性
            model = XGBClassifier()
            # 设置booster
            model._Booster = booster
            # 设置类别信息（使用__dict__绕过只读属性）
            if use_3_class:
                model.__dict__['classes_'] = np.array([0, 1, 2])
                model.__dict__['n_classes_'] = 3
            else:
                model.__dict__['classes_'] = np.array([0, 1])
                model.__dict__['n_classes_'] = 2
            # 设置LabelEncoder（用于预测时的标签转换）
            model._le = LabelEncoder().fit(y_train)
            # 设置最佳迭代和分数（使用__dict__直接设置）
            model.__dict__['best_iteration'] = best_iteration
            model.__dict__['best_score'] = best_score
            
        else:
            # 使用XGBoost默认的early stopping（基于eval_metric）
            logger.info(f"使用XGBoost默认early stopping，基于eval_metric: {params.get('eval_metric', 'mlogloss')}")
        model = XGBClassifier(
            **params,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
        )
        
        # 训练模型，使用early stopping
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        logger.info(f"模型训练完成！最佳轮数: {model.best_iteration + 1 if hasattr(model, 'best_iteration') else 'N/A'}")
        return model
    
    def evaluate_model(self, model, X, y, dataset_name='', use_probability_threshold=True, use_3_class=True, pos_weights=None, symbols=None):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X: 特征
            y: 真实标签
                - 三分类：0=无趋势, 1=上涨, 2=下跌
                - 二分类：0=无趋势, 1=有趋势
            dataset_name: 数据集名称（用于日志）
            use_probability_threshold: 是否使用概率阈值进行预测
            use_3_class: 是否使用三分类
            pos_weights: 趋势位置权重
        
        Returns:
            dict: 评估结果
        """
        # 检查模型是否为None
        if model is None:
            logger.error(f"❌ 模型为None，无法评估{dataset_name}")
            return None
        
        # 预测概率（支持校准与时间平滑）
        y_pred_proba = self._predict_proba(model, X, use_3_class=use_3_class, use_calibration=getattr(config, 'USE_PROB_CALIBRATION', False))
        try:
            y_pred_proba = self._apply_temporal_smoothing(y_pred_proba, symbols=symbols)
            if getattr(config, 'USE_TEMPORAL_SMOOTHING', False):
                logger.info(f"已对概率应用EMA平滑(alpha={getattr(config, 'SMOOTHING_EMA_ALPHA', 0.3)})")
        except Exception as e:
            logger.warning(f"概率时间平滑失败，已跳过：{e}")
        
        # 根据配置或传入阈值选择预测方式
        use_threshold = getattr(config, 'USE_PROBABILITY_THRESHOLD', True) if use_probability_threshold else False
        confidence_threshold = getattr(config, 'PREDICTION_CONFIDENCE_THRESHOLD', 0.6)
        
        # 将标签与预测转换为numpy数组，便于子集筛选
        y_array = np.array(y) if not isinstance(y, np.ndarray) else y
        # 在后续分支中会重新赋值y_pred（numpy数组），此处无需提前转换
        
        if use_3_class:
            # 三分类评估
            if use_threshold:
                # 使用概率阈值：只在模型有信心时才预测趋势
                # 三分类：proba[:, 0] = P(0=无趋势), proba[:, 1] = P(1=上涨), proba[:, 2] = P(2=下跌)
                p_no_trend = y_pred_proba[:, 0]  # P(0=无趋势)
                p_up_trend = y_pred_proba[:, 1]   # P(1=上涨)
                p_down_trend = y_pred_proba[:, 2]  # P(2=下跌)
                
                # 统计概率分布
                logger.info(f"概率分布统计:")
                logger.info(f"  P(无趋势) 均值: {p_no_trend.mean():.4f}, 中位数: {np.median(p_no_trend):.4f}")
                logger.info(f"  P(上涨) 均值: {p_up_trend.mean():.4f}, 中位数: {np.median(p_up_trend):.4f}")
                logger.info(f"  P(下跌) 均值: {p_down_trend.mean():.4f}, 中位数: {np.median(p_down_trend):.4f}")
                
                use_class_thresh = getattr(config, 'USE_CLASS_SPECIFIC_THRESHOLDS', False)
                # 支持动态传入阈值（通过self.dynamic_thresholds）
                dyn_thr = getattr(self, 'dynamic_thresholds', None)
                if dyn_thr is not None:
                    use_class_thresh = True
                    thr0 = float(dyn_thr.get('no_trend', confidence_threshold))
                    thr1 = float(dyn_thr.get('up_trend', confidence_threshold))
                    thr2 = float(dyn_thr.get('down_trend', confidence_threshold))
                elif use_class_thresh:
                    thr0 = getattr(config, 'CLASS_PROB_THRESHOLD_NO_TREND', confidence_threshold)
                    thr1 = getattr(config, 'CLASS_PROB_THRESHOLD_UP_TREND', confidence_threshold)
                    thr2 = getattr(config, 'CLASS_PROB_THRESHOLD_DOWN_TREND', confidence_threshold)
                
                if use_class_thresh:
                    # 使用分类别阈值
                    signals = []
                    for p0, p1, p2 in zip(p_no_trend, p_up_trend, p_down_trend):
                        candidates = []
                        if p0 >= thr0:
                            candidates.append((0, p0))
                        if p1 >= thr1:
                            candidates.append((1, p1))
                        if p2 >= thr2:
                            candidates.append((2, p2))
                        if len(candidates) == 0:
                            # 无类别达到阈值，默认不交易（0）
                            signals.append(0)
                        else:
                            # 取达到各自阈值中概率最高者
                            signals.append(max(candidates, key=lambda t: t[1])[0])
                    y_pred = np.array(signals)
                    logger.info(f"使用分类别阈值预测（无趋势={thr0:.2f}, 上涨={thr1:.2f}, 下跌={thr2:.2f}）")
                else:
                    # 全局阈值：最大概率 >= 阈值则采用对应类别，否则0
                    max_probs = np.max(y_pred_proba, axis=1)
                    max_prob_classes = np.argmax(y_pred_proba, axis=1)
                    signals = []
                    for max_prob, max_class in zip(max_probs, max_prob_classes):
                        if max_prob < confidence_threshold:
                            signals.append(0)
                        else:
                            signals.append(max_class)
                    y_pred = np.array(signals)
                    logger.info(f"使用概率阈值预测（置信度阈值: {confidence_threshold}）")
                logger.info(f"  无趋势数: {(y_pred == 0).sum()} / {len(y_pred)} ({(y_pred == 0).sum()/len(y_pred)*100:.1f}%)")
                logger.info(f"  上涨趋势数: {(y_pred == 1).sum()} / {len(y_pred)} ({(y_pred == 1).sum()/len(y_pred)*100:.1f}%)")
                logger.info(f"  下跌趋势数: {(y_pred == 2).sum()} / {len(y_pred)} ({(y_pred == 2).sum()/len(y_pred)*100:.1f}%)")
            else:
                # 直接预测（三分类）
                y_pred = model.predict(X)
                logger.info(f"使用直接预测（三分类）")
            
            # 检查预测结果分布
            if isinstance(y_pred, np.ndarray):
                pred_counts = pd.Series(y_pred).value_counts().sort_index()
            else:
                pred_counts = y_pred.value_counts().sort_index()
            logger.info(f"预测结果分布:")
            class_names = {0: '无趋势', 1: '上涨趋势', 2: '下跌趋势'}
            for class_idx in [0, 1, 2]:
                count = pred_counts.get(class_idx, 0)
                pct = count / len(y_pred) * 100 if len(y_pred) > 0 else 0
                logger.info(f"  类别{class_idx} ({class_names[class_idx]}): {count} ({pct:.2f}%)")
            
            # 检查真实标签分布
            if isinstance(y, pd.Series):
                true_counts = y.value_counts().sort_index()
            else:
                true_counts = pd.Series(y).value_counts().sort_index()
            logger.info(f"真实标签分布:")
            for class_idx in [0, 1, 2]:
                count = true_counts.get(class_idx, 0)
                pct = count / len(y) * 100 if len(y) > 0 else 0
                logger.info(f"  类别{class_idx} ({class_names[class_idx]}): {count} ({pct:.2f}%)")
            
            # 计算准确率
            accuracy = accuracy_score(y, y_pred)
            
            # 分类报告（三分类）
            try:
                class_report = classification_report(y, y_pred, 
                                                    target_names=['无趋势(0)', '上涨趋势(1)', '下跌趋势(2)'],
                                                    output_dict=True,
                                                    zero_division=0)
            except ValueError as e:
                logger.warning(f"分类报告生成失败: {e}")
                class_report = {
                    '无趋势(0)': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    '上涨趋势(1)': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    '下跌趋势(2)': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
                }
                for class_idx in [0, 1, 2]:
                    if class_idx in true_counts.index:
                        class_report[f"{class_names[class_idx]}({class_idx})"]['support'] = int(true_counts[class_idx])
            
            # 混淆矩阵
            cm = confusion_matrix(y, y_pred)
            
            # 总体（macro）指标
            overall_precision = class_report.get('macro avg', {}).get('precision', 0.0)
            overall_recall = class_report.get('macro avg', {}).get('recall', 0.0)
            overall_f1 = class_report.get('macro avg', {}).get('f1-score', 0.0)
            logger.info("\n总体指标:")
            logger.info(f"  命中率(accuracy): {accuracy:.4f}")
            logger.info(f"  Recall(macro):   {overall_recall:.4f}")
            logger.info(f"  Precision(macro):{overall_precision:.4f}")
            
            # 提取各类别指标
            no_trend_precision = class_report['无趋势(0)']['precision']
            no_trend_recall = class_report['无趋势(0)']['recall']
            no_trend_f1 = class_report['无趋势(0)']['f1-score']
            no_trend_support = class_report['无趋势(0)']['support']
            
            up_trend_precision = class_report['上涨趋势(1)']['precision']
            up_trend_recall = class_report['上涨趋势(1)']['recall']
            up_trend_f1 = class_report['上涨趋势(1)']['f1-score']
            up_trend_support = class_report['上涨趋势(1)']['support']
            
            down_trend_precision = class_report['下跌趋势(2)']['precision']
            down_trend_recall = class_report['下跌趋势(2)']['recall']
            down_trend_f1 = class_report['下跌趋势(2)']['f1-score']
            down_trend_support = class_report['下跌趋势(2)']['support']
            
            logger.info(f"\n{dataset_name}数据集评估结果:")
            logger.info(f"准确率: {accuracy:.4f}")
            logger.info(f"\n无趋势(0):")
            logger.info(f"  Precision: {no_trend_precision:.4f}, Recall: {no_trend_recall:.4f}, F1: {no_trend_f1:.4f}, Support: {int(no_trend_support)}")
            logger.info(f"\n上涨趋势(1):")
            logger.info(f"  Precision: {up_trend_precision:.4f}, Recall: {up_trend_recall:.4f}, F1: {up_trend_f1:.4f}, Support: {int(up_trend_support)}")
            logger.info(f"\n下跌趋势(2):")
            logger.info(f"  Precision: {down_trend_precision:.4f}, Recall: {down_trend_recall:.4f}, F1: {down_trend_f1:.4f}, Support: {int(down_trend_support)}")
            logger.info(f"\n{'='*60}")
            logger.info(f"完整分类报告:\n{classification_report(y, y_pred, target_names=['无趋势(0)', '上涨趋势(1)', '下跌趋势(2)'])}")
            logger.info(f"\n混淆矩阵:\n{cm}")
            
            # 针对高权重样本（trend_position_weight >= 0.7）的子集评估
            subset_metrics = {}
            if pos_weights is not None:
                weight_array = np.array(pos_weights)
                subset_mask = weight_array >= 0.7
                subset_count = subset_mask.sum()
                if subset_count > 0:
                    y_mid = y_array[subset_mask]
                    y_pred_mid = y_pred[subset_mask]
                    try:
                        class_report_mid = classification_report(y_mid, y_pred_mid,
                                                                 target_names=['无趋势(0)', '上涨趋势(1)', '下跌趋势(2)'],
                                                                 output_dict=True,
                                                                 zero_division=0)
                        # 仅趋势样本(1/2)的命中率
                        try:
                            trend_mask_mid = np.isin(y_mid, [1, 2])
                            if trend_mask_mid.any():
                                mid_trend_only_accuracy = float((np.array(y_mid)[trend_mask_mid] == np.array(y_pred_mid)[trend_mask_mid]).mean())
                            else:
                                mid_trend_only_accuracy = 0.0
                        except Exception:
                            mid_trend_only_accuracy = 0.0
                        subset_metrics = {
                            'mid_trend_precision': (class_report_mid['上涨趋势(1)']['precision'] + class_report_mid['下跌趋势(2)']['precision']) / 2,
                            'mid_trend_recall': (class_report_mid['上涨趋势(1)']['recall'] + class_report_mid['下跌趋势(2)']['recall']) / 2,
                            'mid_trend_f1': (class_report_mid['上涨趋势(1)']['f1-score'] + class_report_mid['下跌趋势(2)']['f1-score']) / 2,
                            'mid_accuracy': float((np.array(y_mid) == np.array(y_pred_mid)).mean()),
                            # 分类别高权重识别正确率（Recall和Precision）
                            'mid_up_accuracy': class_report_mid['上涨趋势(1)']['recall'],
                            'mid_down_accuracy': class_report_mid['下跌趋势(2)']['recall'],
                            'mid_no_trend_accuracy': class_report_mid['无趋势(0)']['recall'],
                            'mid_up_precision': class_report_mid['上涨趋势(1)']['precision'],
                            'mid_down_precision': class_report_mid['下跌趋势(2)']['precision'],
                            'mid_no_trend_precision': class_report_mid['无趋势(0)']['precision'],
                            'mid_trend_only_accuracy': mid_trend_only_accuracy,
                        }
                        logger.info(f"\n{dataset_name} 高权重样本评估（trend_position_weight >= 0.7，样本数: {subset_count}）:")
                        logger.info(f"  趋势Precision: {subset_metrics['mid_trend_precision']:.4f}")
                        logger.info(f"  趋势Recall:    {subset_metrics['mid_trend_recall']:.4f}")
                        logger.info(f"  趋势F1:        {subset_metrics['mid_trend_f1']:.4f}")
                        logger.info(f"  命中率(accuracy): {subset_metrics['mid_accuracy']:.4f}")
                        logger.info(f"  趋势内命中率(仅1/2): {subset_metrics['mid_trend_only_accuracy']:.4f}")
                    except ValueError as e:
                        logger.warning(f"高权重样本分类报告生成失败: {e}")
                else:
                    logger.info(f"{dataset_name} 中 trend_position_weight >= 0.7 的样本数为0，跳过子集评估")
            else:
                logger.info(f"{dataset_name} 未提供趋势位置权重，跳过高权重子集评估")

            result = {
                'accuracy': accuracy,
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1': overall_f1,
                'y_true': y,
                'y_pred': y_pred,  # 返回0（无趋势）、1（上涨）、2（下跌）
                'y_proba': y_pred_proba,
                # 各类别指标
                'no_trend_precision': no_trend_precision,
                'no_trend_recall': no_trend_recall,
                'no_trend_f1': no_trend_f1,
                'up_trend_precision': up_trend_precision,
                'up_trend_recall': up_trend_recall,
                'up_trend_f1': up_trend_f1,
                'down_trend_precision': down_trend_precision,
                'down_trend_recall': down_trend_recall,
                'down_trend_f1': down_trend_f1,
                # 为了兼容性，保留trend_f1（上涨和下跌的平均F1）
                'trend_f1': (up_trend_f1 + down_trend_f1) / 2 if (up_trend_support > 0 or down_trend_support > 0) else 0.0,
                'trend_precision': (up_trend_precision + down_trend_precision) / 2 if (up_trend_support > 0 or down_trend_support > 0) else 0.0,
                'trend_recall': (up_trend_recall + down_trend_recall) / 2 if (up_trend_support > 0 or down_trend_support > 0) else 0.0,
                # 分类别识别正确率（Recall和Precision）
                'up_trend_accuracy': up_trend_recall,  # Recall
                'down_trend_accuracy': down_trend_recall,  # Recall
                'up_trend_precision': up_trend_precision,
                'down_trend_precision': down_trend_precision,
            }
            result.update(subset_metrics)
            return result
        else:
            # 二分类评估（保持原有逻辑）
            if use_threshold:
                # 使用概率阈值：只在模型有信心时才交易
                # 二分类：proba[:, 0] = P(0=无趋势), proba[:, 1] = P(1=有趋势)
                p_no_trend = y_pred_proba[:, 0]  # P(0=无趋势)
                p_trend = y_pred_proba[:, 1]     # P(1=有趋势)
                
                # 统计概率分布，用于诊断
                logger.info(f"概率分布统计:")
                logger.info(f"  P(有趋势) 均值: {p_trend.mean():.4f}, 中位数: {np.median(p_trend):.4f}")
                logger.info(f"  P(有趋势) 范围: [{p_trend.min():.4f}, {p_trend.max():.4f}]")
                logger.info(f"  P(有趋势) >= {confidence_threshold} 的样本数: {(p_trend >= confidence_threshold).sum()} / {len(p_trend)} ({(p_trend >= confidence_threshold).sum()/len(p_trend)*100:.1f}%)")
                
                signals = []
                for p_nt, p_t in zip(p_no_trend, p_trend):
                    if p_t < confidence_threshold:
                        # 置信度不够，不交易（返回0=无趋势）
                        signals.append(0)
                    else:
                        # 有趋势，返回1
                        signals.append(1)
                
                y_pred = np.array(signals)
                logger.info(f"使用概率阈值预测（置信度阈值: {confidence_threshold}）")
                logger.info(f"  有趋势信号数: {(y_pred == 1).sum()} / {len(y_pred)} ({(y_pred == 1).sum()/len(y_pred)*100:.1f}%)")
                logger.info(f"  无趋势数: {(y_pred == 0).sum()} / {len(y_pred)} ({(y_pred == 0).sum()/len(y_pred)*100:.1f}%)")
            else:
                # 直接预测（二分类）
                y_pred = model.predict(X)
                logger.info(f"使用直接预测（二分类）")
            
            # 检查预测结果分布
            if isinstance(y_pred, np.ndarray):
                pred_counts = pd.Series(y_pred).value_counts().sort_index()
            else:
                pred_counts = y_pred.value_counts().sort_index()
            logger.info(f"预测结果分布:")
            for class_idx in [0, 1]:
                count = pred_counts.get(class_idx, 0)
                pct = count / len(y_pred) * 100 if len(y_pred) > 0 else 0
                logger.info(f"  类别{class_idx} ({'无趋势' if class_idx == 0 else '有趋势'}): {count} ({pct:.2f}%)")
            
            # 检查真实标签分布
            if isinstance(y, pd.Series):
                true_counts = y.value_counts().sort_index()
            else:
                true_counts = pd.Series(y).value_counts().sort_index()
            logger.info(f"真实标签分布:")
            for class_idx in [0, 1]:
                count = true_counts.get(class_idx, 0)
                pct = count / len(y) * 100 if len(y) > 0 else 0
                logger.info(f"  类别{class_idx} ({'无趋势' if class_idx == 0 else '有趋势'}): {count} ({pct:.2f}%)")
            
            # 计算准确率
            accuracy = accuracy_score(y, y_pred)
            
            # 分类报告（二分类：无趋势 vs 有趋势）
            try:
                class_report = classification_report(y, y_pred, 
                                                    target_names=['无趋势(0)', '有趋势(1)'],
                                                    output_dict=True,
                                                    zero_division=0)
            except ValueError as e:
                logger.warning(f"分类报告生成失败: {e}")
                class_report = {
                    '无趋势(0)': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    '有趋势(1)': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
                }
                if 0 in true_counts.index:
                    class_report['无趋势(0)']['support'] = int(true_counts[0])
                if 1 in true_counts.index:
                    class_report['有趋势(1)']['support'] = int(true_counts[1])
            
            # 混淆矩阵
            cm = confusion_matrix(y, y_pred)
            
            # 总体（macro）指标
            overall_precision = class_report.get('macro avg', {}).get('precision', 0.0)
            overall_recall = class_report.get('macro avg', {}).get('recall', 0.0)
            overall_f1 = class_report.get('macro avg', {}).get('f1-score', 0.0)
            logger.info("\n总体指标:")
            logger.info(f"  命中率(accuracy): {accuracy:.4f}")
            logger.info(f"  Recall(macro):   {overall_recall:.4f}")
            logger.info(f"  Precision(macro):{overall_precision:.4f}")
            
            # 提取有趋势的指标（对交易更重要）
            trend_precision = class_report['有趋势(1)']['precision']
            trend_recall = class_report['有趋势(1)']['recall']
            trend_f1 = class_report['有趋势(1)']['f1-score']
            trend_support = class_report['有趋势(1)']['support']
            
            no_trend_precision = class_report['无趋势(0)']['precision']
            no_trend_recall = class_report['无趋势(0)']['recall']
            no_trend_f1 = class_report['无趋势(0)']['f1-score']
            no_trend_support = class_report['无趋势(0)']['support']
            
            logger.info(f"\n{dataset_name}数据集评估结果:")
            logger.info(f"准确率: {accuracy:.4f}")
            logger.info(f"\n有趋势(1)（主指标）:")
            logger.info(f"  Precision: {trend_precision:.4f} (发出有趋势信号时，确实有趋势的比例)")
            logger.info(f"  Recall:    {trend_recall:.4f} (实际有趋势中，被正确识别的比例)")
            logger.info(f"  F1-score:  {trend_f1:.4f}")
            logger.info(f"  Support:   {int(trend_support)} (样本数)")
            logger.info(f"\n无趋势(0):")
            logger.info(f"  Precision: {no_trend_precision:.4f}")
            logger.info(f"  Recall:    {no_trend_recall:.4f}")
            logger.info(f"  F1-score:  {no_trend_f1:.4f}")
            logger.info(f"  Support:   {int(no_trend_support)} (样本数)")
            logger.info(f"\n{'='*60}")
            logger.info(f"趋势F1（主指标）: {trend_f1:.4f}")
            logger.info(f"{'='*60}")
            
            logger.info(f"\n完整分类报告:\n{classification_report(y, y_pred)}")
            logger.info(f"\n混淆矩阵:\n{confusion_matrix(y, y_pred)}")
            
            subset_metrics = {}
            if pos_weights is not None:
                weight_array = np.array(pos_weights)
                subset_mask = weight_array >= 0.7
                subset_count = subset_mask.sum()
                if subset_count > 0:
                    y_mid = y_array[subset_mask]
                    y_pred_mid = y_pred[subset_mask]
                    try:
                        class_report_mid = classification_report(y_mid, y_pred_mid,
                                                                 target_names=['无趋势(0)', '有趋势(1)'],
                                                                 output_dict=True,
                                                                 zero_division=0)
                        subset_metrics = {
                            'mid_trend_precision': class_report_mid['有趋势(1)']['precision'],
                            'mid_trend_recall': class_report_mid['有趋势(1)']['recall'],
                            'mid_trend_f1': class_report_mid['有趋势(1)']['f1-score']
                        }
                        logger.info(f"\n{dataset_name} 高权重样本评估（trend_position_weight >= 0.7，样本数: {subset_count}）:")
                        logger.info(f"  趋势Precision: {subset_metrics['mid_trend_precision']:.4f}")
                        logger.info(f"  趋势Recall:    {subset_metrics['mid_trend_recall']:.4f}")
                        logger.info(f"  趋势F1:        {subset_metrics['mid_trend_f1']:.4f}")
                    except ValueError as e:
                        logger.warning(f"高权重样本分类报告生成失败: {e}")
                else:
                    logger.info(f"{dataset_name} 中 trend_position_weight >= 0.7 的样本数为0，跳过子集评估")
            else:
                logger.info(f"{dataset_name} 未提供趋势位置权重，跳过高权重子集评估")

            result = {
                'accuracy': accuracy,
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1': overall_f1,
                'y_true': y,
                'y_pred': y_pred,  # 返回0（无趋势）和1（有趋势）
                'y_proba': y_pred_proba,
                # 趋势检测指标
                'trend_precision': trend_precision,
                'trend_recall': trend_recall,
                'trend_f1': trend_f1,  # 主指标
                'no_trend_precision': no_trend_precision,
                'no_trend_recall': no_trend_recall,
                'no_trend_f1': no_trend_f1
            }
            result.update(subset_metrics)
            return result
    
    def get_feature_importance(self, model, feature_names):
        """
        获取特征重要性
        
        Args:
            model: 训练好的模型（XGBClassifier）
            feature_names: 特征名称列表
        
        Returns:
            DataFrame: 特征重要性排序
        """
        def _extract_importances_from_single(estimator):
            # 优先用 feature_importances_
            if hasattr(estimator, "feature_importances_"):
                imp = np.asarray(estimator.feature_importances_, dtype=float)
                # 对齐特征长度
                if len(imp) != len(feature_names):
                    try:
                        booster = estimator.get_booster()
                        score = booster.get_score(importance_type='gain')
                        imp_map = np.zeros(len(feature_names), dtype=float)
                        for i in range(len(feature_names)):
                            imp_map[i] = float(score.get(f"f{i}", 0.0))
                        imp = imp_map
                    except Exception:
                        imp = np.zeros(len(feature_names), dtype=float)
                return imp
            # 回退到原生 booster.get_score
            try:
                booster = estimator.get_booster()
                score = booster.get_score(importance_type='gain')
                imp = np.zeros(len(feature_names), dtype=float)
                for i in range(len(feature_names)):
                    imp[i] = float(score.get(f"f{i}", 0.0))
                return imp
            except Exception:
                return np.zeros(len(feature_names), dtype=float)
        
        # 兼容 Stage2 OVR 包装器：合成两个头的特征重要性（取平均）
        if bool(getattr(model, 'is_ovr', False)) and hasattr(model, 'model_up') and hasattr(model, 'model_down'):
            imp_up = _extract_importances_from_single(model.model_up)
            imp_down = _extract_importances_from_single(model.model_down)
            imp = (imp_up + imp_down) / 2.0
        else:
            imp = _extract_importances_from_single(model)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': imp
        }).sort_values('importance', ascending=False)
        return importance_df
    
    # ---------- 概率校准与时间平滑 ----------
    @staticmethod
    def _softmax(z):
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        sum_exp = np.sum(exp_z, axis=1, keepdims=True)
        return exp_z / np.clip(sum_exp, 1e-12, None)
    
    @staticmethod
    def _sigmoid(x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _predict_proba(self, model, X, use_3_class=True, use_calibration=False):
        """
        返回概率，若开启校准且存在参数则使用温度缩放。
        """
        if use_calibration and isinstance(self.calibration, dict) and self.calibration.get('method') == 'temperature':
            T = float(self.calibration.get('T', 1.0))
            try:
                raw = model.predict(X, output_margin=True)
            except Exception:
                # 兼容性回退
                return model.predict_proba(X)
            if use_3_class:
                logits = np.asarray(raw, dtype=float)
                proba = self._softmax(logits / max(T, 1e-6))
            else:
                logit = np.asarray(raw, dtype=float)
                p1 = self._sigmoid(logit / max(T, 1e-6))
                p0 = 1.0 - p1
                proba = np.vstack([p0, p1]).T
            return proba
        else:
            return model.predict_proba(X)
    
    def _fit_temperature_calibration(self, model, X_val, y_val, use_3_class=True):
        """
        在验证集上拟合温度缩放T，最小化NLL。
        """
        if not getattr(config, 'USE_PROB_CALIBRATION', False):
            return None
        method = getattr(config, 'CALIBRATION_METHOD', 'none')
        if method != 'temperature':
            return None
        try:
            raw = model.predict(X_val, output_margin=True)
        except Exception as e:
            logger.warning(f"获取raw margin失败，跳过温度缩放: {e}")
            return None
        T_grid = getattr(config, 'CALIBRATION_T_GRID', [0.5, 0.7, 1.0, 1.3, 1.5, 2.0])
        y_true = np.array(y_val).astype(int)
        def nll_for_T(T):
            T = float(T)
            if use_3_class:
                logits = np.asarray(raw, dtype=float)
                proba = self._softmax(logits / max(T, 1e-6))
                rows = np.arange(len(y_true))
                p = np.clip(proba[rows, y_true], 1e-12, 1.0)
            else:
                logit = np.asarray(raw, dtype=float)
                p1 = self._sigmoid(logit / max(T, 1e-6))
                p = np.where(y_true == 1, np.clip(p1, 1e-12, 1.0), np.clip(1.0 - p1, 1e-12, 1.0))
            return -np.mean(np.log(p))
        best_T, best_nll = None, np.inf
        for T in T_grid:
            try:
                val = nll_for_T(T)
                if val < best_nll:
                    best_nll = val
                    best_T = T
            except Exception:
                continue
        if best_T is not None:
            self.calibration = {'method': 'temperature', 'T': float(best_T)}
            logger.info(f"验证集温度缩放拟合完成：T={best_T:.4f}, NLL={best_nll:.6f}")
        else:
            logger.info("温度缩放拟合未找到更优T，保持未校准")
        return self.calibration
    
    def _apply_temporal_smoothing(self, proba, symbols=None):
        """
        对概率做EMA时间平滑；若提供symbols则按标的分别平滑。
        """
        if not getattr(config, 'USE_TEMPORAL_SMOOTHING', False):
            return proba
        alpha = float(getattr(config, 'SMOOTHING_EMA_ALPHA', 0.3))
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        n, c = proba.shape
        smoothed = np.empty_like(proba)
        if symbols is None:
            smoothed[0] = proba[0]
            for i in range(1, n):
                smoothed[i] = alpha * proba[i] + (1 - alpha) * smoothed[i - 1]
        else:
            symbols_arr = symbols.values if isinstance(symbols, pd.Series) else np.asarray(symbols)
            unique_syms = pd.unique(symbols_arr)
            smoothed[:] = proba
            for sym in unique_syms:
                idx = np.where(symbols_arr == sym)[0]
                if len(idx) == 0:
                    continue
                smoothed[idx[0]] = proba[idx[0]]
                for k in range(1, len(idx)):
                    i = idx[k]
                    j = idx[k-1]
                    smoothed[i] = alpha * proba[i] + (1 - alpha) * smoothed[j]
        return smoothed
    
    def _apply_class_thresholds(self, proba, thr0, thr1, thr2):
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
        return np.array(preds, dtype=int)
    
    def _compute_metric(self, y_true, y_pred, metric='accuracy'):
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'trend_f1':
            from sklearn.metrics import classification_report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            up_f1 = report.get('1', {}).get('f1-score', 0.0)
            down_f1 = report.get('2', {}).get('f1-score', 0.0)
            return (up_f1 + down_f1) / 2.0
        else:
            return accuracy_score(y_true, y_pred)
    
    def _compute_new_threshold_score(self, y_true, y_pred, pos_weights, target_up_acc=0.75, target_down_acc=0.65):
        """
        新的阈值评分函数：基于高置信度子集（trend_position_weight >= 0.7）
        
        计算三个指标：
        1. up_acc_high: 高置信度样本中上涨预测的准确率（precision）
        2. down_acc_high: 高置信度样本中下跌预测的准确率（precision）
        3. trend_f1_high: 高置信度样本中趋势方向的F1分数
        
        组合分数：
        overall_score = 0.45 * score_up + 0.35 * score_down + 0.20 * trend_f1_high
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            pos_weights: 趋势位置权重（用于筛选高置信度样本）
            target_up_acc: 上涨目标准确率（默认0.75）
            target_down_acc: 下跌目标准确率（默认0.65）
        
        Returns:
            float: 组合分数
        """
        try:
            # 筛选高置信度样本（trend_position_weight >= 0.7）
            weight_array = np.array(pos_weights)
            high_conf_mask = weight_array >= 0.7
            
            if not high_conf_mask.any():
                # 如果没有高置信度样本，回退到传统评分
                return accuracy_score(y_true, y_pred)
            
            y_true_high = np.array(y_true)[high_conf_mask]
            y_pred_high = np.array(y_pred)[high_conf_mask]
            
            # 获取实际存在的类别
            unique_true = np.unique(y_true_high)
            unique_pred = np.unique(y_pred_high)
            all_classes = np.unique(np.concatenate([unique_true, unique_pred])).astype(int).tolist()
            
            # 计算分类报告（使用labels参数指定实际存在的类别）
            from sklearn.metrics import classification_report, precision_score, f1_score
            
            # 始终使用labels参数，确保只计算实际存在的类别
            class_report = classification_report(
                y_true_high, y_pred_high,
                labels=all_classes,
                output_dict=True,
                zero_division=0
            )
            
            # 从分类报告中提取指标（使用类别索引作为键，classification_report返回的键是字符串）
            # 1. up_acc_high: 上涨预测的准确率（precision）
            up_acc_high = class_report.get('1', {}).get('precision', 0.0) if 1 in all_classes else 0.0
            
            # 2. down_acc_high: 下跌预测的准确率（precision）
            down_acc_high = class_report.get('2', {}).get('precision', 0.0) if 2 in all_classes else 0.0
            
            # 3. trend_f1_high: 趋势方向的F1分数（上涨和下跌的平均F1）
            up_f1 = class_report.get('1', {}).get('f1-score', 0.0) if 1 in all_classes else 0.0
            down_f1 = class_report.get('2', {}).get('f1-score', 0.0) if 2 in all_classes else 0.0
            
            # 如果只有一个趋势类别存在，使用该类的F1；如果都不存在，设为0
            if up_f1 > 0 and down_f1 > 0:
                trend_f1_high = (up_f1 + down_f1) / 2.0
            elif up_f1 > 0:
                trend_f1_high = up_f1
            elif down_f1 > 0:
                trend_f1_high = down_f1
            else:
                trend_f1_high = 0.0
            
            # 计算归一化分数
            score_up = min(up_acc_high / target_up_acc, 1.0)
            score_down = min(down_acc_high / target_down_acc, 1.0)
            
            # 组合分数
            overall_score = (
                0.45 * score_up +      # 主要目标：提高高置信度上涨准确率
                0.35 * score_down +    # 次要目标：提高下跌准确率
                0.20 * trend_f1_high   # 保持整体稳定性，但优先级较低
            )
            
            return overall_score
            
        except Exception as e:
            logger.warning(f"新评分函数计算失败，回退到传统评分: {e}")
            return accuracy_score(y_true, y_pred)
    
    def optimize_class_thresholds(self, y_true, proba, grid=None, metric='accuracy'):
        """
        在验证集上网格搜索三分类阈值，返回最佳阈值与得分
        """
        if grid is None:
            grid = getattr(config, 'THRESH_SEARCH_GRID', {
                'no_trend': [0.5],
                'up_trend': [0.5],
                'down_trend': [0.5],
            })
        best = {'score': -1.0, 'no_trend': None, 'up_trend': None, 'down_trend': None}
        for thr0 in grid.get('no_trend', [0.5]):
            for thr1 in grid.get('up_trend', [0.5]):
                for thr2 in grid.get('down_trend', [0.5]):
                    y_pred = self._apply_class_thresholds(proba, thr0, thr1, thr2)
                    score = self._compute_metric(y_true, y_pred, metric=metric)
                    if score > best['score']:
                        best.update({'score': score, 'no_trend': thr0, 'up_trend': thr1, 'down_trend': thr2})
        return best
    
    # -------- 两阶段：评估、寻优与每标的阈值 --------
    def _two_stage_predict_labels(self, p_trend, p_down, stage1_thr, stage2_thr_up, stage2_thr_down):
        """
        组合两阶段概率与阈值，输出最终3类标签（0=无趋势, 1=上涨, 2=下跌）
        p_trend: shape (n,) -> P(trend)
        p_down:  shape (n,) -> P(down | trend) 二分类下'stage2'对down的概率；则P(up)=1-p_down
        """
        p_trend = np.asarray(p_trend, dtype=float).reshape(-1)
        p_down = np.asarray(p_down, dtype=float).reshape(-1)
        n = p_trend.shape[0]
        # 将阈值统一为逐样本数组
        def as_array_thr(thr, name):
            if np.isscalar(thr):
                return np.full(n, float(thr), dtype=float)
            thr_arr = np.asarray(thr, dtype=float).reshape(-1)
            if thr_arr.shape[0] != n:
                # 长度不匹配时，回退为全局标量（取均值或首值）
                val = float(thr_arr[0]) if thr_arr.shape[0] > 0 else 0.6
                return np.full(n, val, dtype=float)
            return thr_arr
        s1_arr = as_array_thr(stage1_thr, 'stage1_thr')
        s2_up_arr = as_array_thr(stage2_thr_up, 'stage2_thr_up')
        s2_down_arr = as_array_thr(stage2_thr_down, 'stage2_thr_down')
        preds = np.zeros(n, dtype=int)
        p_up = 1.0 - p_down
        for i in range(n):
            if p_trend[i] < s1_arr[i]:
                preds[i] = 0
            else:
                cand = []
                if p_up[i] >= s2_up_arr[i]:
                    cand.append((1, p_up[i]))
                if p_down[i] >= s2_down_arr[i]:
                    cand.append((2, p_down[i]))
                if len(cand) == 0:
                    # 若均未过阈值，取更大者
                    preds[i] = 1 if p_up[i] >= p_down[i] else 2
                else:
                    preds[i] = max(cand, key=lambda t: t[1])[0]
        return preds
    
    def evaluate_two_stage_model(self, model_stage1, model_stage2, X, y, dataset_name='', symbols=None, thresholds=None, metric='accuracy', pos_weights=None):
        """
        评估两阶段模型（Stage1: 是否趋势；Stage2: 涨/跌）
        thresholds: 可为None，或包含
          {
            'stage1': {'global': {'trend': ...}, 'per_symbol': {'AAPL': ...}},
            'stage2': {'global': {'up_trend': ..., 'down_trend': ...}, 'per_symbol': {'AAPL': {'up_trend': ..., 'down_trend': ...}}}
          }
        """
        # 概率
        proba1 = model_stage1.predict_proba(X)  # [:,1]为P(trend)
        p_trend = proba1[:, 1]
        proba2 = model_stage2.predict_proba(X)  # [:,1]为P(down)，[:,0]为P(up)
        p_down = proba2[:, 1]
        
        # 阈值
        stage1_thr_global = float(getattr(config, 'PREDICTION_CONFIDENCE_THRESHOLD', 0.6))
        stage2_thr_up_global = float(getattr(config, 'CLASS_PROB_THRESHOLD_UP_TREND', 0.55))
        stage2_thr_down_global = float(getattr(config, 'CLASS_PROB_THRESHOLD_DOWN_TREND', 0.60))
        per_symbol_s1 = {}
        per_symbol_s2 = {}
        if isinstance(thresholds, dict):
            s1 = thresholds.get('stage1', {})
            s2 = thresholds.get('stage2', {})
            stage1_thr_global = s1.get('global', {}).get('trend', stage1_thr_global)
            stage2_thr_up_global = s2.get('global', {}).get('up_trend', stage2_thr_up_global)
            stage2_thr_down_global = s2.get('global', {}).get('down_trend', stage2_thr_down_global)
            per_symbol_s1 = s1.get('per_symbol', {}) or {}
            per_symbol_s2 = s2.get('per_symbol', {}) or {}
        
        # 逐样本应用每标的阈值（若提供）
        if symbols is not None:
            if not isinstance(symbols, pd.Series):
                symbols = pd.Series(symbols)
            thr1_list = []
            thr_up_list = []
            thr_down_list = []
            for sym in symbols.values:
                thr1_list.append(float(per_symbol_s1.get(sym, stage1_thr_global)))
                local_s2 = per_symbol_s2.get(sym, {})
                thr_up_list.append(float(local_s2.get('up_trend', stage2_thr_up_global)))
                thr_down_list.append(float(local_s2.get('down_trend', stage2_thr_down_global)))
            thr1_arr = np.array(thr1_list, dtype=float)
            thr_up_arr = np.array(thr_up_list, dtype=float)
            thr_down_arr = np.array(thr_down_list, dtype=float)
            y_pred = self._two_stage_predict_labels(p_trend, p_down, thr1_arr, thr_up_arr, thr_down_arr)
        else:
            y_pred = self._two_stage_predict_labels(p_trend, p_down, stage1_thr_global, stage2_thr_up_global, stage2_thr_down_global)
        
        # 评估
        accuracy = accuracy_score(y, y_pred)
        try:
            class_report = classification_report(y, y_pred, 
                                                target_names=['无趋势(0)', '上涨趋势(1)', '下跌趋势(2)'],
                                                output_dict=True,
                                                zero_division=0)
        except Exception:
            class_report = {}
        cm = confusion_matrix(y, y_pred)
        # 总体（macro）指标
        overall_precision = class_report.get('macro avg', {}).get('precision', 0.0)
        overall_recall = class_report.get('macro avg', {}).get('recall', 0.0)
        overall_f1 = class_report.get('macro avg', {}).get('f1-score', 0.0)
        logger.info("\n总体指标:")
        logger.info(f"  命中率(accuracy): {accuracy:.4f}")
        logger.info(f"  Recall(macro):   {overall_recall:.4f}")
        logger.info(f"  Precision(macro):{overall_precision:.4f}")
        
        up_trend_f1 = class_report.get('上涨趋势(1)', {}).get('f1-score', 0.0)
        down_trend_f1 = class_report.get('下跌趋势(2)', {}).get('f1-score', 0.0)
        trend_f1 = (up_trend_f1 + down_trend_f1) / 2.0
        
        logger.info(f"\n{dataset_name}（两阶段）准确率: {accuracy:.4f} | 趋势F1: {trend_f1:.4f}")
        logger.info(f"完整分类报告:\n{classification_report(y, y_pred, target_names=['无趋势(0)', '上涨趋势(1)', '下跌趋势(2)'])}")
        logger.info(f"混淆矩阵:\n{cm}")
        
        subset_metrics = {}
        if pos_weights is not None:
            try:
                weight_array = np.array(pos_weights)
                subset_mask = weight_array >= 0.7
                subset_count = subset_mask.sum()
                if subset_count > 0:
                    y_mid = np.array(y)[subset_mask]
                    y_pred_mid = np.array(y_pred)[subset_mask]
                    class_report_mid = classification_report(y_mid, y_pred_mid,
                                                             target_names=['无趋势(0)', '上涨趋势(1)', '下跌趋势(2)'],
                                                             output_dict=True,
                                                             zero_division=0)
                    # 仅趋势样本(1/2)的命中率
                    try:
                        trend_mask_mid = np.isin(y_mid, [1, 2])
                        if trend_mask_mid.any():
                            mid_trend_only_accuracy = float((np.array(y_mid)[trend_mask_mid] == np.array(y_pred_mid)[trend_mask_mid]).mean())
                        else:
                            mid_trend_only_accuracy = 0.0
                    except Exception:
                        mid_trend_only_accuracy = 0.0
                    subset_metrics = {
                        'mid_trend_precision': (class_report_mid['上涨趋势(1)']['precision'] + class_report_mid['下跌趋势(2)']['precision']) / 2,
                        'mid_trend_recall': (class_report_mid['上涨趋势(1)']['recall'] + class_report_mid['下跌趋势(2)']['recall']) / 2,
                        'mid_trend_f1': (class_report_mid['上涨趋势(1)']['f1-score'] + class_report_mid['下跌趋势(2)']['f1-score']) / 2,
                        'mid_accuracy': float((np.array(y_mid) == np.array(y_pred_mid)).mean()),
                        # 分类别高权重识别率（Recall和Precision）
                        'mid_up_accuracy': class_report_mid['上涨趋势(1)']['recall'],
                        'mid_down_accuracy': class_report_mid['下跌趋势(2)']['recall'],
                        'mid_no_trend_accuracy': class_report_mid['无趋势(0)']['recall'],
                        'mid_up_precision': class_report_mid['上涨趋势(1)']['precision'],
                        'mid_down_precision': class_report_mid['下跌趋势(2)']['precision'],
                        'mid_no_trend_precision': class_report_mid['无趋势(0)']['precision'],
                        'mid_trend_only_accuracy': mid_trend_only_accuracy,
                    }
                    logger.info(f"\n{dataset_name} 高权重样本评估（trend_position_weight >= 0.7，样本数: {subset_count}）:")
                    logger.info(f"  趋势Precision: {subset_metrics['mid_trend_precision']:.4f}")
                    logger.info(f"  趋势Recall:    {subset_metrics['mid_trend_recall']:.4f}")
                    logger.info(f"  趋势F1:        {subset_metrics['mid_trend_f1']:.4f}")
                    logger.info(f"  命中率(accuracy): {subset_metrics['mid_accuracy']:.4f}")
                    logger.info(f"  趋势内命中率(仅1/2): {mid_trend_only_accuracy:.4f}")
                else:
                    logger.info(f"{dataset_name} 中 trend_position_weight >= 0.7 的样本数为0，跳过子集评估")
            except Exception as e:
                logger.warning(f"两阶段高权重样本分类报告生成失败: {e}")
        
        # 分类别识别正确率（Recall和Precision）
        up_trend_accuracy = class_report.get('上涨趋势(1)', {}).get('recall', 0.0)
        down_trend_accuracy = class_report.get('下跌趋势(2)', {}).get('recall', 0.0)
        up_trend_precision = class_report.get('上涨趋势(1)', {}).get('precision', 0.0)
        down_trend_precision = class_report.get('下跌趋势(2)', {}).get('precision', 0.0)
        
        result = {
            'accuracy': accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'trend_f1': trend_f1,
            'y_true': y,
            'y_pred': y_pred,
            'p_trend': p_trend,
            'p_down': p_down,
            'up_trend_accuracy': up_trend_accuracy,  # Recall
            'down_trend_accuracy': down_trend_accuracy,  # Recall
            'up_trend_precision': up_trend_precision,
            'down_trend_precision': down_trend_precision,
        }
        result.update(subset_metrics)
        return result
    
    def _optimize_two_stage_thresholds(self, y_val, p_trend_val, p_down_val, symbols_val=None, metric='accuracy', pos_weights=None):
        """
        在验证集上寻优两阶段阈值：先全局，再每标的
        返回 thresholds 字典，结构见 evaluate_two_stage_model 注释
        
        Args:
            y_val: 验证集真实标签
            p_trend_val: Stage1概率 P(trend)
            p_down_val: Stage2概率 P(down | trend)
            symbols_val: 标的序列（可选）
            metric: 评分指标（如果使用新评分函数则忽略）
            pos_weights: 趋势位置权重（用于高置信度子集筛选）
        """
        grid1 = getattr(config, 'THRESH_SEARCH_GRID_STAGE1_TREND', [0.5, 0.6])
        grid2 = getattr(config, 'THRESH_SEARCH_GRID', {
            'up_trend': [0.55, 0.60, 0.65],
            'down_trend': [0.60, 0.65, 0.70]
        })
        
        # 目标准确率水平（从config读取，或使用默认值）
        target_up_acc = getattr(config, 'THRESH_TARGET_UP_ACC', 0.75)
        target_down_acc = getattr(config, 'THRESH_TARGET_DOWN_ACC', 0.65)
        
        # 使用新的评分函数（基于高置信度子集）
        use_new_scoring = getattr(config, 'USE_NEW_THRESH_SCORING', True)
        
        # 全局最优
        best = {'score': -1.0, 'stage1_trend': None, 'up_trend': None, 'down_trend': None}
        for t1 in grid1:
            for tu in grid2.get('up_trend', [0.6]):
                for td in grid2.get('down_trend', [0.65]):
                    y_pred = self._two_stage_predict_labels(p_trend_val, p_down_val, t1, tu, td)
                    
                    if use_new_scoring and pos_weights is not None:
                        # 使用新的评分函数：基于高置信度子集
                        score = self._compute_new_threshold_score(
                            y_val, y_pred, pos_weights, target_up_acc, target_down_acc
                        )
                    else:
                        # 使用传统评分函数
                        score = self._compute_metric(y_val, y_pred, metric=metric)
                    
                    if score > best['score']:
                        best.update({'score': score, 'stage1_trend': float(t1), 'up_trend': float(tu), 'down_trend': float(td)})
        thresholds = {
            'stage1': {'global': {'trend': best['stage1_trend']}, 'per_symbol': {}},
            'stage2': {'global': {'up_trend': best['up_trend'], 'down_trend': best['down_trend']}, 'per_symbol': {}},
            'score': float(best['score'])
        }
        # 每标的寻优（可选）
        if getattr(config, 'USE_PER_SYMBOL_THRESHOLDS', True) and symbols_val is not None:
            if not isinstance(symbols_val, pd.Series):
                symbols_val = pd.Series(symbols_val)
            df_val = pd.DataFrame({
                'symbol': symbols_val.values,
                'y_true': np.array(y_val),
                'p_trend': np.array(p_trend_val),
                'p_down': np.array(p_down_val)
            })
            # 如果有pos_weights，也加入到DataFrame中
            if pos_weights is not None:
                if not isinstance(pos_weights, pd.Series):
                    pos_weights = pd.Series(pos_weights)
                df_val['pos_weight'] = pos_weights.values
            else:
                df_val['pos_weight'] = 0.0
            
            per_s1 = {}
            per_s2 = {}
            for sym, g in df_val.groupby('symbol'):
                y_g = g['y_true'].values
                pt = g['p_trend'].values
                pdn = g['p_down'].values
                pos_w_g = g['pos_weight'].values if 'pos_weight' in g.columns else None
                best_g = {'score': -1.0, 'stage1_trend': thresholds['stage1']['global']['trend'], 
                          'up_trend': thresholds['stage2']['global']['up_trend'], 
                          'down_trend': thresholds['stage2']['global']['down_trend']}
                for t1 in grid1:
                    for tu in grid2.get('up_trend', [thresholds['stage2']['global']['up_trend']]):
                        for td in grid2.get('down_trend', [thresholds['stage2']['global']['down_trend']]):
                            y_pred = self._two_stage_predict_labels(pt, pdn, t1, tu, td)
                            
                            if use_new_scoring and pos_w_g is not None:
                                # 使用新的评分函数：基于高置信度子集
                                score = self._compute_new_threshold_score(
                                    y_g, y_pred, pos_w_g, target_up_acc, target_down_acc
                                )
                            else:
                                # 使用传统评分函数
                                score = self._compute_metric(y_g, y_pred, metric=metric)
                            
                            if score > best_g['score']:
                                best_g.update({'score': score, 'stage1_trend': float(t1), 'up_trend': float(tu), 'down_trend': float(td)})
                per_s1[sym] = float(best_g['stage1_trend'])
                per_s2[sym] = {'up_trend': float(best_g['up_trend']), 'down_trend': float(best_g['down_trend'])}
            thresholds['stage1']['per_symbol'] = per_s1
            thresholds['stage2']['per_symbol'] = per_s2
        return thresholds
    
    def save_per_symbol_accuracy(self, y_true, y_pred, symbols, dataset_name='dataset'):
        """
        计算并保存每个标的的正确率
        """
        try:
            if symbols is None:
                logger.warning("未提供symbols，跳过按标的正确率统计")
                return None
            # 转为Series，长度对齐
            if not isinstance(symbols, pd.Series):
                symbols = pd.Series(symbols)
            y_true_s = pd.Series(y_true)
            y_pred_s = pd.Series(y_pred)
            n = min(len(symbols), len(y_true_s), len(y_pred_s))
            symbols = symbols.iloc[:n]
            y_true_s = y_true_s.iloc[:n]
            y_pred_s = y_pred_s.iloc[:n]
            
            df_metric = pd.DataFrame({
                'symbol': symbols.values,
                'y_true': y_true_s.values,
                'y_pred': y_pred_s.values
            })
            # 每标的准确率
            grouped = df_metric.groupby('symbol')
            result_df = grouped.apply(lambda g: pd.Series({
                'samples': len(g),
                'accuracy': (g['y_true'].values == g['y_pred'].values).mean()
            })).reset_index()
            # 按accuracy降序、samples降序排序
            result_df = result_df.sort_values(by=['accuracy', 'samples'], ascending=[False, False]).reset_index(drop=True)
            # 保存到 models/result
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result_dir = os.path.join(project_root, 'models', 'result')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(result_dir, f'per_symbol_accuracy_{dataset_name}_{timestamp}.csv')
            result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"{dataset_name} 每标的正确率已保存: {file_path}")
            return file_path
        except Exception as e:
            logger.warning(f"保存按标的正确率失败: {e}")
            return None
    
    def save_model(self, model, feature_names, model_name='trend_xgboost', suffix: str = None):
        """
        保存模型
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            model_name: 模型名称
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_stub = f"{model_name}_{timestamp}" if not suffix else f"{model_name}_{suffix}_{timestamp}"
        model_file = os.path.join(self.model_dir, f"{file_stub}.pkl")
        
        # 保存模型和特征名称
        model_data = {
            'model': model,
            'feature_names': feature_names,
            # 保存行业编码器，确保推理阶段编码一致
            'encoders': {
                'sector_encoder': self.sector_encoder,
                'industry_encoder': self.industry_encoder,
                'sub_industry_encoder': self.sub_industry_encoder,
            },
            'calibration': self.calibration,
            'timestamp': timestamp
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"模型已保存到: {model_file}")
        
        # 同时保存为XGBoost原生格式（XGBClassifier也支持）
        try:
            xgb_file = os.path.join(self.model_dir, f"{file_stub}.xgb")
            model.get_booster().save_model(xgb_file)
            logger.info(f"模型(XGBoost格式)已保存到: {xgb_file}")
        except Exception as e:
            logger.warning(f"保存XGBoost格式失败: {e}")
        
        return model_file
    
    def save_two_stage_model(self, model_stage1, model_stage2, feature_names, model_name='trend_xgboost_two_stage', suffix: str = None):
        """
        保存两阶段模型
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_stub = f"{model_name}_{timestamp}" if not suffix else f"{model_name}_{suffix}_{timestamp}"
        model_file = os.path.join(self.model_dir, f"{file_stub}.pkl")
        bundle = {
            'two_stage': True,
            'model_stage1': model_stage1,
            'model_stage2': model_stage2,
            'stage2_ovr': bool(getattr(model_stage2, 'is_ovr', False)),
            'feature_names': feature_names,
            'encoders': {
                'sector_encoder': self.sector_encoder,
                'industry_encoder': self.industry_encoder,
                'sub_industry_encoder': self.sub_industry_encoder,
            },
            'calibration': self.calibration,
            'timestamp': timestamp
        }
        with open(model_file, 'wb') as f:
            pickle.dump(bundle, f)
        logger.info(f"两阶段模型已保存到: {model_file}")
        try:
            # 也保存stage2为XGBoost原生格式做参考
            if bool(getattr(model_stage2, 'is_ovr', False)):
                up_file = os.path.join(self.model_dir, f"{file_stub}_stage2_up.xgb")
                down_file = os.path.join(self.model_dir, f"{file_stub}_stage2_down.xgb")
                try:
                    model_stage2.model_up.get_booster().save_model(up_file)
                    logger.info(f"两阶段Stage2-OVR 上涨头(XGBoost)已保存: {up_file}")
                except Exception as e:
                    logger.warning(f"保存Stage2-OVR 上涨头失败：{e}")
                try:
                    model_stage2.model_down.get_booster().save_model(down_file)
                    logger.info(f"两阶段Stage2-OVR 下跌头(XGBoost)已保存: {down_file}")
                except Exception as e:
                    logger.warning(f"保存Stage2-OVR 下跌头失败：{e}")
            else:
                xgb_file = os.path.join(self.model_dir, f"{file_stub}.xgb")
            model_stage2.get_booster().save_model(xgb_file)
            logger.info(f"两阶段Stage2模型(XGBoost格式)已保存到: {xgb_file}")
        except Exception as e:
            logger.warning(f"保存两阶段XGBoost格式失败: {e}")
        return model_file
    
    def train(self):
        """完整的训练流程"""
        logger.info("=" * 60)
        logger.info("开始XGBoost趋势预测模型训练")
        logger.info("=" * 60)
        
        # Walk-Forward CV 选参（可选）
        def run_walk_forward_cv_if_enabled(train_df_full):
            if not getattr(config, 'USE_PURGED_WALK_FORWARD_CV', False):
                return None
            try:
                logger.info("\n[CV] 开始Purged Walk-Forward验证与选参...")
                num_folds = int(getattr(config, 'CV_NUM_FOLDS', 4))
                val_days = int(getattr(config, 'CV_VAL_WINDOW_DAYS', 60))
                purge_gap = int(getattr(config, 'CV_PURGE_GAP_DAYS', 5))
                anchored = bool(getattr(config, 'CV_ANCHORED', True))
                select_metric = getattr(config, 'CV_SELECT_METRIC', 'accuracy')
                grid = getattr(config, 'XGB_CV_PARAM_GRID', {})
                
                df = train_df_full.copy()
                if 'date' not in df.columns:
                    logger.warning("[CV] 训练数据缺少date列，跳过CV")
                    return None
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                dates = pd.to_datetime(df['date'].unique())
                dates = np.array(sorted(dates))
                if len(dates) < val_days * 2:
                    logger.warning("[CV] 可用日期不足，跳过CV")
                    return None
                # 生成验证窗口起点（尽量靠近后部）
                fold_starts = []
                latest_end = dates[-1]
                # 以val_days为步长，向前构造窗口，再取最后num_folds个
                i = len(dates) - val_days
                while i - val_days >= 0 and len(fold_starts) < num_folds * 2:
                    fold_starts.append(dates[i - val_days])
                    i -= val_days
                fold_starts = sorted(fold_starts)[-num_folds:]
                if len(fold_starts) == 0:
                    logger.warning("[CV] 无法生成折，跳过CV")
                    return None
                
                # 组装参数组合
                from itertools import product
                keys = list(grid.keys())
                values = [grid[k] for k in keys]
                param_candidates = [dict(zip(keys, v)) for v in product(*values)] or [{}]
                
                best_score = -1.0
                best_params = None
                
                for cand in param_candidates:
                    fold_scores = []
                    logger.info(f"[CV] 参数候选: {cand}")
                    for fs in fold_starts:
                        val_start = fs
                        val_end = val_start + pd.Timedelta(days=val_days-1)
                        # 训练集区间
                        if anchored:
                            train_start = dates[0]
                        else:
                            train_start = val_start - pd.Timedelta(days=val_days*2)  # 简易滚动
                        train_end = val_start - pd.Timedelta(days=purge_gap)
                        
                        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
                        val_mask = (df['date'] >= val_start) & (df['date'] <= val_end)
                        df_tr = df.loc[train_mask].copy()
                        df_va = df.loc[val_mask].copy()
                        if len(df_tr) < 100 or len(df_va) < 50:
                            continue
                        
                        # 重置编码器
                        from sklearn.preprocessing import LabelEncoder
                        self.sector_encoder = LabelEncoder()
                        self.industry_encoder = LabelEncoder()
                        self.sub_industry_encoder = LabelEncoder()
                        
                        # 特征
                        X_tr, y_tr, feat_cols, pos_w_tr, q_w_tr, t_w_tr, _, use3_tr = self.prepare_features(
                            df_tr,
                            use_time_features=getattr(config, 'USE_TIME_FEATURES', True),
                            use_raw_ohlcv=getattr(config, 'USE_RAW_OHLCV', False),
                            use_industry_features=getattr(config, 'USE_INDUSTRY_FEATURES', True),
                            fit_encoders=True
                        )
                        X_va, y_va, _, pos_w_va, q_w_va, t_w_va, _, use3_va = self.prepare_features(
                            df_va,
                            use_time_features=getattr(config, 'USE_TIME_FEATURES', True),
                            use_raw_ohlcv=getattr(config, 'USE_RAW_OHLCV', False),
                            use_industry_features=getattr(config, 'USE_INDUSTRY_FEATURES', True),
                            fit_encoders=False
                        )
                        # 训练与评估
                        model_cv = self.train_model(
                            X_tr, y_tr, X_va, y_va,
                            params=cand,
                            use_class_weights=getattr(config, 'USE_CLASS_WEIGHTS', True),
                            pos_weights=pos_w_tr,
                            quality_weights=q_w_tr,
                            time_weights=t_w_tr,
                            use_3_class=use3_tr
                        )
                        if model_cv is None:
                            continue
                        res = self.evaluate_model(model_cv, X_va, y_va, 'CV验证', use_probability_threshold=True, use_3_class=use3_va, pos_weights=pos_w_va)
                        if res is None:
                            continue
                        score = res['accuracy'] if select_metric == 'accuracy' else res.get('trend_f1', 0.0)
                        fold_scores.append(score)
                    if len(fold_scores) > 0:
                        avg_score = float(np.mean(fold_scores))
                        logger.info(f"[CV] 参数 {cand} 平均{select_metric}: {avg_score:.4f}（{len(fold_scores)}折）")
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = cand
                if best_params is not None:
                    logger.info(f"[CV] 最优参数: {best_params}, 平均{select_metric}: {best_score:.4f}")
                else:
                    logger.info("[CV] 未找到更优参数，保持默认")
                return best_params
            except Exception as e:
                logger.warning(f"[CV] 执行失败，跳过：{e}")
                return None
        
        # 1. 加载数据
        logger.info("\n1. 加载数据...")
        try:
            train_df = self.load_data(self.train_dir, use_all_symbols=True, dataset_name='训练集')
            val_df = self.load_data(self.val_dir, use_all_symbols=True, dataset_name='验证集')
            test_df = self.load_data(self.test_dir, use_all_symbols=True, dataset_name='测试集')
        except FileNotFoundError as e:
            logger.error(f"数据加载失败: {e}")
            logger.error("请先运行 data_splitter.py 划分数据")
            return None
        
        # 可选：利用Walk-Forward CV对参数进行选优
        best_params = run_walk_forward_cv_if_enabled(train_df)
        
        # 验证时间顺序：训练集 < 验证集 < 测试集
        logger.info("\n验证数据集时间顺序...")
        if len(train_df) > 0 and len(val_df) > 0:
            train_max_date = train_df['date'].max()
            val_min_date = val_df['date'].min()
            if train_max_date >= val_min_date:
                logger.error(f"❌ 时间顺序错误：训练集最大日期 ({train_max_date}) >= 验证集最小日期 ({val_min_date})")
                logger.error("这会导致数据泄露！请检查数据划分逻辑。")
                return None
            else:
                logger.info(f"✅ 训练集 < 验证集: {train_max_date} < {val_min_date}")
        
        if len(val_df) > 0 and len(test_df) > 0:
            val_max_date = val_df['date'].max()
            test_min_date = test_df['date'].min()
            if val_max_date >= test_min_date:
                logger.error(f"❌ 时间顺序错误：验证集最大日期 ({val_max_date}) >= 测试集最小日期 ({test_min_date})")
                logger.error("这会导致数据泄露！请检查数据划分逻辑。")
                return None
            else:
                logger.info(f"✅ 验证集 < 测试集: {val_max_date} < {test_min_date}")
        
        logger.info("✅ 所有数据集时间顺序验证通过！")
        
        # 2. 准备特征和标签
        logger.info("\n2. 准备特征和标签...")
        # 可以从config读取配置，或直接设置
        use_time_features = getattr(config, 'USE_TIME_FEATURES', True)
        use_raw_ohlcv = getattr(config, 'USE_RAW_OHLCV', False)
        use_industry_features = getattr(config, 'USE_INDUSTRY_FEATURES', True)
        
        # 检查是否使用三分类模型
        use_3_class = getattr(config, 'USE_3_CLASS_MODEL', True)
        
        # 训练集：拟合编码器
        X_train, y_train, feature_names, pos_weights_train, quality_weights_train, time_weights_train, symbols_train, use_3_class_train = self.prepare_features(
            train_df, 
            use_time_features=use_time_features,
            use_raw_ohlcv=use_raw_ohlcv,
            use_industry_features=use_industry_features,
            fit_encoders=True  # 在训练集上拟合编码器
        )
        
        # 上采样下跌趋势样本（如果启用）
        if getattr(config, 'USE_UPSAMPLING_DOWN_TREND', True) and use_3_class_train:
            logger.info("\n2.1. 上采样下跌趋势样本...")
            X_train, y_train, pos_weights_train, quality_weights_train, time_weights_train, symbols_train = self.upsample_down_trend(
                X_train, y_train, pos_weights_train, quality_weights_train, time_weights_train, symbols_train
            )
        # 验证集和测试集：使用已拟合的编码器
        X_val, y_val, _, pos_weights_val, quality_weights_val, time_weights_val, symbols_val, use_3_class_val = self.prepare_features(
            val_df,
            use_time_features=use_time_features,
            use_raw_ohlcv=use_raw_ohlcv,
            use_industry_features=use_industry_features,
            fit_encoders=False  # 使用已拟合的编码器
        )
        X_test, y_test, _, pos_weights_test, quality_weights_test, time_weights_test, symbols_test, use_3_class_test = self.prepare_features(
            test_df,
            use_time_features=use_time_features,
            use_raw_ohlcv=use_raw_ohlcv,
            use_industry_features=use_industry_features,
            fit_encoders=False  # 使用已拟合的编码器
        )
        
        # 确保use_3_class一致
        if use_3_class_train != use_3_class_val or use_3_class_train != use_3_class_test:
            logger.warning("⚠️ 不同数据集的模型类型不一致，使用训练集的设置")
        use_3_class = use_3_class_train
        
        # 确保特征顺序一致
        X_val = X_val[feature_names]
        X_test = X_test[feature_names]
        
        # 3. 训练模型
        logger.info("\n3. 训练模型...")
        use_two_stage = bool(getattr(config, 'USE_TWO_STAGE_CLASSIFIER', False) and use_3_class)
        use_class_weights = getattr(config, 'USE_CLASS_WEIGHTS', True)
        model_params = best_params if best_params is not None else None
        test_split_results = None
        importance_df = None
        if use_two_stage:
            logger.info("启用两阶段分类：Stage1=是否趋势(0/1)，Stage2=涨/跌（二分类）")
            # Stage1: 是否趋势
            y_train_s1 = (pd.Series(y_train).astype(int) != 0).astype(int)
            y_val_s1 = (pd.Series(y_val).astype(int) != 0).astype(int)
            model_stage1 = self.train_model(
                X_train, y_train_s1, X_val, y_val_s1,
                params=None,
                use_class_weights=use_class_weights,
                pos_weights=pos_weights_train,
                quality_weights=quality_weights_train,
                time_weights=time_weights_train,
                use_3_class=False
            )
            if model_stage1 is None:
                logger.error("❌ Stage1训练失败")
                return None
            # Stage2
            use_stage2_ovr = bool(getattr(config, 'USE_STAGE2_OVR', False))
            if use_stage2_ovr:
                logger.info("Stage2采用 OVR：训练 (上涨 vs 其余) 与 (下跌 vs 其余) 两个二分类头")
                # 头A：上涨 vs 其余
                y_train_up = (pd.Series(y_train).astype(int) == 1).astype(int)
                y_val_up = (pd.Series(y_val).astype(int) == 1).astype(int)
                model_stage2_up = self.train_model(
                    X_train, y_train_up, X_val, y_val_up,
                    params=None,
                    use_class_weights=False,  # 避免被Stage2特化权重误判，改用位置/质量/时间权重
                    pos_weights=pos_weights_train,
                    quality_weights=quality_weights_train,
                    time_weights=time_weights_train,
                    use_3_class=False
                )
                if model_stage2_up is None:
                    logger.error("❌ Stage2-OVR 上涨头训练失败")
                    return None
                # 头B：下跌 vs 其余
                y_train_down = (pd.Series(y_train).astype(int) == 2).astype(int)
                y_val_down = (pd.Series(y_val).astype(int) == 2).astype(int)
                model_stage2_down = self.train_model(
                    X_train, y_train_down, X_val, y_val_down,
                    params=None,
                    use_class_weights=False,
                    pos_weights=pos_weights_train,
                    quality_weights=quality_weights_train,
                    time_weights=time_weights_train,
                    use_3_class=False
                )
                if model_stage2_down is None:
                    logger.error("❌ Stage2-OVR 下跌头训练失败")
                    return None
                model_stage2 = Stage2OVR(model_stage2_up, model_stage2_down)
            else:
                    # 原Stage2：涨/跌（二分类，down=1, up=0；仅在趋势样本上训练）
                mask_tr_train = pd.Series(y_train).isin([1, 2]).values
                mask_tr_val = pd.Series(y_val).isin([1, 2]).values
                X_train_s2 = X_train[mask_tr_train].copy()
                y_train_s2 = (pd.Series(y_train)[mask_tr_train] == 2).astype(int)
                X_val_s2 = X_val[mask_tr_val].copy()
                y_val_s2 = (pd.Series(y_val)[mask_tr_val] == 2).astype(int)
                model_stage2 = self.train_model(
                    X_train_s2, y_train_s2, X_val_s2, y_val_s2,
                    params=None,
                    use_class_weights=use_class_weights,
                    pos_weights=(pos_weights_train[mask_tr_train] if pos_weights_train is not None else None),
                    quality_weights=(quality_weights_train[mask_tr_train] if quality_weights_train is not None else None),
                    time_weights=(time_weights_train[mask_tr_train] if time_weights_train is not None else None),
                    use_3_class=False
                )
            if model_stage2 is None:
                logger.error("❌ Stage2训练失败（趋势样本不足？）")
                return None
            # 初评
            eval_train = self.evaluate_two_stage_model(model_stage1, model_stage2, X_train, y_train, '训练集(两阶段)', symbols=symbols_train, pos_weights=pos_weights_train)
            eval_val = self.evaluate_two_stage_model(model_stage1, model_stage2, X_val, y_val, '验证集(两阶段)', symbols=symbols_val, pos_weights=pos_weights_val)
            eval_test = self.evaluate_two_stage_model(model_stage1, model_stage2, X_test, y_test, '测试集(两阶段)', symbols=symbols_test, pos_weights=pos_weights_test)
            # 阈值寻优（全局+每标的）
            metric = getattr(config, 'THRESH_OPTIMIZE_METRIC', 'accuracy')
            thresholds_ts = self._optimize_two_stage_thresholds(
                y_val=np.array(y_val),
                p_trend_val=eval_val['p_trend'],
                p_down_val=eval_val['p_down'],
                symbols_val=symbols_val,
                metric=metric,
                pos_weights=pos_weights_val
            )
            try:
                project_root = os.path.dirname(os.path.abspath(__file__))
                result_dir = os.path.join(project_root, 'models', 'result')
                os.makedirs(result_dir, exist_ok=True)
                thr_file = os.path.join(result_dir, f"class_thresholds_two_stage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(thr_file, 'w', encoding='utf-8') as f:
                    json.dump({'two_stage': True, 'metric': metric, **thresholds_ts}, f, ensure_ascii=False, indent=2)
                logger.info(f"两阶段阈值已保存: {thr_file}")
            except Exception as e:
                logger.warning(f"保存两阶段阈值失败：{e}")
            # 复评
            eval_train = self.evaluate_two_stage_model(model_stage1, model_stage2, X_train, y_train, '训练集(两阶段-寻优阈值)', symbols=symbols_train, thresholds=thresholds_ts, pos_weights=pos_weights_train)
            eval_val = self.evaluate_two_stage_model(model_stage1, model_stage2, X_val, y_val, '验证集(两阶段-寻优阈值)', symbols=symbols_val, thresholds=thresholds_ts, pos_weights=pos_weights_val)
            eval_test = self.evaluate_two_stage_model(model_stage1, model_stage2, X_test, y_test, '测试集(两阶段-寻优阈值)', symbols=symbols_test, thresholds=thresholds_ts, pos_weights=pos_weights_test)
            # 特征重要性（Stage2）
            logger.info("\n5. 特征重要性分析...")
            importance_df = self.get_feature_importance(model_stage2, feature_names)
            logger.info("\nTop 20 重要特征（Stage2-涨跌判别）:")
            logger.info(importance_df.head(20).to_string(index=False))
            importance_file = os.path.join(self.model_dir, 'feature_importance_stage2.csv')
            # 备份现有文件（如果存在）
            if os.path.exists(importance_file):
                backup_date = datetime.now().strftime('%Y%m%d')
                backup_file = os.path.join(self.model_dir, f'feature_importance_stage2_{backup_date}.csv')
                try:
                    shutil.copy2(importance_file, backup_file)
                    logger.info(f"已备份原文件到: {backup_file}")
                except Exception as e:
                    logger.warning(f"备份文件失败: {e}")
            importance_df.to_csv(importance_file, index=False, encoding='utf-8-sig')
            logger.info(f"\n特征重要性(Stage2)已保存到: {importance_file}")
            # 每标的正确率
            logger.info("\n6. 每标的正确率统计与保存...")
            self.save_per_symbol_accuracy(eval_train['y_true'], eval_train['y_pred'], symbols_train, dataset_name='train')
            self.save_per_symbol_accuracy(eval_val['y_true'], eval_val['y_pred'], symbols_val, dataset_name='validation')
            self.save_per_symbol_accuracy(eval_test['y_true'], eval_test['y_pred'], symbols_test, dataset_name='test')
            # 统计测试集Stage2的置信度-准确率关系并保存CSV
            try:
                proba_s2_test = model_stage2.predict_proba(X_test)
                p_up = proba_s2_test[:, 0]   # P(上涨)
                p_down = proba_s2_test[:, 1] # P(下跌)
                # 预测标签：0=上涨，1=下跌
                y_pred_s2 = (p_down > p_up).astype(int)
                y_true_test = np.array(y_test)  # 真实3类标签：0=无趋势, 1=上涨, 2=下跌
                # 概率分桶（0.05步长）
                edges = np.round(np.linspace(0.0, 1.0, 21), 2)
                interval_index = pd.IntervalIndex.from_breaks(edges, closed='right')
                rows = []
                # up_trend: 仅统计预测为上涨的样本，按p_up分桶
                mask_up_pred = (y_pred_s2 == 0)
                probs_up = p_up[mask_up_pred]
                y_true_up = y_true_test[mask_up_pred]
                bins_up = pd.cut(probs_up, bins=edges, right=True, include_lowest=True)
                df_up = pd.DataFrame({'bin': bins_up, 'is_correct': (y_true_up == 1).astype(int)})
                agg_up = df_up.groupby('bin', observed=True).agg(
                    n_samples=('is_correct', 'size'),
                    n_correct=('is_correct', 'sum')
                ).reindex(interval_index, fill_value=0)
                for iv, row in agg_up.iterrows():
                    low = float(iv.left)
                    high = float(iv.right)
                    n_samples = int(row['n_samples'])
                    n_correct = int(row['n_correct'])
                    n_wrong = int(n_samples - n_correct)
                    acc = (n_correct / n_samples) if n_samples > 0 else float('nan')
                    rows.append({
                        'class': 'up_trend',
                        'prob_low': low,
                        'prob_high': high,
                        'n_samples': n_samples,
                        'n_correct': n_correct,
                        'n_wrong': n_wrong,
                        'accuracy': acc
                    })
                # down_trend: 仅统计预测为下跌的样本，按p_down分桶
                mask_down_pred = (y_pred_s2 == 1)
                probs_down = p_down[mask_down_pred]
                y_true_down = y_true_test[mask_down_pred]
                bins_down = pd.cut(probs_down, bins=edges, right=True, include_lowest=True)
                df_down = pd.DataFrame({'bin': bins_down, 'is_correct': (y_true_down == 2).astype(int)})
                agg_down = df_down.groupby('bin', observed=True).agg(
                    n_samples=('is_correct', 'size'),
                    n_correct=('is_correct', 'sum')
                ).reindex(interval_index, fill_value=0)
                for iv, row in agg_down.iterrows():
                    low = float(iv.left)
                    high = float(iv.right)
                    n_samples = int(row['n_samples'])
                    n_correct = int(row['n_correct'])
                    n_wrong = int(n_samples - n_correct)
                    acc = (n_correct / n_samples) if n_samples > 0 else float('nan')
                    rows.append({
                        'class': 'down_trend',
                        'prob_low': low,
                        'prob_high': high,
                        'n_samples': n_samples,
                        'n_correct': n_correct,
                        'n_wrong': n_wrong,
                        'accuracy': acc
                    })
                conf_df = pd.DataFrame(
                    rows,
                    columns=['class', 'prob_low', 'prob_high', 'n_samples', 'n_correct', 'n_wrong', 'accuracy']
                )
                ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_path = os.path.join(self.model_dir, f"confidence_accuracy_stage2_{ts_stamp}.csv")
                conf_df.to_csv(out_path, index=False, encoding='utf-8-sig')
                logger.info(f"Stage2 置信度-准确率统计已保存: {out_path}")
            except Exception as e:
                logger.warning(f"Stage2 置信度-准确率统计失败：{e}")
            # 保存模型（文件名带测试集高权重命中率后缀，便于直观识别）
            logger.info("\n7. 保存模型...")
            suffix_metric = None
            try:
                mid_acc = eval_test.get('mid_accuracy', None)
                if mid_acc is None:
                    mid_acc = eval_test.get('accuracy', None)
                if mid_acc is not None:
                    suffix_metric = f"midacc_{float(mid_acc):.4f}"
            except Exception:
                pass
            model_file = self.save_two_stage_model(model_stage1, model_stage2, feature_names, suffix=suffix_metric)
            # 总结
            logger.info("\n" + "=" * 60)
            logger.info("训练完成！（两阶段）")
            logger.info("=" * 60)
            logger.info(f"\n整体准确率:")
            logger.info(f"  训练集: {eval_train['accuracy']:.4f}")
            logger.info(f"  验证集: {eval_val['accuracy']:.4f}")
            logger.info(f"  测试集: {eval_test['accuracy']:.4f}")
            # 分类别识别率（Recall和Precision）
            logger.info(f"\n分类别识别率（Recall）和精确率（Precision）：")
            logger.info(f"  训练集: 上涨 Recall={eval_train.get('up_trend_accuracy', 0.0):.4f} Precision={eval_train.get('up_trend_precision', 0.0):.4f} | 下跌 Recall={eval_train.get('down_trend_accuracy', 0.0):.4f} Precision={eval_train.get('down_trend_precision', 0.0):.4f}")
            logger.info(f"  验证集: 上涨 Recall={eval_val.get('up_trend_accuracy', 0.0):.4f} Precision={eval_val.get('up_trend_precision', 0.0):.4f} | 下跌 Recall={eval_val.get('down_trend_accuracy', 0.0):.4f} Precision={eval_val.get('down_trend_precision', 0.0):.4f}")
            logger.info(f"  测试集: 上涨 Recall={eval_test.get('up_trend_accuracy', 0.0):.4f} Precision={eval_test.get('up_trend_precision', 0.0):.4f} | 下跌 Recall={eval_test.get('down_trend_accuracy', 0.0):.4f} Precision={eval_test.get('down_trend_precision', 0.0):.4f}")
            logger.info(f"\n总体Recall(macro):")
            logger.info(f"  训练集: {eval_train.get('overall_recall', 0.0):.4f}")
            logger.info(f"  验证集: {eval_val.get('overall_recall', 0.0):.4f}")
            logger.info(f"  测试集: {eval_test.get('overall_recall', 0.0):.4f}")
            logger.info(f"\n趋势检测指标（主指标）:")
            logger.info(f"  训练集趋势F1: {eval_train['trend_f1']:.4f}")
            logger.info(f"  验证集趋势F1: {eval_val['trend_f1']:.4f}")
            logger.info(f"  测试集趋势F1: {eval_test['trend_f1']:.4f} ⭐")
            if 'mid_trend_f1' in eval_train:
                logger.info(f"  训练集高权重趋势F1(>=0.7): {eval_train['mid_trend_f1']:.4f}")
                if 'mid_accuracy' in eval_train:
                    logger.info(f"  训练集高权重命中率(>=0.7): {eval_train['mid_accuracy']:.4f}")
                if 'mid_trend_recall' in eval_train:
                    logger.info(f"  训练集高权重Recall(>=0.7): {eval_train['mid_trend_recall']:.4f}")
                if 'mid_up_accuracy' in eval_train or 'mid_down_accuracy' in eval_train:
                    logger.info(f"  训练集高权重分类别正确率(>=0.7): 上涨 Accuracy={eval_train.get('mid_up_precision', 0.0):.4f} Recall={eval_train.get('mid_up_accuracy', 0.0):.4f} | 下跌 Accuracy={eval_train.get('mid_down_precision', 0.0):.4f} Recall={eval_train.get('mid_down_accuracy', 0.0):.4f}")
            if 'mid_trend_f1' in eval_val:
                logger.info(f"  验证集高权重趋势F1(>=0.7): {eval_val['mid_trend_f1']:.4f}")
                if 'mid_accuracy' in eval_val:
                    logger.info(f"  验证集高权重命中率(>=0.7): {eval_val['mid_accuracy']:.4f}")
                if 'mid_trend_recall' in eval_val:
                    logger.info(f"  验证集高权重Recall(>=0.7): {eval_val['mid_trend_recall']:.4f}")
                if 'mid_up_accuracy' in eval_val or 'mid_down_accuracy' in eval_val:
                    logger.info(f"  验证集高权重分类别正确率(>=0.7): 上涨 Accuracy={eval_val.get('mid_up_precision', 0.0):.4f} Recall={eval_val.get('mid_up_accuracy', 0.0):.4f} | 下跌 Accuracy={eval_val.get('mid_down_precision', 0.0):.4f} Recall={eval_val.get('mid_down_accuracy', 0.0):.4f}")
            if 'mid_trend_f1' in eval_test:
                logger.info(f"  测试集高权重趋势F1(>=0.7): {eval_test['mid_trend_f1']:.4f}")
                if 'mid_accuracy' in eval_test:
                    logger.info(f"  测试集高权重命中率(>=0.7): {eval_test['mid_accuracy']:.4f}")
                if 'mid_trend_recall' in eval_test:
                    logger.info(f"  测试集高权重Recall(>=0.7): {eval_test['mid_trend_recall']:.4f}")
                if 'mid_up_accuracy' in eval_test or 'mid_down_accuracy' in eval_test:
                    logger.info(f"  测试集高权重分类别正确率(>=0.7): 上涨 Accuracy={eval_test.get('mid_up_precision', 0.0):.4f} Recall={eval_test.get('mid_up_accuracy', 0.0):.4f} | 下跌 Accuracy={eval_test.get('mid_down_precision', 0.0):.4f} Recall={eval_test.get('mid_down_accuracy', 0.0):.4f}")
            logger.info(f"\n模型文件: {model_file}")
            return {
                'model': {'stage1': model_stage1, 'stage2': model_stage2},
                'feature_names': feature_names,
                'train_results': eval_train,
                'val_results': eval_val,
                'test_results': eval_test,
                'test_split_results': None,
                'importance_df': importance_df
            }
        else:
            # 原三分类流程
            # 使用类别权重，给有趋势更高的权重，避免模型全猜0
            model = self.train_model(X_train, y_train, X_val, y_val, 
                                    use_class_weights=use_class_weights,
                                    pos_weights=pos_weights_train,
                                    quality_weights=quality_weights_train,
                                    time_weights=time_weights_train,
                                    params=model_params,
                                    use_3_class=use_3_class)
            if model is None:
                logger.error("❌ 模型训练失败，无法继续评估")
                return None
            try:
                self._fit_temperature_calibration(model, X_val, y_val, use_3_class=use_3_class)
            except Exception as e:
                logger.warning(f"温度缩放拟合失败，跳过校准：{e}")
            logger.info("\n4. 评估模型...")
            use_prob_threshold = getattr(config, 'USE_PROBABILITY_THRESHOLD', True)
            logger.info(f"预测方式: {'概率阈值' if use_prob_threshold else '直接argmax'}")
            if use_prob_threshold:
                if use_3_class and getattr(config, 'USE_CLASS_SPECIFIC_THRESHOLDS', False):
                    thr0 = getattr(config, 'CLASS_PROB_THRESHOLD_NO_TREND', 0.5)
                    thr1 = getattr(config, 'CLASS_PROB_THRESHOLD_UP_TREND', 0.5)
                    thr2 = getattr(config, 'CLASS_PROB_THRESHOLD_DOWN_TREND', 0.5)
                    logger.info(f"使用分类别阈值: 无趋势={thr0} / 上涨={thr1} / 下跌={thr2}")
                else:
                    confidence_threshold = getattr(config, 'PREDICTION_CONFIDENCE_THRESHOLD', 0.6)
                    logger.info(f"置信度阈值: {confidence_threshold} (最大概率必须 >= 此值才交易)")
            train_results = self.evaluate_model(model, X_train, y_train, '训练集', use_probability_threshold=use_prob_threshold, use_3_class=use_3_class, pos_weights=pos_weights_train, symbols=symbols_train)
            val_results = self.evaluate_model(model, X_val, y_val, '验证集', use_probability_threshold=use_prob_threshold, use_3_class=use_3_class, pos_weights=pos_weights_val, symbols=symbols_val)
            test_results = self.evaluate_model(model, X_test, y_test, '测试集', use_probability_threshold=use_prob_threshold, use_3_class=use_3_class, pos_weights=pos_weights_test, symbols=symbols_test)
            if train_results is None or val_results is None or test_results is None:
                logger.error("❌ 模型评估失败，无法继续")
                return None
            if use_3_class and getattr(config, 'USE_CLASS_SPECIFIC_THRESHOLDS', True):
                try:
                    metric = getattr(config, 'THRESH_OPTIMIZE_METRIC', 'accuracy')
                    grid = getattr(config, 'THRESH_SEARCH_GRID', None)
                    y_val_true = y_val
                    y_val_proba = val_results['y_proba']
                    best_thr = self.optimize_class_thresholds(y_val_true, y_val_proba, grid=grid, metric=metric)
                    logger.info(f"验证集阈值寻优完成（指标={metric}）: {best_thr}")
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    result_dir = os.path.join(project_root, 'models', 'result')
                    os.makedirs(result_dir, exist_ok=True)
                    thr_file = os.path.join(result_dir, f"class_thresholds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(thr_file, 'w', encoding='utf-8') as f:
                        json.dump(best_thr, f, ensure_ascii=False, indent=2)
                    logger.info(f"阈值已保存: {thr_file}")
                    self.dynamic_thresholds = {
                        'no_trend': best_thr['no_trend'],
                        'up_trend': best_thr['up_trend'],
                        'down_trend': best_thr['down_trend']
                    }
                    logger.info("\n4.2 使用寻优阈值重新评估验证/测试集...")
                    val_results = self.evaluate_model(model, X_val, y_val, '验证集(寻优阈值)', use_probability_threshold=True, use_3_class=use_3_class, pos_weights=pos_weights_val, symbols=symbols_val)
                    test_results = self.evaluate_model(model, X_test, y_test, '测试集(寻优阈值)', use_probability_threshold=True, use_3_class=use_3_class, pos_weights=pos_weights_test, symbols=symbols_test)
                except Exception as e:
                    logger.warning(f"阈值寻优失败，保留原阈值: {e}")
            # 可选：test_early/test_late
            if getattr(config, 'SPLIT_TEST_INTO_EARLY_LATE', False):
                logger.info("\n4.1. 评估test_early/test_late合并数据...")
                try:
                    test_early_df = self.load_data(self.test_early_dir, use_all_symbols=True, dataset_name='test_early')
                    test_late_df = self.load_data(self.test_late_dir, use_all_symbols=True, dataset_name='test_late')
                    combined_df = pd.concat([test_early_df, test_late_df], ignore_index=True)
                    if len(combined_df) > 0:
                        X_test_split, y_test_split, _, pos_weights_split, quality_weights_split, time_weights_split, symbols_split, _ = self.prepare_features(
                            combined_df,
                            use_time_features=use_time_features,
                            use_raw_ohlcv=use_raw_ohlcv,
                            use_industry_features=use_industry_features,
                            fit_encoders=False
                        )
                        X_test_split = X_test_split[feature_names]
                        test_split_results = self.evaluate_model(
                            model,
                            X_test_split,
                            y_test_split,
                            'test_early_late_combined',
                            use_probability_threshold=use_prob_threshold,
                            use_3_class=use_3_class,
                            pos_weights=pos_weights_split,
                            symbols=symbols_split
                        )
                        try:
                            self.save_per_symbol_accuracy(test_split_results['y_true'], test_split_results['y_pred'], symbols_split, dataset_name='test_early_late_combined')
                        except Exception as e:
                            logger.warning(f"保存test_early_late_combined每标的正确率失败: {e}")
                    else:
                        logger.info("test_early/test_late 合并后没有样本，跳过评估")
                except FileNotFoundError:
                    logger.warning("未找到test_early或test_late数据，跳过评估")
                except Exception as e:
                    logger.warning(f"评估test_early/test_late时出错: {e}")
            # 特征重要性
            logger.info("\n5. 特征重要性分析...")
            importance_df = self.get_feature_importance(model, feature_names)
            logger.info("\nTop 20 重要特征:")
            logger.info(importance_df.head(20).to_string(index=False))
            importance_file = os.path.join(self.model_dir, 'feature_importance.csv')
            importance_df.to_csv(importance_file, index=False, encoding='utf-8-sig')
            logger.info(f"\n特征重要性已保存到: {importance_file}")
            # 每标的正确率
            logger.info("\n6. 每标的正确率统计与保存...")
            self.save_per_symbol_accuracy(train_results['y_true'], train_results['y_pred'], symbols_train, dataset_name='train')
            self.save_per_symbol_accuracy(val_results['y_true'], val_results['y_pred'], symbols_val, dataset_name='validation')
            self.save_per_symbol_accuracy(test_results['y_true'], test_results['y_pred'], symbols_test, dataset_name='test')
            # 保存模型（文件名带测试集高权重命中率后缀，便于直观识别）
            logger.info("\n7. 保存模型...")
            suffix_metric = None
            try:
                mid_acc = test_results.get('mid_accuracy', None)
                if mid_acc is None:
                    mid_acc = test_results.get('accuracy', None)
                if mid_acc is not None:
                    suffix_metric = f"midacc_{float(mid_acc):.4f}"
            except Exception:
                pass
            model_file = self.save_model(model, feature_names, suffix=suffix_metric)
            # 总结
            logger.info("\n" + "=" * 60)
            logger.info("训练完成！")
            logger.info("=" * 60)
            logger.info(f"\n整体准确率:")
            logger.info(f"  训练集: {train_results['accuracy']:.4f}")
            logger.info(f"  验证集: {val_results['accuracy']:.4f}")
            logger.info(f"  测试集: {test_results['accuracy']:.4f}")
            # 分类别识别正确率（Recall视角）
            logger.info(f"\n分类别识别正确率（Recall）：")
            logger.info(f"  训练集: 上涨={train_results.get('up_trend_accuracy', train_results.get('up_trend_recall', 0.0)):.4f} | 下跌={train_results.get('down_trend_accuracy', train_results.get('down_trend_recall', 0.0)):.4f}")
            logger.info(f"  验证集: 上涨={val_results.get('up_trend_accuracy', val_results.get('up_trend_recall', 0.0)):.4f} | 下跌={val_results.get('down_trend_accuracy', val_results.get('down_trend_recall', 0.0)):.4f}")
            logger.info(f"  测试集: 上涨={test_results.get('up_trend_accuracy', test_results.get('up_trend_recall', 0.0)):.4f} | 下跌={test_results.get('down_trend_accuracy', test_results.get('down_trend_recall', 0.0)):.4f}")
            logger.info(f"\n总体Recall(macro):")
            logger.info(f"  训练集: {train_results.get('overall_recall', 0.0):.4f}")
            logger.info(f"  验证集: {val_results.get('overall_recall', 0.0):.4f}")
            logger.info(f"  测试集: {test_results.get('overall_recall', 0.0):.4f}")
            # 分类别识别率（Recall和Precision）
            logger.info(f"\n分类别识别率（Recall）和精确率（Precision）：")
            logger.info(f"  训练集: 上涨 Recall={train_results.get('up_trend_accuracy', 0.0):.4f} Precision={train_results.get('up_trend_precision', 0.0):.4f} | 下跌 Recall={train_results.get('down_trend_accuracy', 0.0):.4f} Precision={train_results.get('down_trend_precision', 0.0):.4f}")
            logger.info(f"  验证集: 上涨 Recall={val_results.get('up_trend_accuracy', 0.0):.4f} Precision={val_results.get('up_trend_precision', 0.0):.4f} | 下跌 Recall={val_results.get('down_trend_accuracy', 0.0):.4f} Precision={val_results.get('down_trend_precision', 0.0):.4f}")
            logger.info(f"  测试集: 上涨 Recall={test_results.get('up_trend_accuracy', 0.0):.4f} Precision={test_results.get('up_trend_precision', 0.0):.4f} | 下跌 Recall={test_results.get('down_trend_accuracy', 0.0):.4f} Precision={test_results.get('down_trend_precision', 0.0):.4f}")
            logger.info(f"\n趋势检测指标（主指标）:")
            logger.info(f"  训练集趋势F1: {train_results['trend_f1']:.4f}")
            logger.info(f"    有趋势: Precision={train_results['trend_precision']:.4f}, Recall={train_results['trend_recall']:.4f}")
            if 'mid_trend_f1' in train_results:
                logger.info(f"    高权重趋势F1(>=0.7): {train_results['mid_trend_f1']:.4f}")
                if 'mid_accuracy' in train_results:
                    logger.info(f"    高权重命中率(>=0.7): {train_results['mid_accuracy']:.4f}")
                if 'mid_trend_recall' in train_results:
                    logger.info(f"    高权重Recall(>=0.7): {train_results['mid_trend_recall']:.4f}")
                if 'mid_up_accuracy' in train_results or 'mid_down_accuracy' in train_results:
                    logger.info(f"    高权重分类别识别率(>=0.7): 上涨 Recall={train_results.get('mid_up_accuracy', 0.0):.4f} Precision={train_results.get('mid_up_precision', 0.0):.4f} | 下跌 Recall={train_results.get('mid_down_accuracy', 0.0):.4f} Precision={train_results.get('mid_down_precision', 0.0):.4f} | 无趋势 Recall={train_results.get('mid_no_trend_accuracy', 0.0):.4f} Precision={train_results.get('mid_no_trend_precision', 0.0):.4f}")
                    if 'mid_trend_only_accuracy' in train_results:
                        logger.info(f"    高权重趋势内命中率(仅1/2): {train_results.get('mid_trend_only_accuracy', 0.0):.4f}")
            logger.info(f"  验证集趋势F1: {val_results['trend_f1']:.4f}")
            logger.info(f"    有趋势: Precision={val_results['trend_precision']:.4f}, Recall={val_results['trend_recall']:.4f}")
            if 'mid_trend_f1' in val_results:
                logger.info(f"    高权重趋势F1(>=0.7): {val_results['mid_trend_f1']:.4f}")
                if 'mid_accuracy' in val_results:
                    logger.info(f"    高权重命中率(>=0.7): {val_results['mid_accuracy']:.4f}")
                if 'mid_trend_recall' in val_results:
                    logger.info(f"    高权重Recall(>=0.7): {val_results['mid_trend_recall']:.4f}")
                if 'mid_up_accuracy' in val_results or 'mid_down_accuracy' in val_results:
                    logger.info(f"    高权重分类别识别率(>=0.7): 上涨={val_results.get('mid_up_accuracy', 0.0):.4f} | 下跌={val_results.get('mid_down_accuracy', 0.0):.4f} | 无趋势={val_results.get('mid_no_trend_accuracy', 0.0):.4f}")
                    if 'mid_trend_only_accuracy' in val_results:
                        logger.info(f"    高权重趋势内命中率(仅1/2): {val_results.get('mid_trend_only_accuracy', 0.0):.4f}")
            logger.info(f"  测试集趋势F1: {test_results['trend_f1']:.4f} ⭐")
            logger.info(f"    有趋势: Precision={test_results['trend_precision']:.4f}, Recall={test_results['trend_recall']:.4f}")
            if 'mid_trend_f1' in test_results:
                logger.info(f"    高权重趋势F1(>=0.7): {test_results['mid_trend_f1']:.4f}")
                if 'mid_accuracy' in test_results:
                    logger.info(f"    高权重命中率(>=0.7): {test_results['mid_accuracy']:.4f}")
                if 'mid_trend_recall' in test_results:
                    logger.info(f"    高权重Recall(>=0.7): {test_results['mid_trend_recall']:.4f}")
                if 'mid_up_accuracy' in test_results or 'mid_down_accuracy' in test_results:
                    logger.info(f"    高权重分类别识别率(>=0.7): 上涨={test_results.get('mid_up_accuracy', 0.0):.4f} | 下跌={test_results.get('mid_down_accuracy', 0.0):.4f} | 无趋势={test_results.get('mid_no_trend_accuracy', 0.0):.4f}")
                    if 'mid_trend_only_accuracy' in test_results:
                        logger.info(f"    高权重趋势内命中率(仅1/2): {test_results.get('mid_trend_only_accuracy', 0.0):.4f}")
            if test_split_results is not None:
                logger.info(f"\n测试集（test_early + test_late）综合评估:")
                logger.info(f"  准确率: {test_split_results['accuracy']:.4f}")
                logger.info(f"  趋势F1: {test_split_results['trend_f1']:.4f}")
                if 'mid_trend_f1' in test_split_results:
                    logger.info(f"  高权重样本趋势F1 (>=0.7): {test_split_results['mid_trend_f1']:.4f}")
            logger.info(f"\n模型文件: {model_file}")
            return {
                'model': model,
                'feature_names': feature_names,
                'train_results': train_results,
                'val_results': val_results,
                'test_results': test_results,
                'test_split_results': test_split_results,
                'importance_df': importance_df
            }


def main():
    """主函数"""
    trainer = TrendXGBoostTrainer()
    trainer.train()


if __name__ == '__main__':
    main()

