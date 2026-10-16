"""
XGBoost趋势预测模型训练
"""

import os
import sys

# Add parent directory to path to import config
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
import glob
from datetime import datetime
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    # 设置中文字体，避免中文显示警告
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
try:
    # package方式（python -m）优先
    from .simulation_core import simulate  # type: ignore
except Exception:
    # 脚本方式兜底
    from trade_simulation.simulation_core import simulate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        # apply目录路径
        self.apply_dir = os.path.join(project_root, 'apply')
    
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
            weight_down_trend = getattr(config, 'CLASS_WEIGHT_DOWN_TREND_3', 1.0)
            
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
            weight_no_trend = getattr(config, 'CLASS_WEIGHT_NO_TREND', 1.0)
            weight_trend = getattr(config, 'CLASS_WEIGHT_TREND', 2.0)
            
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
        model = XGBClassifier(
            **params,
            n_estimators=n_estimators,  # 设置较大的树数量，通过early stopping控制
            early_stopping_rounds=early_stopping_rounds,  # 验证集连续多少轮不改善则停止
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
                        subset_metrics = {
                            'mid_trend_precision': (class_report_mid['上涨趋势(1)']['precision'] + class_report_mid['下跌趋势(2)']['precision']) / 2,
                            'mid_trend_recall': (class_report_mid['上涨趋势(1)']['recall'] + class_report_mid['下跌趋势(2)']['recall']) / 2,
                            'mid_trend_f1': (class_report_mid['上涨趋势(1)']['f1-score'] + class_report_mid['下跌趋势(2)']['f1-score']) / 2,
                            'mid_accuracy': float((np.array(y_mid) == np.array(y_pred_mid)).mean())
                        }
                        logger.info(f"\n{dataset_name} 高权重样本评估（trend_position_weight >= 0.7，样本数: {subset_count}）:")
                        logger.info(f"  趋势Precision: {subset_metrics['mid_trend_precision']:.4f}")
                        logger.info(f"  趋势Recall:    {subset_metrics['mid_trend_recall']:.4f}")
                        logger.info(f"  趋势F1:        {subset_metrics['mid_trend_f1']:.4f}")
                        logger.info(f"  命中率(accuracy): {subset_metrics['mid_accuracy']:.4f}")
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
                'trend_recall': (up_trend_recall + down_trend_recall) / 2 if (up_trend_support > 0 or down_trend_support > 0) else 0.0
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
        # XGBClassifier使用feature_importances_属性
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        # 转换为DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
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
                    subset_metrics = {
                        'mid_trend_precision': (class_report_mid['上涨趋势(1)']['precision'] + class_report_mid['下跌趋势(2)']['precision']) / 2,
                        'mid_trend_recall': (class_report_mid['上涨趋势(1)']['recall'] + class_report_mid['下跌趋势(2)']['recall']) / 2,
                        'mid_trend_f1': (class_report_mid['上涨趋势(1)']['f1-score'] + class_report_mid['下跌趋势(2)']['f1-score']) / 2,
                        'mid_accuracy': float((np.array(y_mid) == np.array(y_pred_mid)).mean())
                    }
                    logger.info(f"\n{dataset_name} 高权重样本评估（trend_position_weight >= 0.7，样本数: {subset_count}）:")
                    logger.info(f"  趋势Precision: {subset_metrics['mid_trend_precision']:.4f}")
                    logger.info(f"  趋势Recall:    {subset_metrics['mid_trend_recall']:.4f}")
                    logger.info(f"  趋势F1:        {subset_metrics['mid_trend_f1']:.4f}")
                    logger.info(f"  命中率(accuracy): {subset_metrics['mid_accuracy']:.4f}")
                else:
                    logger.info(f"{dataset_name} 中 trend_position_weight >= 0.7 的样本数为0，跳过子集评估")
            except Exception as e:
                logger.warning(f"两阶段高权重样本分类报告生成失败: {e}")
        
        result = {
            'accuracy': accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'trend_f1': trend_f1,
            'y_true': y,
            'y_pred': y_pred,
            'p_trend': p_trend,
            'p_down': p_down
        }
        result.update(subset_metrics)
        return result
    
    def _optimize_two_stage_thresholds(self, y_val, p_trend_val, p_down_val, symbols_val=None, metric='accuracy'):
        """
        在验证集上寻优两阶段阈值：先全局，再每标的
        返回 thresholds 字典，结构见 evaluate_two_stage_model 注释
        """
        grid1 = getattr(config, 'THRESH_SEARCH_GRID_STAGE1_TREND', [0.5, 0.6])
        grid2 = getattr(config, 'THRESH_SEARCH_GRID', {
            'up_trend': [0.55, 0.60, 0.65],
            'down_trend': [0.60, 0.65, 0.70]
        })
        # 全局最优
        best = {'score': -1.0, 'stage1_trend': None, 'up_trend': None, 'down_trend': None}
        for t1 in grid1:
            for tu in grid2.get('up_trend', [0.6]):
                for td in grid2.get('down_trend', [0.65]):
                    y_pred = self._two_stage_predict_labels(p_trend_val, p_down_val, t1, tu, td)
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
            per_s1 = {}
            per_s2 = {}
            for sym, g in df_val.groupby('symbol'):
                y_g = g['y_true'].values
                pt = g['p_trend'].values
                pdn = g['p_down'].values
                best_g = {'score': -1.0, 'stage1_trend': thresholds['stage1']['global']['trend'], 
                          'up_trend': thresholds['stage2']['global']['up_trend'], 
                          'down_trend': thresholds['stage2']['global']['down_trend']}
                for t1 in grid1:
                    for tu in grid2.get('up_trend', [thresholds['stage2']['global']['up_trend']]):
                        for td in grid2.get('down_trend', [thresholds['stage2']['global']['down_trend']]):
                            y_pred = self._two_stage_predict_labels(pt, pdn, t1, tu, td)
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
            # 也保存stage2为XGB原生格式做参考
            xgb_file = os.path.join(self.model_dir, f"{file_stub}.xgb")
            model_stage2.get_booster().save_model(xgb_file)
            logger.info(f"两阶段Stage2模型(XGBoost格式)已保存到: {xgb_file}")
        except Exception as e:
            logger.warning(f"保存两阶段XGBoost格式失败: {e}")
        return model_file
    
    def plot_total_assets(self, daily_balances_df, output_path, initial_capital=None):
        """
        绘制总资产随时间变化的图表
        
        Args:
            daily_balances_df: 每日余额DataFrame，包含date和balance列
            output_path: 图表保存路径
            initial_capital: 初始资金（用于绘制基准线）
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib不可用，跳过图表生成")
            return
        
        if daily_balances_df is None or len(daily_balances_df) == 0:
            logger.warning("每日余额数据为空，跳过图表生成")
            return
        
        try:
            # 确保日期列存在且为datetime类型
            if 'date' not in daily_balances_df.columns or 'balance' not in daily_balances_df.columns:
                logger.warning("每日余额数据缺少date或balance列，跳过图表生成")
                return
            
            df = daily_balances_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['balance'], linewidth=2, label='总资产', color='#2E86AB')
            
            # 如果有初始资金，绘制基准线
            if initial_capital is not None:
                plt.axhline(y=initial_capital, color='gray', linestyle='--', linewidth=1, 
                           label=f'初始资金 ({initial_capital:,.0f})', alpha=0.7)
            
            # 计算总收益率
            if len(df) > 0:
                final_balance = df['balance'].iloc[-1]
                if initial_capital is not None and initial_capital > 0:
                    total_return = (final_balance - initial_capital) / initial_capital * 100
                    plt.title(f'总资产变化曲线\n最终资产: {final_balance:,.2f} | 总收益率: {total_return:.2f}%', 
                             fontsize=14, fontweight='bold')
                else:
                    plt.title(f'总资产变化曲线\n最终资产: {final_balance:,.2f}', 
                             fontsize=14, fontweight='bold')
            
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('总资产 ($)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"总资产图表已保存: {output_path}")
        except Exception as e:
            logger.warning(f"生成总资产图表失败: {e}")
            try:
                plt.close()
            except:
                pass
    
    def load_model_from_apply(self, apply_dir=None):
        """
        从 apply/model 目录加载最新的模型
        如果模型名称包含 long_only 或 short_only，自动寻找配对模型
        """
        if apply_dir is None:
            apply_dir = self.apply_dir
        
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
        bundle = {
            'model_path': model_path,  # 保存模型路径，用于后续打印
            'feature_names': data['feature_names'],
            'encoders': data.get('encoders', None),
            'calibration': data.get('calibration', None),
            'timestamp': data.get('timestamp', 'unknown'),
            'two_stage': data.get('two_stage', False),
            'long_only': data.get('long_only', False),
            'short_only': data.get('short_only', False),
            'has_paired_model': paired_data is not None,
        }
        
        if bundle['two_stage']:
            bundle['model_stage1'] = data.get('model_stage1')
            # 兼容普通两阶段与长多两阶段
            if bundle['long_only']:
                bundle['model_stage2_up'] = data.get('model_stage2_up')
                if bundle['model_stage1'] is None or bundle['model_stage2_up'] is None:
                    raise ValueError("长多两阶段模型缺少 stage1 或 stage2_up")
            elif bundle['short_only']:
                bundle['model_stage2_down'] = data.get('model_stage2_down')
                if bundle['model_stage1'] is None or bundle['model_stage2_down'] is None:
                    raise ValueError("短空两阶段模型缺少 stage1 或 stage2_down")
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
        
        # 附加编码器
        if bundle['encoders']:
            encoders = bundle['encoders']
            if encoders.get('sector_encoder'):
                self.sector_encoder = encoders['sector_encoder']
            if encoders.get('industry_encoder'):
                self.industry_encoder = encoders['industry_encoder']
            if encoders.get('sub_industry_encoder'):
                self.sub_industry_encoder = encoders['sub_industry_encoder']
            logger.info("已加载并附加LabelEncoders")
        return bundle
    
    def train(self, symbol=None, start_date=None, end_date=None):
        """
        使用apply目录中的模型进行仿真（不训练）
        
        Args:
            symbol: 可选，指定要回测的单个标的（如 'AAPL'）。如果为None，则回测所有标的
            start_date: 可选，指定回测开始日期（格式：'YYYY-MM-DD'）。如果为None，则使用config.SIM_START_DATE
            end_date: 可选，指定回测结束日期（格式：'YYYY-MM-DD'）。如果为None，则使用config.SIM_END_DATE
        """
        logger.info("=" * 60)
        logger.info("使用apply目录中的模型进行仿真（跳过训练）")
        if symbol:
            logger.info(f"回测标的: {symbol}")
        if start_date:
            logger.info(f"自定义开始日期: {start_date}")
        if end_date:
            logger.info(f"自定义结束日期: {end_date}")
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
        
        # 1. 加载数据（仅使用 data/xgb/test/all_symbols.csv 作为仿真数据源）
        logger.info("\n1. 加载数据（仅使用 data/xgb/test/all_symbols.csv）...")
        try:
            full_df = self.load_data(self.test_dir, use_all_symbols=True, dataset_name='测试/留出集')
        except FileNotFoundError as e:
            logger.error(f"数据加载失败: {e}")
            logger.error("请先运行 data_splitter.py 或确保 data/xgb/test/all_symbols.csv 存在")
            return None
        # 统一时间并排序
        if 'date' not in full_df.columns:
            logger.error("数据缺少 date 列")
            return None
        full_df['date'] = pd.to_datetime(full_df['date'])
        full_df = full_df.sort_values('date').reset_index(drop=True)
        # 使用固定日期范围：优先使用传入参数，否则从测试集获取，最后使用config
        if len(full_df) == 0:
            logger.error("数据为空")
            return None
        
        # 确定日期范围：优先使用传入参数，其次使用config中的SIM_START_DATE和SIM_END_DATE，最后从测试集获取
        # 优先级：用户传入参数 > config配置 > 测试集日期范围
        if start_date is not None:
            start_date_fixed = pd.to_datetime(start_date)
            logger.info(f"使用用户指定的开始日期: {start_date_fixed.date()}")
        else:
            # 优先使用config中的SIM_START_DATE
            if hasattr(config, 'SIM_START_DATE') and config.SIM_START_DATE:
                start_date_fixed = pd.to_datetime(config.SIM_START_DATE)
                logger.info(f"使用config.SIM_START_DATE: {start_date_fixed.date()}")
            else:
                # 如果config中没有，尝试从测试集获取
                try:
                    test_df = self.load_data(self.test_dir, use_all_symbols=True, dataset_name='测试集')
                    if len(test_df) > 0 and 'date' in test_df.columns:
                        test_df['date'] = pd.to_datetime(test_df['date'])
                        start_date_fixed = test_df['date'].min()
                        logger.info(f"从测试集获取开始日期: {start_date_fixed.date()}")
                    else:
                        start_date_fixed = pd.to_datetime('2024-05-01')
                        logger.warning(f"测试集为空，使用默认开始日期: {start_date_fixed.date()}")
                except (FileNotFoundError, Exception) as e:
                    start_date_fixed = pd.to_datetime('2024-05-01')
                    logger.warning(f"无法加载测试集数据，使用默认开始日期: {start_date_fixed.date()}")
        
        if end_date is not None:
            end_date_fixed = pd.to_datetime(end_date)
            logger.info(f"使用用户指定的结束日期: {end_date_fixed.date()}")
        else:
            # 优先使用config中的SIM_END_DATE
            if hasattr(config, 'SIM_END_DATE') and config.SIM_END_DATE:
                end_date_fixed = pd.to_datetime(config.SIM_END_DATE)
                logger.info(f"使用config.SIM_END_DATE: {end_date_fixed.date()}")
            else:
                # 如果config中没有，尝试从测试集获取
                try:
                    test_df = self.load_data(self.test_dir, use_all_symbols=True, dataset_name='测试集')
                    if len(test_df) > 0 and 'date' in test_df.columns:
                        test_df['date'] = pd.to_datetime(test_df['date'])
                        end_date_fixed = test_df['date'].max()
                        logger.info(f"从测试集获取结束日期: {end_date_fixed.date()}")
                    else:
                        end_date_fixed = pd.to_datetime('2025-11-01')
                        logger.warning(f"测试集为空，使用默认结束日期: {end_date_fixed.date()}")
                except (FileNotFoundError, Exception) as e:
                    end_date_fixed = pd.to_datetime('2025-11-01')
                    logger.warning(f"无法加载测试集数据，使用默认结束日期: {end_date_fixed.date()}")
        
        logger.info(f"使用日期范围：{start_date_fixed.date()} ~ {end_date_fixed.date()}")
        
        # 按日期过滤
        sim_df = full_df[(full_df['date'] >= start_date_fixed) & (full_df['date'] <= end_date_fixed)].copy()
        
        # 如果指定了标的，只保留该标的的数据
        if symbol is not None:
            symbol_upper = str(symbol).upper()
            if 'symbol' not in sim_df.columns:
                logger.error("数据中缺少 'symbol' 列，无法按标的过滤")
                return None
            before_count = len(sim_df)
            sim_df = sim_df[sim_df['symbol'].str.upper() == symbol_upper].copy()
            after_count = len(sim_df)
            logger.info(f"按标的过滤：{symbol_upper}，过滤前 {before_count} 行，过滤后 {after_count} 行")
            if after_count == 0:
                logger.error(f"未找到标的 {symbol_upper} 在指定日期范围内的数据")
                return None
        if len(sim_df) < 20:
            logger.error(f"数据不足：sim={len(sim_df)}")
            logger.error(f"请检查数据源是否包含 {start_date_fixed.date()} ~ {end_date_fixed.date()} 的数据")
            return None
        logger.info(f"日期过滤完成：仿真={len(sim_df)} 行")
        logger.info(f"仿真起止：{sim_df['date'].min()} ~ {sim_df['date'].max()}")
        
        # 2. 从apply目录加载模型（跳过训练）
        logger.info("\n2. 从apply目录加载模型...")
        try:
            model_bundle = self.load_model_from_apply()
            feature_names = model_bundle['feature_names']
            use_two_stage = model_bundle.get('two_stage', False)
            is_long_only = bool(model_bundle.get('long_only', False))
            logger.info(f"模型特征数: {len(feature_names)}")
            if use_two_stage and is_long_only:
                logger.info("模型类型: 两阶段（长多：Stage1趋势识别 + Stage2上涨vs下跌）")
            elif use_two_stage:
                logger.info("模型类型: 两阶段（通用涨/跌模型）")
            else:
                logger.info("模型类型: 单模型（三分类或二分类）")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None
        
        # 3. 准备仿真集特征
        logger.info("\n3. 准备仿真集特征...")
        use_time_features = getattr(config, 'USE_TIME_FEATURES', True)
        use_raw_ohlcv = getattr(config, 'USE_RAW_OHLCV', False)
        use_industry_features = getattr(config, 'USE_INDUSTRY_FEATURES', True)
        
        # 准备仿真集特征（使用已加载的编码器）
        # 保存原始索引映射，以便后续对齐
        sim_df_indexed = sim_df.reset_index(drop=True)
        X_sim, y_sim, _, pos_w_sim, _, _, symbols_sim, use_3_class = self.prepare_features(
            sim_df_indexed,
            use_time_features=use_time_features,
            use_raw_ohlcv=use_raw_ohlcv,
            use_industry_features=use_industry_features,
            fit_encoders=False  # 使用已加载的编码器
        )
        # 确保特征顺序一致
        X_sim = X_sim[feature_names]
        
        # 记录过滤后的数据量
        logger.info(f"特征准备完成：原始数据 {len(sim_df_indexed)} 行，过滤后 {len(X_sim)} 行")
        if len(X_sim) < len(sim_df_indexed):
            logger.warning(f"数据过滤：移除了 {len(sim_df_indexed) - len(X_sim)} 行（可能是标签为NaN的行）")
        
        # 4. 加载阈值
        logger.info("\n4. 加载阈值...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_dir = os.path.join(project_root, 'models', 'result')
        thresholds_ts = None
        try:
            cands = []
            cands += glob.glob(os.path.join(result_dir, "class_thresholds_two_stage_*.json"))
            cands += glob.glob(os.path.join(result_dir, "class_thresholds_*.json"))
            cands = sorted(cands, key=os.path.getmtime, reverse=True)
            if cands:
                with open(cands[0], 'r', encoding='utf-8') as f:
                    thresholds_ts = json.load(f)
                logger.info(f"已加载阈值文件: {cands[0]}")
        except Exception as e:
            logger.warning(f"加载阈值失败，将使用默认阈值: {e}")
        
        # 5. 对仿真集进行预测
        logger.info("\n5. 对仿真集进行预测...")
        # 检查是否优先使用 with_prediction 数据
        prefer_with_prediction = getattr(config, 'SIM_PREFER_WITH_PREDICTION_DATA', False)
        
        # 优先使用 data/with_prediction/all_with_prediction.csv 直接仿真（如果配置允许）
        if prefer_with_prediction:
            try:
                all_pred_path = os.path.join(project_root, 'data', 'with_prediction', 'all_with_prediction.csv')
                if os.path.exists(all_pred_path):
                    logger.info("检测到 all_with_prediction.csv，优先使用其作为仿真输入")
                    df_ap = pd.read_csv(all_pred_path)
                    if 'date' not in df_ap.columns:
                        raise ValueError("all_with_prediction 缺少 date 列")
                    df_ap['date'] = pd.to_datetime(df_ap['date'])
                    
                    # 检查原始数据的日期范围
                    original_min_date = df_ap['date'].min()
                    original_max_date = df_ap['date'].max()
                    logger.info(f"all_with_prediction.csv 原始数据日期范围: {original_min_date.date()} ~ {original_max_date.date()}")
                    logger.info(f"设定的模拟日期范围: {start_date_fixed.date()} ~ {end_date_fixed.date()}")
                    
                    # 固定窗口过滤
                    df_ap = df_ap[(df_ap['date'] >= start_date_fixed) & (df_ap['date'] <= end_date_fixed)].copy()
                    
                    # 检查过滤后的数据范围
                    if len(df_ap) > 0:
                        filtered_min_date = df_ap['date'].min()
                        filtered_max_date = df_ap['date'].max()
                        logger.info(f"过滤后数据日期范围: {filtered_min_date.date()} ~ {filtered_max_date.date()}")
                        
                        # 如果数据范围小于设定范围，给出警告
                        if filtered_max_date < end_date_fixed:
                            logger.warning(f"⚠️  数据只到 {filtered_max_date.date()}，小于设定的结束日期 {end_date_fixed.date()}")
                            logger.warning(f"⚠️  模拟将只运行到 {filtered_max_date.date()}，请运行 apply_model.py 生成到 {end_date_fixed.date()} 的预测数据")
                        if filtered_min_date > start_date_fixed:
                            logger.warning(f"⚠️  数据从 {filtered_min_date.date()} 开始，大于设定的开始日期 {start_date_fixed.date()}")
                    else:
                        logger.error(f"过滤后数据为空，请检查 all_with_prediction.csv 是否包含 {start_date_fixed.date()} ~ {end_date_fixed.date()} 的数据")
                    
                    if 'symbol' not in df_ap.columns:
                        raise ValueError("all_with_prediction 缺少 symbol 列")
                    
                    # 如果指定了标的，只保留该标的的数据
                    if symbol is not None:
                        symbol_upper = str(symbol).upper()
                        before_count = len(df_ap)
                        df_ap = df_ap[df_ap['symbol'].str.upper() == symbol_upper].copy()
                        after_count = len(df_ap)
                        logger.info(f"按标的过滤 all_with_prediction：{symbol_upper}，过滤前 {before_count} 行，过滤后 {after_count} 行")
                        if after_count == 0:
                            logger.error(f"未找到标的 {symbol_upper} 在 all_with_prediction 中指定日期范围内的数据")
                            raise ValueError(f"未找到标的 {symbol_upper} 的数据")
                    # 概率列重命名兼容
                    rename_map = {}
                    if 'mdl_p_no_trend' not in df_ap.columns and 'p_no_trend' in df_ap.columns:
                        rename_map['p_no_trend'] = 'mdl_p_no_trend'
                    if 'mdl_p_up' not in df_ap.columns and 'p_up' in df_ap.columns:
                        rename_map['p_up'] = 'mdl_p_up'
                    if 'mdl_p_down' not in df_ap.columns and 'p_down' in df_ap.columns:
                        rename_map['p_down'] = 'mdl_p_down'
                    if rename_map:
                        df_ap = df_ap.rename(columns=rename_map)
                    # 仅有 p_trend 时的兜底（用 mdl_pred 将 p_trend 分到 up/down）
                    if ('mdl_p_up' not in df_ap.columns or 'mdl_p_down' not in df_ap.columns) and ('p_trend' in df_ap.columns):
                        if 'mdl_p_no_trend' not in df_ap.columns and 'p_no_trend' in df_ap.columns:
                            df_ap['mdl_p_no_trend'] = df_ap['p_no_trend'].astype(float)
                        elif 'mdl_p_no_trend' not in df_ap.columns:
                            df_ap['mdl_p_no_trend'] = (1.0 - pd.to_numeric(df_ap['p_trend'], errors='coerce').fillna(0.0)).clip(0.0, 1.0)
                        if 'mdl_pred' in df_ap.columns:
                            p_tr = pd.to_numeric(df_ap['p_trend'], errors='coerce').fillna(0.0)
                            df_ap['mdl_p_up'] = np.where(pd.to_numeric(df_ap['mdl_pred'], errors='coerce').fillna(0).astype(int) == 1, p_tr, 0.0)
                            df_ap['mdl_p_down'] = np.where(pd.to_numeric(df_ap['mdl_pred'], errors='coerce').fillna(0).astype(int) == 2, p_tr, 0.0)
                        else:
                            df_ap['mdl_p_up'] = 0.0
                            df_ap['mdl_p_down'] = 0.0
                    # 必需列检查
                    required_cols = ['date', 'symbol', 'open', 'mdl_pred', 'mdl_p_no_trend', 'mdl_p_up', 'mdl_p_down']
                    missing = [c for c in required_cols if c not in df_ap.columns]
                    if missing:
                        raise ValueError(f"all_with_prediction 缺少必要列: {missing}")
                    sim_df_all = df_ap[required_cols].copy()
                    # 诊断信息
                    logger.info(f"使用 all_with_prediction：{len(sim_df_all)} 行，起止 {sim_df_all['date'].min()} ~ {sim_df_all['date'].max()}，标的 {sim_df_all['symbol'].nunique()}")
                    # 执行模拟
                    allow_long = bool(getattr(config, 'STRAT_ALLOW_LONG', True))
                    allow_short = bool(getattr(config, 'STRAT_ALLOW_SHORT', True))
                    trades_df, summary_df, final_cash, stopped_early, daily_positions_df, daily_balances_df, sharpe_ratio = simulate(
                        df_all=sim_df_all,
                        up_thr=float(getattr(config, 'CLASS_PROB_THRESHOLD_UP_TREND', 0.55)),
                        alloc_cap=float(getattr(config, 'STRAT_ALLOC_CAP', 1.0)),
                        per_symbol_cap=float(getattr(config, 'STRAT_PER_SYMBOL_CAP', 0.25)),
                        fee_bps=float(getattr(config, 'STRAT_FEE_BPS', 2.0)),
                        slippage_bps=float(getattr(config, 'STRAT_SLIPPAGE_BPS', 1.0)),
                        initial_capital=float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0)),
                        allow_long=allow_long,
                        allow_short=allow_short,
                        start_date=str(getattr(config, 'SIM_START_DATE', '')),
                        end_date=str(getattr(config, 'SIM_END_DATE', '')),
                    )
                    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "result", "trading_result")
                    os.makedirs(out_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    trades_path = os.path.join(out_dir, f"trade_sim_oos_{ts}.csv")
                    try:
                        if 'entry_date' in trades_df.columns:
                            trades_df = trades_df.sort_values('entry_date').reset_index(drop=True)
                    except Exception:
                        pass
                    # 重新排列列顺序：确保包含所需字段并置前
                    desired = [
                        "entry_date", "exit_date", "symbol", "entry_price", "exit_price",
                        "price_diff", "shares", "open_cash_flow", "close_cash_flow",
                        "net_pnl", "profit_pct", "equity_before_open", "equity_after_close",
                        "entry_p_no", "entry_p_up", "entry_p_down",
                        "exit_p_no", "exit_p_up", "exit_p_down"
                    ]
                    cols = [c for c in desired if c in trades_df.columns] + [c for c in trades_df.columns if c not in desired]
                    trades_df = trades_df[cols]
                    trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
                    summary_path = os.path.join(out_dir, f"trade_sim_oos_summary_{ts}.csv")
                    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
                    positions_path = os.path.join(out_dir, f"trade_sim_oos_daily_positions_{ts}.csv")
                    daily_positions_df.to_csv(positions_path, index=False, encoding='utf-8-sig')
                    balances_path = os.path.join(out_dir, f"trade_sim_oos_daily_balances_{ts}.csv")
                    daily_balances_df.to_csv(balances_path, index=False, encoding='utf-8-sig')
                    # 绘制总资产图表
                    chart_path = os.path.join(out_dir, f"trade_sim_oos_total_assets_{ts}.png")
                    initial_capital = float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0))
                    self.plot_total_assets(daily_balances_df, chart_path, initial_capital=initial_capital)
                    logger.info(f"仿真交易完成：{trades_path}，汇总：{summary_path}")
                    if sharpe_ratio is not None:
                        total_return = (final_cash - float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0))) / float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0)) * 100
                        logger.info(f"总收益率：{total_return:.2f}%")
                        logger.info(f"夏普比：{sharpe_ratio:.4f}")
                    # 打印使用的模型信息
                    model_path = model_bundle.get('model_path', '未知')
                    model_name = os.path.basename(model_path) if model_path != '未知' else '未知'
                    has_paired = bool(model_bundle.get('has_paired_model', False))
                    if has_paired:
                        paired_model_path = model_bundle.get('paired_model_path', '未知')
                        paired_model_name = os.path.basename(paired_model_path) if paired_model_path != '未知' else '未知'
                        model_type = "长多+短空两阶段（合并）"
                        logger.info(f"\n使用的主模型: {model_name}")
                        logger.info(f"  路径: {model_path}")
                        logger.info(f"使用的配对模型: {paired_model_name}")
                        logger.info(f"  路径: {paired_model_path}")
                        logger.info(f"模型类型: {model_type}")
                    else:
                        is_long_only = bool(model_bundle.get('long_only', False))
                        is_short_only = bool(model_bundle.get('short_only', False))
                        if is_long_only:
                            model_type = "长多两阶段"
                        elif is_short_only:
                            model_type = "短空两阶段"
                        else:
                            model_type = "两阶段" if use_two_stage else "单模型"
                        logger.info(f"\n使用的模型: {model_name}")
                        logger.info(f"  路径: {model_path}")
                        logger.info(f"模型类型: {model_type}")
                    if 'timestamp' in model_bundle:
                        logger.info(f"模型时间戳: {model_bundle.get('timestamp', '未知')}")
                    # 成功执行后直接返回，避免重复运行
                    return {
                        'model': model_bundle.get('model') if not use_two_stage else {
                            'stage1': model_bundle.get('model_stage1'),
                            'stage2': model_bundle.get('model_stage2') if not model_bundle.get('long_only', False) else model_bundle.get('model_stage2_up')
                        },
                        'feature_names': feature_names
                    }
                else:
                    logger.info("all_with_prediction.csv 不存在，将使用实时预测")
            except Exception as e:
                logger.warning(f"使用 all_with_prediction 直接仿真失败，将退回到模型内预测路径：{e}")
                import traceback
                logger.debug(f"异常详情：{traceback.format_exc()}")
        else:
            logger.info("SIM_PREFER_WITH_PREDICTION_DATA=False，将使用实时预测（不使用 with_prediction 文件）")
        if use_two_stage:
            is_long_only = bool(model_bundle.get('long_only', False))
            is_short_only = bool(model_bundle.get('short_only', False))
            has_paired = bool(model_bundle.get('has_paired_model', False))
            model_stage1 = model_bundle['model_stage1']
            
            # 使用两阶段模型预测
            if has_paired:
                # 有配对模型：合并 long_only 和 short_only 的输出
                paired_model_stage1 = model_bundle.get('paired_model_stage1')
                
                # Stage1: 使用两个模型的 Stage1 平均
                proba1 = model_stage1.predict_proba(X_sim)
                p_trend = proba1[:, 1]
                if paired_model_stage1 is not None:
                    proba1_paired = paired_model_stage1.predict_proba(X_sim)
                    p_trend = (p_trend + proba1_paired[:, 1]) / 2.0
                    logger.info("使用两个模型的 Stage1 平均作为 P(trend)")
                
                # Stage2: 根据模型类型获取对应的 Stage2
                if is_long_only:
                    # 主模型是 long_only，配对模型是 short_only
                    model_stage2_up = model_bundle.get('model_stage2_up')
                    paired_model_stage2_down = model_bundle.get('paired_model_stage2_down')
                    if model_stage2_up is None or paired_model_stage2_down is None:
                        raise ValueError("配对模型缺少必要的 Stage2 模型")
                    proba2_up = model_stage2_up.predict_proba(X_sim)  # [:,1] = P(up | trend)
                    proba2_down = paired_model_stage2_down.predict_proba(X_sim)  # [:,1] = P(down | trend)
                    p_up_cond = proba2_up[:, 1]
                    p_down_cond = proba2_down[:, 1]
                    logger.info("合并 long_only 和 short_only 模型的输出")
                elif is_short_only:
                    # 主模型是 short_only，配对模型是 long_only
                    model_stage2_down = model_bundle.get('model_stage2_down')
                    paired_model_stage2_up = model_bundle.get('paired_model_stage2_up')
                    if model_stage2_down is None or paired_model_stage2_up is None:
                        raise ValueError("配对模型缺少必要的 Stage2 模型")
                    proba2_down = model_stage2_down.predict_proba(X_sim)  # [:,1] = P(down | trend)
                    proba2_up = paired_model_stage2_up.predict_proba(X_sim)  # [:,1] = P(up | trend)
                    p_down_cond = proba2_down[:, 1]
                    p_up_cond = proba2_up[:, 1]
                    logger.info("合并 short_only 和 long_only 模型的输出")
                else:
                    raise ValueError("配对模型但无法确定模型类型")
                
                # 计算无条件概率
                p_up = p_trend * p_up_cond
                p_down = p_trend * p_down_cond
                p_no = 1.0 - p_trend
                
                # 预测标签：按无条件概率argmax得到 0/1/2
                stacked = np.vstack([p_no, p_up, p_down]).T
                mdl_pred = np.argmax(stacked, axis=1).astype(int)
                preds = pd.DataFrame({
                    'mdl_pred': mdl_pred,
                    'mdl_p_no_trend': p_no,
                    'mdl_p_up': p_up,
                    'mdl_p_down': p_down,
                })
                # 使用过滤后的数据（与X_sim对齐）
                sim_df_filtered = sim_df_indexed.iloc[:len(X_sim)].copy() if len(X_sim) <= len(sim_df_indexed) else sim_df_indexed.copy()
                if 'date' in sim_df_filtered.columns:
                    sim_df_filtered['date'] = pd.to_datetime(sim_df_filtered['date'])
                # 构建包含策略所需列的数据框（包括MA20等）
                base_cols = ['date', 'symbol', 'open']
                strategy_cols = ['MA20']  # 策略可能需要的列
                available_cols = base_cols + [col for col in strategy_cols if col in sim_df_filtered.columns]
                sim_df_all = pd.concat(
                    [sim_df_filtered[available_cols].reset_index(drop=True),
                     preds.reset_index(drop=True)],
                    axis=1
                )
            elif is_long_only:
                # 长多两阶段：Stage1 提供 P(trend)，Stage2_up 提供 P(up | trend)
                model_stage2_up = model_bundle['model_stage2_up']
                proba1 = model_stage1.predict_proba(X_sim)
                p_trend = proba1[:, 1]
                proba2_up = model_stage2_up.predict_proba(X_sim)
                p_up_cond = proba2_up[:, 1]
                # 组合为无条件概率
                p_up = p_trend * p_up_cond
                p_down = p_trend * (1.0 - p_up_cond)
                p_no = 1.0 - p_trend
                # 预测标签：按无条件概率argmax得到 0/1/2
                stacked = np.vstack([p_no, p_up, p_down]).T
                mdl_pred = np.argmax(stacked, axis=1).astype(int)
                preds = pd.DataFrame({
                    'mdl_pred': mdl_pred,
                    'mdl_p_no_trend': p_no,
                    'mdl_p_up': p_up,
                    'mdl_p_down': p_down,
                })
                # 使用过滤后的数据（与X_sim对齐）
                sim_df_filtered = sim_df_indexed.iloc[:len(X_sim)].copy() if len(X_sim) <= len(sim_df_indexed) else sim_df_indexed.copy()
                if 'date' in sim_df_filtered.columns:
                    sim_df_filtered['date'] = pd.to_datetime(sim_df_filtered['date'])
                # 构建包含策略所需列的数据框（包括MA20等）
                base_cols = ['date', 'symbol', 'open']
                strategy_cols = ['MA20']  # 策略可能需要的列
                available_cols = base_cols + [col for col in strategy_cols if col in sim_df_filtered.columns]
                sim_df_all = pd.concat(
                    [sim_df_filtered[available_cols].reset_index(drop=True),
                     preds.reset_index(drop=True)],
                    axis=1
                )
            elif is_short_only:
                # 短空两阶段：Stage1 提供 P(trend)，Stage2_down 提供 P(down | trend)
                model_stage2_down = model_bundle['model_stage2_down']
                proba1 = model_stage1.predict_proba(X_sim)
                p_trend = proba1[:, 1]
                proba2_down = model_stage2_down.predict_proba(X_sim)
                p_down_cond = proba2_down[:, 1]
                # 组合为无条件概率
                p_down = p_trend * p_down_cond
                p_up = p_trend * (1.0 - p_down_cond)
                p_no = 1.0 - p_trend
                # 预测标签：按无条件概率argmax得到 0/1/2
                stacked = np.vstack([p_no, p_up, p_down]).T
                mdl_pred = np.argmax(stacked, axis=1).astype(int)
                preds = pd.DataFrame({
                    'mdl_pred': mdl_pred,
                    'mdl_p_no_trend': p_no,
                    'mdl_p_up': p_up,
                    'mdl_p_down': p_down,
                })
                # 使用过滤后的数据（与X_sim对齐）
                sim_df_filtered = sim_df_indexed.iloc[:len(X_sim)].copy() if len(X_sim) <= len(sim_df_indexed) else sim_df_indexed.copy()
                if 'date' in sim_df_filtered.columns:
                    sim_df_filtered['date'] = pd.to_datetime(sim_df_filtered['date'])
                # 构建包含策略所需列的数据框（包括MA20等）
                base_cols = ['date', 'symbol', 'open']
                strategy_cols = ['MA20']  # 策略可能需要的列
                available_cols = base_cols + [col for col in strategy_cols if col in sim_df_filtered.columns]
                sim_df_all = pd.concat(
                    [sim_df_filtered[available_cols].reset_index(drop=True),
                     preds.reset_index(drop=True)],
                    axis=1
                )
            else:
                # 非长多/短空模型：使用标准的 model_stage2，并通过 evaluate_two_stage_model 得到预测概率
                model_stage2 = model_bundle.get('model_stage2')
                if model_stage2 is None:
                    raise ValueError("两阶段模型缺少 model_stage2，且不是长多/短空模型")

                eval_sim = self.evaluate_two_stage_model(
                    model_stage1,
                    model_stage2,
                    X_sim,
                    y_sim,
                    '仿真集(两阶段)',
                    symbols=symbols_sim,
                    thresholds=thresholds_ts,
                    pos_weights=pos_w_sim
                )

                # 组装仿真输入表
                # 使用过滤后的数据（与X_sim对齐）
                sim_df_filtered = sim_df_indexed.iloc[:len(X_sim)].copy() if len(X_sim) <= len(sim_df_indexed) else sim_df_indexed.copy()
                if 'date' in sim_df_filtered.columns:
                    sim_df_filtered['date'] = pd.to_datetime(sim_df_filtered['date'])
                
                # 确保预测结果长度与过滤后的数据一致
                n_preds = len(eval_sim['y_pred'])
                n_data = len(sim_df_filtered)
                if n_preds != n_data:
                    logger.warning(f"预测结果长度({n_preds})与数据长度({n_data})不匹配，将截断或填充")
                    min_len = min(n_preds, n_data)
                    preds = pd.DataFrame({
                        'mdl_pred': eval_sim['y_pred'][:min_len]
                    })
                    p_trend = eval_sim['p_trend'][:min_len]
                    p_down_cond = eval_sim['p_down'][:min_len]
                    sim_df_filtered = sim_df_filtered.iloc[:min_len].copy()
                else:
                    preds = pd.DataFrame({
                        'mdl_pred': eval_sim['y_pred']
                    })
                    p_trend = eval_sim['p_trend']
                    p_down_cond = eval_sim['p_down']
                
                # 计算无条件概率
                p_down = p_trend * p_down_cond
                p_up = p_trend * (1.0 - p_down_cond)
                p_no = 1.0 - p_trend
                preds['mdl_p_no_trend'] = p_no
                preds['mdl_p_up'] = p_up
                preds['mdl_p_down'] = p_down
                
                # 合并到 sim_df_filtered（包含策略所需列）
                base_cols = ['date', 'symbol', 'open']
                strategy_cols = ['MA20']  # 策略可能需要的列
                available_cols = base_cols + [col for col in strategy_cols if col in sim_df_filtered.columns]
                sim_df_all = pd.concat(
                    [sim_df_filtered[available_cols].reset_index(drop=True),
                     preds.reset_index(drop=True)],
                    axis=1
                )
            
            # 验证数据完整性
            logger.info(f"仿真输入表准备完成：{len(sim_df_all)} 行")
            logger.info(f"日期范围：{sim_df_all['date'].min()} 到 {sim_df_all['date'].max()}")
            logger.info(f"标的数量：{sim_df_all['symbol'].nunique()}")
            logger.info(f"预测分布：无趋势={(preds['mdl_pred']==0).sum()}, 上涨={(preds['mdl_pred']==1).sum()}, 下跌={(preds['mdl_pred']==2).sum()}")
            
            # 输出完整的交易逻辑
            allow_long = bool(getattr(config, 'STRAT_ALLOW_LONG', True))
            allow_short = bool(getattr(config, 'STRAT_ALLOW_SHORT', True))
            logger.info("\n" + "=" * 80)
            logger.info("完整交易策略说明（两阶段模型）")
            logger.info("=" * 80)
            logger.info(f"做多功能：{'已启用' if allow_long else '已禁用'}")
            logger.info(f"做空功能：{'已启用' if allow_short else '已禁用'}")
            # 止盈 / 止损配置说明
            use_tp = bool(getattr(config, 'STRAT_USE_TAKE_PROFIT', False))
            use_sl = bool(getattr(config, 'STRAT_USE_STOP_LOSS', False))
            tp_pct = float(getattr(config, 'STRAT_TAKE_PROFIT_PCT', 0.10))
            sl_pct = float(getattr(config, 'STRAT_STOP_LOSS_PCT', 0.08))
            logger.info("")
            logger.info("止盈 / 止损配置：")
            logger.info(f"  - 止盈开关：{'开启' if use_tp else '关闭'}（阈值: {tp_pct * 100:.1f}%）")
            logger.info(f"  - 止损开关：{'开启' if use_sl else '关闭'}（阈值: {sl_pct * 100:.1f}%）")
            logger.info("")
            if allow_long:
                long_conf_thr = float(getattr(config, 'STRAT_LONG_CONFIDENCE_THRESHOLD', 0.90))
                long_other_thr = float(getattr(config, 'STRAT_LONG_OTHER_THRESHOLD', 0.2))
                long_sum_thr = float(getattr(config, 'STRAT_LONG_SUM_THRESHOLD', 0.3))
                logger.info("【开仓条件（做多）】")
                logger.info(f"  - 前一日信号满足：p_up >= {long_conf_thr} 且 p_no <= {long_other_thr} 且 p_dn <= {long_other_thr} 且 (p_no + p_dn) <= {long_sum_thr}")
                logger.info("  - 当前无持仓")
                logger.info("  - 每次下注10%的总市值（现金+持仓市值），向下取整")
                logger.info("  - 受单标的上限约束（per_symbol_cap）")
                logger.info("  - 受总资金上限约束（alloc_cap）")
                logger.info("")
            if allow_short:
                short_conf_thr = float(getattr(config, 'STRAT_SHORT_CONFIDENCE_THRESHOLD', 0.90))
                short_other_thr = float(getattr(config, 'STRAT_SHORT_OTHER_THRESHOLD', 0.12))
                short_sum_thr = float(getattr(config, 'STRAT_SHORT_SUM_THRESHOLD', 0.2))
                logger.info("【开仓条件（做空）】")
                logger.info(f"  - 前一日信号满足：p_dn >= {short_conf_thr} 且 p_no <= {short_other_thr} 且 p_up <= {short_other_thr} 且 (p_no + p_up) <= {short_sum_thr}")
                logger.info("  - 当前无持仓")
                logger.info("  - 每次下注10%的总市值，向下取整")
                logger.info("  - 受单标的上限约束（per_symbol_cap）")
                logger.info("  - 受总资金上限约束（alloc_cap）")
                logger.info("")
            if allow_long:
                logger.info("【平仓条件（做多）】")
                logger.info("  - 趋势反转：从上涨(1)变成下跌(2)或无趋势(0)")
                logger.info("  - 无趋势：前一日预测为无趋势(0)")
                logger.info("  - 使用前一日预测（pred_prev）判断，避免未来信息泄漏")
                logger.info("")
            if allow_short:
                logger.info("【平仓条件（做空）】")
                logger.info("  - 趋势反转：从下跌(2)变成上涨(1)或无趋势(0)")
                logger.info("  - 无趋势：前一日预测为无趋势(0)")
                logger.info("  - 使用前一日预测（pred_prev）判断，避免未来信息泄漏")
            logger.info("")
            logger.info("【成交价和费用】")
            logger.info("  - 成交价：开盘价（open）")
            logger.info("  - 费用：开/平各一次（fee_bps + slippage_bps）")
            logger.info("  - 做多：买入时支付费用，卖出时支付费用")
            logger.info("  - 做空：卖出时支付费用，平仓买入时支付费用")
            logger.info("")
            logger.info("【资金管理】")
            logger.info(f"  - 初始资金：{getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0)}")
            logger.info("  - 总市值 = 现金 + 持仓市值（做多为正，做空为负）")
            logger.info("  - 如果现金 < 0，停止交易")
            logger.info("=" * 80)
            logger.info("")
            
            # 执行模拟
            trades_df, summary_df, final_cash, stopped_early, daily_positions_df, daily_balances_df, sharpe_ratio = simulate(
                df_all=sim_df_all,
                up_thr=float(getattr(config, 'CLASS_PROB_THRESHOLD_UP_TREND', 0.55)),
                alloc_cap=float(getattr(config, 'STRAT_ALLOC_CAP', 1.0)),
                per_symbol_cap=float(getattr(config, 'STRAT_PER_SYMBOL_CAP', 0.25)),
                fee_bps=float(getattr(config, 'STRAT_FEE_BPS', 2.0)),
                slippage_bps=float(getattr(config, 'STRAT_SLIPPAGE_BPS', 1.0)),
                initial_capital=float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0)),
                start_date=str(getattr(config, 'SIM_START_DATE', '')),
                end_date=str(getattr(config, 'SIM_END_DATE', '')),
            )
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "result", "trading_result")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            trades_path = os.path.join(out_dir, f"trade_sim_oos_{ts}.csv")
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
            trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
            summary_path = os.path.join(out_dir, f"trade_sim_oos_summary_{ts}.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            positions_path = os.path.join(out_dir, f"trade_sim_oos_daily_positions_{ts}.csv")
            daily_positions_df.to_csv(positions_path, index=False, encoding='utf-8-sig')
            balances_path = os.path.join(out_dir, f"trade_sim_oos_daily_balances_{ts}.csv")
            daily_balances_df.to_csv(balances_path, index=False, encoding='utf-8-sig')
            # 绘制总资产图表
            chart_path = os.path.join(out_dir, f"trade_sim_oos_total_assets_{ts}.png")
            initial_capital = float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0))
            self.plot_total_assets(daily_balances_df, chart_path, initial_capital=initial_capital)
            logger.info(f"仿真交易完成：{trades_path}，汇总：{summary_path}")
            if sharpe_ratio is not None:
                total_return = (final_cash - float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0))) / float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0)) * 100
                logger.info(f"总收益率：{total_return:.2f}%")
                logger.info(f"夏普比：{sharpe_ratio:.4f}")
            # 打印使用的模型信息
            model_path = model_bundle.get('model_path', '未知')
            model_name = os.path.basename(model_path) if model_path != '未知' else '未知'
            has_paired = bool(model_bundle.get('has_paired_model', False))
            if has_paired:
                paired_model_path = model_bundle.get('paired_model_path', '未知')
                paired_model_name = os.path.basename(paired_model_path) if paired_model_path != '未知' else '未知'
                logger.info(f"\n使用的主模型: {model_name}")
                logger.info(f"  路径: {model_path}")
                logger.info(f"使用的配对模型: {paired_model_name}")
                logger.info(f"  路径: {paired_model_path}")
                logger.info(f"模型类型: 长多+短空两阶段（合并）")
            else:
                is_long_only = bool(model_bundle.get('long_only', False))
                is_short_only = bool(model_bundle.get('short_only', False))
                if is_long_only:
                    logger.info(f"\n使用的模型: {model_name}")
                    logger.info(f"  路径: {model_path}")
                    logger.info(f"模型类型: 长多两阶段")
                elif is_short_only:
                    logger.info(f"\n使用的模型: {model_name}")
                    logger.info(f"  路径: {model_path}")
                    logger.info(f"模型类型: 短空两阶段")
                else:
                    logger.info(f"\n使用的模型: {model_name}")
                    logger.info(f"  路径: {model_path}")
                    logger.info(f"模型类型: 两阶段（通用涨/跌模型）")
            if 'timestamp' in model_bundle:
                logger.info(f"模型时间戳: {model_bundle.get('timestamp', '未知')}")
            # 返回用于后续可能调试或复用的模型信息
            # 注意：这里不再直接引用局部变量 model_stage2，避免在long_only/short_only场景下未定义
            if has_paired:
                model_info = {
                    'stage1': model_stage1,
                    'paired_stage1': model_bundle.get('paired_model_stage1'),
                    'stage2_up': model_bundle.get('model_stage2_up'),
                    'stage2_down': model_bundle.get('model_stage2_down'),
                }
            elif is_long_only:
                model_info = {
                    'stage1': model_stage1,
                    'stage2_up': model_bundle.get('model_stage2_up'),
                }
            elif is_short_only:
                model_info = {
                    'stage1': model_stage1,
                    'stage2_down': model_bundle.get('model_stage2_down'),
                }
            else:
                model_info = {
                    'stage1': model_stage1,
                    'stage2': model_bundle.get('model_stage2'),
                }
            return {
                'model': model_info,
                'feature_names': feature_names
            }
        else:
            # 单模型预测
            model = model_bundle['model']
            # 加载阈值（单模型）
            thresholds_single = None
            if thresholds_ts and not thresholds_ts.get('two_stage', False):
                thresholds_single = thresholds_ts
            # 评估仿真集
            sim_results = self.evaluate_model(model, X_sim, y_sim, '仿真集', use_probability_threshold=True, use_3_class=use_3_class, pos_weights=pos_w_sim, symbols=symbols_sim)
            # 组装仿真输入表
            # 使用过滤后的数据（与X_sim对齐）
            sim_df_filtered = sim_df_indexed.iloc[:len(X_sim)].copy() if len(X_sim) <= len(sim_df_indexed) else sim_df_indexed.copy()
            if 'date' in sim_df_filtered.columns:
                sim_df_filtered['date'] = pd.to_datetime(sim_df_filtered['date'])
            
            # 确保预测结果长度与过滤后的数据一致
            n_preds = len(sim_results['y_pred'])
            n_data = len(sim_df_filtered)
            if n_preds != n_data:
                logger.warning(f"预测结果长度({n_preds})与数据长度({n_data})不匹配，将截断或填充")
                min_len = min(n_preds, n_data)
                preds = pd.DataFrame({
                    'mdl_pred': sim_results['y_pred'][:min_len]
                })
                proba = sim_results['y_proba'][:min_len]
                sim_df_filtered = sim_df_filtered.iloc[:min_len].copy()
            else:
                preds = pd.DataFrame({
                    'mdl_pred': sim_results['y_pred']
                })
                proba = sim_results['y_proba']
            
            if use_3_class:
                preds['mdl_p_no_trend'] = proba[:, 0]
                preds['mdl_p_up'] = proba[:, 1]
                preds['mdl_p_down'] = proba[:, 2]
            else:
                preds['mdl_p_no_trend'] = proba[:, 0]
                preds['mdl_p_up'] = proba[:, 1]
                preds['mdl_p_down'] = 1.0 - proba[:, 1]  # 仅近似
            
            # 合并到 sim_df_filtered（包含策略所需列）
            base_cols = ['date', 'symbol', 'open']
            strategy_cols = ['MA20']  # 策略可能需要的列
            available_cols = base_cols + [col for col in strategy_cols if col in sim_df_filtered.columns]
            sim_df_all = pd.concat([sim_df_filtered[available_cols].reset_index(drop=True), 
                                   preds.reset_index(drop=True)], axis=1)
            
            # 验证数据完整性
            logger.info(f"仿真输入表准备完成：{len(sim_df_all)} 行")
            logger.info(f"日期范围：{sim_df_all['date'].min()} 到 {sim_df_all['date'].max()}")
            logger.info(f"标的数量：{sim_df_all['symbol'].nunique()}")
            logger.info(f"预测分布：无趋势={(preds['mdl_pred']==0).sum()}, 上涨={(preds['mdl_pred']==1).sum()}, 下跌={(preds['mdl_pred']==2).sum()}")
            
            # 输出完整的交易逻辑
            allow_long = bool(getattr(config, 'STRAT_ALLOW_LONG', True))
            allow_short = bool(getattr(config, 'STRAT_ALLOW_SHORT', True))
            logger.info("\n" + "=" * 80)
            logger.info("完整交易策略说明（单模型）")
            logger.info("=" * 80)
            logger.info(f"做多功能：{'已启用' if allow_long else '已禁用'}")
            logger.info(f"做空功能：{'已启用' if allow_short else '已禁用'}")
            # 止盈 / 止损配置说明
            use_tp = bool(getattr(config, 'STRAT_USE_TAKE_PROFIT', False))
            use_sl = bool(getattr(config, 'STRAT_USE_STOP_LOSS', False))
            tp_pct = float(getattr(config, 'STRAT_TAKE_PROFIT_PCT', 0.10))
            sl_pct = float(getattr(config, 'STRAT_STOP_LOSS_PCT', 0.08))
            logger.info("")
            logger.info("止盈 / 止损配置：")
            logger.info(f"  - 止盈开关：{'开启' if use_tp else '关闭'}（阈值: {tp_pct * 100:.1f}%）")
            logger.info(f"  - 止损开关：{'开启' if use_sl else '关闭'}（阈值: {sl_pct * 100:.1f}%）")
            logger.info("")
            if allow_long:
                long_conf_thr = float(getattr(config, 'STRAT_LONG_CONFIDENCE_THRESHOLD', 0.90))
                long_other_thr = float(getattr(config, 'STRAT_LONG_OTHER_THRESHOLD', 0.2))
                long_sum_thr = float(getattr(config, 'STRAT_LONG_SUM_THRESHOLD', 0.3))
                logger.info("【开仓条件（做多）】")
                logger.info(f"  - 前一日信号满足：p_up >= {long_conf_thr} 且 p_no <= {long_other_thr} 且 p_dn <= {long_other_thr} 且 (p_no + p_dn) <= {long_sum_thr}")
                logger.info("  - 当前无持仓")
                logger.info("  - 每次下注10%的总市值（现金+持仓市值），向下取整")
                logger.info("  - 受单标的上限约束（per_symbol_cap）")
                logger.info("  - 受总资金上限约束（alloc_cap）")
                logger.info("")
            if allow_short:
                short_conf_thr = float(getattr(config, 'STRAT_SHORT_CONFIDENCE_THRESHOLD', 0.90))
                short_other_thr = float(getattr(config, 'STRAT_SHORT_OTHER_THRESHOLD', 0.12))
                short_sum_thr = float(getattr(config, 'STRAT_SHORT_SUM_THRESHOLD', 0.2))
                logger.info("【开仓条件（做空）】")
                logger.info(f"  - 前一日信号满足：p_dn >= {short_conf_thr} 且 p_no <= {short_other_thr} 且 p_up <= {short_other_thr} 且 (p_no + p_up) <= {short_sum_thr}")
                logger.info("  - 当前无持仓")
                logger.info("  - 每次下注10%的总市值，向下取整")
                logger.info("  - 受单标的上限约束（per_symbol_cap）")
                logger.info("  - 受总资金上限约束（alloc_cap）")
                logger.info("")
            if allow_long:
                logger.info("【平仓条件（做多）】")
                logger.info("  - 趋势反转：从上涨(1)变成下跌(2)或无趋势(0)")
                logger.info("  - 无趋势：前一日预测为无趋势(0)")
                logger.info("  - 使用前一日预测（pred_prev）判断，避免未来信息泄漏")
                logger.info("")
            if allow_short:
                logger.info("【平仓条件（做空）】")
                logger.info("  - 趋势反转：从下跌(2)变成上涨(1)或无趋势(0)")
                logger.info("  - 无趋势：前一日预测为无趋势(0)")
                logger.info("  - 使用前一日预测（pred_prev）判断，避免未来信息泄漏")
            logger.info("")
            logger.info("【成交价和费用】")
            logger.info("  - 成交价：开盘价（open）")
            logger.info("  - 费用：开/平各一次（fee_bps + slippage_bps）")
            logger.info("  - 做多：买入时支付费用，卖出时支付费用")
            logger.info("  - 做空：卖出时支付费用，平仓买入时支付费用")
            logger.info("")
            logger.info("【资金管理】")
            logger.info(f"  - 初始资金：{getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0)}")
            logger.info("  - 总市值 = 现金 + 持仓市值（做多为正，做空为负）")
            logger.info("  - 如果现金 < 0，停止交易")
            logger.info("=" * 80)
            logger.info("")
            
            # 执行模拟
            trades_df, summary_df, final_cash, stopped_early, daily_positions_df, daily_balances_df, sharpe_ratio = simulate(
                df_all=sim_df_all,
                up_thr=float(getattr(config, 'CLASS_PROB_THRESHOLD_UP_TREND', 0.55)),
                alloc_cap=float(getattr(config, 'STRAT_ALLOC_CAP', 1.0)),
                per_symbol_cap=float(getattr(config, 'STRAT_PER_SYMBOL_CAP', 0.25)),
                fee_bps=float(getattr(config, 'STRAT_FEE_BPS', 2.0)),
                slippage_bps=float(getattr(config, 'STRAT_SLIPPAGE_BPS', 1.0)),
                initial_capital=float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0)),
                start_date=str(getattr(config, 'SIM_START_DATE', '')),
                end_date=str(getattr(config, 'SIM_END_DATE', '')),
            )
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "result", "trading_result")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            trades_path = os.path.join(out_dir, f"trade_sim_oos_{ts}.csv")
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
            trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
            summary_path = os.path.join(out_dir, f"trade_sim_oos_summary_{ts}.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            positions_path = os.path.join(out_dir, f"trade_sim_oos_daily_positions_{ts}.csv")
            daily_positions_df.to_csv(positions_path, index=False, encoding='utf-8-sig')
            balances_path = os.path.join(out_dir, f"trade_sim_oos_daily_balances_{ts}.csv")
            daily_balances_df.to_csv(balances_path, index=False, encoding='utf-8-sig')
            # 绘制总资产图表
            chart_path = os.path.join(out_dir, f"trade_sim_oos_total_assets_{ts}.png")
            initial_capital = float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0))
            self.plot_total_assets(daily_balances_df, chart_path, initial_capital=initial_capital)
            logger.info(f"仿真交易完成：{trades_path}，汇总：{summary_path}")
            if sharpe_ratio is not None:
                total_return = (final_cash - float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0))) / float(getattr(config, 'STRAT_INITIAL_CAPITAL', 100000.0)) * 100
                logger.info(f"总收益率：{total_return:.2f}%")
                logger.info(f"夏普比：{sharpe_ratio:.4f}")
            # 打印使用的模型信息
            model_path = model_bundle.get('model_path', '未知')
            model_name = os.path.basename(model_path) if model_path != '未知' else '未知'
            logger.info(f"\n使用的模型: {model_name}")
            logger.info(f"  路径: {model_path}")
            logger.info(f"模型类型: 单模型（三分类或二分类）")
            if 'timestamp' in model_bundle:
                logger.info(f"模型时间戳: {model_bundle.get('timestamp', '未知')}")
            return {
                'model': model,
                'feature_names': feature_names
            }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='XGBoost趋势预测模型回测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 回测所有标的（使用config中的日期范围）
  python xgboost_simulation.py
  
  # 回测单个标的（使用config中的日期范围）
  python xgboost_simulation.py --symbol AAPL
  
  # 回测单个标的，指定时间区间
  python xgboost_simulation.py --symbol AAPL --start-date 2024-01-01 --end-date 2024-12-31
  
  # 回测所有标的，指定时间区间
  python xgboost_simulation.py --start-date 2024-01-01 --end-date 2024-12-31
        """
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='要回测的单个标的代码（如 AAPL）。如果不指定，则回测所有标的'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='回测开始日期（格式：YYYY-MM-DD）。如果不指定，则使用config.SIM_START_DATE'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='回测结束日期（格式：YYYY-MM-DD）。如果不指定，则使用config.SIM_END_DATE'
    )
    
    args = parser.parse_args()
    
    trainer = TrendXGBoostTrainer()
    trainer.train(symbol=args.symbol, start_date=args.start_date, end_date=args.end_date)


if __name__ == '__main__':
    main()

