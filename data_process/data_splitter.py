"""
数据划分工具
按时间顺序划分训练集、验证集、测试集
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 确保可导入项目根目录下的 config.py（支持直接运行本文件）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSplitter:
    """数据划分器"""
    
    def __init__(self):
        """初始化数据划分器"""
        # 获取项目根目录（上级目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, config.DATA_DIR)
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # 创建xgb目录下的train、validation、test目录
        self.xgb_dir = os.path.join(self.data_dir, 'xgb')
        self.train_dir = os.path.join(self.xgb_dir, 'train')
        self.val_dir = os.path.join(self.xgb_dir, 'validation')
        self.test_dir = os.path.join(self.xgb_dir, 'test')
        self.test_early_dir = os.path.join(self.xgb_dir, 'test_early')
        self.test_late_dir = os.path.join(self.xgb_dir, 'test_late')
        
        for dir_path, dir_name in [(self.train_dir, 'train'), 
                                   (self.val_dir, 'validation'), 
                                   (self.test_dir, 'test'),
                                   (self.test_early_dir, 'test_early'),
                                   (self.test_late_dir, 'test_late')]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"创建{dir_name}目录: {dir_path}")
            else:
                logger.info(f"{dir_name}目录已存在: {dir_path}")
    
    def split_time_series(self, df, train_ratio=None, val_ratio=None, test_ratio=None):
        """
        按时间顺序和比例划分数据集
        
        Args:
            df: DataFrame，必须包含'date'列，且已按时间排序
            train_ratio: 训练集比例，如果为None则使用config中的值
            val_ratio: 验证集比例，如果为None则使用config中的值
            test_ratio: 测试集比例，如果为None则自动计算
        
        Returns:
            train_df, val_df, test_df
        """
        if train_ratio is None:
            train_ratio = config.TRAIN_RATIO
        if val_ratio is None:
            val_ratio = config.VAL_RATIO
        if test_ratio is None:
            test_ratio = 1 - train_ratio - val_ratio
        
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            logger.warning(f"比例总和不为1，自动调整test_ratio为 {1 - train_ratio - val_ratio}")
            test_ratio = 1 - train_ratio - val_ratio
        
        # 确保按时间排序
        if 'date' not in df.columns:
            raise ValueError("DataFrame必须包含'date'列")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        # 确保按时间严格排序（如果有symbol列，也按symbol排序以保持一致性）
        if 'symbol' in df.columns:
            df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        else:
            df = df.sort_values('date').reset_index(drop=True)
        
        # 验证时间顺序
        date_diff = df['date'].diff()
        if (date_diff < pd.Timedelta(0)).any():
            logger.warning("数据中存在日期倒退，已重新排序")
            if 'symbol' in df.columns:
                df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                df = df.sort_values('date').reset_index(drop=True)
        
        # 移除标签为NaN的行（最后m行无法计算未来指标）
        # 这些行不能用于训练，但可以保留在数据中用于分析
        df_with_label = df[df['label'].notna()].copy() if 'label' in df.columns else df.copy()
        
        # ===== 新逻辑：按 unique_dates 切分（更符合“最新15%日期为测试集”） =====
        use_unique_dates = bool(getattr(config, 'SPLIT_RATIO_BY_UNIQUE_DATES', True))
        if use_unique_dates:
            unique_dates = pd.Series(df_with_label['date'].dt.normalize().unique()).sort_values().to_list()
            m = len(unique_dates)
            if m == 0:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            train_end_date_idx = int(m * train_ratio)
            val_end_date_idx = int(m * (train_ratio + val_ratio))
            # 边界保护
            train_end_date_idx = max(0, min(train_end_date_idx, m))
            val_end_date_idx = max(train_end_date_idx, min(val_end_date_idx, m))

            train_dates = set(unique_dates[:train_end_date_idx])
            val_dates = set(unique_dates[train_end_date_idx:val_end_date_idx])
            test_dates = set(unique_dates[val_end_date_idx:])

            train_df = df_with_label[df_with_label['date'].dt.normalize().isin(train_dates)].copy()
            val_df = df_with_label[df_with_label['date'].dt.normalize().isin(val_dates)].copy()
            test_df = df_with_label[df_with_label['date'].dt.normalize().isin(test_dates)].copy()

            logger.info("按 unique_dates 切分（SPLIT_RATIO_BY_UNIQUE_DATES=True）")
            logger.info(f"  unique_dates 总数: {m}")
            logger.info(f"  train/val/test unique_dates: {len(train_dates)}/{len(val_dates)}/{len(test_dates)}")
        else:
            # ===== 旧逻辑：按样本行数切分，再向后对齐到下一个日期避免边界重叠 =====
            n = len(df_with_label)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            # 按时间顺序划分（使用iloc确保顺序）
            # 确保边界不重叠：找到第一个与训练集最后日期不同的行作为验证集起点
            train_df = df_with_label.iloc[:train_end].copy()
            
            if train_end < n:
                # 找到验证集的起始位置：第一个日期大于训练集最大日期的行
                train_max_date = train_df['date'].max()
                val_start_idx = train_end
                # 跳过所有与训练集最大日期相同的行
                while val_start_idx < n and df_with_label.iloc[val_start_idx]['date'] <= train_max_date:
                    val_start_idx += 1
                
                if val_start_idx < n:
                    # 验证集：从val_start_idx到val_end（同样需要处理边界）
                    val_df_temp = df_with_label.iloc[val_start_idx:val_end].copy()
                    if len(val_df_temp) > 0:
                        val_max_date = val_df_temp['date'].max()
                        test_start_idx = val_end
                        # 跳过所有与验证集最大日期相同的行
                        while test_start_idx < n and df_with_label.iloc[test_start_idx]['date'] <= val_max_date:
                            test_start_idx += 1
                        
                        val_df = val_df_temp
                        test_df = df_with_label.iloc[test_start_idx:].copy() if test_start_idx < n else pd.DataFrame()
                    else:
                        val_df = val_df_temp
                        test_df = df_with_label.iloc[val_end:].copy()
                else:
                    # 如果找不到验证集起点，使用原始切分但记录警告
                    logger.warning("无法找到验证集起点，使用原始切分（可能存在边界重叠）")
                    val_df = df_with_label.iloc[train_end:val_end].copy()
                    test_df = df_with_label.iloc[val_end:].copy()
            else:
                # 训练集已经包含所有数据
                val_df = pd.DataFrame()
                test_df = pd.DataFrame()
        
        # 再次确保每个数据集内部按时间排序
        if len(train_df) > 0:
            if 'symbol' in train_df.columns:
                train_df = train_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                train_df = train_df.sort_values('date').reset_index(drop=True)
        
        if len(val_df) > 0:
            if 'symbol' in val_df.columns:
                val_df = val_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                val_df = val_df.sort_values('date').reset_index(drop=True)
        
        if len(test_df) > 0:
            if 'symbol' in test_df.columns:
                test_df = test_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                test_df = test_df.sort_values('date').reset_index(drop=True)
        
        n = len(df_with_label)
        logger.info(f"数据划分完成:")
        logger.info(f"  总数据量: {len(df)} 行（含NaN标签: {len(df) - n} 行）")
        logger.info(f"  训练集: {len(train_df)} 行 ({(len(train_df)/n*100 if n else 0):.1f}%)")
        logger.info(f"  验证集: {len(val_df)} 行 ({(len(val_df)/n*100 if n else 0):.1f}%)")
        logger.info(f"  测试集: {len(test_df)} 行 ({(len(test_df)/n*100 if n else 0):.1f}%)")
        
        # 显示时间范围
        if len(train_df) > 0:
            logger.info(f"  训练集时间范围: {train_df['date'].min()} 到 {train_df['date'].max()}")
        if len(val_df) > 0:
            logger.info(f"  验证集时间范围: {val_df['date'].min()} 到 {val_df['date'].max()}")
        if len(test_df) > 0:
            logger.info(f"  测试集时间范围: {test_df['date'].min()} 到 {test_df['date'].max()}")
        
        return train_df, val_df, test_df
    
    def split_by_date(self, df, train_end_date=None, val_end_date=None):
        """
        按具体日期划分数据集
        
        Args:
            df: DataFrame，必须包含'date'列
            train_end_date: 训练集结束日期（字符串或datetime），如果为None则使用config中的值
            val_end_date: 验证集结束日期（字符串或datetime），如果为None则使用config中的值
        
        Returns:
            train_df, val_df, test_df
        """
        if train_end_date is None:
            train_end_date = config.TRAIN_END_DATE
        if val_end_date is None:
            val_end_date = config.VAL_END_DATE
        
        # 确保按时间排序
        if 'date' not in df.columns:
            raise ValueError("DataFrame必须包含'date'列")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        # 确保按时间严格排序（如果有symbol列，也按symbol排序以保持一致性）
        if 'symbol' in df.columns:
            df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        else:
            df = df.sort_values('date').reset_index(drop=True)
        
        # 验证时间顺序
        date_diff = df['date'].diff()
        if (date_diff < pd.Timedelta(0)).any():
            logger.warning("数据中存在日期倒退，已重新排序")
            if 'symbol' in df.columns:
                df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                df = df.sort_values('date').reset_index(drop=True)
        
        # 移除标签为NaN的行
        df_with_label = df[df['label'].notna()].copy() if 'label' in df.columns else df.copy()
        
        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)
        
        # 确保边界不重叠：训练集使用 <=，验证集使用 > 且 <=，测试集使用 >
        train_df = df_with_label[df_with_label['date'] <= train_end].copy()
        
        # 如果训练集为空，直接返回空DataFrame
        if len(train_df) == 0:
            logger.warning("训练集为空，请检查train_end_date设置")
            val_df = pd.DataFrame()
            test_df = pd.DataFrame()
            return train_df, val_df, test_df
        
        # 获取训练集的最大日期，确保验证集从下一个日期开始
        train_max_date = train_df['date'].max()
        val_df = df_with_label[(df_with_label['date'] > train_max_date) & 
                               (df_with_label['date'] <= val_end)].copy()
        
        # 如果验证集为空，测试集从val_end之后开始
        if len(val_df) == 0:
            test_df = df_with_label[df_with_label['date'] > val_end].copy()
        else:
            # 获取验证集的最大日期，确保测试集从下一个日期开始
            val_max_date = val_df['date'].max()
            test_df = df_with_label[df_with_label['date'] > val_max_date].copy()
        
        # 确保每个数据集内部按时间排序
        if len(train_df) > 0:
            if 'symbol' in train_df.columns:
                train_df = train_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                train_df = train_df.sort_values('date').reset_index(drop=True)
        
        if len(val_df) > 0:
            if 'symbol' in val_df.columns:
                val_df = val_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                val_df = val_df.sort_values('date').reset_index(drop=True)
        
        if len(test_df) > 0:
            if 'symbol' in test_df.columns:
                test_df = test_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            else:
                test_df = test_df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"按日期划分完成:")
        logger.info(f"  总数据量: {len(df)} 行（含NaN标签: {len(df) - len(df_with_label)} 行）")
        logger.info(f"  训练集: {len(train_df)} 行")
        logger.info(f"  验证集: {len(val_df)} 行")
        logger.info(f"  测试集: {len(test_df)} 行")
        
        if len(train_df) > 0:
            logger.info(f"  训练集时间范围: {train_df['date'].min()} 到 {train_df['date'].max()}")
        if len(val_df) > 0:
            logger.info(f"  验证集时间范围: {val_df['date'].min()} 到 {val_df['date'].max()}")
        if len(test_df) > 0:
            logger.info(f"  测试集时间范围: {test_df['date'].min()} 到 {test_df['date'].max()}")
        
        return train_df, val_df, test_df
    
    def validate_split(self, train_df, val_df, test_df):
        """
        验证数据划分结果
        
        Args:
            train_df, val_df, test_df: 划分后的数据集
        
        Returns:
            dict: 验证结果
        """
        results = {
            'time_order_valid': True,
            'label_distribution': {},
            'warnings': []
        }
        
        # 检查时间顺序（严格小于关系：训练集 < 验证集 < 测试集）
        if len(train_df) > 0 and len(val_df) > 0:
            train_max_date = train_df['date'].max()
            val_min_date = val_df['date'].min()
            if train_max_date >= val_min_date:
                results['time_order_valid'] = False
                results['warnings'].append(
                    f"训练集和验证集时间顺序不正确：训练集最大日期 ({train_max_date}) >= 验证集最小日期 ({val_min_date})"
                )
        
        if len(val_df) > 0 and len(test_df) > 0:
            val_max_date = val_df['date'].max()
            test_min_date = test_df['date'].min()
            if val_max_date >= test_min_date:
                results['time_order_valid'] = False
                results['warnings'].append(
                    f"验证集和测试集时间顺序不正确：验证集最大日期 ({val_max_date}) >= 测试集最小日期 ({test_min_date})"
                )
        
        # 检查标签分布
        if 'label' in train_df.columns:
            train_labels = train_df['label'].value_counts().to_dict()
            val_labels = val_df['label'].value_counts().to_dict() if len(val_df) > 0 else {}
            test_labels = test_df['label'].value_counts().to_dict() if len(test_df) > 0 else {}
            
            results['label_distribution'] = {
                'train': train_labels,
                'val': val_labels,
                'test': test_labels
            }
            
            # 检查类别平衡
            for name, labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
                if len(labels) > 0:
                    total = sum(labels.values())
                    max_ratio = max(labels.values()) / total if total > 0 else 0
                    if max_ratio > 0.8:
                        results['warnings'].append(f"{name}集标签严重不平衡（最大类别占比{max_ratio:.1%}）")
        
        return results
    
    def split_symbol(self, symbol):
        """
        划分单个标的的数据
        
        Args:
            symbol: 标的代码
        """
        logger.info(f"开始划分 {symbol} 的数据...")
        
        # 读取处理后的数据
        processed_file = os.path.join(self.processed_dir, f"{symbol}_processed.csv")
        if not os.path.exists(processed_file):
            logger.warning(f"未找到 {symbol} 的处理后数据文件: {processed_file}")
            return None
        
        try:
            df = pd.read_csv(processed_file)
            logger.info(f"成功读取 {symbol} 的数据，共 {len(df)} 条记录")
        except Exception as e:
            logger.error(f"读取 {symbol} 数据失败: {e}")
            return None
        
        # 根据配置选择划分方式
        if config.USE_DATE_SPLIT:
            train_df, val_df, test_df = self.split_by_date(df)
        else:
            train_df, val_df, test_df = self.split_time_series(df)
        
        # 验证划分结果
        validation = self.validate_split(train_df, val_df, test_df)
        
        if not validation['time_order_valid']:
            logger.error(f"{symbol} 数据划分时间顺序不正确！")
            for warning in validation['warnings']:
                logger.error(f"  {warning}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"{symbol}: {warning}")
        
        # 打印标签分布
        if validation['label_distribution']:
            logger.info(f"{symbol} 标签分布:")
            for split_name, labels in validation['label_distribution'].items():
                if labels:
                    logger.info(f"  {split_name}: {labels}")
        
        # 保存划分后的数据到对应的文件夹
        train_file = os.path.join(self.train_dir, f"{symbol}.csv")
        val_file = os.path.join(self.val_dir, f"{symbol}.csv")
        test_file = os.path.join(self.test_dir, f"{symbol}.csv")
        
        train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
        logger.info(f"{symbol} 训练集已保存到: {train_file}")
        
        if len(val_df) > 0:
            val_df.to_csv(val_file, index=False, encoding='utf-8-sig')
            logger.info(f"{symbol} 验证集已保存到: {val_file}")
        
        test_df.to_csv(test_file, index=False, encoding='utf-8-sig')
        logger.info(f"{symbol} 测试集已保存到: {test_file}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'validation': validation
        }
    
    def merge_all_symbols(self):
        """
        合并所有标的的数据到单个文件
        将train、validation、test文件夹中的所有文件分别合并
        确保按时间顺序排列（先按日期，再按symbol）
        """
        logger.info("开始合并所有标的的数据...")
        
        for split_name, split_dir in [('train', self.train_dir), 
                                      ('validation', self.val_dir), 
                                      ('test', self.test_dir)]:
            if not os.path.exists(split_dir):
                logger.warning(f"{split_name}目录不存在，跳过合并")
                continue
            
            # 获取该文件夹中的所有CSV文件（排除all_symbols.csv）
            csv_files = [f for f in os.listdir(split_dir) 
                        if f.endswith('.csv') and f != 'all_symbols.csv']
            
            if not csv_files:
                logger.warning(f"{split_name}目录中没有CSV文件")
                continue
            
            logger.info(f"合并{split_name}数据，找到 {len(csv_files)} 个文件...")
            
            all_data = []
            for csv_file in csv_files:
                symbol = csv_file.replace('.csv', '')
                file_path = os.path.join(split_dir, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    # 确保数据按时间排序
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date').reset_index(drop=True)
                    # 添加标的代码列（如果不存在）
                    if 'symbol' not in df.columns:
                        df['symbol'] = symbol
                    all_data.append(df)
                    logger.info(f"  加载 {symbol}: {len(df)} 行，时间范围: {df['date'].min()} 到 {df['date'].max()}")
                except Exception as e:
                    logger.warning(f"读取 {file_path} 失败: {e}")
                    continue
            
            if all_data:
                # 合并所有数据
                merged_df = pd.concat(all_data, ignore_index=True)
                # 按日期和symbol排序（确保时间顺序）
                if 'date' in merged_df.columns:
                    merged_df['date'] = pd.to_datetime(merged_df['date'])
                    merged_df = merged_df.sort_values(['date', 'symbol']).reset_index(drop=True)
                    
                    # 验证时间顺序
                    date_diff = merged_df['date'].diff()
                    if (date_diff < pd.Timedelta(0)).any():
                        logger.warning(f"{split_name}数据合并后时间顺序有问题，存在日期倒退的情况")
                    else:
                        logger.info(f"{split_name}数据已按时间顺序排列")
                
                # 保存合并后的文件
                merged_file = os.path.join(split_dir, 'all_symbols.csv')
                merged_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
                logger.info(f"{split_name}数据合并完成，共 {len(merged_df)} 行，已保存到: {merged_file}")
                logger.info(f"  时间范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
            else:
                logger.warning(f"{split_name}没有有效数据可合并")
    
    def split_all_symbols(self):
        """
        划分所有标的的数据
        先合并所有股票数据，再统一按时间划分，确保时间顺序正确
        """
        if not os.path.exists(self.processed_dir):
            logger.error(f"未找到processed目录: {self.processed_dir}")
            return
        
        # 获取所有处理后数据文件
        processed_files = [f for f in os.listdir(self.processed_dir) 
                          if f.endswith('_processed.csv')]
        
        if not processed_files:
            logger.warning(f"processed目录中没有找到数据文件")
            return
        
        symbols = [f.replace('_processed.csv', '') for f in processed_files]
        logger.info(f"找到 {len(symbols)} 个标的需要划分: {symbols}")
        
        # 第一步：合并所有股票的数据
        logger.info("=" * 60)
        logger.info("第一步：合并所有股票的数据...")
        logger.info("=" * 60)
        
        all_data = []
        for symbol in symbols:
            processed_file = os.path.join(self.processed_dir, f"{symbol}_processed.csv")
            if not os.path.exists(processed_file):
                logger.warning(f"未找到 {symbol} 的处理后数据文件: {processed_file}")
                continue
            
            try:
                df = pd.read_csv(processed_file)
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                all_data.append(df)
                logger.info(f"  加载 {symbol}: {len(df)} 行")
            except Exception as e:
                logger.error(f"读取 {symbol} 数据失败: {e}")
                continue
        
        if not all_data:
            logger.error("没有有效数据可划分")
            return
        
        # 合并所有数据
        merged_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"合并完成，共 {len(merged_df)} 行数据")
        
        # 确保按时间排序
        if 'date' not in merged_df.columns:
            logger.error("数据缺少'date'列")
            return
        
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        logger.info(f"合并后时间范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
        
        # 第二步：统一按时间划分
        logger.info("=" * 60)
        logger.info("第二步：统一按时间划分数据...")
        logger.info("=" * 60)
        
        if config.USE_DATE_SPLIT:
            train_df, val_df, test_df = self.split_by_date(merged_df)
        else:
            train_df, val_df, test_df = self.split_time_series(merged_df)
        
        # 验证划分结果
        validation = self.validate_split(train_df, val_df, test_df)
        
        if not validation['time_order_valid']:
            logger.error("数据划分时间顺序不正确！")
            for warning in validation['warnings']:
                logger.error(f"  {warning}")
            return None
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"  {warning}")
        
        # 打印标签分布
        if validation['label_distribution']:
            logger.info("标签分布:")
            for split_name, labels in validation['label_distribution'].items():
                if labels:
                    logger.info(f"  {split_name}: {labels}")
        
        # 第三步：保存划分后的数据
        logger.info("=" * 60)
        logger.info("第三步：保存划分后的数据...")
        logger.info("=" * 60)
        
        # 按股票分组保存（可选，用于分析）
        for symbol in symbols:
            # 训练集
            train_symbol = train_df[train_df['symbol'] == symbol].copy()
            if len(train_symbol) > 0:
                train_file = os.path.join(self.train_dir, f"{symbol}.csv")
                train_symbol.to_csv(train_file, index=False, encoding='utf-8-sig')
            
            # 验证集
            val_symbol = val_df[val_df['symbol'] == symbol].copy()
            if len(val_symbol) > 0:
                val_file = os.path.join(self.val_dir, f"{symbol}.csv")
                val_symbol.to_csv(val_file, index=False, encoding='utf-8-sig')
            
            # 测试集
            test_symbol = test_df[test_df['symbol'] == symbol].copy()
            if len(test_symbol) > 0:
                test_file = os.path.join(self.test_dir, f"{symbol}.csv")
                test_symbol.to_csv(test_file, index=False, encoding='utf-8-sig')
        
        # 保存合并后的all_symbols.csv
        train_file = os.path.join(self.train_dir, 'all_symbols.csv')
        val_file = os.path.join(self.val_dir, 'all_symbols.csv')
        test_file = os.path.join(self.test_dir, 'all_symbols.csv')
        
        train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
        logger.info(f"训练集已保存到: {train_file} ({len(train_df)} 行)")
        
        if len(val_df) > 0:
            val_df.to_csv(val_file, index=False, encoding='utf-8-sig')
            logger.info(f"验证集已保存到: {val_file} ({len(val_df)} 行)")
        
        test_df.to_csv(test_file, index=False, encoding='utf-8-sig')
        logger.info(f"测试集已保存到: {test_file} ({len(test_df)} 行)")
        
        # 第四步：如果启用，将测试集进一步分为test_early和test_late
        if getattr(config, 'SPLIT_TEST_INTO_EARLY_LATE', False) and len(test_df) > 0:
            logger.info("=" * 60)
            logger.info("第四步：将测试集分为test_early和test_late...")
            logger.info("=" * 60)
            
            test_early_end_date = pd.to_datetime(getattr(config, 'TEST_EARLY_END_DATE', '2025-03-31'))
            # test_early起始日期：2024-07-01（VAL_END_DATE之后）
            test_early_start_date = pd.to_datetime('2024-07-01')
            
            # test_early: 2024-07-01 ~ TEST_EARLY_END_DATE
            test_early_df = test_df[(test_df['date'] >= test_early_start_date) & (test_df['date'] <= test_early_end_date)].copy()
            # test_late: TEST_EARLY_END_DATE之后的所有数据
            test_late_df = test_df[test_df['date'] > test_early_end_date].copy()
            
            logger.info(f"test_early时间范围: {test_early_start_date.date()} ~ {test_early_end_date.date()}")
            logger.info(f"test_late时间范围: {test_early_end_date.date()} ~ {test_df['date'].max().date()}")
            logger.info(f"test_early样本数: {len(test_early_df)}")
            logger.info(f"test_late样本数: {len(test_late_df)}")
            
            # 按股票分组保存test_early
            for symbol in symbols:
                test_early_symbol = test_early_df[test_early_df['symbol'] == symbol].copy()
                if len(test_early_symbol) > 0:
                    test_early_file = os.path.join(self.test_early_dir, f"{symbol}.csv")
                    test_early_symbol.to_csv(test_early_file, index=False, encoding='utf-8-sig')
            
            # 按股票分组保存test_late
            for symbol in symbols:
                test_late_symbol = test_late_df[test_late_df['symbol'] == symbol].copy()
                if len(test_late_symbol) > 0:
                    test_late_file = os.path.join(self.test_late_dir, f"{symbol}.csv")
                    test_late_symbol.to_csv(test_late_file, index=False, encoding='utf-8-sig')
            
            # 保存合并后的all_symbols.csv
            test_early_file = os.path.join(self.test_early_dir, 'all_symbols.csv')
            test_late_file = os.path.join(self.test_late_dir, 'all_symbols.csv')
            
            if len(test_early_df) > 0:
                test_early_df.to_csv(test_early_file, index=False, encoding='utf-8-sig')
                logger.info(f"test_early已保存到: {test_early_file} ({len(test_early_df)} 行)")
            
            if len(test_late_df) > 0:
                test_late_df.to_csv(test_late_file, index=False, encoding='utf-8-sig')
                logger.info(f"test_late已保存到: {test_late_file} ({len(test_late_df)} 行)")
        
        logger.info("=" * 60)
        logger.info("数据划分完成！")
        logger.info("=" * 60)
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'validation': validation
        }


def main():
    """主函数"""
    splitter = DataSplitter()
    splitter.split_all_symbols()


if __name__ == '__main__':
    main()

