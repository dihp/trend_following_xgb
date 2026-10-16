"""
趋势标签生成器
从processed数据中生成趋势标签
"""

import os
import sys
# 添加项目根目录到路径，以便导入根目录的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import config
import logging
from datetime import datetime
from scipy import stats
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 控制台输出
    ]
)
logger = logging.getLogger(__name__)

# 添加文件日志处理器（用于保存到文件）
def setup_file_logger(log_dir):
    """设置文件日志处理器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'labeler_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"日志文件已保存到: {log_file}")
    return log_file


class TrendLabeler:
    """趋势标签生成器"""
    
    def __init__(self):
        """初始化标签生成器"""
        # 获取项目根目录（上级目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, config.DATA_DIR)
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # 创建trend_visualizations目录用于保存日志
        self.viz_dir = os.path.join(project_root, 'trend_visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
        
        # 设置文件日志
        self.log_file = setup_file_logger(self.viz_dir)
        
        # 读取详细日志开关
        self.detailed_logging = getattr(config, 'LABEL_DETAILED_LOGGING', True)
        
        # 读取多线程配置
        self.use_multithreading = getattr(config, 'LABEL_USE_MULTITHREADING', True)
        self.max_workers = getattr(config, 'LABEL_MAX_WORKERS', None)  # None表示使用CPU核心数
        
        # 线程锁（用于线程安全的日志和计数）
        self._lock = threading.Lock()
    
    def _log_detail(self, message):
        """记录详细日志（根据开关决定是否记录）"""
        if self.detailed_logging:
            logger.info(message)
    
    def _build_segments_from_labels(self, labels: pd.Series):
        """
        根据 label（可包含{1,-1,0}与NaN）构建连续段序列。
        返回列表：[{'start':i, 'end':j, 'value':v}, ...]，包含无趋势段(value=0)。
        """
        segs = []
        if labels is None or len(labels) == 0:
            return segs
        arr = pd.Series(labels).fillna(0).astype(int).values
        n = len(arr)
        start = 0
        cur = arr[0]
        for i in range(1, n):
            if arr[i] != cur:
                segs.append({'start': start, 'end': i-1, 'value': cur})
                start = i
                cur = arr[i]
        segs.append({'start': start, 'end': n-1, 'value': cur})
        return segs
    
    def _compute_incremental_cutoff_index(self, df: pd.DataFrame) -> int:
        """
        计算增量重算的起始index（左闭），左侧数据冻结不再改写。
        策略：从倒数N段的起点开始，左侧再加BUFFER_DAYS，且至少回看MIN_LOOKBACK_DAYS。
        若无旧label/不满足条件，则返回0（全量）。
        """
        try:
            if not bool(getattr(config, 'INCR_LABELING_ENABLED', False)):
                return 0
            if 'label' not in df.columns:
                return 0
            old_labels = df['label']
            if old_labels.isna().all():
                return 0
            segs = self._build_segments_from_labels(old_labels)
            if len(segs) == 0:
                return 0
            n = len(segs)
            last_n = int(getattr(config, 'INCR_LAST_N_SEGMENTS', 3))
            idx_seg = max(0, n - last_n)  # 第 max(0, k-N) 段
            base_start = int(segs[idx_seg]['start'])
            
            # 日期缓冲与最小回看
            buffer_days = int(getattr(config, 'INCR_BUFFER_DAYS', 7))
            min_lookback_days = int(getattr(config, 'INCR_MIN_LOOKBACK_DAYS', 60))
            
            # 若存在date列，按日期计算；否则退化为按行数
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'])
                base_date = pd.to_datetime(dates.iloc[base_start]) if base_start < len(dates) else dates.min()
                # 左侧缓冲
                cutoff_date = base_date - pd.Timedelta(days=buffer_days)
                # 至少回看
                min_cutoff_date = dates.max() - pd.Timedelta(days=min_lookback_days)
                final_cutoff_date = min(cutoff_date, min_cutoff_date)  # 取更早的一个，扩大回看
                # 对齐到索引
                cutoff_idx = int(np.searchsorted(dates.values, np.datetime64(final_cutoff_date), side='left'))
                cutoff_idx = max(0, min(cutoff_idx, len(df) - 1))
            else:
                cutoff_idx = max(0, base_start - buffer_days)
                # 至少回看 min_lookback_days 行
                cutoff_idx = min(cutoff_idx, max(0, len(df) - min_lookback_days))
            
            # 安全性：留出极小的前置窗口，避免0窗口
            cutoff_idx = max(0, min(cutoff_idx, len(df) - 1))
            if cutoff_idx <= 0:
                return 0
            
            # 日志
            self._log_detail(f"[增量] 启用增量标注：倒数{last_n}段起点={base_start}，缓冲={buffer_days}天，min回看={min_lookback_days}天 -> cutoff_idx={cutoff_idx}")
            return cutoff_idx
        except Exception as e:
            logger.warning(f"[增量] 计算增量起点失败，退化为全量：{e}")
            return 0
    
    def _calculate_r2_fast(self, prices, start_idx, window_len):
        """
        快速计算R²（向量化，避免linregress的开销）
        
        Args:
            prices: 价格数组
            start_idx: 窗口起始索引
            window_len: 窗口长度
        
        Returns:
            float: R²值
        """
        if window_len < 2:
            return 0.0
        
        end_idx = start_idx + window_len
        if end_idx > len(prices):
            return 0.0
        
        window_prices = prices[start_idx:end_idx]
        
        if len(window_prices) < 2:
            return 0.0
        
        # x是时间索引 [0, 1, 2, ..., window_len-1]
        n = window_len
        x_mean = (n - 1) / 2.0  # 0到n-1的均值
        y_mean = np.mean(window_prices)
        
        # 预计算x的统计量（对于固定窗口大小，这些是常数）
        # 对于[0,1,2,...,n-1]，方差是固定的：n*(n+1)/12
        if n == 1:
            var_x = 0.0
        else:
            var_x = (n * n - 1) / 12.0  # 对于[0,1,2,...,n-1]的方差
        
        # 计算协方差和y的方差（向量化）
        x_centered = np.arange(n, dtype=np.float64) - x_mean
        y_centered = window_prices - y_mean
        
        # 使用向量化计算协方差
        cov_xy = np.dot(x_centered, y_centered) / n
        var_y = np.var(window_prices, ddof=0)  # 使用总体方差（与linregress一致）
        
        if var_x == 0 or var_y <= 0:
            return 0.0
        
        # R² = (cov(x,y) / (std(x) * std(y)))^2
        std_x = np.sqrt(var_x)
        std_y = np.sqrt(var_y)
        correlation = cov_xy / (std_x * std_y)
        r_squared = correlation ** 2
        
        return max(0.0, min(1.0, r_squared))  # 确保在[0,1]范围内
    
    class _RollingR2Calculator:
        """滚动R²计算器（O(1)更新）"""
        def __init__(self, prices, window_size):
            self.prices = prices
            self.window_size = window_size
            self.n = window_size
            
            # 预计算x的统计量（对于固定窗口大小）
            self.x_values = np.arange(window_size)
            self.x_mean = np.mean(self.x_values)
            self.x_var = np.var(self.x_values)
            self.x_centered = self.x_values - self.x_mean
            
            # 缓存当前窗口的统计量
            self.current_start = 0
            self.current_y_sum = 0.0
            self.current_y2_sum = 0.0
            self.current_xy_sum = 0.0
            self._initialize_window(0)
        
        def _initialize_window(self, start_idx):
            """初始化窗口统计量"""
            end_idx = start_idx + self.window_size
            window_prices = self.prices[start_idx:end_idx]
            
            self.current_start = start_idx
            self.current_y_sum = np.sum(window_prices)
            self.current_y2_sum = np.sum(window_prices ** 2)
            self.current_xy_sum = np.dot(self.x_centered, window_prices - np.mean(window_prices))
        
        def get_r2(self, start_idx):
            """获取指定起始位置的R²值"""
            if start_idx != self.current_start:
                self._initialize_window(start_idx)
            
            end_idx = start_idx + self.window_size
            window_prices = self.prices[start_idx:end_idx]
            
            if len(window_prices) < 2:
                return 0.0
            
            y_mean = self.current_y_sum / self.n
            y_var = (self.current_y2_sum / self.n) - (y_mean ** 2)
            
            if self.x_var == 0 or y_var <= 0:
                return 0.0
            
            # 使用预计算的x_centered
            y_centered = window_prices - y_mean
            cov_xy = np.dot(self.x_centered, y_centered) / self.n
            
            correlation = cov_xy / (np.sqrt(self.x_var) * np.sqrt(y_var))
            r_squared = correlation ** 2
            
            return max(0.0, min(1.0, r_squared))
    
    def _search_trends_in_gap(self, df, gap_start, gap_end, min_window, max_window,
                              direction_ratio_threshold, r2_threshold, r2_relaxed_threshold,
                              sliding_window_size, use_dynamic_return, dynamic_k, min_return, 
                              close_prices=None, daily_returns=None, log_returns=None):
        """
        在间隙中搜索新趋势（辅助方法）
        
        Args:
            close_prices: 缓存的价格数组（可选，用于优化性能）
            daily_returns: 预计算的日收益率数组（可选，用于优化性能）
            log_returns: 预计算的对数收益率数组（可选，用于优化性能）
        
        Returns:
            list: 找到的趋势段列表
        """
        gap_trend_segments = []
        gap_i = gap_start
        
        while gap_i < gap_end - min_window + 1:
            gap_found_trend = False
            gap_best_segment = None
            gap_best_window_len = 0
            
            # 尝试从最小窗口扩展到最大窗口
            for gap_window_len in range(min_window, min(max_window + 1, gap_end - gap_i + 2)):
                gap_seg_end_idx = gap_i + gap_window_len - 1
                
                if gap_seg_end_idx > gap_end:
                    break
                
                # 获取窗口内的价格数据
                # 使用缓存的价格数据（如果提供）
                if close_prices is not None:
                    gap_prices = close_prices[gap_i:gap_seg_end_idx+1]
                else:
                    gap_prices = df['close'].iloc[gap_i:gap_seg_end_idx+1].values
                
                if len(gap_prices) < 2:
                    continue
                
                # 1. 计算方向一致性（使用预计算的日收益率）
                if daily_returns is not None:
                    gap_window_daily_returns = daily_returns[gap_i:gap_seg_end_idx]
                else:
                    gap_window_daily_returns = np.diff(gap_prices) / gap_prices[:-1]
                
                gap_up_days = np.sum(gap_window_daily_returns > 0)
                gap_down_days = np.sum(gap_window_daily_returns < 0)
                gap_total_days = len(gap_window_daily_returns)
                
                if gap_total_days == 0:
                    continue
                
                gap_up_ratio = gap_up_days / gap_total_days
                gap_down_ratio = gap_down_days / gap_total_days
                gap_direction_consistency = max(gap_up_ratio, gap_down_ratio)
                
                # 提前计算收益率和R²，用于判断是否放宽方向一致性要求（使用预计算的对数收益率）
                if log_returns is not None:
                    gap_window_log_returns = log_returns[gap_i:gap_seg_end_idx]
                else:
                    gap_window_log_returns = np.diff(np.log(gap_prices))
                
                gap_total_log_return = np.sum(gap_window_log_returns)
                gap_abs_total_return = abs(gap_total_log_return)
                
                # 使用快速R²计算方法
                gap_overall_r_squared = self._calculate_r2_fast(close_prices if close_prices is not None else gap_prices, 
                                                                 gap_i, gap_window_len)
                
                # 优化：对于震荡但整体趋势明显的市场，放宽方向一致性要求
                gap_effective_direction_threshold = direction_ratio_threshold
                if gap_overall_r_squared >= 0.8 and gap_abs_total_return >= 0.2:
                    gap_effective_direction_threshold = max(0.55, direction_ratio_threshold * 0.92)
                
                if gap_direction_consistency < gap_effective_direction_threshold:
                    continue
                
                gap_direction = 1 if gap_up_ratio >= gap_down_ratio else -1
                
                # 2. 使用滑动窗口计算R²（使用快速方法）
                gap_sliding_r2_values = []
                gap_sliding_window = min(sliding_window_size, gap_window_len)
                gap_slide_step = max(1, gap_sliding_window // 4)
                
                # 使用快速R²计算方法
                gap_prices_array = close_prices if close_prices is not None else gap_prices
                for gap_slide_start in range(0, gap_window_len - gap_sliding_window + 1, gap_slide_step):
                    gap_slide_abs_start = gap_i + gap_slide_start
                    r_squared = self._calculate_r2_fast(gap_prices_array, gap_slide_abs_start, gap_sliding_window)
                    if r_squared > 0:  # 只添加有效的R²值
                        gap_sliding_r2_values.append(r_squared)
                
                if len(gap_sliding_r2_values) == 0:
                    continue
                
                gap_min_sliding_r2 = min(gap_sliding_r2_values)
                
                # 3. 计算整体窗口的R²（已在前面计算，这里重用）
                # gap_overall_r_squared 已在前面计算
                
                # 4. 计算收益率幅度（已在前面计算，这里重用）
                # gap_total_log_return 和 gap_abs_total_return 已在前面计算
                
                if len(gap_window_log_returns) > 1:
                    gap_sigma_seg = np.std(gap_window_log_returns)
                else:
                    gap_sigma_seg = 0
                
                if use_dynamic_return:
                    gap_A_min = dynamic_k * gap_sigma_seg * np.sqrt(gap_window_len)
                else:
                    gap_A_min = np.log(1 + min_return)
                
                if gap_abs_total_return < gap_A_min:
                    continue
                
                if (gap_direction == 1 and gap_total_log_return < 0) or (gap_direction == -1 and gap_total_log_return > 0):
                    continue
                
                # 5. R²判断（间隙搜索中也使用加速检测）
                gap_r2_passed = False
                
                # 检测趋势是否在加速
                gap_is_accelerating = False
                if gap_window_len >= 20:
                    gap_mid_point = gap_window_len // 2
                    gap_first_half_prices = gap_prices[:gap_mid_point]
                    gap_second_half_prices = gap_prices[gap_mid_point:]
                    
                    if len(gap_first_half_prices) > 1 and len(gap_second_half_prices) > 1:
                        gap_first_half_return = np.sum(np.diff(np.log(gap_first_half_prices)))
                        gap_second_half_return = np.sum(np.diff(np.log(gap_second_half_prices)))
                        
                        if gap_direction == 1:  # 上涨趋势
                            if gap_second_half_return > gap_first_half_return and gap_second_half_return > 0:
                                gap_is_accelerating = True
                        else:  # 下跌趋势
                            if gap_second_half_return < gap_first_half_return and gap_second_half_return < 0:
                                gap_is_accelerating = True
                
                # 根据是否加速调整R²阈值
                gap_effective_r2_threshold = r2_threshold
                if gap_is_accelerating:
                    gap_effective_r2_threshold = max(r2_relaxed_threshold, r2_threshold * 0.7)
                
                if gap_min_sliding_r2 >= gap_effective_r2_threshold:
                    gap_r2_passed = True
                elif gap_overall_r_squared >= gap_effective_r2_threshold:
                    gap_r2_passed = True
                elif gap_overall_r_squared >= r2_relaxed_threshold and gap_direction_consistency >= direction_ratio_threshold:
                    gap_r2_passed = True
                
                if not gap_r2_passed:
                    continue
                
                # 所有条件满足，记录这个趋势段
                gap_found_trend = True
                
                # 找到极值点（使用与初始检测相同的逻辑）
                gap_start_price = gap_prices[0]
                if gap_direction == 1:  # 上涨趋势
                    gap_max_price = gap_prices.max()
                    gap_max_indices = np.where(gap_prices == gap_max_price)[0]
                    gap_max_relative_idx = gap_max_indices[-1]
                    gap_max_idx = gap_i + gap_max_relative_idx
                    
                    # 检查最高点是否就是终点
                    if gap_max_idx == gap_seg_end_idx:
                        # 检查最低点
                        gap_min_price = gap_prices.min()
                        gap_min_indices = np.where(gap_prices == gap_min_price)[0]
                        gap_min_relative_idx = gap_min_indices[0]
                        gap_min_idx = gap_i + gap_min_relative_idx
                        
                        gap_drop_from_start = np.log(gap_start_price) - np.log(gap_min_price)
                        gap_rise_from_start = np.log(gap_max_price) - np.log(gap_start_price)
                        
                        if gap_drop_from_start > gap_rise_from_start and gap_min_relative_idx > 0:
                            gap_seg_end_idx = gap_min_idx
                            gap_window_len = gap_min_idx - gap_i + 1
                            gap_extreme_idx = gap_min_idx
                            gap_extreme_price = gap_min_price
                            gap_direction = -1
                        else:
                            gap_extreme_idx = gap_max_idx
                            gap_extreme_price = gap_max_price
                    else:
                        gap_extreme_idx = gap_max_idx
                        gap_extreme_price = gap_max_price
                else:  # 下跌趋势
                    gap_min_price = gap_prices.min()
                    gap_min_indices = np.where(gap_prices == gap_min_price)[0]
                    gap_extreme_relative_idx = gap_min_indices[-1]
                    gap_extreme_idx = gap_i + gap_extreme_relative_idx
                    gap_extreme_price = gap_min_price
                
                gap_best_segment = {
                    'start_idx': gap_i,
                    'end_idx': gap_seg_end_idx,
                    'direction': gap_direction,
                    'direction_consistency': gap_direction_consistency,
                    'r_squared': gap_overall_r_squared,
                    'min_sliding_r2': gap_min_sliding_r2,
                    'total_return': gap_total_log_return,
                    'sigma_seg': gap_sigma_seg,
                    'A_min': gap_A_min,
                    'window_len': gap_window_len,
                    'extreme_idx': gap_extreme_idx,
                    'extreme_price': gap_extreme_price
                }
                gap_best_window_len = gap_window_len
                
                if gap_window_len >= max_window:
                    break
            
            if gap_found_trend and gap_best_segment is not None:
                # 对间隙中找到的趋势段也进行头部截断
                gap_seg_start = gap_best_segment['start_idx']
                gap_seg_end = gap_best_segment['end_idx']
                gap_seg_direction = gap_best_segment['direction']
                
                # 使用缓存的价格数据（如果提供）
                if close_prices is not None:
                    gap_seg_prices = close_prices[gap_seg_start:gap_seg_end+1]
                else:
                    gap_seg_prices = df['close'].iloc[gap_seg_start:gap_seg_end+1].values
                
                if gap_seg_direction == 1:  # 上涨趋势：找最低点
                    gap_seg_min_price = gap_seg_prices.min()
                    gap_seg_min_indices = np.where(gap_seg_prices == gap_seg_min_price)[0]
                    gap_seg_new_start_relative_idx = gap_seg_min_indices[0]
                    gap_seg_new_start_idx = gap_seg_start + gap_seg_new_start_relative_idx
                    gap_seg_new_start_price = gap_seg_min_price
                else:  # 下跌趋势：找最高点
                    gap_seg_max_price = gap_seg_prices.max()
                    gap_seg_max_indices = np.where(gap_seg_prices == gap_seg_max_price)[0]
                    gap_seg_new_start_relative_idx = gap_seg_max_indices[0]
                    gap_seg_new_start_idx = gap_seg_start + gap_seg_new_start_relative_idx
                    gap_seg_new_start_price = gap_seg_max_price
                
                if gap_seg_new_start_idx > gap_seg_start:
                    gap_best_segment['start_idx'] = gap_seg_new_start_idx
                    gap_best_segment['window_len'] = gap_seg_end - gap_seg_new_start_idx + 1
                
                gap_best_segment['start_extreme_idx'] = gap_seg_new_start_idx
                gap_best_segment['start_extreme_price'] = gap_seg_new_start_price
                
                gap_trend_segments.append(gap_best_segment)
                gap_i = gap_best_segment['extreme_idx'] + 1
                logger.debug(f"在间隙 [{gap_start}-{gap_end}] 中检测到新趋势段: [{gap_best_segment['start_idx']}-{gap_best_segment['end_idx']}]")
            else:
                gap_i += 1
        
        return gap_trend_segments
    
    def generate_trend_label(self, df):
        """
        生成趋势标签（最终版）

        标签：
        +1 = 上涨趋势
        -1 = 下跌趋势
         0 = 无趋势
        """
        try:
            close = df["close"].astype(float).values
            n = len(close)
            labels = np.zeros(n, dtype=float)

            if n < 10:
                return pd.Series([np.nan] * n, index=df.index, name="label")

            # ========= 1. 读取配置 =========
            direction_ratio_threshold = getattr(config, "LABEL_DIRECTION_RATIO_THRESHOLD", 0.6)
            r2_threshold = getattr(config, "LABEL_R2_THRESHOLD", 0.7)
            min_return = getattr(config, "LABEL_MIN_RETURN", 0.05)
            use_dynamic_return = getattr(config, "LABEL_USE_DYNAMIC_RETURN", True)
            dynamic_k = getattr(config, "LABEL_DYNAMIC_RETURN_K", 2.0)
            min_window = getattr(config, "LABEL_MIN_WINDOW", 20)
            max_window = getattr(config, "LABEL_MAX_WINDOW", 150)
            merge_gap = getattr(config, "LABEL_MERGE_GAP", 10)

            # 下跌用更宽松的阈值（A 档：尽量合并成大段）
            dir_ratio_threshold_down = direction_ratio_threshold * 0.8
            r2_threshold_down = r2_threshold * 0.6
            min_return_down = min_return * 0.7
            dynamic_k_down = dynamic_k * 0.7

            log_price = np.log(close)
            log_ret = np.diff(log_price, prepend=log_price[0])

            segments = []

            # ========= 2. 从左到右寻找趋势段 =========
            last_valid_start = n - min_window
            if last_valid_start < 0:
                return pd.Series([np.nan] * n, index=df.index, name="label")

            i = 0
            while i <= last_valid_start:
                best_seg = None
                best_score = 0.0

                max_L = min(max_window, n - i)

                for L in range(min_window, max_L + 1):
                    start = i
                    end = i + L - 1

                    total_lr = log_price[end] - log_price[start]
                    abs_ret = abs(total_lr)
                    if abs_ret <= 0:
                        continue

                    direction = 1 if total_lr > 0 else -1

                    # --- 幅度阈值：上涨 / 下跌 分开 ---
                    if use_dynamic_return:
                        window_sigma = np.std(log_ret[start + 1: end + 1])
                        if window_sigma == 0 or np.isnan(window_sigma):
                            continue
                        dyn_min_up = dynamic_k * window_sigma * np.sqrt(L)
                        dyn_min_down = dynamic_k_down * window_sigma * np.sqrt(L)
                    else:
                        dyn_min_up = 0.0
                        dyn_min_down = 0.0

                    if direction > 0:
                        ret_threshold = max(min_return, dyn_min_up)
                    else:
                        ret_threshold = max(min_return_down, dyn_min_down)

                    if abs_ret < ret_threshold:
                        continue

                    # --- 方向一致性 ---
                    seg_ret = log_ret[start + 1: end + 1]
                    up_days = np.sum(seg_ret > 0)
                    down_days = np.sum(seg_ret < 0)
                    eff_days = up_days + down_days
                    if eff_days == 0:
                        continue

                    if direction > 0:
                        dir_ratio = up_days / eff_days
                        if dir_ratio < direction_ratio_threshold:
                            continue
                    else:
                        dir_ratio = down_days / eff_days
                        if dir_ratio < dir_ratio_threshold_down:
                            continue

                    # --- R² ---
                    r2 = self._calculate_r2_fast(close, start, L)
                    if direction > 0:
                        if r2 < r2_threshold:
                            continue
                    else:
                        if r2 < r2_threshold_down:
                            continue

                    # --- 评分：幅度 * sqrt(L) * R² ---
                    score = abs_ret * np.sqrt(L) * r2
                    if score > best_score:
                        best_score = score
                        best_seg = {
                            "start": start,
                            "end": end,
                            "direction": direction,
                            "length": L,
                            "r2": r2,
                            "total_lr": total_lr,
                            "score": score,
                        }

                if best_seg is None:
                    i += 1
                else:
                    segments.append(best_seg)
                    i = best_seg["end"] + 1

            # ========= 3. 合并同向趋势段 =========
            if not segments:
                labels[:] = 0.0
            else:
                merged = [segments[0]]
                for seg in segments[1:]:
                    last = merged[-1]
                    gap = seg["start"] - last["end"] - 1

                    if seg["direction"] == last["direction"] and gap <= merge_gap:
                        # 同向且间隔小 → 尝试合并
                        new_start = last["start"]
                        new_end = seg["end"]
                        L = new_end - new_start + 1
                        total_lr = log_price[new_end] - log_price[new_start]
                        abs_ret = abs(total_lr)
                        r2 = self._calculate_r2_fast(close, new_start, L)

                        # === 新增：同向“大跳动”强制合并（基于波动率动态检测）===
                        # 若合并后的候选段内存在“同向大跳动”，则：
                        # - 强制合并（不因R²/幅度阈值而拆段）
                        # - 以避免爆发行情把同一趋势切碎
                        big_jump_multiplier = getattr(config, "LABEL_BIG_JUMP_SIGMA_MULTIPLIER", 2.5)
                        same_dir_big_move = False
                        try:
                            # seg_log_ret: (new_start, new_end] 的对数收益
                            seg_log_ret = log_ret[new_start + 1: new_end + 1]
                            if seg_log_ret is not None and len(seg_log_ret) >= 2:
                                seg_s = pd.Series(seg_log_ret)
                                rolling_std = seg_s.rolling(window=20, min_periods=2).std(ddof=0)
                                same_dir_big_move = bool(
                                    (
                                        (seg_s * seg["direction"] > 0)
                                        & (seg_s.abs() >= big_jump_multiplier * rolling_std)
                                    ).any()
                                )
                        except Exception as _e:
                            # 安全降级：检测失败时不触发强制合并，回退到原逻辑
                            same_dir_big_move = False

                        if same_dir_big_move:
                            # 强制合并：忽略R²与abs_ret阈值
                            last["start"] = new_start
                            last["end"] = new_end
                            last["length"] = L
                            last["r2"] = r2
                            last["total_lr"] = total_lr
                            last["score"] = abs_ret * np.sqrt(L) * max(r2, 0)
                        elif seg["direction"] == -1:
                            # 下跌：宽松合并
                            last["start"] = new_start
                            last["end"] = new_end
                            last["length"] = L
                            last["r2"] = r2
                            last["total_lr"] = total_lr
                            last["score"] = abs_ret * np.sqrt(L) * max(r2, 0)
                        else:
                            # 上涨：合并条件稍严格
                            if abs_ret >= min_return and r2 >= (r2_threshold * 0.8):
                                last["start"] = new_start
                                last["end"] = new_end
                                last["length"] = L
                                last["r2"] = r2
                                last["total_lr"] = total_lr
                                last["score"] = abs_ret * np.sqrt(L) * r2
                            else:
                                merged.append(seg)
                    else:
                        merged.append(seg)

                # ========= 4. 极点修正每一段的起止 =========
                refined = []
                for seg in merged:
                    s = seg["start"]
                    e = seg["end"]
                    d = seg["direction"]

                    # 太短的段不修，直接保留
                    if e - s + 1 < min_window:
                        refined.append(seg)
                        continue

                    prices = close[s: e + 1]

                    if d == -1:
                        # 下跌：先找最高点作为真正起点，再从那里往后找最低点
                        local_high_idx = int(np.argmax(prices))
                        new_start = s + local_high_idx
                        if new_start >= e:
                            refined.append(seg)
                            continue

                        prices2 = close[new_start: e + 1]
                        local_low_idx = int(np.argmin(prices2))
                        new_end = new_start + local_low_idx
                    else:
                        # 上涨：先找最低点作为真正起点，再从那里往后找最高点
                        local_low_idx = int(np.argmin(prices))
                        new_start = s + local_low_idx
                        if new_start >= e:
                            refined.append(seg)
                            continue

                        prices2 = close[new_start: e + 1]
                        local_high_idx = int(np.argmax(prices2))
                        new_end = new_start + local_high_idx

                    # 基本合法性检查
                    L_new = new_end - new_start + 1
                    if L_new < min_window:
                        refined.append(seg)
                        continue

                    total_lr_new = log_price[new_end] - log_price[new_start]
                    if d == -1 and total_lr_new >= 0:
                        refined.append(seg)
                        continue
                    if d == 1 and total_lr_new <= 0:
                        refined.append(seg)
                        continue

                    # 更新R²和评分
                    r2_new = self._calculate_r2_fast(close, new_start, L_new)
                    seg["start"] = new_start
                    seg["end"] = new_end
                    seg["length"] = L_new
                    seg["total_lr"] = total_lr_new
                    seg["r2"] = r2_new
                    seg["score"] = abs(total_lr_new) * np.sqrt(L_new) * max(r2_new, 0)

                    refined.append(seg)

                merged = refined

                # ========= 5. 写入标签 =========
                for seg in merged:
                    labels[seg["start"]: seg["end"] + 1] = seg["direction"]

            return pd.Series(labels, index=df.index, name="label")

        except Exception as e:
            logger.warning(f"生成趋势标签失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.Series([np.nan] * len(df), index=df.index, name="label")

    def calculate_trend_position_weight(self, df, labels):
        """
        计算趋势位置权重
        
        对于每个趋势段，根据样本在趋势中的位置分配权重：
        - 前20%：权重0.3（趋势刚开始，识别错了惩罚小）
        - 20%-80%：权重1.0（趋势中段，识别错了惩罚大）
        - 后20%：权重0.7（趋势后段，识别错了惩罚中等）
        
        Args:
            df: DataFrame，包含date列
            labels: Series，趋势标签（0=无趋势, 1=上涨趋势, -1=下跌趋势）
        
        Returns:
            Series: 趋势位置权重（无趋势样本权重为1.0）
        """
        try:
            weights = pd.Series(1.0, index=df.index, name='trend_position_weight')
            
            # 找到所有趋势段（连续的label=1或-1）
            trend_segments = []
            in_trend = False
            trend_start = None
            
            for i in range(len(labels)):
                if pd.isna(labels.iloc[i]):
                    if in_trend:
                        # 趋势段结束
                        trend_segments.append((trend_start, i - 1))
                        in_trend = False
                    continue
                
                if labels.iloc[i] == 1 or labels.iloc[i] == -1:  # 有趋势（上涨或下跌）
                    if not in_trend:
                        # 趋势段开始
                        trend_start = i
                        in_trend = True
                else:  # 无趋势
                    if in_trend:
                        # 趋势段结束
                        trend_segments.append((trend_start, i - 1))
                        in_trend = False
            
            # 处理最后一个趋势段
            if in_trend:
                trend_segments.append((trend_start, len(labels) - 1))
            
            # 从config读取权重配置
            head_weight = getattr(config, 'TREND_POSITION_HEAD_WEIGHT', 0.3)
            middle_weight = getattr(config, 'TREND_POSITION_MIDDLE_WEIGHT', 1.0)
            tail_weight = getattr(config, 'TREND_POSITION_TAIL_WEIGHT', 0.7)
            head_threshold = getattr(config, 'TREND_POSITION_HEAD_THRESHOLD', 0.2)
            tail_threshold = getattr(config, 'TREND_POSITION_TAIL_THRESHOLD', 0.8)
            
            # 为每个趋势段内的样本分配权重
            for start_idx, end_idx in trend_segments:
                trend_len = end_idx - start_idx + 1
                if trend_len <= 0:
                    continue
                
                for pos in range(trend_len):
                    idx = start_idx + pos
                    x = pos / max(trend_len, 1)  # 在趋势段中的相对位置（0-1）
                    
                    if x < head_threshold:
                        # 趋势头部：权重较小（趋势刚开始，识别错了惩罚小）
                        weights.iloc[idx] = head_weight
                    elif x < tail_threshold:
                        # 趋势中段：权重最大（趋势最稳定，识别错了惩罚大）
                        weights.iloc[idx] = middle_weight
                    else:
                        # 趋势尾部：权重中等（趋势后段，识别错了惩罚中等）
                        weights.iloc[idx] = tail_weight
            
            logger.info(f"趋势位置权重计算完成:")
            logger.info(f"  趋势段数量: {len(trend_segments)}")
            logger.info(f"  权重配置: 头部={head_weight}, 中段={middle_weight}, 尾部={tail_weight}")
            logger.info(f"  阈值配置: 头部阈值={head_threshold}, 尾部阈值={tail_threshold}")
            logger.info(f"  权重分布: 头部={((weights == head_weight).sum())}, 中段={((weights == middle_weight).sum())}, 尾部={((weights == tail_weight).sum())}, 无趋势={((weights == 1.0).sum())}")
            
            return weights
        except Exception as e:
            logger.warning(f"计算趋势位置权重失败: {e}")
            return pd.Series(1.0, index=df.index, name='trend_position_weight')
    
    def label_symbol(self, symbol):
        """
        为单个标的生成趋势标签
        
        Args:
            symbol: 标的代码
        """
        logger.info(f"开始为 {symbol} 生成趋势标签...")
        
        # 读取processed数据
        processed_file = os.path.join(self.processed_dir, f"{symbol}_processed.csv")
        if not os.path.exists(processed_file):
            logger.warning(f"未找到 {symbol} 的processed数据文件: {processed_file}")
            return
        
        try:
            df = pd.read_csv(processed_file)
            logger.info(f"成功读取 {symbol} 的数据，共 {len(df)} 条记录")
        except Exception as e:
            logger.error(f"读取 {symbol} 数据失败: {e}")
            return
        
        # 检查必要的列
        if 'close' not in df.columns:
            logger.error(f"{symbol} 数据缺少 'close' 列")
            return
        
        # 生成趋势标签（支持增量）
        logger.info(f"生成 {symbol} 的趋势标签...")
        try:
            cutoff_idx = self._compute_incremental_cutoff_index(df)
            if cutoff_idx <= 0:
                # 全量重算
                df['label'] = self.generate_trend_label(df)
                self._log_detail(f"[增量] 未启用或无需增量：全量重算")
            else:
                # 仅对 cutoff_idx 之后的窗口重算，左侧冻结
                left_index = df.index[:cutoff_idx]
                right_df = df.iloc[cutoff_idx:].copy()
                new_right_labels = self.generate_trend_label(right_df)
                # 如果原先无label列，先填充左侧为0
                if 'label' not in df.columns:
                    df['label'] = 0
                # 合并：左侧保留旧值，右侧覆盖为新值
                df.loc[right_df.index, 'label'] = new_right_labels
                logger.info(f"[增量] 仅重算区间：idx>={cutoff_idx}，左侧冻结不变")
            
            # 统计标签分布
            label_counts = df['label'].value_counts()
            logger.info(f"{symbol} 标签分布: {dict(label_counts)}")
            logger.info(f"{symbol} 标签NaN数量: {df['label'].isna().sum()}")
            
            # 计算趋势位置权重
            logger.info(f"计算 {symbol} 的趋势位置权重...")
            try:
                df['trend_position_weight'] = self.calculate_trend_position_weight(
                    df, df['label']
                )
                logger.info(f"{symbol} 趋势位置权重计算完成")
            except Exception as e:
                logger.warning(f"{symbol} 趋势位置权重计算失败: {e}")
                df['trend_position_weight'] = 1.0
            
        except Exception as e:
            logger.warning(f"{symbol} 趋势标签生成失败: {e}")
            df['label'] = np.nan
            df['trend_position_weight'] = 1.0
        
        # 保存更新后的数据
        df.to_csv(processed_file, index=False, encoding='utf-8-sig')
        logger.info(f"{symbol} 标签生成完成，数据已保存到: {processed_file}")
    
    def label_all_symbols(self):
        """为所有标的生成趋势标签"""
        if not os.path.exists(self.processed_dir):
            logger.error(f"未找到processed目录: {self.processed_dir}")
            return
        
        # 获取所有processed数据文件
        processed_files = [f for f in os.listdir(self.processed_dir) 
                          if f.endswith('_processed.csv')]
        
        if not processed_files:
            logger.warning(f"processed目录中没有找到数据文件")
            return
        
        symbols = [f.replace('_processed.csv', '') for f in processed_files]
        logger.info(f"找到 {len(symbols)} 个标的需要生成标签: {symbols}")
        
        completed_count = 0
        failed_count = 0
        
        # 使用多线程处理
        if self.use_multithreading and len(symbols) > 1:
            logger.info(f"使用多线程处理（最大线程数: {self.max_workers or '自动'})")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_symbol = {executor.submit(self.label_symbol, symbol): symbol for symbol in symbols}
                
                # 处理完成的任务
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        future.result()  # 获取结果，如果有异常会抛出
                        with self._lock:
                            completed_count += 1
                            logger.info(f"✅ [{completed_count}/{len(symbols)}] {symbol} 标签生成完成")
                    except Exception as e:
                        with self._lock:
                            failed_count += 1
                            logger.error(f"❌ 为 {symbol} 生成标签时发生错误: {e}")
                            import traceback
                            traceback.print_exc()
        else:
            # 单线程处理
            if not self.use_multithreading:
                logger.info("多线程已禁用，使用单线程处理")
            for symbol in symbols:
                try:
                    self.label_symbol(symbol)
                    completed_count += 1
                    logger.info(f"✅ [{completed_count}/{len(symbols)}] {symbol} 标签生成完成")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ 为 {symbol} 生成标签时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        logger.info("="*60)
        logger.info(f"所有标的标签生成完成！")
        logger.info(f"  成功: {completed_count} 个")
        logger.info(f"  失败: {failed_count} 个")
        logger.info("="*60)


def main():
    """主函数：打标签+可视化"""
    labeler = TrendLabeler()
    
    # 生成标签
    logger.info("\n" + "="*60)
    logger.info("开始生成趋势标签...")
    logger.info("="*60)
    labeler.label_all_symbols()
    
    # 标签生成完成后，自动执行趋势可视化
    logger.info("\n" + "="*60)
    logger.info("标签生成完成，开始自动执行趋势可视化...")
    logger.info("="*60)
    
    try:
        from data_process.visualize_trends import TrendVisualizer
        visualizer = TrendVisualizer()
        visualizer.visualize_all_symbols()
        logger.info("\n✅ 趋势可视化完成！")
    except Exception as e:
        logger.warning(f"⚠️ 趋势可视化失败（不影响标签生成结果）: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
