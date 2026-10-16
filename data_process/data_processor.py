"""
数据处理器
从unprocessed文件夹读取数据，计算技术指标，保存到processed文件夹
"""

import os
import pandas as pd
import numpy as np
import config
import logging
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# 可选：静默 pandas 的碎片化 PerformanceWarning（不影响结果，仅避免日志刷屏）
try:
    if bool(getattr(config, 'SUPPRESS_PANDAS_PERFORMANCE_WARNINGS', True)):
        from pandas.errors import PerformanceWarning
        warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass

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
    
    log_file = os.path.join(log_dir, f'data_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"日志文件已保存到: {log_file}")
    return log_file


class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        """初始化数据处理器"""
        # 获取项目根目录（上级目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, config.DATA_DIR)
        self.unprocessed_dir = os.path.join(self.data_dir, 'unprocessed')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # 创建trend_visualizations目录用于保存日志
        self.viz_dir = os.path.join(project_root, 'trend_visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
        
        # 设置文件日志
        self.log_file = setup_file_logger(self.viz_dir)
        
        # 创建processed目录
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            logger.info(f"创建processed目录: {self.processed_dir}")
        else:
            logger.info(f"processed目录已存在: {self.processed_dir}")
    
    # ===== 技术指标增强 =====
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """
        计算MACD（EMA版），滞后1日
        返回: macd, macd_signal, macd_hist
        """
        try:
            close = df['close'].astype(float)
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_hist = macd - macd_signal
            return macd.shift(1), macd_signal.shift(1), macd_hist.shift(1)
        except Exception as e:
            logger.warning(f"计算MACD失败: {e}")
            n = len(df)
            nan_s = pd.Series([np.nan] * n)
            return nan_s, nan_s, nan_s
    
    def calculate_bollinger(self, df, period=20, k=2.0):
        """
        计算布林带带宽与带内位置，滞后1日
        带宽: (upper - lower) / middle
        带内位置: (close - middle) / (upper - lower)
        """
        try:
            close = df['close'].astype(float)
            mid = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper = mid + k * std
            lower = mid - k * std
            width = (upper - lower) / (mid.replace(0, np.nan))
            pos = (close - mid) / ((upper - lower).replace(0, np.nan))
            return width.shift(1), pos.shift(1)
        except Exception as e:
            logger.warning(f"计算布林带失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n, name='BBandWidth'), pd.Series([np.nan] * n, name='BBandPos')
    
    def calculate_adx(self, df, period=14):
        """
        计算ADX与正/负方向指标（简化Wilder），滞后1日
        返回: adx, plus_di, minus_di
        """
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            prev_close = close.shift(1)
            up_move = high.diff()
            down_move = (-low.diff())
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            tr1 = (high - low).abs()
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            tr_n = tr.rolling(window=period).sum()
            plus_dm_n = pd.Series(plus_dm, index=df.index).rolling(window=period).sum()
            minus_dm_n = pd.Series(minus_dm, index=df.index).rolling(window=period).sum()
            plus_di = 100.0 * (plus_dm_n / tr_n.replace(0, np.nan))
            minus_di = 100.0 * (minus_dm_n / tr_n.replace(0, np.nan))
            dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
            adx = dx.rolling(window=period).mean()
            return adx.shift(1), plus_di.shift(1), minus_di.shift(1)
        except Exception as e:
            logger.warning(f"计算ADX失败: {e}")
            n = len(df)
            nan_s = pd.Series([np.nan] * n)
            return nan_s, nan_s, nan_s
    
    def calculate_pct_from_ma(self, df, ma_series, name):
        """
        计算价格相对某均线的偏离（close/MA - 1），最终滞后1日
        """
        try:
            pct = (df['close'].astype(float) / ma_series.replace(0, np.nan)) - 1.0
            return pct.shift(1).rename(name)
        except Exception as e:
            logger.warning(f"计算{name}失败: {e}")
            return pd.Series([np.nan] * len(df), name=name)
    
    # ===== 市场外生特征（SPY/QQQ/VIX） =====
    def _load_unprocessed_symbol(self, symbol):
        path = os.path.join(self.unprocessed_dir, f"{symbol}_unprocessed.csv")
        if not os.path.exists(path):
            return None
        try:
            d = pd.read_csv(path)
            if 'date' not in d.columns or 'close' not in d.columns:
                return None
            d['date'] = pd.to_datetime(d['date'])
            d = d.sort_values('date').reset_index(drop=True)
            return d
        except Exception:
            return None
    
    def _compute_mkt_features(self, symbol_prefix, df_src):
        """
        计算市场特征（返回 DataFrame: date + 多列），全部滞后1日
        - Return{3,5,10,20}D
        - Vol{5,10,20}D
        对于VIX:
        - Level（close）
        - Change1D（pct_change）
        """
        try:
            d = df_src.copy()
            out = pd.DataFrame({'date': d['date']})
            if symbol_prefix in ['SPY', 'QQQ']:
                for p in [3, 5, 10, 20]:
                    out[f'MKT_{symbol_prefix}_Return{p}D'] = d['close'].pct_change(periods=p).shift(1)
                for p in [5, 10, 20]:
                    out[f'MKT_{symbol_prefix}_Vol{p}D'] = d['close'].pct_change().rolling(window=p).std().shift(1)
            elif symbol_prefix == 'VIX':
                out['MKT_VIX_Level'] = d['close'].shift(1)
                out['MKT_VIX_Change1D'] = d['close'].pct_change().shift(1)
            return out
        except Exception as e:
            logger.warning(f"计算市场特征失败({symbol_prefix}): {e}")
            return pd.DataFrame(columns=['date'])
        
    def calculate_rsi(self, df, period=14):
        """
        计算RSI (相对强弱指数)
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            period: RSI周期，默认14
            
        Returns:
            Series: RSI值（滞后一日）
        """
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return rsi.shift(1)
        except Exception as e:
            logger.warning(f"计算RSI失败: {e}")
            return pd.Series([np.nan] * len(df), name='RSI')
    
    def calculate_ma(self, df, period, column='close'):
        """
        计算移动平均线
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame
            period: 移动平均周期
            column: 计算移动平均的列名
            
        Returns:
            Series: 移动平均值（滞后一日）
        """
        try:
            ma = df[column].rolling(window=period).mean()
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return ma.shift(1)
        except Exception as e:
            logger.warning(f"计算MA{period}失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'MA{period}')
    
    def calculate_volume_ratio(self, df, period):
        """
        计算成交量均值比
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'volume'列
            period: 计算周期（交易日数）
            
        Returns:
            Series: 成交量均值比（滞后一日）
        """
        try:
            avg_volume = df['volume'].rolling(window=period).mean()
            volume_ratio = df['volume'] / avg_volume
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return volume_ratio.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日成交量均值比失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'VolumeRatio{period}')
    
    def calculate_volume_momentum(self, df, period=10):
        """
        计算成交量动量
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'volume'列
            period: 计算周期
            
        Returns:
            Series: 成交量动量（滞后一日）
        """
        try:
            # pct_change()计算的是相对于N天前的变化，已经滞后了
            # 但为了确保在时间t时只使用t-1及之前的数据，我们再次滞后
            volume_momentum = df['volume'].pct_change(periods=period).shift(1)
            return volume_momentum
        except Exception as e:
            logger.warning(f"计算成交量动量失败: {e}")
            return pd.Series([np.nan] * len(df), name='VolumeMomentum')
    
    def calculate_atr(self, df, period=14):
        """
        计算Average True Range (ATR)
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'high', 'low', 'close'列
            period: ATR周期，默认14
            
        Returns:
            Series: ATR值（滞后一日）
        """
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return atr.shift(1)
        except Exception as e:
            logger.warning(f"计算ATR失败: {e}")
            return pd.Series([np.nan] * len(df), name='ATR')
    
    def calculate_monthly_volatility(self, df, period=20):
        """
        计算月波动率（使用20个交易日，约1个月）
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期，默认20个交易日
            
        Returns:
            Series: 月波动率（年化，滞后一日）
        """
        try:
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=period).std() * np.sqrt(252)  # 年化波动率
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return volatility.shift(1)
        except Exception as e:
            logger.warning(f"计算月波动率失败: {e}")
            return pd.Series([np.nan] * len(df), name='MonthlyVolatility')
    
    def calculate_52week_volatility(self, df):
        """
        计算52周历史波动率（使用252个交易日，约1年）
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            
        Returns:
            Series: 52周波动率（年化，滞后一日）
        """
        try:
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=252).std() * np.sqrt(252)  # 年化波动率
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return volatility.shift(1)
        except Exception as e:
            logger.warning(f"计算52周波动率失败: {e}")
            return pd.Series([np.nan] * len(df), name='Volatility52W')
    
    def calculate_price_position(self, df, period=252):
        """
        计算价格相对位置（当前价格在N天内的位置，0-1之间）
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期，默认252个交易日（约1年）
            
        Returns:
            Series: 价格相对位置（滞后一日）
        """
        try:
            highest = df['close'].rolling(window=period).max()
            lowest = df['close'].rolling(window=period).min()
            price_position = (df['close'] - lowest) / (highest - lowest)
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return price_position.shift(1)
        except Exception as e:
            logger.warning(f"计算价格相对位置失败: {e}")
            return pd.Series([np.nan] * len(df), name='PricePosition')
    
    def calculate_obv(self, df):
        """
        计算OBV (能量潮指标)
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'和'volume'列
            
        Returns:
            Series: OBV值（滞后一日）
        """
        try:
            obv = pd.Series([0.0] * len(df), name='OBV')
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return obv.shift(1)
        except Exception as e:
            logger.warning(f"计算OBV失败: {e}")
            return pd.Series([np.nan] * len(df), name='OBV')
    
    def calculate_daily_change_pct(self, df):
        """
        计算当天涨跌百分比
        注意：pct_change()已经滞后了（计算的是相对于前一天的变化），但为了明确，我们再次滞后一日
        
        Args:
            df: DataFrame，需要包含'close'列
            
        Returns:
            Series: 涨跌百分比（滞后一日）
        """
        try:
            # pct_change()计算的是相对于前一天的变化，已经滞后了
            # 但为了确保在时间t时只使用t-1及之前的数据，我们再次滞后
            return df['close'].pct_change().shift(1) * 100
        except Exception as e:
            logger.warning(f"计算涨跌百分比失败: {e}")
            return pd.Series([np.nan] * len(df), name='DailyChangePct')
    
    # ===== 新增特征计算方法 =====
    def calculate_ret_1d(self, df):
        """计算单日收益（滞后一日）"""
        try:
            close = df['close'].astype(float)
            return close.pct_change(1).shift(1)
        except Exception as e:
            logger.warning(f"计算ret_1d失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_ret_multi(self, df, period):
        """计算多日收益（滞后一日）"""
        try:
            close = df['close'].astype(float)
            return close.pct_change(period).shift(1)
        except Exception as e:
            logger.warning(f"计算ret_{period}d失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_ret_slope(self, df, period):
        """计算收益的线性拟合斜率（标准化斜率，滞后一日）"""
        try:
            close = df['close'].astype(float)
            ret = close.pct_change(1)
            slopes = pd.Series([np.nan] * len(df), index=df.index)
            for i in range(period, len(df)):
                window_ret = ret.iloc[i-period+1:i+1]
                if len(window_ret.dropna()) >= period // 2:
                    x = np.arange(len(window_ret))
                    y = window_ret.values
                    mask = ~np.isnan(y)
                    if mask.sum() >= period // 2:
                        slope = np.polyfit(x[mask], y[mask], 1)[0]
                        slopes.iloc[i] = slope
            return slopes.shift(1)
        except Exception as e:
            logger.warning(f"计算ret_slope_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_down_speed(self, df, period):
        """计算下跌速度（只保留负值，滞后一日）"""
        try:
            close = df['close'].astype(float)
            ret = close.pct_change(period)
            return ret.clip(upper=0).shift(1)  # 只保留负值，上涨时为0
        except Exception as e:
            logger.warning(f"计算down_speed_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_ret_acc(self, df, period):
        """计算收益加速度（滞后一日）"""
        try:
            close = df['close'].astype(float)
            ret_1d = close.pct_change(1)
            ret_period = close.pct_change(period)
            acc = ret_1d - ret_period / period
            return acc.shift(1)
        except Exception as e:
            logger.warning(f"计算ret_acc_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_vol_ratio_5_20(self, df):
        """计算短期vs长期波动率比（滞后一日）"""
        try:
            close = df['close'].astype(float)
            ret_1d = close.pct_change(1)
            std_5 = ret_1d.rolling(window=5).std()
            std_20 = ret_1d.rolling(window=20).std()
            return (std_5 / (std_20 + 1e-8)).shift(1)
        except Exception as e:
            logger.warning(f"计算vol_ratio_5_20失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_big_down_cnt(self, df, period, threshold=-0.02):
        """计算连续大阴线统计（滞后一日）"""
        try:
            close = df['close'].astype(float)
            ret_1d = close.pct_change(1)
            big_down = (ret_1d <= threshold).astype(int)
            return big_down.rolling(window=period).sum().shift(1)
        except Exception as e:
            logger.warning(f"计算big_down_cnt_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_gap_down_rate(self, df):
        """计算跳空低开率（滞后一日）"""
        try:
            open_price = df['open'].astype(float)
            close = df['close'].astype(float)
            prev_close = close.shift(1)
            return (open_price / prev_close - 1).shift(1)
        except Exception as e:
            logger.warning(f"计算gap_down_rate失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_high_lower_streak(self, df, period):
        """计算高点抬低次数（连续N天high_diff_1 < 0的天数，滞后一日）"""
        try:
            high = df['high'].astype(float)
            high_diff_1 = high.diff(1)
            is_lower = (high_diff_1 < 0).astype(int)
            return is_lower.rolling(window=period).sum().shift(1)
        except Exception as e:
            logger.warning(f"计算high_lower_streak_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_low_newlow_streak(self, df, period):
        """计算低点创新低次数（连续N天low_diff_1 < 0的天数，滞后一日）"""
        try:
            low = df['low'].astype(float)
            low_diff_1 = low.diff(1)
            is_newlow = (low_diff_1 < 0).astype(int)
            return is_newlow.rolling(window=period).sum().shift(1)
        except Exception as e:
            logger.warning(f"计算low_newlow_streak_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_pos_in_range(self, df, period=20):
        """计算相对近高点/近低点的位置（滞后一日）"""
        try:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            recent_max = high.rolling(window=period).max()
            recent_min = low.rolling(window=period).min()
            range_size = recent_max - recent_min + 1e-8
            return ((close - recent_min) / range_size).shift(1)
        except Exception as e:
            logger.warning(f"计算pos_in_range_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_vol_ratio_vol(self, df):
        """计算成交量放大比（5日/20日，滞后一日）"""
        try:
            volume = df['volume'].astype(float)
            vol_ma_5 = volume.rolling(window=5).mean()
            vol_ma_20 = volume.rolling(window=20).mean()
            return (vol_ma_5 / (vol_ma_20 + 1e-8)).shift(1)
        except Exception as e:
            logger.warning(f"计算vol_ratio_5_20_vol失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_down_heavy_cnt(self, df, period=10, vol_threshold=1.5):
        """计算下跌放量指示（滞后一日）"""
        try:
            close = df['close'].astype(float)
            volume = df['volume'].astype(float)
            ret_1d = close.pct_change(1)
            vol_ma_5 = volume.rolling(window=5).mean()
            vol_ma_20 = volume.rolling(window=20).mean()
            vol_ratio = vol_ma_5 / (vol_ma_20 + 1e-8)
            down_and_heavy = ((ret_1d < 0) & (vol_ratio > vol_threshold)).astype(int)
            return down_and_heavy.rolling(window=period).sum().shift(1)
        except Exception as e:
            logger.warning(f"计算down_heavy_cnt_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_obv_slope(self, df, period):
        """计算OBV斜率（滞后一日）"""
        try:
            obv = self.calculate_obv(df)
            slopes = pd.Series([np.nan] * len(df), index=df.index)
            for i in range(period, len(df)):
                window_obv = obv.iloc[i-period+1:i+1]
                if len(window_obv.dropna()) >= period // 2:
                    x = np.arange(len(window_obv))
                    y = window_obv.values
                    mask = ~np.isnan(y)
                    if mask.sum() >= period // 2:
                        slope = np.polyfit(x[mask], y[mask], 1)[0]
                        slopes.iloc[i] = slope
            return slopes.shift(1)
        except Exception as e:
            logger.warning(f"计算obv_slope_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_rsi_slope(self, df, period, rsi_period=14):
        """计算RSI斜率（滞后一日）"""
        try:
            rsi = self.calculate_rsi(df, period=rsi_period)
            rsi_prev = rsi.shift(period)
            return (rsi - rsi_prev).shift(1)
        except Exception as e:
            logger.warning(f"计算rsi_slope_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_macd_hist_slope(self, df, period):
        """计算MACD柱子斜率（滞后一日）"""
        try:
            _, _, macd_hist = self.calculate_macd(df)
            macd_hist_prev = macd_hist.shift(period)
            return (macd_hist - macd_hist_prev).shift(1)
        except Exception as e:
            logger.warning(f"计算macd_hist_slope_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_drop_from_max(self, df, period):
        """计算相对近期高点跌幅（滞后一日）"""
        try:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            recent_max = high.rolling(window=period).max()
            return (close / recent_max - 1).shift(1)
        except Exception as e:
            logger.warning(f"计算drop_from_max_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_rebound_from_min(self, df, period):
        """计算相对近期低点反弹幅度（滞后一日）"""
        try:
            close = df['close'].astype(float)
            low = df['low'].astype(float)
            recent_min = low.rolling(window=period).min()
            return (close / recent_min - 1).shift(1)
        except Exception as e:
            logger.warning(f"计算rebound_from_min_{period}失败: {e}")
            return pd.Series([np.nan] * len(df))
    
    def calculate_high_in_period(self, df, period=20):
        """
        计算N个交易日内的最高点
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'high'列
            period: 计算周期，默认20个交易日
            
        Returns:
            Series: N日最高点（滞后一日）
        """
        try:
            high_period = df['high'].rolling(window=period).max()
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return high_period.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日最高点失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'High{period}D')
    
    def calculate_return(self, df, period):
        """
        计算过去N天的收益率
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期（天数）
            
        Returns:
            Series: N日收益率（滞后一日）
        """
        try:
            returns = df['close'].pct_change(periods=period)
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return returns.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日收益率失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'Return{period}D')
    
    def calculate_return_volatility(self, df, period):
        """
        计算过去N天的收益波动率
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期（天数）
            
        Returns:
            Series: N日收益波动率（滞后一日）
        """
        try:
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=period).std()
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return volatility.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日收益波动率失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'ReturnVolatility{period}D')

    def _safe_div(self, numerator, denominator, eps=1e-12):
        """
        安全除法：处理0/NaN/inf
        """
        try:
            num = pd.to_numeric(numerator, errors='coerce')
            den = pd.to_numeric(denominator, errors='coerce')
            out = num / (den.replace(0, np.nan) + eps)
            return out.replace([np.inf, -np.inf], np.nan)
        except Exception:
            # 保底：返回全NaN
            n = len(numerator) if hasattr(numerator, '__len__') else 0
            return pd.Series([np.nan] * n)
    
    def calculate_r2(self, df, period):
        """
        计算过去N天的拟合优度（趋势线稳定性，R²）
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期（天数）
            
        Returns:
            Series: N日R²（滞后一日）
        """
        try:
            from scipy import stats
            r2_values = pd.Series([np.nan] * len(df), name=f'R2_{period}D', index=df.index)
            
            for i in range(period, len(df)):
                # 使用t-1及之前的数据计算R²
                window_data = df['close'].iloc[i-period:i]
                if len(window_data) < 2:
                    continue
                
                x = np.arange(len(window_data))
                y = window_data.values
                
                # 计算线性回归的R²
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                r2 = r_value ** 2
                r2_values.iloc[i] = r2
            
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return r2_values.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日R²失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'R2_{period}D')
    
    def calculate_max_drawdown(self, df, period=5):
        """
        计算过去N天的最大回撤
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期（天数），默认5天
            
        Returns:
            Series: N日最大回撤（滞后一日）
        """
        try:
            max_drawdown = pd.Series([np.nan] * len(df), name=f'MaxDrawdown{period}D', index=df.index)
            
            for i in range(period, len(df)):
                # 使用t-1及之前的数据计算最大回撤
                window_data = df['close'].iloc[i-period:i]
                if len(window_data) < 2:
                    continue
                
                # 计算滚动最高点
                rolling_max = window_data.expanding().max()
                # 计算回撤 = (当前价格 - 最高点) / 最高点
                drawdown = (window_data - rolling_max) / rolling_max
                max_drawdown.iloc[i] = drawdown.min()
            
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return max_drawdown.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日最大回撤失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'MaxDrawdown{period}D')
    
    def calculate_max_rebound(self, df, period=5):
        """
        计算过去N天的最大反弹
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期（天数），默认5天
        
        Returns:
            Series: N日最大反弹（滞后一日）
        """
        try:
            max_rebound = pd.Series([np.nan] * len(df), name=f'MaxRebound{period}D', index=df.index)
            
            for i in range(period, len(df)):
                # 使用t-1及之前的数据计算最大反弹
                window_data = df['close'].iloc[i-period:i]
                if len(window_data) < 2:
                    continue
                
                # 计算滚动最低点
                rolling_min = window_data.expanding().min()
                # 计算反弹 = (当前价格 - 最低点) / 最低点
                rebound = (window_data - rolling_min) / rolling_min
                max_rebound.iloc[i] = rebound.max()
            
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return max_rebound.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日最大反弹失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'MaxRebound{period}D')
    
    def calculate_volume_rolling(self, df, period):
        """
        计算过去N天的交易量（滚动均值）
        注意：滞后一日，确保在时间t时只使用t-1及之前的数据
        
        Args:
            df: DataFrame，需要包含'volume'列
            period: 计算周期（天数）
            
        Returns:
            Series: N日交易量滚动均值（滞后一日）
        """
        try:
            volume_rolling = df['volume'].rolling(window=period).mean()
            # 滞后一日，确保在时间t时只使用t-1及之前的数据
            return volume_rolling.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日交易量滚动均值失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'Volume{period}D')
    
    # ================== 高性价比特征增强 ==================
    def calculate_overnight_return(self, df):
        """
        隔夜收益率：(今日开盘 - 昨日收盘) / 昨日收盘，滞后1日
        """
        try:
            prev_close = df['close'].shift(1)
            ovn = (df['open'] - prev_close) / prev_close
            return ovn.shift(1)
        except Exception as e:
            logger.warning(f"计算隔夜收益率失败: {e}")
            return pd.Series([np.nan] * len(df), name='OvernightReturn')
    
    def calculate_intraday_return(self, df):
        """
        盘中收益率：(收盘 - 开盘) / 开盘，滞后1日
        """
        try:
            intraday = (df['close'] - df['open']) / df['open'].replace(0, np.nan)
            return intraday.shift(1)
        except Exception as e:
            logger.warning(f"计算盘中收益率失败: {e}")
            return pd.Series([np.nan] * len(df), name='IntradayReturn')
    
    def calculate_gap_pct(self, df):
        """
        缺口百分比：(今日开盘 - 昨日收盘) / 昨日收盘，滞后1日
        """
        try:
            prev_close = df['close'].shift(1)
            gap = (df['open'] - prev_close) / prev_close
            return gap.shift(1)
        except Exception as e:
            logger.warning(f"计算缺口百分比失败: {e}")
            return pd.Series([np.nan] * len(df), name='GapPct')
    
    def calculate_high_low_range_pct(self, df):
        """
        当日高低振幅占比：(high - low) / open，滞后1日
        """
        try:
            rng = (df['high'] - df['low']) / df['open'].replace(0, np.nan)
            return rng.shift(1)
        except Exception as e:
            logger.warning(f"计算高低振幅占比失败: {e}")
            return pd.Series([np.nan] * len(df), name='HighLowRangePct')
    
    def calculate_pct_from_rolling_extrema(self, df, period=20):
        """
        相对N日极值的距离：
        - PctFrom{period}DHigh = close / rolling_max(close, period) - 1
        - PctFrom{period}DLow  = close / rolling_min(close, period) - 1
        均滞后1日
        """
        try:
            roll_high = df['close'].rolling(window=period).max()
            roll_low = df['close'].rolling(window=period).min()
            pct_high = (df['close'] / roll_high.replace(0, np.nan)) - 1.0
            pct_low = (df['close'] / roll_low.replace(0, np.nan)) - 1.0
            return pct_high.shift(1), pct_low.shift(1)
        except Exception as e:
            logger.warning(f"计算相对{period}日极值距离失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n, name=f'PctFrom{period}DHigh'), pd.Series([np.nan] * n, name=f'PctFrom{period}DLow')
    
    def calculate_days_since_rolling_high_low(self, df, period=20):
        """
        距离最近N日高/低点的天数：DaysSince{period}DHigh / DaysSince{period}DLow，滞后1日
        取窗口内最近极值到窗口末端的距离（0表示当天即极值）
        """
        try:
            def days_since_high(x):
                idx = np.argmax(x)
                return (len(x) - 1) - idx
            def days_since_low(x):
                idx = np.argmin(x)
                return (len(x) - 1) - idx
            dsh = df['close'].rolling(window=period).apply(days_since_high, raw=True)
            dsl = df['close'].rolling(window=period).apply(days_since_low, raw=True)
            return dsh.shift(1), dsl.shift(1)
        except Exception as e:
            logger.warning(f"计算最近{period}日极点天数失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n, name=f'DaysSince{period}DHigh'), pd.Series([np.nan] * n, name=f'DaysSince{period}DLow')
    
    def calculate_shadow_ratios(self, df):
        """
        蜡烛影线/实体特征（滞后1日）：
        - UpperShadowPct = (high - max(open, close)) / (high - low)
        - LowerShadowPct = (min(open, close) - low) / (high - low)
        - BodyPct        = |close - open| / (high - low)
        - UpperBodyRatio = (high - max(open, close)) / |close - open|
        - LowerBodyRatio = (min(open, close) - low) / |close - open|
        """
        try:
            eps = 1e-12
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            open_ = df['open'].astype(float)
            close = df['close'].astype(float)
            rng = (high - low).replace(0, np.nan)
            body = (close - open_).abs().replace(0, np.nan)
            upper = (high - np.maximum(open_, close))
            lower = (np.minimum(open_, close) - low)
            upper_pct = (upper / (rng + eps)).shift(1)
            lower_pct = (lower / (rng + eps)).shift(1)
            body_pct = (body / (rng + eps)).shift(1)
            upper_body = (upper / (body + eps)).shift(1)
            lower_body = (lower / (body + eps)).shift(1)
            return upper_pct, lower_pct, body_pct, upper_body, lower_body
        except Exception as e:
            logger.warning(f"计算影线/实体特征失败: {e}")
            n = len(df)
            nan_s = pd.Series([np.nan] * n)
            return nan_s, nan_s, nan_s, nan_s, nan_s
    
    def calculate_parkinson_vol(self, df, period=10):
        """
        Parkinson 波动率（基于高低价），滞后1日
        σ_p = sqrt( (1/(4 ln2)) * mean( ln(H/L)^2 ) )
        """
        try:
            hl = (df['high'] / df['low']).replace(0, np.nan)
            term = np.log(hl)**2
            factor = 1.0 / (4.0 * np.log(2.0))
            pv = (factor * term.rolling(window=period).mean()).apply(np.sqrt)
            return pv.shift(1)
        except Exception as e:
            logger.warning(f"计算Parkinson波动率失败: {e}")
            return pd.Series([np.nan] * len(df), name=f'ParkinsonVol{period}D')
    
    def calculate_return_skew_kurt(self, df, period=20):
        """
        收益偏度/峰度（滚动），滞后1日
        """
        try:
            ret = df['close'].pct_change()
            skew = ret.rolling(window=period).skew()
            # 近似峰度：使用rolling.apply计算Fisher Kurtosis
            def kurt_fn(x):
                x = x[~np.isnan(x)]
                if len(x) < 4:
                    return np.nan
                m = np.mean(x)
                s = np.std(x, ddof=0)
                if s == 0:
                    return np.nan
                k = np.mean(((x - m)/s)**4) - 3.0
                return k
            kurt = ret.rolling(window=period).apply(kurt_fn, raw=False)
            return skew.shift(1), kurt.shift(1)
        except Exception as e:
            logger.warning(f"计算收益偏度/峰度失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n, name=f'ReturnSkew{period}D'), pd.Series([np.nan] * n, name=f'ReturnKurt{period}D')
    
    def calculate_ma_slope(self, series):
        """
        MA斜率（简单差分），滞后1日
        """
        try:
            slope = series.diff()
            return slope.shift(1)
        except Exception as e:
            logger.warning(f"计算MA斜率失败: {e}")
            return pd.Series([np.nan] * len(series))
    
    def calculate_rolling_skewness_kurtosis(self, df, period):
        """
        计算滚动偏度和峰度（多个窗口）
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 滚动窗口周期
            
        Returns:
            tuple: (skewness, kurtosis) Series，滞后1日
        """
        try:
            from scipy import stats
            returns = df['close'].pct_change().fillna(0)
            # 滚动偏度
            skew = returns.rolling(window=period, min_periods=3).apply(
                lambda x: stats.skew(x) if len(x.dropna()) >= 3 else np.nan, raw=False
            )
            # 滚动峰度（Fisher Kurtosis）
            def kurt_fn(x):
                x = x[~np.isnan(x)]
                if len(x) < 4:
                    return np.nan
                m = np.mean(x)
                s = np.std(x, ddof=0)
                if s == 0:
                    return np.nan
                k = np.mean(((x - m)/s)**4) - 3.0
                return k
            kurt = returns.rolling(window=period, min_periods=4).apply(kurt_fn, raw=False)
            return skew.shift(1), kurt.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日滚动偏度/峰度失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n), pd.Series([np.nan] * n)
    
    def calculate_rolling_atr_volatility(self, df, period):
        """
        计算滚动ATR和波动率（多个窗口）
        
        Args:
            df: DataFrame，需要包含'high', 'low', 'close'列
            period: 滚动窗口周期
            
        Returns:
            tuple: (atr, volatility) Series，滞后1日
        """
        try:
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period, min_periods=1).mean()
            
            # 波动率（收益率的标准差）
            returns = df['close'].pct_change().fillna(0)
            volatility = returns.rolling(window=period, min_periods=2).std()
            
            return atr.shift(1), volatility.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日滚动ATR/波动率失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n), pd.Series([np.nan] * n)
    
    def calculate_advanced_momentum(self, df, period):
        """
        计算高阶动量指标
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期
            
        Returns:
            tuple: (momentum, roc, momentum_accel) Series，滞后1日
        """
        try:
            close = df['close']
            # 价格动量（当前价格相对于N天前的变化率）
            momentum = close.pct_change(periods=period)
            # ROC (Rate of Change)
            roc = (close - close.shift(period)) / close.shift(period)
            # 价格加速度（动量的变化率）
            momentum_accel = momentum.diff() if period >= 5 else pd.Series([np.nan] * len(df))
            return momentum.shift(1), roc.shift(1), momentum_accel.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日高阶动量指标失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n), pd.Series([np.nan] * n), pd.Series([np.nan] * n)
    
    def calculate_volume_change_features(self, df, period):
        """
        计算成交量变化特征
        
        Args:
            df: DataFrame，需要包含'volume'列
            period: 计算周期
            
        Returns:
            tuple: (volume_change, volume_ratio, volume_momentum) Series，滞后1日
        """
        try:
            volume = df['volume']
            # 成交量变化率
            volume_change = volume.pct_change(periods=period)
            # 成交量移动平均比
            if period >= 5:
                volume_ma = volume.rolling(window=period, min_periods=1).mean()
                volume_ratio = volume / volume_ma
            else:
                volume_ratio = pd.Series([np.nan] * len(df), index=df.index)
            # 成交量动量（使用线性回归斜率）
            def volume_momentum_fn(x):
                if len(x) < 2:
                    return 0
                return np.polyfit(range(len(x)), x, 1)[0]
            volume_momentum = volume.rolling(window=min(period, 5), min_periods=2).apply(
                volume_momentum_fn, raw=False
            )
            return volume_change.shift(1), volume_ratio.shift(1), volume_momentum.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日成交量变化特征失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n), pd.Series([np.nan] * n), pd.Series([np.nan] * n)
    
    def calculate_price_slope(self, df, period):
        """
        计算不同时间窗口的价格斜率
        
        Args:
            df: DataFrame，需要包含'close'列
            period: 计算周期
        
        Returns:
            tuple: (slope, slope_norm) Series，滞后1日
        """
        try:
            close = df['close']
            # 使用线性回归计算价格趋势斜率
            def slope_fn(x):
                if len(x) < 2:
                    return 0
                return np.polyfit(range(len(x)), x, 1)[0]
            slope = close.rolling(window=period, min_periods=2).apply(slope_fn, raw=False)
            # 归一化斜率（斜率相对于价格的比例）
            slope_norm = slope / (close + 1e-8)
            return slope.shift(1), slope_norm.shift(1)
        except Exception as e:
            logger.warning(f"计算{period}日价格斜率失败: {e}")
            n = len(df)
            return pd.Series([np.nan] * n), pd.Series([np.nan] * n)
    
    def _filter_features(self, df, expected_feature_names):
        """
        过滤DataFrame，只保留特征表中的列，并删除不应该存在的特征列
        
        Args:
            df: 要过滤的DataFrame
            expected_feature_names: 期望的特征列表
            
        Returns:
            过滤后的DataFrame
        """
        # 定义不应该存在的特征列（会导致未来信息泄露或不应作为特征的列）
        # 注意：date和symbol是必需的元数据，应该保留
        exclude_columns = []
        
        # 1. 标签列（不应作为特征）
        exclude_columns.extend(['label', 'label_old', 'processed_label', 'trend_position_weight'])
        
        # 2. 未来信息泄露列（这些特征基于未来数据计算，与标签高度相关）
        exclude_columns.extend(['NormalizedReturn', 'Slope', 'R2', 'TNR'])
        
        # 3. 从config读取额外排除列（用于AB实验/快速禁用一组新特征）
        try:
            extra_excludes = getattr(config, 'EXTRA_EXCLUDE_COLUMNS', []) or []
            if isinstance(extra_excludes, (list, tuple, set)):
                exclude_columns.extend(list(extra_excludes))
        except Exception:
            pass
        
        # 删除不应该存在的列
        columns_to_drop_exclude = [col for col in df.columns if col in exclude_columns]
        if columns_to_drop_exclude:
            logger.info(f"删除不应该存在的特征列 ({len(columns_to_drop_exclude)} 个): {columns_to_drop_exclude}")
            df = df.drop(columns=columns_to_drop_exclude)
        
        # 如果提供了expected_feature_names，则进一步过滤
        if not expected_feature_names:
            return df
        
        # 基础列（必须保留）
        base_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        # 保留基础列和特征表中的列
        columns_to_keep = [col for col in base_columns if col in df.columns]
        columns_to_keep.extend([col for col in expected_feature_names if col in df.columns])
        # 删除不在特征表中的列
        columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
        if columns_to_drop:
            logger.info(f"删除不在特征表中的列 ({len(columns_to_drop)} 个): {columns_to_drop[:10]}{'...' if len(columns_to_drop) > 10 else ''}")
        return df[columns_to_keep]
    
    def process_symbol(self, symbol):
        """
        处理单个标的的数据
        
        Args:
            symbol: 标的代码
        """
        logger.info(f"开始处理 {symbol} 的数据...")
        
        # 读取未处理数据
        unprocessed_file = os.path.join(self.unprocessed_dir, f"{symbol}_unprocessed.csv")
        if not os.path.exists(unprocessed_file):
            logger.warning(f"未找到 {symbol} 的未处理数据文件: {unprocessed_file}")
            return
        
        try:
            df = pd.read_csv(unprocessed_file)
            logger.info(f"成功读取 {symbol} 的数据，共 {len(df)} 条记录")
        except Exception as e:
            logger.error(f"读取 {symbol} 数据失败: {e}")
            return
        
        # 检查必要的列
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"{symbol} 数据缺少必要的列: {missing_columns}")
            return
        
        # 确保日期列为datetime类型
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 创建结果DataFrame，先复制原始数据
        result_df = df.copy()
        
        # 标记无法计算的指标
        unavailable_indicators = []
        
        # 提前进行增量判断：若无新增列且无新日期，则直接跳过计算
        processed_file = os.path.join(self.processed_dir, f"{symbol}_processed.csv")
        incr_enabled = getattr(config, 'INCR_DATA_PROCESSING_ENABLED', True)
        # 从config读取期望的特征列表
        expected_feature_names = getattr(config, 'EXPECTED_FEATURE_NAMES', [])
        if not expected_feature_names:
            logger.warning(f"config.EXPECTED_FEATURE_NAMES 为空，将计算所有特征")
        try:
            if os.path.exists(processed_file):
                old_df = pd.read_csv(processed_file)
                if 'date' in old_df.columns:
                    old_df['date'] = pd.to_datetime(old_df['date'])
                    latest_old_date = old_df['date'].max()
                    latest_new_date = result_df['date'].max()
                    has_new_date = bool(pd.to_datetime(latest_new_date) > pd.to_datetime(latest_old_date))
                else:
                    has_new_date = True
                # 需要新增的列
                existing_cols = set(old_df.columns)
                new_cols_needed = [c for c in expected_feature_names if c not in existing_cols]
                # 如果启用了增量处理，且无新增列且无新日期，则跳过计算
                if incr_enabled and (not has_new_date) and (len(new_cols_needed) == 0):
                    logger.info(f"{symbol} 无新增列且无新日期，跳过计算与保存（增量处理已启用）")
                    return
        except Exception as e:
            if incr_enabled:
                logger.warning(f"{symbol} 增量判断失败，继续全量计算：{e}")
            else:
                logger.info(f"{symbol} 增量处理已禁用，执行全量计算")
        
        # 计算各种指标（只计算特征表中的特征）
        logger.info(f"计算 {symbol} 的技术指标...")
        if expected_feature_names:
            logger.info(f"  只计算特征表中的 {len(expected_feature_names)} 个特征")
        
        # 1. RSI
        
        
        try:
            result_df['RSI'] = self.calculate_rsi(df, period=14)
        except Exception as e:
            logger.warning(f"{symbol} RSI计算失败: {e}")
            unavailable_indicators.append('RSI')
            result_df['RSI'] = np.nan
        
        # 2. MA5
        try:
            result_df['MA5'] = self.calculate_ma(df, period=5, column='close')
        except Exception as e:
            logger.warning(f"{symbol} MA5计算失败: {e}")
            unavailable_indicators.append('MA5')
            result_df['MA5'] = np.nan
        
        # 3. MA20
        try:
            result_df['MA20'] = self.calculate_ma(df, period=20, column='close')
        except Exception as e:
            logger.warning(f"{symbol} MA20计算失败: {e}")
            unavailable_indicators.append('MA20')
            result_df['MA20'] = np.nan
        # 3b. MA50/MA100
        try:
            result_df['MA50'] = self.calculate_ma(df, period=50, column='close')
        except Exception as e:
            logger.warning(f"{symbol} MA50计算失败: {e}")
            unavailable_indicators.append('MA50')
            result_df['MA50'] = np.nan
        try:
            result_df['MA100'] = self.calculate_ma(df, period=100, column='close')
        except Exception as e:
            logger.warning(f"{symbol} MA100计算失败: {e}")
            unavailable_indicators.append('MA100')
            result_df['MA100'] = np.nan
        
        # 4. 5个交易日成交量均值比
        try:
            result_df['VolumeRatio5D'] = self.calculate_volume_ratio(df, period=5)
        except Exception as e:
            logger.warning(f"{symbol} 5日成交量均值比计算失败: {e}")
            unavailable_indicators.append('VolumeRatio5D')
            result_df['VolumeRatio5D'] = np.nan
        
        # 6. 10交易日成交量均值比
        try:
            result_df['VolumeRatio10D'] = self.calculate_volume_ratio(df, period=10)
        except Exception as e:
            logger.warning(f"{symbol} 10日成交量均值比计算失败: {e}")
            unavailable_indicators.append('VolumeRatio10D')
            result_df['VolumeRatio10D'] = np.nan
        
        # 7. 成交量动量
        try:
            result_df['VolumeMomentum'] = self.calculate_volume_momentum(df, period=10)
        except Exception as e:
            logger.warning(f"{symbol} 成交量动量计算失败: {e}")
            unavailable_indicators.append('VolumeMomentum')
            result_df['VolumeMomentum'] = np.nan
        
        # 8. ATR
        try:
            result_df['ATR'] = self.calculate_atr(df, period=14)
        except Exception as e:
            logger.warning(f"{symbol} ATR计算失败: {e}")
            unavailable_indicators.append('ATR')
            result_df['ATR'] = np.nan
        # 8b. ADX/DI
        try:
            adx, pdi, mdi = self.calculate_adx(df, period=14)
            result_df['ADX14'] = adx
            result_df['PlusDI14'] = pdi
            result_df['MinusDI14'] = mdi
        except Exception as e:
            logger.warning(f"{symbol} ADX/DI计算失败: {e}")
            unavailable_indicators.extend(['ADX14','PlusDI14','MinusDI14'])
            result_df['ADX14'] = np.nan
            result_df['PlusDI14'] = np.nan
            result_df['MinusDI14'] = np.nan
        
        # 9. 月波动率
        try:
            result_df['MonthlyVolatility'] = self.calculate_monthly_volatility(df, period=20)
        except Exception as e:
            logger.warning(f"{symbol} 月波动率计算失败: {e}")
            unavailable_indicators.append('MonthlyVolatility')
            result_df['MonthlyVolatility'] = np.nan
        
        # 10. 52周历史波动率
        try:
            result_df['Volatility52W'] = self.calculate_52week_volatility(df)
        except Exception as e:
            logger.warning(f"{symbol} 52周波动率计算失败: {e}")
            unavailable_indicators.append('Volatility52W')
            result_df['Volatility52W'] = np.nan
        
        # 11. 价格相对位置
        try:
            result_df['PricePosition'] = self.calculate_price_position(df, period=252)
        except Exception as e:
            logger.warning(f"{symbol} 价格相对位置计算失败: {e}")
            unavailable_indicators.append('PricePosition')
            result_df['PricePosition'] = np.nan
        
        # 12. OBV
        try:
            result_df['OBV'] = self.calculate_obv(df)
        except Exception as e:
            logger.warning(f"{symbol} OBV计算失败: {e}")
            unavailable_indicators.append('OBV')
            result_df['OBV'] = np.nan
        
        # 12b. MACD
        try:
            macd, macds, macdh = self.calculate_macd(df, fast=12, slow=26, signal=9)
            result_df['MACD'] = macd
            result_df['MACDSignal'] = macds
            result_df['MACDHist'] = macdh
        except Exception as e:
            logger.warning(f"{symbol} MACD计算失败: {e}")
            unavailable_indicators.extend(['MACD','MACDSignal','MACDHist'])
            result_df['MACD'] = np.nan
            result_df['MACDSignal'] = np.nan
            result_df['MACDHist'] = np.nan
        
        # 12c. 布林带
        try:
            bbw, bbpos = self.calculate_bollinger(df, period=20, k=2.0)
            result_df['BBandWidth20'] = bbw
            result_df['BBandPos20'] = bbpos
        except Exception as e:
            logger.warning(f"{symbol} 布林带计算失败: {e}")
            unavailable_indicators.extend(['BBandWidth20','BBandPos20'])
            result_df['BBandWidth20'] = np.nan
            result_df['BBandPos20'] = np.nan
        
        # 13. 当天涨跌百分比
        try:
            result_df['DailyChangePct'] = self.calculate_daily_change_pct(df)
        except Exception as e:
            logger.warning(f"{symbol} 涨跌百分比计算失败: {e}")
            unavailable_indicators.append('DailyChangePct')
            result_df['DailyChangePct'] = np.nan
        
        # 14. 20个交易日内的高点（可以根据需要调整周期）
        try:
            result_df['High20D'] = self.calculate_high_in_period(df, period=20)
        except Exception as e:
            logger.warning(f"{symbol} 20日最高点计算失败: {e}")
            unavailable_indicators.append('High20D')
            result_df['High20D'] = np.nan
        
        # 15. 过去3, 5, 10, 30天的收益率
        for period in [3, 5, 10, 30]:
            try:
                result_df[f'Return{period}D'] = self.calculate_return(df, period=period)
            except Exception as e:
                logger.warning(f"{symbol} {period}日收益率计算失败: {e}")
                unavailable_indicators.append(f'Return{period}D')
                result_df[f'Return{period}D'] = np.nan
        
        # 16. 过去5, 10, 30天的收益波动率
        for period in [5, 10, 30]:
            try:
                result_df[f'ReturnVolatility{period}D'] = self.calculate_return_volatility(df, period=period)
            except Exception as e:
                logger.warning(f"{symbol} {period}日收益波动率计算失败: {e}")
                unavailable_indicators.append(f'ReturnVolatility{period}D')
                result_df[f'ReturnVolatility{period}D'] = np.nan

        # 16b. 趋势强度：Return{t}D / (ReturnVolatility{t}D * sqrt(t))
        # 说明：ReturnVolatility{t}D 是“日收益”的滚动std；区间t的波动规模近似为 std * sqrt(t)
        for period in [5, 10, 30]:
            ts_col = f'TrendStrength_RV{period}D'
            if expected_feature_names and (ts_col not in expected_feature_names):
                continue
            try:
                ret_col = f'Return{period}D'
                vol_col = f'ReturnVolatility{period}D'
                if (ret_col in result_df.columns) and (vol_col in result_df.columns):
                    denom = result_df[vol_col].astype(float) * np.sqrt(float(period))
                    result_df[ts_col] = self._safe_div(result_df[ret_col], denom)
                else:
                    result_df[ts_col] = np.nan
            except Exception as e:
                logger.warning(f"{symbol} {ts_col}计算失败: {e}")
                unavailable_indicators.append(ts_col)
                result_df[ts_col] = np.nan
        
        # 17. 过去5, 10天的拟合优度（趋势线稳定性，R²）
        for period in [5, 10]:
            try:
                result_df[f'R2_{period}D'] = self.calculate_r2(df, period=period)
            except Exception as e:
                logger.warning(f"{symbol} {period}日R²计算失败: {e}")
                unavailable_indicators.append(f'R2_{period}D')
                result_df[f'R2_{period}D'] = np.nan
        
        # 18. max_drawdown_5d, max_rebound_5d（最近走势波动范围，5天）
        try:
            result_df['MaxDrawdown5D'] = self.calculate_max_drawdown(df, period=5)
        except Exception as e:
            logger.warning(f"{symbol} 5日最大回撤计算失败: {e}")
            unavailable_indicators.append('MaxDrawdown5D')
            result_df['MaxDrawdown5D'] = np.nan
        
        try:
            result_df['MaxRebound5D'] = self.calculate_max_rebound(df, period=5)
        except Exception as e:
            logger.warning(f"{symbol} 5日最大反弹计算失败: {e}")
            unavailable_indicators.append('MaxRebound5D')
            result_df['MaxRebound5D'] = np.nan
        
        # 19. 过去3, 5, 10, 30天的交易量（滚动均值）
        for period in [3, 5, 10, 30]:
            try:
                result_df[f'Volume{period}D'] = self.calculate_volume_rolling(df, period=period)
            except Exception as e:
                logger.warning(f"{symbol} {period}日交易量滚动均值计算失败: {e}")
                unavailable_indicators.append(f'Volume{period}D')
                result_df[f'Volume{period}D'] = np.nan
        
        # 20+. 增强特征：隔夜/盘中/缺口/振幅
        try:
            result_df['OvernightReturn'] = self.calculate_overnight_return(df)
        except Exception as e:
            logger.warning(f"{symbol} OvernightReturn计算失败: {e}")
            unavailable_indicators.append('OvernightReturn')
            result_df['OvernightReturn'] = np.nan
        try:
            result_df['IntradayReturn'] = self.calculate_intraday_return(df)
        except Exception as e:
            logger.warning(f"{symbol} IntradayReturn计算失败: {e}")
            unavailable_indicators.append('IntradayReturn')
            result_df['IntradayReturn'] = np.nan
        try:
            result_df['GapPct'] = self.calculate_gap_pct(df)
        except Exception as e:
            logger.warning(f"{symbol} GapPct计算失败: {e}")
            unavailable_indicators.append('GapPct')
            result_df['GapPct'] = np.nan
        try:
            result_df['HighLowRangePct'] = self.calculate_high_low_range_pct(df)
        except Exception as e:
            logger.warning(f"{symbol} HighLowRangePct计算失败: {e}")
            unavailable_indicators.append('HighLowRangePct')
            result_df['HighLowRangePct'] = np.nan
        
        # 影线/实体
        try:
            usp, lsp, bpct, ubr, lbr = self.calculate_shadow_ratios(df)
            result_df['UpperShadowPct'] = usp
            result_df['LowerShadowPct'] = lsp
            result_df['BodyPct'] = bpct
            result_df['UpperBodyRatio'] = ubr
            result_df['LowerBodyRatio'] = lbr
        except Exception as e:
            logger.warning(f"{symbol} 影线/实体特征计算失败: {e}")
            unavailable_indicators.extend(['UpperShadowPct','LowerShadowPct','BodyPct','UpperBodyRatio','LowerBodyRatio'])
            result_df['UpperShadowPct'] = np.nan
            result_df['LowerShadowPct'] = np.nan
            result_df['BodyPct'] = np.nan
            result_df['UpperBodyRatio'] = np.nan
            result_df['LowerBodyRatio'] = np.nan
        
        # 与20日极值的距离/天数
        try:
            ph, pl = self.calculate_pct_from_rolling_extrema(df, period=20)
            result_df['PctFrom20DHigh'] = ph
            result_df['PctFrom20DLow'] = pl
        except Exception as e:
            logger.warning(f"{symbol} 相对20日极值距离计算失败: {e}")
            unavailable_indicators.extend(['PctFrom20DHigh','PctFrom20DLow'])
            result_df['PctFrom20DHigh'] = np.nan
            result_df['PctFrom20DLow'] = np.nan
        try:
            dsh, dsl = self.calculate_days_since_rolling_high_low(df, period=20)
            result_df['DaysSince20DHigh'] = dsh
            result_df['DaysSince20DLow'] = dsl
        except Exception as e:
            logger.warning(f"{symbol} 最近20日极点天数计算失败: {e}")
            unavailable_indicators.extend(['DaysSince20DHigh','DaysSince20DLow'])
            result_df['DaysSince20DHigh'] = np.nan
            result_df['DaysSince20DLow'] = np.nan
        # 60日版本
        try:
            ph60, pl60 = self.calculate_pct_from_rolling_extrema(df, period=60)
            result_df['PctFrom60DHigh'] = ph60
            result_df['PctFrom60DLow'] = pl60
        except Exception as e:
            logger.warning(f"{symbol} 相对60日极值距离计算失败: {e}")
            unavailable_indicators.extend(['PctFrom60DHigh','PctFrom60DLow'])
            result_df['PctFrom60DHigh'] = np.nan
            result_df['PctFrom60DLow'] = np.nan
        try:
            dsh60, dsl60 = self.calculate_days_since_rolling_high_low(df, period=60)
            result_df['DaysSince60DHigh'] = dsh60
            result_df['DaysSince60DLow'] = dsl60
        except Exception as e:
            logger.warning(f"{symbol} 最近60日极点天数计算失败: {e}")
            unavailable_indicators.extend(['DaysSince60DHigh','DaysSince60DLow'])
            result_df['DaysSince60DHigh'] = np.nan
            result_df['DaysSince60DLow'] = np.nan
        
        # Parkinson波动率
        for period in [5, 10]:
            try:
                result_df[f'ParkinsonVol{period}D'] = self.calculate_parkinson_vol(df, period=period)
            except Exception as e:
                logger.warning(f"{symbol} ParkinsonVol{period}D计算失败: {e}")
                unavailable_indicators.append(f'ParkinsonVol{period}D')
                result_df[f'ParkinsonVol{period}D'] = np.nan

        # Parkinson 版本趋势强度：Return{t}D / (ParkinsonVol{t}D * sqrt(t))
        # 注：ParkinsonVol{t}D近似“日频波动率估计”（基于高低价），区间t波动规模近似 sigma_daily * sqrt(t)
        for period in [5, 10]:
            ts_col = f'TrendStrength_PV{period}D'
            if expected_feature_names and (ts_col not in expected_feature_names):
                continue
            try:
                ret_col = f'Return{period}D'
                pv_col = f'ParkinsonVol{period}D'
                if (ret_col in result_df.columns) and (pv_col in result_df.columns):
                    denom = result_df[pv_col].astype(float) * np.sqrt(float(period))
                    result_df[ts_col] = self._safe_div(result_df[ret_col], denom)
                else:
                    result_df[ts_col] = np.nan
            except Exception as e:
                logger.warning(f"{symbol} {ts_col}计算失败: {e}")
                unavailable_indicators.append(ts_col)
                result_df[ts_col] = np.nan
        
        # 收益偏度/峰度
        for period in [10, 20]:
            try:
                skew, kurt = self.calculate_return_skew_kurt(df, period=period)
                result_df[f'ReturnSkew{period}D'] = skew
                result_df[f'ReturnKurt{period}D'] = kurt
            except Exception as e:
                logger.warning(f"{symbol} ReturnSkew/Kurt{period}D计算失败: {e}")
                unavailable_indicators.extend([f'ReturnSkew{period}D', f'ReturnKurt{period}D'])
                result_df[f'ReturnSkew{period}D'] = np.nan
                result_df[f'ReturnKurt{period}D'] = np.nan
        
        # 多尺度加速度/斜率（基于已计算的滞后特征）
        try:
            if 'Return10D' in result_df.columns and 'Return5D' in result_df.columns:
                result_df['ReturnAccel10_5'] = (result_df['Return10D'] - result_df['Return5D']).astype(float)
            else:
                result_df['ReturnAccel10_5'] = np.nan
        except Exception as e:
            logger.warning(f"{symbol} ReturnAccel10_5计算失败: {e}")
            unavailable_indicators.append('ReturnAccel10_5')
            result_df['ReturnAccel10_5'] = np.nan
        try:
            if 'MA5' in result_df.columns:
                result_df['MA5Slope'] = self.calculate_ma_slope(result_df['MA5'])
            else:
                result_df['MA5Slope'] = np.nan
            if 'MA20' in result_df.columns:
                result_df['MA20Slope'] = self.calculate_ma_slope(result_df['MA20'])
            else:
                result_df['MA20Slope'] = np.nan
            if 'MA50' in result_df.columns:
                result_df['MA50Slope'] = self.calculate_ma_slope(result_df['MA50'])
            else:
                result_df['MA50Slope'] = np.nan
            if 'MA100' in result_df.columns:
                result_df['MA100Slope'] = self.calculate_ma_slope(result_df['MA100'])
            else:
                result_df['MA100Slope'] = np.nan
            # 价格相对均线偏离
            try:
                result_df['PctFromMA20'] = self.calculate_pct_from_ma(df, result_df['MA20'], 'PctFromMA20')
            except:
                result_df['PctFromMA20'] = np.nan
            try:
                result_df['PctFromMA50'] = self.calculate_pct_from_ma(df, result_df['MA50'], 'PctFromMA50')
            except:
                result_df['PctFromMA50'] = np.nan
            try:
                result_df['PctFromMA100'] = self.calculate_pct_from_ma(df, result_df['MA100'], 'PctFromMA100')
            except:
                result_df['PctFromMA100'] = np.nan
        except Exception as e:
            logger.warning(f"{symbol} MA斜率计算失败: {e}")
            unavailable_indicators.extend(['MA5Slope','MA20Slope','MA50Slope','MA100Slope','PctFromMA20','PctFromMA50','PctFromMA100'])
            result_df['MA5Slope'] = np.nan
            result_df['MA20Slope'] = np.nan
            result_df['MA50Slope'] = np.nan
            result_df['MA100Slope'] = np.nan
            result_df['PctFromMA20'] = np.nan
            result_df['PctFromMA50'] = np.nan
            result_df['PctFromMA100'] = np.nan
        
        # ===== 新增技术指标特征（滚动偏度/峰度、ATR/波动率、高阶动量、成交量变化、价格斜率） =====
        # 仅计算 EXPECTED_FEATURE_NAMES 中包含的列，避免无用重算
        windows = [5, 10, 20, 30, 60]
        # 1) 滚动偏度/峰度
        need_sk_kurt = any(
            (f"Skewness_{p}D" in expected_feature_names) or (f"Kurtosis_{p}D" in expected_feature_names)
            for p in windows
        )
        if need_sk_kurt:
            for period in windows:
                do_sk = f'Skewness_{period}D' in expected_feature_names
                do_ku = f'Kurtosis_{period}D' in expected_feature_names
                if not (do_sk or do_ku):
                    continue
                try:
                    skew, kurt = self.calculate_rolling_skewness_kurtosis(df, period)
                    if do_sk:
                        result_df[f'Skewness_{period}D'] = skew
                    if do_ku:
                        result_df[f'Kurtosis_{period}D'] = kurt
                except Exception as e:
                    logger.warning(f"{symbol} Skewness/Kurtosis_{period}D计算失败: {e}")
                    if do_sk:
                        unavailable_indicators.append(f'Skewness_{period}D')
                        result_df[f'Skewness_{period}D'] = np.nan
                    if do_ku:
                        unavailable_indicators.append(f'Kurtosis_{period}D')
                        result_df[f'Kurtosis_{period}D'] = np.nan
        
        # 2) 滚动 ATR/波动率
        need_atr_vol = any(
            (f"ATR_{p}D" in expected_feature_names) or (f"Volatility_{p}D" in expected_feature_names)
            for p in windows
        )
        if need_atr_vol:
            for period in windows:
                do_atr = f'ATR_{period}D' in expected_feature_names
                do_vol = f'Volatility_{period}D' in expected_feature_names
                if not (do_atr or do_vol):
                    continue
                try:
                    atr, vol = self.calculate_rolling_atr_volatility(df, period)
                    if do_atr:
                        result_df[f'ATR_{period}D'] = atr
                    if do_vol:
                        result_df[f'Volatility_{period}D'] = vol
                except Exception as e:
                    logger.warning(f"{symbol} ATR/Volatility_{period}D计算失败: {e}")
                    if do_atr:
                        unavailable_indicators.append(f'ATR_{period}D')
                        result_df[f'ATR_{period}D'] = np.nan
                    if do_vol:
                        unavailable_indicators.append(f'Volatility_{period}D')
                        result_df[f'Volatility_{period}D'] = np.nan
        
        # 3) 高阶动量指标
        need_mom = any(
            (f"Momentum_{p}D" in expected_feature_names) or
            (f"ROC_{p}D" in expected_feature_names) or
            (f"MomentumAccel_{p}D" in expected_feature_names)
            for p in windows
        )
        if need_mom:
            for period in windows:
                do_m = f'Momentum_{period}D' in expected_feature_names
                do_roc = f'ROC_{period}D' in expected_feature_names
                do_acc = f'MomentumAccel_{period}D' in expected_feature_names
                if not (do_m or do_roc or do_acc):
                    continue
                try:
                    momentum, roc, momentum_accel = self.calculate_advanced_momentum(df, period)
                    if do_m:
                        result_df[f'Momentum_{period}D'] = momentum
                    if do_roc:
                        result_df[f'ROC_{period}D'] = roc
                    if do_acc:
                        result_df[f'MomentumAccel_{period}D'] = momentum_accel if period >= 5 else np.nan
                except Exception as e:
                    logger.warning(f"{symbol} Momentum/ROC/Accel_{period}D计算失败: {e}")
                    if do_m:
                        unavailable_indicators.append(f'Momentum_{period}D')
                        result_df[f'Momentum_{period}D'] = np.nan
                    if do_roc:
                        unavailable_indicators.append(f'ROC_{period}D')
                        result_df[f'ROC_{period}D'] = np.nan
                    if do_acc:
                        unavailable_indicators.append(f'MomentumAccel_{period}D')
                        result_df[f'MomentumAccel_{period}D'] = np.nan
        
        # 4) 成交量变化特征
        need_volchg = any(
            (f"VolumeChange_{p}D" in expected_feature_names) or
            (f"VolumeRatio_{p}D" in expected_feature_names) or
            (f"VolumeMomentum_{p}D" in expected_feature_names)
            for p in windows
        )
        if need_volchg:
            for period in windows:
                do_vc = f'VolumeChange_{period}D' in expected_feature_names
                do_vr = f'VolumeRatio_{period}D' in expected_feature_names
                do_vm = f'VolumeMomentum_{period}D' in expected_feature_names
                if not (do_vc or do_vr or do_vm):
                    continue
                try:
                    vol_change, vol_ratio, vol_momentum = self.calculate_volume_change_features(df, period)
                    if do_vc:
                        result_df[f'VolumeChange_{period}D'] = vol_change
                    if do_vr:
                        result_df[f'VolumeRatio_{period}D'] = vol_ratio if period >= 5 else np.nan
                    if do_vm:
                        result_df[f'VolumeMomentum_{period}D'] = vol_momentum if period >= 5 else np.nan
                except Exception as e:
                    logger.warning(f"{symbol} VolumeChange/Ratio/Momentum_{period}D计算失败: {e}")
                    if do_vc:
                        unavailable_indicators.append(f'VolumeChange_{period}D')
                        result_df[f'VolumeChange_{period}D'] = np.nan
                    if do_vr:
                        unavailable_indicators.append(f'VolumeRatio_{period}D')
                        result_df[f'VolumeRatio_{period}D'] = np.nan
                    if do_vm:
                        unavailable_indicators.append(f'VolumeMomentum_{period}D')
                        result_df[f'VolumeMomentum_{period}D'] = np.nan
        
        # 5) 不同时间窗口的价格斜率
        need_slope = any(
            (f"Slope_{p}D" in expected_feature_names) or (f"SlopeNorm_{p}D" in expected_feature_names)
            for p in windows
        )
        if need_slope:
            for period in windows:
                do_s = f'Slope_{period}D' in expected_feature_names
                do_sn = f'SlopeNorm_{period}D' in expected_feature_names
                if not (do_s or do_sn):
                    continue
                try:
                    slope, slope_norm = self.calculate_price_slope(df, period)
                    if do_s:
                        result_df[f'Slope_{period}D'] = slope
                    if do_sn:
                        result_df[f'SlopeNorm_{period}D'] = slope_norm
                except Exception as e:
                    logger.warning(f"{symbol} Slope/SlopeNorm_{period}D计算失败: {e}")
                    if do_s:
                        unavailable_indicators.append(f'Slope_{period}D')
                        result_df[f'Slope_{period}D'] = np.nan
                    if do_sn:
                        unavailable_indicators.append(f'SlopeNorm_{period}D')
                        result_df[f'SlopeNorm_{period}D'] = np.nan
        
        # ===== 新增下跌趋势特征 =====
        # 1. 单日 & 多日收益
        try:
            result_df['ret_1d'] = self.calculate_ret_1d(df)
        except Exception as e:
            logger.warning(f"{symbol} ret_1d计算失败: {e}")
            result_df['ret_1d'] = np.nan
        for period in [3, 5, 10, 20]:
            try:
                result_df[f'ret_{period}d'] = self.calculate_ret_multi(df, period)
            except Exception as e:
                logger.warning(f"{symbol} ret_{period}d计算失败: {e}")
                result_df[f'ret_{period}d'] = np.nan

        # 1b. 月波动率对应的20D趋势强度：ret_20d / (MonthlyVolatility折算到20D)
        # MonthlyVolatility = std_20 * sqrt(252)（年化）；折算到20日区间波动规模：std_20*sqrt(20) = MonthlyVolatility*sqrt(20/252)
        ts_col = 'TrendStrength_Monthly20D'
        if (not expected_feature_names) or (ts_col in expected_feature_names):
            try:
                if ('ret_20d' in result_df.columns) and ('MonthlyVolatility' in result_df.columns):
                    denom = result_df['MonthlyVolatility'].astype(float) * np.sqrt(20.0 / 252.0)
                    result_df[ts_col] = self._safe_div(result_df['ret_20d'], denom)
                else:
                    result_df[ts_col] = np.nan
            except Exception as e:
                logger.warning(f"{symbol} {ts_col}计算失败: {e}")
                result_df[ts_col] = np.nan
        
        # 2. 标准化斜率（趋势强度）
        for period in [5, 10, 20]:
            try:
                result_df[f'ret_slope_{period}'] = self.calculate_ret_slope(df, period)
            except Exception as e:
                logger.warning(f"{symbol} ret_slope_{period}计算失败: {e}")
                result_df[f'ret_slope_{period}'] = np.nan
        
        # 3. 下跌速度（单向）
        for period in [3, 5, 10]:
            try:
                result_df[f'down_speed_{period}'] = self.calculate_down_speed(df, period)
            except Exception as e:
                logger.warning(f"{symbol} down_speed_{period}计算失败: {e}")
                result_df[f'down_speed_{period}'] = np.nan
        
        # 4. 收益加速度
        for period in [3, 5]:
            try:
                result_df[f'ret_acc_{period}'] = self.calculate_ret_acc(df, period)
            except Exception as e:
                logger.warning(f"{symbol} ret_acc_{period}计算失败: {e}")
                result_df[f'ret_acc_{period}'] = np.nan
        
        # 5. 短期vs长期波动率比
        try:
            result_df['vol_ratio_5_20'] = self.calculate_vol_ratio_5_20(df)
        except Exception as e:
            logger.warning(f"{symbol} vol_ratio_5_20计算失败: {e}")
            result_df['vol_ratio_5_20'] = np.nan
        
        # 6. 连续大阴线统计
        for period in [5, 10]:
            try:
                result_df[f'big_down_cnt_{period}'] = self.calculate_big_down_cnt(df, period, threshold=-0.02)
            except Exception as e:
                logger.warning(f"{symbol} big_down_cnt_{period}计算失败: {e}")
                result_df[f'big_down_cnt_{period}'] = np.nan
        
        # 7. 跳空低开
        try:
            result_df['gap_down_rate'] = self.calculate_gap_down_rate(df)
        except Exception as e:
            logger.warning(f"{symbol} gap_down_rate计算失败: {e}")
            result_df['gap_down_rate'] = np.nan
        
        # 8. 高点抬低次数
        for period in [5, 10]:
            try:
                result_df[f'high_lower_streak_{period}'] = self.calculate_high_lower_streak(df, period)
            except Exception as e:
                logger.warning(f"{symbol} high_lower_streak_{period}计算失败: {e}")
                result_df[f'high_lower_streak_{period}'] = np.nan
        
        # 9. 低点创新低次数
        for period in [5, 10]:
            try:
                result_df[f'low_newlow_streak_{period}'] = self.calculate_low_newlow_streak(df, period)
            except Exception as e:
                logger.warning(f"{symbol} low_newlow_streak_{period}计算失败: {e}")
                result_df[f'low_newlow_streak_{period}'] = np.nan
        
        # 10. 相对近高点/近低点的位置
        try:
            result_df['pos_in_range_20'] = self.calculate_pos_in_range(df, period=20)
        except Exception as e:
            logger.warning(f"{symbol} pos_in_range_20计算失败: {e}")
            result_df['pos_in_range_20'] = np.nan
        
        # 11. 成交量放大比
        try:
            result_df['vol_ratio_5_20_vol'] = self.calculate_vol_ratio_vol(df)
        except Exception as e:
            logger.warning(f"{symbol} vol_ratio_5_20_vol计算失败: {e}")
            result_df['vol_ratio_5_20_vol'] = np.nan
        
        # 12. 下跌放量指示
        try:
            result_df['down_heavy_cnt_10'] = self.calculate_down_heavy_cnt(df, period=10, vol_threshold=1.5)
        except Exception as e:
            logger.warning(f"{symbol} down_heavy_cnt_10计算失败: {e}")
            result_df['down_heavy_cnt_10'] = np.nan
        
        # 13. OBV斜率
        for period in [10, 20]:
            try:
                result_df[f'obv_slope_{period}'] = self.calculate_obv_slope(df, period)
            except Exception as e:
                logger.warning(f"{symbol} obv_slope_{period}计算失败: {e}")
                result_df[f'obv_slope_{period}'] = np.nan
        
        # 14. RSI斜率
        for period in [3, 5]:
            try:
                result_df[f'rsi_slope_{period}'] = self.calculate_rsi_slope(df, period, rsi_period=14)
            except Exception as e:
                logger.warning(f"{symbol} rsi_slope_{period}计算失败: {e}")
                result_df[f'rsi_slope_{period}'] = np.nan
        
        # 15. MACD柱子斜率
        for period in [3, 5]:
            try:
                result_df[f'macd_hist_slope_{period}'] = self.calculate_macd_hist_slope(df, period)
            except Exception as e:
                logger.warning(f"{symbol} macd_hist_slope_{period}计算失败: {e}")
                result_df[f'macd_hist_slope_{period}'] = np.nan
        
        # 16. 相对近期高点跌幅
        for period in [10, 20]:
            try:
                result_df[f'drop_from_max_{period}'] = self.calculate_drop_from_max(df, period)
            except Exception as e:
                logger.warning(f"{symbol} drop_from_max_{period}计算失败: {e}")
                result_df[f'drop_from_max_{period}'] = np.nan
        
        # 17. 相对近期低点反弹幅度
        try:
            result_df['rebound_from_min_10'] = self.calculate_rebound_from_min(df, period=10)
        except Exception as e:
            logger.warning(f"{symbol} rebound_from_min_10计算失败: {e}")
            result_df['rebound_from_min_10'] = np.nan
        
        # ===== 优化：去碎片化DataFrame以提高性能 =====
        # 由于多次对result_df赋值导致碎片化，在保存前进行去碎片化
        result_df = result_df.copy()
        
        # ===== 合并市场外生特征（SPY/QQQ/VIX）=====
        try:
            mkt_cols_all = []
            spy_df = self._load_unprocessed_symbol('SPY')
            if spy_df is not None:
                m_spy = self._compute_mkt_features('SPY', spy_df)
                result_df = result_df.merge(m_spy, on='date', how='left')
                mkt_cols_all.extend([c for c in m_spy.columns if c != 'date'])
            qqq_df = self._load_unprocessed_symbol('QQQ')
            if qqq_df is not None:
                m_qqq = self._compute_mkt_features('QQQ', qqq_df)
                result_df = result_df.merge(m_qqq, on='date', how='left')
                mkt_cols_all.extend([c for c in m_qqq.columns if c != 'date'])
            vix_df = self._load_unprocessed_symbol('VIX')
            if vix_df is not None:
                m_vix = self._compute_mkt_features('VIX', vix_df)
                result_df = result_df.merge(m_vix, on='date', how='left')
                mkt_cols_all.extend([c for c in m_vix.columns if c != 'date'])
            if mkt_cols_all:
                logger.info(f"{symbol} 已合并市场外生特征: {mkt_cols_all}")
        except Exception as e:
            logger.warning(f"{symbol} 合并市场外生特征失败: {e}")
        # 趋势标签与权重迁移至 labeler.py 统一计算（此处不再生成）
        
        # 打印无法计算的指标
        if unavailable_indicators:
            logger.warning(f"{symbol} 无法计算的指标: {', '.join(unavailable_indicators)}")
        
        # 保存处理后的数据（简化增量逻辑：仅检查“最新一天是否有新增行”与“是否有新增列”）
        processed_file = os.path.join(self.processed_dir, f"{symbol}_processed.csv")
        if os.path.exists(processed_file):
            try:
                old_df = pd.read_csv(processed_file)
                if 'date' not in old_df.columns:
                    raise ValueError("已处理数据缺少date列")
                old_df['date'] = pd.to_datetime(old_df['date'])
                result_df['date'] = pd.to_datetime(result_df['date'])
                # 1) 是否有新增列
                new_cols = [c for c in result_df.columns if c not in old_df.columns]
                # 2) 是否有新增的日期行（仅比较到最新一天）
                last_old_date = old_df['date'].max()
                new_rows = result_df[result_df['date'] > last_old_date].copy()
                # 如果既没有新增列也没有新增行，则直接跳过
                if (not new_cols) and new_rows.empty:
                    logger.info(f"{symbol} 无新增列且无新日期数据，跳过保存")
                    return
                updated = old_df.copy()
                # 回填新增列（仅合并新列到历史行；已有列不改）
                if new_cols:
                    try:
                        backfill = result_df[['date'] + new_cols].copy()
                        updated = updated.merge(backfill, on='date', how='left', suffixes=('', '_new'))
                        for col in new_cols:
                            newc = f"{col}_new"
                            if newc in updated.columns:
                                updated[col] = updated[newc].where(~updated[newc].isna(), updated.get(col, np.nan))
                                updated.drop(columns=[newc], inplace=True)
                        logger.info(f"{symbol} 新增列已回填: {new_cols}")
                    except Exception as e:
                        logger.warning(f"{symbol} 新增列回填失败，跳过新增列回填: {e}")
                # 追加新日期行
                if not new_rows.empty:
                    # 确保新行包含旧表的所有列，没有的列补NaN，并按旧表列顺序排列
                    for col in updated.columns:
                        if col not in new_rows.columns:
                            new_rows[col] = np.nan
                    new_rows = new_rows[updated.columns]
                    merged = pd.concat([updated, new_rows], ignore_index=True)
                else:
                    merged = updated
                merged = merged.sort_values('date').reset_index(drop=True)
                # 过滤列：只保留特征表中的列
                merged = self._filter_features(merged, expected_feature_names)
                merged.to_csv(processed_file, index=False, encoding='utf-8-sig')
                logger.info(f"{symbol} 增量更新完成（仅检查最新一天与新增列）。保存到: {processed_file}")
                logger.info(f"{symbol} 最终列数: {len(merged.columns)}，行数: {len(merged)}")
            except Exception as e:
                logger.warning(f"{symbol} 增量合并失败，回退为整表覆盖保存: {e}")
                result_df = self._filter_features(result_df, expected_feature_names)
                result_df.to_csv(processed_file, index=False, encoding='utf-8-sig')
                logger.info(f"{symbol} 处理完成（整表覆盖保存），数据已保存到: {processed_file}")
        else:
            result_df = self._filter_features(result_df, expected_feature_names)
        result_df.to_csv(processed_file, index=False, encoding='utf-8-sig')
        logger.info(f"{symbol} 处理完成，数据已保存到: {processed_file}")
        logger.info(f"{symbol} 共计算了 {len(result_df.columns) - len(df.columns)} 个技术指标")
    
    def process_all_symbols(self):
        """处理所有标的的数据（使用多线程加速）"""
        if not os.path.exists(self.unprocessed_dir):
            logger.error(f"未找到unprocessed目录: {self.unprocessed_dir}")
            return
        
        # 获取所有未处理数据文件
        unprocessed_files = [f for f in os.listdir(self.unprocessed_dir) 
                            if f.endswith('_unprocessed.csv')]
        
        if not unprocessed_files:
            logger.warning(f"unprocessed目录中没有找到数据文件")
            return
        
        symbols = [f.replace('_unprocessed.csv', '') for f in unprocessed_files]
        logger.info(f"找到 {len(symbols)} 个标的需要处理: {symbols}")
        
        # 获取线程数配置
        max_workers = getattr(config, 'PROCESSOR_MAX_WORKERS', 4)
        if max_workers is None:
            max_workers = min(len(symbols), (os.cpu_count() or 1) + 4)
        
        logger.info(f"使用多线程处理，线程数: {max_workers}")
        
        # 使用线程池并行处理
        completed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {executor.submit(self.process_symbol, symbol): symbol 
                               for symbol in symbols}
            
            # 处理完成的任务
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    future.result()  # 获取结果，如果有异常会在这里抛出
                    completed_count += 1
                    logger.info(f"✅ [{completed_count}/{len(symbols)}] {symbol} 处理完成")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ 处理 {symbol} 时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        logger.info("="*60)
        logger.info(f"所有标的处理完成！")
        logger.info(f"  成功: {completed_count} 个")
        logger.info(f"  失败: {failed_count} 个")
        logger.info("="*60)


def main():
    """主函数"""
    processor = DataProcessor()
    processor.process_all_symbols()
    
    # 数据处理完成后，自动执行趋势可视化
    logger.info("\n" + "="*60)
    logger.info("数据处理完成，开始自动执行趋势可视化...")
    logger.info("="*60)
    
    try:
        from data_process.visualize_trends import TrendVisualizer
        visualizer = TrendVisualizer()
        visualizer.visualize_all_symbols()
        logger.info("\n✅ 趋势可视化完成！")
    except Exception as e:
        logger.warning(f"⚠️ 趋势可视化失败（不影响数据处理结果）: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

