"""
IBKR数据收集程序
使用ib_insync库从Interactive Brokers收集日K线数据
"""

import os
import sys
# 添加项目根目录到路径，以便导入根目录的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from datetime import datetime, timedelta
import pytz
import eventkit_fix  # 导入EventKit修复文件，解决大小写问题
from ib_insync import IB, Stock, Index, Forex, Future, util
import config
import logging
import yfinance as yf
import json
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IBKRDataCollector:
    """IBKR数据收集器"""
    
    def __init__(self, host=config.IBKR_HOST, port=config.IBKR_PORT):
        """
        初始化IBKR连接
        
        Args:
            host: IBKR TWS或Gateway的IP地址
            port: IBKR端口号
        """
        self.ib = IB()
        self.host = host
        self.port = port
        
        # 获取项目根目录（上级目录），确保data目录在项目根目录下
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, config.DATA_DIR)
        
        # 创建数据目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")
        else:
            logger.info(f"数据目录已存在: {self.data_dir}")
        
        # 创建未处理数据目录
        self.unprocessed_dir = os.path.join(self.data_dir, 'unprocessed')
        if not os.path.exists(self.unprocessed_dir):
            os.makedirs(self.unprocessed_dir)
            logger.info(f"创建未处理数据目录: {self.unprocessed_dir}")
        else:
            logger.info(f"未处理数据目录已存在: {self.unprocessed_dir}")
        
        # 元数据文件路径（存储industry、sector、sub-industry信息）
        self.metadata_file = os.path.join(self.data_dir, 'symbols_metadata.json')
    
    def connect(self):
        """连接到IBKR"""
        try:
            self.ib.connect(self.host, self.port, clientId=1)
            logger.info(f"成功连接到IBKR: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"连接IBKR失败: {e}")
            return False
    
    def disconnect(self):
        """断开IBKR连接"""
        try:
            self.ib.disconnect()
            logger.info("已断开IBKR连接")
        except Exception as e:
            logger.error(f"断开连接时出错: {e}")
    
    def create_contract(self, symbol_config):
        """
        创建合约对象
        
        Args:
            symbol_config: 标的配置字典
            
        Returns:
            Contract对象
        """
        if symbol_config['secType'] == 'STK':
            contract = Stock(
                symbol_config['symbol'],
                symbol_config['exchange'],
                symbol_config['currency']
            )
        elif symbol_config['secType'] == 'IND':
            contract = Index(
                symbol_config['symbol'],
                symbol_config['exchange'],
                symbol_config['currency']
            )
        elif symbol_config['secType'] == 'CASH':
            # 外汇对：XAUUSD 等
            contract = Forex(
                symbol_config['symbol'],
                symbol_config['exchange'],
                symbol_config['currency']
            )
        elif symbol_config['secType'] == 'FUT':
            # 期货合约
            contract = Future(
                symbol=symbol_config['symbol'],
                exchange=symbol_config['exchange'],
                currency=symbol_config['currency']
            )
            # 如果指定了合约月份，则设置
            if 'lastTradeDateOrContractMonth' in symbol_config and symbol_config['lastTradeDateOrContractMonth']:
                contract.lastTradeDateOrContractMonth = symbol_config['lastTradeDateOrContractMonth']
        else:
            # 可以扩展支持其他类型的合约
            raise ValueError(f"不支持的secType: {symbol_config['secType']}")
        
        return contract
    
    def get_end_date_time(self):
        """
        根据当前EST时间获取数据截止日期
        
        规则：
        - 如果现在是EST 16:00前，返回昨天
        - 如果现在是EST 16:00后，返回今天
        
        Returns:
            str: 格式化的日期时间字符串，用于IBKR API
        """
        # 获取EST时区
        est = pytz.timezone('US/Eastern')
        
        # 获取当前EST时间
        now_est = datetime.now(est)
        
        # 判断是否在16:00之前
        if now_est.hour < 16:
            # 16:00前，使用昨天
            end_date = (now_est - timedelta(days=1)).date()
            logger.info(f"当前EST时间: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}，使用昨天作为截止日期: {end_date}")
        else:
            # 16:00后，使用今天
            end_date = now_est.date()
            logger.info(f"当前EST时间: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}，使用今天作为截止日期: {end_date}")
        
        # 将日期转换为IBKR需要的格式（YYYYMMDD HH:MM:SS）
        # 使用当天的16:00作为截止时间（市场收盘时间）
        end_datetime_str = end_date.strftime('%Y%m%d') + ' 16:00:00'
        return end_datetime_str
    
    def get_historical_data(self, contract, duration=None, barSize='1 day', since_date=None):
        """
        获取历史K线数据，如果不足10年则尽可能多地读取数据
        
        Args:
            contract: 合约对象
            duration: 数据持续时间，如果为None则使用config中的HISTORICAL_DAYS
            barSize: K线大小，'1 day' 表示日K线
            since_date: 增量抓取的起始日期（不含当天），若提供则仅抓取此日期之后的数据
            
        Returns:
            tuple: (处理后的DataFrame, 原始DataFrame)，包含日期、开盘、高点、低点、收盘、交易量
        """
        # 获取截止日期时间
        end_date_time = self.get_end_date_time()
        
        # 定义尝试的时间段（从长到短），优先增量抓取
        if since_date is not None:
            try:
                # 计算需要补充的天数（加1天缓冲）
                est = pytz.timezone('US/Eastern')
                # end_date_time 格式为 'YYYYMMDD 16:00:00'，取日期部分
                end_date = datetime.strptime(end_date_time.split(' ')[0], '%Y%m%d').date()
                # since_date可能包含时间，统一为date
                since_d = since_date.date() if hasattr(since_date, 'date') else pd.to_datetime(since_date).date()
                days = (end_date - since_d).days + 1
                if days <= 0:
                    logger.info(f"{contract.symbol} 已是最新，无需增量抓取（since={since_d}, end={end_date}）")
                    return None, None
                # 限制最小与上限，避免过大范围
                days = max(1, min(days, config.HISTORICAL_DAYS))
                durations = [f'{days} D', '3 M', '1 M']  # 回退方案
                logger.info(f"增量抓取 {contract.symbol}: 自 {since_d} 之后，共约 {days} 天")
            except Exception as e:
                logger.warning(f"计算增量天数失败，回退为全量策略: {e}")
                since_date = None  # 回退
        
        if since_date is None:
            if duration is None:
                # 使用配置的天数，转换为年数
                target_days = config.HISTORICAL_DAYS
                target_years = target_days / 365.0
                
                # 生成尝试的时间段列表：10年 -> 5年 -> 3年 -> 2年 -> 1年 -> 6个月 -> 3个月 -> 1个月
                durations = []
                if target_years >= 10:
                    durations = ['10 Y', '5 Y', '3 Y', '2 Y', '1 Y', '6 M', '3 M', '1 M']
                elif target_years >= 5:
                    durations = ['5 Y', '3 Y', '2 Y', '1 Y', '6 M', '3 M', '1 M']
                elif target_years >= 3:
                    durations = ['3 Y', '2 Y', '1 Y', '6 M', '3 M', '1 M']
                elif target_years >= 2:
                    durations = ['2 Y', '1 Y', '6 M', '3 M', '1 M']
                elif target_years >= 1:
                    durations = ['1 Y', '6 M', '3 M', '1 M']
                else:
                    # 如果配置的天数少于1年，使用天数格式
                    durations = [f'{target_days} D', '6 M', '3 M', '1 M']
            else:
                durations = [duration]
        
        # 尝试不同的时间段，直到成功获取数据
        for attempt_duration in durations:
            try:
                logger.info(f"尝试获取 {contract.symbol} 的 {attempt_duration} 历史数据（截止日期: {end_date_time}）...")
                
                # 请求历史数据
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_date_time,
                    durationStr=attempt_duration,
                    barSizeSetting=barSize,
                    whatToShow='TRADES',
                    useRTH=True,  # 仅使用常规交易时间
                    formatDate=1
                )
                
                if not bars or len(bars) == 0:
                    logger.warning(f"未获取到 {contract.symbol} 的 {attempt_duration} 历史数据，尝试更短的时间段...")
                    continue
                
                # 转换为DataFrame（原始数据）
                df_raw = util.df(bars)
                
                if df_raw.empty:
                    logger.warning(f"{contract.symbol} 的 {attempt_duration} 数据为空，尝试更短的时间段...")
                    continue
                
                # 保存（增量合并）未处理数据
                self.save_or_merge_unprocessed_data(df_raw, contract.symbol)
                
                # 创建处理后的数据副本
                df = df_raw.copy()
                
                # 重命名列以匹配中文需求
                df.rename(columns={
                    'date': '日期',
                    'open': '开盘价',
                    'high': '最高价',
                    'low': '最低价',
                    'close': '收盘价',
                    'volume': '交易量'
                }, inplace=True)
                
                # 确保日期列为datetime类型
                df['日期'] = pd.to_datetime(df['日期'])
                
                # 选择需要的列
                df = df[['日期', '开盘价', '最高价', '最低价', '收盘价', '交易量']]
                
                logger.info(f"成功获取 {contract.symbol} 的 {len(df)} 条日K线数据（时间段: {attempt_duration}）")
                return df, df_raw
                
            except Exception as e:
                logger.warning(f"获取 {contract.symbol} 的 {attempt_duration} 历史数据时出错: {e}，尝试更短的时间段...")
                continue
        
        # 所有时间段都失败了
        logger.error(f"无法获取 {contract.symbol} 的历史数据，已尝试所有时间段")
        return None, None
    
    def get_industry_info(self, symbol):
        """
        从yfinance API获取标的的行业信息
        
        Args:
            symbol: 标的代码
            
        Returns:
            dict: 包含industry、sector、sub-industry的字典
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            industry_info = {
                'symbol': symbol,
                'industry': info.get('industry', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'sub_industry': info.get('industryDisp', 'N/A'),  # 子行业信息
                'long_name': info.get('longName', 'N/A'),
                'market_cap': info.get('marketCap', None),
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"成功获取 {symbol} 的行业信息: {industry_info['sector']} - {industry_info['industry']}")
            return industry_info
            
        except Exception as e:
            logger.warning(f"获取 {symbol} 行业信息失败: {e}")
            return {
                'symbol': symbol,
                'industry': 'N/A',
                'sector': 'N/A',
                'sub_industry': 'N/A',
                'long_name': 'N/A',
                'market_cap': None,
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
    
    def load_metadata(self):
        """加载已保存的元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载元数据文件失败: {e}")
                return {}
        return {}
    
    def save_metadata(self, metadata):
        """
        保存元数据到JSON文件
        
        Args:
            metadata: 元数据字典
        """
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"元数据已保存到: {self.metadata_file}")
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def update_metadata(self, symbol, industry_info):
        """
        更新单个标的的元数据
        
        Args:
            symbol: 标的代码
            industry_info: 行业信息字典
        """
        metadata = self.load_metadata()
        metadata[symbol] = industry_info
        self.save_metadata(metadata)
    
    def load_existing_unprocessed(self, symbol):
        """
        读取已存在的未处理数据，并返回DataFrame与最后一个日期
        """
        filename = os.path.join(self.unprocessed_dir, f"{symbol}_unprocessed.csv")
        if not os.path.exists(filename):
            return None, None
        try:
            df_exist = pd.read_csv(filename)
            if 'date' in df_exist.columns:
                df_exist['date'] = pd.to_datetime(df_exist['date'])
                last_date = df_exist['date'].max()
            else:
                last_date = None
            return df_exist, last_date
        except Exception as e:
            logger.warning(f"读取已存在未处理数据失败，将视为不存在: {e}")
            return None, None
    
    def save_or_merge_unprocessed_data(self, df_raw, symbol):
        """
        保存未处理数据到unprocessed文件夹
        
        Args:
            df_raw: 原始数据DataFrame（未处理）
            symbol: 标的代码
        """
        if df_raw is None or df_raw.empty:
            logger.warning(f"{symbol} 未处理数据为空，跳过保存")
            return
        
        filename = os.path.join(self.unprocessed_dir, f"{symbol}_unprocessed.csv")
        # 合并去重
        df_exist, _ = self.load_existing_unprocessed(symbol)
        if df_exist is not None and not df_exist.empty:
            try:
                # 统一日期类型
                if 'date' in df_raw.columns:
                    df_raw['date'] = pd.to_datetime(df_raw['date'])
                if 'date' in df_exist.columns:
                    df_exist['date'] = pd.to_datetime(df_exist['date'])
                df_merged = pd.concat([df_exist, df_raw], ignore_index=True)
                if 'date' in df_merged.columns:
                    df_merged = df_merged.drop_duplicates(subset=['date']).sort_values('date')
                else:
                    df_merged = df_merged.drop_duplicates()
                df_merged.to_csv(filename, index=False, encoding='utf-8-sig')
                logger.info(f"未处理数据已合并保存到: {filename}（新增 {len(df_merged)-len(df_exist)} 条）")
                return
            except Exception as e:
                logger.warning(f"合并未处理数据失败，改为覆盖保存: {e}")
        df_raw.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"未处理数据已保存到: {filename}")
    
    def collect_all_symbols(self):
        """收集所有配置中的标的的数据"""
        if not self.connect():
            logger.error("无法连接到IBKR，请检查TWS或IB Gateway是否运行")
            return
        
        try:
            for symbol_config in config.SYMBOLS:
                symbol = symbol_config['symbol']
                logger.info(f"开始收集 {symbol} 的数据...")
                
                # 获取行业信息（从yfinance API）
                logger.info(f"正在获取 {symbol} 的行业信息...")
                industry_info = self.get_industry_info(symbol)
                self.update_metadata(symbol, industry_info)
                
                # 添加延迟，避免API请求过快
                time.sleep(0.5)
                
                # 创建合约
                contract = self.create_contract(symbol_config)
                
                # 解析合约（确保合约信息正确）
                self.ib.qualifyContracts(contract)
                
                # 增量：读取已存在的最后日期，仅抓取新数据
                _, last_date = self.load_existing_unprocessed(symbol)
                df, df_raw = self.get_historical_data(contract, since_date=last_date)
                
                if df is None:
                    logger.info(f"{symbol} 无需更新或未获取到新数据")
                
        finally:
            self.disconnect()
    
    def collect_single_symbol(self, symbol_name):
        """
        收集单个标的的数据
        
        Args:
            symbol_name: 标的代码
        """
        if not self.connect():
            logger.error("无法连接到IBKR，请检查TWS或IB Gateway是否运行")
            return
        
        try:
            # 查找对应的配置
            symbol_config = None
            for config_item in config.SYMBOLS:
                if config_item['symbol'] == symbol_name:
                    symbol_config = config_item
                    break
            
            if symbol_config is None:
                logger.error(f"未找到标的 {symbol_name} 的配置")
                return
            
            logger.info(f"开始收集 {symbol_name} 的数据...")
            
            # 获取行业信息（从yfinance API）
            logger.info(f"正在获取 {symbol_name} 的行业信息...")
            industry_info = self.get_industry_info(symbol_name)
            self.update_metadata(symbol_name, industry_info)
            
            # 创建合约
            contract = self.create_contract(symbol_config)
            
            # 解析合约
            self.ib.qualifyContracts(contract)
            
            # 增量：读取已存在的最后日期，仅抓取新数据
            _, last_date = self.load_existing_unprocessed(symbol_name)
            df, df_raw = self.get_historical_data(contract, since_date=last_date)
            
            if df is None:
                logger.info(f"{symbol_name} 无需更新或未获取到新数据")
                
        finally:
            self.disconnect()


def main():
    """主函数"""
    collector = IBKRDataCollector()
    
    # 收集所有标的的数据
    collector.collect_all_symbols()
    
    # 如果只想收集单个标的，可以使用：
    # collector.collect_single_symbol('AAPL')


if __name__ == '__main__':
    main()

