"""
标签处理器
保留原始label列，并将processed_label更新为向后移动3天，用于预测3天后的趋势方向
"""

import os
import pandas as pd
import numpy as np
import config
import logging
from datetime import datetime
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 控制台输出
    ]
)
logger = logging.getLogger(__name__)


class LabelProcessor:
    """标签处理器：将label列向后移动3天"""
    
    def __init__(self, shift_days=3):
        """
        初始化标签处理器
        
        Args:
            shift_days: label列向后移动的天数（默认3天）
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(project_root, config.DATA_DIR)
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.shift_days = shift_days
        
        logger.info(f"标签处理器初始化完成")
        logger.info(f"  label列向后移动天数: {shift_days}天")
    
    def process_symbol(self, symbol):
        """
        处理单个标的的标签
        
        Args:
            symbol: 股票代码
        """
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
        
        # 检查是否有label列
        if 'label' not in df.columns:
            logger.warning(f"{symbol} 数据缺少 'label' 列，跳过处理")
            return
        
        # 计算未来label（向后移动指定天数）
        original_label = df['label'].copy()
        future_label = original_label.shift(-self.shift_days)
        
        # 末尾的NaN值填充为0（无法预测未来，标记为无趋势）
        nan_count = future_label.isna().sum()
        if nan_count > 0:
            future_label = future_label.fillna(0)
            logger.info(f"{symbol} 末尾{nan_count}个未来label的NaN值已填充为0")
        
        # 确保未来label为整数类型
        future_label = future_label.astype(int)
        
        # 将未来label写入processed_label列（保留原label列）
        if 'processed_label' in df.columns:
            df['processed_label'] = future_label
            logger.info(f"{symbol} 已更新processed_label列")
        else:
            label_col_idx = df.columns.get_loc('label')
            df.insert(label_col_idx + 1, 'processed_label', future_label)
            logger.info(f"{symbol} 已插入processed_label列")
        
        # 统计原始与未来label的分布
        original_counts = original_label.value_counts().sort_index()
        future_counts = df['processed_label'].value_counts().sort_index()
        
        logger.info(f"{symbol} processed_label已向后移动{self.shift_days}天")
        logger.info(f"  原始label分布: {dict(original_counts)}")
        logger.info(f"  未来processed_label分布: {dict(future_counts)}")
 
        # 保存更新后的数据
        df.to_csv(processed_file, index=False, encoding='utf-8-sig')
        logger.info(f"{symbol} 标签处理完成，数据已保存到: {processed_file}")
    
    def process_all_symbols(self):
        """处理所有标的的标签"""
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
        logger.info(f"找到 {len(symbols)} 个标的需要处理标签: {symbols}")
        
        completed_count = 0
        failed_count = 0
        
        for symbol in symbols:
            try:
                self.process_symbol(symbol)
                completed_count += 1
                logger.info(f"✅ [{completed_count}/{len(symbols)}] {symbol} 标签处理完成")
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ 为 {symbol} 处理标签时发生错误: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("="*60)
        logger.info(f"所有标的标签处理完成！")
        logger.info(f"  成功: {completed_count} 个")
        logger.info(f"  失败: {failed_count} 个")
        logger.info("="*60)


def main():
    """主函数"""
    # 从config读取参数（如果存在），否则使用默认值3天
    shift_days = getattr(config, 'LABEL_SHIFT_DAYS', 3)
    processor = LabelProcessor(shift_days=shift_days)
    
    logger.info("\n" + "="*60)
    logger.info("开始处理标签（将label列向后移动）...")
    logger.info("="*60)
    processor.process_all_symbols()


if __name__ == '__main__':
    main()

