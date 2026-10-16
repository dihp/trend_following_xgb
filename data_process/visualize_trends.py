"""
可视化data_processor标记的趋势
在K线图上标记出趋势段
"""

import os
import sys
import pandas as pd
import numpy as np
# 在多线程环境中使用非交互式后端（避免GUI线程问题）
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrendVisualizer:
    """趋势可视化器"""
    
    def __init__(self):
        """初始化"""
        # 获取项目根目录（上级目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, config.DATA_DIR)
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # 创建可视化输出目录
        self.viz_dir = os.path.join(project_root, 'trend_visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            logger.info(f"创建可视化目录: {self.viz_dir}")
    
    def find_trend_segments(self, df):
        """
        从标签中找出连续的趋势段
        
        Args:
            df: DataFrame，包含label列（1=上涨趋势, -1=下跌趋势, 0=无趋势）
        
        Returns:
            list: 趋势段列表，每个元素是(start_idx, end_idx, direction)
        """
        trend_segments = []
        
        if 'label' not in df.columns:
            return trend_segments
        
        in_trend = False
        trend_start = None
        current_direction = None
        
        for i in range(len(df)):
            label = df['label'].iloc[i]
            
            if pd.isna(label) or label == 0:
                if in_trend:
                    # 趋势段结束
                    trend_end = i - 1
                    trend_segments.append((trend_start, trend_end, current_direction))
                    in_trend = False
                    current_direction = None
                continue
            
            if label == 1 or label == -1:  # 有趋势（上涨或下跌）
                if not in_trend:
                    # 趋势段开始
                    trend_start = i
                    current_direction = int(label)  # 1或-1
                    in_trend = True
                elif int(label) != current_direction:
                    # 趋势方向改变，结束当前趋势段，开始新趋势段
                    trend_end = i - 1
                    trend_segments.append((trend_start, trend_end, current_direction))
                    trend_start = i
                    current_direction = int(label)
        
        # 处理最后一个趋势段
        if in_trend:
            trend_end = len(df) - 1
            trend_segments.append((trend_start, trend_end, current_direction))
        
        return trend_segments
    
    def visualize_symbol(self, symbol):
        """
        可视化单个股票的趋势
        
        Args:
            symbol: 股票代码
        """
        processed_file = os.path.join(self.processed_dir, f'{symbol}_processed.csv')
        
        if not os.path.exists(processed_file):
            logger.warning(f"未找到 {symbol} 的处理后数据文件: {processed_file}")
            return
        
        try:
            df = pd.read_csv(processed_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"处理 {symbol}，共 {len(df)} 条数据")
            
            # 找出趋势段
            trend_segments = self.find_trend_segments(df)
            logger.info(f"  找到 {len(trend_segments)} 个趋势段")
            
            # 创建图表（只显示价格图和日期）
            fig = plt.figure(figsize=(18, 8))
            ax = fig.add_subplot(1, 1, 1)
            
            dates = df['date'].values
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # 绘制K线（使用收盘价线图和阴影表示高低价区间）
            ax.plot(dates, closes, 'k-', linewidth=1.5, label='收盘价', alpha=0.8, zorder=3)
            ax.fill_between(dates, lows, highs, alpha=0.2, color='gray', label='高低价区间', zorder=1)
            
            # 标记趋势段
            for start_idx, end_idx, direction in trend_segments:
                seg_dates = dates[start_idx:end_idx+1]
                seg_closes = closes[start_idx:end_idx+1]
                
                # 根据方向选择颜色
                color = 'green' if direction == 1 else 'red'
                alpha = 0.3
                
                # 绘制趋势段背景
                start_date = dates[start_idx]
                end_date = dates[end_idx]
                ax.axvspan(start_date, end_date, alpha=alpha, color=color, zorder=0)
                
                # 绘制趋势段价格线（加粗）
                # 只为第一个趋势段添加图例标签
                label_text = None
                if trend_segments and start_idx == trend_segments[0][0]:
                    label_text = f'{"上升" if direction == 1 else "下降"}趋势'
                
                ax.plot(seg_dates, seg_closes, color=color, linewidth=3, 
                        alpha=0.7, zorder=4, label=label_text if label_text else '')
            
            ax.set_xlabel('日期', fontsize=12, fontweight='bold')
            ax.set_ylabel('价格 (USD)', fontsize=12, fontweight='bold')
            ax.set_title(f'{symbol} - K线图与趋势识别', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 格式化x轴日期
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 尝试使用tight_layout，如果失败则使用subplots_adjust
            try:
                plt.tight_layout()
            except Exception as e:
                logger.debug(f"tight_layout失败，使用subplots_adjust: {e}")
                plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.15)
            
            # 保存图表
            output_file = os.path.join(self.viz_dir, f'{symbol}_trends.png')
            try:
                # 直接保存，savefig会自动处理绘制
                plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            except Exception as e:
                # 如果tight失败，尝试不使用tight
                try:
                    plt.savefig(output_file, dpi=150, facecolor='white', edgecolor='none')
                except Exception as e2:
                    logger.error(f"保存图表失败: {e2}")
                    raise
            finally:
                plt.close(fig)
            
            logger.info(f"  ✅ 图表已保存到: {output_file}")
            
            # 输出统计信息
            if trend_segments:
                total_trend_days = sum(end_idx - start_idx + 1 for start_idx, end_idx, _ in trend_segments)
                up_trends = sum(1 for _, _, d in trend_segments if d == 1)
                down_trends = sum(1 for _, _, d in trend_segments if d == -1)
                
                logger.info(f"  趋势段统计:")
                logger.info(f"    总趋势段数: {len(trend_segments)} (上升: {up_trends}, 下降: {down_trends})")
                logger.info(f"    趋势总天数: {total_trend_days} / {len(df)} ({total_trend_days/len(df)*100:.1f}%)")
                if trend_segments:
                    avg_length = np.mean([end_idx - start_idx + 1 for start_idx, end_idx, _ in trend_segments])
                    logger.info(f"    平均趋势段长度: {avg_length:.1f} 日")
            
        except Exception as e:
            logger.error(f"可视化 {symbol} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_all_symbols(self):
        """可视化所有股票的趋势"""
        if not os.path.exists(self.processed_dir):
            logger.error(f"未找到processed目录: {self.processed_dir}")
            return
        
        processed_files = [f for f in os.listdir(self.processed_dir) 
                          if f.endswith('_processed.csv')]
        
        if not processed_files:
            logger.warning("没有找到已处理的数据文件")
            return
        
        symbols = [f.replace('_processed.csv', '') for f in processed_files]
        
        logger.info("="*60)
        logger.info(f"开始可视化 {len(symbols)} 个股票的趋势")
        logger.info("="*60)
        
        # 顺序处理（不使用多线程，避免matplotlib线程安全问题）
        completed_count = 0
        failed_count = 0
        
        for i, symbol in enumerate(symbols, 1):
            try:
                self.visualize_symbol(symbol)
                completed_count += 1
                logger.info(f"✅ [{completed_count}/{len(symbols)}] {symbol} 可视化完成")
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ 可视化 {symbol} 时发生错误: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("\n" + "="*60)
        logger.info("所有可视化图表已生成完成！")
        logger.info(f"  成功: {completed_count} 个")
        logger.info(f"  失败: {failed_count} 个")
        logger.info("="*60)
        logger.info(f"图表保存位置: {self.viz_dir}")


def main():
    """主函数"""
    visualizer = TrendVisualizer()
    visualizer.visualize_all_symbols()


if __name__ == '__main__':
    main()

