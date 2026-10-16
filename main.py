"""
主程序：自动执行完整的数据处理流程
按照顺序执行：data_processor -> labeler -> visualizer -> label_processor -> data_splitter
"""

import os
import sys
import logging
import glob
import subprocess
from datetime import datetime
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 控制台输出
    ]
)
logger = logging.getLogger(__name__)

def _parse_backup_timestamp(name: str, prefix: str):
    """
    解析备份文件名中的时间戳：prefix_YYYYMMDD_HHMMSS.csv
    返回 datetime 或 None。
    """
    m = re.match(rf"^{re.escape(prefix)}_(\d{{8}})_(\d{{6}})\.csv$", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except Exception:
        return None


def prune_apply_signal_backups(apply_dir: str, prefix: str, keep_max_files: int, logger_obj=None):
    """
    清理 apply 目录中某一类信号文件的“带时间戳备份”，保留最新 keep_max_files 个。
    仅匹配：prefix_YYYYMMDD_HHMMSS.csv（不会匹配 prefix.csv 当前文件）。
    """
    if logger_obj is None:
        logger_obj = logger
    try:
        keep_max_files = int(keep_max_files)
    except Exception:
        keep_max_files = 21
    # 安全保护：避免误配置导致“一个都不保留”
    if keep_max_files < 1:
        keep_max_files = 1
    if not apply_dir or (not os.path.isdir(apply_dir)):
        return

    cands = []
    for name in os.listdir(apply_dir):
        ts = _parse_backup_timestamp(name, prefix)
        if ts is None:
            continue
        p = os.path.join(apply_dir, name)
        if not os.path.isfile(p):
            continue
        cands.append((ts, p))

    cands.sort(key=lambda x: x[0])  # oldest first（按文件名时间戳）
    before = len(cands)
    deleted = 0
    while len(cands) > keep_max_files:
        _, p = cands.pop(0)
        try:
            os.remove(p)
            deleted += 1
        except Exception as e:
            logger_obj.warning(f"清理 apply 信号文件失败：{p}，原因：{e}")
    after = len(cands)
    if before > 0:
        logger_obj.info(f"[apply-prune] {prefix}: 备份文件 {before} -> {after}（删除 {deleted}），保留最新 {keep_max_files} 个")


def clean_unused_symbols():
    """
    清理不在 config.SYMBOLS 中的标的数据
    包括 data/unprocessed、data/processed、data/with_prediction 目录中的文件
    """
    logger.info("=" * 80)
    logger.info("[清理步骤] 开始清理不在配置中的标的数据...")
    logger.info("=" * 80)
    
    try:
        import config
        
        # 从 config.SYMBOLS 中提取所有有效的 symbol（大写）
        valid_symbols = set()
        if hasattr(config, 'SYMBOLS') and isinstance(config.SYMBOLS, list):
            for item in config.SYMBOLS:
                if isinstance(item, dict) and 'symbol' in item:
                    symbol = str(item['symbol']).upper().strip()
                    if symbol:
                        valid_symbols.add(symbol)
        
        if not valid_symbols:
            logger.warning("config.SYMBOLS 为空或格式不正确，跳过清理")
            return
        
        logger.info(f"配置中的有效标的: {sorted(valid_symbols)}")
        
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # 需要清理的目录和文件模式
        dirs_to_clean = [
            ('unprocessed', '{symbol}_unprocessed.csv'),
            ('processed', '{symbol}_processed.csv'),
            ('with_prediction', '{symbol}_with_prediction.csv'),
        ]
        
        total_deleted = 0
        
        for dir_name, file_pattern in dirs_to_clean:
            dir_path = os.path.join(project_root, 'data', dir_name)
            if not os.path.exists(dir_path):
                logger.info(f"目录不存在，跳过: {dir_path}")
                continue
            
            # 获取所有匹配模式的文件
            pattern = file_pattern.format(symbol='*')
            files = glob.glob(os.path.join(dir_path, pattern))
            
            deleted_count = 0
            for file_path in files:
                filename = os.path.basename(file_path)
                # 提取 symbol（去掉后缀）
                if dir_name == 'unprocessed':
                    symbol = filename.replace('_unprocessed.csv', '')
                elif dir_name == 'processed':
                    symbol = filename.replace('_processed.csv', '')
                elif dir_name == 'with_prediction':
                    symbol = filename.replace('_with_prediction.csv', '')
                else:
                    continue
                
                symbol = symbol.upper().strip()
                
                # 跳过 all_with_prediction.csv（合并文件）
                if symbol == 'ALL' and dir_name == 'with_prediction':
                    logger.info(f"保留合并文件: {filename}")
                    continue
                
                # 如果 symbol 不在有效列表中，删除文件
                if symbol not in valid_symbols:
                    try:
                        os.remove(file_path)
                        logger.info(f"删除文件: {file_path} (标的: {symbol})")
                        deleted_count += 1
                        total_deleted += 1
                    except Exception as e:
                        logger.warning(f"删除文件失败: {file_path}, 错误: {e}")
            
            if deleted_count > 0:
                logger.info(f"在 {dir_name} 目录中删除了 {deleted_count} 个文件")
            else:
                logger.info(f"在 {dir_name} 目录中未发现需要删除的文件")
        
        if total_deleted > 0:
            logger.info(f"✅ [清理步骤] 清理完成，共删除 {total_deleted} 个文件")
        else:
            logger.info("✅ [清理步骤] 无需清理，所有文件都在配置中")
        
    except Exception as e:
        logger.error(f"❌ [清理步骤] 清理失败: {e}")
        import traceback
        traceback.print_exc()
        # 清理失败不影响后续流程，继续执行


def run_pipeline(skip_collect: bool = False):
    """
    按顺序执行完整流程。

    Args:
        skip_collect: True 时跳过 data_process/data_collector.py（不重新获取数据），其余训练/处理流程保持一致。
    """
    logger.info("=" * 80)
    logger.info("开始执行完整数据处理流程" + ("（跳过数据收集）" if skip_collect else ""))
    logger.info("=" * 80)
    logger.info("执行顺序：")
    logger.info("  0. 清理不在配置中的标的数据")
    if skip_collect:
        logger.info("  1. [跳过] data_collector - 收集IBKR数据（使用已有本地数据）")
    else:
        logger.info("  1. data_collector - 收集IBKR数据")
    logger.info("  2. data_processor - 计算技术指标")
    logger.info("  3. labeler - 生成趋势标签")
    logger.info("  4. visualizer - 可视化趋势")
    logger.info("  5. label_processor - 处理标签（向后移动3天）")
    logger.info("  6. data_splitter - 划分训练/验证/测试集")
    logger.info("  7. train_long_only_model - 训练长多模型")
    logger.info("  8. train_short_only_model - 训练短空模型")
    logger.info("  9. apply_model - 应用模型进行预测")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    import config
    
    # 0. 清理不在配置中的标的数据
    clean_unused_symbols()
    
    # 1. 收集IBKR数据
    logger.info("\n" + "=" * 80)
    if skip_collect:
        logger.info("[步骤 1/9] 跳过收集IBKR数据（使用已有本地数据）")
        logger.info("=" * 80)
    else:
        logger.info("[步骤 1/9] 开始收集IBKR数据...")
        logger.info("=" * 80)
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(project_root, 'data_process', 'data_collector.py')
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=project_root,
                capture_output=False,
                text=True
            )
            if result.returncode == 0:
                logger.info("✅ [步骤 1/9] IBKR数据收集完成")
            else:
                logger.error(f"❌ [步骤 1/9] IBKR数据收集失败，退出码: {result.returncode}")
                return
        except Exception as e:
            logger.error(f"❌ [步骤 1/9] IBKR数据收集失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 2. 数据处理（计算技术指标）
    logger.info("\n" + "=" * 80)
    logger.info("[步骤 2/9] 开始数据处理（计算技术指标）...")
    logger.info("=" * 80)
    try:
        from data_process.data_processor import DataProcessor
        processor = DataProcessor()
        processor.process_all_symbols()
        logger.info("✅ [步骤 2/9] 数据处理完成")
    except Exception as e:
        logger.error(f"❌ [步骤 2/9] 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 生成趋势标签
    logger.info("\n" + "=" * 80)
    logger.info("[步骤 3/9] 开始生成趋势标签...")
    logger.info("=" * 80)
    try:
        from trend_model.labeler import TrendLabeler
        labeler = TrendLabeler()
        labeler.label_all_symbols()
        logger.info("✅ [步骤 3/9] 趋势标签生成完成")
    except Exception as e:
        logger.error(f"❌ [步骤 3/9] 趋势标签生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 趋势可视化
    logger.info("\n" + "=" * 80)
    logger.info("[步骤 4/9] 开始趋势可视化...")
    logger.info("=" * 80)
    try:
        from data_process.visualize_trends import TrendVisualizer
        visualizer = TrendVisualizer()
        visualizer.visualize_all_symbols()
        logger.info("✅ [步骤 4/9] 趋势可视化完成")
    except Exception as e:
        logger.warning(f"⚠️ [步骤 4/9] 趋势可视化失败（不影响后续流程）: {e}")
        import traceback
        traceback.print_exc()
        # 可视化失败不影响后续流程，继续执行
    
    # 5. 处理标签（向后移动3天）
    logger.info("\n" + "=" * 80)
    logger.info("[步骤 5/9] 开始处理标签（向后移动3天）...")
    logger.info("=" * 80)
    try:
        from label_processor import LabelProcessor
        shift_days = getattr(config, 'LABEL_SHIFT_DAYS', 3)
        processor = LabelProcessor(shift_days=shift_days)
        processor.process_all_symbols()
        logger.info("✅ [步骤 5/9] 标签处理完成")
    except Exception as e:
        logger.error(f"❌ [步骤 5/9] 标签处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 数据划分
    logger.info("\n" + "=" * 80)
    logger.info("[步骤 6/9] 开始数据划分...")
    logger.info("=" * 80)
    try:
        from data_process.data_splitter import DataSplitter
        splitter = DataSplitter()
        splitter.split_all_symbols()
        logger.info("✅ [步骤 6/9] 数据划分完成")
    except Exception as e:
        logger.error(f"❌ [步骤 6/9] 数据划分失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 训练长多模型
    logger.info("\n" + "=" * 80)
    logger.info("[步骤 7/9] 开始训练长多模型...")
    logger.info("=" * 80)
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(project_root, 'trend_model', 'train_long_only_model.py')
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            logger.info("✅ [步骤 7/9] 长多模型训练完成")
        else:
            logger.error(f"❌ [步骤 7/9] 长多模型训练失败，退出码: {result.returncode}")
            return
    except Exception as e:
        logger.error(f"❌ [步骤 7/9] 长多模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 8. 训练短空模型
    logger.info("\n" + "=" * 80)
    logger.info("[步骤 8/9] 开始训练短空模型...")
    logger.info("=" * 80)
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(project_root, 'trend_model', 'train_short_only_model.py')
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            logger.info("✅ [步骤 8/9] 短空模型训练完成")
        else:
            logger.error(f"❌ [步骤 8/9] 短空模型训练失败，退出码: {result.returncode}")
            return
    except Exception as e:
        logger.error(f"❌ [步骤 8/9] 短空模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 9. 应用模型进行预测
    logger.info("\n" + "=" * 80)
    logger.info("[步骤 9/9] 开始应用模型进行预测...")
    logger.info("=" * 80)
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(project_root, 'apply_model.py')
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            logger.info("✅ [步骤 9/9] 模型应用完成")
        else:
            logger.error(f"❌ [步骤 9/9] 模型应用失败，退出码: {result.returncode}")
            return
    except Exception as e:
        logger.error(f"❌ [步骤 9/9] 模型应用失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 完成
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 所有步骤完成！")
    logger.info(f"总耗时: {duration}")
    logger.info("=" * 80)

    # 10. 清理 apply 目录中历史信号文件（只清理带时间戳的备份文件，保留 current 的 .csv）
    try:
        if bool(getattr(config, "APPLY_SIGNAL_AUTO_PRUNE", True)):
            apply_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apply")
            keep_max = int(getattr(config, "APPLY_SIGNAL_KEEP_MAX_FILES", 21))
            prune_apply_signal_backups(apply_dir, "all_trend_signals", keep_max, logger_obj=logger)
            prune_apply_signal_backups(apply_dir, "current_position_signals", keep_max, logger_obj=logger)
            prune_apply_signal_backups(apply_dir, "high_confidence_signals", keep_max, logger_obj=logger)
    except Exception as e:
        logger.warning(f"清理 apply 历史信号文件失败（不影响主流程）：{e}")


def main():
    """主函数：按顺序执行完整流程"""
    run_pipeline(skip_collect=False)


if __name__ == '__main__':
    main()

