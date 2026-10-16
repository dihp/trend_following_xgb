"""
AB实验：对比“原有特征” vs “原有特征 + 趋势强度(收益/波动率)”在测试集的正确率。

用法（建议在已完成数据处理/划分后运行）：
  python3 ab_train_trend_strength.py

说明：
- 训练脚本复用 trend_model/train_xgboost.py 的 TrendXGBoostTrainer.train()
- baseline 通过 config.EXTRA_EXCLUDE_COLUMNS 排除趋势强度列实现
- 若你的 XGB 数据集还没包含新列，请先跑：
    python3 main.py
  或至少执行：数据处理(step2) + 数据划分(step6)
"""

import os
import sys
import json
from datetime import datetime

# 确保可导入项目模块
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from trend_model.train_xgboost import TrendXGBoostTrainer


def _check_feature_presence():
    """检查XGB训练数据里是否包含趋势强度特征列。"""
    try:
        import pandas as pd
        train_path = os.path.join(project_root, config.DATA_DIR, 'xgb', 'train', 'all_symbols.csv')
        if not os.path.exists(train_path):
            return False, f"未找到训练数据文件: {train_path}"
        df = pd.read_csv(train_path, nrows=5)
        cols = set(df.columns)
        need = set(getattr(config, 'TREND_STRENGTH_FEATURE_NAMES', []))
        missing = sorted(list(need - cols))
        if missing:
            return False, f"训练数据缺少趋势强度列: {missing}"
        return True, "OK"
    except Exception as e:
        return False, f"检查训练数据特征列失败: {e}"


def _run(tag: str, exclude_strength: bool):
    # 设置额外排除列
    strength_cols = list(getattr(config, 'TREND_STRENGTH_FEATURE_NAMES', []))
    config.EXTRA_EXCLUDE_COLUMNS = strength_cols if exclude_strength else []

    trainer = TrendXGBoostTrainer()
    res = trainer.train()

    test_acc = float(res.get('test_results', {}).get('accuracy', float('nan')))
    out = {
        'tag': tag,
        'exclude_strength': bool(exclude_strength),
        'test_accuracy': test_acc,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'trend_strength_features': strength_cols,
    }
    return out


def main():
    ok, msg = _check_feature_presence()
    if not ok:
        print(f"[提示] {msg}")
        print("[提示] 请先重新生成XGB数据集（确保新列进入 data/XGB/... ）。例如：python3 main.py")
        # 仍允许继续跑（如果你只想看脚本跑通），但结果会等同于“无新特征”

    print("\n========== AB: baseline (排除趋势强度特征) ==========")
    baseline = _run(tag='baseline', exclude_strength=True)
    print(f"[baseline] test_accuracy = {baseline['test_accuracy']:.4f}")

    print("\n========== AB: enhanced (包含趋势强度特征) ==========")
    enhanced = _run(tag='enhanced', exclude_strength=False)
    print(f"[enhanced] test_accuracy = {enhanced['test_accuracy']:.4f}")

    diff = enhanced['test_accuracy'] - baseline['test_accuracy']
    print("\n========== 结果对比 ==========")
    print(f"baseline  test_accuracy: {baseline['test_accuracy']:.4f}")
    print(f"enhanced  test_accuracy: {enhanced['test_accuracy']:.4f}")
    print(f"delta(enhanced-baseline): {diff:.4f}")

    # 保存结果
    out_dir = os.path.join(project_root, 'models', 'result')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"ab_trend_strength_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({'baseline': baseline, 'enhanced': enhanced, 'diff': diff}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存AB结果: {out_file}")


if __name__ == '__main__':
    main()


