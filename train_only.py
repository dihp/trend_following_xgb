"""
Train-only 入口：
复用 main.py 的完整训练/处理流程，但跳过 data_process/data_collector.py（不重新获取数据）。

用法：
    python train_only.py
"""

from main import run_pipeline


def main():
    run_pipeline(skip_collect=True)


if __name__ == "__main__":
    main()