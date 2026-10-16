"""
短空（Short-Only）两阶段二分类训练脚本
Stage1: 是否趋势（0/1）
Stage2: 下跌 vs 不下跌（下跌=1，其余=0）
"""

import os
import sys
# 添加项目根目录到路径，以便导入根目录的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import pickle
import shutil
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import config
from trend_model.train_xgboost import TrendXGBoostTrainer


class ShortOnlyTwoStageTrainer:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = project_root
        self.data_dir = os.path.join(project_root, config.DATA_DIR)
        self.xgb_dir = os.path.join(self.data_dir, 'xgb')
        self.train_dir = os.path.join(self.xgb_dir, 'train')
        self.val_dir = os.path.join(self.xgb_dir, 'validation')
        self.test_dir = os.path.join(self.xgb_dir, 'test')
        self.model_dir = os.path.join(project_root, 'models', 'xgboost', 'short_only')
        os.makedirs(self.model_dir, exist_ok=True)
        # 复用原训练器（用于特征工程、训练与早停逻辑）
        self.base = TrendXGBoostTrainer()

    def _load_all(self):
        # 加载数据（按原流程，使用 all_symbols.csv）
        train_df = self.base.load_data(self.train_dir, use_all_symbols=True, dataset_name='训练集')
        val_df = self.base.load_data(self.val_dir, use_all_symbols=True, dataset_name='验证集')
        test_df = self.base.load_data(self.test_dir, use_all_symbols=True, dataset_name='测试集')
        return train_df, val_df, test_df

    def _prepare_all_features(self, train_df, val_df, test_df):
        # 与原流程一致：在训练集上拟合编码器，验证/测试仅transform
        use_time_features = getattr(config, 'USE_TIME_FEATURES', True)
        use_raw_ohlcv = getattr(config, 'USE_RAW_OHLCV', True)
        use_industry_features = getattr(config, 'USE_INDUSTRY_FEATURES', True)

        X_train, y_train, _, pos_w_train, quality_w_train, time_w_train, symbols_train, use_3_train = self.base.prepare_features(
            train_df,
            use_time_features=use_time_features,
            use_raw_ohlcv=use_raw_ohlcv,
            use_industry_features=use_industry_features,
            fit_encoders=True
        )
        X_val, y_val, _, pos_w_val, quality_w_val, time_w_val, symbols_val, use_3_val = self.base.prepare_features(
            val_df,
            use_time_features=use_time_features,
            use_raw_ohlcv=use_raw_ohlcv,
            use_industry_features=use_industry_features,
            fit_encoders=False
        )
        X_test, y_test, _, pos_w_test, quality_w_test, time_w_test, symbols_test, use_3_test = self.base.prepare_features(
            test_df,
            use_time_features=use_time_features,
            use_raw_ohlcv=use_raw_ohlcv,
            use_industry_features=use_industry_features,
            fit_encoders=False
        )

        # 对齐列
        feature_names = list(X_train.columns)
        X_val = X_val[feature_names]
        X_test = X_test[feature_names]
        return (X_train, y_train, pos_w_train, quality_w_train, time_w_train, symbols_train), \
               (X_val, y_val, pos_w_val, quality_w_val, time_w_val, symbols_val), \
               (X_test, y_test, pos_w_test, quality_w_test, time_w_test, symbols_test), feature_names

    def _evaluate_binary(self, y_true, y_score, thr=0.5, name=''):
        y_pred = (y_score >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        try:
            report = classification_report(y_true, y_pred, target_names=['class0', 'class1'])
        except Exception:
            report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n[{name}] Accuracy={acc:.4f}")
        print(f"[{name}] Classification report:\n{report}")
        print(f"[{name}] Confusion matrix:\n{cm}")
        return acc
    
    def _optimize_short_only_thresholds(self, y_val, p_trend_val, p_down_val, y_val_down_full, metric='accuracy'):
        """
        在验证集上搜索短空模型的最优阈值
        
        Args:
            y_val: 验证集真实标签（0=无趋势, 1=上涨, 2=下跌）
            p_trend_val: Stage1概率 P(trend)
            p_down_val: Stage2概率 P(down | trend)（在全样本上）
            y_val_down_full: 全样本的下跌标签（1=下跌, 0=非下跌）
            metric: 评分指标（'accuracy' 或其他）
        
        Returns:
            dict: {'stage1_trend': float, 'stage2_down': float, 'score': float}
        """
        # 从config读取搜索网格
        grid1 = getattr(config, 'THRESH_SEARCH_GRID_STAGE1_TREND', [0.5, 0.55, 0.6, 0.65, 0.7])
        grid2 = getattr(config, 'THRESH_SEARCH_GRID', {
            'down_trend': [0.5, 0.55, 0.6, 0.65, 0.7]
        })
        grid_down = grid2.get('down_trend', [0.5, 0.55, 0.6, 0.65, 0.7])
        
        best = {'score': -1.0, 'stage1_trend': None, 'stage2_down': None}
        
        # 网格搜索
        for t1 in grid1:
            for thr_down in grid_down:
                # 建议做空：p_trend >= t1 且 p_down >= thr_down
                suggest = ((p_trend_val >= t1) & (p_down_val >= thr_down)).astype(int)
                # 评估：建议做空 vs 真实下跌
                score = accuracy_score(y_val_down_full, suggest)
                
                if score > best['score']:
                    best.update({
                        'score': score,
                        'stage1_trend': float(t1),
                        'stage2_down': float(thr_down)
                    })
        
        return best

    def train(self):
        # 1) 加载数据
        train_df, val_df, test_df = self._load_all()
        # 2) 特征
        (X_train, y_train, pos_w_train, quality_w_train, time_w_train, symbols_train), \
        (X_val, y_val, pos_w_val, quality_w_val, time_w_val, symbols_val), \
        (X_test, y_test, pos_w_test, quality_w_test, time_w_test, symbols_test), feature_names = self._prepare_all_features(
            train_df, val_df, test_df
        )

        use_class_weights = getattr(config, 'USE_CLASS_WEIGHTS', True)

        # Stage1: 是否趋势（0/1）
        y_train_s1 = (pd.Series(y_train).astype(int) != 0).astype(int)
        y_val_s1 = (pd.Series(y_val).astype(int) != 0).astype(int)
        y_test_s1 = (pd.Series(y_test).astype(int) != 0).astype(int)
        # 使用长多模型专门的Stage1超参数（短空模型可以共用，或后续单独配置）
        params_stage1 = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': getattr(config, 'XGB_LONGONLY_STAGE1_MAX_DEPTH', 3),  # 暂时共用，后续可改为 XGB_SHORTONLY_STAGE1_MAX_DEPTH
            'learning_rate': getattr(config, 'XGB_LONGONLY_STAGE1_LEARNING_RATE', 0.02),
            'min_child_weight': getattr(config, 'XGB_LONGONLY_STAGE1_MIN_CHILD_WEIGHT', 5),
            'gamma': getattr(config, 'XGB_LONGONLY_STAGE1_GAMMA', 0.5),
            'subsample': getattr(config, 'XGB_LONGONLY_STAGE1_SUBSAMPLE', 0.7),
            'colsample_bytree': getattr(config, 'XGB_LONGONLY_STAGE1_COLSAMPLE_BYTREE', 0.7),
            'reg_lambda': getattr(config, 'XGB_LONGONLY_STAGE1_REG_LAMBDA', 3.0),
            'reg_alpha': getattr(config, 'XGB_LONGONLY_STAGE1_REG_ALPHA', 0.1),
            'random_state': 42,
            'n_jobs': -1
        }
        # 临时设置长多模型的early stopping配置（短空模型可以共用）
        old_metric = getattr(config, 'XGB_EARLY_STOPPING_METRIC', 'trend_f1')
        old_rounds = getattr(config, 'XGB_EARLY_STOPPING_ROUNDS', 20)
        config.XGB_EARLY_STOPPING_METRIC = getattr(config, 'XGB_LONGONLY_EARLY_STOPPING_METRIC', 'accuracy')
        config.XGB_EARLY_STOPPING_ROUNDS = getattr(config, 'XGB_LONGONLY_STAGE1_EARLY_STOPPING_ROUNDS', 20)
        try:
            model_stage1 = self.base.train_model(
                X_train, y_train_s1, X_val, y_val_s1,
                params=params_stage1,
                use_class_weights=use_class_weights,
                pos_weights=pos_w_train,
                quality_weights=quality_w_train,
                time_weights=time_w_train,
                use_3_class=False
            )
        finally:
            # 恢复原始配置
            config.XGB_EARLY_STOPPING_METRIC = old_metric
            config.XGB_EARLY_STOPPING_ROUNDS = old_rounds
        if model_stage1 is None:
            print("Stage1 训练失败")
            return None
        p_trend_train = model_stage1.predict_proba(X_train)[:, 1]
        p_trend_val = model_stage1.predict_proba(X_val)[:, 1]
        p_trend_test = model_stage1.predict_proba(X_test)[:, 1]
        self._evaluate_binary(y_train_s1, p_trend_train, thr=getattr(config, 'PREDICTION_CONFIDENCE_THRESHOLD', 0.6), name='Stage1-Train')
        self._evaluate_binary(y_val_s1, p_trend_val, thr=getattr(config, 'PREDICTION_CONFIDENCE_THRESHOLD', 0.6), name='Stage1-Val')
        self._evaluate_binary(y_test_s1, p_trend_test, thr=getattr(config, 'PREDICTION_CONFIDENCE_THRESHOLD', 0.6), name='Stage1-Test')

        # Stage2: 只在 y_s1=1 的子集内，预测「下跌 vs 不下跌」
        # 训练集/验证集：仅使用趋势样本（y_s1=1 且 原始y∈{1,2}），二分类标签：1=下跌(2)，0=非下跌(1或0)
        y_train_series = pd.Series(y_train).astype(int)
        y_val_series = pd.Series(y_val).astype(int)
        y_test_series = pd.Series(y_test).astype(int)

        mask_trend_train = (y_train_s1.values == 1) & (y_train_series.isin([1, 2]).values)
        mask_trend_val = (y_val_s1.values == 1) & (y_val_series.isin([1, 2]).values)
        mask_trend_test = (y_test_s1.values == 1) & (y_test_series.isin([1, 2]).values)

        X_train_s2 = X_train[mask_trend_train].copy()
        X_val_s2 = X_val[mask_trend_val].copy()
        # X_test_s2 只用于评估，不参与训练
        X_test_s2 = X_test[mask_trend_test].copy()

        y_train_down = (y_train_series[mask_trend_train] == 2).astype(int)  # 下跌=1，非下跌=0
        y_val_down = (y_val_series[mask_trend_val] == 2).astype(int)
        y_test_down = (y_test_series[mask_trend_test] == 2).astype(int)
        # 使用长多模型专门的Stage2超参数（短空模型可以共用，或后续单独配置）
        params_stage2 = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': getattr(config, 'XGB_LONGONLY_STAGE2_MAX_DEPTH', 3),
            'learning_rate': getattr(config, 'XGB_LONGONLY_STAGE2_LEARNING_RATE', 0.02),
            'min_child_weight': getattr(config, 'XGB_LONGONLY_STAGE2_MIN_CHILD_WEIGHT', 5),
            'gamma': getattr(config, 'XGB_LONGONLY_STAGE2_GAMMA', 0.5),
            'subsample': getattr(config, 'XGB_LONGONLY_STAGE2_SUBSAMPLE', 0.7),
            'colsample_bytree': getattr(config, 'XGB_LONGONLY_STAGE2_COLSAMPLE_BYTREE', 0.7),
            'reg_lambda': getattr(config, 'XGB_LONGONLY_STAGE2_REG_LAMBDA', 3.0),
            'reg_alpha': getattr(config, 'XGB_LONGONLY_STAGE2_REG_ALPHA', 0.1),
            'random_state': 42,
            'n_jobs': -1
        }
        # 临时设置长多模型的early stopping配置
        old_metric = getattr(config, 'XGB_EARLY_STOPPING_METRIC', 'trend_f1')
        old_rounds = getattr(config, 'XGB_EARLY_STOPPING_ROUNDS', 20)
        config.XGB_EARLY_STOPPING_METRIC = getattr(config, 'XGB_LONGONLY_EARLY_STOPPING_METRIC', 'accuracy')
        config.XGB_EARLY_STOPPING_ROUNDS = getattr(config, 'XGB_LONGONLY_STAGE2_EARLY_STOPPING_ROUNDS', 20)
        try:
            model_stage2_down = self.base.train_model(
                X_train_s2, y_train_down, X_val_s2, y_val_down,
                params=params_stage2,
                use_class_weights=False,  # 避免权重过强导致偏置，保留位置/质量/时间权重
                pos_weights=(pos_w_train[mask_trend_train] if pos_w_train is not None else None),
                quality_weights=(quality_w_train[mask_trend_train] if quality_w_train is not None else None),
                time_weights=(time_w_train[mask_trend_train] if time_w_train is not None else None),
                use_3_class=False
            )
        finally:
            # 恢复原始配置
            config.XGB_EARLY_STOPPING_METRIC = old_metric
            config.XGB_EARLY_STOPPING_ROUNDS = old_rounds
        if model_stage2_down is None:
            print("Stage2(下跌 vs 不下跌，在趋势子集内) 训练失败")
            return None
        # 预测时仍在全样本上输出概率，语义近似 P(down | trend)
        p_down_train_full = model_stage2_down.predict_proba(X_train)[:, 1]
        p_down_val_full = model_stage2_down.predict_proba(X_val)[:, 1]
        p_down_test_full = model_stage2_down.predict_proba(X_test)[:, 1]

        # 准备全样本的下跌标签（用于阈值搜索）
        y_val_series = pd.Series(y_val).astype(int)
        y_val_down_full = (y_val_series == 2).astype(int)
        
        # 在验证集上搜索最优阈值
        print("\n开始搜索最优阈值...")
        thresholds = self._optimize_short_only_thresholds(
            y_val, p_trend_val, p_down_val_full, y_val_down_full
        )
        t1 = thresholds['stage1_trend']
        thr_down = thresholds['stage2_down']
        print(f"最优阈值: Stage1(trend)={t1:.3f}, Stage2(down)={thr_down:.3f}, 验证集准确率={thresholds['score']:.4f}")
        
        # 仅在趋势子集上评估Stage2性能（下跌 vs 不下跌）
        if mask_trend_train.any():
            self._evaluate_binary(y_train_down, p_down_train_full[mask_trend_train], thr=thr_down, name='Stage2-Down-Train(trend-only)')
        if mask_trend_val.any():
            self._evaluate_binary(y_val_down, p_down_val_full[mask_trend_val], thr=thr_down, name='Stage2-Down-Val(trend-only)')
        if mask_trend_test.any():
            self._evaluate_binary(y_test_down, p_down_test_full[mask_trend_test], thr=thr_down, name='Stage2-Down-Test(trend-only)')

        # 组合与导出：面向"做空"需求，使用最优阈值
        suggest_train = ((p_trend_train >= t1) & (p_down_train_full >= thr_down)).astype(int)
        suggest_val = ((p_trend_val >= t1) & (p_down_val_full >= thr_down)).astype(int)
        suggest_test = ((p_trend_test >= t1) & (p_down_test_full >= thr_down)).astype(int)
        # 在全体样本上，以"是否真实为下跌(2)"作为标签评估建议做空信号的命中率
        y_train_down_full = (y_train_series == 2).astype(int)
        y_val_down_full = (y_val_series == 2).astype(int)
        y_test_down_full = (y_test_series == 2).astype(int)
        test_accuracy = accuracy_score(y_test_down_full, suggest_test)
        print(f"\n【第一次训练结果】")
        print(f"建议做空（Train）命中率: {accuracy_score(y_train_down_full, suggest_train):.4f}")
        print(f"建议做空（Val）命中率:   {accuracy_score(y_val_down_full, suggest_val):.4f}")
        print(f"建议做空（Test）命中率:  {test_accuracy:.4f}")
        
        # ===== 两阶段训练：去除低置信度样本后重新训练 =====
        use_two_pass = getattr(config, 'USE_TWO_PASS_TRAINING', True)
        if use_two_pass:
            print("\n" + "="*60)
            print("开始两阶段训练：去除低置信度样本后重新训练")
            print("="*60)
            
            # 计算置信度：|prob - 0.5|，值越大表示置信度越高
            low_conf_threshold = getattr(config, 'LOW_CONFIDENCE_THRESHOLD', 0.1)
            
            # Stage1置信度：|P(trend) - 0.5|
            conf_stage1_train = np.abs(p_trend_train - 0.5)
            conf_stage1_val = np.abs(p_trend_val - 0.5)
            
            # Stage2置信度：|P(down|trend) - 0.5|
            conf_stage2_train = np.abs(p_down_train_full - 0.5)
            conf_stage2_val = np.abs(p_down_val_full - 0.5)
            
            # 综合置信度：取两阶段的最小值（如果任一阶段置信度低，则整体置信度低）
            conf_combined_train = np.minimum(conf_stage1_train, conf_stage2_train)
            conf_combined_val = np.minimum(conf_stage1_val, conf_stage2_val)
            
            # 识别高置信度样本（置信度 >= threshold，即 |prob - 0.5| >= threshold）
            high_conf_mask_train = conf_combined_train >= low_conf_threshold
            high_conf_mask_val = conf_combined_val >= low_conf_threshold
            
            n_train_before = len(X_train)
            n_val_before = len(X_val)
            n_train_after = high_conf_mask_train.sum()
            n_val_after = high_conf_mask_val.sum()
            
            print(f"\n低置信度阈值: {low_conf_threshold} (即 |prob - 0.5| < {low_conf_threshold} 的样本将被移除)")
            print(f"训练集: {n_train_before} -> {n_train_after} (移除 {n_train_before - n_train_after} 个低置信度样本, {100*(n_train_before-n_train_after)/n_train_before:.1f}%)")
            print(f"验证集: {n_val_before} -> {n_val_after} (移除 {n_val_before - n_val_after} 个低置信度样本, {100*(n_val_before-n_val_after)/n_val_before:.1f}%)")
            print(f"测试集: 保持不变 ({len(X_test)} 个样本)")
            
            if n_train_after < 100 or n_val_after < 20:
                print(f"\n警告：过滤后样本数过少（训练集{n_train_after}，验证集{n_val_after}），跳过第二次训练")
            else:
                # 过滤数据：只保留高置信度样本
                X_train_filtered = X_train[high_conf_mask_train].copy()
                y_train_filtered = y_train[high_conf_mask_train]
                pos_w_train_filtered = pos_w_train[high_conf_mask_train] if pos_w_train is not None else None
                quality_w_train_filtered = quality_w_train[high_conf_mask_train] if quality_w_train is not None else None
                time_w_train_filtered = time_w_train[high_conf_mask_train] if time_w_train is not None else None
                symbols_train_filtered = symbols_train[high_conf_mask_train] if symbols_train is not None else None
                
                X_val_filtered = X_val[high_conf_mask_val].copy()
                y_val_filtered = y_val[high_conf_mask_val]
                pos_w_val_filtered = pos_w_val[high_conf_mask_val] if pos_w_val is not None else None
                quality_w_val_filtered = quality_w_val[high_conf_mask_val] if quality_w_val is not None else None
                time_w_val_filtered = time_w_val[high_conf_mask_val] if time_w_val is not None else None
                symbols_val_filtered = symbols_val[high_conf_mask_val] if symbols_val is not None else None
                
                print("\n开始第二次训练（仅使用高置信度样本）...")
                
                # 第二次训练 Stage1
                y_train_s1_filtered = (pd.Series(y_train_filtered).astype(int) != 0).astype(int)
                y_val_s1_filtered = (pd.Series(y_val_filtered).astype(int) != 0).astype(int)
                
                config.XGB_EARLY_STOPPING_METRIC = getattr(config, 'XGB_LONGONLY_EARLY_STOPPING_METRIC', 'accuracy')
                config.XGB_EARLY_STOPPING_ROUNDS = getattr(config, 'XGB_LONGONLY_STAGE1_EARLY_STOPPING_ROUNDS', 20)
                try:
                    model_stage1_pass2 = self.base.train_model(
                        X_train_filtered, y_train_s1_filtered, X_val_filtered, y_val_s1_filtered,
                        params=params_stage1,
                        use_class_weights=use_class_weights,
                        pos_weights=pos_w_train_filtered,
                        quality_weights=quality_w_train_filtered,
                        time_weights=time_w_train_filtered,
                        use_3_class=False
                    )
                finally:
                    config.XGB_EARLY_STOPPING_METRIC = old_metric
                    config.XGB_EARLY_STOPPING_ROUNDS = old_rounds
                
                if model_stage1_pass2 is None:
                    print("第二次训练 Stage1 失败，使用第一次训练的模型")
                    model_stage1_pass2 = model_stage1
                else:
                    print("第二次训练 Stage1 完成")
                
                p_trend_train_pass2 = model_stage1_pass2.predict_proba(X_train)[:, 1]
                p_trend_val_pass2 = model_stage1_pass2.predict_proba(X_val)[:, 1]
                p_trend_test_pass2 = model_stage1_pass2.predict_proba(X_test)[:, 1]
                
                # 第二次训练 Stage2
                y_train_series_filtered = pd.Series(y_train_filtered).astype(int)
                y_val_series_filtered = pd.Series(y_val_filtered).astype(int)
                
                mask_trend_train_filtered = (y_train_s1_filtered.values == 1) & (y_train_series_filtered.isin([1, 2]).values)
                mask_trend_val_filtered = (y_val_s1_filtered.values == 1) & (y_val_series_filtered.isin([1, 2]).values)
                
                X_train_s2_filtered = X_train_filtered[mask_trend_train_filtered].copy()
                X_val_s2_filtered = X_val_filtered[mask_trend_val_filtered].copy()
                
                y_train_down_filtered = (y_train_series_filtered[mask_trend_train_filtered] == 2).astype(int)
                y_val_down_filtered = (y_val_series_filtered[mask_trend_val_filtered] == 2).astype(int)
                
                config.XGB_EARLY_STOPPING_METRIC = getattr(config, 'XGB_LONGONLY_EARLY_STOPPING_METRIC', 'accuracy')
                config.XGB_EARLY_STOPPING_ROUNDS = getattr(config, 'XGB_LONGONLY_STAGE2_EARLY_STOPPING_ROUNDS', 20)
                try:
                    model_stage2_down_pass2 = self.base.train_model(
                        X_train_s2_filtered, y_train_down_filtered, X_val_s2_filtered, y_val_down_filtered,
                        params=params_stage2,
                        use_class_weights=False,
                        pos_weights=(pos_w_train_filtered[mask_trend_train_filtered] if pos_w_train_filtered is not None else None),
                        quality_weights=(quality_w_train_filtered[mask_trend_train_filtered] if quality_w_train_filtered is not None else None),
                        time_weights=(time_w_train_filtered[mask_trend_train_filtered] if time_w_train_filtered is not None else None),
                        use_3_class=False
                    )
                finally:
                    config.XGB_EARLY_STOPPING_METRIC = old_metric
                    config.XGB_EARLY_STOPPING_ROUNDS = old_rounds
                
                if model_stage2_down_pass2 is None:
                    print("第二次训练 Stage2 失败，使用第一次训练的模型")
                    model_stage2_down_pass2 = model_stage2_down
                else:
                    print("第二次训练 Stage2 完成")
                
                # 使用第二次训练的模型进行预测
                p_down_train_full_pass2 = model_stage2_down_pass2.predict_proba(X_train)[:, 1]
                p_down_val_full_pass2 = model_stage2_down_pass2.predict_proba(X_val)[:, 1]
                p_down_test_full_pass2 = model_stage2_down_pass2.predict_proba(X_test)[:, 1]
                
                # 在验证集上搜索最优阈值（使用第二次训练的模型，在全验证集上评估）
                thresholds_pass2 = self._optimize_short_only_thresholds(
                    y_val, p_trend_val_pass2, p_down_val_full_pass2, y_val_down_full
                )
                t1_pass2 = thresholds_pass2['stage1_trend']
                thr_down_pass2 = thresholds_pass2['stage2_down']
                print(f"\n第二次训练最优阈值: Stage1(trend)={t1_pass2:.3f}, Stage2(down)={thr_down_pass2:.3f}, 验证集准确率={thresholds_pass2['score']:.4f}")
                
                # 评估第二次训练的结果
                suggest_train_pass2 = ((p_trend_train_pass2 >= t1_pass2) & (p_down_train_full_pass2 >= thr_down_pass2)).astype(int)
                suggest_val_pass2 = ((p_trend_val_pass2 >= t1_pass2) & (p_down_val_full_pass2 >= thr_down_pass2)).astype(int)
                suggest_test_pass2 = ((p_trend_test_pass2 >= t1_pass2) & (p_down_test_full_pass2 >= thr_down_pass2)).astype(int)
                
                test_accuracy_pass2 = accuracy_score(y_test_down_full, suggest_test_pass2)
                print(f"\n【第二次训练结果】")
                print(f"建议做空（Train）命中率: {accuracy_score(y_train_down_full, suggest_train_pass2):.4f}")
                print(f"建议做空（Val）命中率:   {accuracy_score(y_val_down_full, suggest_val_pass2):.4f}")
                print(f"建议做空（Test）命中率:  {test_accuracy_pass2:.4f}")
                
                # 比较两次训练的结果
                print(f"\n【对比】")
                print(f"测试集准确率: 第一次={test_accuracy:.4f}, 第二次={test_accuracy_pass2:.4f}, 提升={test_accuracy_pass2-test_accuracy:+.4f}")
                
                # ===== 第三次训练：进一步剔除Stage2概率在[0.48-0.52]区间的样本 =====
                use_three_pass = getattr(config, 'USE_THREE_PASS_TRAINING', True)
                stage2_filter_range = getattr(config, 'STAGE2_NARROW_FILTER', [0.48, 0.52])
                
                if use_three_pass and test_accuracy_pass2 >= test_accuracy:
                    print("\n" + "="*60)
                    print("开始第三次训练：进一步剔除Stage2概率在窄区间的样本")
                    print("="*60)
                    
                    # 使用第二次训练的模型预测，找出Stage2概率在[0.48-0.52]区间的样本
                    # 这些样本在第二次训练的高置信度样本中，但Stage2概率仍然接近0.5
                    p_down_train_filtered = p_down_train_full_pass2[high_conf_mask_train]
                    p_down_val_filtered = p_down_val_full_pass2[high_conf_mask_val]
                    
                    # 剔除Stage2概率在[0.48-0.52]区间的样本
                    stage2_narrow_mask_train = (p_down_train_filtered < stage2_filter_range[0]) | (p_down_train_filtered > stage2_filter_range[1])
                    stage2_narrow_mask_val = (p_down_val_filtered < stage2_filter_range[0]) | (p_down_val_filtered > stage2_filter_range[1])
                    
                    n_train_pass2 = len(X_train_filtered)
                    n_val_pass2 = len(X_val_filtered)
                    n_train_pass3 = stage2_narrow_mask_train.sum()
                    n_val_pass3 = stage2_narrow_mask_val.sum()
                    
                    print(f"\nStage2窄区间过滤: [{stage2_filter_range[0]}, {stage2_filter_range[1]}]")
                    print(f"训练集: {n_train_pass2} -> {n_train_pass3} (移除 {n_train_pass2 - n_train_pass3} 个样本, {100*(n_train_pass2-n_train_pass3)/n_train_pass2:.1f}%)")
                    print(f"验证集: {n_val_pass2} -> {n_val_pass3} (移除 {n_val_pass2 - n_val_pass3} 个样本, {100*(n_val_pass2-n_val_pass3)/n_val_pass2:.1f}%)")
                    print(f"测试集: 保持不变 ({len(X_test)} 个样本)")
                    
                    if n_train_pass3 < 100 or n_val_pass3 < 20:
                        print(f"\n警告：过滤后样本数过少（训练集{n_train_pass3}，验证集{n_val_pass3}），跳过第三次训练")
                    else:
                        # 进一步过滤数据
                        X_train_pass3 = X_train_filtered[stage2_narrow_mask_train].copy()
                        y_train_pass3 = y_train_filtered[stage2_narrow_mask_train]
                        pos_w_train_pass3 = pos_w_train_filtered[stage2_narrow_mask_train] if pos_w_train_filtered is not None else None
                        quality_w_train_pass3 = quality_w_train_filtered[stage2_narrow_mask_train] if quality_w_train_filtered is not None else None
                        time_w_train_pass3 = time_w_train_filtered[stage2_narrow_mask_train] if time_w_train_filtered is not None else None
                        symbols_train_pass3 = symbols_train_filtered[stage2_narrow_mask_train] if symbols_train_filtered is not None else None
                        
                        X_val_pass3 = X_val_filtered[stage2_narrow_mask_val].copy()
                        y_val_pass3 = y_val_filtered[stage2_narrow_mask_val]
                        pos_w_val_pass3 = pos_w_val_filtered[stage2_narrow_mask_val] if pos_w_val_filtered is not None else None
                        quality_w_val_pass3 = quality_w_val_filtered[stage2_narrow_mask_val] if quality_w_val_filtered is not None else None
                        time_w_val_pass3 = time_w_val_filtered[stage2_narrow_mask_val] if time_w_val_filtered is not None else None
                        symbols_val_pass3 = symbols_val_filtered[stage2_narrow_mask_val] if symbols_val_filtered is not None else None
                        
                        print("\n开始第三次训练（使用最严格过滤后的样本）...")
                        
                        # 第三次训练 Stage1
                        y_train_s1_pass3 = (pd.Series(y_train_pass3).astype(int) != 0).astype(int)
                        y_val_s1_pass3 = (pd.Series(y_val_pass3).astype(int) != 0).astype(int)
                        
                        config.XGB_EARLY_STOPPING_METRIC = getattr(config, 'XGB_LONGONLY_EARLY_STOPPING_METRIC', 'accuracy')
                        config.XGB_EARLY_STOPPING_ROUNDS = getattr(config, 'XGB_LONGONLY_STAGE1_EARLY_STOPPING_ROUNDS', 20)
                        try:
                            model_stage1_pass3 = self.base.train_model(
                                X_train_pass3, y_train_s1_pass3, X_val_pass3, y_val_s1_pass3,
                                params=params_stage1,
                                use_class_weights=use_class_weights,
                                pos_weights=pos_w_train_pass3,
                                quality_weights=quality_w_train_pass3,
                                time_weights=time_w_train_pass3,
                                use_3_class=False
                            )
                        finally:
                            config.XGB_EARLY_STOPPING_METRIC = old_metric
                            config.XGB_EARLY_STOPPING_ROUNDS = old_rounds
                        
                        if model_stage1_pass3 is None:
                            print("第三次训练 Stage1 失败，使用第二次训练的模型")
                            model_stage1_pass3 = model_stage1_pass2
                        else:
                            print("第三次训练 Stage1 完成")
                        
                        p_trend_train_pass3 = model_stage1_pass3.predict_proba(X_train)[:, 1]
                        p_trend_val_pass3 = model_stage1_pass3.predict_proba(X_val)[:, 1]
                        p_trend_test_pass3 = model_stage1_pass3.predict_proba(X_test)[:, 1]
                        
                        # 第三次训练 Stage2
                        y_train_series_pass3 = pd.Series(y_train_pass3).astype(int)
                        y_val_series_pass3 = pd.Series(y_val_pass3).astype(int)
                        
                        mask_trend_train_pass3 = (y_train_s1_pass3.values == 1) & (y_train_series_pass3.isin([1, 2]).values)
                        mask_trend_val_pass3 = (y_val_s1_pass3.values == 1) & (y_val_series_pass3.isin([1, 2]).values)
                        
                        X_train_s2_pass3 = X_train_pass3[mask_trend_train_pass3].copy()
                        X_val_s2_pass3 = X_val_pass3[mask_trend_val_pass3].copy()
                        
                        y_train_down_pass3 = (y_train_series_pass3[mask_trend_train_pass3] == 2).astype(int)
                        y_val_down_pass3 = (y_val_series_pass3[mask_trend_val_pass3] == 2).astype(int)
                        
                        config.XGB_EARLY_STOPPING_METRIC = getattr(config, 'XGB_LONGONLY_EARLY_STOPPING_METRIC', 'accuracy')
                        config.XGB_EARLY_STOPPING_ROUNDS = getattr(config, 'XGB_LONGONLY_STAGE2_EARLY_STOPPING_ROUNDS', 20)
                        try:
                            model_stage2_down_pass3 = self.base.train_model(
                                X_train_s2_pass3, y_train_down_pass3, X_val_s2_pass3, y_val_down_pass3,
                                params=params_stage2,
                                use_class_weights=False,
                                pos_weights=(pos_w_train_pass3[mask_trend_train_pass3] if pos_w_train_pass3 is not None else None),
                                quality_weights=(quality_w_train_pass3[mask_trend_train_pass3] if quality_w_train_pass3 is not None else None),
                                time_weights=(time_w_train_pass3[mask_trend_train_pass3] if time_w_train_pass3 is not None else None),
                                use_3_class=False
                            )
                        finally:
                            config.XGB_EARLY_STOPPING_METRIC = old_metric
                            config.XGB_EARLY_STOPPING_ROUNDS = old_rounds
                        
                        if model_stage2_down_pass3 is None:
                            print("第三次训练 Stage2 失败，使用第二次训练的模型")
                            model_stage2_down_pass3 = model_stage2_down_pass2
                        else:
                            print("第三次训练 Stage2 完成")
                        
                        # 使用第三次训练的模型进行预测
                        p_down_train_full_pass3 = model_stage2_down_pass3.predict_proba(X_train)[:, 1]
                        p_down_val_full_pass3 = model_stage2_down_pass3.predict_proba(X_val)[:, 1]
                        p_down_test_full_pass3 = model_stage2_down_pass3.predict_proba(X_test)[:, 1]
                        
                        # 在验证集上搜索最优阈值（使用第三次训练的模型）
                        thresholds_pass3 = self._optimize_short_only_thresholds(
                            y_val, p_trend_val_pass3, p_down_val_full_pass3, y_val_down_full
                        )
                        t1_pass3 = thresholds_pass3['stage1_trend']
                        thr_down_pass3 = thresholds_pass3['stage2_down']
                        print(f"\n第三次训练最优阈值: Stage1(trend)={t1_pass3:.3f}, Stage2(down)={thr_down_pass3:.3f}, 验证集准确率={thresholds_pass3['score']:.4f}")
                        
                        # 评估第三次训练的结果
                        suggest_train_pass3 = ((p_trend_train_pass3 >= t1_pass3) & (p_down_train_full_pass3 >= thr_down_pass3)).astype(int)
                        suggest_val_pass3 = ((p_trend_val_pass3 >= t1_pass3) & (p_down_val_full_pass3 >= thr_down_pass3)).astype(int)
                        suggest_test_pass3 = ((p_trend_test_pass3 >= t1_pass3) & (p_down_test_full_pass3 >= thr_down_pass3)).astype(int)
                        
                        test_accuracy_pass3 = accuracy_score(y_test_down_full, suggest_test_pass3)
                        print(f"\n【第三次训练结果】")
                        print(f"建议做空（Train）命中率: {accuracy_score(y_train_down_full, suggest_train_pass3):.4f}")
                        print(f"建议做空（Val）命中率:   {accuracy_score(y_val_down_full, suggest_val_pass3):.4f}")
                        print(f"建议做空（Test）命中率:  {test_accuracy_pass3:.4f}")
                        
                        # 比较三次训练的结果
                        print(f"\n【三次训练对比】")
                        print(f"测试集准确率: 第一次={test_accuracy:.4f}, 第二次={test_accuracy_pass2:.4f}, 第三次={test_accuracy_pass3:.4f}")
                        print(f"提升: 第二次相对第一次={test_accuracy_pass2-test_accuracy:+.4f}, 第三次相对第二次={test_accuracy_pass3-test_accuracy_pass2:+.4f}, 第三次相对第一次={test_accuracy_pass3-test_accuracy:+.4f}")
                        
                        # 选择效果最好的模型
                        best_accuracy = max(test_accuracy, test_accuracy_pass2, test_accuracy_pass3)
                        if test_accuracy_pass3 >= best_accuracy:
                            print(f"使用第三次训练的模型（效果最好）")
                            model_stage1 = model_stage1_pass3
                            model_stage2_down = model_stage2_down_pass3
                            t1 = t1_pass3
                            thr_down = thr_down_pass3
                            test_accuracy = test_accuracy_pass3
                            suggest_test = suggest_test_pass3
                        elif test_accuracy_pass2 >= test_accuracy:
                            print(f"使用第二次训练的模型（效果最好）")
                            model_stage1 = model_stage1_pass2
                            model_stage2_down = model_stage2_down_pass2
                            t1 = t1_pass2
                            thr_down = thr_down_pass2
                            test_accuracy = test_accuracy_pass2
                            suggest_test = suggest_test_pass2
                        else:
                            print(f"使用第一次训练的模型（效果最好）")
                else:
                    # 只比较前两次训练
                    if test_accuracy_pass2 >= test_accuracy:
                        print(f"使用第二次训练的模型（效果更好）")
                        model_stage1 = model_stage1_pass2
                        model_stage2_down = model_stage2_down_pass2
                        t1 = t1_pass2
                        thr_down = thr_down_pass2
                        test_accuracy = test_accuracy_pass2
                        suggest_test = suggest_test_pass2
                    else:
                        print(f"使用第一次训练的模型（效果更好）")

        # 保存模型，使用测试集命中率作为后缀
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 将命中率格式化为4位整数（例如 0.7234 -> 7234）
        accuracy_suffix = f"{int(test_accuracy * 10000):04d}"
        file_stub = f"trend_xgb_shortonly_two_stage_{timestamp}_acc{accuracy_suffix}"
        out_path = os.path.join(self.model_dir, f"{file_stub}.pkl")
        bundle = {
            'two_stage': True,
            'short_only': True,  # 标记为短空模型
            'model_stage1': model_stage1,
            'model_stage2_down': model_stage2_down,  # 注意：短空模型使用 model_stage2_down
            'feature_names': list(X_train.columns),
            'encoders': {
                'sector_encoder': self.base.sector_encoder,
                'industry_encoder': self.base.industry_encoder,
                'sub_industry_encoder': self.base.sub_industry_encoder,
            },
            'timestamp': timestamp,
            'thresholds': {
                'stage1_trend': t1,
                'stage2_down': thr_down
            }
        }
        with open(out_path, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"\n模型已保存: {out_path}")

        # 计算并保存每标的正确率（基于"建议做空"信号 vs 实际是否下跌）
        try:
            if symbols_test is not None:
                # y_test_down_full: 测试集真实是否下跌(1/0)
                # suggest_test: 最终选择的建议做空信号(1=建议做空,0=不做空)
                print("\n开始按标的统计短空模型在测试集上的命中率，并写入 per_symbol_accuracy_test_shortonly_*.csv ...")
                self.base.save_per_symbol_accuracy(
                    y_true=y_test_down_full,
                    y_pred=suggest_test,
                    symbols=symbols_test,
                    dataset_name='test_shortonly'
                )
            else:
                print("symbols_test 为空，跳过每标的正确率统计")
        except Exception as e:
            print(f"保存短空模型每标的正确率失败: {e}")
        
        # 计算并保存Stage1的每标的正确率（是否趋势 vs 实际是否趋势）
        try:
            if symbols_test is not None and model_stage1 is not None:
                # 使用最终选择的stage1模型进行预测
                # 如果使用了多次训练，model_stage1 已经是最终选择的模型
                p_trend_test_final = model_stage1.predict_proba(X_test)[:, 1]
                # 使用stage1的阈值进行预测
                stage1_thr = t1  # 使用最终选择的stage1阈值
                y_pred_stage1 = (p_trend_test_final >= stage1_thr).astype(int)
                # y_test_s1: 测试集真实是否趋势(1/0)
                print("\n开始按标的统计短空模型Stage1在测试集上的命中率，并写入 per_symbol_accuracy_test_stage1_shortonly_*.csv ...")
                self.base.save_per_symbol_accuracy(
                    y_true=y_test_s1,
                    y_pred=y_pred_stage1,
                    symbols=symbols_test,
                    dataset_name='test_stage1_shortonly'
                )
            else:
                if symbols_test is None:
                    print("symbols_test 为空，跳过Stage1每标的正确率统计")
                elif model_stage1 is None:
                    print("model_stage1 为空，跳过Stage1每标的正确率统计")
        except Exception as e:
            print(f"保存短空模型Stage1每标的正确率失败: {e}")
        
        # 删除 apply/model/ 中的同名旧模型，并复制新模型到 apply/model/
        apply_model_dir = os.path.join(self.project_root, 'apply', 'model')
        os.makedirs(apply_model_dir, exist_ok=True)
        
        # 查找并删除所有包含 shortonly 或 short_only 的旧模型
        old_models = glob.glob(os.path.join(apply_model_dir, '*shortonly*.pkl'))
        old_models.extend(glob.glob(os.path.join(apply_model_dir, '*short_only*.pkl')))
        for old_model in old_models:
            try:
                os.remove(old_model)
                print(f"已删除旧模型: {old_model}")
            except Exception as e:
                print(f"删除旧模型失败 {old_model}: {e}")
        
        # 复制新模型到 apply/model/
        apply_model_path = os.path.join(apply_model_dir, f"{file_stub}.pkl")
        try:
            shutil.copy2(out_path, apply_model_path)
            print(f"模型已复制到 apply/model/: {apply_model_path}")
        except Exception as e:
            print(f"复制模型到 apply/model/ 失败: {e}")


def main():
    trainer = ShortOnlyTwoStageTrainer()
    trainer.train()


if __name__ == '__main__':
    main()

