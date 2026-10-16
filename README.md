# XGBoost Trend Classification

A Python research pipeline for collecting daily market data, engineering features, constructing trend labels, and evaluating XGBoost classifiers. It includes experiments with three-class, two-stage, and separate long/short models.

## Problem

Can daily price-derived features distinguish upward trends, downward trends, and non-trending periods? How do labeling choices, temporal splits, class imbalance, and decision thresholds affect the apparent quality of a classifier?

## Data

The acquisition code uses Interactive Brokers, with Yahoo Finance support also present in the source. The configured universe is in `config.py`. Raw prices, processed features, labels, prediction exports, logs, trained models, and generated images are excluded from this repository. Reproduction requires your own authorized data and, where applicable, a local IB Gateway/TWS session.

## Method

1. Collect and incrementally update historical daily bars.
2. Compute technical features and construct trend labels.
3. Process labels and split observations into training, validation, and test sets using the implemented temporal split routines.
4. Train XGBoost models with configurable class weighting/resampling and validation-based threshold selection.
5. Evaluate classification reports, confusion matrices, and per-symbol results; export probabilities for downstream research.
6. Explore separate strategy simulations using the model outputs.

Temporal ordering alone does not prove the absence of leakage. Retrospective trend labels, label shifts, feature generation, cross-symbol date alignment, and repeated threshold tuning all require explicit review before interpreting results as out-of-sample evidence. Classification accuracy is not a trading return.

## Result

The repository provides the end-to-end research implementation. Numerical model scores are not claimed here because no frozen dataset/model/evaluation bundle is distributed or newly reproduced for this release. Local model files and prediction records remain private. The project should be assessed by its pipeline design, evaluation code, and stated limitations rather than an unverified performance headline.

## What I learned

The definition and timing of a label can matter as much as the estimator. Class imbalance, per-symbol behavior, and threshold selection can make a single aggregate accuracy misleading. A useful research workflow needs to separate classification evaluation from transaction-cost-aware trading evaluation.

## Setup

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Review `config.py` for the symbol universe, broker connection, data paths, label definitions, and model settings. Dependencies are not a frozen environment lockfile.

## Usage

Individual pipeline entry points include:

```bash
python -m data_process.data_collector
python -m data_process.data_processor
python -m trend_model.labeler
python label_processor.py
python -m data_process.data_splitter
python -m trend_model.train_xgboost
```

Each stage needs the preceding stage's local output. The complete pipeline is `python main.py`; the current orchestration also performs visualization, trains separate long/short models, and applies the resulting models. It includes cleanup of data for symbols no longer in the configuration, so use a dedicated project data directory. `apply_model.py` requires locally trained artifacts and is not runnable from a fresh clone alone.

## Source guide

- `data_process/`: collection, feature engineering, temporal splitting, and plots.
- `trend_model/`: trend labeling and XGBoost training/evaluation.
- `label_processor.py`: downstream label transformation.
- `trade_simulation/`: downstream strategy simulation experiments.
- `apply_model.py`: local model inference and prediction exports.
- `main.py`: full workflow orchestration.
- `config.py`: research configuration and strategy parameters.

The README reflects the current directory layout. Older references to root-level `data_collector.py`, `annotate_predictions.py`, and `trade_policy.py` were removed because those paths are absent in this snapshot. Source syntax was checked; training, broker access, and live prediction were not run during publication preparation.
