#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train LightGBM on large parquet datasets (~2M+ rows) with sane defaults.
- Auto dtype downcast to save RAM
- Auto scale_pos_weight for imbalance
- Early stopping on AUC-PR (average_precision)
- Exports: metrics.json, feature_importance.csv/png, PR & ROC curves, confusion matrix
- Optional test set eval
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


TARGET_COL = "is_helpful"  # change if your target differs


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser("train_lightgbm_v2_full")
    ap.add_argument("--train", required=True, help="Path to train parquet file")
    ap.add_argument("--test", default=None, help="Optional path to test parquet file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio split from training file (ignored if you supply separate val file in future)")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--boost_rounds", type=int, default=3000)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=0.035)
    ap.add_argument("--num_leaves", type=int, default=128)
    ap.add_argument("--feature_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_freq", type=int, default=2)
    ap.add_argument("--min_child_samples", type=int, default=60)
    ap.add_argument("--max_bin", type=int, default=255)
    ap.add_argument("--bin_construct_sample_cnt", type=int, default=200000)
    ap.add_argument("--reg_alpha", type=float, default=0.1)
    ap.add_argument("--reg_lambda", type=float, default=0.2)
    ap.add_argument("--scale_pos_weight", default="auto", help="'auto' or a float value, e.g. 12.5")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for training if available")
    ap.add_argument("--positive_label", type=int, default=1, help="Value of positive class in TARGET_COL")
    ap.add_argument("--neg_label", type=int, default=0, help="Value of negative class in TARGET_COL")
    ap.add_argument("--feature_cols", default=None, help="CSV list of feature columns to use. If not set, auto-detect numeric columns minus target.")
    ap.add_argument("--drop_cols", default=None, help="CSV list of columns to drop before training.")
    ap.add_argument("--sample_rows", type=int, default=0, help="Optional row cap for debugging (0 = use all).")
    return ap.parse_args()


def to_efficient_dtypes(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    for c in numeric_cols:
        if c not in df.columns:
            continue
        dt = str(df[c].dtype)
        if dt.startswith("float"):
            df[c] = df[c].astype("float32")
        elif dt.startswith(("int", "uint")):
            # Keep as int32 to be safe with missing values -> later filled
            df[c] = df[c].astype("int32")
        elif dt == "bool":
            df[c] = df[c].astype("uint8")
        # others left as-is
    return df


def load_parquet(path: str, columns: Optional[List[str]] = None, sample_rows: int = 0) -> pd.DataFrame:
    log(f"Loading parquet: {path}")
    df = pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)
    if sample_rows and sample_rows > 0 and len(df) > sample_rows:
        log(f"Sampling first {sample_rows} rows for debug")
        df = df.iloc[:sample_rows].copy()
    return df


def detect_feature_columns(df: pd.DataFrame, target_col: str, specified_cols: Optional[List[str]]) -> List[str]:
    if specified_cols:
        return [c for c in specified_cols if c in df.columns and c != target_col]
    # By default: numeric columns except target
    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    feature_cols = [c for c in num_cols if c != target_col]
    return feature_cols


def apply_drops(df: pd.DataFrame, drop_cols: Optional[List[str]]) -> pd.DataFrame:
    if drop_cols:
        keep = [c for c in df.columns if c not in drop_cols]
        return df[keep].copy()
    return df


def compute_spw(y: np.ndarray, pos_label: int) -> float:
    pos = float((y == pos_label).sum())
    neg = float(len(y) - pos)
    return neg / max(pos, 1.0)


def make_lgb_dataset(X: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> lgb.Dataset:
    return lgb.Dataset(X[feature_names].values, label=y, feature_name=feature_names, free_raw_data=False)


def train_and_eval(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: List[str],
    params: dict,
    out_dir: str,
    boost_rounds: int,
    early_stopping_rounds: int,
) -> Tuple[lgb.Booster, dict]:
    train_data = make_lgb_dataset(X_train, y_train, feature_names)
    val_data = make_lgb_dataset(X_val, y_val, feature_names)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=200),
    ]

    log("Start training...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=boost_rounds,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Predictions on validation
    val_pred = model.predict(X_val[feature_names].values, num_iteration=model.best_iteration)
    metrics = compute_metrics(y_val, val_pred, pos_label=params.get("pos_label", 1))
    metrics["best_iteration"] = int(model.best_iteration)

    save_metrics(metrics, os.path.join(out_dir, "metrics_val.json"))
    save_feature_importance(model, feature_names, out_dir)
    plot_curves(y_val, val_pred, out_dir, prefix="val")
    return model, metrics


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, pos_label: int = 1) -> dict:
    y_true_bin = (y_true == pos_label).astype(int)
    ap = average_precision_score(y_true_bin, y_prob)
    try:
        roc = roc_auc_score(y_true_bin, y_prob)
    except Exception:
        roc = None
    # Default threshold = 0.5; user can tune later per PR curve
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])
    report = classification_report(y_true_bin, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    metrics = {
        "average_precision": ap,
        "roc_auc": roc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics


def save_metrics(metrics: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(f"Saved metrics → {path}")


def save_feature_importance(model: lgb.Booster, feature_names: List[str], out_dir: str) -> None:
    imp_gain = model.feature_importance(importance_type="gain")
    imp_split = model.feature_importance(importance_type="split")
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "gain": imp_gain,
        "split": imp_split,
    }).sort_values("gain", ascending=False)
    csv_path = os.path.join(out_dir, "feature_importance.csv")
    df_imp.to_csv(csv_path, index=False)
    log(f"Saved feature_importance → {csv_path}")

    # Plot top 40 by gain
    topn = min(40, len(df_imp))
    plt.figure(figsize=(10, max(4, topn * 0.25)))
    plt.barh(df_imp["feature"].head(topn)[::-1], df_imp["gain"].head(topn)[::-1])
    plt.xlabel("Gain")
    plt.title("Top Feature Importance (gain)")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    log(f"Saved feature_importance.png → {png_path}")


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, prefix: str = "val") -> None:
    y_true_bin = (y_true == 1).astype(int)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true_bin, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve ({prefix})")
    plt.tight_layout()
    pr_path = os.path.join(out_dir, f"{prefix}_pr_curve.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    log(f"Saved {prefix}_pr_curve.png → {pr_path}")

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC curve ({prefix})")
        plt.tight_layout()
        roc_path = os.path.join(out_dir, f"{prefix}_roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        log(f"Saved {prefix}_roc_curve.png → {roc_path}")
    except Exception as e:
        log(f"ROC curve skipped: {e}")


def main():
    args = parse_args()
    ensure_dir(args.out)

    # Load train parquet minimally (only needed columns if provided)
    cols_to_load = None
    if args.feature_cols:
        specified = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        cols_to_load = list(set(specified + [TARGET_COL]))
    train_df = load_parquet(args.train, columns=cols_to_load, sample_rows=args.sample_rows)

    # Optional drop columns
    drop_cols = [c.strip() for c in args.drop_cols.split(",")] if args.drop_cols else None
    train_df = apply_drops(train_df, drop_cols)

    if TARGET_COL not in train_df.columns:
        log(f"FATAL: target column '{TARGET_COL}' not found in training data.")
        sys.exit(2)

    # Detect features
    feature_cols = detect_feature_columns(train_df, TARGET_COL, specified_cols=[c.strip() for c in args.feature_cols.split(",")] if args.feature_cols else None)
    if not feature_cols:
        log("FATAL: No feature columns detected. Provide --feature_cols explicitly.")
        sys.exit(2)

    # Fill NA
    train_df[feature_cols] = train_df[feature_cols].fillna(0)

    # Downcast to save RAM
    train_df = to_efficient_dtypes(train_df, feature_cols)

    # Train/Val split
    y = train_df[TARGET_COL].values
    X = train_df[feature_cols]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.random_state, stratify=y
    )
    log(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Build params
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "min_child_samples": args.min_child_samples,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "bin_construct_sample_cnt": args.bin_construct_sample_cnt,
        "verbose": -1,
        "num_threads": args.threads,
        "pos_label": args.positive_label,
    }
    if args.gpu:
        params.update({"device": "gpu"})

    # scale_pos_weight
    if isinstance(args.scale_pos_weight, str) and args.scale_pos_weight == "auto":
        spw = compute_spw(y_train, pos_label=args.positive_label)
    else:
        spw = float(args.scale_pos_weight)
    params["scale_pos_weight"] = spw
    log(f"scale_pos_weight = {spw:.4f}")

    # Train & eval on validation
    model, val_metrics = train_and_eval(
        X_train, y_train, X_val, y_val, feature_cols, params, args.out, args.boost_rounds, args.early_stopping_rounds
    )

    # Save model (LightGBM text)
    model_path = os.path.join(args.out, "model.txt")
    model.save_model(model_path, num_iteration=model.best_iteration)
    log(f"Saved model → {model_path}")

    # Optional: evaluate on held-out test set
    if args.test:
        # Load only needed columns + target
        cols = list(set(feature_cols + [TARGET_COL]))
        test_df = load_parquet(args.test, columns=cols, sample_rows=args.sample_rows)
        test_df = apply_drops(test_df, drop_cols)
        test_df[feature_cols] = test_df[feature_cols].fillna(0)
        test_df = to_efficient_dtypes(test_df, feature_cols)

        y_test = test_df[TARGET_COL].values
        X_test = test_df[feature_cols]
        test_prob = model.predict(X_test.values, num_iteration=model.best_iteration)
        test_metrics = compute_metrics(y_test, test_prob, pos_label=args.positive_label)
        save_metrics(test_metrics, os.path.join(args.out, "metrics_test.json"))
        plot_curves(y_test, test_prob, args.out, prefix="test")
        log("Test evaluation done.")

    # Save a quick run manifest
    manifest = {
        "train_path": args.train,
        "test_path": args.test,
        "out_dir": args.out,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "val_ratio": args.val_ratio,
        "params": params,
        "best_iteration": int(model.best_iteration),
    }
    with open(os.path.join(args.out, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log("Run manifest saved. Done.")


if __name__ == "__main__":
    main()
