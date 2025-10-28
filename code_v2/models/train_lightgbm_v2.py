# code_v2/models/train_lightgbm_v2.py
# Author: Võ Thị Diễm Thanh (Features & Models)
# Day 5-6: Train LightGBM với features V2 và NULL handling
#
# Cải tiến so với V1:
# - Sử dụng features V2 (NULL-safe)
# - Early stopping
# - Feature importance analysis
# - Better hyperparameters từ tuning V1
#
# Usage:
#   python code_v2/models/train_lightgbm_v2.py \
#       --train data/processed_v2/train.parquet \
#       --test data/processed_v2/test.parquet \
#       --feature_set v3 \
#       --out output/lightgbm_v2

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score, accuracy_score,
    precision_recall_curve, roc_curve
)
import lightgbm as lgb
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Train parquet path")
    ap.add_argument("--test", required=True, help="Test parquet path")
    ap.add_argument("--feature_set", default="v3", choices=["baseline", "v1", "v2", "v3", "full"])
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--sample", type=int, default=None, help="Sample size for faster training")
    return ap.parse_args()

def get_feature_columns(feature_set="v3"):
    """Define feature sets (matching metadata_features_v2.py)"""
    baseline_features = [
        "star_rating", "review_length", "review_length_log"
    ]
    
    v1_features = baseline_features + [
        "rating_deviation", "is_long_review",
        "user_review_count", "product_review_count"
    ]
    
    v2_features = v1_features + [
        "user_avg_rating", "user_helpful_ratio",
        "product_avg_rating", "product_helpful_ratio",
        "price", "price_log",
        "product_avg_rating_meta", "product_total_ratings"
    ]
    
    v3_features = v2_features + [
        "has_metadata", "has_price", "has_product_rating",
        "is_expensive", "user_consistency",
        "meta_review_rating_gap",
        "category_review_count", "is_popular_category",
        # Text features
        "text_length", "word_count", "sentence_count",
        "exclamation_count", "question_count",
        # Sentiment features
        "sentiment_compound", "sentiment_pos", "sentiment_neg",
        "sentiment_strength", "is_polarized",
        "sentiment_rating_alignment"
    ]
    
    full_features = v3_features + [
        "user_avg_review_length", "product_avg_review_length",
        "product_rating_stddev",
        "day_of_week", "is_weekend", "is_peak_hour",
        "is_holiday_season", "quarter",
        "rating_x_length", "user_product_activity",
        "price_x_rating", "user_experience_score",
        "category_price_percentile", "category_rating_percentile",
        "avg_word_length", "uppercase_ratio"
    ]
    
    feature_map = {
        "baseline": baseline_features,
        "v1": v1_features,
        "v2": v2_features,
        "v3": v3_features,
        "full": full_features
    }
    
    return feature_map.get(feature_set, v3_features)

def load_data(path, feature_cols, sample_size=None):
    """Load parquet và select features"""
    print(f"[INFO] Loading data from {path}")
    df = pd.read_parquet(path)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"[INFO] Sampled {len(df):,} rows")
    
    # Check available columns
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]
    
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
    
    X = df[available_features]
    y = df["is_helpful"]
    
    print(f"[INFO] Features: {len(available_features)}")
    print(f"[INFO] Samples: {len(X):,}")
    print(f"[INFO] Positive rate: {y.mean():.2%}")
    
    return X, y, available_features

def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM with early stopping"""
    
    if params is None:
        # Best params từ tuning V1 (Day 7)
        params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': 10.0,  # Adjust based on class imbalance
            'max_depth': -1,
            'verbose': -1
        }
    
    print("\n[INFO] Training LightGBM...")
    print(f"  Params: {params}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train with early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    print(f"\n[INFO] Best iteration: {model.best_iteration}")
    print(f"[INFO] Best score: {model.best_score['val']['average_precision']:.4f}")
    
    return model

def evaluate_model(model, X, y, name="Test"):
    """Evaluate model and return metrics"""
    print(f"\n[INFO] Evaluating on {name} set...")
    
    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        "auc_pr": average_precision_score(y, y_pred_proba),
        "auc_roc": roc_auc_score(y, y_pred_proba),
        "accuracy": accuracy_score(y, y_pred)
    }
    
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics, y_pred_proba

def plot_feature_importance(model, feature_names, out_dir):
    """Plot and save feature importance"""
    importance = model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n[INFO] Top 20 Feature Importances:")
    print(feature_imp.head(20).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(feature_imp['feature'].head(20), feature_imp['importance'].head(20))
    plt.xlabel('Importance (Gain)')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=300)
    print(f"[INFO] Feature importance plot saved to {out_dir}/feature_importance.png")
    
    # Save to CSV
    feature_imp.to_csv(f"{out_dir}/feature_importance.csv", index=False)
    
    return feature_imp

def plot_pr_curve(y_true, y_pred_proba, out_dir, name="Test"):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pr_curve_{name.lower()}.png", dpi=300)
    print(f"[INFO] PR curve saved to {out_dir}/pr_curve_{name.lower()}.png")

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("TRAIN LIGHTGBM V2 (NULL-SAFE)")
    print("="*80)
    print(f"Author: Võ Thị Diễm Thanh")
    print(f"Feature set: {args.feature_set}")
    print("="*80 + "\n")
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature columns
    feature_cols = get_feature_columns(args.feature_set)
    print(f"[INFO] Using {len(feature_cols)} features")
    
    # Load data
    X_train, y_train, train_features = load_data(args.train, feature_cols, args.sample)
    X_test, y_test, test_features = load_data(args.test, feature_cols)
    
    # Use available features only
    common_features = list(set(train_features) & set(test_features))
    print(f"[INFO] Common features: {len(common_features)}")
    
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    # Train model
    model = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # Evaluate
    train_metrics, _ = evaluate_model(model, X_train, y_train, "Train")
    test_metrics, y_pred_proba = evaluate_model(model, X_test, y_test, "Test")
    
    # Feature importance
    feature_imp = plot_feature_importance(model, common_features, out_dir)
    
    # Plot PR curve
    plot_pr_curve(y_test, y_pred_proba, out_dir, "Test")
    
    # Save model
    model_path = out_dir / "model.txt"
    model.save_model(str(model_path))
    print(f"\n[INFO] Model saved to {model_path}")
    
    # Save metrics
    results = {
        "model": "LightGBM V2",
        "author": "Võ Thị Diễm Thanh",
        "feature_set": args.feature_set,
        "n_features": len(common_features),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_iteration": int(model.best_iteration),
        "features": common_features
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Metrics saved to {out_dir}/metrics.json")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"\nTest AUC-PR: {test_metrics['auc_pr']:.4f}")
    print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Compare với V1
    v1_auc_pr = 0.7180  # From final report
    improvement = (test_metrics['auc_pr'] - v1_auc_pr) / v1_auc_pr * 100
    print(f"\nImprovement vs V1: {improvement:+.2f}%")
    
    if test_metrics['auc_pr'] >= 0.72:
        print("✓ Target achieved (≥ 0.72)!")
    else:
        print(f"Target: 0.72 (need +{(0.72 - test_metrics['auc_pr']):.4f})")

if __name__ == "__main__":
    main()
