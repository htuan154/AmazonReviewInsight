# code_v2/models/predict_pipeline_v2.py
# Author: Lê Đăng Hoàng Tuấn (Infrastructure)
# Day 7: Prediction pipeline với NULL handling
#
# Cải tiến so với V1:
# - 100% coverage (không drop records do NULL)
# - Validate input data quality
# - Batch processing
#
# Usage:
#   python code_v2/models/predict_pipeline_v2.py \
#       --test data/test_v2.parquet \
#       --model output/lightgbm_v2/model.txt \
#       --features output/lightgbm_v2/metrics.json \
#       --out output/submission_v2.csv

import argparse
import json
import pandas as pd
import lightgbm as lgb
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Test parquet path")
    ap.add_argument("--model", required=True, help="Trained LightGBM model path")
    ap.add_argument("--features", required=True, help="Metrics JSON with feature list")
    ap.add_argument("--out", required=True, help="Output submission CSV")
    ap.add_argument("--batch_size", type=int, default=100000, help="Batch size for prediction")
    return ap.parse_args()

def validate_test_data(df, required_features):
    """Validate test data quality"""
    print("\n[INFO] Validating test data...")
    
    # Check missing features
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"[ERROR] Missing features: {missing_features}")
        raise ValueError(f"Test data missing {len(missing_features)} features")
    
    # Check NULL values in key features
    null_counts = df[required_features].isnull().sum()
    high_null_features = null_counts[null_counts > 0]
    
    if len(high_null_features) > 0:
        print("[WARNING] Features with NULL values:")
        for feat, count in high_null_features.items():
            pct = count / len(df) * 100
            print(f"  {feat}: {count:,} ({pct:.2f}%)")
        
        # Fill NULLs with defaults
        print("\n[INFO] Filling NULL values with defaults...")
        fill_values = {feat: 0.0 for feat in high_null_features.index}
        df[required_features] = df[required_features].fillna(fill_values)
        print("  ✓ NULLs filled")
    else:
        print("  ✓ No NULL values in features")
    
    # Check review_id
    if "review_id" not in df.columns:
        raise ValueError("Test data must have 'review_id' column")
    
    null_ids = df["review_id"].isnull().sum()
    if null_ids > 0:
        raise ValueError(f"Found {null_ids} NULL review_ids")
    
    print(f"  ✓ All {len(df):,} records have valid review_id")
    
    return df

def predict_batch(model, X, batch_size=100000):
    """Predict in batches to handle large datasets"""
    print(f"\n[INFO] Predicting in batches of {batch_size:,}...")
    
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    predictions = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_X = X.iloc[start_idx:end_idx]
        batch_pred = model.predict(batch_X)
        predictions.extend(batch_pred)
        
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"  Batch {i+1}/{n_batches} ({end_idx:,}/{n_samples:,} samples)")
    
    return predictions

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("PREDICTION PIPELINE V2 (100% COVERAGE)")
    print("="*80)
    print(f"Author: Lê Đăng Hoàng Tuấn")
    print("="*80 + "\n")
    
    # Load model
    print(f"[INFO] Loading model from {args.model}")
    model = lgb.Booster(model_file=args.model)
    print(f"  ✓ Model loaded (iteration {model.current_iteration()})")
    
    # Load feature list
    print(f"\n[INFO] Loading feature list from {args.features}")
    with open(args.features, "r") as f:
        metrics = json.load(f)
    
    required_features = metrics["features"]
    print(f"  ✓ {len(required_features)} features required")
    
    # Load test data
    print(f"\n[INFO] Loading test data from {args.test}")
    df_test = pd.read_parquet(args.test)
    print(f"  ✓ Loaded {len(df_test):,} test samples")
    
    # Validate data
    df_test = validate_test_data(df_test, required_features)
    
    # Prepare features
    X_test = df_test[required_features]
    review_ids = df_test["review_id"]
    
    # Predict
    y_pred_proba = predict_batch(model, X_test, args.batch_size)
    
    # Create submission
    submission = pd.DataFrame({
        "review_id": review_ids,
        "probability_helpful": y_pred_proba
    })
    
    # Validate submission
    print("\n[INFO] Validating submission...")
    
    # Check for duplicates
    n_duplicates = submission["review_id"].duplicated().sum()
    if n_duplicates > 0:
        print(f"[WARNING] Found {n_duplicates} duplicate review_ids")
    
    # Check probability range
    proba = submission["probability_helpful"]
    if proba.min() < 0 or proba.max() > 1:
        print(f"[WARNING] Probability out of range: [{proba.min():.4f}, {proba.max():.4f}]")
    
    # Statistics
    print(f"\n[INFO] Submission Statistics:")
    print(f"  Total predictions: {len(submission):,}")
    print(f"  Mean probability: {proba.mean():.4f}")
    print(f"  Std probability: {proba.std():.4f}")
    print(f"  Min probability: {proba.min():.4f}")
    print(f"  Max probability: {proba.max():.4f}")
    
    # Coverage
    coverage = len(submission) / len(df_test) * 100
    print(f"\n[INFO] Coverage: {coverage:.2f}% ({len(submission):,}/{len(df_test):,})")
    
    if coverage < 100:
        print(f"[WARNING] Missing {len(df_test) - len(submission):,} predictions!")
    else:
        print("  ✓ 100% coverage achieved!")
    
    # Save submission
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission.to_csv(out_path, index=False)
    print(f"\n[INFO] Submission saved to {out_path}")
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETED")
    print("="*80)
    
    # Compare với V1
    v1_coverage = 37.7  # From final report
    improvement = coverage - v1_coverage
    print(f"\nV1 Coverage: {v1_coverage:.1f}%")
    print(f"V2 Coverage: {coverage:.1f}%")
    print(f"Improvement: +{improvement:.1f}%")
    
    if coverage == 100:
        print("\n✓ Production-ready: 100% coverage!")

if __name__ == "__main__":
    main()
