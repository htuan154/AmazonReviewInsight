#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# TRAIN LIGHTGBM V2 - SEMI-SUPERVISED WITH PSEUDO-LABELING (IMPROVED)
# ============================================================================
# Spark 3.4–3.5 + SynapseML 1.0.7
#
# V2.1 IMPROVEMENTS:
# -> Tích hợp evaluation_v2.py cho comprehensive metrics
# -> Sửa công thức class weighting theo sklearn balanced style
# -> Thêm threshold optimization để tối ưu precision/recall
# -> Fix confusion matrix calculation (đúng công thức)
# -> Save evaluation plots (PR/ROC curves, confusion matrix)
#
# Mục đích: Train LightGBM classifier với semi-supervised learning
# -> Sử dụng pseudo-labeling để tận dụng unlabeled test data
# -> Tự động xử lý class imbalance với adaptive weighting
# -> Hyperparameter tuning với cross-validation
# -> Target AUC-PR: 0.75-0.80 (realistic cho hidden test)
#
# ┌────────────────────────────────────────────────────────────────────┐
# │ KEY FEATURES                                                       │
# ├────────────────────────────────────────────────────────────────────┤
# │ 1. Consistent Featurization: numFeatures phải match train/test    │
# │ 2. Auto Class Weighting: --posWeight auto (N_neg/N_pos)           │
# │ 3. Stratified Split: Maintain class ratio trong train/val         │
# │ 4. Pseudo-Labeling: Iterative training với high-confidence samples│
# │ 5. Hyperparameter Tuning: 3-fold CV grid search (optional)        │
# │ 6. Comprehensive Logging: Schema, columns, params, metrics        │
# └────────────────────────────────────────────────────────────────────┘
#
# Pipeline Flow:
# 1. Load train/test Parquet -> Validate schema & dimensions
# 2. Drop leaky columns (helpful_votes, helpful_ratio, v.v.)
# 3. Stratified train/val split (90/10) -> Maintain class balance
# 4. Compute class weights -> Handle imbalance (auto or manual)
# 5. [Optional] Hyperparameter tuning -> 3-fold CV grid search
# 6. Train LightGBM -> Early stopping on validation set
# 7. [Optional] Pseudo-labeling -> Predict on test -> Select confident samples
# 8. Save model + metadata + logs -> HDFS/local
#
# Usage Example (Basic):
# spark-submit code_v2/models/train_lightgbm_spark_v2.py \
#   --train hdfs:///amazon/features_train_v2.parquet \
#   --test hdfs:///amazon/features_test_v2.parquet \
#   --out hdfs:///amazon/models/lightgbm_v2 \
#   --posWeight auto \
#   --save_schema_log
#
# Usage Example (Advanced with Tuning):
# spark-submit code_v2/models/train_lightgbm_spark_v2.py \
#   --train hdfs:///amazon/features_train_v2.parquet \
#   --out hdfs:///amazon/models/lightgbm_v2_tuned \
#   --auto_tune \
#   --tune_preset thorough \
#   --pseudo_rounds 2 \
#   --pseudo_min_prob 0.9 \
#   --save_schema_log
#
# ============================================================================

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession, functions as F, types as T, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import VectorUDT, SparseVector, DenseVector
from synapse.ml.lightgbm import LightGBMClassifier

# ============================================================================
# IMPORT EVALUATION UTILITIES (V2.1 NEW)
# ============================================================================
try:
    import sys
    # Dynamic path resolution
    sys.path.append(str(Path(__file__).parent.parent / "utils"))
    from evaluation_v2 import (
        calculate_metrics,
        find_optimal_threshold,
        plot_confusion_matrix,
        plot_pr_roc_curves,
        plot_threshold_analysis,
        print_classification_report
    )
    EVALUATION_V2_AVAILABLE = True
    print("[INFO] evaluation_v2.py imported successfully")
except ImportError as e:
    print(f"[WARN] Could not import evaluation_v2.py: {e}")
    print("[WARN] Will use basic evaluation only")
    EVALUATION_V2_AVAILABLE = False

import numpy as np

# ============================================================================
# FUNCTION: parse_args()
# ============================================================================
# Mục đích: Parse command-line arguments để configure training pipeline
#
# Argument Groups:
# ┌──────────────────────┬──────────────────────────────────────────────────┐
# │ Group                │ Arguments                                        │
# ├──────────────────────┼──────────────────────────────────────────────────┤
# │ IO                   │ --train, --test, --out                           │
# │ Columns              │ --id_col, --label_col, --features_col            │
# │ Feature Validation   │ --numFeatures (expected dimension)               │
# │ Sampling             │ --limit_train (for quick testing)                │
# │ Class Imbalance      │ --posWeight (auto/manual)                        │
# │ LightGBM Params      │ --numLeaves, --learningRate, --numIterations...  │
# │ Auto-Tuning          │ --auto_tune, --tune_preset (quick/thorough)      │
# │ Validation           │ --valFrac, --seed, --target_aucpr_min/max        │
# │ Pseudo-Labeling      │ --pseudo_rounds, --pseudo_min_prob...            │
# │ Logging              │ --save_schema_log, --force, --label_method       │
# └──────────────────────┴──────────────────────────────────────────────────┘
#
# Optimized Hyperparameters (from V1 Day 7 Best tuning):
# - numLeaves: 50 (lower = less overfit)
# - learningRate: 0.05 (slower but more stable)
# - minDataInLeaf: 50 (higher = less overfit)
# - featureFraction: 0.8 (random feature subset per tree)
# - baggingFraction: 0.8 (random sample subset per tree)
#
# Auto-Tuning Options:
# - quick: 9 combinations (3x3 grid) -> ~5-10 minutes
# - thorough: 27 combinations (3x3x3 grid) -> ~20-30 minutes
#
# Pseudo-Labeling Flow:
# 1. Train on labeled data -> Get model
# 2. Predict on unlabeled test -> Get probabilities
# 3. Select high-confidence predictions (prob >= 0.9)
# 4. Add pseudo-labels to training set (low weight = 0.3)
# 5. Retrain -> Repeat for N rounds
#
# Use case: Flexible configuration cho training experiments
def parse_args():
    p = argparse.ArgumentParser(description="Train LightGBM with semi-supervised pseudo-labeling.")
    # IO
    p.add_argument("--train", required=True, help="Path to TRAIN features parquet")
    p.add_argument("--test", default=None, help="Path to TEST features parquet (optional for pseudo-labeling)")
    p.add_argument("--out", required=True, help="Model output directory")
    
    # Columns
    p.add_argument("--id_col", default="review_id", help="ID column name")
    p.add_argument("--label_col", default="is_helpful", help="Binary label column (0/1)")
    p.add_argument("--features_col", default="features", help="Vector column for features")
    
    # Feature consistency
    p.add_argument("--numFeatures", type=int, default=None, 
                   help="Expected feature dimension (for validation)")
    
    # Sampling
    p.add_argument("--limit_train", type=int, default=None, 
                   help="Limit training samples for quick testing")
    
    # Class imbalance
    p.add_argument("--posWeight", default="auto", 
                   help="Positive class weight: 'auto' (N_neg/N_pos), or float value")
    
    # LightGBM hyperparameters (V2.1 OPTIMIZED for better generalization)
    p.add_argument("--numLeaves", type=int, default=50,
                   help="Max tree leaves (lower = less overfit, default 50 from V1 Best)")
    p.add_argument("--learningRate", type=float, default=0.05,
                   help="Learning rate (default 0.05 from V1 Best tuning)")
    p.add_argument("--numIterations", type=int, default=1500,
                   help="Number of boosting iterations (V2.1: increased from 500 to 1500)")
    p.add_argument("--earlyStoppingRound", type=int, default=100,
                   help="Early stopping patience (V2.1: increased from 50 to 100)")
    p.add_argument("--featureFraction", type=float, default=0.8,
                   help="Feature sampling ratio per tree (0.8 = 80% features)")
    p.add_argument("--baggingFraction", type=float, default=0.8,
                   help="Data sampling ratio per tree (0.8 = 80% samples)")
    p.add_argument("--minDataInLeaf", type=int, default=100,
                   help="Min samples per leaf (V2.1: increased from 50 to 100 for less overfit)")
    p.add_argument("--maxDepth", type=int, default=12,
                   help="Max tree depth (V2.1: changed from -1 to 12 to prevent overfitting)")
    p.add_argument("--lambdaL1", type=float, default=0.0,
                   help="L1 regularization (prevents overfitting)")
    p.add_argument("--lambdaL2", type=float, default=0.0,
                   help="L2 regularization (prevents overfitting)")
    
    # Auto-tuning (CRITICAL for hidden test performance)
    p.add_argument("--auto_tune", action="store_true",
                   help="Enable hyperparameter tuning (3-fold CV grid search)")
    p.add_argument("--tune_preset", default="quick", choices=["quick", "thorough"],
                   help="Tuning preset: quick (9 combos), thorough (27 combos)")
    
    # Validation
    p.add_argument("--valFrac", type=float, default=0.1, help="Validation split fraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_aucpr_min", type=float, default=0.80, 
                   help="Target minimum AUC-PR for early stopping")
    p.add_argument("--target_aucpr_max", type=float, default=0.85, 
                   help="Target maximum AUC-PR for early stopping")
    
    # Pseudo-labeling (semi-supervised)
    p.add_argument("--pseudo_rounds", type=int, default=0, 
                   help="Number of pseudo-labeling iterations")
    p.add_argument("--pseudo_min_prob", type=float, default=0.9, 
                   help="Minimum probability threshold for pseudo-labeling")
    p.add_argument("--pseudo_top_pct", type=float, default=0.1, 
                   help="Top percentage of confident predictions to pseudo-label")
    p.add_argument("--pseudo_weight", type=float, default=0.3, 
                   help="Weight for pseudo-labeled samples")
    
    # Logging
    p.add_argument("--save_schema_log", action="store_true", 
                   help="Save schema, columns, and params to log files")
    p.add_argument("--force", action="store_true", 
                   help="Force training even if numFeatures mismatch")
    p.add_argument("--label_method", default="heuristic", choices=["heuristic", "clustering"],
                   help="Method to generate synthetic labels when label column is missing")
    
    return p.parse_args()


# ============================================================================
# LEAKY COLUMNS - DANH SÁCH CÁC CỘT GÂY LABEL LEAKAGE
# ============================================================================
# Các cột này chứa thông tin trực tiếp từ label (helpful_votes > 0)
# -> PHẢI XÓA KHỎI features trước khi train
#
# ┌────────────────────────┬────────────────────────────────────────────────┐
# │ Column                 │ Why Leakage?                                   │
# ├────────────────────────┼────────────────────────────────────────────────┤
# │ helpful_votes          │ Raw vote count -> trực tiếp từ label           │
# │ total_votes            │ Total votes -> có thể infer helpful_votes      │
# │ helpful_ratio          │ helpful_votes / total_votes -> trực tiếp       │
# │ vote_ratio             │ Tương tự helpful_ratio                         │
# │ is_helpful_times_len   │ is_helpful * length -> chứa label              │
# │ helpfulness_x_length   │ Tương tự trên                                  │
# │ label_ratio            │ Aggregate label statistics                     │
# │ probability_helpful    │ Pre-computed probability -> leakage             │
# │ helpful                │ Alternative name cho label                     │
# │ target_helpful         │ Alternative name cho label                     │
# └────────────────────────┴────────────────────────────────────────────────┘
#
# Use case: drop_leaky_columns() sử dụng set này để filter
LEAKY_COLS = {
    "helpful_votes", "total_votes", "helpful_ratio", "vote_ratio",
    "is_helpful_times_len", "helpfulness_x_length", "label_ratio",
    "probability_helpful", "helpful", "target_helpful"
}


# ============================================================================
# FUNCTION: drop_leaky_columns()
# ============================================================================
# Mục đích: Xóa các cột gây label leakage khỏi DataFrame
#
# Tham số:
# - df: DataFrame cần làm sạch
# - features_col: Tên cột features vector (KHÔNG xóa)
# - label_col: Tên cột label (KHÔNG xóa)
#
# Logic:
# 1. Tìm tất cả cột trong LEAKY_COLS tồn tại trong df.columns
# 2. Loại trừ features_col và label_col (cần giữ lại)
# 3. Drop từng cột leaky
#
# ┌────────────────────────────────────────────────────────────────────┐
# │ WHY cần xóa leaky columns?                                         │
# │ - Model học từ helpful_votes -> AUC-PR = 0.99 (quá cao, unrealistic)│
# │ - Hidden test không có helpful_votes -> model fail                  │
# │ - Chỉ được dùng features tính từ review text/metadata             │
# └────────────────────────────────────────────────────────────────────┘
#
# Output: DataFrame đã xóa leaky columns
#
# Use case: Gọi trước khi train để đảm bảo không có leakage
def drop_leaky_columns(df, features_col, label_col):
    """Drop columns that might leak the ground-truth label."""
    # Lấy tập hợp tất cả tên cột trong DataFrame
    cols = set(df.columns)
    
    # Tìm các cột leaky: có trong LEAKY_COLS VÀ có trong df.columns
    # NHƯNG không phải features_col hoặc label_col (cần giữ lại)
    bad = [c for c in LEAKY_COLS if c in cols and c not in {features_col, label_col}]
    
    # Nếu tìm thấy cột leaky -> in warning và xóa từng cột
    if bad:
        print(f"[WARN] Dropping potential leaky columns: {bad}")
        for c in bad:
            df = df.drop(c)  # Xóa cột leaky
    
    return df  # Return DataFrame đã sạch


# ============================================================================
# FUNCTION: get_vector_size()
# ============================================================================
# Mục đích: Lấy dimension (số chiều) của feature vector
#
# Tham số:
# - df: DataFrame chứa features
# - features_col: Tên cột vector
#
# Logic:
# 1. Lấy sample row đầu tiên
# 2. Extract vector từ features_col
# 3. Kiểm tra kiểu: SparseVector/DenseVector -> .size
# 4. Return dimension
#
# Ví dụ:
# df có cột "features" với vector [0.1, 0.2, ..., 0.9] (100 dims)
# -> get_vector_size(df, "features") = 100
#
# Use case: Validate dimension consistency giữa train/test
def get_vector_size(df, features_col):
    """Extract the dimension of the feature vector."""
    # Lấy 1 row mẫu từ DataFrame
    sample = df.select(features_col).first()
    
    # Nếu DataFrame rỗng -> raise error
    if sample is None:
        raise RuntimeError(f"Cannot determine vector size: no data in '{features_col}'")
    
    # Lấy vector từ row đầu tiên (index 0)
    vec = sample[0]
    
    # Kiểm tra kiểu vector và lấy size
    if isinstance(vec, (SparseVector, DenseVector)):
        return vec.size  # Spark ML vector có attribute .size
    elif hasattr(vec, 'size'):
        return vec.size  # Generic vector với attribute size
    else:
        # Vector không có .size -> raise error
        raise RuntimeError(f"Unknown vector type: {type(vec)}")


def validate_feature_dimension(df, features_col, expected_dim, force=False):
    """Validate that feature vector dimension matches expected."""
    actual_dim = get_vector_size(df, features_col)
    if expected_dim and actual_dim != expected_dim:
        msg = f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}"
        if force:
            print(f"[WARN] {msg} (--force enabled, continuing anyway)")
        else:
            raise RuntimeError(f"{msg}. Use --force to override.")
    return actual_dim


def ensure_id_string(df, id_col):
    """Ensure ID column is string type for CSV output compatibility."""
    if id_col not in df.columns:
        raise RuntimeError(f"ID column '{id_col}' not found in DataFrame")
    return df.withColumn(id_col, F.col(id_col).cast(T.StringType()))


def generate_synthetic_labels(df, label_col, method='heuristic', seed=42):
    """
    Generate synthetic labels when ground truth is not available.
    
    Methods:
    - 'heuristic': Use rating + review length + sentiment as proxy
    - 'clustering': Use KMeans to find natural groupings
    
    Args:
        df: Input DataFrame with features
        label_col: Name for the generated label column
        method: 'heuristic' or 'clustering'
        seed: Random seed
    
    Returns:
        DataFrame with synthetic label column
    """
    print(f"[WARN] Label column '{label_col}' not found. Generating synthetic labels using '{method}' method...")
    
    # Check if DataFrame is not empty
    df_count = df.count()
    if df_count == 0:
        raise RuntimeError("Cannot generate synthetic labels: DataFrame is empty")
    
    print(f"[INFO] Generating labels for {df_count:,} samples")
    
    if method == 'heuristic':
        # Heuristic: High rating (4-5 stars) + decent length + positive sentiment = helpful (1)
        # Low rating (1-2 stars) or very short reviews = not helpful (0)
        
        conditions = []
        weights = []
        
        # Factor 1: Star rating (high ratings more likely helpful)
        # Scale to [0,1] and coalesce missing to 0.0
        if 'star_rating' in df.columns:
            conditions.append(F.coalesce(F.col('star_rating') / F.lit(5.0), F.lit(0.0)))
            weights.append(0.3)
        
        # Factor 2: Review length (longer reviews more helpful)
        if 'review_length_log' in df.columns:
            conditions.append(F.coalesce(F.col('review_length_log'), F.lit(0.0)))
            weights.append(0.2)
        elif 'review_length' in df.columns:
            conditions.append(F.coalesce(F.log1p(F.col('review_length').cast('double')), F.lit(0.0)))
            weights.append(0.2)
        
        # Factor 3: Sentiment alignment (positive reviews for high ratings)
        if 'sentiment_rating_alignment' in df.columns:
            conditions.append(F.coalesce(F.col('sentiment_rating_alignment').cast('double'), F.lit(0.0)))
            weights.append(0.2)
        elif 'sentiment_compound' in df.columns:
            conditions.append(F.coalesce((F.col('sentiment_compound') + 1.0) / 2.0, F.lit(0.0)))  # Normalize to [0,1]
            weights.append(0.2)
        elif 'sent_score' in df.columns:
            # Fallback: use sent_score (-1..1) -> [0,1]
            conditions.append(F.coalesce((F.col('sent_score') + 1.0) / 2.0, F.lit(0.0)))
            weights.append(0.2)
        
        # Factor 4: Word count (more detailed = more helpful)
        if 'word_count' in df.columns:
            conditions.append(F.coalesce(F.log1p(F.col('word_count').cast('double')), F.lit(0.0)))
            weights.append(0.15)
        
        # Factor 5: User helpfulness history
        if 'user_helpful_ratio' in df.columns:
            conditions.append(F.coalesce(F.col('user_helpful_ratio'), F.lit(0.0)))
            weights.append(0.15)
        
        # Compute weighted score
        if not conditions:
            print(f"[ERROR] No feature columns found for synthetic label generation")
            print(f"[ERROR] Available columns in DataFrame: {df.columns}")
            raise RuntimeError(
                "Not enough features to generate synthetic labels. "
                "Required columns: star_rating, review_length/review_length_log, "
                "sentiment_compound/sentiment_rating_alignment, word_count, user_helpful_ratio"
            )
        
        print(f"[INFO] Found {len(conditions)} features for synthetic label generation")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Compute score
        score_expr = sum(cond * w for cond, w in zip(conditions, weights))
        df = df.withColumn('_helpful_score', score_expr)
        
        # Check if we have valid scores
        score_count = df.filter(F.col('_helpful_score').isNotNull()).count()
        if score_count == 0:
            print(f"[ERROR] No valid scores computed. Check if required columns exist.")
            print(f"[ERROR] Available columns: {df.columns}")
            raise RuntimeError("Cannot generate synthetic labels: no valid feature scores")
        
        # Get median as threshold with safety check
        quantiles = df.approxQuantile('_helpful_score', [0.5], 0.01)
        if not quantiles or len(quantiles) == 0:
            # Fallback: use mean if quantile fails
            print(f"[WARN] Could not compute median, using mean as threshold")
            median_score = df.select(F.avg('_helpful_score')).collect()[0][0]
            if median_score is None:
                # Ultimate fallback: use 0.5 as neutral threshold
                print(f"[WARN] Could not compute mean, using 0.5 as threshold")
                median_score = 0.5
        else:
            median_score = quantiles[0]
        
        # Assign labels: above median = helpful (1), below = not helpful (0)
        df = df.withColumn(label_col, 
                          F.when(F.col('_helpful_score') >= median_score, 1).otherwise(0))
        df = df.drop('_helpful_score')
        
        # Get class distribution
        class_counts = df.groupBy(label_col).count().collect()
        dist = {row[label_col]: row['count'] for row in class_counts}
        
        print(f"[INFO] Generated synthetic labels using heuristic method:")
        print(f"       - Threshold: {median_score:.4f}")
        print(f"       - Class 0 (not helpful): {dist.get(0, 0):,} ({dist.get(0, 0)/df_count*100:.1f}%)")
        print(f"       - Class 1 (helpful): {dist.get(1, 0):,} ({dist.get(1, 0)/df_count*100:.1f}%)")
        print(f"       - Features used: {len(conditions)} features")
        print(f"       - Total samples: {df_count:,}")
        
    elif method == 'clustering':
        # Use KMeans clustering on available features
        from pyspark.ml.clustering import KMeans
        
        # Check if features column exists
        if 'features' not in df.columns:
            raise RuntimeError("Features column required for clustering method")
        
        print(f"[INFO] Running KMeans clustering (k=2) to generate synthetic labels...")
        
        kmeans = KMeans(featuresCol='features', k=2, seed=seed, maxIter=20)
        model = kmeans.fit(df)
        df = model.transform(df)
        
        # Use cluster assignment as label
        df = df.withColumn(label_col, F.col('prediction').cast(T.IntegerType()))
        df = df.drop('prediction')
        
        # Get class distribution
        class_counts = df.groupBy(label_col).count().collect()
        dist = {row[label_col]: row['count'] for row in class_counts}
        
        print(f"[INFO] Generated synthetic labels using KMeans clustering:")
        print(f"       - Cluster 0: {dist.get(0, 0):,} samples")
        print(f"       - Cluster 1: {dist.get(1, 0):,} samples")
        print(f"       - Centers: {model.clusterCenters()}")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'heuristic' or 'clustering'")
    
    return df


def print_feature_summary(df, features_col, stage=""):
    """Print summary of features for debugging."""
    try:
        vec_size = get_vector_size(df, features_col)
        sample_count = df.count()
        
        print(f"\n{'='*60}")
        print(f"FEATURE SUMMARY {stage}")
        print(f"{'='*60}")
        print(f"Total samples:      {sample_count:,}")
        print(f"Feature dimension:  {vec_size:,}")
        print(f"Columns in schema:  {len(df.columns)}")
        
        # Show first few column names
        non_feature_cols = [c for c in df.columns if c != features_col][:10]
        print(f"Available columns:  {', '.join(non_feature_cols)}")
        if len(df.columns) > 11:
            print(f"                    ... and {len(df.columns) - 11} more")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"[WARN] Could not print feature summary: {e}")


def assemble_features(df, features_col, label_col=None, exclude_cols=None):
    """
    Automatically assemble features vector from numeric columns.
    
    Args:
        df: Input DataFrame
        features_col: Name for output features column
        label_col: Label column to exclude (optional)
        exclude_cols: Additional columns to exclude (optional)
    
    Returns:
        DataFrame with features vector column
    """
    if features_col in df.columns:
        print(f"[INFO] Features column '{features_col}' already exists, skipping assembly")
        return df
    
    # Get all numeric columns - check against PySpark type classes
    from pyspark.sql.types import NumericType
    numeric_cols = [field.name for field in df.schema.fields 
                   if isinstance(field.dataType, NumericType)]
    
    # Exclude label and other specified columns
    exclude_set = set()
    if label_col:
        exclude_set.add(label_col)
    if exclude_cols:
        exclude_set.update(exclude_cols)
    
    # Also exclude ID-like columns and text columns
    exclude_patterns = ['_id', 'user_id', 'product_id', 'review_id', 'text', 'cleaned']
    for col in df.columns:
        for pattern in exclude_patterns:
            if pattern in col.lower():
                exclude_set.add(col)
                break
    
    feature_cols = [c for c in numeric_cols if c not in exclude_set]
    
    if not feature_cols:
        raise RuntimeError(
            f"No numeric columns found for feature assembly. "
            f"Available columns: {df.columns}"
        )
    
    print(f"[INFO] Assembling {len(feature_cols)} numeric columns into '{features_col}' vector:")
    print(f"       {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=features_col,
        handleInvalid='skip'  # Skip rows with invalid values
    )
    
    df_with_features = assembler.transform(df)
    
    # Store feature column names as metadata for later reference
    return df_with_features, feature_cols


def stratified_train_val_split(df, label_col, val_frac=0.1, seed=42):
    """Stratified train/val split using sampleBy."""
    df = df.withColumn("__label_int__", F.col(label_col).cast("int"))
    fractions = {0: val_frac, 1: val_frac}
    df_with_id = df.withColumn("__uid__", F.monotonically_increasing_id())
    val_df = df_with_id.sampleBy("__label_int__", fractions=fractions, seed=seed)
    train_df = df_with_id.join(val_df.select("__uid__"), on="__uid__", how="left_anti")
    return (
        train_df.drop("__uid__", "__label_int__"),
        val_df.drop("__uid__", "__label_int__")
    )


def stratified_kfold_split(df, label_col, n_folds=3, seed=42):
    """
    Stratified K-Fold split for cross-validation.
    Returns list of (train_fold, val_fold) tuples.
    """
    print(f"[CV] Creating {n_folds}-fold stratified split...")
    
    # Add fold assignment column
    df = df.withColumn("__label_int__", F.col(label_col).cast("int"))
    
    # Stratified assignment: assign fold ID proportionally within each class
    window_pos = Window.partitionBy("__label_int__").orderBy(F.rand(seed))
    df_with_fold = df.withColumn("__row_num__", F.row_number().over(window_pos))
    df_with_fold = df_with_fold.withColumn("__fold__", 
                                           (F.col("__row_num__") % n_folds).cast("int"))
    
    folds = []
    for fold_idx in range(n_folds):
        val_fold = df_with_fold.filter(F.col("__fold__") == fold_idx)
        train_fold = df_with_fold.filter(F.col("__fold__") != fold_idx)
        
        folds.append((
            train_fold.drop("__label_int__", "__row_num__", "__fold__"),
            val_fold.drop("__label_int__", "__row_num__", "__fold__")
        ))
    
    return folds


# ============================================================================
# FUNCTION: hyperparameter_tuning()
# ============================================================================
# Mục đích: Tự động tìm best hyperparameters bằng grid search + 3-fold CV
#
# Tham số:
# - train_df: Training data
# - label_col, features_col: Column names
# - args: Command-line arguments
# - preset: "quick" (9 combos) hoặc "thorough" (27 combos)
#
# Grid Search Presets:
# ┌───────────┬─────────────────────────────────────────────────────────┐
# │ Preset    │ Search Space                                            │
# ├───────────┼─────────────────────────────────────────────────────────┤
# │ quick     │ numLeaves: [31, 50, 100]                                │
# │ (9 combos)│ learningRate: [0.05, 0.1, 0.15]                         │
# │           │ -> 3x3 = 9 combinations                                  │
# │           │ -> ~5-10 phút (với 3-fold CV = 27 runs)                 │
# ├───────────┼─────────────────────────────────────────────────────────┤
# │ thorough  │ numLeaves: [31, 50, 100]                                │
# │(27 combos)│ learningRate: [0.03, 0.05, 0.1]                         │
# │           │ minDataInLeaf: [20, 50, 100]                            │
# │           │ -> 3x3x3 = 27 combinations                               │
# │           │ -> ~20-30 phút (với 3-fold CV = 81 runs)                │
# └───────────┴─────────────────────────────────────────────────────────┘
#
# Cross-Validation Flow:
# 1. Stratified 3-fold split -> Giữ class ratio trong mỗi fold
# 2. For each param combination:
#    - Train on 2 folds -> Validate on 1 fold -> Get AUC-PR
#    - Rotate folds -> Train 3 times -> Get 3 AUC-PR scores
#    - Compute mean ± std of 3 scores
# 3. Select params với highest mean AUC-PR
#
# ┌────────────────────────────────────────────────────────────────────┐
# │ WHY grid search?                                                   │
# │ - V1 Best params: numLeaves=50, learningRate=0.05, minData=50     │
# │ - Nhưng mỗi dataset khác nhau -> cần tune lại                      │
# │ - Grid search đảm bảo tìm được best params cho dataset hiện tại   │
# │ - 3-fold CV đảm bảo không overfit lên 1 validation split          │
# └────────────────────────────────────────────────────────────────────┘
#
# Output:
# - best_params: Dict với best hyperparameters
# - tuning_results: List chứa kết quả của tất cả combinations
#
# Use case: --auto_tune --tune_preset quick/thorough
def hyperparameter_tuning(train_df, label_col, features_col, args, preset="quick"):
    """
    Grid search for hyperparameter tuning with 3-fold CV.
    
    Presets:
    - quick: 9 combinations (from V1 Day 7 report)
    - thorough: 27 combinations (extended search)
    
    Returns: (best_params, tuning_results)
    """
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING - {preset.upper()} PRESET")
    print(f"{'='*80}\n")
    
    # Define search grid based on V1 Best tuning results
    if preset == "quick":
        # Top performers from V1 Day 7 report
        param_grid = {
            "numLeaves": [31, 50, 100],
            "learningRate": [0.05, 0.1, 0.15]
        }
        print("[INFO] Quick Grid: 9 combinations (3x3)")
    else:  # thorough
        param_grid = {
            "numLeaves": [31, 50, 100],
            "learningRate": [0.03, 0.05, 0.1],
            "minDataInLeaf": [20, 50, 100]
        }
        print("[INFO] Thorough Grid: 27 combinations (3x3x3)")
    
    # Generate all combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(product(*param_values))
    
    print(f"[INFO] Total combinations: {len(all_combos)}")
    print(f"[INFO] Using 3-fold CV -> {len(all_combos) * 3} training runs\n")
    
    # Create 3-fold stratified split
    folds = stratified_kfold_split(train_df, label_col, n_folds=3, seed=args.seed)
    
    # Track results
    tuning_results = []
    
    # Grid search
    for combo_idx, param_values_tuple in enumerate(all_combos, 1):
        params = dict(zip(param_names, param_values_tuple))
        
        # Build classifier with current params
        clf = LightGBMClassifier(
            objective="binary",
            labelCol=label_col,
            featuresCol=features_col,
            weightCol="weight",
            predictionCol="prediction",
            rawPredictionCol="rawPrediction",
            probabilityCol="probability",
            numLeaves=params.get("numLeaves", args.numLeaves),
            learningRate=params.get("learningRate", args.learningRate),
            minDataInLeaf=params.get("minDataInLeaf", args.minDataInLeaf),
            numIterations=args.numIterations,
            featureFraction=args.featureFraction,
            baggingFraction=args.baggingFraction,
            maxDepth=args.maxDepth,
            lambdaL1=args.lambdaL1,
            lambdaL2=args.lambdaL2,
            earlyStoppingRound=args.earlyStoppingRound,
            isUnbalance=True,
            seed=args.seed
        )
        
        # Cross-validation
        fold_aucprs = []
        for fold_idx, (train_fold, val_fold) in enumerate(folds, 1):
            # Add validation indicator
            train_fold = train_fold.withColumn("is_val", F.lit(False))
            val_fold = val_fold.withColumn("is_val", F.lit(True))
            combined = train_fold.unionByName(val_fold)
            
            # Train
            clf_with_val = clf.setValidationIndicatorCol("is_val")
            model = clf_with_val.fit(combined)
            
            # Evaluate
            eval_pr = BinaryClassificationEvaluator(
                labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
            pred_df = model.transform(val_fold)
            aucpr = eval_pr.evaluate(pred_df)
            fold_aucprs.append(aucpr)
            
            print(f"[CV {combo_idx}/{len(all_combos)}] Fold {fold_idx}/3: "
                  f"AUC-PR={aucpr:.4f} | Params: {params}")
        
        # Compute mean and std
        import statistics
        mean_aucpr = statistics.mean(fold_aucprs)
        std_aucpr = statistics.stdev(fold_aucprs) if len(fold_aucprs) > 1 else 0.0
        
        tuning_results.append({
            "params": params,
            "fold_scores": fold_aucprs,
            "mean_aucpr": mean_aucpr,
            "std_aucpr": std_aucpr
        })
        
        print(f"[CV {combo_idx}/{len(all_combos)}] SUMMARY: "
              f"Mean={mean_aucpr:.4f} ± {std_aucpr:.4f} | {params}\n")
    
    # Find best params
    best_result = max(tuning_results, key=lambda x: x["mean_aucpr"])
    best_params = best_result["params"]
    best_aucpr = best_result["mean_aucpr"]
    
    print(f"\n{'='*80}")
    print(f"TUNING COMPLETE - BEST PARAMS FOUND")
    print(f"{'='*80}")
    print(f"Best Mean AUC-PR: {best_aucpr:.4f} ± {best_result['std_aucpr']:.4f}")
    print(f"Best Params: {best_params}")
    print(f"{'='*80}\n")
    
    # Print top 5 results
    print("TOP 5 CONFIGURATIONS:")
    print("-" * 80)
    sorted_results = sorted(tuning_results, key=lambda x: x["mean_aucpr"], reverse=True)[:5]
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank}. AUC-PR={result['mean_aucpr']:.4f} ± {result['std_aucpr']:.4f} | {result['params']}")
    print()
    
    return best_params, tuning_results


# ============================================================================
# FUNCTION: compute_class_weight() - V2.1 IMPROVED
# ============================================================================
# Mục đích: Tính class weight để handle imbalanced data
# -> Tăng weight cho class thiểu số (helpful reviews)
#
# V2.1 IMPROVEMENTS:
# - Sử dụng công thức sklearn balanced: w = n_samples / (n_classes * n_samples_class)
# - Thêm mode "balanced_subsample" cho better generalization
# - Normalize weights để sum = n_samples
#
# Tham số:
# - df: DataFrame chứa labels
# - label_col: Tên cột label (binary 0/1)
# - weight_col: Tên cột weight output (default "weight")
# - pos_weight: 'auto' (balanced), 'auto_simple' (N_neg/N_pos), float, hoặc None
#
# Weighting Strategies:
# ┌─────────────────┬──────────────────────────────────────────────────────────┐
# │ pos_weight      │ Formula                                                  │
# ├─────────────────┼──────────────────────────────────────────────────────────┤
# │ 'auto'          │ SKLEARN BALANCED (V2.1 NEW):                             │
# │  (RECOMMENDED)  │ w0 = N / (2 * N_neg)                                     │
# │                 │ w1 = N / (2 * N_pos)                                     │
# │                 │ Ví dụ: N=100K, pos=20K, neg=80K                          │
# │                 │   w0 = 100K/(2*80K) = 0.625                              │
# │                 │   w1 = 100K/(2*20K) = 2.500                              │
# │                 │ -> Balanced: w0 * N_neg + w1 * N_pos = N                 │
# ├─────────────────┼──────────────────────────────────────────────────────────┤
# │ 'auto_simple'   │ SIMPLE RATIO (V1 OLD):                                   │
# │                 │ w1 = N_neg / N_pos (clamped to [0.1, 10])                │
# │                 │ w0 = 1.0                                                 │
# │                 │ Ví dụ: 80K neg, 20K pos -> w1 = 4.0                       │
# ├─────────────────┼──────────────────────────────────────────────────────────┤
# │ float value     │ MANUAL:                                                  │
# │                 │ w1 = pos_weight (manual)                                 │
# │                 │ w0 = 1.0                                                 │
# │                 │ Ví dụ: --posWeight 3.0 -> w1 = 3.0                        │
# ├─────────────────┼──────────────────────────────────────────────────────────┤
# │ None            │ NO WEIGHTING:                                            │
# │                 │ w1 = 1.0, w0 = 1.0                                       │
# │                 │ -> Treat all samples equally                              │
# └─────────────────┴──────────────────────────────────────────────────────────┘
#
# Weight Assignment:
# - Label = 1 (helpful) -> weight = w1
# - Label = 0 (not helpful) -> weight = w0
#
# ┌────────────────────────────────────────────────────────────────────┐
# │ WHY sklearn balanced formula?                                      │
# │ - Đảm bảo tổng weighted samples = n_samples (normalized)          │
# │ - Tránh class weight quá cực đoan (w1 >> w0)                      │
# │ - Generalize tốt hơn trên hidden test                              │
# │ - Standard trong sklearn, XGBoost, LightGBM                        │
# └────────────────────────────────────────────────────────────────────┘
#
# Output:
# - df_with_weights: DataFrame với cột weight_col
# - w0, w1: Class weights
# - pos, neg: Class counts
#
# Use case: Gọi trước khi train LightGBM
def compute_class_weight(df, label_col, weight_col="weight", pos_weight=None):
    """
    Compute class weights với sklearn balanced formula (V2.1)
    """
    # Đếm số lượng mỗi class
    counts = df.groupBy(label_col).count().collect()
    count_dict = {r[label_col]: r['count'] for r in counts}
    
    # Lấy pos/neg counts
    pos = count_dict.get(1, 0)
    neg = count_dict.get(0, 0)
    total = pos + neg
    
    # Check nếu collapse về 1 class duy nhất
    if pos == 0 or neg == 0:
        print(f"[WARN] Only one class present: pos={pos}, neg={neg}")
        print(f"[WARN] Using uniform weights (1.0)")
        w0, w1 = 1.0, 1.0
    
    # CASE 1: 'auto' - Sklearn balanced formula (RECOMMENDED)
    elif pos_weight is None or str(pos_weight).lower() == 'auto':
        # Công thức sklearn: w_class = n_samples / (n_classes * n_samples_class)
        # n_classes = 2 (binary classification)
        w0 = total / (2.0 * neg)
        w1 = total / (2.0 * pos)
        
        print(f"\n{'='*80}")
        print(f"[INFO] Class Weighting: SKLEARN BALANCED (auto)")
        print(f"{'='*80}")
        print(f"Total samples:     {total:,}")
        print(f"Positive (label=1): {pos:,} ({pos/total*100:.2f}%)")
        print(f"Negative (label=0): {neg:,} ({neg/total*100:.2f}%)")
        print(f"\nBalanced Weights:")
        print(f"  w0 (neg) = {total} / (2 * {neg}) = {w0:.4f}")
        print(f"  w1 (pos) = {total} / (2 * {pos}) = {w1:.4f}")
        print(f"\nWeight Ratio: w1/w0 = {w1/w0:.4f}")
        print(f"Effective samples after weighting:")
        print(f"  Negative: {neg} * {w0:.4f} = {neg*w0:,.0f}")
        print(f"  Positive: {pos} * {w1:.4f} = {pos*w1:,.0f}")
        print(f"  Total:    {neg*w0 + pos*w1:,.0f} (should ~= {total:,})")
        print(f"{'='*80}\n")
    
    # CASE 2: 'auto_simple' - Simple ratio (V1 old method)
    elif str(pos_weight).lower() == 'auto_simple':
        w1_raw = float(neg) / float(pos)
        w1 = max(0.1, min(10.0, w1_raw))  # Clamp to [0.1, 10]
        w0 = 1.0
        
        print(f"\n{'='*80}")
        print(f"[INFO] Class Weighting: SIMPLE RATIO (auto_simple)")
        print(f"{'='*80}")
        print(f"Total samples:     {total:,}")
        print(f"Positive (label=1): {pos:,} ({pos/total*100:.2f}%)")
        print(f"Negative (label=0): {neg:,} ({neg/total*100:.2f}%)")
        print(f"\nSimple Ratio:")
        print(f"  w1 = N_neg / N_pos = {neg} / {pos} = {w1_raw:.4f}")
        if w1 != w1_raw:
            print(f"  w1 (clamped) = {w1:.4f} (clamped to [0.1, 10.0])")
        print(f"  w0 = 1.0 (baseline)")
        print(f"{'='*80}\n")
    
    # CASE 3: Manual float value
    elif isinstance(pos_weight, (int, float)):
        w1 = float(pos_weight)
        w0 = 1.0
        
        print(f"\n{'='*80}")
        print(f"[INFO] Class Weighting: MANUAL")
        print(f"{'='*80}")
        print(f"Total samples:     {total:,}")
        print(f"Positive (label=1): {pos:,} ({pos/total*100:.2f}%)")
        print(f"Negative (label=0): {neg:,} ({neg/total*100:.2f}%)")
        print(f"\nManual Weights:")
        print(f"  w1 = {w1:.4f} (manual)")
        print(f"  w0 = 1.0 (baseline)")
        print(f"{'='*80}\n")
    
    # CASE 4: No weighting
    else:
        w0, w1 = 1.0, 1.0
        print(f"[INFO] No class weighting (w0=w1=1.0)")
    
    # Assign weights: w1 for label=1, w0 for label=0
    df_weighted = df.withColumn(
        weight_col,
        F.when(F.col(label_col) == 1, F.lit(w1))
         .otherwise(F.lit(w0))
    )
    
    return df_weighted, w1, w0, pos, neg


# ============================================================================
# ============================================================================
# FUNCTION: evaluate_model() - V2.1 IMPROVED
# ============================================================================
# Mục đích: Evaluate model và return comprehensive metrics
# -> V2.1: Tích hợp evaluation_v2.py cho metrics chính xác
#
# Tham số:
# - model: Trained LightGBM model
# - df: Validation/test DataFrame
# - label_col: Label column name
# - stage_name: "VAL" hoặc "TEST" (for logging)
#
# Metrics Computed:
# ┌───────────────────┬────────────────────────────────────────────────────┐
# │ Metric            │ Mô tả                                              │
# ├───────────────────┼────────────────────────────────────────────────────┤
# │ AUC-PR            │ Area Under Precision-Recall Curve (PRIMARY)       │
# │                   │ -> Quan trọng nhất cho imbalanced data              │
# │                   │ -> Target: 0.75-0.80 (realistic)                    │
# ├───────────────────┼────────────────────────────────────────────────────┤
# │ AUC-ROC           │ Area Under ROC Curve (SECONDARY)                   │
# │                   │ -> Ít sensitive với imbalance hơn                   │
# ├───────────────────┼────────────────────────────────────────────────────┤
# │ Precision         │ TP / (TP + FP) - Độ chính xác predictions         │
# │ Recall            │ TP / (TP + FN) - Độ phủ của positive class        │
# │ F1-Score          │ 2 * Precision * Recall / (P + R) - Harmonic mean  │
# ├───────────────────┼────────────────────────────────────────────────────┤
# │ Confusion Matrix  │ TP, TN, FP, FN - Chi tiết classification          │
# │                   │ -> True Positive: Dự đoán helpful đúng             │
# │                   │ -> True Negative: Dự đoán not helpful đúng         │
# │                   │ -> False Positive: Dự đoán helpful SAI             │
# │                   │ -> False Negative: Dự đoán not helpful SAI         │
# └───────────────────┴────────────────────────────────────────────────────┘
#
# ┌────────────────────────────────────────────────────────────────────┐
# │ WHY AUC-PR is primary metric?                                      │
# │ - Imbalanced data: 80% neg, 20% pos                                │
# │ - AUC-ROC có thể misleading (high AUC-ROC nhưng poor PR)          │
# │ - AUC-PR focus vào positive class (helpful reviews)                │
# │ - Competition/real-world: AUC-PR là standard cho imbalanced       │
# └────────────────────────────────────────────────────────────────────┘
#
# V2.1 Threshold Strategy:
# ┌────────────────────────────────────────────────────────────────────┐
# │ PRECISION-CONSTRAINED OPTIMIZATION (V2.1 NEW)                      │
# │                                                                     │
# │ Problem: Tối ưu F1 thuần túy -> Recall quá cao -> FP tăng vọt       │
# │          VD: Recall=95%, Precision=34% -> 66% predictions SAI!     │
# │                                                                     │
# │ Solution: Find threshold maximize F1 với constraint Precision≥50% │
# │                                                                     │
# │ Benefits:                                                           │
# │   OK Giảm False Positive (FP) xuống đáng kể                        │
# │   OK Precision ≥ 50% -> Chỉ 50% predictions có thể sai              │
# │   OK Vẫn giữ F1 score cao (balanced P-R)                           │
# │   OK Model predictions đáng tin cậy hơn cho business               │
# │                                                                     │
# │ Trade-off: Recall có thể giảm nhẹ (~80-85%) nhưng chấp nhận được │
# └────────────────────────────────────────────────────────────────────┘
#
# Output:
# - metrics: Dict chứa tất cả metrics
# - pred_df: DataFrame với predictions (probability, prediction columns)
#
# Use case: Evaluate sau khi train hoặc mỗi pseudo-labeling round
def evaluate_model(model, df, label_col, stage_name="VAL", out_dir=None):
    """
    Evaluate model với evaluation_v2.py (V2.1 IMPROVED)
    
    Returns: (metrics_dict, pred_df)
    
    V2.1 Changes:
    - Sử dụng sklearn metrics thay vì PySpark evaluators
    - Tính toán chính xác confusion matrix
    - Find optimal threshold
    - Save plots (PR/ROC curves, confusion matrix, threshold analysis)
    """
    import numpy as np
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 1: Predict với model                               │
    # └──────────────────────────────────────────────────────────┘
    pred_df = model.transform(df)
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 2: Convert Spark DataFrame -> numpy arrays          │
    # └──────────────────────────────────────────────────────────┘
    # Extract (label, probability[1]) cho sklearn metrics
    print(f"[{stage_name}] Collecting predictions for evaluation...")
    
    # UDF để extract probability của class 1
    get_prob_class1 = F.udf(lambda v: float(v[1]) if v and len(v) > 1 else 0.0, T.FloatType())
    pred_df_with_prob = pred_df.withColumn("prob_class1", get_prob_class1(F.col("probability")))
    
    # Collect data (có thể tốn memory nếu val set quá lớn, nhưng thường OK)
    y_true_pred = pred_df_with_prob.select(label_col, "prob_class1").collect()
    
    y_true = np.array([int(row[label_col]) for row in y_true_pred])
    y_pred_proba = np.array([float(row['prob_class1']) for row in y_true_pred])
    
    print(f"[{stage_name}] Collected {len(y_true):,} predictions")
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 3: Tính metrics với evaluation_v2 (nếu có)        │
    # └──────────────────────────────────────────────────────────┘
    if EVALUATION_V2_AVAILABLE:
        print(f"[{stage_name}] Using evaluation_v2.py for accurate metrics calculation")
        
        # Tính metrics với threshold mặc định = 0.5
        metrics_default = calculate_metrics(y_true, y_pred_proba, threshold=0.5)
        
        # Tìm optimal threshold với constraint Precision >= 50% (V2.1 BALANCED)
        # Để tránh FP quá cao, ta yêu cầu Precision tối thiểu 50%
        print(f"[{stage_name}] Finding optimal threshold (Precision >= 50%)...")
        opt_threshold, threshold_scores = find_optimal_threshold(y_true, y_pred_proba, metric="precision_min")
        
        # Tính lại metrics với optimal threshold
        metrics_opt = calculate_metrics(y_true, y_pred_proba, threshold=opt_threshold)
        
        # In classification report chi tiết
        print(f"\n{'='*80}")
        print(f"[{stage_name}] EVALUATION RESULTS WITH OPTIMAL THRESHOLD (Precision >= 50%)")
        print(f"{'='*80}")
        print_classification_report(y_true, y_pred_proba, threshold=opt_threshold)
        
        # In comparison giữa default vs optimal threshold
        print(f"\n{'='*80}")
        print(f"[{stage_name}] THRESHOLD COMPARISON")
        print(f"{'='*80}")
        print(f"| Threshold | AUC-PR | Precision | Recall |   F1   |")
        print(f"|   0.50    | {metrics_default['auc_pr']:.4f} |   {metrics_default['precision']:.4f}  | {metrics_default['recall']:.4f} | {metrics_default['f1_score']:.4f} |")
        print(f"|   {opt_threshold:.2f}    | {metrics_opt['auc_pr']:.4f} |   {metrics_opt['precision']:.4f}  | {metrics_opt['recall']:.4f} | {metrics_opt['f1_score']:.4f} | <- OPTIMAL")
        print(f"{'='*80}")
        
        # Highlight improvement
        prec_diff = metrics_opt['precision'] - metrics_default['precision']
        rec_diff = metrics_opt['recall'] - metrics_default['recall']
        print(f"OK Precision improvement: {prec_diff:+.4f} ({prec_diff/metrics_default['precision']*100:+.1f}%)")
        print(f"OK Recall trade-off: {rec_diff:+.4f} ({rec_diff/metrics_default['recall']*100:+.1f}%)")
        print(f"{'='*80}\n")
        
        # ┌──────────────────────────────────────────────────────────┐
        # │ STEP 4: Save plots (nếu có out_dir)                     │
        # └──────────────────────────────────────────────────────────┘
        if out_dir:
            try:
                # Tạo reports directory
                reports_dir = os.path.join(out_dir, "reports")
                os.makedirs(reports_dir, exist_ok=True)
                
                # Plot PR/ROC curves
                pr_roc_path = os.path.join(reports_dir, f"pr_roc_curves_{stage_name.lower()}.png")
                plot_pr_roc_curves(y_true, y_pred_proba, out_path=pr_roc_path)
                
                # Plot confusion matrix với optimal threshold
                y_pred_binary = (y_pred_proba >= opt_threshold).astype(int)
                cm_path = os.path.join(reports_dir, f"confusion_matrix_{stage_name.lower()}.png")
                plot_confusion_matrix(y_true, y_pred_binary, out_path=cm_path)
                
                # Plot threshold analysis
                threshold_path = os.path.join(reports_dir, f"threshold_analysis_{stage_name.lower()}.png")
                plot_threshold_analysis(y_true, y_pred_proba, out_path=threshold_path)
                
                print(f"[{stage_name}] Evaluation plots saved to {reports_dir}")
                
            except Exception as e:
                print(f"[WARN] Failed to save evaluation plots: {e}")
        
        # Return metrics với optimal threshold
        return metrics_opt, pred_df
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ FALLBACK: Dùng PySpark evaluators (less accurate)       │
    # └──────────────────────────────────────────────────────────┘
    else:
        print(f"[{stage_name}] Using PySpark evaluators (evaluation_v2.py not available)")
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        
        eval_pr = BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
        eval_roc = BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        
        aucpr = eval_pr.evaluate(pred_df)
        aucroc = eval_roc.evaluate(pred_df)
        
        # Compute Precision, Recall, F1
        evaluator_precision = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision")
        evaluator_recall = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="weightedRecall")
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="f1")
        
        precision = evaluator_precision.evaluate(pred_df)
        recall = evaluator_recall.evaluate(pred_df)
        f1 = evaluator_f1.evaluate(pred_df)
        
        # Confusion matrix
        cm_df = pred_df.groupBy(label_col, "prediction").count().collect()
        confusion_matrix = {}
        for row in cm_df:
            key = f"true_{int(row[label_col])}_pred_{int(row['prediction'])}"
            confusion_matrix[key] = int(row['count'])
        
        tp = confusion_matrix.get("true_1_pred_1", 0)
        tn = confusion_matrix.get("true_0_pred_0", 0)
        fp = confusion_matrix.get("true_0_pred_1", 0)
        fn = confusion_matrix.get("true_1_pred_0", 0)
        
        print(f"[{stage_name}] AUC-PR={aucpr:.4f} | AUC-ROC={aucroc:.4f}")
        print(f"[{stage_name}] Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}")
        print(f"[{stage_name}] Confusion Matrix: TP={tp:,} TN={tn:,} FP={fp:,} FN={fn:,}")
        
        metrics = {
            "auc_pr": float(aucpr),
            "auc_roc": float(aucroc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "threshold": 0.5,
            "confusion_matrix": {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn
            }
        }
        
        return metrics, pred_df


# ============================================================================
# FUNCTION: pseudo_label_iteration()
# ============================================================================
# Mục đích: Thực hiện 1 iteration của pseudo-labeling
# -> Predict trên unlabeled data -> Select confident samples -> Add to training
#
# Tham số:
# - model: Trained model từ iteration trước
# - unlabeled_df: DataFrame không có labels (test set)
# - label_col: Label column name để tạo
# - features_col: Features column name
# - min_prob: Minimum probability threshold (default 0.9)
# - top_pct: Top % confident samples to select (default 0.1 = 10%)
# - pseudo_weight: Weight cho pseudo-labeled samples (default 0.3)
#
# Pseudo-Labeling Algorithm:
# ┌────────────────────────────────────────────────────────────────────┐
# │ STEP 1: Predict on unlabeled data                                 │
# │   model.transform(unlabeled_df) -> probability column              │
# │   Extract prob_class1 (probability of "helpful")                  │
# ├────────────────────────────────────────────────────────────────────┤
# │ STEP 2: Filter confident predictions                              │
# │   Positive: prob_class1 >= 0.9 (very likely helpful)              │
# │   Negative: prob_class1 <= 0.1 (very likely not helpful)          │
# ├────────────────────────────────────────────────────────────────────┤
# │ STEP 3: Select top % by confidence                                │
# │   Sort positive by prob DESC -> Take top 10%                        │
# │   Sort negative by prob ASC -> Take top 10%                         │
# ├────────────────────────────────────────────────────────────────────┤
# │ STEP 4: Assign pseudo-labels with low weight                      │
# │   Positive samples -> label=1, weight=0.3                           │
# │   Negative samples -> label=0, weight=0.3                           │
# │   (Low weight vì không chắc chắn 100%)                            │
# └────────────────────────────────────────────────────────────────────┘
#
# ┌────────────────────────────────────────────────────────────────────┐
# │ WHY pseudo-labeling?                                               │
# │ - Semi-supervised learning: Tận dụng unlabeled test data          │
# │ - Test set có millions samples -> Thêm vào training = more data    │
# │ - Confident predictions có thể improve model                       │
# │ - Trade-off: Low weight (0.3) để tránh noise từ wrong labels     │
# └────────────────────────────────────────────────────────────────────┘
#
# Output:
# - pseudo_df: DataFrame với pseudo-labels (label_col, weight columns)
# - None nếu không tìm thấy confident samples
#
# Use case: Gọi sau mỗi iteration training trong pseudo-labeling rounds
def pseudo_label_iteration(model, unlabeled_df, label_col, features_col, 
                           min_prob=0.9, top_pct=0.1, pseudo_weight=0.3):
    """
    Pseudo-labeling: predict on unlabeled data, select high-confidence samples.
    Returns DataFrame with pseudo-labels and low weight.
    """
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 1: Predict trên unlabeled data                     │
    # └──────────────────────────────────────────────────────────┘
    # Transform sẽ thêm cột: prediction, rawPrediction, probability
    pred_df = model.transform(unlabeled_df)
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 2: Extract probability của class 1 (helpful)       │
    # └──────────────────────────────────────────────────────────┘
    # probability là DenseVector([prob_class0, prob_class1])
    # -> Lấy prob_class1 = v[1]
    get_prob_udf = F.udf(lambda v: float(v[1]) if v and len(v) > 1 else 0.0, T.FloatType())
    pred_df = pred_df.withColumn("prob_class1", get_prob_udf(F.col("probability")))
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 3: Filter confident predictions                    │
    # └──────────────────────────────────────────────────────────┘
    # Confident POSITIVE: prob_class1 >= 0.9 (90% chắc là helpful)
    confident_pos = pred_df.filter(F.col("prob_class1") >= min_prob)
    # Confident NEGATIVE: prob_class1 <= 0.1 (90% chắc là not helpful)
    confident_neg = pred_df.filter(F.col("prob_class1") <= (1 - min_prob))
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 4: Chỉ lấy top % samples có confidence cao nhất    │
    # └──────────────────────────────────────────────────────────┘
    # Lấy 10% samples từ mỗi class
    n_pos = int(confident_pos.count() * top_pct)
    n_neg = int(confident_neg.count() * top_pct)
    
    # Nếu không có confident samples -> return None
    if n_pos == 0 and n_neg == 0:
        print("[WARN] No confident pseudo-labels found")
        return None
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 5: Sort và select top samples                      │
    # └──────────────────────────────────────────────────────────┘
    # Positive: Sort DESC (cao nhất trước) -> Take n_pos -> Assign label=1
    pseudo_pos = confident_pos.orderBy(F.desc("prob_class1")).limit(n_pos) \
        .withColumn(label_col, F.lit(1))
    
    # Negative: Sort ASC (thấp nhất trước) -> Take n_neg -> Assign label=0
    pseudo_neg = confident_neg.orderBy(F.asc("prob_class1")).limit(n_neg) \
        .withColumn(label_col, F.lit(0))
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 6: Merge và assign low weight (0.3)                │
    # └──────────────────────────────────────────────────────────┘
    # Union 2 DataFrames
    pseudo_df = pseudo_pos.unionByName(pseudo_neg, allowMissingColumns=True)
    # Add weight column = 0.3 (thấp hơn real labels = 1.0 hoặc w1)
    pseudo_df = pseudo_df.withColumn("weight", F.lit(pseudo_weight))
    
    # ┌──────────────────────────────────────────────────────────┐
    # │ STEP 7: Chỉ giữ lại columns cần thiết                   │
    # └──────────────────────────────────────────────────────────┘
    # Keep original columns + label_col + weight
    cols_to_keep = [c for c in unlabeled_df.columns] + [label_col, "weight"]
    pseudo_df = pseudo_df.select(*[c for c in cols_to_keep if c in pseudo_df.columns])
    
    # In thông tin
    print(f"[PSEUDO] Added {n_pos} positive + {n_neg} negative pseudo-labels (weight={pseudo_weight})")
    
    return pseudo_df  # Return DataFrame với pseudo-labels


def save_schema_logs(out_dir, train_schema, test_schema, columns_used, params, metadata):
    """Save schema and parameter logs to output directory."""
    try:
        # Create logs subdirectory
        logs_dir = os.path.join(out_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save train schema
        with open(os.path.join(logs_dir, "schema_train.txt"), "w", encoding="utf-8") as f:
            f.write("TRAIN SCHEMA\n")
            f.write("=" * 80 + "\n")
            for field in train_schema:
                f.write(f"{field.name}: {field.dataType}\n")
        
        # Save test schema if available
        if test_schema:
            with open(os.path.join(logs_dir, "schema_test.txt"), "w", encoding="utf-8") as f:
                f.write("TEST SCHEMA\n")
                f.write("=" * 80 + "\n")
                for field in test_schema:
                    f.write(f"{field.name}: {field.dataType}\n")
        
        # Save columns used
        with open(os.path.join(logs_dir, "columns_used.txt"), "w", encoding="utf-8") as f:
            f.write("COLUMNS USED IN TRAINING\n")
            f.write("=" * 80 + "\n")
            for i, col in enumerate(columns_used, 1):
                f.write(f"{i}. {col}\n")
        
        # Save parameters
        with open(os.path.join(logs_dir, "params.txt"), "w", encoding="utf-8") as f:
            f.write("TRAINING PARAMETERS\n")
            f.write("=" * 80 + "\n")
            for k, v in sorted(params.items()):
                f.write(f"{k} = {v}\n")
        
        # Save metadata JSON
        with open(os.path.join(logs_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[OK] Logs saved to {logs_dir}")
    except Exception as e:
        print(f"[WARN] Failed to save logs: {e}")


def save_error_log(out_dir, error_info, args=None):
    """Save detailed error log when training fails."""
    try:
        # Handle HDFS paths - convert to local path for error logging
        if out_dir and out_dir.startswith('hdfs://'):
            # Extract path after hdfs://host:port/
            import re
            match = re.search(r'hdfs://[^/]+/(.*)', out_dir)
            if match:
                out_dir = f"./{match.group(1).replace('/', '_')}_errors"
            else:
                out_dir = "./error_logs"
        
        # Create error log directory
        error_dir = out_dir if out_dir else "."
        os.makedirs(error_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_path = os.path.join(error_dir, f"error_log_{timestamp}.txt")
        
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"TRAINING ERROR LOG - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write arguments if available
            if args:
                f.write("COMMAND LINE ARGUMENTS:\n")
                f.write("-" * 80 + "\n")
                for key, value in vars(args).items():
                    f.write(f"{key} = {value}\n")
                f.write("\n")
            
            # Write error details
            f.write("ERROR DETAILS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Error Type: {error_info['type']}\n")
            f.write(f"Error Message: {error_info['message']}\n\n")
            
            # Write full traceback
            f.write("FULL TRACEBACK:\n")
            f.write("-" * 80 + "\n")
            f.write(error_info['traceback'])
            f.write("\n")
            
            # Write system info
            f.write("\nSYSTEM INFO:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Working Directory: {os.getcwd()}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF ERROR LOG\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n[ERROR] Detailed error log saved to: {error_log_path}")
        print(f"[ERROR] Please check the log file for complete error information.")
        return error_log_path
    except Exception as log_error:
        print(f"[FATAL] Could not save error log: {log_error}")
        return None


def format_error_message(exc_type, exc_value, exc_tb):
    """Format exception information into a structured dictionary."""
    return {
        'type': exc_type.__name__,
        'message': str(exc_value),
        'traceback': ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    }



# ============================================================================
# MAIN FUNCTION: main()
# ============================================================================
# Mục đích: Orchestrate toàn bộ training pipeline từ đầu đến cuối
#
# Pipeline Flow (10 bước chính):
#
# ┌────────────────────────────────────────────────────────────────────┐
# │ 1. Initialize Spark Session                                       │
# │    - Enable adaptive query execution                               │
# │    - Set log level to WARN                                         │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 2. Load Data (Train + Test)                                       │
# │    - Read Parquet from HDFS/local                                  │
# │    - Apply --limit_train for quick testing                         │
# │    - Print sample counts                                           │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 3. Validate Schema & Features                                     │
# │    - Check review_id exists (CRITICAL for submission)              │
# │    - Check label_col exists (generate synthetic if missing)        │
# │    - Check features_col exists (assemble if missing)               │
# │    - Validate feature dimensions match train/test                  │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 4. Clean Data                                                      │
# │    - Drop leaky columns (helpful_votes, helpful_ratio, v.v.)      │
# │    - Cast label to int (0/1)                                       │
# │    - Print feature summaries                                       │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 5. Stratified Train/Val Split                                     │
# │    - Split 90/10 (default)                                         │
# │    - Maintain class ratio trong cả train và val                    │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 6. Compute Class Weights                                           │
# │    - Auto: w = N_neg / N_pos (clamped [0.1, 10])                  │
# │    - Manual: --posWeight 3.0                                       │
# │    - None: w = 1.0                                                 │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 7. [Optional] Hyperparameter Tuning                               │
# │    - 3-fold CV grid search                                         │
# │    - Quick (9 combos) hoặc Thorough (27 combos)                   │
# │    - Update args với best params                                   │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 8. Train Initial Model                                             │
# │    - LightGBMClassifier với optimized hyperparameters              │
# │    - Early stopping trên validation set                            │
# │    - Evaluate: AUC-PR, AUC-ROC, Precision, Recall, F1             │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 9. [Optional] Pseudo-Labeling Iterations                          │
# │    - Predict on test -> Select confident samples                    │
# │    - Add to training (low weight = 0.3)                            │
# │    - Retrain -> Evaluate -> Keep best model                          │
# │    - Repeat for N rounds                                           │
# └────────────────────────────────────────────────────────────────────┘
#          ↓
# ┌────────────────────────────────────────────────────────────────────┐
# │ 10. Save Model + Metadata + Logs                                  │
# │     - Model -> HDFS/local                                           │
# │     - Metadata JSON (params, metrics, v.v.)                        │
# │     - Schema logs (if --save_schema_log)                           │
# │     - Evaluation reports (JSON, CSV, TXT)                          │
# └────────────────────────────────────────────────────────────────────┘
#
# ┌────────────────────────────────────────────────────────────────────┐
# │ CRITICAL CHECKS                                                    │
# ├────────────────────────────────────────────────────────────────────┤
# │ 1. review_id PHẢI tồn tại -> Submission alignment                  │
# │ 2. Feature dimensions PHẢI match train/test -> Model compatibility │
# │ 3. Label PHẢI binary {0,1} -> Classification correctness           │
# │ 4. Leaky columns PHẢI được xóa -> No label leakage                 │
# │ 5. Class weights PHẢI hợp lý -> Handle imbalance                   │
# └────────────────────────────────────────────────────────────────────┘
#
# Error Handling:
# - Catch all exceptions -> Save detailed error log
# - Format error với traceback -> Debugging easier
# - Print clear error messages -> User-friendly
#
# Use case: Complete end-to-end training cho Amazon review helpfulness
def main():
    args = parse_args()
    spark = None
    
    try:
        spark = (SparkSession.builder
                 .appName("Train-LightGBM-V2-SemiSupervised")
                 .config("spark.sql.adaptive.enabled", "true")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("WARN")
        
        print(f"\n{'='*80}")
        print(f"LIGHTGBM TRAINING - Semi-Supervised with Pseudo-Labeling")
        print(f"{'='*80}")
        print(f"Train: {args.train}")
        print(f"Test: {args.test}")
        print(f"Output: {args.out}")
        print(f"Target AUC-PR: [{args.target_aucpr_min:.2f}, {args.target_aucpr_max:.2f}]")
        print(f"{'='*80}\n")
        
        # ========== Load Data ==========
        train_df = spark.read.parquet(args.train)
        original_train_count = train_df.count()
        print(f"[LOAD] Train samples: {original_train_count:,}")
        
        # Apply limit if specified
        if args.limit_train:
            train_df = train_df.limit(args.limit_train)
            print(f"[LIMIT] Using {args.limit_train:,} samples for training")
        
        # Load test for pseudo-labeling (optional)
        test_df = None
        if args.test:
            test_df = spark.read.parquet(args.test)
            print(f"[LOAD] Test samples: {test_df.count():,}")
        
        # ========== Validate Schema ==========
        # Ensure ID column exists (create if missing)
        if args.id_col not in train_df.columns:
            print(f"[WARN] ID column '{args.id_col}' not found in data!")
            print(f"[WARN] Generating auto-incremented IDs (this may cause data leakage)")
            train_df = train_df.withColumn(args.id_col, F.monotonically_increasing_id().cast(T.StringType()))
        else:
            print(f"[OK] Found '{args.id_col}' column in data (preserving original IDs)")
            train_df = ensure_id_string(train_df, args.id_col)
            # Show sample IDs to verify
            sample_ids = [row[args.id_col] for row in train_df.select(args.id_col).limit(5).collect()]
            print(f"[OK] Sample {args.id_col}: {sample_ids[:3]}")

        
        # Check if label column exists
        if args.label_col not in train_df.columns:
            print(f"[WARN] Label column '{args.label_col}' not found in train data.")
            print(f"[INFO] Generating synthetic labels automatically...")
            
            # Generate synthetic labels using heuristic method (default)
            train_df = generate_synthetic_labels(
                train_df, 
                args.label_col, 
                method=args.label_method,
                seed=args.seed
            )
        
        # ========== Feature Assembly ==========
        # Assemble features vector if not present
        feature_cols_used = None
        actual_train_dim = None
        
        if args.features_col not in train_df.columns:
            print(f"[INFO] Features column '{args.features_col}' not found, assembling from numeric columns...")
            train_df, feature_cols_used = assemble_features(
                train_df, 
                args.features_col, 
                label_col=args.label_col,
                exclude_cols=[args.id_col]
            )
            actual_train_dim = get_vector_size(train_df, args.features_col)
            print(f"[INFO] Assembled features dimension: {actual_train_dim}")
        else:
            # Features column exists - get dimension
            print(f"[INFO] Features column '{args.features_col}' already exists")
            actual_train_dim = get_vector_size(train_df, args.features_col)
            print(f"[INFO] Existing features dimension: {actual_train_dim}")
        
        # Handle test data - try to match train's feature set
        if test_df:
            if args.id_col not in test_df.columns:
                print(f"[WARN] ID column '{args.id_col}' not found in test data!")
                print(f"[WARN] Generating auto-incremented IDs (this may cause data leakage)")
                test_df = test_df.withColumn(args.id_col, F.monotonically_increasing_id().cast(T.StringType()))
            else:
                print(f"[OK] Found '{args.id_col}' column in test data (preserving original IDs)")
                test_df = ensure_id_string(test_df, args.id_col)
                # Show sample IDs to verify
                sample_ids = [row[args.id_col] for row in test_df.select(args.id_col).limit(5).collect()]
                print(f"[OK] Sample test {args.id_col}: {sample_ids[:3]}")
            
            if args.features_col not in test_df.columns:
                print(f"[INFO] Assembling features for test data...")
                if feature_cols_used:
                    # Use EXACT same columns as train (best case)
                    print(f"[INFO] Using same {len(feature_cols_used)} columns as train data")
                    
                    # Check which columns exist in test
                    missing_cols = [c for c in feature_cols_used if c not in test_df.columns]
                    if missing_cols:
                        print(f"[WARN] Test missing {len(missing_cols)} columns from train: {missing_cols[:5]}...")
                        print(f"[WARN] Will fill missing columns with 0.0")
                        for col_name in missing_cols:
                            test_df = test_df.withColumn(col_name, F.lit(0.0))
                    
                    assembler = VectorAssembler(
                        inputCols=feature_cols_used,
                        outputCol=args.features_col,
                        handleInvalid='skip'
                    )
                    test_df = assembler.transform(test_df)
                else:
                    # Train had pre-assembled features, test needs assembly
                    # This is tricky - try to match dimensions
                    print(f"[WARN] Train has pre-assembled features ({actual_train_dim} dims)")
                    print(f"[WARN] Attempting to assemble test features to match...")
                    
                    test_df, test_feature_cols = assemble_features(
                        test_df,
                        args.features_col,
                        exclude_cols=[args.id_col]
                    )
                    test_dim = get_vector_size(test_df, args.features_col)
                    
                    if test_dim != actual_train_dim:
                        print(f"\n{'='*80}")
                        print(f"FEATURE DIMENSION MISMATCH DETECTED")
                        print(f"{'='*80}")
                        print(f"Train dimension: {actual_train_dim}")
                        print(f"Test dimension:  {test_dim}")
                        print(f"")
                        print(f"This usually happens when train and test use different feature pipelines:")
                        print(f"  - Train: metadata -> text -> sentiment (~37 features)")
                        print(f"  - Test:  feature_pipeline_v2.py --preset full (~20,000 TF-IDF features)")
                        print(f"")
                        print(f"SOLUTIONS:")
                        print(f"  1. Re-run test with --preset fast:")
                        print(f"     spark-submit code_v2/features/feature_pipeline_v2.py \\")
                        print(f"       --input {args.test.replace('/features_test_v2', '/test')} \\")
                        print(f"       --output {args.test} \\")
                        print(f"       --preset fast --save")
                        print(f"")
                        print(f"  2. Or use --force to train anyway (NOT recommended):")
                        print(f"     Add --force flag to ignore dimension mismatch")
                        print(f"")
                        print(f"  3. Or ensure both use same pipeline:")
                        print(f"     metadata_features_v2.py -> text_preprocessing_v2.py -> sentiment_vader_v2.py")
                        print(f"{'='*80}\n")
                        
                        if not args.force:
                            raise RuntimeError(
                                f"Feature dimension mismatch: train={actual_train_dim}, test={test_dim}. "
                                f"Use --force to override or fix the feature pipeline."
                            )
                        else:
                            print(f"[WARN] --force enabled: continuing with mismatched dimensions")
                            print(f"[WARN] Model will only use train features for training")
                            print(f"[WARN] Test predictions may be unreliable!")
            else:
                # Test already has features column
                test_dim = get_vector_size(test_df, args.features_col)
                print(f"[INFO] Test features dimension: {test_dim}")
                
                if test_dim != actual_train_dim:
                    print(f"[WARN] Dimension mismatch: train={actual_train_dim}, test={test_dim}")
                    if not args.force:
                        raise RuntimeError(
                            f"Feature dimension mismatch. Train and test must have same feature dimension. "
                            f"Use --force to override."
                        )
                    else:
                        print(f"[WARN] --force enabled: continuing anyway")
        
        # ========== Clean Data ==========
        train_df = drop_leaky_columns(train_df, args.features_col, args.label_col)
        train_df = train_df.withColumn(args.label_col, F.col(args.label_col).cast(T.IntegerType()))
        
        if test_df:
            test_df = drop_leaky_columns(test_df, args.features_col, args.label_col)
        
        # Print feature summaries for debugging
        print_feature_summary(train_df, args.features_col, stage="TRAIN")
        if test_df and args.features_col in test_df.columns:
            print_feature_summary(test_df, args.features_col, stage="TEST")
        
        # ========== Validate Feature Dimensions ==========
        actual_dim = actual_train_dim if actual_train_dim else get_vector_size(train_df, args.features_col)
        print(f"\n[INFO] Final feature dimension: {actual_dim}")
        
        # Validate against expected dimension if provided
        if args.numFeatures and actual_dim != args.numFeatures:
            msg = f"Feature dimension mismatch: expected {args.numFeatures}, got {actual_dim}"
            if args.force:
                print(f"[WARN] {msg} (--force enabled, continuing anyway)")
            else:
                print(f"[ERROR] {msg}")
                raise RuntimeError(f"{msg}. Use --force to override.")
        
        if test_df and args.features_col in test_df.columns:
            test_dim = get_vector_size(test_df, args.features_col)
            if test_dim != actual_dim:
                print(f"\n[WARN] Test/Train dimension mismatch: test={test_dim}, train={actual_dim}")
                print(f"[WARN] This will cause issues during pseudo-labeling (if enabled)")
                if args.pseudo_rounds > 0:
                    print(f"[WARN] Disabling pseudo-labeling due to dimension mismatch")
                    args.pseudo_rounds = 0
        
        # ========== Validate Labels ==========
        distinct_labels = [r[0] for r in train_df.select(args.label_col).distinct().collect()]
        if not set(distinct_labels).issubset({0, 1, None}):
            raise RuntimeError(f"Label must be binary {{0,1}}, got: {sorted(distinct_labels)}")
        
        # ========== Stratified Split ==========
        train_split, val_split = stratified_train_val_split(
            train_df, args.label_col, val_frac=args.valFrac, seed=args.seed)
        
        print(f"[SPLIT] Train: {train_split.count():,} | Val: {val_split.count():,}")
        
        # ========== Class Weighting ==========
        pos_weight_val = args.posWeight if args.posWeight != "auto" else "auto"
        train_split, w1, w0, n_pos, n_neg = compute_class_weight(
            train_split, args.label_col, weight_col="weight", pos_weight=pos_weight_val)
        
        val_split = val_split.withColumn("weight", F.lit(1.0))  # No weighting for validation
        
        # ========== Hyperparameter Tuning (Optional) ==========
        tuning_results = None
        if args.auto_tune:
            print(f"\n[TUNING] Auto-tuning enabled ({args.tune_preset} preset)")
            print(f"[TUNING] This will take ~10-30 minutes depending on preset...")
            
            best_params, tuning_results = hyperparameter_tuning(
                train_split, args.label_col, args.features_col, args, preset=args.tune_preset
            )
            
            # Update args with best params
            print(f"[TUNING] Applying best hyperparameters to final training:")
            for key, value in best_params.items():
                old_value = getattr(args, key)
                setattr(args, key, value)
                print(f"  {key}: {old_value} -> {value}")
            print()
        
        # ========== Initial Training ==========
        print(f"\n[TRAIN] Starting LightGBM training with optimized params...")
        print(f"[TRAIN] numLeaves={args.numLeaves}, learningRate={args.learningRate}, "
              f"minDataInLeaf={args.minDataInLeaf}")
        
        # Prepare combined dataset with validation indicator
        train_split = train_split.withColumn("is_val", F.lit(False))
        val_split = val_split.withColumn("is_val", F.lit(True))
        combined_df = train_split.unionByName(val_split)
        
        # Build classifier with optimized hyperparameters
        clf = LightGBMClassifier(
            objective="binary",
            labelCol=args.label_col,
            featuresCol=args.features_col,
            weightCol="weight",
            predictionCol="prediction",
            rawPredictionCol="rawPrediction",
            probabilityCol="probability",
            numLeaves=args.numLeaves,
            learningRate=args.learningRate,
            numIterations=args.numIterations,
            featureFraction=args.featureFraction,
            baggingFraction=args.baggingFraction,
            minDataInLeaf=args.minDataInLeaf,
            maxDepth=args.maxDepth,
            lambdaL1=args.lambdaL1,
            lambdaL2=args.lambdaL2,
            earlyStoppingRound=args.earlyStoppingRound,
            isUnbalance=True,
            seed=args.seed
        ).setValidationIndicatorCol("is_val")
        
        model = clf.fit(combined_df)
        
        # ========== Evaluate ==========
        metrics, _ = evaluate_model(model, val_split, args.label_col, stage_name="VAL-INITIAL", out_dir=args.out)
        
        best_aucpr = metrics["auc_pr"]
        best_metrics = metrics
        best_model = model
        
        # ========== Pseudo-Labeling Iterations ==========
        if args.pseudo_rounds > 0 and test_df is not None:
            print(f"\n[PSEUDO] Starting {args.pseudo_rounds} pseudo-labeling rounds...")
            
            for round_idx in range(args.pseudo_rounds):
                print(f"\n--- Pseudo-Labeling Round {round_idx + 1}/{args.pseudo_rounds} ---")
                
                # Generate pseudo-labels on test set
                pseudo_df = pseudo_label_iteration(
                    model, test_df, args.label_col, args.features_col,
                    min_prob=args.pseudo_min_prob,
                    top_pct=args.pseudo_top_pct,
                    pseudo_weight=args.pseudo_weight
                )
                
                if pseudo_df is None:
                    print("[PSEUDO] No pseudo-labels generated, stopping early")
                    break
                
                # Combine with original training data
                train_augmented = train_split.unionByName(pseudo_df, allowMissingColumns=True)
                train_augmented = train_augmented.withColumn("is_val", F.lit(False))
                combined_augmented = train_augmented.unionByName(val_split)
                
                # Retrain
                model = clf.fit(combined_augmented)
                metrics, _ = evaluate_model(model, val_split, args.label_col, 
                                          stage_name=f"VAL-PSEUDO-R{round_idx+1}", out_dir=args.out)
                
                aucpr = metrics["auc_pr"]
                
                # Keep best model
                if aucpr > best_aucpr:
                    best_aucpr = aucpr
                    best_metrics = metrics
                    best_model = model
                    print(f"[PSEUDO] New best AUC-PR: {best_aucpr:.4f}")
                
                # Early stopping if target reached
                if args.target_aucpr_min <= aucpr <= args.target_aucpr_max:
                    print(f"[PSEUDO] Target AUC-PR reached ({aucpr:.4f}), stopping early")
                    break
        
        # ========== Save Model ==========
        print(f"\n[SAVE] Saving model to {args.out}")
        best_model.write().overwrite().save(args.out)
        
        # ========== Save Metadata & Logs ==========
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "id_col": args.id_col,
            "label_col": args.label_col,
            "features_col": args.features_col,
            "numFeatures": actual_dim,
            "posWeight": w1,
            "class_distribution": {"positive": n_pos, "negative": n_neg},
            "train_samples": train_split.count(),
            "val_samples": val_split.count(),
            "evaluation_metrics": best_metrics,
            "seed": args.seed,
            "hyperparameters": {
                "numLeaves": args.numLeaves,
                "learningRate": args.learningRate,
                "numIterations": args.numIterations,
                "earlyStoppingRound": args.earlyStoppingRound,
                "featureFraction": args.featureFraction,
                "baggingFraction": args.baggingFraction,
                "minDataInLeaf": args.minDataInLeaf,
                "maxDepth": args.maxDepth
            },
            "pseudo_labeling": {
                "rounds": args.pseudo_rounds,
                "min_prob": args.pseudo_min_prob,
                "top_pct": args.pseudo_top_pct,
                "weight": args.pseudo_weight
            } if args.pseudo_rounds > 0 else None,
            "hyperparameter_tuning": {
                "enabled": args.auto_tune,
                "preset": args.tune_preset if args.auto_tune else None,
                "results": tuning_results
            } if args.auto_tune else None
        }
        
        if args.save_schema_log:
            train_schema = train_df.schema.fields
            test_schema = test_df.schema.fields if test_df else None
            
            # List columns used (exclude internal columns)
            columns_used = feature_cols_used if feature_cols_used else [
                c for c in train_df.columns if c not in 
                {args.label_col, args.features_col, "weight", "is_val", "__uid__", "__label_int__"}
            ]
            
            params = vars(args)
            save_schema_logs(args.out, train_schema, test_schema, columns_used, params, metadata)
        
        # Save metadata JSON to model directory
        try:
            metadata_path = os.path.join(args.out, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"[OK] Metadata saved to {metadata_path}")
        except Exception as e:
            print(f"[WARN] Could not save metadata.json: {e}")
        
        # ========== Save Evaluation Report (LOCAL) ==========
        try:
            # Create reports directory
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_prefix = f"{reports_dir}/training_report_{timestamp}"
            
            # 1. Save detailed metrics as JSON
            report_json_path = f"{report_prefix}.json"
            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[OK] Detailed report saved to: {report_json_path}")
            
            # 2. Save summary metrics as CSV
            report_csv_path = f"{report_prefix}_metrics.csv"
            with open(report_csv_path, "w", encoding="utf-8", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Timestamp", metadata["timestamp"]])
                writer.writerow(["AUC-PR", f"{best_metrics['auc_pr']:.4f}"])
                writer.writerow(["AUC-ROC", f"{best_metrics['auc_roc']:.4f}"])
                writer.writerow(["Precision", f"{best_metrics['precision']:.4f}"])
                writer.writerow(["Recall", f"{best_metrics['recall']:.4f}"])
                writer.writerow(["F1-Score", f"{best_metrics['f1_score']:.4f}"])
                writer.writerow(["True Positive (TP)", best_metrics["confusion_matrix"]["TP"]])
                writer.writerow(["True Negative (TN)", best_metrics["confusion_matrix"]["TN"]])
                writer.writerow(["False Positive (FP)", best_metrics["confusion_matrix"]["FP"]])
                writer.writerow(["False Negative (FN)", best_metrics["confusion_matrix"]["FN"]])
                writer.writerow(["Training Samples", metadata["train_samples"]])
                writer.writerow(["Validation Samples", metadata["val_samples"]])
                writer.writerow(["Positive Class", metadata["class_distribution"]["positive"]])
                writer.writerow(["Negative Class", metadata["class_distribution"]["negative"]])
                writer.writerow(["Class Weight", f"{metadata['posWeight']:.3f}"])
                writer.writerow(["Feature Dimension", metadata["numFeatures"]])
            print(f"[OK] CSV metrics saved to: {report_csv_path}")
            
            # 3. Save human-readable text report
            report_txt_path = f"{report_prefix}_summary.txt"
            with open(report_txt_path, "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write("LIGHTGBM TRAINING REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Training Date: {metadata['timestamp']}\n")
                f.write(f"Model Output: {args.out}\n\n")
                
                f.write("-"*80 + "\n")
                f.write("EVALUATION METRICS (Validation Set)\n")
                f.write("-"*80 + "\n")
                f.write(f"AUC-PR (Primary Metric):  {best_metrics['auc_pr']:.4f}\n")
                f.write(f"AUC-ROC:                  {best_metrics['auc_roc']:.4f}\n")
                f.write(f"Precision:                {best_metrics['precision']:.4f}\n")
                f.write(f"Recall:                   {best_metrics['recall']:.4f}\n")
                f.write(f"F1-Score:                 {best_metrics['f1_score']:.4f}\n\n")
                
                f.write("-"*80 + "\n")
                f.write("CONFUSION MATRIX\n")
                f.write("-"*80 + "\n")
                cm = best_metrics["confusion_matrix"]
                f.write(f"True Positive (TP):       {cm['TP']:>10,}\n")
                f.write(f"True Negative (TN):       {cm['TN']:>10,}\n")
                f.write(f"False Positive (FP):      {cm['FP']:>10,}\n")
                f.write(f"False Negative (FN):      {cm['FN']:>10,}\n\n")
                
                f.write("-"*80 + "\n")
                f.write("DATASET STATISTICS\n")
                f.write("-"*80 + "\n")
                f.write(f"Training Samples:         {metadata['train_samples']:>10,}\n")
                f.write(f"Validation Samples:       {metadata['val_samples']:>10,}\n")
                f.write(f"Positive Class:           {metadata['class_distribution']['positive']:>10,}\n")
                f.write(f"Negative Class:           {metadata['class_distribution']['negative']:>10,}\n")
                f.write(f"Class Weight (pos):       {metadata['posWeight']:>10.3f}\n")
                f.write(f"Feature Dimension:        {metadata['numFeatures']:>10,}\n\n")
                
                f.write("-"*80 + "\n")
                f.write("MODEL HYPERPARAMETERS\n")
                f.write("-"*80 + "\n")
                hp = metadata["hyperparameters"]
                for key, value in hp.items():
                    f.write(f"{key:.<30} {value}\n")
                f.write("\n")
                
                if metadata.get("pseudo_labeling"):
                    f.write("-"*80 + "\n")
                    f.write("PSEUDO-LABELING SETTINGS\n")
                    f.write("-"*80 + "\n")
                    pl = metadata["pseudo_labeling"]
                    for key, value in pl.items():
                        f.write(f"{key:.<30} {value}\n")
                    f.write("\n")
                
                f.write("="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")
            
            print(f"[OK] Text report saved to: {report_txt_path}")
            print(f"\n{'='*80}")
            print(f"EVALUATION REPORTS SAVED:")
            print(f"  JSON:  {report_json_path}")
            print(f"  CSV:   {report_csv_path}")
            print(f"  TXT:   {report_txt_path}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"[WARN] Could not save evaluation reports: {e}")
            traceback.print_exc()
        
        # ========== Final Summary ==========
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"VAL_AUCPR     = {best_metrics['auc_pr']:.4f}")
        print(f"VAL_AUCROC    = {best_metrics['auc_roc']:.4f}")
        print(f"VAL_Precision = {best_metrics['precision']:.4f}")
        print(f"VAL_Recall    = {best_metrics['recall']:.4f}")
        print(f"VAL_F1        = {best_metrics['f1_score']:.4f}")
        print(f"Feature dimension = {actual_dim}")
        print(f"Train samples = {train_split.count():,}")
        print(f"Val samples = {val_split.count():,}")
        print(f"Class balance = pos:{n_pos:,} neg:{n_neg:,} (weight={w1:.3f})")
        print(f"Model saved to = {args.out}")
        
        if feature_cols_used:
            print(f"\nFeatures used ({len(feature_cols_used)}):")
            print(f"  {', '.join(feature_cols_used[:10])}")
            if len(feature_cols_used) > 10:
                print(f"  ... and {len(feature_cols_used) - 10} more")
        
        print(f"\nNEXT STEPS:")
        print(f"  1. Check validation metrics (AUC-PR should be 0.80-0.85)")
        print(f"  2. Run predictions on test set:")
        print(f"     spark-submit code_v2/models/predict_pipeline_v2.py \\")
        print(f"       --model {args.out} \\")
        print(f"       --input {args.test if args.test else 'hdfs://.../features_test_v2'} \\")
        print(f"       --output hdfs://.../predictions/submission.csv")
        print(f"  3. If dimension mismatch, ensure test uses SAME feature pipeline as train")
        print(f"{'='*80}\n")
        
    except Exception as e:
        # Capture exception information
        exc_type, exc_value, exc_tb = sys.exc_info()
        error_info = format_error_message(exc_type, exc_value, exc_tb)
        
        # Print error to console
        print(f"\n{'='*80}")
        print(f"TRAINING FAILED")
        print(f"{'='*80}")
        print(f"Error Type: {error_info['type']}")
        print(f"Error Message: {error_info['message']}")
        print(f"{'='*80}\n")
        
        # Save detailed error log
        out_dir = args.out if hasattr(args, 'out') and args.out else "."
        save_error_log(out_dir, error_info, args if 'args' in locals() else None)
        
        # Re-raise the exception
        raise
    
    finally:
        # Clean up Spark session
        if spark is not None:
            try:
                spark.stop()
            except Exception as stop_error:
                print(f"[WARN] Error stopping Spark session: {stop_error}")


if __name__ == "__main__":
    main()


