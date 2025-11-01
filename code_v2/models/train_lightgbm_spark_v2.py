#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spark 3.4–3.5 + SynapseML 1.0.7
# Train LightGBM with semi-supervised pseudo-labeling support:
# - Consistent featurization (numFeatures must match train/test)
# - Automatic class weight handling (--posWeight auto/manual)
# - Stratified train/val split with AUC-PR optimization (target: 0.80-0.85)
# - Pseudo-labeling rounds for unlabeled data
# - Comprehensive logging (schema, columns, params)
#
# Example:
# spark-submit ... train_lightgbm_spark_v2.py \
#   --train hdfs://.../output_v2/features_train_v2 \
#   --test hdfs://.../output_v2/features_test_v2 \
#   --out hdfs://.../output_v2/models/lightgbm_v2 \
#   --limit_train 1000000 \
#   --posWeight auto \
#   --pseudo_rounds 2 \
#   --save_schema_log

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pyspark.sql import SparkSession, functions as F, types as T, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import VectorUDT, SparseVector, DenseVector
from synapse.ml.lightgbm import LightGBMClassifier

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
    
    # LightGBM hyperparameters (optimized for hidden test generalization)
    p.add_argument("--numLeaves", type=int, default=50,
                   help="Max tree leaves (lower = less overfit, default 50 from V1 Best)")
    p.add_argument("--learningRate", type=float, default=0.05,
                   help="Learning rate (default 0.05 from V1 Best tuning)")
    p.add_argument("--numIterations", type=int, default=500)
    p.add_argument("--earlyStoppingRound", type=int, default=50)
    p.add_argument("--featureFraction", type=float, default=0.8)
    p.add_argument("--baggingFraction", type=float, default=0.8)
    p.add_argument("--minDataInLeaf", type=int, default=50,
                   help="Min samples per leaf (higher = less overfit, default 50)")
    p.add_argument("--maxDepth", type=int, default=-1)
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


LEAKY_COLS = {
    "helpful_votes", "total_votes", "helpful_ratio", "vote_ratio",
    "is_helpful_times_len", "helpfulness_x_length", "label_ratio",
    "probability_helpful", "helpful", "target_helpful"
}


def drop_leaky_columns(df, features_col, label_col):
    """Drop columns that might leak the ground-truth label."""
    cols = set(df.columns)
    bad = [c for c in LEAKY_COLS if c in cols and c not in {features_col, label_col}]
    if bad:
        print(f"[WARN] Dropping potential leaky columns: {bad}")
        for c in bad:
            df = df.drop(c)
    return df


def get_vector_size(df, features_col):
    """Extract the dimension of the feature vector."""
    sample = df.select(features_col).first()
    if sample is None:
        raise RuntimeError(f"Cannot determine vector size: no data in '{features_col}'")
    vec = sample[0]
    if isinstance(vec, (SparseVector, DenseVector)):
        return vec.size
    elif hasattr(vec, 'size'):
        return vec.size
    else:
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


def compute_class_weight(df, label_col, weight_col="weight", pos_weight=None):
    """
    Compute class weight for imbalance handling.
    pos_weight: 'auto' → N_neg/N_pos, or float value, or None (no weighting)
    """
    agg = df.groupBy().agg(
        F.sum(F.when(F.col(label_col) == 1, 1).otherwise(0)).alias("pos"),
        F.sum(F.when(F.col(label_col) == 0, 1).otherwise(0)).alias("neg"),
    ).collect()[0]
    pos, neg = int(agg["pos"]), int(agg["neg"])
    
    if pos == 0 or neg == 0:
        raise RuntimeError(f"Class collapse detected (pos={pos}, neg={neg}). Check labels.")
    
    if pos_weight == "auto":
        w1 = max(0.1, min(10.0, float(neg) / float(pos)))  # clamp to [0.1, 10]
        print(f"[INFO] Auto-computed posWeight = {w1:.3f} (neg={neg}, pos={pos})")
    elif pos_weight is not None:
        w1 = float(pos_weight)
        print(f"[INFO] Using manual posWeight = {w1:.3f}")
    else:
        w1 = 1.0
        print(f"[INFO] No class weighting (posWeight = 1.0)")
    
    df = df.withColumn(weight_col, 
                       F.when(F.col(label_col) == 1, F.lit(w1)).otherwise(F.lit(1.0)))
    return df, w1, pos, neg


def evaluate_model(model, df, label_col, stage_name="VAL"):
    """
    Evaluate model and return comprehensive metrics.
    Returns: (aucpr, aucroc, precision, recall, f1, confusion_matrix, pred_df)
    """
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    eval_pr = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    eval_roc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    
    pred_df = model.transform(df)
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
    
    # Compute Confusion Matrix
    cm_df = pred_df.groupBy(label_col, "prediction").count().collect()
    confusion_matrix = {}
    for row in cm_df:
        key = f"true_{int(row[label_col])}_pred_{int(row['prediction'])}"
        confusion_matrix[key] = int(row['count'])
    
    # Extract TP, TN, FP, FN
    tp = confusion_matrix.get("true_1_pred_1", 0)
    tn = confusion_matrix.get("true_0_pred_0", 0)
    fp = confusion_matrix.get("true_0_pred_1", 0)
    fn = confusion_matrix.get("true_1_pred_0", 0)
    
    print(f"[{stage_name}] AUC-PR={aucpr:.4f} | AUC-ROC={aucroc:.4f}")
    print(f"[{stage_name}] Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}")
    print(f"[{stage_name}] Confusion Matrix: TP={tp:,} TN={tn:,} FP={fp:,} FN={fn:,}")
    
    metrics = {
        "aucpr": float(aucpr),
        "aucroc": float(aucroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }
    }
    
    return metrics, pred_df


def pseudo_label_iteration(model, unlabeled_df, label_col, features_col, 
                           min_prob=0.9, top_pct=0.1, pseudo_weight=0.3):
    """
    Pseudo-labeling: predict on unlabeled data, select high-confidence samples.
    Returns DataFrame with pseudo-labels and low weight.
    """
    pred_df = model.transform(unlabeled_df)
    
    # Extract probability for class 1
    get_prob_udf = F.udf(lambda v: float(v[1]) if v and len(v) > 1 else 0.0, T.FloatType())
    pred_df = pred_df.withColumn("prob_class1", get_prob_udf(F.col("probability")))
    
    # Select confident positive and negative samples
    confident_pos = pred_df.filter(F.col("prob_class1") >= min_prob)
    confident_neg = pred_df.filter(F.col("prob_class1") <= (1 - min_prob))
    
    # Take top % by confidence
    n_pos = int(confident_pos.count() * top_pct)
    n_neg = int(confident_neg.count() * top_pct)
    
    if n_pos == 0 and n_neg == 0:
        print("[WARN] No confident pseudo-labels found")
        return None
    
    pseudo_pos = confident_pos.orderBy(F.desc("prob_class1")).limit(n_pos) \
        .withColumn(label_col, F.lit(1))
    pseudo_neg = confident_neg.orderBy(F.asc("prob_class1")).limit(n_neg) \
        .withColumn(label_col, F.lit(0))
    
    pseudo_df = pseudo_pos.unionByName(pseudo_neg, allowMissingColumns=True)
    pseudo_df = pseudo_df.withColumn("weight", F.lit(pseudo_weight))
    
    # Keep only necessary columns
    cols_to_keep = [c for c in unlabeled_df.columns] + [label_col, "weight"]
    pseudo_df = pseudo_df.select(*[c for c in cols_to_keep if c in pseudo_df.columns])
    
    print(f"[PSEUDO] Added {n_pos} positive + {n_neg} negative pseudo-labels (weight={pseudo_weight})")
    return pseudo_df


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
                        print(f"  - Train: metadata → text → sentiment (~37 features)")
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
                        print(f"     metadata_features_v2.py → text_preprocessing_v2.py → sentiment_vader_v2.py")
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
        train_split, w1, n_pos, n_neg = compute_class_weight(
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
        metrics, _ = evaluate_model(model, val_split, args.label_col, stage_name="VAL-INITIAL")
        
        best_aucpr = metrics["aucpr"]
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
                                          stage_name=f"VAL-PSEUDO-R{round_idx+1}")
                
                aucpr = metrics["aucpr"]
                
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
                writer.writerow(["AUC-PR", f"{best_metrics['aucpr']:.4f}"])
                writer.writerow(["AUC-ROC", f"{best_metrics['aucroc']:.4f}"])
                writer.writerow(["Precision", f"{best_metrics['precision']:.4f}"])
                writer.writerow(["Recall", f"{best_metrics['recall']:.4f}"])
                writer.writerow(["F1-Score", f"{best_metrics['f1']:.4f}"])
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
                f.write(f"AUC-PR (Primary Metric):  {best_metrics['aucpr']:.4f}\n")
                f.write(f"AUC-ROC:                  {best_metrics['aucroc']:.4f}\n")
                f.write(f"Precision:                {best_metrics['precision']:.4f}\n")
                f.write(f"Recall:                   {best_metrics['recall']:.4f}\n")
                f.write(f"F1-Score:                 {best_metrics['f1']:.4f}\n\n")
                
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
        print(f"VAL_AUCPR     = {best_metrics['aucpr']:.4f}")
        print(f"VAL_AUCROC    = {best_metrics['aucroc']:.4f}")
        print(f"VAL_Precision = {best_metrics['precision']:.4f}")
        print(f"VAL_Recall    = {best_metrics['recall']:.4f}")
        print(f"VAL_F1        = {best_metrics['f1']:.4f}")
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


