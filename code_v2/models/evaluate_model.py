#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate trained LightGBM model on validation/test data with detailed metrics

Usage:
  spark-submit evaluate_model.py \
    --model_path hdfs://.../lightgbm_v11 \
    --data hdfs://.../features_train_final \
    --out_dir d:/reports/evaluation \
    [--sample_size 50000]
"""

import argparse
import json
import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from evaluation_v2 import (
    calculate_metrics, find_optimal_threshold,
    plot_confusion_matrix, plot_pr_roc_curves,
    plot_threshold_analysis, print_classification_report
)

try:
    from synapse.ml.lightgbm import LightGBMClassificationModel
except:
    LightGBMClassificationModel = None


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained LightGBM model")
    p.add_argument("--model_path", required=True, help="Path to trained model")
    p.add_argument("--data", required=True, help="Path to features parquet (with is_helpful label)")
    p.add_argument("--out_dir", required=True, help="Output directory for reports")
    p.add_argument("--sample_size", type=int, default=None, help="Sample size (None=all)")
    p.add_argument("--id_col", default="review_id", help="ID column name")
    p.add_argument("--features_col", default="features", help="Features column name")
    p.add_argument("--label_col", default="is_helpful", help="Label column name")
    return p.parse_args()


def build_spark():
    return (
        SparkSession.builder
        .appName("EvaluateLightGBM")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def load_model_metadata(spark, model_path):
    """Load metadata.json if exists"""
    try:
        from py4j.java_gateway import java_import
        java_import(spark._jvm, "org.apache.hadoop.fs.*")
        
        jvm = spark._jvm
        conf = spark._jsc.hadoopConfiguration()
        Path = jvm.org.apache.hadoop.fs.Path
        
        metadata_path = f"{model_path.rstrip('/')}/metadata.json"
        p = Path(metadata_path)
        fs = p.getFileSystem(conf)
        
        if fs.exists(p):
            stream = fs.open(p)
            content = stream.readLine()
            metadata = json.loads(content)
            return metadata
    except Exception as e:
        print(f"[WARN] Could not load metadata: {e}")
    
    return {}


def detect_model_kind(spark, model_path):
    """Detect if model is Pipeline or standalone LightGBM"""
    try:
        jvm = spark._jvm
        conf = spark._jsc.hadoopConfiguration()
        Path = jvm.org.apache.hadoop.fs.Path
        
        has_stages = Path(f"{model_path}/stages").getFileSystem(conf).exists(Path(f"{model_path}/stages"))
        has_booster = Path(f"{model_path}/complexParams/lightGBMBooster").getFileSystem(conf).exists(
            Path(f"{model_path}/complexParams/lightGBMBooster"))
        
        if has_stages:
            return "pipeline"
        if has_booster:
            return "lgbm"
    except:
        pass
    
    return "unknown"


def load_model(spark, model_path):
    """Load model (Pipeline or LightGBM)"""
    kind = detect_model_kind(spark, model_path)
    
    if kind == "pipeline":
        return PipelineModel.load(model_path)
    elif kind == "lgbm":
        if LightGBMClassificationModel is None:
            raise RuntimeError("LightGBMClassificationModel not available")
        return LightGBMClassificationModel.load(model_path)
    else:
        raise RuntimeError(f"Could not detect model type at {model_path}")


def main():
    args = parse_args()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*80)
    print("=== MODEL EVALUATION ===")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load metadata
    print(f"\n[INFO] Loading model metadata from {args.model_path}")
    metadata = load_model_metadata(spark, args.model_path)
    
    print("\n[INFO] Model Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Load model
    print(f"\n[INFO] Loading model from {args.model_path}")
    model = load_model(spark, args.model_path)
    print(f"[INFO] Model type: {type(model).__name__}")
    
    # Load data
    print(f"\n[INFO] Loading data from {args.data}")
    df = spark.read.parquet(args.data)
    
    # Check label column
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in data. "
                        f"Available columns: {df.columns}")
    
    # Sample if needed
    if args.sample_size:
        print(f"[INFO] Sampling {args.sample_size:,} rows...")
        df = df.sample(False, min(1.0, args.sample_size / df.count()), seed=42)
    
    total_count = df.count()
    print(f"[INFO] Evaluation data: {total_count:,} rows")
    
    # Class distribution
    class_dist = df.groupBy(args.label_col).count().collect()
    class_counts = {row[args.label_col]: row['count'] for row in class_dist}
    
    print("\n[INFO] Class Distribution:")
    for label, count in sorted(class_counts.items()):
        pct = count / total_count * 100
        print(f"  Class {label}: {count:,} ({pct:.2f}%)")
    
    # Predict
    print(f"\n[INFO] Running prediction...")
    predictions = model.transform(df)
    
    # Extract probability
    from pyspark.sql.types import DoubleType
    
    @F.udf(DoubleType())
    def extract_prob(v):
        try:
            if v is not None:
                return float(v[1])
        except:
            pass
        return 0.0
    
    predictions = predictions.withColumn(
        "prob_helpful",
        extract_prob(F.col("probability"))
    )
    
    # Collect results
    print(f"\n[INFO] Collecting results...")
    results = predictions.select(
        args.id_col,
        args.label_col,
        "prob_helpful"
    ).toPandas()
    
    y_true = results[args.label_col].values
    y_pred_proba = results["prob_helpful"].values
    
    print(f"[INFO] Evaluation set: {len(y_true):,} samples")
    print(f"  Positive class: {(y_true == 1).sum():,} ({(y_true == 1).mean()*100:.2f}%)")
    print(f"  Negative class: {(y_true == 0).sum():,} ({(y_true == 0).mean()*100:.2f}%)")
    
    # Calculate metrics at default threshold (0.5)
    print("\n" + "="*80)
    print("=== METRICS AT THRESHOLD = 0.5 ===")
    print("="*80)
    
    from sklearn.metrics import confusion_matrix
    
    y_pred_50 = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_50)
    
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (TN): {tn:,}")
    print(f"  False Positives (FP): {fp:,}")
    print(f"  False Negatives (FN): {fn:,}")
    print(f"  True Positives  (TP): {tp:,}")
    
    metrics_50 = calculate_metrics(y_true, y_pred_proba, threshold=0.5)
    
    print(f"\nMetrics:")
    print(f"  AUC-PR:    {metrics_50['auc_pr']:.4f}")
    print(f"  AUC-ROC:   {metrics_50['auc_roc']:.4f}")
    print(f"  Accuracy:  {metrics_50['accuracy']:.4f}")
    print(f"  Precision: {metrics_50['precision']:.4f}")
    print(f"  Recall:    {metrics_50['recall']:.4f}")
    print(f"  F1 Score:  {metrics_50['f1_score']:.4f}")
    
    # Find optimal threshold
    print("\n" + "="*80)
    print("=== OPTIMAL THRESHOLD ANALYSIS ===")
    print("="*80)
    
    opt_t, scores = find_optimal_threshold(y_true, y_pred_proba, metric="f1")
    
    y_pred_opt = (y_pred_proba >= opt_t).astype(int)
    cm_opt = confusion_matrix(y_true, y_pred_opt)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    
    print(f"\nConfusion Matrix at Optimal Threshold ({opt_t:.2f}):")
    print(f"  True Negatives  (TN): {tn_opt:,}")
    print(f"  False Positives (FP): {fp_opt:,}")
    print(f"  False Negatives (FN): {fn_opt:,}")
    print(f"  True Positives  (TP): {tp_opt:,}")
    
    metrics_opt = calculate_metrics(y_true, y_pred_proba, threshold=opt_t)
    
    print(f"\nMetrics:")
    print(f"  AUC-PR:    {metrics_opt['auc_pr']:.4f}")
    print(f"  AUC-ROC:   {metrics_opt['auc_roc']:.4f}")
    print(f"  Accuracy:  {metrics_opt['accuracy']:.4f}")
    print(f"  Precision: {metrics_opt['precision']:.4f}")
    print(f"  Recall:    {metrics_opt['recall']:.4f}")
    print(f"  F1 Score:  {metrics_opt['f1_score']:.4f}")
    
    # Classification report
    print_classification_report(y_true, y_pred_proba, threshold=opt_t)
    
    # Generate plots
    print("\n" + "="*80)
    print("=== GENERATING PLOTS ===")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Confusion matrix
    cm_path = os.path.join(args.out_dir, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(y_true, y_pred_opt, out_path=cm_path)
    
    # PR/ROC curves
    curves_path = os.path.join(args.out_dir, f"pr_roc_curves_{timestamp}.png")
    plot_pr_roc_curves(y_true, y_pred_proba, out_path=curves_path)
    
    # Threshold analysis
    thresh_path = os.path.join(args.out_dir, f"threshold_analysis_{timestamp}.png")
    plot_threshold_analysis(y_true, y_pred_proba, out_path=thresh_path)
    
    # Save summary report
    report_path = os.path.join(args.out_dir, f"evaluation_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Data Path: {args.data}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Metadata:\n")
        for key, value in metadata.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nDataset Statistics:\n")
        f.write(f"  Total Samples: {total_count:,}\n")
        f.write(f"  Positive Class: {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total_count*100:.2f}%)\n")
        f.write(f"  Negative Class: {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total_count*100:.2f}%)\n")
        
        f.write(f"\n" + "="*80 + "\n")
        f.write("METRICS AT THRESHOLD = 0.5\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN: {tn:,}  |  FP: {fp:,}\n")
        f.write(f"  FN: {fn:,}  |  TP: {tp:,}\n\n")
        
        for key, value in metrics_50.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\n" + "="*80 + "\n")
        f.write(f"OPTIMAL THRESHOLD = {opt_t:.2f}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN: {tn_opt:,}  |  FP: {fp_opt:,}\n")
        f.write(f"  FN: {fn_opt:,}  |  TP: {tp_opt:,}\n\n")
        
        for key, value in metrics_opt.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\n[INFO] Evaluation report saved to {report_path}")
    
    print("\n" + "="*80)
    print("=== EVALUATION COMPLETE ===")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - Report:      {report_path}")
    print(f"  - Conf Matrix: {cm_path}")
    print(f"  - PR/ROC:      {curves_path}")
    print(f"  - Threshold:   {thresh_path}")
    
    spark.stop()


if __name__ == "__main__":
    main()
