# code/models/baseline_dummy.py
# Day 1 Fix: Baseline benchmark với dummy classifier
"""
Usage:
    spark-submit code/models/baseline_dummy.py \
        --train hdfs://localhost:9000/datasets/amazon/movies/parquet/train \
        --test hdfs://localhost:9000/datasets/amazon/movies/parquet/test \
        --out output/baseline_dummy.json
"""

import argparse, json, time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Train parquet path")
    ap.add_argument("--test", required=True, help="Test parquet path")
    ap.add_argument("--out", default="output/baseline_dummy.json", help="Output JSON path")
    return ap.parse_args()

def main():
    args = parse_args()
    
    spark = SparkSession.builder\
        .appName("BaselineDummy")\
        .config("spark.driver.memory", "4g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("BASELINE DUMMY CLASSIFIER")
    print(f"{'='*60}\n")
    
    # Load train data
    print(f"Loading train from: {args.train}")
    train = spark.read.parquet(args.train).select("is_helpful")
    
    # Load test data
    print(f"Loading test from: {args.test}")
    test = spark.read.parquet(args.test).select("is_helpful")
    
    # Get class distribution from training set
    train_dist = train.groupBy("is_helpful").count().collect()
    train_stats = {int(r["is_helpful"]): int(r["count"]) for r in train_dist}
    
    total_train = sum(train_stats.values())
    pos_train = train_stats.get(1, 0)
    neg_train = train_stats.get(0, 0)
    
    print(f"\nTrain set:")
    print(f"  Total: {total_train:,}")
    print(f"  Positive: {pos_train:,} ({pos_train/total_train:.2%})")
    print(f"  Negative: {neg_train:,} ({neg_train/total_train:.2%})")
    
    # Strategy 1: Always predict majority class (most_frequent)
    majority_class = 0 if neg_train > pos_train else 1
    print(f"\nStrategy 1: Always predict class {majority_class} (majority)")
    
    test_pd = test.toPandas()
    y_true = test_pd["is_helpful"].values
    
    # Most frequent
    y_pred_freq = np.full(len(y_true), majority_class)
    y_prob_freq = np.full(len(y_true), 1.0 if majority_class == 1 else 0.0)
    
    # Accuracy
    acc_freq = (y_pred_freq == y_true).mean()
    
    # AUC-PR (always predict same class = no discrimination)
    # For most_frequent, AUC-PR ≈ positive ratio
    pos_ratio = (y_true == 1).mean()
    ap_freq = pos_ratio  # Baseline is just the positive ratio
    
    print(f"  Accuracy: {acc_freq:.4f}")
    print(f"  AUC-PR: {ap_freq:.4f} (= positive ratio)")
    
    # Strategy 2: Random stratified (predict based on class distribution)
    print(f"\nStrategy 2: Random stratified (prob = class ratio)")
    
    pos_prob = pos_train / total_train
    y_prob_random = np.full(len(y_true), pos_prob)
    
    # AUC-PR for random
    ap_random = average_precision_score(y_true, y_prob_random)
    
    print(f"  Positive probability: {pos_prob:.4f}")
    print(f"  AUC-PR: {ap_random:.4f}")
    
    # Strategy 3: Uniform random (0.5)
    print(f"\nStrategy 3: Uniform random (prob = 0.5)")
    
    y_prob_uniform = np.full(len(y_true), 0.5)
    ap_uniform = average_precision_score(y_true, y_prob_uniform)
    
    print(f"  AUC-PR: {ap_uniform:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("BASELINE BENCHMARKS")
    print(f"{'='*60}")
    print(f"1. Most Frequent (always predict {majority_class}):")
    print(f"   - Accuracy: {acc_freq:.4f}")
    print(f"   - AUC-PR: {ap_freq:.4f}")
    print(f"\n2. Random Stratified (prob = {pos_prob:.4f}):")
    print(f"   - AUC-PR: {ap_random:.4f}")
    print(f"\n3. Uniform Random (prob = 0.5):")
    print(f"   - AUC-PR: {ap_uniform:.4f}")
    
    print(f"\n{'='*60}")
    print("TARGET FOR DAY 3 BASELINE MODEL:")
    print(f"AUC-PR > {max(ap_freq, ap_random, ap_uniform):.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    results = {
        "timestamp": time.time(),
        "train_samples": total_train,
        "test_samples": len(y_true),
        "positive_ratio": float(pos_ratio),
        "baselines": {
            "most_frequent": {
                "accuracy": float(acc_freq),
                "auc_pr": float(ap_freq)
            },
            "random_stratified": {
                "auc_pr": float(ap_random)
            },
            "uniform_random": {
                "auc_pr": float(ap_uniform)
            }
        },
        "target_auc_pr": float(max(ap_freq, ap_random, ap_uniform))
    }
    
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.out}")
    
    spark.stop()

if __name__ == "__main__":
    main()
