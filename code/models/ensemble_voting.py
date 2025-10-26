#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Voting Classifier - Day 6
Weighted average of LogisticRegression (0.3) + LightGBM (0.7) predictions

Strategy:
- Load both trained models (LogReg 0.5441 AUC-PR, LightGBM 0.6548 AUC-PR)
- Apply both models to validation set
- Weighted average: 0.3 * P(LogReg) + 0.7 * P(LightGBM)
- Expected: AUC-PR ~0.68-0.70 (+3-7% improvement)

Author: Lê Đăng Hoàng Tuấn & Võ Thị Diễm Thanh
Date: October 27, 2025
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import json
import time
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Import SynapseML LightGBM to register the class with Py4J
try:
    from synapse.ml.lightgbm import LightGBMClassificationModel, LightGBMClassifier
    print("✓ SynapseML LightGBM imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: Could not import SynapseML LightGBM: {e}")
    print("   Attempting to continue anyway...")


def evaluate_model(predictions, label_col="is_helpful"):
    """Evaluate predictions with multiple metrics"""
    
    # AUC-ROC
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="probability_ensemble",
        metricName="areaUnderROC"
    )
    auc_roc = evaluator_roc.evaluate(predictions)
    
    # AUC-PR (PRIMARY METRIC)
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="probability_ensemble",
        metricName="areaUnderPR"
    )
    auc_pr = evaluator_pr.evaluate(predictions)
    
    # Convert probability to binary prediction (threshold=0.5)
    predictions = predictions.withColumn(
        "prediction",
        F.when(F.col("probability_ensemble") >= 0.5, 1.0).otherwise(0.0)
    )
    
    # Accuracy
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)
    
    # Precision
    evaluator_prec = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    precision = evaluator_prec.evaluate(predictions)
    
    # Recall
    evaluator_rec = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    recall = evaluator_rec.evaluate(predictions)
    
    # F1
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)
    
    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    parser = argparse.ArgumentParser(description="Ensemble Voting - Day 6")
    parser.add_argument("--test", required=True, help="Path to test parquet (HDFS)")
    parser.add_argument("--model_logreg", required=True, help="Path to LogReg model")
    parser.add_argument("--model_lightgbm", required=True, help="Path to LightGBM model")
    parser.add_argument("--weight_logreg", type=float, default=0.3, help="Weight for LogReg (default: 0.3)")
    parser.add_argument("--weight_lightgbm", type=float, default=0.7, help="Weight for LightGBM (default: 0.7)")
    parser.add_argument("--out", required=True, help="Output directory for metrics")
    args = parser.parse_args()
    
    # Validate weights
    if abs(args.weight_logreg + args.weight_lightgbm - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0: {args.weight_logreg} + {args.weight_lightgbm} = {args.weight_logreg + args.weight_lightgbm}")
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("EnsembleVoting") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "6g") \
        .getOrCreate()
    
    try:
        print("\n" + "="*80)
        print("ENSEMBLE VOTING CLASSIFIER - DAY 6")
        print("="*80)
        print(f"Test data:      {args.test}")
        print(f"LogReg model:   {args.model_logreg}")
        print(f"LightGBM model: {args.model_lightgbm}")
        print(f"Weights: LogReg={args.weight_logreg}, LightGBM={args.weight_lightgbm}")
        print(f"Output:         {args.out}")
        
        start_time = time.time()
        
        # Load test data
        print("\n[1/5] Loading test data...")
        test_df = spark.read.parquet(args.test)
        n_test = test_df.count()
        print(f"   Test records: {n_test:,}")
        
        # Fill NULL values in metadata features (critical for VectorAssembler)
        print("   Filling NULL values in metadata features...")
        test_df = test_df.fillna({
            'price': 0.0,
            'product_avg_rating_meta': 3.5,  # median rating
            'product_total_ratings': 0.0,
            'sentiment_compound': 0.0,
            'sentiment_pos': 0.0,
            'sentiment_neg': 0.0,
            'sentiment_neu': 1.0,  # default neutral
            'rating_deviation': 0.0,
            'is_long_review': 0,
            'review_length_log': 0.0
        })
        
        # Load LogReg model
        print("\n[2/5] Loading LogisticRegression model...")
        logreg_model = PipelineModel.load(args.model_logreg)
        print(f"   Model loaded: {args.model_logreg}")
        
        # Load LightGBM model
        print("\n[3/5] Loading LightGBM model...")
        lightgbm_model = PipelineModel.load(args.model_lightgbm)
        print(f"   Model loaded: {args.model_lightgbm}")
        
        # Apply both models
        print("\n[4/5] Applying both models to test set...")
        
        print("   [4.1] Applying LogisticRegression...")
        logreg_pred = logreg_model.transform(test_df)
        
        # Extract probability for class 1 from LogReg
        # LogReg probability column is a vector, need to extract element [1]
        from pyspark.ml.functions import vector_to_array
        logreg_pred = logreg_pred.withColumn(
            "prob_logreg",
            vector_to_array(F.col("probability"))[1]
        )
        
        print("   [4.2] Applying LightGBM...")
        lightgbm_pred = lightgbm_model.transform(test_df)
        
        # Extract probability for class 1 from LightGBM
        lightgbm_pred = lightgbm_pred.withColumn(
            "prob_lightgbm",
            vector_to_array(F.col("probability"))[1]
        )
        
        # Merge predictions
        print("   [4.3] Merging predictions...")
        # Join on index (or use monotonically_increasing_id if no index)
        logreg_pred = logreg_pred.withColumn("row_id", F.monotonically_increasing_id())
        lightgbm_pred = lightgbm_pred.withColumn("row_id", F.monotonically_increasing_id())
        
        # Select needed columns
        logreg_probs = logreg_pred.select("row_id", "prob_logreg", "is_helpful")
        lightgbm_probs = lightgbm_pred.select("row_id", "prob_lightgbm")
        
        # Join
        ensemble_pred = logreg_probs.join(lightgbm_probs, on="row_id", how="inner")
        
        # Weighted average
        print(f"   [4.4] Computing weighted average: {args.weight_logreg} * LogReg + {args.weight_lightgbm} * LightGBM...")
        ensemble_pred = ensemble_pred.withColumn(
            "probability_ensemble",
            (F.col("prob_logreg") * args.weight_logreg) + (F.col("prob_lightgbm") * args.weight_lightgbm)
        )
        
        # Show sample
        print("\n   Sample predictions:")
        ensemble_pred.select([
            "is_helpful", "prob_logreg", "prob_lightgbm", "probability_ensemble"
        ]).show(10, truncate=False)
        
        # Evaluate ensemble
        print("\n[5/5] Evaluating Ensemble Voting model...")
        metrics = evaluate_model(ensemble_pred, label_col="is_helpful")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Display results
        print("\n" + "="*80)
        print("ENSEMBLE VOTING RESULTS")
        print("="*80)
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"AUC-PR:    {metrics['auc_pr']:.4f}  <-- PRIMARY METRIC")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"\nTime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Compare with individual models
        print("\n" + "="*80)
        print("COMPARISON WITH INDIVIDUAL MODELS")
        print("="*80)
        print("Day 5 Results (from metrics.json):")
        print("  - Tuned LogReg:  AUC-PR = 0.5441")
        print("  - LightGBM:      AUC-PR = 0.6548")
        print(f"\nDay 6 Ensemble:    AUC-PR = {metrics['auc_pr']:.4f}")
        
        improvement_vs_logreg = ((metrics['auc_pr'] - 0.5441) / 0.5441) * 100
        improvement_vs_lightgbm = ((metrics['auc_pr'] - 0.6548) / 0.6548) * 100
        
        print(f"\nImprovement vs LogReg:   {improvement_vs_logreg:+.1f}%")
        print(f"Improvement vs LightGBM: {improvement_vs_lightgbm:+.1f}%")
        
        if metrics['auc_pr'] > 0.6548:
            print("\n✅ SUCCESS! Ensemble outperforms LightGBM!")
        elif metrics['auc_pr'] > 0.6500:
            print("\n✅ Good! Ensemble close to LightGBM performance")
        else:
            print("\n⚠️  Ensemble underperforms. Consider adjusting weights.")
        
        # Check Day 6 target
        if metrics['auc_pr'] >= 0.70:
            print(f"\n🎯 DAY 6 TARGET ACHIEVED! AUC-PR = {metrics['auc_pr']:.4f} >= 0.70")
        else:
            gap = 0.70 - metrics['auc_pr']
            print(f"\n📊 Day 6 Target: 0.70, Current: {metrics['auc_pr']:.4f}, Gap: {gap:.4f}")
        
        # Save metrics
        output_metrics = {
            "model": "Ensemble Voting (LogReg + LightGBM)",
            "weights": {
                "logreg": args.weight_logreg,
                "lightgbm": args.weight_lightgbm
            },
            "metrics": {
                "auc_roc": float(metrics['auc_roc']),
                "auc_pr": float(metrics['auc_pr']),
                "accuracy": float(metrics['accuracy']),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "f1": float(metrics['f1'])
            },
            "comparison": {
                "logreg_auc_pr": 0.5441,
                "lightgbm_auc_pr": 0.6548,
                "ensemble_auc_pr": float(metrics['auc_pr']),
                "improvement_vs_logreg_pct": float(improvement_vs_logreg),
                "improvement_vs_lightgbm_pct": float(improvement_vs_lightgbm)
            },
            "test_records": n_test,
            "time_seconds": elapsed
        }
        
        import os
        os.makedirs(args.out, exist_ok=True)
        metrics_path = os.path.join(args.out, "metrics.json")
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(output_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Metrics saved to: {metrics_path}")
        
        print("\n" + "="*80)
        print("ENSEMBLE VOTING COMPLETED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
