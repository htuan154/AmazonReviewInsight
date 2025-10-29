"""
Hyperparameter Tuning for LightGBM - FINAL VERSION (100% ASCII)
Day 7: Recovery Plan to achieve AUC-PR >= 0.70
"""

import argparse
import json
import os
import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from synapse.ml.lightgbm import LightGBMClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for LightGBM')
    parser.add_argument('--train', required=True, help='Path to training parquet')
    parser.add_argument('--test', required=True, help='Path to test parquet')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (9 combinations)')
    return parser.parse_args()


def create_feature_pipeline(num_features=20000, min_doc_freq=10):
    """Create TF-IDF + Metadata feature pipeline"""
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures",
                          numFeatures=num_features)
    idf = IDF(inputCol="rawFeatures", outputCol="text_features",
              minDocFreq=min_doc_freq)
    
    metadata_cols = [
        "star_rating", "review_length_log",
        "sentiment_compound", "sentiment_pos", "sentiment_neu", "sentiment_neg",
        "rating_deviation", "is_long_review",
        "price", "product_avg_rating_meta", "product_total_ratings"
    ]
    
    assembler = VectorAssembler(
        inputCols=["text_features"] + metadata_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    
    return Pipeline(stages=[tokenizer, hashingTF, idf, assembler])


def manual_cv_tuning(spark, train_df, feature_pipeline, param_combos, num_folds=3):
    """Manual cross-validation with parameter grid"""
    print("\n" + "="*80)
    print("STARTING MANUAL CROSS-VALIDATION TUNING")
    print("="*80)
    print("[DATA] Total Combinations: {}".format(len(param_combos)))
    print("[DATA] CV Folds: {}".format(num_folds))
    print("[DATA] Total Trainings: {}".format(len(param_combos) * num_folds))
    print("="*80 + "\n")
    
    start_time = time.time()
    
    if "row_id" not in train_df.columns:
        train_df = train_df.withColumn("row_id", monotonically_increasing_id())
    
    train_df = train_df.cache()
    total_count = train_df.count()
    fold_size = total_count // num_folds
    
    print("[DATA] Training Data: {:,} records".format(total_count))
    print("[DATA] Each Fold: ~{:,} records\n".format(fold_size))
    
    print("[INFO] Fitting feature pipeline...")
    pipeline_start = time.time()
    feature_model = feature_pipeline.fit(train_df)
    pipeline_time = time.time() - pipeline_start
    print("[OK] Pipeline fitted in {:.1f}s\n".format(pipeline_time))
    
    print("[INFO] Transforming training data...")
    transform_start = time.time()
    train_transformed = feature_model.transform(train_df).cache()
    transform_time = time.time() - transform_start
    print("[OK] Data transformed in {:.1f}s\n".format(transform_time))
    
    all_results = []
    best_score = 0.0
    best_params = None
    best_idx = -1
    
    for idx, params in enumerate(param_combos, 1):
        print("\n" + "-"*80)
        print("[LOOP] Combination {}/{}".format(idx, len(param_combos)))
        print("-"*80)
        print("Parameters:")
        for k, v in params.items():
            print("  - {}: {}".format(k, v))
        print()
        
        fold_scores = []
        fold_times = []
        
        for fold in range(num_folds):
            fold_start = time.time()
            
            validation_df = train_transformed.filter((col("row_id") % num_folds) == fold)
            train_fold_df = train_transformed.filter((col("row_id") % num_folds) != fold)
            
            lightgbm = LightGBMClassifier(
                objective="binary",
                featuresCol="features",
                labelCol="is_helpful",
                isUnbalance=True,
                verbosity=-1,
                numThreads=4,
                numLeaves=params.get('numLeaves', 31),
                learningRate=params.get('learningRate', 0.1),
                numIterations=params.get('numIterations', 100),
                minDataInLeaf=params.get('minDataInLeaf', 20),
                featureFraction=params.get('featureFraction', 0.8)
            )
            
            model = lightgbm.fit(train_fold_df)
            predictions = model.transform(validation_df)
            
            evaluator = BinaryClassificationEvaluator(
                labelCol="is_helpful",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderPR"
            )
            score = evaluator.evaluate(predictions)
            
            fold_time = time.time() - fold_start
            fold_scores.append(score)
            fold_times.append(fold_time)
            
            print("  Fold {}: AUC-PR = {:.4f} ({:.1f}s)".format(fold+1, score, fold_time))
        
        avg_score = sum(fold_scores) / len(fold_scores)
        std_score = (sum((s - avg_score)**2 for s in fold_scores) / len(fold_scores)) ** 0.5
        avg_time = sum(fold_times) / len(fold_times)
        
        print("\n  [DATA] CV Results:")
        print("    - Mean AUC-PR: {:.4f} (+/-{:.4f})".format(avg_score, std_score))
        print("    - Min: {:.4f}, Max: {:.4f}".format(min(fold_scores), max(fold_scores)))
        print("    - Avg Time/Fold: {:.1f}s".format(avg_time))
        
        result = {
            'combination': idx,
            'params': params,
            'cv_scores': fold_scores,
            'mean_score': avg_score,
            'std_score': std_score,
            'min_score': min(fold_scores),
            'max_score': max(fold_scores),
            'avg_time_seconds': avg_time
        }
        all_results.append(result)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_idx = idx
            print("  [WIN] NEW BEST! (Previous: {:.4f})".format(best_score))
        
        elapsed = time.time() - start_time
        avg_time_per_combo = elapsed / idx
        remaining_combos = len(param_combos) - idx
        eta_seconds = avg_time_per_combo * remaining_combos
        eta_minutes = eta_seconds / 60
        
        print("\n  [TIME] Progress: {}/{} ({:.1f}%)".format(
            idx, len(param_combos), 100*idx/len(param_combos)))
        print("  [TIME] Elapsed: {:.1f}m | ETA: {:.1f}m".format(
            elapsed/60, eta_minutes))
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TUNING COMPLETED!")
    print("="*80)
    print("[TIME] Total Time: {:.1f} minutes ({:.1f} seconds)".format(
        total_time/60, total_time))
    print("[WIN] Best Combination: #{}".format(best_idx))
    print("[WIN] Best CV Score: {:.4f}".format(best_score))
    print("\n[NOTE] Best Parameters:")
    for k, v in best_params.items():
        print("  - {}: {}".format(k, v))
    print("="*80 + "\n")
    
    train_df.unpersist()
    train_transformed.unpersist()
    
    return best_params, best_score, all_results, total_time, feature_model


def train_final_model(spark, train_df, test_df, feature_model, best_params):
    """Train final model with best parameters"""
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*80)
    
    start_time = time.time()
    
    print("[INFO] Transforming train/test data...")
    train_transformed = feature_model.transform(train_df)
    test_transformed = feature_model.transform(test_df)
    
    lightgbm = LightGBMClassifier(
        objective="binary",
        featuresCol="features",
        labelCol="is_helpful",
        isUnbalance=True,
        verbosity=1,
        numThreads=4,
        numLeaves=best_params.get('numLeaves', 31),
        learningRate=best_params.get('learningRate', 0.1),
        numIterations=best_params.get('numIterations', 100),
        minDataInLeaf=best_params.get('minDataInLeaf', 20),
        featureFraction=best_params.get('featureFraction', 0.8)
    )
    
    print("\n[TOOL] Training on {:,} records...".format(train_df.count()))
    model = lightgbm.fit(train_transformed)
    train_time = time.time() - start_time
    
    print("[OK] Training completed in {:.1f}s ({:.2f}m)".format(
        train_time, train_time/60))
    
    print("\n[DATA] Evaluating on test set...")
    predictions = model.transform(test_transformed)
    
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="is_helpful",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol="is_helpful",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    auc_pr = evaluator_pr.evaluate(predictions)
    auc_roc = evaluator_roc.evaluate(predictions)
    
    from pyspark.sql.functions import when
    accuracy_df = predictions.withColumn(
        "correct",
        when(col("prediction") == col("is_helpful"), 1).otherwise(0)
    )
    accuracy = accuracy_df.agg({"correct": "avg"}).collect()[0][0]
    
    print("\n" + "="*80)
    print("FINAL MODEL RESULTS")
    print("="*80)
    print("[TARGET] AUC-PR:  {:.4f}".format(auc_pr))
    print("[TARGET] AUC-ROC: {:.4f}".format(auc_roc))
    print("[TARGET] Accuracy: {:.4f}".format(accuracy))
    print("="*80 + "\n")
    
    feature_importance = model.getFeatureImportances()
    
    metrics = {
        'auc_pr': auc_pr,
        'auc_roc': auc_roc,
        'accuracy': accuracy,
        'train_time_seconds': train_time,
        'train_time_minutes': train_time / 60,
        'feature_importance': feature_importance
    }
    
    return model, metrics, feature_model


def save_results(output_dir, best_params, best_cv_score, all_results, total_tuning_time,
                 final_metrics, baseline_auc_pr=0.6548):
    """Save all tuning results"""
    os.makedirs(output_dir, exist_ok=True)
    
    improvement_pct = ((final_metrics['auc_pr'] - baseline_auc_pr) / baseline_auc_pr) * 100
    target_achieved = final_metrics['auc_pr'] >= 0.70
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'LightGBM (Hyperparameter Tuned)',
        'tuning_strategy': 'Grid Search with 3-fold Cross-Validation',
        
        'tuning_summary': {
            'total_combinations_tested': len(all_results),
            'cv_folds': 3,
            'total_trainings': len(all_results) * 3,
            'total_tuning_time_seconds': total_tuning_time,
            'total_tuning_time_minutes': total_tuning_time / 60,
            'best_cv_score': best_cv_score
        },
        
        'best_parameters': best_params,
        
        'test_metrics': {
            'auc_pr': final_metrics['auc_pr'],
            'auc_roc': final_metrics['auc_roc'],
            'accuracy': final_metrics['accuracy']
        },
        
        'final_training': {
            'train_time_seconds': final_metrics['train_time_seconds'],
            'train_time_minutes': final_metrics['train_time_minutes']
        },
        
        'comparison_vs_baseline': {
            'baseline_model': 'LightGBM V2 (Day 5)',
            'baseline_auc_pr': baseline_auc_pr,
            'tuned_auc_pr': final_metrics['auc_pr'],
            'absolute_improvement': final_metrics['auc_pr'] - baseline_auc_pr,
            'percent_improvement': improvement_pct,
            'target': 0.70,
            'target_achieved': target_achieved,
            'gap_to_target': final_metrics['auc_pr'] - 0.70 if not target_achieved else 0
        },
        
        'top_10_configurations': sorted(
            all_results,
            key=lambda x: x['mean_score'],
            reverse=True
        )[:10],
        
        'feature_importance': final_metrics['feature_importance']
    }
    
    results_file = os.path.join(output_dir, 'tuning_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("[OK] Results saved to: {}".format(results_file))
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*80)
    print("[FIND] Combinations Tested: {}".format(len(all_results)))
    print("[FIND] Total Trainings: {}".format(len(all_results) * 3))
    print("[TIME] Total Time: {:.1f} minutes".format(total_tuning_time/60))
    print("\n[WIN] BEST CV SCORE: {:.4f}".format(best_cv_score))
    print("[TARGET] TEST AUC-PR: {:.4f}".format(final_metrics['auc_pr']))
    print("\n[DATA] IMPROVEMENT vs BASELINE (V2):")
    print("   Baseline: {:.4f}".format(baseline_auc_pr))
    print("   Tuned:    {:.4f}".format(final_metrics['auc_pr']))
    print("   Gain:     {:+.2f}%".format(improvement_pct))
    print("\n[TARGET] TARGET STATUS:")
    if target_achieved:
        print("   [OK] TARGET ACHIEVED! AUC-PR = {:.4f} >= 0.70".format(
            final_metrics['auc_pr']))
    else:
        gap = 0.70 - final_metrics['auc_pr']
        print("   [WARN] Gap to target: -{:.4f} ({:.1f}%)".format(
            gap, -gap/0.70*100))
    print("="*80 + "\n")
    
    return target_achieved


def main():
    args = parse_args()
    
    print("\n[INFO] Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("LightGBM_Hyperparameter_Tuning") \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("[OK] Spark initialized!")
    print("   Version: {}".format(spark.version))
    print("   Driver Memory: 8GB")
    print("   Executor Memory: 6GB\n")
    
    print("[INFO] Loading training data from: {}".format(args.train))
    train_df = spark.read.parquet(args.train)
    print("   [OK] Loaded {:,} training records".format(train_df.count()))
    
    print("[INFO] Loading test data from: {}".format(args.test))
    test_df = spark.read.parquet(args.test)
    print("   [OK] Loaded {:,} test records\n".format(test_df.count()))
    
    feature_pipeline = create_feature_pipeline()
    
    if args.quick:
        print("[INFO] QUICK MODE: Testing 9 combinations\n")
        param_combos = [
            {'numLeaves': 31, 'learningRate': 0.05},
            {'numLeaves': 31, 'learningRate': 0.1},
            {'numLeaves': 31, 'learningRate': 0.15},
            {'numLeaves': 50, 'learningRate': 0.05},
            {'numLeaves': 50, 'learningRate': 0.1},
            {'numLeaves': 50, 'learningRate': 0.15},
            {'numLeaves': 100, 'learningRate': 0.05},
            {'numLeaves': 100, 'learningRate': 0.1},
            {'numLeaves': 100, 'learningRate': 0.15},
        ]
    else:
        print("[INFO] FULL MODE: Testing 81 combinations\n")
        param_combos = []
        for leaves in [20, 31, 50]:
            for lr in [0.05, 0.1, 0.15]:
                for iters in [100, 150, 200]:
                    for min_leaf in [10, 20, 30]:
                        param_combos.append({
                            'numLeaves': leaves,
                            'learningRate': lr,
                            'numIterations': iters,
                            'minDataInLeaf': min_leaf
                        })
    
    best_params, best_cv_score, all_results, total_tuning_time, feature_model = manual_cv_tuning(
        spark, train_df, feature_pipeline, param_combos, num_folds=3
    )
    
    model, final_metrics, feature_model = train_final_model(
        spark, train_df, test_df, feature_model, best_params
    )
    
    model_path = os.path.join(args.out, 'model')
    print("\n[SAVE] Saving model to: {}".format(model_path))
    model.write().overwrite().save(model_path)
    print("   [OK] Model saved!")
    
    pipeline_path = os.path.join(args.out, 'feature_pipeline')
    print("[SAVE] Saving feature pipeline to: {}".format(pipeline_path))
    feature_model.write().overwrite().save(pipeline_path)
    print("   [OK] Pipeline saved!")
    
    target_achieved = save_results(
        args.out, best_params, best_cv_score, all_results,
        total_tuning_time, final_metrics
    )
    
    if target_achieved:
        print("\n" + "*** SUCCESS! TARGET ACHIEVED! AUC-PR >= 0.70 ***")
        print("*** DAY 6  COMPLETED 100%! ***\n")
    else:
        print("\n*** Target not reached. Next steps: ***")
        print("1. Try FULL MODE (without --quick)")
        print("2. Consider stacking ensemble")
        print("3. Advanced feature engineering\n")
    
    spark.stop()


if __name__ == '__main__':
    main()
