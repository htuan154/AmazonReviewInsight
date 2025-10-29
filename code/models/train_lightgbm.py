# code/models/train_lightgbm.py
# Day 5 - LightGBM Model (Tree-based, better with non-linear patterns)
"""
Usage:
    spark-submit --driver-memory 8g --executor-memory 6g \
        --packages com.microsoft.azure:synapseml_2.12:0.11.4 \
        code/models/train_lightgbm.py \
        --train hdfs://localhost:9000/datasets/amazon/movies/parquet/train \
        --test hdfs://localhost:9000/datasets/amazon/movies/parquet/test \
        --out output/lightgbm_model
"""

import argparse, json, time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from synapse.ml.lightgbm import LightGBMClassifier

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--numFeatures", type=int, default=20000, help="TF-IDF features")
    ap.add_argument("--minDocFreq", type=int, default=5, help="Min doc frequency")
    return ap.parse_args()

def main():
    args = parse_args()
    spark = SparkSession.builder\
        .appName("Day5-LightGBM")\
        .config("spark.driver.memory", "8g")\
        .config("spark.executor.memory", "6g")\
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.11.4")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("DAY 5 - LIGHTGBM TRAINING")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading train from: {args.train}")
    feature_cols = [
        "review_id", "review_text", "clean_text",
        "star_rating", "review_length", "review_length_log",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "rating_deviation", "is_long_review",
        "price", "product_avg_rating_meta", "product_total_ratings",
        "is_helpful"
    ]
    
    train_raw = spark.read.parquet(args.train)
    available_cols = [c for c in feature_cols if c in train_raw.columns]
    
    train = train_raw.select(*available_cols).na.fill({
        "review_text": "", "clean_text": "",
        "star_rating": 0.0, "review_length": 0, "review_length_log": 0.0,
        "sentiment_compound": 0.0, "sentiment_pos": 0.0, 
        "sentiment_neg": 0.0, "sentiment_neu": 0.0,
        "rating_deviation": 0.0, "is_long_review": 0,
        "price": 0.0, "product_avg_rating_meta": 0.0, "product_total_ratings": 0
    })
    
    print(f"Loading test from: {args.test}")
    val_raw = spark.read.parquet(args.test)
    val = val_raw.select(*available_cols).na.fill({
        "review_text": "", "clean_text": "",
        "star_rating": 0.0, "review_length": 0, "review_length_log": 0.0,
        "sentiment_compound": 0.0, "sentiment_pos": 0.0,
        "sentiment_neg": 0.0, "sentiment_neu": 0.0,
        "rating_deviation": 0.0, "is_long_review": 0,
        "price": 0.0, "product_avg_rating_meta": 0.0, "product_total_ratings": 0
    })
    
    print(f"\nTrain records: {train.count():,}")
    print(f"Test records: {val.count():,}")

    # Class distribution
    counts = train.groupBy("is_helpful").count().collect()
    cnt = {int(r["is_helpful"]): int(r["count"]) for r in counts}
    pos, neg = cnt.get(1, 1), cnt.get(0, 1)
    pos_weight = float(neg) / float(max(pos,1))
    
    print(f"\nClass distribution:")
    print(f"  Helpful (1): {pos:,} ({pos/(pos+neg)*100:.1f}%)")
    print(f"  Not Helpful (0): {neg:,} ({neg/(pos+neg)*100:.1f}%)")
    print(f"  Imbalance ratio: {pos_weight:.3f}")

    # Build pipeline
    print(f"\n{'='*60}")
    print("BUILDING PIPELINE WITH LIGHTGBM")
    print(f"{'='*60}\n")
    
    metadata_features = [
        "star_rating", "review_length_log",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "rating_deviation", "is_long_review",
        "price", "product_avg_rating_meta", "product_total_ratings"
    ]
    metadata_features = [f for f in metadata_features if f in available_cols]
    
    print(f"Metadata features ({len(metadata_features)}):")
    for i, feat in enumerate(metadata_features, 1):
        print(f"  {i}. {feat}")
    
    # Pipeline stages
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="raw_features", 
                          numFeatures=args.numFeatures)
    idf = IDF(inputCol="raw_features", outputCol="tfidf", 
              minDocFreq=args.minDocFreq)
    assembler = VectorAssembler(inputCols=["tfidf"] + metadata_features, 
                                outputCol="features")
    
    # LightGBM with optimized parameters
    print(f"\nLightGBM parameters:")
    print(f"  objective: binary")
    print(f"  numLeaves: 31 (default, good starting point)")
    print(f"  numIterations: 100")
    print(f"  learningRate: 0.1")
    print(f"  featureFraction: 0.8 (use 80% features per tree)")
    print(f"  baggingFraction: 0.8 (use 80% data per iteration)")
    print(f"  baggingFreq: 5 (bagging every 5 iterations)")
    print(f"  maxDepth: -1 (no limit)")
    print(f"  minSumHessianInLeaf: 0.001")
    print(f"  isUnbalance: true (handle class imbalance)")
    
    lgbm = LightGBMClassifier(
        featuresCol="features",
        labelCol="is_helpful",
        objective="binary",
        numLeaves=31,
        numIterations=100,
        learningRate=0.1,
        featureFraction=0.8,
        baggingFraction=0.8,
        baggingFreq=5,
        maxDepth=-1,
        minSumHessianInLeaf=0.001,
        isUnbalance=True  # Handle class imbalance
    )
    
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler, lgbm])
    
    # Train
    print(f"\n{'='*60}")
    print("TRAINING LIGHTGBM MODEL")
    print(f"{'='*60}\n")
    
    print("Starting training...")
    start_time = time.time()
    
    model = pipeline.fit(train)
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    
    preds = model.transform(val)
    
    eval_pr = BinaryClassificationEvaluator(
        labelCol="is_helpful", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    eval_roc = BinaryClassificationEvaluator(
        labelCol="is_helpful", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    eval_acc = MulticlassClassificationEvaluator(
        labelCol="is_helpful", predictionCol="prediction", metricName="accuracy")
    
    auc_pr = eval_pr.evaluate(preds)
    auc_roc = eval_roc.evaluate(preds)
    accuracy = eval_acc.evaluate(preds)
    
    print(f"LightGBM performance:")
    print(f"  AUC-PR: {auc_pr:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Comparison with previous models
    print(f"\n{'='*60}")
    print("COMPARISON WITH PREVIOUS MODELS")
    print(f"{'='*60}\n")
    
    day3_pr = 0.4353
    day4_pr = 0.4490
    
    print(f"Model Evolution:")
    print(f"  Day 3 Baseline (LogReg):        AUC-PR = {day3_pr:.4f}")
    print(f"  Day 4 Full Features (LogReg):   AUC-PR = {day4_pr:.4f} (+{(day4_pr-day3_pr)/day3_pr*100:.1f}%)")
    print(f"  Day 5 LightGBM (this model):    AUC-PR = {auc_pr:.4f} (+{(auc_pr-day4_pr)/day4_pr*100:.1f}% vs Day 4)")
    print(f"                                             (+{(auc_pr-day3_pr)/day3_pr*100:.1f}% vs Baseline)")
    
    if auc_pr >= 0.55:
        print(f"\n*** TARGET ACHIEVED! AUC-PR >= 0.55 ***")
    elif auc_pr >= 0.50:
        print(f"\n*** GOOD PROGRESS! AUC-PR >= 0.50 ***")
    elif auc_pr > day4_pr:
        print(f"\n*** IMPROVEMENT! Better than Day 4 LogReg ***")
    else:
        print(f"\nWARNING: Performance not improved vs Day 4")
        print(f"   -> LogReg may be better for this problem (linear separable)")
        print(f"   -> Or need to tune more LightGBM hyperparameters")
    
    # Feature importance (from LightGBM model)
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (TOP 20)")
    print(f"{'='*60}\n")
    
    lgbm_model = model.stages[-1]
    feature_importances = lgbm_model.getFeatureImportances()
    
    # Get metadata feature names (TF-IDF indices can't be interpreted)
    feature_names = ["TF-IDF"] + metadata_features
    
    # For LightGBM, importance is split/gain based
    # We'll print metadata feature importance only (interpretable)
    print(f"Note: TF-IDF has {args.numFeatures} dimensions (not shown individually)")
    print(f"\nMetadata feature importance:")
    
    # Extract metadata importances (indices after TF-IDF dimension)
    tfidf_dim = args.numFeatures
    metadata_importances = []
    
    for i, feat_name in enumerate(metadata_features):
        idx = tfidf_dim + i
        if idx < len(feature_importances):
            importance = float(feature_importances[idx])
            metadata_importances.append((feat_name, importance))
    
    # Sort by importance
    metadata_importances.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (feat, imp) in enumerate(metadata_importances, 1):
        print(f"  {rank:2d}. {feat:30s} {imp:8.2f}")
    
    # Save model
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}\n")
    
    model_path = f"{args.out}/model"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save metrics
    metrics = {
        "timestamp": time.time(),
        "model_type": "LightGBM",
        "train_time_seconds": train_time,
        "train_time_minutes": train_time / 60,
        "parameters": {
            "numFeatures": args.numFeatures,
            "minDocFreq": args.minDocFreq,
            "objective": "binary",
            "numLeaves": 31,
            "numIterations": 100,
            "learningRate": 0.1,
            "featureFraction": 0.8,
            "baggingFraction": 0.8,
            "baggingFreq": 5,
            "isUnbalance": True
        },
        "test_metrics": {
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "accuracy": accuracy
        },
        "comparison": {
            "day3_baseline": day3_pr,
            "day4_logreg": day4_pr,
            "improvement_vs_day4_pct": (auc_pr - day4_pr) / day4_pr * 100,
            "improvement_vs_baseline_pct": (auc_pr - day3_pr) / day3_pr * 100
        },
        "feature_importance": {
            "metadata": [{"feature": f, "importance": float(i)} 
                        for f, i in metadata_importances]
        }
    }
    
    metrics_path = f"{args.out}/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    print(f"\n{'='*60}")
    print("DAY 5 - LIGHTGBM TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Target: >= 0.55 ({auc_pr >= 0.55 and '*** ACHIEVED ***' or '*** NOT YET ***'})")
    
    spark.stop()

if __name__ == "__main__":
    main()
