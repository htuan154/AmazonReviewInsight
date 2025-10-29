# code/models/tune_logreg.py
# Day 5 - Hyperparameter Tuning for Logistic Regression
"""
Usage:
    spark-submit --driver-memory 8g --executor-memory 6g \
        code/models/tune_logreg.py \
        --train hdfs://localhost:9000/datasets/amazon/movies/parquet/train \
        --test hdfs://localhost:9000/datasets/amazon/movies/parquet/test \
        --out output/logreg_tuned
"""

import argparse, json, time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv_folds", type=int, default=3, help="Cross-validation folds")
    return ap.parse_args()

def main():
    args = parse_args()
    spark = SparkSession.builder\
        .appName("Day5-LogReg-Tuning")\
        .config("spark.driver.memory", "8g")\
        .config("spark.executor.memory", "6g")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("DAY 5 - LOGISTIC REGRESSION HYPERPARAMETER TUNING")
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

    # Class weights
    counts = train.groupBy("is_helpful").count().collect()
    cnt = {int(r["is_helpful"]): int(r["count"]) for r in counts}
    pos, neg = cnt.get(1, 1), cnt.get(0, 1)
    pos_weight = float(neg) / float(max(pos,1))
    
    print(f"\nClass distribution: pos={pos:,} neg={neg:,} weight={pos_weight:.3f}")
    train = train.withColumn("weight", 
        F.when(F.col("is_helpful")==1, F.lit(pos_weight)).otherwise(F.lit(1.0)))

    # Build pipeline
    print(f"\n{'='*60}")
    print("BUILDING PIPELINE WITH PARAMETER GRID")
    print(f"{'='*60}\n")
    
    metadata_features = [
        "star_rating", "review_length_log",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "rating_deviation", "is_long_review",
        "price", "product_avg_rating_meta", "product_total_ratings"
    ]
    metadata_features = [f for f in metadata_features if f in available_cols]
    
    print(f"Metadata features ({len(metadata_features)}): {metadata_features}")
    
    # Pipeline stages
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=20000)
    idf = IDF(inputCol="raw_features", outputCol="tfidf", minDocFreq=5)
    assembler = VectorAssembler(inputCols=["tfidf"] + metadata_features, outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="is_helpful", weightCol="weight")
    
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler, lr])
    
    # Parameter grid
    print("\nParameter grid:")
    print("  numFeatures: [10000, 20000, 50000]")
    print("  minDocFreq: [3, 5, 10]")
    print("  regParam: [0.0, 0.001, 0.01]")
    print("  elasticNetParam: [0.0, 0.5]")
    print("  maxIter: [50, 100]")
    print(f"  Total combinations: 3 x 3 x 3 x 2 x 2 = 108")
    
    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [10000, 20000, 50000]) \
        .addGrid(idf.minDocFreq, [3, 5, 10]) \
        .addGrid(lr.regParam, [0.0, 0.001, 0.01]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
        .addGrid(lr.maxIter, [50, 100]) \
        .build()
    
    print(f"\nActual combinations: {len(paramGrid)}")
    
    # Cross-validation
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({args.cv_folds}-FOLD)")
    print(f"{'='*60}\n")
    
    evaluator = BinaryClassificationEvaluator(
        labelCol="is_helpful",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
    
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=args.cv_folds,
        seed=42,
        parallelism=2  # Run 2 models in parallel
    )
    
    print("Starting cross-validation...")
    print("This will take 20-30 minutes...")
    start_time = time.time()
    
    cvModel = cv.fit(train)
    cv_time = time.time() - start_time
    
    print(f"\nCross-validation completed in {cv_time/60:.1f} minutes")
    
    # Best model
    print(f"\n{'='*60}")
    print("BEST MODEL PARAMETERS")
    print(f"{'='*60}\n")
    
    bestModel = cvModel.bestModel
    stages = bestModel.stages
    
    best_hashingTF = stages[2]
    best_idf = stages[3]
    best_lr = stages[-1]
    
    print(f"Best parameters:")
    print(f"  numFeatures: {best_hashingTF.getNumFeatures()}")
    print(f"  minDocFreq: {best_idf.getMinDocFreq()}")
    print(f"  regParam: {best_lr.getRegParam()}")
    print(f"  elasticNetParam: {best_lr.getElasticNetParam()}")
    print(f"  maxIter: {best_lr.getMaxIter()}")
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    
    preds = bestModel.transform(val)
    
    eval_pr = BinaryClassificationEvaluator(
        labelCol="is_helpful", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    eval_roc = BinaryClassificationEvaluator(
        labelCol="is_helpful", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    eval_acc = MulticlassClassificationEvaluator(
        labelCol="is_helpful", predictionCol="prediction", metricName="accuracy")
    
    auc_pr = eval_pr.evaluate(preds)
    auc_roc = eval_roc.evaluate(preds)
    accuracy = eval_acc.evaluate(preds)
    
    print(f"Best model performance:")
    print(f"  AUC-PR: {auc_pr:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # CV metrics
    avg_metrics = cvModel.avgMetrics
    print(f"\nCross-validation metrics (avg AUC-PR):")
    top_5_idx = sorted(range(len(avg_metrics)), key=lambda i: avg_metrics[i], reverse=True)[:5]
    for i, idx in enumerate(top_5_idx, 1):
        print(f"  {i}. Config {idx}: {avg_metrics[idx]:.4f}")
    
    # Save
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}\n")
    
    model_path = f"{args.out}/model"
    bestModel.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save metrics
    metrics = {
        "timestamp": time.time(),
        "cv_folds": args.cv_folds,
        "cv_time_minutes": cv_time / 60,
        "num_param_combinations": len(paramGrid),
        "best_params": {
            "numFeatures": int(best_hashingTF.getNumFeatures()),
            "minDocFreq": int(best_idf.getMinDocFreq()),
            "regParam": float(best_lr.getRegParam()),
            "elasticNetParam": float(best_lr.getElasticNetParam()),
            "maxIter": int(best_lr.getMaxIter())
        },
        "test_metrics": {
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "accuracy": accuracy
        },
        "cv_avg_metrics": {
            "best": float(max(avg_metrics)),
            "worst": float(min(avg_metrics)),
            "mean": float(sum(avg_metrics) / len(avg_metrics))
        }
    }
    
    metrics_path = f"{args.out}/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    print(f"\n{'='*60}")
    print("DAY 5 - HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*60}")
    print(f"Best AUC-PR: {auc_pr:.4f}")
    print(f"Improvement vs Day 4 (0.4490): {(auc_pr - 0.4490) / 0.4490 * 100:+.1f}%")
    
    spark.stop()

if __name__ == "__main__":
    main()
