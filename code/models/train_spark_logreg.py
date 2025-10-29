# code/models/train_spark_logreg.py
# Day 3 - Baseline Logistic Regression Model
# Usage:
# spark-submit --driver-memory 6g --executor-memory 4g \
#   code/models/train_spark_logreg.py \
#   --train hdfs://localhost:9000/datasets/amazon/movies/parquet/train \
#   --test hdfs://localhost:9000/datasets/amazon/movies/parquet/test \
#   --out hdfs://localhost:9000/datasets/amazon/movies/models/logreg_baseline \
#   --numFeatures 20000 --minDF 5

import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Train parquet path")
    ap.add_argument("--test", required=True, help="Test parquet path")
    ap.add_argument("--out", required=True, help="Model output path")
    ap.add_argument("--numFeatures", type=int, default=20000)
    ap.add_argument("--minDF", type=int, default=5)
    ap.add_argument("--sample", type=int, default=0, help="Sample size for testing (0=full)")
    return ap.parse_args()

def main():
    args = parse_args()
    spark = SparkSession.builder\
        .appName("Day3-Baseline-LogReg")\
        .config("spark.driver.memory", "6g")\
        .config("spark.executor.memory", "4g")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("DAY 3 - BASELINE LOGISTIC REGRESSION")
    print(f"{'='*60}\n")

    # Read train and test sets
    print(f"Loading train from: {args.train}")
    train = (spark.read.parquet(args.train)
          .select("review_id","review_text","star_rating","review_length","is_helpful")
          .na.fill({"review_text":"", "star_rating":0.0, "review_length":0})
         )
    
    print(f"Loading test from: {args.test}")
    val = (spark.read.parquet(args.test)
          .select("review_id","review_text","star_rating","review_length","is_helpful")
          .na.fill({"review_text":"", "star_rating":0.0, "review_length":0})
         )
    
    # Sample if requested (for testing)
    if args.sample > 0:
        print(f"\nSampling {args.sample:,} train records...")
        train = train.limit(args.sample)
        val = val.limit(args.sample // 4)
    
    train_count = train.count()
    val_count = val.count()
    print(f"\nTrain records: {train_count:,}")
    print(f"Test records: {val_count:,}")

    # Class weights for imbalance
    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION & WEIGHTING")
    print(f"{'='*60}")
    
    counts = train.groupBy("is_helpful").count().collect()
    cnt = {int(r["is_helpful"]): int(r["count"]) for r in counts}
    pos, neg = cnt.get(1, 1), cnt.get(0, 1)
    pos_weight = float(neg) / float(max(pos,1))
    
    print(f"\nTrain set:")
    print(f"  Positive (helpful=1): {pos:,} ({pos/(pos+neg):.2%})")
    print(f"  Negative (helpful=0): {neg:,} ({neg/(pos+neg):.2%})")
    print(f"  Imbalance ratio: {neg/pos:.2f}:1")
    print(f"  pos_weight: {pos_weight:.3f}")
    
    train = train.withColumn("weight", F.when(F.col("is_helpful")==1, F.lit(pos_weight)).otherwise(F.lit(1.0)))

    # Text → TF-IDF Pipeline
    print(f"\n{'='*60}")
    print("BUILDING TF-IDF PIPELINE")
    print(f"{'='*60}")
    print(f"  numFeatures: {args.numFeatures:,}")
    print(f"  minDocFreq: {args.minDF}")
    
    tok = Tokenizer(inputCol="review_text", outputCol="tokens")
    swr = StopWordsRemover(inputCol="tokens", outputCol="clean")
    htf = HashingTF(inputCol="clean", outputCol="tf", numFeatures=args.numFeatures)
    idf = IDF(inputCol="tf", outputCol="tfidf", minDocFreq=args.minDF)

    # Assemble features
    print("\nFeature columns: tfidf, star_rating, review_length")
    assembler = VectorAssembler(inputCols=["tfidf","star_rating","review_length"], outputCol="features")

    # Logistic Regression
    print(f"\n{'='*60}")
    print("TRAINING LOGISTIC REGRESSION")
    print(f"{'='*60}")
    print("  maxIter: 80")
    print("  regParam: 0.0 (no regularization)")
    print("  elasticNetParam: 0.0")
    print("  weightCol: weight (class balancing)")
    
    lr = LogisticRegression(featuresCol="features", labelCol="is_helpful",
                            weightCol="weight", maxIter=80, regParam=0.0, elasticNetParam=0.0)

    pipe = Pipeline(stages=[tok, swr, htf, idf, assembler, lr])
    
    print("\nFitting pipeline...")
    model = pipe.fit(train)

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    print("\nTransforming test data...")
    pred = model.transform(val).select("is_helpful","rawPrediction","probability","prediction")
    
    # AUC-PR
    evaluator = BinaryClassificationEvaluator(labelCol="is_helpful",
                                              rawPredictionCol="rawPrediction",
                                              metricName="areaUnderPR")
    ap = float(evaluator.evaluate(pred))
    
    # AUC-ROC
    evaluator_roc = BinaryClassificationEvaluator(labelCol="is_helpful",
                                                   rawPredictionCol="rawPrediction",
                                                   metricName="areaUnderROC")
    auc = float(evaluator_roc.evaluate(pred))
    
    # Accuracy
    correct = pred.filter(F.col("prediction") == F.col("is_helpful")).count()
    accuracy = correct / val_count
    
    print(f"\nTest Results:")
    print(f"  AUC-PR: {ap:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # Save model & metrics
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    print(f"Model path: {args.out}")
    
    model.write().overwrite().save(args.out)
    
    import time
    metrics = {
        "train_pos": int(pos),
        "train_neg": int(neg),
        "pos_weight": float(pos_weight),
        "test_auc_pr": float(ap),
        "test_auc_roc": float(auc),
        "test_accuracy": float(accuracy),
        "numFeatures": args.numFeatures,
        "minDF": args.minDF,
        "timestamp": time.time()
    }
    
    (spark.createDataFrame([tuple(metrics.values())], list(metrics.keys()))
          .write.mode("overwrite").json(args.out + "/metrics.json"))
    
    print(f"Metrics saved to: {args.out}/metrics.json")

    print(f"\n{'='*60}")
    print("DAY 3 - BASELINE MODEL COMPLETED")
    print(f"{'='*60}")
    print(f"Train: pos={pos:,} neg={neg:,} pos_weight={pos_weight:.3f}")
    print(f"Test: AUC-PR={ap:.4f} AUC-ROC={auc:.4f} Accuracy={accuracy:.4f}")
    print(f"\nNext steps:")
    print(f"  - Compare with baseline dummy (~0.25)")
    print(f"  - Try different hyperparameters")
    print(f"  - Add more features (sentiment, metadata)")
    
    spark.stop()

if __name__ == "__main__":
    main()
