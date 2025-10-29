# code/models/train_logreg_v2.py
# Day 4 - Logistic Regression với Features V2 (sentiment + metadata)
"""
Usage:
    spark-submit --driver-memory 6g --executor-memory 4g \
        code/models/train_logreg_v2.py \
        --train hdfs://localhost:9000/datasets/amazon/movies/parquet/train \
        --test hdfs://localhost:9000/datasets/amazon/movies/parquet/test \
        --out output/logreg_v2 \
        --numFeatures 20000 --minDF 5
"""

import argparse, json, time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Train parquet path")
    ap.add_argument("--test", required=True, help="Test parquet path")
    ap.add_argument("--out", required=True, help="Model output path")
    ap.add_argument("--numFeatures", type=int, default=20000)
    ap.add_argument("--minDF", type=int, default=5)
    ap.add_argument("--regParam", type=float, default=0.0, help="L2 regularization")
    ap.add_argument("--elasticNetParam", type=float, default=0.0, help="ElasticNet mixing (0=L2, 1=L1)")
    ap.add_argument("--maxIter", type=int, default=100)
    ap.add_argument("--sample", type=int, default=0, help="Sample size for testing (0=full)")
    return ap.parse_args()

def main():
    args = parse_args()
    spark = SparkSession.builder\
        .appName("Day4-LogReg-V2")\
        .config("spark.driver.memory", "6g")\
        .config("spark.executor.memory", "4g")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("DAY 4 - LOGISTIC REGRESSION V2 (FULL FEATURES)")
    print(f"{'='*60}\n")

    # Read train and test - BAY GIO SU DUNG TAT CA FEATURES!
    print(f"Loading train from: {args.train}")
    
    # Columns co trong features_v1
    feature_cols = [
        "review_id", "review_text", "clean_text",
        "star_rating", "review_length", "review_length_log",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "rating_deviation", "is_long_review",
        "product_title", "brand", "category",  # metadata
        "price", "product_avg_rating_meta", "product_total_ratings",  # metadata numeric
        "is_helpful"
    ]
    
    # Chi lay columns ton tai
    train_raw = spark.read.parquet(args.train)
    available_cols = [c for c in feature_cols if c in train_raw.columns]
    
    train = train_raw.select(*available_cols).na.fill({
        "review_text": "",
        "clean_text": "",
        "star_rating": 0.0,
        "review_length": 0,
        "review_length_log": 0.0,
        "sentiment_compound": 0.0,
        "sentiment_pos": 0.0,
        "sentiment_neg": 0.0,
        "sentiment_neu": 0.0,
        "rating_deviation": 0.0,
        "is_long_review": 0,
        "price": 0.0,
        "product_avg_rating_meta": 0.0,
        "product_total_ratings": 0
    })
    
    print(f"Loading test from: {args.test}")
    val_raw = spark.read.parquet(args.test)
    val = val_raw.select(*available_cols).na.fill({
        "review_text": "",
        "clean_text": "",
        "star_rating": 0.0,
        "review_length": 0,
        "review_length_log": 0.0,
        "sentiment_compound": 0.0,
        "sentiment_pos": 0.0,
        "sentiment_neg": 0.0,
        "sentiment_neu": 0.0,
        "rating_deviation": 0.0,
        "is_long_review": 0,
        "price": 0.0,
        "product_avg_rating_meta": 0.0,
        "product_total_ratings": 0
    })
    
    # Sample if requested
    if args.sample > 0:
        print(f"\nSampling {args.sample:,} train records...")
        train = train.limit(args.sample)
        val = val.limit(args.sample // 4)
    
    train_count = train.count()
    val_count = val.count()
    print(f"\nTrain records: {train_count:,}")
    print(f"Test records: {val_count:,}")
    print(f"Available features: {len(available_cols)-2} (excluding review_id, is_helpful)")

    # Class weights
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
    
    train = train.withColumn("weight", 
        F.when(F.col("is_helpful")==1, F.lit(pos_weight)).otherwise(F.lit(1.0)))

    # Build pipeline with ALL FEATURES
    print(f"\n{'='*60}")
    print("BUILDING TF-IDF + METADATA PIPELINE")
    print(f"{'='*60}")
    print(f"  numFeatures: {args.numFeatures:,}")
    print(f"  minDocFreq: {args.minDF}")
    print(f"\nText feature: clean_text (preprocessed)")
    print(f"Metadata features:")
    
    # Danh sach metadata features
    metadata_features = []
    
    # Numeric features
    numeric_features = [
        "star_rating",
        "review_length_log",
        "sentiment_compound",
        "sentiment_pos", 
        "sentiment_neg",
        "sentiment_neu",
        "rating_deviation",
        "is_long_review"
    ]
    
    # Them product metadata neu co
    if "price" in available_cols:
        numeric_features.append("price")
    if "product_avg_rating_meta" in available_cols:
        numeric_features.append("product_avg_rating_meta")
    if "product_total_ratings" in available_cols:
        numeric_features.append("product_total_ratings")
    
    # Loc chi giu features co trong data
    numeric_features = [f for f in numeric_features if f in available_cols]
    
    for feat in numeric_features:
        print(f"  + {feat}")
    
    metadata_features.extend(numeric_features)
    
    # TF-IDF pipeline
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="raw_features", 
                          numFeatures=args.numFeatures)
    idf = IDF(inputCol="raw_features", outputCol="tfidf", minDocFreq=args.minDF)
    
    # Vector assembler - ket hop TF-IDF + metadata
    assembler = VectorAssembler(
        inputCols=["tfidf"] + metadata_features,
        outputCol="features"
    )
    
    # Logistic Regression
    print(f"\n{'='*60}")
    print("TRAINING LOGISTIC REGRESSION V2")
    print(f"{'='*60}")
    print(f"  maxIter: {args.maxIter}")
    print(f"  regParam: {args.regParam}")
    print(f"  elasticNetParam: {args.elasticNetParam}")
    print(f"  weightCol: weight (class balancing)")
    
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="is_helpful",
        weightCol="weight",
        maxIter=args.maxIter,
        regParam=args.regParam,
        elasticNetParam=args.elasticNetParam
    )
    
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler, lr])
    
    print("\nFitting pipeline...")
    start_time = time.time()
    model = pipeline.fit(train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f} seconds")

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*60}\n")
    
    print("Transforming test data...")
    preds = model.transform(val)
    
    # Metrics
    eval_auc = BinaryClassificationEvaluator(
        labelCol="is_helpful",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
    
    eval_roc = BinaryClassificationEvaluator(
        labelCol="is_helpful",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    eval_acc = MulticlassClassificationEvaluator(
        labelCol="is_helpful",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    auc_pr = eval_auc.evaluate(preds)
    auc_roc = eval_roc.evaluate(preds)
    accuracy = eval_acc.evaluate(preds)
    
    print(f"Test Results:")
    print(f"  AUC-PR: {auc_pr:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Save model
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    
    model_path = f"{args.out}/model"
    print(f"Model path: {model_path}")
    model.save(model_path)
    
    # Save metrics
    metrics = {
        "timestamp": time.time(),
        "train_samples": train_count,
        "test_samples": val_count,
        "train_pos": pos,
        "train_neg": neg,
        "pos_weight": pos_weight,
        "test_auc_pr": auc_pr,
        "test_auc_roc": auc_roc,
        "test_accuracy": accuracy,
        "numFeatures": args.numFeatures,
        "minDF": args.minDF,
        "regParam": args.regParam,
        "elasticNetParam": args.elasticNetParam,
        "maxIter": args.maxIter,
        "train_time_seconds": train_time,
        "features_used": metadata_features + ["tfidf"],
        "num_metadata_features": len(metadata_features)
    }
    
    metrics_path = f"{args.out}/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Feature importance (chi cho metadata, ko bao gom TF-IDF)
    lr_model = model.stages[-1]
    coeffs = lr_model.coefficients.toArray()
    
    # TF-IDF chiem numFeatures dau tien
    tfidf_coeffs = coeffs[:args.numFeatures]
    metadata_coeffs = coeffs[args.numFeatures:]
    
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (METADATA ONLY)")
    print(f"{'='*60}")
    
    if len(metadata_coeffs) == len(metadata_features):
        feat_importance = sorted(
            zip(metadata_features, metadata_coeffs),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        print("\nTop 10 most important metadata features:")
        for i, (feat, coef) in enumerate(feat_importance[:10], 1):
            sign = "+" if coef > 0 else "-"
            print(f"  {i:2d}. {feat:30s} {sign} {abs(coef):.4f}")
    
    print(f"\n{'='*60}")
    print("DAY 4 - MODEL V2 COMPLETED")
    print(f"{'='*60}")
    print(f"Train: pos={pos:,} neg={neg:,} pos_weight={pos_weight:.3f}")
    print(f"Test: AUC-PR={auc_pr:.4f} AUC-ROC={auc_roc:.4f} Accuracy={accuracy:.4f}")
    print(f"\nNext steps:")
    print(f"  - Compare with Day 3 baseline (AUC-PR=0.4353)")
    print(f"  - Try hyperparameter tuning")
    print(f"  - Experiment with LightGBM (Day 5)")
    
    spark.stop()

if __name__ == "__main__":
    main()
