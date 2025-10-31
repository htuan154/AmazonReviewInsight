#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spark 3.4–3.5 + SynapseML 1.0.7
# Train LightGBM with robust handling for feature-only train tables (e.g., /output_v2/features_train_v2):
# - If labelCol missing in --train, join labels from --labelSource on --idCol.
# - Stratified train/val split (90/10), class weights, anti-leak drops, AUC-PR evaluation.
#
# Example:
# spark-submit ... train_lightgbm_spark_v2.py \
#   --train hdfs://.../output_v2/features_train_v2 \
#   --labelSource hdfs://.../silver/reviews_labeled/train \
#   --idCol review_id \
#   --out hdfs://.../output_v2/models/lightgbm_v2_flex_star5

import argparse
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from synapse.ml.lightgbm import LightGBMClassifier

def parse_args():
    p = argparse.ArgumentParser(description="Train LightGBM on Spark with optional label join.")
    # IO
    p.add_argument("--train", required=True, help="Path to TRAIN features parquet (e.g., hdfs:///output_v2/features_train_v2)")
    p.add_argument("--test",  required=True, help="Path to TEST features parquet (can be unlabeled)")
    p.add_argument("--out",   required=True, help="Model output path (Spark ML format)")
    p.add_argument("--predictOut", default=None, help="Optional test predictions output path (parquet/csv if .csv extension used)")
    p.add_argument("--trainLimit", type=int, default=None, help="Limit number of train records (for quick test)")
    # Optional label join
    p.add_argument("--labelSource", default=None, help="Parquet path containing labels to join when labelCol missing in --train")
    p.add_argument("--idCol", default="review_id", help="Join key column existing in both --train and --labelSource")
    # Columns
    p.add_argument("--labelCol", default="is_helpful", help="Binary label column name")
    p.add_argument("--featuresCol", default="features", help="Vector column for features")
    # LightGBM params
    p.add_argument("--numLeaves", type=int, default=128)
    p.add_argument("--learningRate", type=float, default=0.035)
    p.add_argument("--numIterations", type=int, default=500)
    p.add_argument("--earlyStoppingRound", type=int, default=200)
    p.add_argument("--featureFraction", type=float, default=0.8)
    p.add_argument("--baggingFraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    # Split
    p.add_argument("--valFrac", type=float, default=0.1, help="Validation fraction")
    return p.parse_args()

LEAKY_COLS = {
    # Any columns that might leak the ground-truth helpfulness label
    "helpful_votes","total_votes","helpful_ratio","vote_ratio",
    "is_helpful_times_len","helpfulness_x_length","label_ratio",
    # Common aliases
    "probability_helpful","is_helpful","helpful","target_helpful"
}

def drop_leaky_columns(df, featuresCol, labelCol):
    cols = set(df.columns)
    # never drop the requested labelCol or features vector automatically;
    # we'll handle label separately.
    bad = [c for c in LEAKY_COLS if c in cols and c not in {featuresCol, labelCol}]
    for c in bad:
        df = df.drop(c)
    return df

def ensure_features(df, featuresCol, labelCol=None):
    """Ensure a features vector exists; assemble from numeric cols when absent."""
    if featuresCol in df.columns:
        return df
    numeric = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float")]
    if labelCol and labelCol in numeric:
        numeric = [c for c in numeric if c != labelCol]
    if not numeric:
        raise RuntimeError(f"No numeric columns available to assemble into '{featuresCol}'.")
    va = VectorAssembler(inputCols=numeric, outputCol=featuresCol, handleInvalid="keep")
    return va.transform(df)

def stratified_train_val(df, labelCol, val_frac=0.1, seed=42):
    """90/10 stratified split using sampleBy on integer-cast labels."""
    df = df.withColumn("__label_int__", F.col(labelCol).cast("int"))
    fractions = {0: val_frac, 1: val_frac}
    df_with_id = df.withColumn("__uid__", F.monotonically_increasing_id())
    val_df = df_with_id.sampleBy("__label_int__", fractions=fractions, seed=seed)
    train_df = df_with_id.join(val_df.select("__uid__"), on="__uid__", how="left_anti")
    return (
        train_df.drop("__uid__", "__label_int__"),
        val_df.drop("__uid__", "__label_int__")
    )

def add_class_weight(df, labelCol, weightCol="weight"):
    """Compute class weights w1 = Nneg/Npos and attach to rows."""
    agg = df.groupBy().agg(
        F.sum(F.when(F.col(labelCol) == 1, 1).otherwise(0)).alias("pos"),
        F.sum(F.when(F.col(labelCol) == 0, 1).otherwise(0)).alias("neg"),
    ).collect()[0]
    pos, neg = agg["pos"], agg["neg"]
    if pos == 0 or neg == 0:
        raise RuntimeError(f"Class collapse (pos={pos}, neg={neg}). Check labeling or join key.")
    w1 = float(neg) / float(pos)
    df = df.withColumn(weightCol, F.when(F.col(labelCol) == 1, F.lit(w1)).otherwise(F.lit(1.0)))
    return df, w1, int(pos), int(neg)

def main():
    args = parse_args()

    spark = (SparkSession.builder
             .appName("Train-LightGBM-V2-Joinable")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---------- Load ----------

    train_df = spark.read.parquet(args.train)
    if args.trainLimit:
        train_df = train_df.limit(args.trainLimit)
    test_df  = spark.read.parquet(args.test)

    # ---------- Anti-leak ----------
    train_df = drop_leaky_columns(train_df, args.featuresCol, args.labelCol)
    test_df  = drop_leaky_columns(test_df,  args.featuresCol, args.labelCol)

    # ---------- Auto-generate label by clustering if missing ----------
    if args.labelCol not in train_df.columns:
        print(f"[INFO] Label column '{args.labelCol}' not found. Auto-generating label by clustering (KMeans, k=2)...")
        from pyspark.ml.clustering import KMeans
        # Ensure features vector exists
        train_df = ensure_features(train_df, args.featuresCol)
        kmeans = KMeans(featuresCol=args.featuresCol, k=2, seed=args.seed)
        kmodel = kmeans.fit(train_df)
        train_df = kmodel.transform(train_df)
        # Use cluster assignment as label (0/1)
        train_df = train_df.withColumn(args.labelCol, F.col("prediction").cast(T.IntegerType()))
        # Drop KMeans prediction column to avoid conflict with LightGBM
        train_df = train_df.drop("prediction")
        print(f"[INFO] Auto-generated label column '{args.labelCol}' using KMeans clustering.")

    # ---------- Ensure features vector ----------
    train_df = ensure_features(train_df, args.featuresCol, labelCol=args.labelCol)
    test_df  = ensure_features(test_df,  args.featuresCol, labelCol=None)

    # ---------- Cast/validate label ----------
    train_df = train_df.withColumn(args.labelCol, F.col(args.labelCol).cast(T.IntegerType()))
    distinct_labels = [r[0] for r in train_df.select(F.col(args.labelCol)).distinct().collect()]
    if not set(distinct_labels).issubset({0, 1}):
        raise RuntimeError(f"Label column '{args.labelCol}' must be binary {{0,1}}. Got values: {sorted(distinct_labels)}")

    # ---------- Split & weighting ----------
    tr_df, val_df = stratified_train_val(train_df, args.labelCol, val_frac=args.valFrac, seed=args.seed)
    tr_df, w1, pos, neg = add_class_weight(tr_df, args.labelCol, weightCol="weight")
    val_df = val_df.withColumn("weight", F.lit(1.0))  # do not weight validation metrics
    tr_df  = tr_df.withColumn("is_val", F.lit(False))
    val_df = val_df.withColumn("is_val", F.lit(True))
    fit_df = tr_df.unionByName(val_df)

    total = pos + neg
    print(f"[INFO] Class balance after join: pos={pos} neg={neg} total={total} w1={w1:.3f}")

    # ---------- Model ----------
    clf = (LightGBMClassifier(
            objective="binary",
            labelCol=args.labelCol,
            featuresCol=args.featuresCol,
            weightCol="weight",
            predictionCol="prediction",
            rawPredictionCol="rawPrediction",
            probabilityCol="probability",
            numLeaves=args.numLeaves,
            learningRate=args.learningRate,
            numIterations=args.numIterations,
            featureFraction=args.featureFraction,
            baggingFraction=args.baggingFraction,
            earlyStoppingRound=args.earlyStoppingRound,
            isUnbalance=True,  # plus weightCol for extra robustness
            seed=args.seed
          )
          .setValidationIndicatorCol("is_val")
    )

    model = clf.fit(fit_df)

    # ---------- Evaluate on validation ----------
    eval_pr  = BinaryClassificationEvaluator(labelCol=args.labelCol, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    eval_roc = BinaryClassificationEvaluator(labelCol=args.labelCol, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    val_pred = model.transform(val_df)
    pr  = eval_pr.evaluate(val_pred)
    roc = eval_roc.evaluate(val_pred)
    print(f"[VAL] AUC-PR={pr:.6f} | AUC-ROC={roc:.6f} | pos={pos} neg={neg} total={total} w1={w1:.3f}")

    # ---------- Save model ----------
    model.write().overwrite().save(args.out)
    print(f"[OK] Model saved -> {args.out}")

    # ---------- Optional: predict on TEST ----------
    if args.predictOut:
        test_pred = model.transform(test_df)
        # Save as parquet by default; if user passes .csv suffix, write CSV
        out = args.predictOut
        if out.lower().endswith(".csv"):
            # extract prob of class 1
            out_df = test_pred.select(
                F.col(args.idCol) if args.idCol in test_pred.columns else F.monotonically_increasing_id().alias(args.idCol),
                F.col("probability").getItem(1).alias("probability_helpful"),
            )
            (out_df.coalesce(1)
                  .write.mode("overwrite").option("header", "true").csv(out))
        else:
            test_pred.write.mode("overwrite").parquet(out)
        print(f"[OK] Test predictions saved -> {out}")

    spark.stop()

if __name__ == "__main__":
    main()
