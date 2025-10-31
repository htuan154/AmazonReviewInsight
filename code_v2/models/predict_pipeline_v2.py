#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict script (v2, final, submission-ready)

- Loads either a full Spark ML PipelineModel OR a bare SynapseML LightGBMClassificationModel
- Robustly extracts P(y=1) from probability/rawPrediction for VectorUDT & LightGBM margins
- Writes:
    out_dir/predictions_parquet/...
    out_dir/predictions_csv/...
    out_dir/submission.csv          <-- single CSV file with review_id,probability_helpful
    out_dir/metrics.json            <-- contains Brier score if label present
- Logs schema to code_v2/tmp/predict_schema.txt and error stack to code_v2/tmp/predict_error_log.txt
"""

import argparse
import json
import math
import os
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml import PipelineModel

try:
    from pyspark.ml.functions import vector_to_array
except Exception:
    vector_to_array = None

try:
    from synapse.ml.lightgbm import LightGBMClassificationModel
except Exception:
    LightGBMClassificationModel = None


def build_spark(app_name: str = "AmazonReviewInsight-PredictV2") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    return spark

def fs_exists(spark: SparkSession, path_str: str) -> bool:
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    Path = jvm.org.apache.hadoop.fs.Path
    p = Path(path_str)
    fs = p.getFileSystem(conf)
    return fs.exists(p)

def fs_ls(spark: SparkSession, path_str: str):
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    Path = jvm.org.apache.hadoop.fs.Path
    p = Path(path_str)
    fs = p.getFileSystem(conf)
    if not fs.exists(p):
        return []
    return list(fs.listStatus(p))

def fs_rename(spark: SparkSession, src: str, dst: str) -> bool:
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    Path = jvm.org.apache.hadoop.fs.Path
    fs_src = Path(src).getFileSystem(conf)
    return bool(fs_src.rename(Path(src), Path(dst)))


def detect_model_kind(spark: SparkSession, model_path: str) -> str:
    has_stages = fs_exists(spark, model_path.rstrip("/") + "/stages")
    has_booster = fs_exists(spark, model_path.rstrip("/") + "/complexParams/lightGBMBooster")
    if has_stages:
        return "pipeline"
    if has_booster:
        return "lgbm"
    return "unknown"

def load_model(spark: SparkSession, model_path: str):
    kind = detect_model_kind(spark, model_path)
    if kind == "pipeline":
        return PipelineModel.load(model_path), kind
    if kind == "lgbm":
        if LightGBMClassificationModel is None:
            raise RuntimeError("SynapseML is required to load LightGBM models. Add --packages com.microsoft.azure:synapseml_2.12:1.0.7")
        return LightGBMClassificationModel.load(model_path), kind
    raise RuntimeError("Could not load model: neither PipelineModel nor LightGBMClassificationModel.")

def read_features(spark: SparkSession, path: Optional[str]) -> Optional[DataFrame]:
    if not path:
        return None
    return spark.read.parquet(path)

def ensure_id_column(df: DataFrame, id_col: str) -> DataFrame:
    if id_col in df.columns:
        return df
    for cand in ["review_id", "id", "_id"]:
        if cand in df.columns:
            return df.withColumnRenamed(cand, id_col)
    return df.withColumn(id_col, F.monotonically_increasing_id())


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(z)))

def extract_prob1(df: DataFrame,
                  probability_col: str = "probability",
                  raw_col: str = "rawPrediction") -> DataFrame:
    # 1) vector_to_array when available
    if probability_col in df.columns and vector_to_array is not None:
        try:
            arr = vector_to_array(F.col(probability_col))
            df = df.withColumn("prob1_tmp", F.when(F.size(arr) >= 2, arr[1]).otherwise(arr[0]))
        except Exception:
            pass

    # 2) Access SparseVector 'values' field
    if "prob1_tmp" not in df.columns or df.where(F.col("prob1_tmp").isNull()).head(1):
        def get_prob_from_vec_struct(v):
            if v is None:
                return None
            try:
                arr = list(v)
                if len(arr) >= 2:
                    return float(arr[1])
                if len(arr) == 1:
                    return float(arr[0])
            except Exception:
                return None
            return None
        udf_v = F.udf(get_prob_from_vec_struct, DoubleType())
        if probability_col in df.columns:
            df = df.withColumn(
                "prob1_tmp",
                F.when(
                    F.col("prob1_tmp").isNull() if "prob1_tmp" in df.columns else F.lit(True),
                    udf_v(F.col(probability_col).getField("values"))
                ).otherwise(F.col("prob1_tmp")) if "prob1_tmp" in df.columns else udf_v(F.col(probability_col).getField("values"))
            )

    # 3) Fallback from raw margin(s)
    if "prob1_tmp" not in df.columns or df.where(F.col("prob1_tmp").isNull()).head(1):
        def prob_from_raw(v):
            if v is None:
                return None
            try:
                if isinstance(v, (int, float)):
                    return _sigmoid(float(v))
                arr = list(v)
                if len(arr) == 1:
                    return _sigmoid(float(arr[0]))
                if len(arr) >= 2:
                    m0, m1 = float(arr[0]), float(arr[1])
                    e0, e1 = math.exp(m0), math.exp(m1)
                    s = e0 + e1
                    return e1 / s if s != 0 else None
            except Exception:
                return None
            return None
        udf_raw = F.udf(prob_from_raw, DoubleType())
        if raw_col in df.columns:
            df = df.withColumn(
                "prob1_tmp",
                F.when(
                    F.col("prob1_tmp").isNull() if "prob1_tmp" in df.columns else F.lit(True),
                    udf_raw(F.col(raw_col))
                ).otherwise(F.col("prob1_tmp")) if "prob1_tmp" in df.columns else udf_raw(F.col(raw_col))
            )

    return df.withColumn("prob1", F.col("prob1_tmp")).drop("prob1_tmp")

def compute_brier_score(df: DataFrame, label_col: str = "label", prob_col: str = "prob1"):
    if label_col not in df.columns or prob_col not in df.columns:
        return None
    tmp = df.where((F.col(prob_col).isNotNull()) & (F.col(label_col).isNotNull()))
    if tmp.rdd.isEmpty():
        return None
    brier = tmp.select(F.pow(F.col(prob_col) - F.col(label_col).cast(DoubleType()), 2).alias("sq"))
    val = brier.agg(F.avg("sq").alias("brier")).collect()[0][0]
    return float(val) if val is not None else None

def run_predict(spark: SparkSession,
                model_path: str,
                test_path: str,
                out_dir: str,
                id_col: str = "review_id",
                label_col: str = "label",
                probability_col: str = "probability",
                raw_col: str = "rawPrediction") -> None:

    model, kind = load_model(spark, model_path)

    test_df = read_features(spark, test_path)
    if test_df is None:
        raise RuntimeError("--test_features is required")

    test_df = ensure_id_column(test_df, id_col)

    if kind == "lgbm" and "features" not in test_df.columns:
        raise RuntimeError("Bare LightGBM model requires a 'features' vector column in test dataframe.")

    pred_df = model.transform(test_df)

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    schema_path = os.path.join(tmp_dir, "predict_schema.txt")
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(pred_df._jdf.schema().treeString())
    print(f"[DEBUG] Wrote schema to: {schema_path}")

    pred_df = extract_prob1(pred_df, probability_col=probability_col, raw_col=raw_col)
    brier = compute_brier_score(pred_df, label_col=label_col, prob_col="prob1")

    base = out_dir.rstrip("/")
    out_parquet = base + "/predictions_parquet"
    out_csv_dir = base + "/predictions_csv"
    submission_tmp = base + "/submission_tmp"
    submission_csv = base + "/submission.csv"
    metrics_path = base + "/metrics.json"

    out_cols = [id_col, "prob1", "prediction"]
    if label_col in pred_df.columns:
        out_cols.append(label_col)
    out_pred = pred_df.select(*[c for c in out_cols if c in pred_df.columns])

    (out_pred.coalesce(1)
            .write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(out_parquet))

    (out_pred.coalesce(1)
            .write.mode("overwrite")
            .option("header", True)
            .csv(out_csv_dir))

    submission_df = out_pred.select(
        F.col(id_col).alias("review_id").cast(StringType()),
        F.col("prob1").alias("probability_helpful").cast(DoubleType())
    )
    (submission_df.coalesce(1)
                  .write.mode("overwrite")
                  .option("header", True)
                  .csv(submission_tmp))

    listed = fs_ls(spark, submission_tmp)
    part_src = None
    for st in listed:
        p = st.getPath().toString()
        name = st.getPath().getName()
        if name.startswith("part-") and name.endswith(".csv"):
            part_src = p
            break
    if part_src is None:
        raise RuntimeError("Could not locate part-*.csv in submission_tmp")

    if not fs_rename(spark, part_src, submission_csv):
        raise RuntimeError(f"Failed to rename {part_src} to {submission_csv}")

    metrics = {"brier_score": brier}
    spark.createDataFrame([(json.dumps(metrics, ensure_ascii=False),)], ["value"]) \
         .coalesce(1).write.mode("overwrite").text(metrics_path)

    print("\n==================== PREDICTION SUMMARY ====================")
    print(f"Model path:          {model_path}   (kind={kind})")
    print(f"Test path:           {test_path}")
    print(f"Output (parquet):    {out_parquet}")
    print(f"Output (csv dir):    {out_csv_dir}")
    print(f"Submission (file):   {submission_csv}")
    print(f"metrics.json:        {metrics_path}")
    if brier is not None:
        print(f"Brier Score:         {brier:.6f} (lower is better)")
    else:
        print("Brier Score:         (skipped – label column not found)")
    print("===========================================================\n")


def parse_args():
    p = argparse.ArgumentParser(description="Predict with Spark PipelineModel or SynapseML LightGBM model (submission-ready)")
    p.add_argument("--test_features", required=True, help="Path to test features parquet (HDFS/S3/file)")
    p.add_argument("--model_path", required=True, help="Path to saved model (pipeline or lightgbm)")
    p.add_argument("--out_dir", required=True, help="Output directory (HDFS/S3/file)")

    p.add_argument("--train_features", default=None)
    p.add_argument("--include_text", action="store_true")
    p.add_argument("--numFeatures", type=int, default=None)

    p.add_argument("--id_col", default="review_id", help="ID column in outputs (default: review_id)")
    p.add_argument("--label_col", default="label", help="Label column name (for Brier Score if present)")
    p.add_argument("--probabilityCol", default="probability", help="Model probability column (default: probability)")
    p.add_argument("--rawPredictionCol", default="rawPrediction", help="Model raw prediction column (default: rawPrediction)")
    return p.parse_args()

def main():
    args = parse_args()
    spark = build_spark()
    try:
        run_predict(
            spark=spark,
            model_path=args.model_path,
            test_path=args.test_features,
            out_dir=args.out_dir,
            id_col=args.id_col,
            label_col=args.label_col,
            probability_col=args.probabilityCol,
            raw_col=args.rawPredictionCol,
        )
    except Exception as e:
        import traceback
        tmp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        log_path = os.path.join(tmp_dir, "predict_error_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"\n[ERROR] Wrote error log to: {log_path}\n")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
