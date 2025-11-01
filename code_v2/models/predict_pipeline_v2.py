#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict script (v2, refactored, submission-ready)

Features:
- Loads metadata.json from model directory for numFeatures, id_col, features_col validation
- Strict feature dimension checking with --force override
- Safe probability extraction from Vector/rawPrediction with fallback to sigmoid
- Generates submission.csv with exactly 2 columns: review_id, probability_helpful
- Comprehensive logging: schema, params, sample output, error logs
- Validates output: correct columns, header, probability range [0,1]

Usage:
  spark-submit predict_pipeline_v2.py \
    --model_path hdfs://.../lightgbm_v2 \
    --test hdfs://.../features_test_v2 \
    --out hdfs://.../predictions \
    [--force] [--debug_samples 100]
"""

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark import StorageLevel

try:
    from pyspark.ml.functions import vector_to_array
except Exception:
    vector_to_array = None

try:
    from synapse.ml.lightgbm import LightGBMClassificationModel
except Exception:
    LightGBMClassificationModel = None


def parse_args():
    p = argparse.ArgumentParser(description="Predict with validation and comprehensive logging")
    # Required
    p.add_argument("--model_path", required=True, help="Path to saved model directory")
    p.add_argument("--test", required=True, help="Path to test features parquet")
    p.add_argument("--out", required=True, help="Output directory for predictions and logs")
    
    # Optional
    p.add_argument("--id_col", default=None, help="ID column name (read from metadata if None)")
    p.add_argument("--features_col", default=None, help="Features column name (read from metadata if None)")
    p.add_argument("--label_col", default="is_helpful", help="Label column for metrics (if present)")
    
    # Validation
    p.add_argument("--force", action="store_true", 
                   help="Force prediction even if numFeatures mismatch")
    
    # Logging
    p.add_argument("--debug_samples", type=int, default=100,
                   help="Number of samples to save in debug CSV (0=disable)")
    p.add_argument("--save_full_parquet", action="store_true",
                   help="Save full prediction results as parquet")
    
    return p.parse_args()


def build_spark(app_name: str = "AmazonReviewInsight-PredictV2") -> SparkSession:
    """Initialize Spark session with optimized configs."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.shuffle.partitions", "200")
        # Reduce memory pressure from Parquet vectorized reader
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
        # Smaller file partition bytes to avoid huge tasks when scanning
        .config("spark.sql.files.maxPartitionBytes", "64m")
        # Give the driver a bit more room on local runs
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """Load metadata.json from model directory."""
    try:
        # Try local file system first
        metadata_path = os.path.join(model_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # Try HDFS path
        metadata_hdfs = model_path.rstrip("/") + "/metadata.json"
        # This requires Spark to be initialized, handle in main flow
        return None
    except Exception as e:
        print(f"[WARN] Could not load metadata.json: {e}")
        return None


def load_metadata_hdfs(spark: SparkSession, model_path: str) -> Optional[Dict[str, Any]]:
    """Load metadata.json from HDFS path."""
    try:
        metadata_path = model_path.rstrip("/") + "/metadata.json"
        if not fs_exists(spark, metadata_path):
            return None
        
        # Read as text file
        rdd = spark.sparkContext.textFile(metadata_path)
        content = "\n".join(rdd.collect())
        return json.loads(content)
    except Exception as e:
        print(f"[WARN] Could not load metadata from HDFS: {e}")
        return None


def get_vector_size(df: DataFrame, features_col: str) -> int:
    """Extract the dimension of the feature vector."""
    sample = df.select(features_col).first()
    if sample is None:
        raise RuntimeError(f"Cannot determine vector size: no data in '{features_col}'")
    vec = sample[0]
    if isinstance(vec, (SparseVector, DenseVector)):
        return vec.size
    elif hasattr(vec, 'size'):
        return vec.size
    else:
        raise RuntimeError(f"Unknown vector type: {type(vec)}")


def validate_feature_dimension(df: DataFrame, features_col: str, 
                               expected_dim: Optional[int], force: bool = False) -> int:
    """Validate feature dimension matches expected from metadata."""
    actual_dim = get_vector_size(df, features_col)
    
    if expected_dim is not None and actual_dim != expected_dim:
        msg = (f"\n{'='*80}\n"
               f"FEATURE DIMENSION MISMATCH\n"
               f"{'='*80}\n"
               f"Expected: {expected_dim} features (from model metadata)\n"
               f"Got:      {actual_dim} features (in test data)\n\n"
               f"This error usually means the test features were built with different\n"
               f"parameters than training features (e.g., different HashingTF numFeatures,\n"
               f"different TF-IDF vocabulary, or different feature engineering).\n\n"
               f"SOLUTION:\n"
               f"  Rebuild test features using the EXACT SAME parameters as training:\n"
               f"  - Same numFeatures for HashingTF/CountVectorizer\n"
               f"  - Same text preprocessing pipeline\n"
               f"  - Same feature engineering steps\n\n"
               f"To override this check (NOT RECOMMENDED), use --force flag.\n"
               f"{'='*80}\n")
        
        if force:
            print(f"[WARN] {msg}")
            print("[WARN] --force enabled, continuing anyway (predictions may be incorrect)")
        else:
            raise RuntimeError(msg)
    
    return actual_dim


def ensure_id_string(df: DataFrame, id_col: str) -> DataFrame:
    """Ensure ID column exists and is string type."""
    if id_col not in df.columns:
        # Try common alternatives
        for candidate in ["review_id", "id", "_id"]:
            if candidate in df.columns:
                df = df.withColumnRenamed(candidate, id_col)
                break
        else:
            # Generate sequential IDs
            df = df.withColumn(id_col, F.monotonically_increasing_id().cast(StringType()))
    
    # Ensure string type
    if df.schema[id_col].dataType != StringType():
        df = df.withColumn(id_col, F.col(id_col).cast(StringType()))
    
    return df


# -----------------------------
# Path & logging helpers
# -----------------------------
def is_remote_path(path: str) -> bool:
    return isinstance(path, str) and (path.startswith("hdfs://") or path.startswith("s3://") or path.startswith("abfs://"))


def join_path(base: str, *parts: str) -> str:
    """Join path parts, using forward slashes for remote URIs (HDFS/S3/ABFS)."""
    if is_remote_path(base):
        b = base.rstrip("/")
        rest = "/".join(p.strip("/") for p in parts)
        return f"{b}/{rest}" if rest else b
    return os.path.join(base, *parts)


def get_local_logs_dir(out_dir: str, kind: str = "predict") -> str:
    """Return a safe local directory to store logs when out_dir is remote (HDFS)."""
    if not is_remote_path(out_dir):
        return join_path(out_dir, "logs")
    # Map remote out dir to local tmp logs
    safe_dir = os.path.join(os.getcwd(), "tmp", f"{kind}_logs")
    os.makedirs(safe_dir, exist_ok=True)
    return safe_dir


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
    """Sigmoid function for converting raw prediction to probability."""
    try:
        return 1.0 / (1.0 + math.exp(-float(z)))
    except (OverflowError, ValueError):
        return 0.0 if z < 0 else 1.0


def extract_probability_helpful(df: DataFrame,
                                probability_col: str = "probability",
                                raw_col: str = "rawPrediction") -> DataFrame:
    """
    Extract probability_helpful from model output.
    Tries multiple strategies:
    1. vector_to_array (if available) from probability column
    2. Direct struct access probability.values[1]
    3. UDF extraction from probability vector
    4. Fallback to sigmoid(rawPrediction)
    """
    
    # Strategy 1: vector_to_array (Spark 3.0+)
    if probability_col in df.columns and vector_to_array is not None:
        try:
            arr = vector_to_array(F.col(probability_col))
            df = df.withColumn("probability_helpful", 
                             F.when(F.size(arr) >= 2, arr[1])
                              .otherwise(arr[0]))
            
            # Validate we got valid probabilities
            sample = df.select("probability_helpful").first()
            if sample and sample[0] is not None:
                print("[INFO] Extracted probability using vector_to_array")
                return df.select(df.columns)  # Ensure column exists
        except Exception as e:
            print(f"[DEBUG] vector_to_array failed: {e}")
    
    # Strategy 2: Direct getItem access (for DenseVector/SparseVector)
    if probability_col in df.columns:
        try:
            df = df.withColumn("probability_helpful", 
                             F.col(probability_col).getItem(1))
            
            # Check if it worked
            sample = df.where(F.col("probability_helpful").isNotNull()).first()
            if sample:
                print("[INFO] Extracted probability using getItem(1)")
                return df
        except Exception as e:
            print(f"[DEBUG] getItem(1) failed: {e}")
    
    # Strategy 3: UDF extraction from probability vector
    if probability_col in df.columns:
        def extract_prob1_udf(v):
            if v is None:
                return None
            try:
                # Handle different vector types
                if isinstance(v, (list, tuple)):
                    return float(v[1]) if len(v) > 1 else float(v[0])
                elif hasattr(v, 'toArray'):
                    arr = v.toArray()
                    return float(arr[1]) if len(arr) > 1 else float(arr[0])
                elif hasattr(v, 'values'):
                    vals = list(v.values)
                    return float(vals[1]) if len(vals) > 1 else float(vals[0])
                else:
                    # Try to access as array-like
                    return float(v[1])
            except Exception:
                return None
        
        udf_extract = F.udf(extract_prob1_udf, DoubleType())
        df = df.withColumn("probability_helpful", udf_extract(F.col(probability_col)))
        
        # Check if we got values
        sample = df.where(F.col("probability_helpful").isNotNull()).first()
        if sample:
            print("[INFO] Extracted probability using UDF from probability vector")
            return df
    
    # Strategy 4: Fallback to sigmoid(rawPrediction)
    if raw_col in df.columns:
        print("[WARN] Falling back to sigmoid(rawPrediction)")
        
        def sigmoid_from_raw(v):
            if v is None:
                return None
            try:
                # Single value
                if isinstance(v, (int, float)):
                    return _sigmoid(float(v))
                
                # Vector/array
                if hasattr(v, 'toArray'):
                    arr = v.toArray()
                elif hasattr(v, 'values'):
                    arr = list(v.values)
                else:
                    arr = list(v)
                
                if len(arr) == 1:
                    # Binary classification with single margin
                    return _sigmoid(float(arr[0]))
                elif len(arr) >= 2:
                    # Two-class raw margins, convert to probability
                    m0, m1 = float(arr[0]), float(arr[1])
                    exp_sum = math.exp(m0) + math.exp(m1)
                    return math.exp(m1) / exp_sum if exp_sum > 0 else 0.5
            except Exception:
                return None
            return None
        
        udf_sigmoid = F.udf(sigmoid_from_raw, DoubleType())
        df = df.withColumn("probability_helpful", udf_sigmoid(F.col(raw_col)))
        return df
    
    # If we got here, we couldn't extract probability
    raise RuntimeError(
        f"Could not extract probability from model output. "
        f"Available columns: {df.columns}. "
        f"Neither '{probability_col}' nor '{raw_col}' could be processed."
    )


def validate_submission(df: DataFrame, count: int) -> None:
    """Validate submission DataFrame meets requirements."""
    # Check columns
    if set(df.columns) != {"review_id", "probability_helpful"}:
        raise RuntimeError(
            f"Submission must have exactly 2 columns: review_id, probability_helpful. "
            f"Got: {df.columns}"
        )
    
    # Check types
    if df.schema["review_id"].dataType != StringType():
        raise RuntimeError("review_id must be StringType")
    
    if df.schema["probability_helpful"].dataType != DoubleType():
        raise RuntimeError("probability_helpful must be DoubleType")
    
    # Check probability range
    stats = df.select(
        F.min("probability_helpful").alias("min_prob"),
        F.max("probability_helpful").alias("max_prob"),
        F.count(F.when(F.col("probability_helpful").isNull(), 1)).alias("null_count")
    ).collect()[0]
    
    if stats["null_count"] > 0:
        raise RuntimeError(f"Found {stats['null_count']} NULL probabilities")
    
    min_prob, max_prob = stats["min_prob"], stats["max_prob"]
    if min_prob < 0.0 or max_prob > 1.0:
        raise RuntimeError(
            f"Probabilities must be in [0, 1]. Got range: [{min_prob:.6f}, {max_prob:.6f}]"
        )
    
    print(f"[OK] Submission validation passed:")
    print(f"     - Rows: {count:,}")
    print(f"     - Columns: review_id (string), probability_helpful (double)")
    print(f"     - Probability range: [{min_prob:.6f}, {max_prob:.6f}]")
    print(f"     - No NULL values")


def save_logs(out_dir: str, test_schema, params: Dict, stats: Dict) -> None:
    """Save prediction logs for debugging."""
    try:
        # If out_dir is remote (e.g., HDFS), write logs locally under tmp/predict_logs
        logs_dir = get_local_logs_dir(out_dir, kind="predict")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save test schema
        with open(os.path.join(logs_dir, "schema_test.txt"), "w", encoding="utf-8") as f:
            f.write("TEST DATA SCHEMA\n")
            f.write("=" * 80 + "\n")
            f.write(str(test_schema))
            f.write("\n")
        
        # Save parameters
        with open(os.path.join(logs_dir, "params.txt"), "w", encoding="utf-8") as f:
            f.write("PREDICTION PARAMETERS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            for k, v in sorted(params.items()):
                f.write(f"{k} = {v}\n")
        
        # Save statistics
        with open(os.path.join(logs_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        print(f"[OK] Logs saved to {logs_dir}")
    except Exception as e:
        print(f"[WARN] Failed to save logs: {e}")


def save_error_log(out_dir: str, error_info: Dict, args=None) -> None:
    """Save detailed error log."""
    try:
        # If out_dir is remote, save error logs locally
        if is_remote_path(out_dir):
            local_dir = os.path.join(os.getcwd(), "tmp", "predict_error_logs")
            os.makedirs(local_dir, exist_ok=True)
            target_dir = local_dir
        else:
            target_dir = out_dir
        os.makedirs(target_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_path = os.path.join(target_dir, f"predict_error_log_{timestamp}.txt")
        
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"PREDICTION ERROR LOG - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            if args:
                f.write("COMMAND LINE ARGUMENTS:\n")
                f.write("-" * 80 + "\n")
                for key, value in vars(args).items():
                    f.write(f"{key} = {value}\n")
                f.write("\n")
            
            f.write("ERROR DETAILS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Error Type: {error_info['type']}\n")
            f.write(f"Error Message: {error_info['message']}\n\n")
            
            f.write("FULL TRACEBACK:\n")
            f.write("-" * 80 + "\n")
            f.write(error_info['traceback'])
            f.write("\n")
            
            f.write("\nSYSTEM INFO:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Working Directory: {os.getcwd()}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF ERROR LOG\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n[ERROR] Detailed error log saved to: {error_log_path}")
        return error_log_path
    except Exception as log_error:
        print(f"[FATAL] Could not save error log: {log_error}")
        return None


def format_error_message(exc_type, exc_value, exc_tb) -> Dict:
    """Format exception into structured dictionary."""
    return {
        'type': exc_type.__name__,
        'message': str(exc_value),
        'traceback': ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    }

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
                features_col: str = "features",
                label_col: str = "is_helpful",
                force: bool = False,
                debug_samples: int = 100,
                save_full_parquet: bool = False) -> None:
    """
    Main prediction pipeline with comprehensive validation and logging.
    """
    
    print(f"\n{'='*80}")
    print(f"PREDICTION PIPELINE - v2 Refactored")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Test:  {test_path}")
    print(f"Output: {out_dir}")
    print(f"{'='*80}\n")
    
    # ========== Load Metadata ==========
    metadata = load_metadata(model_path)
    if metadata is None:
        metadata = load_metadata_hdfs(spark, model_path)
    
    if metadata:
        print(f"[INFO] Loaded metadata from model directory")
        # Override defaults with metadata
        if not id_col or id_col == "review_id":
            id_col = metadata.get("id_col", "review_id")
        if not features_col or features_col == "features":
            features_col = metadata.get("features_col", "features")
        expected_num_features = metadata.get("numFeatures")
        print(f"[INFO] Metadata: id_col={id_col}, features_col={features_col}, "
              f"numFeatures={expected_num_features}")
    else:
        print(f"[WARN] No metadata.json found, using defaults")
        expected_num_features = None
    
    # ========== Load Model ==========
    print(f"[LOAD] Loading model from {model_path}...")
    model, model_kind = load_model(spark, model_path)
    print(f"[OK] Model loaded (type: {model_kind})")
    
    # ========== Load Test Data ==========
    print(f"[LOAD] Reading test data from {test_path}...")
    test_df = spark.read.parquet(test_path)
    test_count = test_df.count()
    print(f"[OK] Loaded {test_count:,} test samples")
    
    # ========== Validate Schema ==========
    if features_col not in test_df.columns:
        raise RuntimeError(
            f"Features column '{features_col}' not found in test data. "
            f"Available columns: {test_df.columns}"
        )
    
    # Ensure ID column
    test_df = ensure_id_string(test_df, id_col)
    
    # ========== Validate Feature Dimensions ==========
    actual_dim = validate_feature_dimension(
        test_df, features_col, expected_num_features, force=force
    )
    print(f"[INFO] Feature dimension: {actual_dim}")
    
    # ========== Run Prediction ==========
    print(f"[PREDICT] Running model.transform()...")
    # Save review_id before prediction (model.transform may drop it)
    test_df_with_id = test_df.withColumn("__row_id__", F.monotonically_increasing_id())
    pred_df = model.transform(test_df_with_id)
    print(f"[OK] Prediction complete")
    
    # Log prediction schema for debugging
    pred_schema_str = pred_df._jdf.schema().treeString()
    
    # ========== Extract Probability ==========
    print(f"[EXTRACT] Extracting probability_helpful...")
    pred_df = extract_probability_helpful(
        pred_df, 
        probability_col="probability", 
        raw_col="rawPrediction"
    )
    print(f"[OK] Extracted probability_helpful")
    
    # ========== Restore Review ID ==========
    # If model dropped review_id, join it back from original test_df
    if id_col not in pred_df.columns:
        print(f"[WARN] Column '{id_col}' not found in predictions, joining back from test data...")
        id_mapping = test_df_with_id.select(F.col("__row_id__"), F.col(id_col))
        pred_df = pred_df.join(id_mapping, on="__row_id__", how="left")
        print(f"[OK] Restored '{id_col}' column")
    
    # ========== Create Submission DataFrame ==========
    submission_df = pred_df.select(
        F.col(id_col).alias("review_id").cast(StringType()),
        F.col("probability_helpful").cast(DoubleType())
    )
    # Cache the narrow submission frame so later actions (write, stats) don't re-read the full test parquet
    submission_df = submission_df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # ========== Validate Submission ==========
    print(f"[VALIDATE] Checking submission format...")
    validate_submission(submission_df, test_count)
    
    # ========== Save Submission CSV ==========
    submission_csv = join_path(out_dir, "submission.csv")
    submission_tmp = join_path(out_dir, "submission_tmp")
    
    print(f"[SAVE] Writing submission.csv...")
    (submission_df.coalesce(1)
                  .write.mode("overwrite")
                  .option("header", "true")
                  .csv(submission_tmp))
    
    # Rename part file to submission.csv
    listed = fs_ls(spark, submission_tmp)
    part_src = None
    for st in listed:
        name = st.getPath().getName()
        if name.startswith("part-") and name.endswith(".csv"):
            part_src = st.getPath().toString()
            break
    
    if part_src is None:
        raise RuntimeError("Could not find part-*.csv in submission_tmp")
    
    if not fs_rename(spark, part_src, submission_csv):
        raise RuntimeError(f"Failed to rename {part_src} to {submission_csv}")
    
    print(f"[OK] Submission saved to: {submission_csv}")
    
    # ========== Save Debug Samples ==========
    if debug_samples > 0:
        debug_csv = join_path(out_dir, "debug_sample.csv")
        debug_tmp = join_path(out_dir, "debug_tmp")
        
        print(f"[SAVE] Saving {debug_samples} debug samples...")
        (submission_df.limit(debug_samples).coalesce(1)
                      .write.mode("overwrite")
                      .option("header", "true")
                      .csv(debug_tmp))
        
        # Rename debug file
        listed_debug = fs_ls(spark, debug_tmp)
        for st in listed_debug:
            name = st.getPath().getName()
            if name.startswith("part-") and name.endswith(".csv"):
                fs_rename(spark, st.getPath().toString(), debug_csv)
                break
        
        print(f"[OK] Debug sample saved to: {debug_csv}")
    
    # ========== Save Full Parquet (Optional) ==========
    if save_full_parquet:
        parquet_path = join_path(out_dir, "predictions_full.parquet")
        print(f"[SAVE] Saving full predictions to parquet...")
        pred_df.write.mode("overwrite").parquet(parquet_path)
        print(f"[OK] Full predictions saved to: {parquet_path}")
    
    # ========== Compute Statistics ==========
    prob_stats = submission_df.select(
        F.min("probability_helpful").alias("min"),
        F.max("probability_helpful").alias("max"),
        F.avg("probability_helpful").alias("mean"),
        F.stddev("probability_helpful").alias("stddev")
    ).collect()[0]
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "test_count": test_count,
        "feature_dimension": actual_dim,
        "probability_stats": {
            "min": float(prob_stats["min"]),
            "max": float(prob_stats["max"]),
            "mean": float(prob_stats["mean"]),
            "stddev": float(prob_stats["stddev"]) if prob_stats["stddev"] else 0.0
        }
    }
    
    # ========== Save Logs ==========
    params = {
        "model_path": model_path,
        "test_path": test_path,
        "out_dir": out_dir,
        "id_col": id_col,
        "features_col": features_col,
        "force": force,
        "model_kind": model_kind
    }
    
    save_logs(out_dir, pred_schema_str, params, stats)
    
    # ========== Final Summary ==========
    print(f"\n{'='*80}")
    print(f"PREDICTION COMPLETE")
    print(f"{'='*80}")
    print(f"Test samples:      {test_count:,}")
    print(f"Feature dimension: {actual_dim}")
    print(f"Probability range: [{prob_stats['min']:.6f}, {prob_stats['max']:.6f}]")
    print(f"Mean probability:  {prob_stats['mean']:.6f}")
    print(f"Output files:")
    print(f"  - {submission_csv}")
    if debug_samples > 0:
        print(f"  - {join_path(out_dir, 'debug_sample.csv')} ({debug_samples} samples)")
    if save_full_parquet:
        print(f"  - {join_path(out_dir, 'predictions_full.parquet')}")
    # If out_dir is remote, logs are saved locally
    logs_dir_print = get_local_logs_dir(out_dir, kind="predict") if is_remote_path(out_dir) else join_path(out_dir, 'logs/')
    print(f"  - {logs_dir_print} (schema, params, stats)")
    print(f"{'='*80}\n")
    # Clean up cache
    try:
        submission_df.unpersist()
    except Exception:
        pass


def main():
    args = parse_args()
    spark = None
    
    try:
        spark = build_spark()
        
        run_predict(
            spark=spark,
            model_path=args.model_path,
            test_path=args.test,
            out_dir=args.out,
            id_col=args.id_col if args.id_col else "review_id",
            features_col=args.features_col if args.features_col else "features",
            label_col=args.label_col,
            force=args.force,
            debug_samples=args.debug_samples,
            save_full_parquet=args.save_full_parquet
        )
        
    except Exception as e:
        # Capture exception information
        exc_type, exc_value, exc_tb = sys.exc_info()
        error_info = format_error_message(exc_type, exc_value, exc_tb)
        
        # Print error to console
        print(f"\n{'='*80}")
        print(f"PREDICTION FAILED")
        print(f"{'='*80}")
        print(f"Error Type: {error_info['type']}")
        print(f"Error Message: {error_info['message']}")
        print(f"{'='*80}\n")
        
        # Save detailed error log
        out_dir = args.out if hasattr(args, 'out') and args.out else "."
        save_error_log(out_dir, error_info, args if 'args' in locals() else None)
        
        # Re-raise
        raise
        
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception as stop_error:
                print(f"[WARN] Error stopping Spark: {stop_error}")


if __name__ == "__main__":
    main()

