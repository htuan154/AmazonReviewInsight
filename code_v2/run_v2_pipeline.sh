#!/bin/bash
# run_v2_pipeline.sh
# Complete V2 Pipeline Execution Script
# Team: Lê Đăng Hoàng Tuấn + Võ Thị Diễm Thanh
#
# Usage: bash run_v2_pipeline.sh [mode]
# Modes: all, etl, features, train, predict

set -e  # Exit on error

# ===== Configuration =====
HDFS_BASE="hdfs://localhost:9000"
REVIEWS_PATH="${HDFS_BASE}/datasets/amazon/movies/raw/Movies_and_TV.jsonl"
METADATA_PATH="${HDFS_BASE}/datasets/amazon/movies/raw/meta_Movies_and_TV.jsonl"
OUTPUT_BASE="${HDFS_BASE}/parquet_v2"
LOCAL_OUTPUT="d:/HK7/AmazonReviewInsight/output"

# Spark configs
SPARK_MASTER="yarn"
SPARK_DEPLOY_MODE="client"
DRIVER_MEMORY="6g"
EXECUTOR_MEMORY="4g"

MODE=${1:-all}

echo "============================================"
echo "Amazon Review Insight V2 Pipeline"
echo "Team: Tuấn (Infrastructure) + Thanh (Models)"
echo "Mode: ${MODE}"
echo "============================================"
echo ""

# ===== Step 1: ETL =====
if [ "$MODE" == "all" ] || [ "$MODE" == "etl" ]; then
    echo "[1/5] Running ETL with NULL handling..."
    spark-submit \
        --master ${SPARK_MASTER} \
        --deploy-mode ${SPARK_DEPLOY_MODE} \
        --driver-memory ${DRIVER_MEMORY} \
        --executor-memory ${EXECUTOR_MEMORY} \
        --conf spark.sql.files.maxPartitionBytes=256m \
        code_v2/etl/preprocess_spark_v2.py \
        --reviews ${REVIEWS_PATH} \
        --metadata ${METADATA_PATH} \
        --output ${OUTPUT_BASE}/cleaned
    
    echo "[1/5] ✓ ETL completed"
    echo ""
    
    echo "[2/5] Running train/test split..."
    spark-submit \
        --master ${SPARK_MASTER} \
        --deploy-mode ${SPARK_DEPLOY_MODE} \
        code_v2/etl/train_test_split_v2.py \
        --input ${OUTPUT_BASE}/cleaned \
        --output_train ${OUTPUT_BASE}/train \
        --output_test ${OUTPUT_BASE}/test \
        --test_size 0.2 \
        --seed 42
    
    echo "[2/5] ✓ Train/test split completed"
    echo ""
fi

# ===== Step 2: Feature Engineering =====
if [ "$MODE" == "all" ] || [ "$MODE" == "features" ]; then
    echo "[3/5] Running feature engineering pipeline..."
    spark-submit \
        --master ${SPARK_MASTER} \
        --deploy-mode ${SPARK_DEPLOY_MODE} \
        --driver-memory ${DRIVER_MEMORY} \
        --executor-memory ${EXECUTOR_MEMORY} \
        code_v2/features/feature_pipeline_v2.py \
        --input ${OUTPUT_BASE}/train \
        --output ${OUTPUT_BASE}/features_full \
        --feature_set v3 \
        --include_text \
        --include_sentiment
    
    echo "[3/5] ✓ Feature engineering completed"
    echo ""
fi

# ===== Step 3: Model Training =====
if [ "$MODE" == "all" ] || [ "$MODE" == "train" ]; then
    echo "[4/5] Training LightGBM model..."
    
    # Note: LightGBM training runs locally with pandas-compatible data
    # For full HDFS data, need to convert Spark DataFrame to pandas first
    
    python code_v2/models/train_lightgbm_v2.py \
        --train ${LOCAL_OUTPUT}/train_features_v2.parquet \
        --test ${LOCAL_OUTPUT}/test_features_v2.parquet \
        --output ${LOCAL_OUTPUT}/lightgbm_v2 \
        --feature_set v3
    
    echo "[4/5] ✓ Model training completed"
    echo ""
fi

# ===== Step 4: Prediction =====
if [ "$MODE" == "all" ] || [ "$MODE" == "predict" ]; then
    echo "[5/5] Running prediction pipeline..."
    
    python code_v2/models/predict_pipeline_v2.py \
        --test_features ${LOCAL_OUTPUT}/test_features_v2.parquet \
        --model_path ${LOCAL_OUTPUT}/lightgbm_v2/model.txt \
        --output ${LOCAL_OUTPUT}/submission_v2.csv \
        --batch_size 100000
    
    echo "[5/5] ✓ Prediction completed"
    echo ""
fi

# ===== Summary =====
if [ "$MODE" == "all" ]; then
    echo "============================================"
    echo "Pipeline completed successfully!"
    echo "============================================"
    echo ""
    echo "Outputs:"
    echo "  - Cleaned data: ${OUTPUT_BASE}/cleaned"
    echo "  - Train set: ${OUTPUT_BASE}/train"
    echo "  - Test set: ${OUTPUT_BASE}/test"
    echo "  - Features: ${OUTPUT_BASE}/features_full"
    echo "  - Model: ${LOCAL_OUTPUT}/lightgbm_v2/model.txt"
    echo "  - Metrics: ${LOCAL_OUTPUT}/lightgbm_v2/metrics.json"
    echo "  - Submission: ${LOCAL_OUTPUT}/submission_v2.csv"
    echo ""
    echo "Next steps:"
    echo "  1. Check metrics: cat ${LOCAL_OUTPUT}/lightgbm_v2/metrics.json"
    echo "  2. Validate submission coverage (should be 100%)"
    echo "  3. Compare V1 vs V2 results"
fi
