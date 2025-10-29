"""
Generate submission.csv from tuned LightGBM model

Usage:
    spark-submit --driver-memory 8g --executor-memory 6g \
        --packages com.microsoft.azure:synapseml_2.12:0.11.4 \
        generate_submission.py \
        --test hdfs://localhost:9000/datasets/amazon/movies/parquet/test \
        --model output/lightgbm_tuned/model \
        --pipeline output/lightgbm_tuned/feature_pipeline \
        --out output/submission.csv
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from synapse.ml.lightgbm import LightGBMClassificationModel
from pyspark.sql.functions import col
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Generate submission.csv')
    parser.add_argument('--test', type=str, required=True,
                        help='Path to test parquet')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained LightGBM model')
    parser.add_argument('--pipeline', type=str, required=True,
                        help='Path to feature pipeline')
    parser.add_argument('--out', type=str, required=True,
                        help='Output submission.csv path')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize Spark
    print("[INFO] Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("AmazonReview-Submission") \
        .getOrCreate()
    
    print("[OK] Spark initialized! Version: {}".format(spark.version))
    
    # Load test data
    print("\n[INFO] Loading test data from: {}".format(args.test))
    test_df = spark.read.parquet(args.test)
    num_test = test_df.count()
    print("   [OK] Loaded {} test records".format(num_test))
    
    # Load feature pipeline
    print("\n[INFO] Loading feature pipeline from: {}".format(args.pipeline))
    pipeline = PipelineModel.load(args.pipeline)
    print("   [OK] Pipeline loaded!")
    
    # Transform test data
    print("[INFO] Transforming test data...")
    start_time = time.time()
    test_transformed = pipeline.transform(test_df)
    elapsed = time.time() - start_time
    print("   [OK] Transformed in {:.1f}s".format(elapsed))
    
    # Load model
    print("\n[INFO] Loading LightGBM model from: {}".format(args.model))
    model = LightGBMClassificationModel.load(args.model)
    print("   [OK] Model loaded!")
    
    # Make predictions
    print("[INFO] Generating predictions...")
    start_time = time.time()
    predictions = model.transform(test_transformed)
    elapsed = time.time() - start_time
    print("   [OK] Predictions generated in {:.1f}s".format(elapsed))
    
    # Create submission dataframe
    print("\n[INFO] Creating submission file...")
    submission = predictions.select(
        col("review_id"),
        col("prediction").cast("int").alias("predicted_helpful")
    )
    
    # Validate
    num_predictions = submission.count()
    print("   [DATA] Total predictions: {}".format(num_predictions))
    
    if num_predictions != num_test:
        print("   [WARN] Prediction count mismatch! Expected: {}, Got: {}".format(
            num_test, num_predictions))
    else:
        print("   [OK] Prediction count matches test records!")
    
    # Check for nulls
    null_count = submission.filter(
        col("review_id").isNull() | col("predicted_helpful").isNull()
    ).count()
    
    if null_count > 0:
        print("   [WARN] Found {} null values!".format(null_count))
    else:
        print("   [OK] No null values detected!")
    
    # Check prediction distribution
    helpful_count = submission.filter(col("predicted_helpful") == 1).count()
    not_helpful_count = submission.filter(col("predicted_helpful") == 0).count()
    helpful_pct = (helpful_count / num_predictions) * 100
    
    print("\n[DATA] Prediction Distribution:")
    print("   - Helpful (1): {} ({:.2f}%)".format(helpful_count, helpful_pct))
    print("   - Not Helpful (0): {} ({:.2f}%)".format(
        not_helpful_count, 100 - helpful_pct))
    
    # Save to CSV
    print("\n[INFO] Saving submission to: {}".format(args.out))
    
    # Convert to Pandas and save (for single CSV file)
    submission_pd = submission.toPandas()
    submission_pd.to_csv(args.out, index=False)
    
    print("   [OK] Submission file saved!")
    print("\n[DATA] File Format:")
    print("   - Columns: review_id, predicted_helpful")
    print("   - Rows: {}".format(len(submission_pd)))
    print("   - Type: CSV with header")
    
    # Show sample
    print("\n[DATA] Sample (first 10 rows):")
    print(submission_pd.head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("*** SUBMISSION FILE GENERATED SUCCESSFULLY! ***")
    print("=" * 80)
    print("\n[FILE] Location: {}".format(args.out))
    print("[FILE] Size: {} rows".format(len(submission_pd)))
    print("[FILE] Ready for submission!")
    
    spark.stop()


if __name__ == "__main__":
    main()
