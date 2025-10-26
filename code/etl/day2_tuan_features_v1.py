# code/etl/day2_tuan_features_v1.py
# Day 2 - Tuan: Tich hop text preprocessing va sentiment vao ETL
"""
Usage:
    spark-submit code/etl/day2_tuan_features_v1.py \
        --data hdfs://localhost:9000/datasets/amazon/movies/parquet/reviews \
        --out hdfs://localhost:9000/datasets/amazon/movies/parquet/features_v1 \
        --sample 100000
"""

import argparse, sys
sys.path.append("code/features")

from pyspark.sql import SparkSession, functions as F
from text_preprocessing import clean_text_udf, preprocess_text_column
from sentiment_vader import add_sentiment_features
from metadata_features import add_basic_metadata_features

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input parquet with reviews")
    ap.add_argument("--out", required=True, help="Output parquet with features v1")
    ap.add_argument("--sample", type=int, default=0, help="Sample size for testing (0=full)")
    return ap.parse_args()

def main():
    args = parse_args()
    
    spark = SparkSession.builder\
        .appName("Day2-Tuan-Features-V1")\
        .config("spark.driver.memory", "6g")\
        .config("spark.executor.memory", "4g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("DAY 2 - TUAN: FEATURES V1")
    print(f"{'='*60}\n")
    
    # Read reviews
    print(f"Loading data from: {args.data}")
    df = spark.read.parquet(args.data)
    
    total_count = df.count()
    print(f"Total records: {total_count:,}")
    
    # Sample for testing
    if args.sample > 0 and args.sample < total_count:
        print(f"Sampling {args.sample:,} records for testing...")
        df = df.sample(fraction=args.sample/total_count, seed=42)
    
    # ===== 1) Text Preprocessing =====
    print(f"\n{'='*60}")
    print("1. TEXT PREPROCESSING")
    print(f"{'='*60}")
    
    print("Cleaning review text...")
    df = preprocess_text_column(df, input_col="review_text", output_col="clean_text")
    
    print("Sample cleaned text:")
    df.select("review_text", "clean_text").show(3, truncate=60)
    
    # ===== 2) VADER Sentiment =====
    print(f"\n{'='*60}")
    print("2. VADER SENTIMENT ANALYSIS")
    print(f"{'='*60}")
    
    print("Computing sentiment scores (this may take a while)...")
    df = add_sentiment_features(df, text_col="review_text", prefix="sentiment")
    
    print("Sentiment statistics:")
    df.select(
        F.mean("sentiment_compound").alias("mean_compound"),
        F.stddev("sentiment_compound").alias("std_compound"),
        F.mean("sentiment_pos").alias("mean_pos"),
        F.mean("sentiment_neg").alias("mean_neg")
    ).show()
    
    # ===== 3) Basic Metadata Features =====
    print(f"\n{'='*60}")
    print("3. BASIC METADATA FEATURES")
    print(f"{'='*60}")
    
    print("Adding: review_length_log, is_long_review, rating_deviation...")
    df = add_basic_metadata_features(df)
    
    print("Feature statistics:")
    df.select(
        "review_length",
        "review_length_log",
        "is_long_review",
        "rating_deviation"
    ).describe().show()
    
    # ===== 4) Save Features V1 =====
    print(f"\n{'='*60}")
    print("4. SAVING FEATURES V1")
    print(f"{'='*60}")
    
    print(f"Output path: {args.out}")
    print(f"Columns: {len(df.columns)}")
    
    # Show new features
    new_features = [
        "clean_text",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "review_length_log", "is_long_review", "rating_deviation"
    ]
    
    print(f"\nNew features added:")
    for feat in new_features:
        if feat in df.columns:
            print(f"  + {feat}")
    
    # Save
    (df
     .repartition(8, "year", "month")
     .write.mode("overwrite")
     .partitionBy("year", "month")
     .option("compression", "snappy")
     .parquet(args.out))
    
    print(f"\n{'='*60}")
    print("DAY 2 - TUAN COMPLETED")
    print(f"{'='*60}")
    print(f"Features V1 saved to: {args.out}")
    print(f"\nNext steps:")
    print("  - Run train/test split")
    print("  - Thanh can now use these features for baseline model")
    
    spark.stop()

if __name__ == "__main__":
    main()
