# code_v2/features/feature_pipeline_v2.py
# Author: Võ Thị Diễm Thanh (Features & Models)
# Day 3-4: Feature engineering pipeline tổng hợp
#
# Tích hợp tất cả feature modules:
# - metadata_features_v2
# - text_preprocessing_v2
# - sentiment_vader_v2
#
# Usage:
#   spark-submit code_v2/features/feature_pipeline_v2.py \
#       --data hdfs:///datasets/amazon/movies/parquet_v2/reviews \
#       --out hdfs:///datasets/amazon/movies/features_v2 \
#       --feature_set v3

import argparse
from pyspark.sql import SparkSession, functions as F

# Import feature modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from metadata_features_v2 import create_full_feature_set_v2, select_feature_columns_v2
from text_preprocessing_v2 import add_text_features_v2
from sentiment_vader_v2 import add_sentiment_features_v2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input parquet path")
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--feature_set", default="v3", choices=["baseline", "v1", "v2", "v3", "full"])
    ap.add_argument("--include_text", action="store_true", help="Include text features")
    ap.add_argument("--include_sentiment", action="store_true", help="Include sentiment features")
    ap.add_argument("--sample", type=int, default=None, help="Sample size for testing")
    return ap.parse_args()

def main():
    args = parse_args()
    
    spark = SparkSession.builder\
        .appName("FeaturePipeline-V2")\
        .config("spark.driver.memory", "6g")\
        .config("spark.executor.memory", "4g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING PIPELINE V2")
    print("="*80)
    print(f"Author: Võ Thị Diễm Thanh")
    print(f"Feature set: {args.feature_set}")
    print("="*80 + "\n")
    
    # Load data
    print(f"[INFO] Loading data from {args.data}")
    df = spark.read.parquet(args.data)
    
    if args.sample:
        df = df.sample(fraction=min(1.0, args.sample / df.count()), seed=42)
        print(f"[INFO] Sampled to {df.count():,} rows")
    else:
        print(f"[INFO] Loaded {df.count():,} rows")
    
    # 1. Metadata features (always included)
    print("\n[INFO] === Step 1: Metadata Features ===")
    df = create_full_feature_set_v2(
        df,
        include_user_agg=True,
        include_product_agg=True,
        include_temporal=True,
        include_category=True,
        include_interactions=True
    )
    
    # 2. Text features (optional)
    if args.include_text:
        print("\n[INFO] === Step 2: Text Features ===")
        df = add_text_features_v2(df)
        print("  ✓ Text features added")
    
    # 3. Sentiment features (optional)
    if args.include_sentiment:
        print("\n[INFO] === Step 3: Sentiment Features ===")
        df = add_sentiment_features_v2(df, text_column="review_text")
        print("  ✓ Sentiment features added")
    
    # 4. Select final feature set
    print(f"\n[INFO] === Step 4: Select Feature Set ({args.feature_set}) ===")
    selected_features = select_feature_columns_v2(df, args.feature_set)
    print(f"  Selected {len(selected_features)} features")
    
    # Keep essential columns + features + label
    keep_columns = [
        "review_id", "user_id", "product_id", "ts",
        "year", "month", "is_helpful"
    ] + selected_features
    
    # Filter only existing columns
    keep_columns = [c for c in keep_columns if c in df.columns]
    df_final = df.select(*keep_columns)
    
    print(f"  Final columns: {len(keep_columns)}")
    
    # 5. Validate NULL counts
    print("\n[INFO] === Step 5: Validate Data Quality ===")
    null_counts = df_final.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in selected_features[:10]  # Check first 10 features
    ]).first()
    
    has_nulls = False
    for feat in selected_features[:10]:
        count = null_counts[feat]
        if count > 0:
            print(f"  WARNING: {feat} has {count:,} NULLs")
            has_nulls = True
    
    if not has_nulls:
        print("  ✓ No NULLs in features (checked sample)")
    
    # 6. Write output
    print(f"\n[INFO] === Step 6: Writing Output ===")
    print(f"  Output path: {args.out}")
    
    (df_final
     .repartition(8, "year", "month")
     .write.mode("overwrite")
     .partitionBy("year", "month")
     .option("compression", "snappy")
     .parquet(args.out))
    
    print("  ✓ Output written")
    
    # 7. Summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETED")
    print("="*80)
    print(f"\nFeature set: {args.feature_set}")
    print(f"Total features: {len(selected_features)}")
    print(f"Text features: {'✓' if args.include_text else '✗'}")
    print(f"Sentiment features: {'✓' if args.include_sentiment else '✗'}")
    print(f"\nOutput: {args.out}")
    
    # Show sample
    print("\n[INFO] Sample of engineered features:")
    df_final.select(selected_features[:5]).show(5)
    
    spark.stop()

if __name__ == "__main__":
    main()
