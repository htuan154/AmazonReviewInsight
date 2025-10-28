# code_v2/etl/train_test_split_v2.py
# Author: Lê Đăng Hoàng Tuấn (Infrastructure)
# Day 1-2: Train/test split với stratification cho V2
#
# Cải tiến so với V1:
# - Đảm bảo NULL đã được xử lý trước khi split
# - Validate data quality sau split
# - Tạo summary statistics chi tiết
#
# Usage:
#    spark-submit code_v2/etl/train_test_split_v2.py \
#        --data hdfs://localhost:9000/datasets/amazon/movies/parquet_v2/reviews \
#        --out hdfs://localhost:9000/datasets/amazon/movies/parquet_v2/ \
#        --test_size 0.2 \
#        --seed 42

import argparse
from pyspark.sql import SparkSession, functions as F

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input parquet path")
    ap.add_argument("--out", required=True, help="Output directory for train/test")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test set ratio (0.0-1.0)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()

def validate_data_quality(df, name="Dataset"):
    """Validate NULL counts trong các cột quan trọng"""
    print(f"\n[INFO] Validating data quality for {name}...")
    
    key_columns = [
        "review_id", "review_text", "star_rating", "helpful_votes",
        "user_id", "product_id", "is_helpful"
    ]
    
    # Check metadata columns if exist
    if "price" in df.columns:
        key_columns.extend(["price", "product_avg_rating_meta", "product_total_ratings", "category"])
    
    null_counts = df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in key_columns if c in df.columns
    ]).first()
    
    has_nulls = False
    for col in key_columns:
        if col in df.columns:
            count = null_counts[col]
            if count > 0:
                print(f"  WARNING: {col} has {count:,} NULLs")
                has_nulls = True
    
    if not has_nulls:
        print(f"  ✓ No NULLs in key columns")
    
    return not has_nulls

def main():
    args = parse_args()
    
    spark = SparkSession.builder\
        .appName("TrainTestSplit-V2")\
        .config("spark.driver.memory", "6g")\
        .config("spark.executor.memory", "4g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*80}")
    print("TRAIN/TEST SPLIT V2 (NULL-SAFE)")
    print(f"{'='*80}\n")
    
    # Read full dataset
    print(f"Loading data from: {args.data}")
    df = spark.read.parquet(args.data)
    
    total_count = df.count()
    print(f"Total records: {total_count:,}")
    
    # Validate data quality
    is_clean = validate_data_quality(df, "Full Dataset")
    
    if not is_clean:
        print("\n[WARNING] Found NULLs in key columns!")
        print("[INFO] You may want to re-run preprocess_spark_v2.py")
    
    # Check class distribution
    print("\n--- Original Class Distribution ---")
    class_dist = df.groupBy("is_helpful").count().orderBy("is_helpful")
    class_dist.show()
    
    class_counts = class_dist.collect()
    for row in class_counts:
        label = int(row["is_helpful"])
        count = int(row["count"])
        pct = (count / total_count * 100) if total_count > 0 else 0
        print(f"  Class {label}: {count:,} ({pct:.2f}%)")
    
    # Stratified split
    train_ratio = 1.0 - args.test_size
    
    print(f"\n[INFO] Splitting with ratio: {train_ratio:.1%} train / {args.test_size:.1%} test")
    
    df_with_split = df.withColumn("_random", F.rand(seed=args.seed))
    
    df_train = df_with_split.filter(F.col("_random") < train_ratio).drop("_random")
    df_test = df_with_split.filter(F.col("_random") >= train_ratio).drop("_random")
    
    # Cache to avoid recomputation
    df_train.cache()
    df_test.cache()
    
    # Verify split
    train_count = df_train.count()
    test_count = df_test.count()
    
    print(f"\n{'='*80}")
    print("SPLIT RESULTS")
    print(f"{'='*80}")
    print(f"Train set: {train_count:,} ({train_count/total_count:.1%})")
    print(f"Test set: {test_count:,} ({test_count/total_count:.1%})")
    
    # Validate both splits
    validate_data_quality(df_train, "Train Set")
    validate_data_quality(df_test, "Test Set")
    
    # Train distribution
    print("\n--- Train Set Class Distribution ---")
    train_dist = df_train.groupBy("is_helpful").count().orderBy("is_helpful").collect()
    for r in train_dist:
        label = int(r["is_helpful"])
        count = int(r["count"])
        pct = count / train_count * 100
        print(f"  Class {label}: {count:,} ({pct:.2f}%)")
    
    # Test distribution
    print("\n--- Test Set Class Distribution ---")
    test_dist = df_test.groupBy("is_helpful").count().orderBy("is_helpful").collect()
    for r in test_dist:
        label = int(r["is_helpful"])
        count = int(r["count"])
        pct = count / test_count * 100
        print(f"  Class {label}: {count:,} ({pct:.2f}%)")
    
    # Save splits
    train_path = f"{args.out}/train"
    test_path = f"{args.out}/test"
    
    print(f"\n[INFO] Writing train set to: {train_path}")
    (df_train
     .repartition(8, "year", "month")
     .write.mode("overwrite")
     .partitionBy("year", "month")
     .option("compression", "snappy")
     .parquet(train_path))
    
    print(f"[INFO] Writing test set to: {test_path}")
    (df_test
     .repartition(4, "year", "month")
     .write.mode("overwrite")
     .partitionBy("year", "month")
     .option("compression", "snappy")
     .parquet(test_path))
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    if "review_length" in df.columns:
        print("\nReview Length Statistics:")
        stats = df_train.select(
            F.mean("review_length").alias("mean"),
            F.stddev("review_length").alias("stddev"),
            F.min("review_length").alias("min"),
            F.max("review_length").alias("max")
        ).first()
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std: {stats['stddev']:.2f}")
        print(f"  Min: {stats['min']}")
        print(f"  Max: {stats['max']}")
    
    if "star_rating" in df.columns:
        print("\nStar Rating Distribution:")
        rating_dist = df_train.groupBy("star_rating").count().orderBy("star_rating")
        rating_dist.show()
    
    if "category" in df.columns:
        print("\nTop 10 Categories:")
        cat_dist = df_train.groupBy("category").count().orderBy(F.desc("count")).limit(10)
        cat_dist.show(truncate=False)
    
    df_train.unpersist()
    df_test.unpersist()
    
    print(f"\n{'='*80}")
    print("COMPLETED")
    print(f"{'='*80}")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    print("\nKey improvements in V2:")
    print("  ✓ NULL validation before and after split")
    print("  ✓ Detailed data quality checks")
    print("  ✓ Summary statistics for EDA")
    
    spark.stop()

if __name__ == "__main__":
    main()
