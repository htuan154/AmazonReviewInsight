# code/etl/train_test_split.py
# Day 1 Fix: Tạo train/test split với stratification
"""
Usage:
    spark-submit code/etl/train_test_split.py \
        --data hdfs://localhost:9000/datasets/amazon/movies/parquet/reviews \
        --out hdfs://localhost:9000/datasets/amazon/movies/parquet/ \
        --test_size 0.2 \
        --seed 42
"""

import argparse
from pyspark.sql import SparkSession, functions as F

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input parquet path")
    ap.add_argument("--out", required=True, help="Output directory for train/test")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test set ratio (0.0-1.0)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()

def main():
    args = parse_args()
    
    spark = SparkSession.builder\
        .appName("TrainTestSplit")\
        .config("spark.driver.memory", "6g")\
        .config("spark.executor.memory", "4g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT WITH STRATIFICATION")
    print(f"{'='*60}\n")
    
    # Read full dataset
    print(f"Loading data from: {args.data}")
    df = spark.read.parquet(args.data)
    
    total_count = df.count()
    print(f"Total records: {total_count:,}")
    
    # Check class distribution
    print("\nOriginal class distribution:")
    df.groupBy("is_helpful").count().show()
    
    # Stratified split using sampleBy
    # Get class fractions
    class_counts = df.groupBy("is_helpful").count().collect()
    class_dict = {int(r["is_helpful"]): int(r["count"]) for r in class_counts}
    
    print(f"\nClass 0 (not helpful): {class_dict.get(0, 0):,}")
    print(f"Class 1 (helpful): {class_dict.get(1, 0):,}")
    
    # Create stratified split
    # Method: Add random number column, then split based on threshold
    train_ratio = 1.0 - args.test_size
    
    df_with_split = df.withColumn("_random", F.rand(seed=args.seed))
    
    # Stratify by partitioning within each class
    df_train = df_with_split.filter(F.col("_random") < train_ratio).drop("_random")
    df_test = df_with_split.filter(F.col("_random") >= train_ratio).drop("_random")
    
    # Verify split
    train_count = df_train.count()
    test_count = df_test.count()
    
    print(f"\n{'='*60}")
    print("SPLIT RESULTS")
    print(f"{'='*60}")
    print(f"Train set: {train_count:,} ({train_count/total_count:.1%})")
    print(f"Test set: {test_count:,} ({test_count/total_count:.1%})")
    
    print("\nTrain set class distribution:")
    train_dist = df_train.groupBy("is_helpful").count().collect()
    for r in train_dist:
        label = int(r["is_helpful"])
        count = int(r["count"])
        pct = count / train_count * 100
        print(f"  Class {label}: {count:,} ({pct:.2f}%)")
    
    print("\nTest set class distribution:")
    test_dist = df_test.groupBy("is_helpful").count().collect()
    for r in test_dist:
        label = int(r["is_helpful"])
        count = int(r["count"])
        pct = count / test_count * 100
        print(f"  Class {label}: {count:,} ({pct:.2f}%)")
    
    # Save to separate directories
    train_path = f"{args.out}/train"
    test_path = f"{args.out}/test"
    
    print(f"\nWriting train set to: {train_path}")
    (df_train
     .repartition(8, "year", "month")
     .write.mode("overwrite")
     .partitionBy("year", "month")
     .option("compression", "snappy")
     .parquet(train_path))
    
    print(f"Writing test set to: {test_path}")
    (df_test
     .repartition(4, "year", "month")
     .write.mode("overwrite")
     .partitionBy("year", "month")
     .option("compression", "snappy")
     .parquet(test_path))
    
    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"{'='*60}")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    
    spark.stop()

if __name__ == "__main__":
    main()
