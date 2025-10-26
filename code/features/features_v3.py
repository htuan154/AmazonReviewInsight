#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering V3 - Day 6
Based on insights from Day 5 Feature Analysis and LightGBM Feature Importance
Creates 6 new features proven to correlate with helpfulness

Author: Lê Đăng Hoàng Tuấn & Võ Thị Diễm Thanh
Date: October 27, 2025
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import time
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def create_features_v3(df):
    """
    Create 6 new features based on Day 5 insights:
    
    1. is_very_long_review: review_length > 200 
       (Insight: Long reviews 3.75x more helpful - 44.5% vs 11.9%)
    
    2. is_critical_review: star_rating <= 2
       (Insight: 1-star reviews 49.9% helpful vs 17.1% for 5-star)
    
    3. is_too_positive: sentiment_pos > 0.4
       (Insight: Negative correlation found - overly positive less helpful)
    
    4. price_tier: Categorical [0=cheap, 1=mid, 2=expensive]
       (Insight: Expensive products get more detailed reviews)
    
    5. sentiment_balance: abs(sentiment_pos - sentiment_neg)
       (Insight: Balanced sentiment reviews are more helpful)
    
    6. length_sentiment_interaction: review_length * sentiment_neg
       (Insight: Both length and negativity positively correlate with helpfulness)
    """
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING V3 - Creating 6 New Features")
    print("="*80)
    
    # Feature 1: is_very_long_review
    # Insight: Long reviews (>200 chars) are 3.75x more helpful
    print("\n[1/6] Creating is_very_long_review (review_length > 200)...")
    df = df.withColumn(
        "is_very_long_review",
        F.when(F.col("review_length") > 200, 1).otherwise(0)
    )
    
    # Feature 2: is_critical_review
    # Insight: 1-2 star reviews are 49.9% helpful vs 17.1% for 5-star
    print("[2/6] Creating is_critical_review (star_rating <= 2)...")
    df = df.withColumn(
        "is_critical_review",
        F.when(F.col("star_rating") <= 2, 1).otherwise(0)
    )
    
    # Feature 3: is_too_positive
    # Insight: Overly positive sentiment shows negative correlation
    print("[3/6] Creating is_too_positive (sentiment_pos > 0.4)...")
    df = df.withColumn(
        "is_too_positive",
        F.when(F.col("sentiment_pos") > 0.4, 1).otherwise(0)
    )
    
    # Feature 4: price_tier
    # Insight: Expensive products get more detailed, helpful reviews
    # Calculate price quantiles for categorization
    print("[4/6] Creating price_tier (0=cheap, 1=mid, 2=expensive)...")
    
    # Get 33rd and 67th percentiles for price
    quantiles = df.approxQuantile("price", [0.33, 0.67], 0.01)
    q33, q67 = quantiles[0], quantiles[1]
    print(f"   Price quantiles: 33rd={q33:.2f}, 67th={q67:.2f}")
    
    df = df.withColumn(
        "price_tier",
        F.when(F.col("price") < q33, 0)  # Cheap
         .when(F.col("price") < q67, 1)  # Mid
         .otherwise(2)                   # Expensive
    )
    
    # Feature 5: sentiment_balance
    # Insight: Balanced reviews (not too positive, not too negative) are helpful
    print("[5/6] Creating sentiment_balance (abs(pos - neg))...")
    df = df.withColumn(
        "sentiment_balance",
        F.abs(F.col("sentiment_pos") - F.col("sentiment_neg"))
    )
    
    # Feature 6: length_sentiment_interaction
    # Insight: Both length and negativity correlate with helpfulness
    # Their interaction may capture "long critical reviews"
    print("[6/6] Creating length_sentiment_interaction (length * neg)...")
    df = df.withColumn(
        "length_sentiment_interaction",
        F.col("review_length") * F.col("sentiment_neg")
    )
    
    print("\n✅ All 6 features created successfully!")
    
    # Show sample of new features
    print("\n📊 Sample of new features:")
    df.select([
        "review_length", "star_rating", "sentiment_pos", "sentiment_neg", "price",
        "is_very_long_review", "is_critical_review", "is_too_positive", 
        "price_tier", "sentiment_balance", "length_sentiment_interaction"
    ]).show(10, truncate=False)
    
    # Statistics for new features
    print("\n📈 New Features Statistics:")
    
    # Binary features
    for col in ["is_very_long_review", "is_critical_review", "is_too_positive"]:
        count_1 = df.filter(F.col(col) == 1).count()
        total = df.count()
        pct = (count_1 / total) * 100
        print(f"   {col}: {count_1:,} ({pct:.1f}%)")
    
    # price_tier distribution
    print("\n   price_tier distribution:")
    df.groupBy("price_tier").count().orderBy("price_tier").show()
    
    # Continuous features
    for col in ["sentiment_balance", "length_sentiment_interaction"]:
        stats = df.select(col).describe().collect()
        print(f"\n   {col}:")
        for row in stats:
            print(f"      {row['summary']}: {float(row[col]):.4f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering V3 - Day 6")
    parser.add_argument("--train", required=True, help="Path to train parquet (HDFS)")
    parser.add_argument("--test", required=True, help="Path to test parquet (HDFS)")
    parser.add_argument("--out_train", required=True, help="Output path for train V3")
    parser.add_argument("--out_test", required=True, help="Output path for test V3")
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("FeatureEngineeringV3") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "6g") \
        .getOrCreate()
    
    try:
        print("\n" + "="*80)
        print("FEATURE ENGINEERING V3 - DAY 6")
        print("="*80)
        print(f"Train input:  {args.train}")
        print(f"Test input:   {args.test}")
        print(f"Train output: {args.out_train}")
        print(f"Test output:  {args.out_test}")
        
        # Process Train Set
        print("\n" + "="*80)
        print("PROCESSING TRAIN SET")
        print("="*80)
        start_time = time.time()
        
        train_df = spark.read.parquet(args.train)
        print(f"\n📊 Train set loaded: {train_df.count():,} records")
        
        train_v3 = create_features_v3(train_df)
        
        # Save Train V3
        print(f"\n💾 Saving train V3 to: {args.out_train}")
        train_v3.write.mode("overwrite").parquet(args.out_train)
        
        train_time = time.time() - start_time
        print(f"✅ Train set completed in {train_time:.1f} seconds")
        
        # Process Test Set
        print("\n" + "="*80)
        print("PROCESSING TEST SET")
        print("="*80)
        start_time = time.time()
        
        test_df = spark.read.parquet(args.test)
        print(f"\n📊 Test set loaded: {test_df.count():,} records")
        
        test_v3 = create_features_v3(test_df)
        
        # Save Test V3
        print(f"\n💾 Saving test V3 to: {args.out_test}")
        test_v3.write.mode("overwrite").parquet(args.out_test)
        
        test_time = time.time() - start_time
        print(f"✅ Test set completed in {test_time:.1f} seconds")
        
        # Summary
        print("\n" + "="*80)
        print("FEATURE ENGINEERING V3 COMPLETED!")
        print("="*80)
        print(f"Total time: {train_time + test_time:.1f} seconds")
        print(f"\nNew features added: 6")
        print("  1. is_very_long_review     (binary)")
        print("  2. is_critical_review      (binary)")
        print("  3. is_too_positive         (binary)")
        print("  4. price_tier              (categorical: 0/1/2)")
        print("  5. sentiment_balance       (continuous)")
        print("  6. length_sentiment_interaction (continuous)")
        print("\n🎯 Ready for model training with V3 features!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
