# code/features/day1_thanh_eda.py
# Script demo Day 1 của Thanh: EDA & định nghĩa target
# Chạy local để test các functions

"""
Usage:
    spark-submit code/features/day1_thanh_eda.py \
        --data hdfs://localhost:9000/datasets/amazon/movies/parquet/reviews \
        --sample 100000 \
        --out output/day1_eda_thanh.txt
"""

import argparse
from pyspark.sql import SparkSession, functions as F

# Import các functions từ modules của Thanh
import sys
sys.path.append("code/features")
from text_preprocessing import define_target, analyze_class_imbalance, preprocess_text_column
from sentiment_vader import add_sentiment_features, analyze_sentiment_by_helpfulness

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hdfs://localhost:9000/datasets/amazon/movies/parquet/reviews")
    ap.add_argument("--sample", type=int, default=100000, help="Sample size for EDA")
    ap.add_argument("--out", default="output/day1_eda_thanh.txt")
    return ap.parse_args()

def main():
    args = parse_args()
    
    spark = SparkSession.builder\
        .appName("Day1-Thanh-EDA")\
        .config("spark.driver.memory", "4g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("DAY 1 - THANH: EDA & Target Definition")
    print(f"{'='*60}\n")
    
    # Đọc dữ liệu từ Parquet (đã có từ Day 1 của Tuấn)
    print(f"Loading data from: {args.data}")
    df = spark.read.parquet(args.data)
    
    print(f"Total records: {df.count():,}")
    print(f"\nSchema:")
    df.printSchema()
    
    # Sample để EDA nhanh hơn
    if args.sample and args.sample < df.count():
        print(f"\nSampling {args.sample:,} records for EDA...")
        df_sample = df.sample(fraction=args.sample/df.count(), seed=42)
    else:
        df_sample = df
    
    # ===== 1) Kiểm tra target definition =====
    print(f"\n{'='*60}")
    print("1. TARGET DEFINITION (is_helpful)")
    print(f"{'='*60}")
    
    # is_helpful đã có từ ETL của Tuấn (helpful_votes > 0)
    # Nhưng Thanh kiểm tra lại với các threshold khác nhau
    
    print("\nCurrent target (helpful_votes > 0):")
    df_sample.groupBy("is_helpful").count().show()
    
    print("\nTesting alternative thresholds:")
    for threshold in [0, 1, 2, 5]:
        df_test = df_sample.withColumn(
            f"is_helpful_t{threshold}",
            F.when(F.col("helpful_votes") > threshold, 1).otherwise(0)
        )
        count = df_test.filter(f"is_helpful_t{threshold} = 1").count()
        total = df_test.count()
        print(f"  Threshold > {threshold}: {count:,} helpful ({count/total:.2%})")
    
    # ===== 2) Phân tích class imbalance =====
    print(f"\n{'='*60}")
    print("2. CLASS IMBALANCE ANALYSIS")
    print(f"{'='*60}")
    
    stats = analyze_class_imbalance(df_sample)
    
    # ===== 3) Phân tích text =====
    print(f"\n{'='*60}")
    print("3. TEXT PREPROCESSING TEST")
    print(f"{'='*60}")
    
    # Test preprocessing trên 10 samples
    sample_for_text = df_sample.limit(10)
    df_clean = preprocess_text_column(sample_for_text)
    
    print("\nOriginal vs Cleaned text (first 3 samples):")
    df_clean.select("review_text", "clean_text").show(3, truncate=80)
    
    # ===== 4) Review length distribution =====
    print(f"\n{'='*60}")
    print("4. REVIEW LENGTH ANALYSIS")
    print(f"{'='*60}")
    
    length_stats = df_sample.select(
        F.mean("review_length").alias("mean"),
        F.stddev("review_length").alias("stddev"),
        F.min("review_length").alias("min"),
        F.expr("percentile(review_length, 0.25)").alias("q25"),
        F.expr("percentile(review_length, 0.5)").alias("median"),
        F.expr("percentile(review_length, 0.75)").alias("q75"),
        F.max("review_length").alias("max")
    ).collect()[0]
    
    print(f"Mean: {length_stats['mean']:.1f} words")
    print(f"Std Dev: {length_stats['stddev']:.1f}")
    print(f"Min: {length_stats['min']}")
    print(f"Q25: {length_stats['q25']:.0f}")
    print(f"Median: {length_stats['median']:.0f}")
    print(f"Q75: {length_stats['q75']:.0f}")
    print(f"Max: {length_stats['max']}")
    
    # Phân tích theo helpfulness
    print("\nReview length by helpfulness:")
    df_sample.groupBy("is_helpful")\
        .agg(F.avg("review_length").alias("avg_length"),
             F.stddev("review_length").alias("std_length"))\
        .show()
    
    # ===== 5) Star rating distribution =====
    print(f"\n{'='*60}")
    print("5. STAR RATING ANALYSIS")
    print(f"{'='*60}")
    
    print("\nRating distribution:")
    df_sample.groupBy("star_rating")\
        .count()\
        .orderBy("star_rating")\
        .show()
    
    print("\nRating by helpfulness:")
    df_sample.groupBy("is_helpful")\
        .agg(F.avg("star_rating").alias("avg_rating"),
             F.stddev("star_rating").alias("std_rating"))\
        .show()
    
    # ===== 6) VADER Sentiment (optional - nếu đã cài) =====
    print(f"\n{'='*60}")
    print("6. SENTIMENT ANALYSIS (VADER) - Optional")
    print(f"{'='*60}")
    
    try:
        print("\nTesting VADER sentiment on sample...")
        df_sentiment = add_sentiment_features(df_sample.limit(1000), text_col="review_text")
        
        print("\nSentiment stats:")
        df_sentiment.select(
            F.mean("sentiment_compound").alias("mean_compound"),
            F.stddev("sentiment_compound").alias("std_compound"),
            F.mean("sentiment_pos").alias("mean_pos"),
            F.mean("sentiment_neg").alias("mean_neg")
        ).show()
        
        print("\nSentiment by helpfulness:")
        analyze_sentiment_by_helpfulness(df_sentiment)
        
    except Exception as e:
        print(f"[INFO] VADER not available or error: {e}")
        print("Install with: pip install vaderSentiment")
    
    # ===== Summary =====
    print(f"\n{'='*60}")
    print("DAY 1 SUMMARY - THANH")
    print(f"{'='*60}")
    print("\n[COMPLETED]")
    print("  1. Verified target definition (is_helpful)")
    print("  2. Analyzed class imbalance")
    print(f"     - Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
    print(f"     - Recommended pos_weight: {stats['recommended_pos_weight']:.3f}")
    print("  3. Tested text preprocessing functions")
    print("  4. Analyzed review length distribution")
    print("  5. Analyzed star rating distribution")
    print("  6. (Optional) Sentiment analysis with VADER")
    
    print("\n[NEXT STEPS - Day 2]")
    print("  - Implement full text preprocessing pipeline")
    print("  - Integrate VADER sentiment")
    print("  - Prepare for baseline model training")
    
    spark.stop()

if __name__ == "__main__":
    main()
