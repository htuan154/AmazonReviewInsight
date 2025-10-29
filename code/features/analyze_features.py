# code/features/analyze_features.py
# Day 5 - Deep Dive Feature Analysis
"""
Usage:
    spark-submit --driver-memory 6g code/features/analyze_features.py \
        --data hdfs://localhost:9000/datasets/amazon/movies/parquet/features_v1 \
        --out output/feature_analysis.json
"""

import argparse, json
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def analyze_correlation(df, features):
    """Phân tích correlation giữa features và target"""
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}\n")
    
    correlations = {}
    for feat in features:
        if feat in df.columns:
            corr = df.stat.corr(feat, "is_helpful")
            correlations[feat] = corr
            print(f"{feat:30s} correlation: {corr:+.4f}")
    
    return correlations

def analyze_distribution(df, features):
    """Phân tích distribution của features theo class"""
    print(f"\n{'='*60}")
    print("DISTRIBUTION ANALYSIS (Helpful vs Not Helpful)")
    print(f"{'='*60}\n")
    
    distributions = {}
    
    for feat in features:
        if feat not in df.columns:
            continue
            
        print(f"\n{feat}:")
        
        stats = df.groupBy("is_helpful").agg(
            F.mean(feat).alias("mean"),
            F.stddev(feat).alias("std"),
            F.min(feat).alias("min"),
            F.max(feat).alias("max"),
            F.expr(f"percentile_approx({feat}, 0.25)").alias("q25"),
            F.expr(f"percentile_approx({feat}, 0.5)").alias("median"),
            F.expr(f"percentile_approx({feat}, 0.75)").alias("q75")
        ).collect()
        
        distributions[feat] = {}
        for row in stats:
            label = "helpful" if row["is_helpful"] == 1 else "not_helpful"
            distributions[feat][label] = {
                "mean": float(row["mean"]) if row["mean"] else 0,
                "std": float(row["std"]) if row["std"] else 0,
                "min": float(row["min"]) if row["min"] else 0,
                "max": float(row["max"]) if row["max"] else 0,
                "q25": float(row["q25"]) if row["q25"] else 0,
                "median": float(row["median"]) if row["median"] else 0,
                "q75": float(row["q75"]) if row["q75"] else 0
            }
            
            print(f"  {label:12s}: mean={row['mean']:.4f} std={row['std']:.4f} "
                  f"median={row['median']:.4f}")
        
        # Calculate mean ratio
        if distributions[feat]["helpful"]["mean"] > 0:
            ratio = (distributions[feat]["not_helpful"]["mean"] / 
                    distributions[feat]["helpful"]["mean"])
            print(f"  Ratio (not_helpful/helpful): {ratio:.2f}x")
    
    return distributions

def analyze_categorical(df):
    """Phân tích categorical features"""
    print(f"\n{'='*60}")
    print("CATEGORICAL FEATURE ANALYSIS")
    print(f"{'='*60}\n")
    
    categorical_analysis = {}
    
    # Star rating distribution
    if "star_rating" in df.columns:
        print("\nStar Rating Distribution:")
        star_dist = df.groupBy("star_rating", "is_helpful").count().orderBy("star_rating", "is_helpful")
        
        # Calculate helpful ratio per star rating
        star_stats = df.groupBy("star_rating").agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("is_helpful") == 1, 1).otherwise(0)).alias("helpful_count")
        ).withColumn("helpful_ratio", F.col("helpful_count") / F.col("total")).orderBy("star_rating")
        
        categorical_analysis["star_rating"] = []
        for row in star_stats.collect():
            data = {
                "value": float(row["star_rating"]),
                "total": int(row["total"]),
                "helpful_count": int(row["helpful_count"]),
                "helpful_ratio": float(row["helpful_ratio"])
            }
            categorical_analysis["star_rating"].append(data)
            print(f"  {row['star_rating']:.0f} stars: {row['total']:,} reviews, "
                  f"{row['helpful_ratio']:.1%} helpful")
    
    # Is long review
    if "is_long_review" in df.columns:
        print("\nLong Review vs Short Review:")
        long_stats = df.groupBy("is_long_review").agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("is_helpful") == 1, 1).otherwise(0)).alias("helpful_count")
        ).withColumn("helpful_ratio", F.col("helpful_count") / F.col("total")).orderBy("is_long_review")
        
        categorical_analysis["is_long_review"] = []
        for row in long_stats.collect():
            label = "Long (>100 words)" if row["is_long_review"] == 1 else "Short (<=100 words)"
            data = {
                "value": int(row["is_long_review"]),
                "label": label,
                "total": int(row["total"]),
                "helpful_count": int(row["helpful_count"]),
                "helpful_ratio": float(row["helpful_ratio"])
            }
            categorical_analysis["is_long_review"].append(data)
            print(f"  {label}: {row['total']:,} reviews, "
                  f"{row['helpful_ratio']:.1%} helpful")
    
    return categorical_analysis

def analyze_sentiment_patterns(df):
    """Phân tích patterns của sentiment"""
    print(f"\n{'='*60}")
    print("SENTIMENT PATTERN ANALYSIS")
    print(f"{'='*60}\n")
    
    sentiment_patterns = {}
    
    # Sentiment compound buckets
    if "sentiment_compound" in df.columns:
        print("\nSentiment Compound Buckets:")
        df_with_buckets = df.withColumn(
            "sentiment_bucket",
            F.when(F.col("sentiment_compound") < -0.5, "Very Negative")
            .when(F.col("sentiment_compound") < 0, "Negative")
            .when(F.col("sentiment_compound") == 0, "Neutral")
            .when(F.col("sentiment_compound") < 0.5, "Positive")
            .otherwise("Very Positive")
        )
        
        bucket_stats = df_with_buckets.groupBy("sentiment_bucket").agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("is_helpful") == 1, 1).otherwise(0)).alias("helpful_count")
        ).withColumn("helpful_ratio", F.col("helpful_count") / F.col("total"))
        
        sentiment_patterns["compound_buckets"] = []
        for row in bucket_stats.collect():
            data = {
                "bucket": row["sentiment_bucket"],
                "total": int(row["total"]),
                "helpful_count": int(row["helpful_count"]),
                "helpful_ratio": float(row["helpful_ratio"])
            }
            sentiment_patterns["compound_buckets"].append(data)
            print(f"  {row['sentiment_bucket']:15s}: {row['total']:,} reviews, "
                  f"{row['helpful_ratio']:.1%} helpful")
    
    # Sentiment dominance (which component is highest)
    if all(col in df.columns for col in ["sentiment_pos", "sentiment_neg", "sentiment_neu"]):
        print("\nSentiment Dominance:")
        df_with_dom = df.withColumn(
            "dominant_sentiment",
            F.when(
                (F.col("sentiment_pos") > F.col("sentiment_neg")) & 
                (F.col("sentiment_pos") > F.col("sentiment_neu")), "Positive"
            ).when(
                (F.col("sentiment_neg") > F.col("sentiment_pos")) & 
                (F.col("sentiment_neg") > F.col("sentiment_neu")), "Negative"
            ).otherwise("Neutral")
        )
        
        dom_stats = df_with_dom.groupBy("dominant_sentiment").agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("is_helpful") == 1, 1).otherwise(0)).alias("helpful_count")
        ).withColumn("helpful_ratio", F.col("helpful_count") / F.col("total"))
        
        sentiment_patterns["dominance"] = []
        for row in dom_stats.collect():
            data = {
                "sentiment": row["dominant_sentiment"],
                "total": int(row["total"]),
                "helpful_count": int(row["helpful_count"]),
                "helpful_ratio": float(row["helpful_ratio"])
            }
            sentiment_patterns["dominance"].append(data)
            print(f"  {row['dominant_sentiment']:10s}: {row['total']:,} reviews, "
                  f"{row['helpful_ratio']:.1%} helpful")
    
    return sentiment_patterns

def analyze_product_metadata(df):
    """Phân tích product metadata patterns"""
    print(f"\n{'='*60}")
    print("PRODUCT METADATA PATTERNS")
    print(f"{'='*60}\n")
    
    product_patterns = {}
    
    # Product rating buckets
    if "product_avg_rating_meta" in df.columns:
        print("\nProduct Avg Rating Buckets:")
        df_with_buckets = df.withColumn(
            "product_rating_bucket",
            F.when(F.col("product_avg_rating_meta") < 2, "1-2 stars (Poor)")
            .when(F.col("product_avg_rating_meta") < 3, "2-3 stars (Fair)")
            .when(F.col("product_avg_rating_meta") < 4, "3-4 stars (Good)")
            .when(F.col("product_avg_rating_meta") < 4.5, "4-4.5 stars (Very Good)")
            .otherwise("4.5-5 stars (Excellent)")
        )
        
        rating_stats = df_with_buckets.groupBy("product_rating_bucket").agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("is_helpful") == 1, 1).otherwise(0)).alias("helpful_count")
        ).withColumn("helpful_ratio", F.col("helpful_count") / F.col("total"))
        
        product_patterns["rating_buckets"] = []
        for row in rating_stats.collect():
            data = {
                "bucket": row["product_rating_bucket"],
                "total": int(row["total"]),
                "helpful_count": int(row["helpful_count"]),
                "helpful_ratio": float(row["helpful_ratio"])
            }
            product_patterns["rating_buckets"].append(data)
            print(f"  {row['product_rating_bucket']:25s}: {row['total']:,} reviews, "
                  f"{row['helpful_ratio']:.1%} helpful")
    
    # Price buckets
    if "price" in df.columns:
        print("\nPrice Buckets:")
        df_with_price = df.filter(F.col("price") > 0).withColumn(
            "price_bucket",
            F.when(F.col("price") < 10, "$0-10")
            .when(F.col("price") < 20, "$10-20")
            .when(F.col("price") < 50, "$20-50")
            .when(F.col("price") < 100, "$50-100")
            .otherwise("$100+")
        )
        
        price_stats = df_with_price.groupBy("price_bucket").agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("is_helpful") == 1, 1).otherwise(0)).alias("helpful_count")
        ).withColumn("helpful_ratio", F.col("helpful_count") / F.col("total"))
        
        product_patterns["price_buckets"] = []
        for row in price_stats.collect():
            data = {
                "bucket": row["price_bucket"],
                "total": int(row["total"]),
                "helpful_count": int(row["helpful_count"]),
                "helpful_ratio": float(row["helpful_ratio"])
            }
            product_patterns["price_buckets"].append(data)
            print(f"  {row['price_bucket']:10s}: {row['total']:,} reviews, "
                  f"{row['helpful_ratio']:.1%} helpful")
    
    return product_patterns

def main():
    args = parse_args()
    spark = SparkSession.builder\
        .appName("Day5-Feature-Analysis")\
        .config("spark.driver.memory", "6g")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("DAY 5 - DEEP DIVE FEATURE ANALYSIS")
    print(f"{'='*60}\n")
    
    print(f"Loading data from: {args.data}")
    df = spark.read.parquet(args.data)
    
    total_count = df.count()
    helpful_count = df.filter(F.col("is_helpful") == 1).count()
    print(f"Total records: {total_count:,}")
    print(f"Helpful: {helpful_count:,} ({helpful_count/total_count:.1%})")
    print(f"Not helpful: {total_count - helpful_count:,} ({(total_count-helpful_count)/total_count:.1%})")
    
    # Define numeric features
    numeric_features = [
        "star_rating", "review_length", "review_length_log",
        "sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "rating_deviation", "price", 
        "product_avg_rating_meta", "product_total_ratings"
    ]
    
    # Run analyses
    correlations = analyze_correlation(df, numeric_features)
    distributions = analyze_distribution(df, numeric_features)
    categorical_analysis = analyze_categorical(df)
    sentiment_patterns = analyze_sentiment_patterns(df)
    product_patterns = analyze_product_metadata(df)
    
    # Compile results
    results = {
        "timestamp": str(df.select(F.current_timestamp()).first()[0]),
        "dataset_stats": {
            "total_records": int(total_count),
            "helpful_count": int(helpful_count),
            "helpful_ratio": float(helpful_count / total_count)
        },
        "correlations": correlations,
        "distributions": distributions,
        "categorical_analysis": categorical_analysis,
        "sentiment_patterns": sentiment_patterns,
        "product_patterns": product_patterns
    }
    
    # Save
    print(f"\n{'='*60}")
    print("SAVING ANALYSIS RESULTS")
    print(f"{'='*60}\n")
    
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.out}")
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}\n")
    
    # Top correlated features
    print("Top 5 positively correlated features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, corr) in enumerate(sorted_corr[:5], 1):
        print(f"  {i}. {feat:30s}: {corr:+.4f}")
    
    print("\nTop 5 negatively correlated features:")
    for i, (feat, corr) in enumerate(sorted_corr[-5:], 1):
        print(f"  {i}. {feat:30s}: {corr:+.4f}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*60}")
    
    spark.stop()

if __name__ == "__main__":
    main()
