# code/features/sentiment_vader.py
# Day 1-2 của Thanh: Sentiment Analysis với VADER
# VADER (Valence Aware Dictionary and sEntiment Reasoner) - tốt cho social media text

"""
VADER Sentiment Analysis cho Amazon Reviews

Cài đặt (trên máy local hoặc trong notebook):
    pip install vaderSentiment

Trên Spark cluster, có thể cần:
    spark-submit --py-files vaderSentiment.zip ...
"""

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructType, StructField

def get_vader_sentiment_udf():
    """
    UDF để tính sentiment scores bằng VADER
    
    Returns:
        UDF trả về struct(compound, pos, neu, neg)
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("[WARNING] vaderSentiment not installed. Install: pip install vaderSentiment")
        # Return dummy UDF
        return F.udf(lambda x: (0.0, 0.0, 0.0, 0.0), 
                     StructType([
                         StructField("compound", FloatType(), True),
                         StructField("pos", FloatType(), True),
                         StructField("neu", FloatType(), True),
                         StructField("neg", FloatType(), True)
                     ]))
    
    analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(text):
        if not text or text.strip() == "":
            return (0.0, 0.0, 1.0, 0.0)  # neutral default
        
        scores = analyzer.polarity_scores(text)
        return (
            float(scores['compound']),
            float(scores['pos']),
            float(scores['neu']),
            float(scores['neg'])
        )
    
    schema = StructType([
        StructField("compound", FloatType(), True),
        StructField("pos", FloatType(), True),
        StructField("neu", FloatType(), True),
        StructField("neg", FloatType(), True)
    ])
    
    return F.udf(analyze_sentiment, schema)

def add_sentiment_features(df, text_col="review_text", prefix="sentiment"):
    """
    Thêm sentiment features vào DataFrame
    
    Args:
        df: Spark DataFrame
        text_col: tên cột chứa text cần phân tích
        prefix: prefix cho các cột sentiment mới
    
    Returns:
        DataFrame với các cột mới:
        - {prefix}_compound: điểm tổng hợp (-1 đến +1)
        - {prefix}_pos: tỷ lệ positive
        - {prefix}_neu: tỷ lệ neutral
        - {prefix}_neg: tỷ lệ negative
    """
    vader_udf = get_vader_sentiment_udf()
    
    df = df.withColumn("_sentiment_struct", vader_udf(F.col(text_col)))
    
    return (df
            .withColumn(f"{prefix}_compound", F.col("_sentiment_struct.compound"))
            .withColumn(f"{prefix}_pos", F.col("_sentiment_struct.pos"))
            .withColumn(f"{prefix}_neu", F.col("_sentiment_struct.neu"))
            .withColumn(f"{prefix}_neg", F.col("_sentiment_struct.neg"))
            .drop("_sentiment_struct"))

def sentiment_label_from_compound(df, compound_col="sentiment_compound", output_col="sentiment_label"):
    """
    Chuyển compound score thành label categorical
    
    Quy tắc VADER:
    - compound >= 0.05: positive
    - compound <= -0.05: negative
    - else: neutral
    """
    return df.withColumn(
        output_col,
        F.when(F.col(compound_col) >= 0.05, "positive")
         .when(F.col(compound_col) <= -0.05, "negative")
         .otherwise("neutral")
    )

# ===== Phân tích sentiment theo helpful_votes (Day 1 EDA) =====
def analyze_sentiment_by_helpfulness(df, label_col="is_helpful"):
    """
    So sánh sentiment giữa helpful vs non-helpful reviews
    
    Args:
        df: DataFrame có các cột sentiment và is_helpful
    
    Returns:
        Dict với thống kê
    """
    stats = (df.groupBy(label_col)
             .agg(
                 F.avg("sentiment_compound").alias("avg_compound"),
                 F.avg("sentiment_pos").alias("avg_pos"),
                 F.avg("sentiment_neg").alias("avg_neg"),
                 F.count("*").alias("count")
             )
             .collect())
    
    result = {int(r[label_col]): {
        "count": int(r["count"]),
        "avg_compound": float(r["avg_compound"]),
        "avg_pos": float(r["avg_pos"]),
        "avg_neg": float(r["avg_neg"])
    } for r in stats}
    
    print("\n=== Sentiment Analysis by Helpfulness ===")
    for label, data in result.items():
        label_str = "Helpful" if label == 1 else "Not Helpful"
        print(f"\n{label_str} Reviews (n={data['count']:,}):")
        print(f"  Avg Compound: {data['avg_compound']:.4f}")
        print(f"  Avg Positive: {data['avg_pos']:.4f}")
        print(f"  Avg Negative: {data['avg_neg']:.4f}")
    
    return result

if __name__ == "__main__":
    # Test/Demo
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("VADER-Test").getOrCreate()
    
    # Test data
    test_data = [
        (1, "This product is AMAZING! Best purchase ever!", 10, 1),
        (2, "Terrible quality. Waste of money. Don't buy!", 0, 0),
        (3, "It's okay, nothing special but works fine.", 2, 1),
        (4, "Absolutely love it! Highly recommended!", 15, 1),
        (5, "Disappointing. Not as described.", 0, 0)
    ]
    
    df = spark.createDataFrame(test_data, ["id", "review_text", "helpful_votes", "is_helpful"])
    
    print("\n=== Testing VADER Sentiment Analysis ===")
    
    # Add sentiment features
    df_sentiment = add_sentiment_features(df)
    df_sentiment.select(
        "review_text", 
        "sentiment_compound", 
        "sentiment_pos", 
        "sentiment_neg"
    ).show(truncate=False)
    
    # Add sentiment label
    df_labeled = sentiment_label_from_compound(df_sentiment)
    df_labeled.select("review_text", "sentiment_label").show(truncate=False)
    
    # Analyze by helpfulness
    analyze_sentiment_by_helpfulness(df_sentiment)
    
    spark.stop()
