# code_v2/features/sentiment_vader_v2.py
# Author: Võ Thị Diễm Thanh (Features & Models)
# Day 2-3: Sentiment analysis với VADER - NULL safe
#
# Cải tiến so với V1:
# - Handle NULL text
# - Batch processing để tăng tốc
# - Thêm sentiment categories

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer (singleton)
_analyzer = None

def get_analyzer():
    """Get or create VADER analyzer singleton"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def sentiment_compound_udf():
    """UDF for compound sentiment score"""
    def get_compound(text):
        if text is None or text.strip() == "":
            return 0.0
        analyzer = get_analyzer()
        scores = analyzer.polarity_scores(text)
        return scores['compound']
    
    return F.udf(get_compound, DoubleType())

def sentiment_pos_udf():
    """UDF for positive sentiment score"""
    def get_pos(text):
        if text is None or text.strip() == "":
            return 0.0
        analyzer = get_analyzer()
        scores = analyzer.polarity_scores(text)
        return scores['pos']
    
    return F.udf(get_pos, DoubleType())

def sentiment_neg_udf():
    """UDF for negative sentiment score"""
    def get_neg(text):
        if text is None or text.strip() == "":
            return 0.0
        analyzer = get_analyzer()
        scores = analyzer.polarity_scores(text)
        return scores['neg']
    
    return F.udf(get_neg, DoubleType())

def sentiment_neu_udf():
    """UDF for neutral sentiment score"""
    def get_neu(text):
        if text is None or text.strip() == "":
            return 1.0  # Neutral if no text
        analyzer = get_analyzer()
        scores = analyzer.polarity_scores(text)
        return scores['neu']
    
    return F.udf(get_neu, DoubleType())

def sentiment_category_udf():
    """UDF for sentiment category"""
    def get_category(compound):
        if compound is None:
            return "neutral"
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    return F.udf(get_category, StringType())

def add_sentiment_features_v2(df, text_column="review_text"):
    """
    Thêm sentiment features với NULL handling
    
    Features:
    - sentiment_compound: điểm tổng hợp (-1 to 1)
    - sentiment_pos: điểm positive (0 to 1)
    - sentiment_neg: điểm negative (0 to 1)
    - sentiment_neu: điểm neutral (0 to 1)
    - sentiment_category: categorical (positive/negative/neutral)
    - sentiment_strength: |compound| (độ mạnh)
    - is_polarized: indicator cho sentiment rõ ràng
    """
    print("\n[INFO] Adding sentiment features (VADER)...")
    
    # Get sentiment scores
    df = df.withColumn("sentiment_compound", sentiment_compound_udf()(F.col(text_column)))
    df = df.withColumn("sentiment_pos", sentiment_pos_udf()(F.col(text_column)))
    df = df.withColumn("sentiment_neg", sentiment_neg_udf()(F.col(text_column)))
    df = df.withColumn("sentiment_neu", sentiment_neu_udf()(F.col(text_column)))
    
    # Sentiment category
    df = df.withColumn("sentiment_category", sentiment_category_udf()(F.col("sentiment_compound")))
    
    # Sentiment strength
    df = df.withColumn("sentiment_strength", F.abs(F.col("sentiment_compound")))
    
    # Is polarized (strong opinion)
    df = df.withColumn(
        "is_polarized",
        F.when(F.col("sentiment_strength") >= 0.5, 1).otherwise(0)
    )
    
    # Sentiment-rating alignment (consistent với star_rating?)
    if "star_rating" in df.columns:
        df = df.withColumn(
            "sentiment_rating_alignment",
            F.when(
                # Positive sentiment + high rating
                (F.col("sentiment_compound") > 0.05) & (F.col("star_rating") >= 4),
                1
            ).when(
                # Negative sentiment + low rating
                (F.col("sentiment_compound") < -0.05) & (F.col("star_rating") <= 2),
                1
            ).otherwise(0)
        )
        
        # Sentiment-rating gap
        df = df.withColumn(
            "sentiment_rating_gap",
            F.abs(
                # Normalize compound (-1,1) to (1,5) scale
                ((F.col("sentiment_compound") + 1) / 2 * 4 + 1) - F.col("star_rating")
            )
        )
    
    print("  ✓ Sentiment features added")
    
    return df
# (giữ nguyên các hàm hiện có)

def run():
    import argparse
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--input",     required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--text_col",  default="review_text")  # dùng 'cleaned_text' sau bước preprocess
    p.add_argument("--mode",      default="overwrite")
    p.add_argument("--save",      action="store_true")
    args = p.parse_args()

    spark = SparkSession.builder.appName("SentimentVADER-V2").getOrCreate()
    df_in = spark.read.parquet(args.input)

    # đảm bảo cột tồn tại
    if args.text_col not in df_in.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in input. "
                         f"Available: {df_in.columns}")

    df_out = add_sentiment_features_v2(df_in, text_column=args.text_col)

    if args.save:
        (df_out.repartition(16)
              .write.mode(args.mode)
              .parquet(args.output))
        print(f"✅ Saved to {args.output}")
    else:
        df_out.select("sentiment_compound","sentiment_category").show(10)

    spark.stop()

if __name__ == "__main__":
    run()
