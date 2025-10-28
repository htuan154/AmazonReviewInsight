# code_v2/features/text_preprocessing_v2.py
# Author: Võ Thị Diễm Thanh (Features & Models)
# Day 2: Text preprocessing với NULL handling
#
# Cải tiến so với V1:
# - Handle NULL/empty text
# - Chuẩn hóa Unicode
# - Loại bỏ HTML tags
# - Handle special characters tốt hơn

from pyspark.sql import functions as F
import re

def clean_text_udf():
    """
    UDF để clean text (handle NULL, HTML, special chars)
    """
    def clean(text):
        if text is None or text.strip() == "":
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s\.\!\?\,\-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase
        text = text.lower()
        
        return text
    
    return F.udf(clean, "string")

def add_text_features_v2(df):
    """
    Thêm text features với NULL handling
    
    Features:
    - cleaned_text: text đã được clean
    - text_length: độ dài text (chars)
    - word_count: số từ
    - sentence_count: số câu
    - avg_word_length: độ dài từ trung bình
    - has_text: indicator có text hay không
    - exclamation_count: số dấu !
    - question_count: số dấu ?
    - uppercase_ratio: tỷ lệ chữ hoa
    """
    # Clean text
    clean_udf = clean_text_udf()
    df = df.withColumn("cleaned_text", clean_udf(F.col("review_text")))
    
    # Has text indicator
    df = df.withColumn(
        "has_text",
        F.when(
            (F.col("cleaned_text").isNotNull()) & 
            (F.length(F.col("cleaned_text")) > 0),
            1
        ).otherwise(0)
    )
    
    # Text length (characters)
    df = df.withColumn(
        "text_length",
        F.coalesce(F.length(F.col("cleaned_text")), F.lit(0))
    )
    
    # Word count
    df = df.withColumn(
        "word_count",
        F.when(
            F.col("has_text") == 1,
            F.size(F.split(F.col("cleaned_text"), r"\s+"))
        ).otherwise(0)
    )
    
    # Sentence count (approximation)
    df = df.withColumn(
        "sentence_count",
        F.when(
            F.col("has_text") == 1,
            F.size(F.split(F.col("cleaned_text"), r"[\.!?]+")) - 1
        ).otherwise(0)
    )
    
    # Average word length
    df = df.withColumn(
        "avg_word_length",
        F.when(
            F.col("word_count") > 0,
            F.col("text_length") / F.col("word_count")
        ).otherwise(0.0)
    )
    
    # Exclamation marks
    df = df.withColumn(
        "exclamation_count",
        F.when(
            F.col("has_text") == 1,
            F.size(F.split(F.col("review_text"), "!")) - 1
        ).otherwise(0)
    )
    
    # Question marks
    df = df.withColumn(
        "question_count",
        F.when(
            F.col("has_text") == 1,
            F.size(F.split(F.col("review_text"), r"\?")) - 1
        ).otherwise(0)
    )
    
    # Uppercase ratio (in original text)
    def uppercase_ratio_udf():
        def calc_ratio(text):
            if not text or len(text) == 0:
                return 0.0
            uppercase_count = sum(1 for c in text if c.isupper())
            return uppercase_count / len(text)
        return F.udf(calc_ratio, "double")
    
    df = df.withColumn(
        "uppercase_ratio",
        uppercase_ratio_udf()(F.col("review_text"))
    )
    
    return df

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("TextPreprocessingV2-Test").getOrCreate()
    
    # Test data with various edge cases
    test_data = [
        ("r1", "This is a GREAT product! Highly recommended."),
        ("r2", None),  # NULL text
        ("r3", ""),  # Empty text
        ("r4", "   "),  # Whitespace only
        ("r5", "<html>Bad product</html> with HTML tags"),
        ("r6", "Check out http://example.com for more info!"),
        ("r7", "WHY IS THIS SO EXPENSIVE???"),
    ]
    
    df = spark.createDataFrame(test_data, ["review_id", "review_text"])
    
    print("\n=== Testing Text Preprocessing V2 ===")
    print("\nOriginal Data:")
    df.show(truncate=False)
    
    # Add text features
    df_processed = add_text_features_v2(df)
    
    print("\n--- Cleaned Text ---")
    df_processed.select("review_id", "cleaned_text", "has_text").show(truncate=False)
    
    print("\n--- Text Statistics ---")
    df_processed.select(
        "review_id", "text_length", "word_count", "sentence_count", "avg_word_length"
    ).show()
    
    print("\n--- Special Characters ---")
    df_processed.select(
        "review_id", "exclamation_count", "question_count", "uppercase_ratio"
    ).show()
    
    spark.stop()
