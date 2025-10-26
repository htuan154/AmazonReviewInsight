# code/features/text_preprocessing.py
# Day 1-2 của Thanh: Tiền xử lý văn bản cơ bản
# Sử dụng cho Spark ML Pipeline hoặc standalone

from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram
import re

def clean_text_udf():
    """
    UDF để làm sạch văn bản: lowercase, loại bỏ HTML tags, URLs, 
    ký tự đặc biệt (giữ lại dấu chấm câu cơ bản)
    """
    def clean(text):
        if not text:
            return ""
        # lowercase
        text = text.lower()
        # remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        # remove emails
        text = re.sub(r'\S+@\S+', ' ', text)
        # giữ chữ, số, dấu câu cơ bản (.,!?)
        text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
        # collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return F.udf(clean, "string")

def get_text_pipeline_stages(input_col="review_text", output_col="clean_tokens"):
    """
    Trả về list các Transformer stages cho Spark ML Pipeline:
    1. Clean text (UDF)
    2. Tokenizer
    3. StopWordsRemover
    
    Usage:
        from pyspark.ml import Pipeline
        stages = get_text_pipeline_stages()
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
    """
    from pyspark.ml.feature import SQLTransformer
    
    # Stage 1: Clean text
    clean_transformer = SQLTransformer(
        statement=f"SELECT *, clean_text_udf({input_col}) AS clean_{input_col} FROM __THIS__"
    )
    
    # Stage 2: Tokenize
    tokenizer = RegexTokenizer(
        inputCol=f"clean_{input_col}",
        outputCol="tokens",
        pattern=r'\W+',  # split on non-word characters
        minTokenLength=2
    )
    
    # Stage 3: Remove stopwords
    stop_remover = StopWordsRemover(
        inputCol="tokens",
        outputCol=output_col
    )
    
    return [clean_transformer, tokenizer, stop_remover]

def preprocess_text_column(df, input_col="review_text", output_col="clean_text"):
    """
    Hàm đơn giản để preprocess một cột text trong DataFrame
    Không dùng Pipeline, chỉ apply UDF
    
    Args:
        df: Spark DataFrame
        input_col: tên cột input (mặc định 'review_text')
        output_col: tên cột output (mặc định 'clean_text')
    
    Returns:
        DataFrame với cột mới đã clean
    """
    clean_udf = clean_text_udf()
    return df.withColumn(output_col, clean_udf(F.col(input_col)))

# ===== Định nghĩa is_helpful (Day 1 - Thanh phối hợp với Tuấn) =====
def define_target(df, threshold=0):
    """
    Định nghĩa target label: is_helpful
    
    Baseline: helpful_votes > threshold
    - threshold=0: bất kỳ vote nào > 0 là helpful
    - threshold=2: cần ít nhất 3 votes
    
    Args:
        df: DataFrame có cột 'helpful_votes'
        threshold: ngưỡng để xác định helpful (mặc định 0)
    
    Returns:
        DataFrame với cột 'is_helpful' (0 hoặc 1)
    """
    return df.withColumn(
        "is_helpful",
        F.when(F.col("helpful_votes") > threshold, 1).otherwise(0)
    )

# ===== Thống kê class imbalance (Day 1 EDA) =====
def analyze_class_imbalance(df, label_col="is_helpful"):
    """
    Phân tích tỷ lệ mất cân bằng dữ liệu
    
    Returns:
        Dict với thông tin về class distribution
    """
    counts = df.groupBy(label_col).count().collect()
    stats = {int(r[label_col]): int(r["count"]) for r in counts}
    
    total = sum(stats.values())
    pos = stats.get(1, 0)
    neg = stats.get(0, 0)
    
    imbalance_ratio = neg / max(pos, 1) if pos > 0 else 0
    pos_weight = neg / max(pos, 1) if pos > 0 else 1.0
    
    result = {
        "positive_samples": pos,
        "negative_samples": neg,
        "total_samples": total,
        "positive_ratio": pos / total if total > 0 else 0,
        "imbalance_ratio": imbalance_ratio,
        "recommended_pos_weight": pos_weight
    }
    
    print("\n=== Class Imbalance Analysis ===")
    print(f"Positive (helpful=1): {pos:,} ({result['positive_ratio']:.2%})")
    print(f"Negative (helpful=0): {neg:,} ({1-result['positive_ratio']:.2%})")
    print(f"Imbalance ratio (neg/pos): {imbalance_ratio:.2f}:1")
    print(f"Recommended pos_weight for training: {pos_weight:.3f}")
    
    return result

if __name__ == "__main__":
    # Test/Demo
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("TextPreprocessing-Test").getOrCreate()
    
    # Test data
    test_data = [
        (1, "This is a GREAT product! Check out http://example.com for more info.", 5),
        (2, "Terrible experience... DON'T BUY!!!!", 0),
        (3, "<b>Amazing</b> quality, very happy with my purchase.", 10)
    ]
    
    df = spark.createDataFrame(test_data, ["id", "review_text", "helpful_votes"])
    
    # Test preprocessing
    print("\n=== Testing Text Preprocessing ===")
    df_clean = preprocess_text_column(df)
    df_clean.select("review_text", "clean_text").show(truncate=False)
    
    # Test target definition
    print("\n=== Testing Target Definition ===")
    df_target = define_target(df_clean, threshold=2)
    df_target.select("helpful_votes", "is_helpful").show()
    
    # Test class imbalance
    analyze_class_imbalance(df_target)
    
    spark.stop()
