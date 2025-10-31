# code_v2/features/text_preprocessing_v2.py
# Author: Võ Thị Diễm Thanh (Features & Models)
# Day 2: Text preprocessing với NULL handling + robust input (no-text fallback)

from pyspark.sql import functions as F
import re

# ---------- helpers ----------
def resolve_text_col(df, preferred=None):
    """
    Trả về (col_name, exists_flag).
    - Nếu preferred được truyền và có trong df: dùng luôn.
    - Nếu không, thử lần lượt các alias phổ biến.
    - Nếu không có cột text nào: trả về ("__synthetic_empty__", False)
    """
    if preferred and preferred in df.columns:
        return preferred, True
    candidates = ["review_text", "review_body", "reviewText", "text"]
    for c in candidates:
        if c in df.columns:
            return c, True
    return "__synthetic_empty__", False

def clean_text_udf():
    """UDF để clean text (handle NULL, HTML, special chars)"""
    def clean(text):
        if text is None or str(text).strip() == "":
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

def uppercase_ratio_udf():
    def calc_ratio(text):
        if not text:
            return 0.0
        s = str(text)
        if len(s) == 0:
            return 0.0
        upper = sum(1 for ch in s if ch.isupper())
        return upper / len(s)
    return F.udf(calc_ratio, "double")

# ---------- main featurizer ----------
def add_text_features_v2(df, text_col=None):
    """
    Thêm text features với NULL handling.

    Features:
      - cleaned_text, has_text
      - text_length, word_count, sentence_count, avg_word_length
      - exclamation_count, question_count, uppercase_ratio
    """
    # Xác định cột text
    resolved_col, exists = resolve_text_col(df, preferred=text_col)

    if not exists:
        # Không có cột text: tạo cột rỗng để pipeline không lỗi
        df = df.withColumn(resolved_col, F.lit(""))
        print(
            "[WARN] Không tìm thấy cột text trong input. "
            "Đã tạo cột rỗng và bỏ qua các đặc trưng phụ thuộc nội dung."
        )
    else:
        print(f"[INFO] Dùng cột text: {resolved_col}")

    clean_udf = clean_text_udf()

    # Cleaned text
    df = df.withColumn("cleaned_text", clean_udf(F.col(resolved_col)))

    # Has text
    df = df.withColumn(
        "has_text",
        F.when((F.col("cleaned_text").isNotNull()) & (F.length("cleaned_text") > 0), 1).otherwise(0)
    )

    # Text length (characters)
    df = df.withColumn("text_length", F.coalesce(F.length("cleaned_text"), F.lit(0)))

    # Word count
    df = df.withColumn(
        "word_count",
        F.when(F.col("has_text") == 1, F.size(F.split(F.col("cleaned_text"), r"\s+"))).otherwise(0)
    )

    # Sentence count (approximation)
    df = df.withColumn(
        "sentence_count",
        F.when(F.col("has_text") == 1, F.size(F.split(F.col("cleaned_text"), r"[\.!?]+")) - 1).otherwise(0)
    )

    # Average word length
    df = df.withColumn(
        "avg_word_length",
        F.when(F.col("word_count") > 0, F.col("text_length") / F.col("word_count")).otherwise(0.0)
    )

    # Exclamation / question counts (dùng original text để không mất dấu khi lower/clean)
    df = df.withColumn(
        "exclamation_count",
        F.when(F.col("has_text") == 1, F.size(F.split(F.col(resolved_col), "!")) - 1).otherwise(0)
    )
    df = df.withColumn(
        "question_count",
        F.when(F.col("has_text") == 1, F.size(F.split(F.col(resolved_col), r"\?")) - 1).otherwise(0)
    )

    # Uppercase ratio (trên original text)
    df = df.withColumn("uppercase_ratio", uppercase_ratio_udf()(F.col(resolved_col)))

    return df

def run():
    import argparse
    from pyspark.sql import SparkSession

    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode",   default="overwrite")
    p.add_argument("--save",   action="store_true")
    p.add_argument("--text-col", default=None, help="Tên cột chứa text (tuỳ chọn)")
    args = p.parse_args()

    spark = SparkSession.builder.appName("TextPreprocessingV2").getOrCreate()
    df_in = spark.read.parquet(args.input)

    df_out = add_text_features_v2(df_in, text_col=args.text_col)

    if args.save:
        (df_out.repartition(16)
              .write.mode(args.mode)
              .parquet(args.output))
        print(f"Saved to {args.output}")
    else:
        df_out.show(10, truncate=False)

    spark.stop()

if __name__ == "__main__":
    run()
