#!/usr/bin/env python3
# Spark 3.5.x | Python 3.11
# Tạo features cho test/train: metadata + sentiment + (tuỳ chọn) TF-IDF/Hashing
# -> luôn xuất cột Vector 'features' khớp với mô hình LightGBM trần.
# Bản này KHÔNG dùng Python UDF để tránh lỗi Python worker/socket trên Windows.

import argparse
from typing import List
from pyspark.sql.functions import col, when
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml import Pipeline

TEXT_CANDIDATES: List[str] = [
    "review_text", "review_body", "reviewText", "text", "body", "content"
]
USER_CANDIDATES: List[str] = ["user_id", "reviewerID", "reviewer_id", "customer_id"]
PROD_CANDIDATES: List[str] = ["product_id", "asin", "item_id"]
RATING_CANDIDATES: List[str] = ["star_rating", "overall", "rating", "stars"]
PRICE_CANDIDATES: List[str] = ["price", "product_price", "item_price"]


def add_column_if_missing(df: DataFrame, name: str, dtype: T.DataType = T.StringType(), fill=None) -> DataFrame:
    if name in df.columns:
        return df
    return df.withColumn(name, (F.lit(fill) if fill is not None else F.lit(None)).cast(dtype))


def first_existing(df: DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_columns(df: DataFrame) -> DataFrame:
    # review_text
    txt_col = first_existing(df, TEXT_CANDIDATES)
    if txt_col is None:
        df = df.withColumn("review_text", F.lit("").cast(T.StringType()))
    elif txt_col != "review_text":
        df = df.withColumn("review_text", F.col(txt_col).cast(T.StringType()))

    # user_id
    user_col = first_existing(df, USER_CANDIDATES)
    if user_col and user_col != "user_id":
        df = df.withColumn("user_id", F.col(user_col).cast(T.StringType()))
    else:
        df = add_column_if_missing(df, "user_id", T.StringType())

    # product_id
    prod_col = first_existing(df, PROD_CANDIDATES)
    if prod_col and prod_col != "product_id":
        df = df.withColumn("product_id", F.col(prod_col).cast(T.StringType()))
    else:
        df = add_column_if_missing(df, "product_id", T.StringType())

    # star_rating
    rating_col = first_existing(df, RATING_CANDIDATES)
    if rating_col and rating_col != "star_rating":
        df = df.withColumn("star_rating", F.col(rating_col).cast(T.DoubleType()))
    else:
        df = add_column_if_missing(df, "star_rating", T.DoubleType())

    # price
    price_col = first_existing(df, PRICE_CANDIDATES)
    if price_col and price_col != "price":
        df = df.withColumn("price", F.col(price_col).cast(T.DoubleType()))
    else:
        df = add_column_if_missing(df, "price", T.DoubleType())

    return df


def clean_text_expr(col: F.Column) -> F.Column:
    # lower -> xoá URL/email/html -> bỏ ký tự không chữ số/khoảng trắng -> rút gọn space
    c = F.lower(col)
    c = F.regexp_replace(c, r"https?://\S+|www\.\S+", " ")
    c = F.regexp_replace(c, r"\S+@\S+\.\S+", " ")
    c = F.regexp_replace(c, r"<[^>]+>", " ")
    c = F.regexp_replace(c, r"[^\w\s]", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def add_text_and_sentiment(df: DataFrame) -> DataFrame:
    # cleaned_text
    df = df.withColumn("cleaned_text", clean_text_expr(F.col("review_text")))

    # token array (dùng cho sentiment nhanh gọn, không UDF)
    tokens = F.split(F.col("cleaned_text"), r"\s+")
    df = df.withColumn("word_count", F.when(F.length("cleaned_text") > 0, F.size(tokens)).otherwise(F.lit(0)))
    df = df.withColumn("char_count", F.length("cleaned_text"))
    df = df.withColumn(
        "avg_word_len",
        F.when(F.col("word_count") > 0, (F.col("char_count") / F.col("word_count")).cast("double")).otherwise(F.lit(0.0))
    )
    df = df.withColumn("is_long_review", (F.col("word_count") >= F.lit(100)).cast("int"))

    # sentiment: đếm giao nhau giữa tokens và từ điển nhỏ (mảng literal) -> size(array_intersect)
    pos_list = [
        "good","great","excellent","amazing","awesome","love","loved","like","liked",
        "fantastic","perfect","best","wonderful","happy","satisfied","recommend",
        "enjoy","enjoyed","nice","positive","superb","brilliant","thumbs","up"
    ]
    neg_list = [
        "bad","terrible","awful","hate","hated","dislike","disliked","worst","poor",
        "boring","disappointed","disappointing","broken","waste","refund","return",
        "negative","bug","issue","problem","not","no","never"
    ]
    pos_arr = F.array(*[F.lit(x) for x in pos_list])
    neg_arr = F.array(*[F.lit(x) for x in neg_list])

    df = df.withColumn("sent_pos", F.size(F.array_intersect(tokens, pos_arr)))
    df = df.withColumn("sent_neg", F.size(F.array_intersect(tokens, neg_arr)))
    df = df.withColumn(
        "sent_score",
        F.when(F.col("word_count") > 0, (F.col("sent_pos") - F.col("sent_neg")) / F.col("word_count").cast("double"))
         .otherwise(F.lit(0.0))
    )
    return df


def add_metadata(df: DataFrame) -> DataFrame:
    df = df.withColumn("review_length", F.length(F.col("review_text")))
    df = df.withColumn(
        "review_length_log",
        F.when(F.col("review_length") > 0, F.log1p(F.col("review_length"))).otherwise(F.lit(0.0))
    )
    if "price" in df.columns:
        df = df.withColumn(
            "price_log",
            F.when(F.col("price").isNotNull() & (F.col("price") > 0), F.log1p(F.col("price")))
             .otherwise(F.lit(None).cast("double"))
        )

    df = df.withColumn(
        "rating_deviation",
        F.when(F.col("star_rating").isNotNull(), F.col("star_rating") - F.lit(3.0))
         .otherwise(F.lit(None).cast("double"))
    )

    if "user_id" in df.columns:
        w_user = Window.partitionBy("user_id")
        df = df.withColumn("user_review_count", F.count(F.lit(1)).over(w_user).cast("long"))
        df = df.withColumn("user_avg_rating", F.avg(F.col("star_rating")).over(w_user))
        df = df.withColumn("user_helpful_ratio", F.lit(None).cast("double"))

    if "product_id" in df.columns:
        w_prod = Window.partitionBy("product_id")
        df = df.withColumn("product_review_count", F.count(F.lit(1)).over(w_prod).cast("long"))
        df = df.withColumn("product_avg_rating", F.avg(F.col("star_rating")).over(w_prod))
        df = df.withColumn("product_helpful_ratio", F.lit(None).cast("double"))

    return df


def select_numeric_columns(df: DataFrame) -> list[str]:
    numeric = [
        "review_length","review_length_log","rating_deviation",
        "user_review_count","product_review_count",
        "user_avg_rating","product_avg_rating",
        "user_helpful_ratio","product_helpful_ratio",
        "word_count","char_count","avg_word_len","is_long_review",
        "sent_pos","sent_neg","sent_score","price","price_log","star_rating"
    ]
    return [c for c in numeric if c in df.columns]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["error","append","overwrite","ignore"], default="overwrite")
    p.add_argument("--preset", choices=["full","fast"], default="full",
                   help="full: có TF-IDF/Hashing; fast: chỉ numeric/metadata/sentiment.")
    p.add_argument("--numFeatures", type=int, default=20000, help="Chiều HashingTF (preset full)")
    p.add_argument("--minDF", type=int, default=5, help="Min DF cho IDF (lọc từ quá hiếm)")
    p.add_argument("--save", action="store_true", help="Ghi parquet ra --output")
    return p.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("FeaturePipelineV2").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[INFO] Reading: {args.input}")
    base = spark.read.parquet(args.input)
    base = normalize_columns(base)
    base = add_text_and_sentiment(base)
    base = add_metadata(base)

    numeric_cols = select_numeric_columns(base)

    stages = []
    feature_inputs = []

    if args.preset == "full":
        # Tokenize + HashingTF + IDF cho cleaned_text
        tok = RegexTokenizer(inputCol="cleaned_text", outputCol="tokens", pattern=r"\s+", minTokenLength=1)
        tf  = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=args.numFeatures, binary=False)
        idf = IDF(inputCol="tf", outputCol="text_tfidf", minDocFreq=args.minDF)
        stages += [tok, tf, idf]
        feature_inputs = numeric_cols + ["text_tfidf"]
    else:
        feature_inputs = numeric_cols

    va = VectorAssembler(inputCols=feature_inputs, outputCol="features", handleInvalid="keep")
    stages.append(va)

    pipe = Pipeline(stages=stages)
    model = pipe.fit(base)          # IDF cần fit
    out   = model.transform(base)

    # Cột để quan sát nhanh (không in trùng với numeric)
    debug_cols = [c for c in ["user_id","product_id","review_text","cleaned_text"] if c in out.columns]
    sel_cols = debug_cols + [c for c in numeric_cols if c not in debug_cols]
    if "text_tfidf" in out.columns:
        sel_cols += ["text_tfidf"]
    sel_cols += ["features"]

    out = out.select(*sel_cols)

    print("[INFO] Schema:")
    out.printSchema()
    print("[INFO] Sample:")
    for r in out.limit(5).collect():
        print(r)

    if args.save:
        print(f"[INFO] Writing -> {args.output} (mode={args.mode})")
        out.write.mode(args.mode).parquet(args.output)
        print("[INFO] Done.")

    spark.stop()


if __name__ == "__main__":
    main()
