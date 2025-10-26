# preprocess_spark.py
# Usage:
# spark-submit --master yarn --deploy-mode client \
#   --conf spark.sql.files.maxPartitionBytes=256m \
#   preprocess_spark.py \
#   --reviews hdfs:///datasets/amazon/movies/raw/Movies_and_TV.jsonl \
#   --meta    hdfs:///datasets/amazon/movies/raw/meta_Movies_and_TV.jsonl \
#   --out     hdfs:///datasets/amazon/movies/parquet

import argparse
from pyspark.sql import SparkSession, functions as F, types as T

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--meta", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repartition", type=int, default=8)  # single-node: 4–12 tuỳ CPU/SSD
    return ap.parse_args()

def safe_parse_ts(col):
    # Dataset có timestamp là epoch MILLISECONDS (int) -> convert to timestamp
    # Divide by 1000 to convert milliseconds to seconds
    return F.from_unixtime(col.cast("bigint") / 1000).cast("timestamp")

def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName("ETL-Amazon-Movies")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ===== 1) Đọc reviews JSONL =====
    # Actual schema: asin, helpful_vote, images, parent_asin, rating, text, timestamp, title, user_id, verified_purchase
    reviews = (
        spark.read.json(args.reviews, multiLine=False)
        .select(
            F.col("asin").alias("review_id"),  # Use asin as review_id
            F.col("text").alias("review_text"),
            F.col("rating").alias("star_rating"),
            F.col("helpful_vote").alias("helpful_votes"),
            F.col("user_id").alias("user_id"),
            F.col("parent_asin").alias("product_id"),
            F.col("timestamp").alias("ts_raw")
        )
    )

    # Chuẩn hoá kiểu dữ liệu
    reviews = (
        reviews
        .withColumn("star_rating", F.col("star_rating").cast("double"))
        .withColumn("helpful_votes", F.col("helpful_votes").cast("long"))
        .withColumn("review_text", F.col("review_text").cast("string"))
        .withColumn("ts", safe_parse_ts(F.col("ts_raw")))
        .drop("ts_raw")
    )

    # Cột phân vùng
    reviews = (
        reviews
        .withColumn("year",  F.year("ts"))
        .withColumn("month", F.month("ts"))
    )

    # ===== 2) Đọc metadata JSONL (xử lý nested struct cẩn thận) =====
    if args.meta:
        print(f"[INFO] Reading metadata from {args.meta}")
        
        # Read as RDD first, manually parse JSON to avoid duplicate column errors
        import json
        
        meta_rdd = spark.sparkContext.textFile(args.meta)
        
        def safe_parse_meta(line):
            """Safely extract top-level fields only"""
            try:
                obj = json.loads(line)
                return (
                    obj.get("parent_asin", ""),
                    obj.get("title", ""),
                    obj.get("store", ""),
                    obj.get("main_category", ""),
                    obj.get("price", None),
                    obj.get("average_rating", None),
                    obj.get("rating_number", None)
                )
            except:
                return ("", "", "", "", None, None, None)
        
        meta_data = meta_rdd.map(safe_parse_meta)
        
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
        meta_schema = StructType([
            StructField("parent_asin", StringType(), True),
            StructField("title", StringType(), True),
            StructField("store", StringType(), True),
            StructField("main_category", StringType(), True),
            StructField("price", StringType(), True),
            StructField("average_rating", DoubleType(), True),
            StructField("rating_number", IntegerType(), True)
        ])
        
        meta = spark.createDataFrame(meta_data, schema=meta_schema)
        
        # Clean price field
        meta = meta.withColumn(
            "price_cleaned",
            F.when(F.col("price").isNotNull(),
                   F.regexp_replace(F.col("price").cast("string"), r'[^\d.]', '').cast("double"))
            .otherwise(F.lit(None).cast("double"))
        )
        
        # Rename for join
        meta = (
            meta
            .select(
                F.col("parent_asin").alias("product_id"),
                F.col("title").alias("product_title"),
                F.col("store").alias("brand"),
                F.col("main_category").alias("category"),
                F.col("price_cleaned").alias("price"),
                F.col("average_rating").alias("product_avg_rating_meta"),
                F.col("rating_number").alias("product_total_ratings")
            )
        )
        
        print(f"[INFO] Metadata records: {meta.count():,}")
        
        # Left join
        reviews = reviews.join(meta, on="product_id", how="left")
        print(f"[INFO] Reviews after metadata join: {reviews.count():,}")

    # ===== 3) Thêm đặc trưng cơ bản để sẵn sàng train =====
    reviews = (
        reviews
        .withColumn("review_length", F.size(F.split(F.coalesce(F.col("review_text"), F.lit("")), r"\s+")))
        .withColumn("is_helpful", (F.col("helpful_votes") > F.lit(0)).cast("int"))  # baseline threshold=0, có thể đổi
    )

    # ===== 4) EDA nhanh: phân phối helpful_votes & tỉ lệ class =====
    hv_dist = (
        reviews
        .groupBy("helpful_votes")
        .count()
        .orderBy("helpful_votes")
    )
    # Lưu EDA ra CSV (1 file) để xem nhanh
    hv_path = f"{args.out}/eda_helpful_votes_csv"
    hv_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(hv_path)

    cls_dist = reviews.groupBy("is_helpful").count()
    cls_dist_path = f"{args.out}/eda_class_ratio_csv"
    cls_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(cls_dist_path)

    # ===== 5) Ghi Parquet phân vùng (year, month); nén snappy =====
    out_reviews = f"{args.out}/reviews"
    (
        reviews
        .repartition(args.repartition, "year", "month")
        .write.mode("overwrite")
        .partitionBy("year", "month")
        .option("compression", "snappy")
        .parquet(out_reviews)
    )

    print("=== DONE ===")
    print(f"Parquet -> {out_reviews}")
    print(f"EDA helpful_votes CSV -> {hv_path}")
    print(f"EDA class ratio CSV -> {cls_dist_path}")

    spark.stop()

if __name__ == "__main__":
    main()
