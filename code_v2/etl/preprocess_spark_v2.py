# code_v2/etl/preprocess_spark_v2.py
# Version 2: Xử lý NULL cho metadata để tránh mất 62.3% test data
# 
# Cải tiến so với v1:
# - Impute giá trị NULL thay vì skip
# - price: median per category
# - average_rating: mean rating hoặc 3.0 (neutral)
# - rating_number: 0 cho sản phẩm mới
#
# Usage:
# spark-submit --master yarn --deploy-mode client \
#   --conf spark.sql.files.maxPartitionBytes=256m \
#   code_v2/etl/preprocess_spark_v2.py \
#   --reviews hdfs:///datasets/amazon/movies/raw/Movies_and_TV.jsonl \
#   --meta    hdfs:///datasets/amazon/movies/raw/meta_Movies_and_TV.jsonl \
#   --out     hdfs:///datasets/amazon/movies/parquet_v2

import argparse
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--meta", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repartition", type=int, default=8)
    return ap.parse_args()

def safe_parse_ts(col):
    """Convert epoch milliseconds to timestamp"""
    return F.from_unixtime(col.cast("bigint") / 1000).cast("timestamp")

def impute_metadata_nulls(meta_df):
    """
    Xử lý NULL trong metadata với các chiến lược imputation TỰ ĐỘNG:
    
    Chiến lược:
    1. Phát hiện TẤT CẢ các cột có NULL (không hardcode)
    2. Áp dụng imputation theo kiểu dữ liệu:
       - String: NULL -> "Unknown" hoặc mode
       - Numeric: NULL -> median (per category) hoặc mean
       - Special cases:
         * price: median per category
         * rating: mean per category
         * rating_number: 0
    
    Args:
        meta_df: DataFrame metadata với các trường có NULL
    
    Returns:
        DataFrame đã được impute (không còn NULL ở bất kỳ cột nào)
    """
    print("\n[INFO] === NULL Imputation Strategy (Auto-detect) ===")
    
    # 0. Phát hiện TẤT CẢ các cột có NULL
    print("\n[INFO] Analyzing NULL columns...")
    
    total_rows = meta_df.count()
    null_info = []
    
    for col_name in meta_df.columns:
        null_count = meta_df.filter(F.col(col_name).isNull()).count()
        if null_count > 0:
            null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
            col_type = meta_df.schema[col_name].dataType.typeName()
            null_info.append({
                "column": col_name,
                "null_count": null_count,
                "null_pct": null_pct,
                "type": col_type
            })
            print(f"  {col_name:30s} ({col_type:10s}): {null_count:10,} ({null_pct:6.2f}%)")
    
    if not null_info:
        print("  ✓ No NULL columns found!")
        return meta_df
    
    print(f"\n[INFO] Found {len(null_info)} columns with NULL values")
    
    # 1. Xử lý category NULL trước (cần để impute các trường khác)
    if "category" in meta_df.columns:
        print("\n[INFO] Handling 'category' NULL...")
        meta_df = meta_df.withColumn(
            "category_clean",
            F.coalesce(F.col("category"), F.lit("Unknown"))
        )
    else:
        # Tạo category_clean mặc định nếu không có cột category
        meta_df = meta_df.withColumn("category_clean", F.lit("Unknown"))
    
    # 2. Xử lý từng cột có NULL theo kiểu dữ liệu
    print("\n[INFO] Imputing NULL values by column type...")
    
    imputed_columns = {}  # Track imputed columns
    category_window = Window.partitionBy("category_clean")
    
    for info in null_info:
        col_name = info["column"]
        col_type = info["type"]
        
        print(f"\n  Processing: {col_name} ({col_type})")
        
        # Skip category (đã xử lý)
        if col_name == "category":
            imputed_columns[col_name] = "category_clean"
            continue
        
        # Skip product_id (không impute - là key)
        if col_name in ["product_id", "parent_asin", "asin"]:
            print(f"    -> Skip (key column)")
            continue
        
        # === NUMERIC COLUMNS ===
        if col_type in ["double", "float", "integer", "long"]:
            
            # Special case: price_cleaned
            if col_name == "price_cleaned":
                print(f"    -> Strategy: median per category")
                
                # Global median
                global_median = meta_df.select(
                    F.expr(f"percentile_approx({col_name}, 0.5)").alias("median")
                ).first()["median"]
                
                if global_median is None:
                    global_median = 0.0
                
                print(f"       Global median: {global_median:.2f}")
                
                # Category median
                meta_df = meta_df.withColumn(
                    f"{col_name}_cat_median",
                    F.expr(f"percentile_approx({col_name}, 0.5)").over(category_window)
                )
                
                # Impute
                meta_df = meta_df.withColumn(
                    f"{col_name}_imputed",
                    F.coalesce(
                        F.col(col_name),
                        F.col(f"{col_name}_cat_median"),
                        F.lit(global_median)
                    )
                )
                
                imputed_columns[col_name] = f"{col_name}_imputed"
            
            # Special case: average_rating
            elif col_name == "average_rating":
                print(f"    -> Strategy: mean per category")
                
                # Global mean
                global_mean = meta_df.agg(F.mean(col_name).alias("mean")).first()["mean"]
                if global_mean is None:
                    global_mean = 3.0
                
                print(f"       Global mean: {global_mean:.2f}")
                
                # Category mean
                meta_df = meta_df.withColumn(
                    f"{col_name}_cat_mean",
                    F.avg(col_name).over(category_window)
                )
                
                # Impute
                meta_df = meta_df.withColumn(
                    f"{col_name}_imputed",
                    F.coalesce(
                        F.col(col_name),
                        F.col(f"{col_name}_cat_mean"),
                        F.lit(global_mean),
                        F.lit(3.0)
                    )
                )
                
                imputed_columns[col_name] = f"{col_name}_imputed"
            
            # Special case: rating_number
            elif col_name == "rating_number":
                print(f"    -> Strategy: 0 (new product)")
                meta_df = meta_df.withColumn(
                    f"{col_name}_imputed",
                    F.coalesce(F.col(col_name), F.lit(0))
                )
                imputed_columns[col_name] = f"{col_name}_imputed"
            
            # Generic numeric: use median per category
            else:
                print(f"    -> Strategy: median per category (generic)")
                
                # Global median
                global_median = meta_df.select(
                    F.expr(f"percentile_approx({col_name}, 0.5)").alias("median")
                ).first()["median"]
                
                if global_median is None:
                    global_median = 0.0
                
                # Category median
                meta_df = meta_df.withColumn(
                    f"{col_name}_cat_median",
                    F.expr(f"percentile_approx({col_name}, 0.5)").over(category_window)
                )
                
                # Impute
                meta_df = meta_df.withColumn(
                    f"{col_name}_imputed",
                    F.coalesce(
                        F.col(col_name),
                        F.col(f"{col_name}_cat_median"),
                        F.lit(global_median)
                    )
                )
                
                imputed_columns[col_name] = f"{col_name}_imputed"
        
        # === STRING COLUMNS ===
        elif col_type == "string":
            print(f"    -> Strategy: 'Unknown' (generic)")
            
            meta_df = meta_df.withColumn(
                f"{col_name}_imputed",
                F.coalesce(F.col(col_name), F.lit("Unknown"))
            )
            
            imputed_columns[col_name] = f"{col_name}_imputed"
        
        # === UNSUPPORTED TYPES ===
        else:
            print(f"    -> Strategy: NULL (unsupported type)")
            imputed_columns[col_name] = col_name  # Keep as-is
    
    # 3. Log NULL counts after imputation
    print("\n[INFO] === NULL Counts After Imputation ===")
    
    for orig_col, imputed_col in imputed_columns.items():
        if imputed_col in meta_df.columns:
            null_after = meta_df.filter(F.col(imputed_col).isNull()).count()
            print(f"  {imputed_col:40s}: {null_after:10,}")
    
    # 4. Select final columns (rename imputed -> original)
    print("\n[INFO] Building final schema...")
    
    final_select = []
    
    # Always include product_id
    final_select.append(F.col("product_id"))
    
    # Map other columns
    column_mapping = {
        "title": "product_title",
        "store": "brand",
        "category": "category",
        "price_cleaned": "price",
        "average_rating": "product_avg_rating_meta",
        "rating_number": "product_total_ratings"
    }
    
    for orig_col, final_name in column_mapping.items():
        if orig_col in imputed_columns:
            imputed_col = imputed_columns[orig_col]
            final_select.append(F.col(imputed_col).alias(final_name))
        elif orig_col in meta_df.columns:
            final_select.append(F.col(orig_col).alias(final_name))
    
    meta_df = meta_df.select(*final_select)
    
    print(f"[INFO] Final schema has {len(meta_df.columns)} columns")
    
    return meta_df

def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName("ETL-Amazon-Movies-V2-NullHandling")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "="*80)
    print("ETL V2: NULL HANDLING FOR METADATA")
    print("="*80)

    # ===== 1) Đọc reviews JSONL =====
    print(f"\n[INFO] Reading reviews from {args.reviews}")
    reviews = (
        spark.read.json(args.reviews, multiLine=False)
        .select(
            F.col("asin").alias("review_id"),  # asin = unique review identifier
            F.col("text").alias("review_text"),
            F.col("rating").alias("star_rating"),
            F.col("helpful_vote").alias("helpful_votes"),
            F.col("user_id"),
            F.col("parent_asin").alias("product_id"),
            F.col("timestamp").alias("ts_raw")
        )
    )

    # Chuẩn hoá kiểu dữ liệu
    reviews = (
        reviews
        .withColumn("review_id", F.col("review_id").cast("string"))  # Ensure string type
        .withColumn("star_rating", F.col("star_rating").cast("double"))
        .withColumn("helpful_votes", F.col("helpful_votes").cast("long"))
        .withColumn("review_text", F.col("review_text").cast("string"))
        .withColumn("ts", safe_parse_ts(F.col("ts_raw")))
        .drop("ts_raw")
    )
    
    # Verify review_id uniqueness
    total_count = reviews.count()
    unique_count = reviews.select("review_id").distinct().count()
    
    print(f"[INFO] Total reviews: {total_count:,}")
    print(f"[INFO] Unique review_ids: {unique_count:,}")
    
    if total_count != unique_count:
        print(f"[ERROR] review_id NOT unique! Duplicates: {total_count - unique_count:,}")
        raise ValueError("review_id (asin) must be unique for each review")
    else:
        print(f"[OK] review_id is unique ✓")

    # Cột phân vùng
    reviews = (
        reviews
        .withColumn("year",  F.year("ts"))
        .withColumn("month", F.month("ts"))
    )
    
    print(f"[INFO] Reviews loaded: {reviews.count():,}")

    # ===== 2) Đọc metadata JSONL với xử lý NULL =====
    if args.meta:
        print(f"\n[INFO] Reading metadata from {args.meta}")
        
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
                F.col("price_cleaned"),
                F.col("average_rating"),
                F.col("rating_number")
            )
        )
        
        print(f"[INFO] Metadata records (raw): {meta.count():,}")
        
        # ===== IMPUTE NULL VALUES =====
        meta = impute_metadata_nulls(meta)
        
        print(f"[INFO] Metadata records (after imputation): {meta.count():,}")
        
        # Left join - GIỜ ĐÃY KHÔNG CÒN NULL Ở CÁC TRƯỜNG QUAN TRỌNG
        reviews = reviews.join(meta, on="product_id", how="left")
        
        # Fill remaining NULLs sau join (trường hợp product_id không match)
        reviews = reviews.fillna({
            "price": 0.0,
            "product_avg_rating_meta": 3.0,
            "product_total_ratings": 0,
            "category": "Unknown"
        })
        
        print(f"[INFO] Reviews after metadata join: {reviews.count():,}")
        
        # Kiểm tra NULL sau join
        null_check = reviews.select([
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in ["price", "product_avg_rating_meta", "product_total_ratings", "category"]
        ])
        
        print("\n[INFO] NULL counts after join + fillna:")
        null_check.show()

    # ===== 3) Thêm đặc trưng cơ bản =====
    reviews = (
        reviews
        .withColumn("review_length", F.size(F.split(F.coalesce(F.col("review_text"), F.lit("")), r"\s+")))
        .withColumn("is_helpful", (F.col("helpful_votes") > F.lit(0)).cast("int"))
    )

    # ===== 4) EDA nhanh =====
    hv_dist = (
        reviews
        .groupBy("helpful_votes")
        .count()
        .orderBy("helpful_votes")
        .limit(20)  # Top 20 để tránh quá nhiều dòng
    )
    
    hv_path = f"{args.out}/eda_helpful_votes_csv"
    hv_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(hv_path)

    cls_dist = reviews.groupBy("is_helpful").count()
    cls_dist_path = f"{args.out}/eda_class_ratio_csv"
    cls_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(cls_dist_path)
    
    # Category distribution
    cat_dist = reviews.groupBy("category").count().orderBy(F.desc("count"))
    cat_path = f"{args.out}/eda_category_dist_csv"
    cat_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(cat_path)

    # ===== 5) Ghi Parquet =====
    out_reviews = f"{args.out}/reviews"
    (
        reviews
        .repartition(args.repartition, "year", "month")
        .write.mode("overwrite")
        .partitionBy("year", "month")
        .option("compression", "snappy")
        .parquet(out_reviews)
    )

    print("\n" + "="*80)
    print("=== DONE ===")
    print("="*80)
    print(f"Parquet (V2 - NULL handled) -> {out_reviews}")
    print(f"EDA helpful_votes CSV -> {hv_path}")
    print(f"EDA class ratio CSV -> {cls_dist_path}")
    print(f"EDA category dist CSV -> {cat_path}")
    print("\nKey improvements:")
    print("  ✓ NULL values imputed (price, rating, rating_number)")
    print("  ✓ No records dropped due to handleInvalid='skip'")
    print("  ✓ Ready for production inference")

    spark.stop()

if __name__ == "__main__":
    main()
