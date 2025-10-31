# code_v2/features/metadata_features_v2.py
# Version 2 (No-Label-Leakage): Xử lý NULL-safe + chống rò rỉ nhãn
#
# Thay đổi chính so với bản trước:
# - Sửa "has_metadata" dùng boolean đúng chuẩn (không lỗi AND int).
# - Loại bỏ feature gây rò rỉ: helpfulness_x_length = is_helpful * review_length.
# - user_helpful_ratio / product_helpful_ratio:
#     * TRAIN: dùng Leave-One-Out (LOO) target encoding theo user/product.
#     * TEST : chỉ join thống kê (sum,count) từ TRAIN rồi tính sum/count (không dùng nhãn của test).
# - Thêm tham số mode={train,test} và đường dẫn lưu/đọc stats.
#
# Cách dùng (gợi ý):
#   create_full_feature_set_v2(df,
#       mode="train",
#       user_stats_path="hdfs:///output_v2/stats/user.parquet",
#       product_stats_path="hdfs:///output_v2/stats/product.parquet",
#       spark=spark)
#
#   create_full_feature_set_v2(df_test,
#       mode="test",
#       user_stats_path="hdfs:///output_v2/stats/user.parquet",
#       product_stats_path="hdfs:///output_v2/stats/product.parquet",
#       spark=spark)

from pyspark.sql import functions as F
from pyspark.sql import Window


# =========================
# 1) BASIC METADATA FEATURES
# =========================
def add_basic_metadata_features_v2(df):
    """
    Thêm các đặc trưng metadata cơ bản với NULL handling

    Features:
    - review_length_log: log(review_length + 1)
    - is_long_review: review_length > 100
    - rating_deviation: |star_rating - 3.0|
    - has_price: 1 nếu có price > 0
    - has_product_rating: 1 nếu có meta rating & total_ratings > 0
    - has_metadata: (has_price AND has_product_rating)
    - price_log: log(price + 1) để handle giá 0
    - is_expensive: price > median_price (per category)
    """
    df = (
        df.withColumn("review_length_log", F.log1p(F.col("review_length")))
          .withColumn("is_long_review", (F.col("review_length") > 100).cast("int"))
          .withColumn("rating_deviation", F.abs(F.col("star_rating") - F.lit(3.0)))
    )

    # Indicators về chất lượng/đầy đủ metadata
    df = df.withColumn(
        "has_price",
        F.when(F.col("price").isNull() | (F.col("price") <= 0), F.lit(0)).otherwise(F.lit(1))
    )

    df = df.withColumn(
        "has_product_rating",
        F.when(
            F.col("product_avg_rating_meta").isNull()
            | F.col("product_total_ratings").isNull()
            | (F.col("product_total_ratings") <= 0),
            F.lit(0)
        ).otherwise(F.lit(1))
    )

    # Dùng boolean đúng chuẩn rồi cast về int để tránh lỗi type
    df = df.withColumn(
        "has_metadata",
        ((F.col("has_price") == 1) & (F.col("has_product_rating") == 1)).cast("int")
    )

    df = df.withColumn(
        "price_log",
        F.when(F.col("price").isNotNull() & (F.col("price") > 0), F.log1p(F.col("price")))
         .otherwise(F.lit(0.0))
    )

    # Median price theo category để đánh dấu expensive
    category_window = Window.partitionBy("category")
    df = df.withColumn(
        "category_median_price",
        F.expr("percentile_approx(price, 0.5)").over(category_window)
    ).withColumn(
        "is_expensive",
        F.when(
            F.col("price").isNotNull()
            & F.col("category_median_price").isNotNull()
            & (F.col("price") > F.col("category_median_price")),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    return df


# ======================================
# 2) USER AGG FEATURES (NO LABEL LEAKAGE)
# ======================================
def add_user_aggregate_features_v2(
    df,
    mode: str = "train",
    user_stats_path: str = None,
    spark=None
):
    """
    Tính aggregate theo user với NULL-safe + CHỐNG RÒ RỈ NHÃN.

    - mode='train':
        * Tính các thống kê không dùng target trực tiếp (count, avg rating, avg length, stddev rating).
        * Tính LOO cho user_helpful_ratio:
            ratio = (sum(is_helpful) - is_helpful_row) / max(count-1, 1)
        * Nếu có user_stats_path + spark: ghi (sum, count) ra HDFS để dùng cho TEST.

    - mode='test':
        * Đọc stats đã lưu (sum, count), join theo user_id để tính ratio = sum/count.
        * Tuyệt đối không dùng nhãn của test.
    """
    user_key = "user_id"
    if mode not in ("train", "test"):
        raise ValueError("mode must be 'train' or 'test'")

    if mode == "train":
        w = Window.partitionBy(user_key)

        # Không dùng target trực tiếp (an toàn)
        df = (
            df.withColumn("user_review_count", F.count(F.lit(1)).over(w))
              .withColumn("user_avg_rating", F.coalesce(F.avg("star_rating").over(w), F.lit(3.0)))
              .withColumn("user_avg_review_length", F.coalesce(F.avg("review_length").over(w), F.lit(50.0)))
              .withColumn("user_rating_stddev", F.coalesce(F.stddev("star_rating").over(w), F.lit(0.0)))
        )

        # LOO target encoding cho helpful ratio
        df = (
            df.withColumn("_user_sum_h", F.sum(F.col("is_helpful").cast("double")).over(w))
              .withColumn("_user_cnt", F.count(F.lit(1)).over(w))
              .withColumn(
                  "user_helpful_ratio",
                  F.when(F.col("_user_cnt") > 1,
                         (F.col("_user_sum_h") - F.col("is_helpful").cast("double"))
                         / (F.col("_user_cnt") - F.lit(1)))
                   .otherwise(F.lit(0.0))
              )
        )

        # user_consistency (ngược chuẩn hóa stddev)
        df = df.withColumn(
            "user_consistency",
            F.when(F.col("user_rating_stddev") == 0, F.lit(1.0))
             .otherwise(F.lit(1.0) / (F.lit(1.0) + F.col("user_rating_stddev")))
        )

        # Lưu stats (sum, count) từ TRAIN để TEST dùng
        if user_stats_path and spark is not None:
            user_stats = (
                df.select(user_key, "_user_sum_h", "_user_cnt")
                  .dropDuplicates([user_key])
            )
            (user_stats.write.mode("overwrite").parquet(user_stats_path))

        return df.drop("_user_sum_h", "_user_cnt")

    # mode == "test"
    assert spark is not None and user_stats_path is not None, \
        "Need spark and user_stats_path in test mode"

    stats = (
        spark.read.parquet(user_stats_path)
             .dropDuplicates([user_key])
             .withColumnRenamed("_user_sum_h", "user_sum_h")
             .withColumnRenamed("_user_cnt", "user_cnt")
    )
    df = df.join(stats, on=user_key, how="left").fillna({"user_sum_h": 0.0, "user_cnt": 0})

    # ratio_test = sum/count (KHÔNG dùng nhãn của test)
    df = df.withColumn(
        "user_helpful_ratio",
        F.when(F.col("user_cnt") > 0, F.col("user_sum_h") / F.col("user_cnt")).otherwise(F.lit(0.0))
    )

    # Các thống kê không dùng target có thể tính trực tiếp
    w = Window.partitionBy(user_key)
    df = (
        df.withColumn("user_review_count", F.count(F.lit(1)).over(w))
          .withColumn("user_avg_rating", F.coalesce(F.avg("star_rating").over(w), F.lit(3.0)))
          .withColumn("user_avg_review_length", F.coalesce(F.avg("review_length").over(w), F.lit(50.0)))
          .withColumn("user_rating_stddev", F.coalesce(F.stddev("star_rating").over(w), F.lit(0.0)))
          .withColumn(
              "user_consistency",
              F.when(F.col("user_rating_stddev") == 0, F.lit(1.0))
               .otherwise(F.lit(1.0) / (F.lit(1.0) + F.col("user_rating_stddev")))
          )
    )

    return df


# ========================================
# 3) PRODUCT AGG FEATURES (NO LABEL LEAKAGE)
# ========================================
def add_product_aggregate_features_v2(
    df,
    mode: str = "train",
    product_stats_path: str = None,
    spark=None
):
    """
    Tính aggregate theo product với NULL-safe + CHỐNG RÒ RỈ NHÃN.

    - mode='train':
        * Tính thống kê an toàn: count, avg rating, avg length, stddev rating.
        * LOO cho product_helpful_ratio (sum - is_helpful_row)/(count-1).
        * Nếu có product_stats_path + spark: ghi (sum, count) ra HDFS.

    - mode='test':
        * Join stats từ TRAIN (sum,count), ratio = sum/count.
    """
    prod_key = "product_id"
    if mode not in ("train", "test"):
        raise ValueError("mode must be 'train' or 'test'")

    if mode == "train":
        w = Window.partitionBy(prod_key)
        df = (
            df.withColumn("product_review_count", F.count(F.lit(1)).over(w))
              .withColumn("product_avg_rating", F.coalesce(F.avg("star_rating").over(w), F.lit(3.0)))
              .withColumn("product_avg_review_length", F.coalesce(F.avg("review_length").over(w), F.lit(50.0)))
              .withColumn("product_rating_stddev", F.coalesce(F.stddev("star_rating").over(w), F.lit(0.0)))
        )

        df = (
            df.withColumn("_prod_sum_h", F.sum(F.col("is_helpful").cast("double")).over(w))
              .withColumn("_prod_cnt", F.count(F.lit(1)).over(w))
              .withColumn(
                  "product_helpful_ratio",
                  F.when(F.col("_prod_cnt") > 1,
                         (F.col("_prod_sum_h") - F.col("is_helpful").cast("double"))
                         / (F.col("_prod_cnt") - F.lit(1)))
                   .otherwise(F.lit(0.0))
              )
        )

        # chênh lệch meta rating vs avg review rating
        df = df.withColumn(
            "meta_review_rating_gap",
            F.when(F.col("product_avg_rating_meta").isNotNull(),
                   F.abs(F.col("product_avg_rating_meta") - F.col("product_avg_rating")))
             .otherwise(F.lit(0.0))
        )

        if product_stats_path and spark is not None:
            prod_stats = (
                df.select(prod_key, "_prod_sum_h", "_prod_cnt")
                  .dropDuplicates([prod_key])
            )
            (prod_stats.write.mode("overwrite").parquet(product_stats_path))

        return df.drop("_prod_sum_h", "_prod_cnt")

    # mode == "test"
    assert spark is not None and product_stats_path is not None, \
        "Need spark and product_stats_path in test mode"

    stats = (
        spark.read.parquet(product_stats_path)
             .dropDuplicates([prod_key])
             .withColumnRenamed("_prod_sum_h", "prod_sum_h")
             .withColumnRenamed("_prod_cnt", "prod_cnt")
    )
    df = df.join(stats, on=prod_key, how="left").fillna({"prod_sum_h": 0.0, "prod_cnt": 0})

    df = df.withColumn(
        "product_helpful_ratio",
        F.when(F.col("prod_cnt") > 0, F.col("prod_sum_h") / F.col("prod_cnt")).otherwise(F.lit(0.0))
    )

    w = Window.partitionBy(prod_key)
    df = (
        df.withColumn("product_review_count", F.count(F.lit(1)).over(w))
          .withColumn("product_avg_rating", F.coalesce(F.avg("star_rating").over(w), F.lit(3.0)))
          .withColumn("product_avg_review_length", F.coalesce(F.avg("review_length").over(w), F.lit(50.0)))
          .withColumn("product_rating_stddev", F.coalesce(F.stddev("star_rating").over(w), F.lit(0.0)))
          .withColumn(
              "meta_review_rating_gap",
              F.when(F.col("product_avg_rating_meta").isNotNull(),
                     F.abs(F.col("product_avg_rating_meta") - F.col("product_avg_rating")))
               .otherwise(F.lit(0.0))
          )
    )

    return df


# ============================
# 4) TEMPORAL & CATEGORY FEATS
# ============================
def add_temporal_features_v2(df):
    """
    Temporal features (NULL-safe):
    - day_of_week, hour_of_day, is_weekend, quarter
    - is_peak_hour (9-17h), is_holiday_season (11-12)
    - days_since_epoch
    """
    df = (
        df.withColumn("day_of_week", F.dayofweek("ts"))
          .withColumn("hour_of_day", F.hour("ts"))
          .withColumn("is_weekend", F.when(F.dayofweek("ts").isin([1, 7]), F.lit(1)).otherwise(F.lit(0)))
          .withColumn("quarter", F.quarter("ts"))
          .withColumn("is_peak_hour", F.when(F.col("hour_of_day").between(9, 17), F.lit(1)).otherwise(F.lit(0)))
          .withColumn("is_holiday_season", F.when(F.month("ts").isin([11, 12]), F.lit(1)).otherwise(F.lit(0)))
          .withColumn("days_since_epoch", F.datediff(F.col("ts"), F.lit("1970-01-01")))
    )
    return df


def add_category_features_v2(df):
    """
    Category-based features (NULL-safe):
    - category_review_count, is_popular_category
    - category_price_percentile, category_rating_percentile
    """
    category_window = Window.partitionBy("category")

    df = df.withColumn("category_review_count", F.count(F.lit(1)).over(category_window))
    df = df.withColumn("is_popular_category", F.when(F.col("category_review_count") > 1000, F.lit(1)).otherwise(F.lit(0)))

    # percent_rank theo price trong category
    df = df.withColumn(
        "category_price_percentile",
        F.when(
            F.col("price").isNotNull(),
            F.percent_rank().over(Window.partitionBy("category").orderBy(F.col("price")))
        ).otherwise(F.lit(0.5))
    )

    df = df.withColumn(
        "category_rating_percentile",
        F.percent_rank().over(Window.partitionBy("category").orderBy(F.col("star_rating")))
    )

    return df


# ============================
# 5) INTERACTION (NO LEAKAGE)
# ============================
def add_interaction_features_v2(df):
    """
    Interaction features với NULL handling.
    ĐÃ LOẠI BỎ helpfulness_x_length (rò rỉ nhãn).
    """
    df = (
        df.withColumn("rating_x_length", F.col("star_rating") * F.col("review_length"))
          .withColumn("user_product_activity", F.col("user_review_count") * F.col("product_review_count"))
    )

    if "sentiment_compound" in df.columns:
        df = df.withColumn("deviation_x_sentiment", F.col("rating_deviation") * F.col("sentiment_compound"))

    df = df.withColumn(
        "price_x_rating",
        F.when(F.col("price").isNotNull(), F.col("price") * F.col("star_rating")).otherwise(F.lit(0.0))
    )

    # user_experience_score vẫn OK vì user_helpful_ratio đã được chống leak
    df = df.withColumn(
        "user_experience_score",
        ( (F.col("user_review_count") / F.lit(10.0)) * F.lit(0.4)
          + F.col("user_consistency") * F.lit(0.3)
          + F.col("user_helpful_ratio") * F.lit(0.3) )
    )

    return df


# ======================================
# 6) CREATE FULL FEATURE SET (WITH MODES)
# ======================================
def create_full_feature_set_v2(
    df,
    include_user_agg=True,
    include_product_agg=True,
    include_temporal=True,
    include_interactions=True,
    include_category=True,
    mode: str = "train",
    user_stats_path: str = None,
    product_stats_path: str = None,
    spark=None
):
    """
    Tạo bộ đặc trưng đầy đủ V2 (NULL-safe + an toàn nhãn).

    Args:
        df: DataFrame đầu vào (phải có cột 'is_helpful' ở TRAIN).
        include_*: bật/tắt từng nhóm features.
        mode: 'train' hoặc 'test' (ảnh hưởng đến cách tính helpful_ratio).
        user_stats_path, product_stats_path: đường dẫn lưu/đọc stats (parquet).
        spark: SparkSession (bắt buộc khi mode='test' để đọc stats).

    Returns:
        DataFrame đã gắn đầy đủ đặc trưng.
    """
    if mode not in ("train", "test"):
        raise ValueError("mode must be 'train' or 'test'")

    print("\n[INFO] Creating feature set V2 (NULL-safe, no label leakage)...")

    # Always basic
    df = add_basic_metadata_features_v2(df)
    print("  ✓ Basic metadata features")

    if include_user_agg:
        df = add_user_aggregate_features_v2(
            df, mode=mode, user_stats_path=user_stats_path, spark=spark
        )
        print("  ✓ User aggregate features (LOO/test-join)")

    if include_product_agg:
        df = add_product_aggregate_features_v2(
            df, mode=mode, product_stats_path=product_stats_path, spark=spark
        )
        print("  ✓ Product aggregate features (LOO/test-join)")

    if include_temporal:
        df = add_temporal_features_v2(df)
        print("  ✓ Temporal features")

    if include_category:
        df = add_category_features_v2(df)
        print("  ✓ Category features")

    if include_interactions:
        df = add_interaction_features_v2(df)
        print("  ✓ Interaction features")

    return df


# =============================================
# 7) SELECT FEATURE SETS (đã loại bỏ leakage col)
# =============================================
def select_feature_columns_v2(df, feature_set="v2"):
    """
    Chọn subset features theo level:
        "baseline", "v1", "v2", "v3", "full"
    ĐÃ loại bỏ 'helpfulness_x_length' để tránh rò rỉ.
    """
    baseline_features = [
        "star_rating",
        "review_length",
        "review_length_log",
    ]

    v1_features = baseline_features + [
        "rating_deviation",
        "is_long_review",
        "user_review_count",
        "product_review_count",
    ]

    v2_features = v1_features + [
        "user_avg_rating",
        "user_helpful_ratio",        # ĐÃ LOO/test-join
        "product_avg_rating",
        "product_helpful_ratio",     # ĐÃ LOO/test-join
        "price",
        "price_log",
        "product_avg_rating_meta",
        "product_total_ratings",
    ]

    v3_features = v2_features + [
        "has_metadata",
        "has_price",
        "has_product_rating",
        "is_expensive",
        "user_consistency",
        "meta_review_rating_gap",
        "category_review_count",
        "is_popular_category",
    ]

    full_features = v3_features + [
        "user_avg_review_length",
        "product_avg_review_length",
        "product_rating_stddev",
        "day_of_week",
        "is_weekend",
        "is_peak_hour",
        "is_holiday_season",
        "quarter",
        "rating_x_length",
        "user_product_activity",
        "price_x_rating",
        "user_experience_score",
        "category_price_percentile",
        "category_rating_percentile",
        # TUYỆT ĐỐI KHÔNG có 'helpfulness_x_length' ở đây
    ]

    feature_map = {
        "baseline": baseline_features,
        "v1": v1_features,
        "v2": v2_features,
        "v3": v3_features,
        "full": full_features,
    }

    selected = feature_map.get(feature_set, v2_features)
    existing = set(df.columns)
    return [f for f in selected if f in existing]


# ===============
# 8) SELF-TESTING
# ===============
if __name__ == "__main__":
    import argparse
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, help="Input HDFS path (parquet). If omitted, runs self-test.")
    ap.add_argument("--output", required=False, help="Output HDFS path for features")
    ap.add_argument("--feature-set", default="v2", choices=["baseline","v1","v2","v3","full"])
    ap.add_argument("--mode", default="train", choices=["train","test"])
    ap.add_argument("--user-stats-path", default="hdfs:///output_v2/stats/user.parquet")
    ap.add_argument("--product-stats-path", default="hdfs:///output_v2/stats/product.parquet")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--coalesce", type=int, default=16)
    args = ap.parse_args()

    spark = SparkSession.builder.appName("MetadataFeaturesV2-NoLeakage").getOrCreate()

    if args.input:
        df = spark.read.parquet(args.input)
        df_feats = create_full_feature_set_v2(
            df, mode=args.mode,
            user_stats_path=args.user_stats_path,
            product_stats_path=args.product_stats_path,
            spark=spark
        )
        cols = select_feature_columns_v2(df_feats, args.feature_set)
        out = df_feats.select(["user_id","product_id"] + cols)

        if args.save and args.output:
            (out.coalesce(args.coalesce)
                .write.mode("overwrite").parquet(args.output))
            print(f"[OK] Wrote features to {args.output}")
        out.show(5, truncate=False)
    else:
        # giữ nguyên self-test cũ nếu không truyền --input
        from datetime import datetime
        test_data = [
            ("u1","p1",5.0,150,10,1,datetime(2023,6,15,14,30),"Electronics",99.99,4.5,100),
            ("u1","p2",4.0,80,3,1,datetime(2023,6,16,9,15),"Electronics",None,None,0),
            ("u2","p1",3.0,200,0,0,datetime(2023,6,17,20,45),None,99.99,4.5,100),
            ("u2","p3",5.0,120,5,1,datetime(2023,6,18,11,0),"Books",19.99,4.0,50),
            ("u3","p1",2.0,50,0,0,datetime(2023,6,19,15,30),"Electronics",99.99,4.5,100),
        ]
        df = spark.createDataFrame(test_data, ["user_id","product_id","star_rating","review_length","helpful_votes",
                                               "is_helpful","ts","category","price","product_avg_rating_meta","product_total_ratings"])
        df_train = create_full_feature_set_v2(df, mode="train", spark=spark)
        df_train.select("user_id","product_id","user_review_count","user_helpful_ratio",
                        "product_review_count","product_helpful_ratio","has_metadata").show(truncate=False)
    spark.stop()
