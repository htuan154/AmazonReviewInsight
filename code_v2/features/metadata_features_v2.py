# code_v2/features/metadata_features_v2.py
# Version 2: Xử lý NULL-safe cho các đặc trưng metadata
#
# Cải tiến so với v1:
# - Tất cả các aggregate functions đều handle NULL
# - Thêm features về "missing metadata" (indicator features)
# - Safe window operations với fillna
# - Thêm features về "data quality"

from pyspark.sql import functions as F, Window

def add_basic_metadata_features_v2(df):
    """
    Thêm các đặc trưng metadata cơ bản với NULL handling
    
    Features:
    - review_length_log: log(review_length + 1)
    - is_long_review: review_length > 100
    - rating_deviation: |star_rating - 3.0|
    - has_metadata: indicator nếu có đầy đủ metadata
    - price_log: log(price + 1) để handle giá 0
    - is_expensive: price > median_price (per category)
    """
    # Basic features (giống v1)
    df = (df
          .withColumn("review_length_log", F.log1p(F.col("review_length")))
          .withColumn("is_long_review", (F.col("review_length") > 100).cast("int"))
          .withColumn("rating_deviation", F.abs(F.col("star_rating") - 3.0)))
    
    # NEW: Data quality indicators
    df = df.withColumn(
        "has_price",
        F.when(F.col("price").isNull() | (F.col("price") == 0), 0).otherwise(1)
    )
    
    df = df.withColumn(
        "has_product_rating",
        F.when(
            F.col("product_avg_rating_meta").isNull() | 
            (F.col("product_total_ratings").isNull()) |
            (F.col("product_total_ratings") == 0),
            0
        ).otherwise(1)
    )
    
    df = df.withColumn(
        "has_metadata",
        (F.col("has_price") & F.col("has_product_rating")).cast("int")
    )
    
    # Price features (with NULL safety)
    df = df.withColumn(
        "price_log",
        F.when(F.col("price").isNotNull() & (F.col("price") > 0),
               F.log1p(F.col("price")))
        .otherwise(0.0)
    )
    
    # Category median price (for expensive indicator)
    category_window = Window.partitionBy("category")
    df = df.withColumn(
        "category_median_price",
        F.expr("percentile_approx(price, 0.5)").over(category_window)
    )
    
    df = df.withColumn(
        "is_expensive",
        F.when(
            (F.col("price").isNotNull()) & 
            (F.col("category_median_price").isNotNull()) &
            (F.col("price") > F.col("category_median_price")),
            1
        ).otherwise(0)
    )
    
    return df

def add_user_aggregate_features_v2(df):
    """
    Tính aggregate features theo user với NULL handling
    
    Features:
    - user_review_count: số reviews của user
    - user_avg_rating: rating trung bình (NULL-safe)
    - user_helpful_ratio: tỷ lệ helpful
    - user_avg_review_length: độ dài trung bình
    - user_consistency: std dev của rating (thấp = consistent)
    """
    user_window = Window.partitionBy("user_id")
    
    df = (df
          .withColumn("user_review_count", F.count("*").over(user_window))
          .withColumn("user_avg_rating", 
                     F.coalesce(F.avg("star_rating").over(user_window), F.lit(3.0)))
          .withColumn("user_helpful_ratio", 
                     F.coalesce(F.avg(F.col("is_helpful").cast("double")).over(user_window), F.lit(0.0)))
          .withColumn("user_avg_review_length", 
                     F.coalesce(F.avg("review_length").over(user_window), F.lit(50.0)))
          .withColumn("user_rating_stddev",
                     F.coalesce(F.stddev("star_rating").over(user_window), F.lit(0.0))))
    
    # User consistency (inverted std dev, normalized 0-1)
    df = df.withColumn(
        "user_consistency",
        F.when(F.col("user_rating_stddev") == 0, 1.0)
        .otherwise(1.0 / (1.0 + F.col("user_rating_stddev")))
    )
    
    return df

def add_product_aggregate_features_v2(df):
    """
    Tính aggregate features theo product với NULL handling
    
    Features:
    - product_review_count: số reviews của sản phẩm
    - product_avg_rating: rating trung bình (from reviews, không phải meta)
    - product_helpful_ratio: tỷ lệ helpful
    - product_avg_review_length: độ dài trung bình
    - product_rating_stddev: độ phân tán rating
    - meta_review_rating_gap: chênh lệch giữa meta rating vs review rating
    """
    product_window = Window.partitionBy("product_id")
    
    df = (df
          .withColumn("product_review_count", F.count("*").over(product_window))
          .withColumn("product_avg_rating", 
                     F.coalesce(F.avg("star_rating").over(product_window), F.lit(3.0)))
          .withColumn("product_helpful_ratio", 
                     F.coalesce(F.avg(F.col("is_helpful").cast("double")).over(product_window), F.lit(0.0)))
          .withColumn("product_avg_review_length", 
                     F.coalesce(F.avg("review_length").over(product_window), F.lit(50.0)))
          .withColumn("product_rating_stddev",
                     F.coalesce(F.stddev("star_rating").over(product_window), F.lit(0.0))))
    
    # NEW: Gap between metadata rating vs actual review rating
    df = df.withColumn(
        "meta_review_rating_gap",
        F.when(
            F.col("product_avg_rating_meta").isNotNull(),
            F.abs(F.col("product_avg_rating_meta") - F.col("product_avg_rating"))
        ).otherwise(0.0)
    )
    
    return df

def add_temporal_features_v2(df):
    """
    Thêm temporal features với NULL safety
    
    Features:
    - day_of_week, hour_of_day, is_weekend, quarter (giống v1)
    - is_peak_hour: giờ cao điểm (9-17h)
    - is_holiday_season: tháng 11-12 (Black Friday, Christmas)
    - days_since_epoch: số ngày kể từ 1970-01-01 (for temporal encoding)
    """
    df = (df
          .withColumn("day_of_week", F.dayofweek("ts"))
          .withColumn("hour_of_day", F.hour("ts"))
          .withColumn("is_weekend", F.when(F.dayofweek("ts").isin([1, 7]), 1).otherwise(0))
          .withColumn("quarter", F.quarter("ts")))
    
    # NEW features
    df = df.withColumn(
        "is_peak_hour",
        F.when(F.col("hour_of_day").between(9, 17), 1).otherwise(0)
    )
    
    df = df.withColumn(
        "is_holiday_season",
        F.when(F.month("ts").isin([11, 12]), 1).otherwise(0)
    )
    
    df = df.withColumn(
        "days_since_epoch",
        F.datediff(F.col("ts"), F.lit("1970-01-01"))
    )
    
    return df

def add_interaction_features_v2(df):
    """
    Interaction features với NULL handling
    
    Features:
    - rating_x_length: star_rating * review_length
    - user_product_activity: user_review_count * product_review_count
    - deviation_x_sentiment: rating_deviation * sentiment (nếu có)
    - price_x_rating: price * star_rating
    - helpfulness_x_length: is_helpful * review_length
    - user_experience_score: composite score
    """
    df = (df
          .withColumn("rating_x_length", 
                     F.col("star_rating") * F.col("review_length"))
          .withColumn("user_product_activity",
                     F.col("user_review_count") * F.col("product_review_count")))
    
    # Interaction với sentiment (nếu có cột)
    if "sentiment_compound" in df.columns:
        df = df.withColumn(
            "deviation_x_sentiment",
            F.col("rating_deviation") * F.col("sentiment_compound")
        )
    
    # Price interaction
    df = df.withColumn(
        "price_x_rating",
        F.when(F.col("price").isNotNull(),
               F.col("price") * F.col("star_rating"))
        .otherwise(0.0)
    )
    
    # Helpfulness interaction
    df = df.withColumn(
        "helpfulness_x_length",
        F.col("is_helpful") * F.col("review_length")
    )
    
    # NEW: User experience composite score
    # High score = experienced user (many reviews, consistent, helpful)
    df = df.withColumn(
        "user_experience_score",
        (
            (F.col("user_review_count") / 10.0) * 0.4 +  # Activity weight
            F.col("user_consistency") * 0.3 +              # Consistency weight
            F.col("user_helpful_ratio") * 0.3              # Helpfulness weight
        )
    )
    
    return df

def add_category_features_v2(df):
    """
    Features dựa trên category với NULL handling
    
    Features:
    - category_price_percentile: percentile của price trong category
    - category_rating_percentile: percentile của rating trong category
    - is_popular_category: category có > 1000 reviews
    """
    category_window = Window.partitionBy("category")
    
    # Count reviews per category
    df = df.withColumn(
        "category_review_count",
        F.count("*").over(category_window)
    )
    
    df = df.withColumn(
        "is_popular_category",
        F.when(F.col("category_review_count") > 1000, 1).otherwise(0)
    )
    
    # Price percentile in category (0-1)
    df = df.withColumn(
        "category_price_percentile",
        F.when(
            F.col("price").isNotNull(),
            F.percent_rank().over(
                Window.partitionBy("category").orderBy("price")
            )
        ).otherwise(0.5)  # Median if NULL
    )
    
    # Rating percentile in category
    df = df.withColumn(
        "category_rating_percentile",
        F.percent_rank().over(
            Window.partitionBy("category").orderBy("star_rating")
        )
    )
    
    return df

def create_full_feature_set_v2(df, include_user_agg=True, include_product_agg=True,
                                include_temporal=True, include_interactions=True,
                                include_category=True):
    """
    Tạo bộ đặc trưng đầy đủ V2 với NULL handling
    
    Args:
        df: DataFrame đầu vào
        include_*: flags để bật/tắt từng nhóm features
    
    Returns:
        DataFrame với đầy đủ features (NULL-safe)
    """
    print("\n[INFO] Creating feature set V2 (NULL-safe)...")
    
    # Always add basic features
    df = add_basic_metadata_features_v2(df)
    print("  ✓ Basic metadata features")
    
    if include_user_agg:
        df = add_user_aggregate_features_v2(df)
        print("  ✓ User aggregate features")
    
    if include_product_agg:
        df = add_product_aggregate_features_v2(df)
        print("  ✓ Product aggregate features")
    
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

def select_feature_columns_v2(df, feature_set="v2"):
    """
    Chọn subset features theo level
    
    Args:
        feature_set: "baseline", "v1", "v2", "v3", "full"
    """
    baseline_features = [
        "star_rating",
        "review_length",
        "review_length_log"
    ]
    
    v1_features = baseline_features + [
        "rating_deviation",
        "is_long_review",
        "user_review_count",
        "product_review_count"
    ]
    
    v2_features = v1_features + [
        "user_avg_rating",
        "user_helpful_ratio",
        "product_avg_rating",
        "product_helpful_ratio",
        "price",
        "price_log",
        "product_avg_rating_meta",
        "product_total_ratings"
    ]
    
    v3_features = v2_features + [
        "has_metadata",
        "has_price",
        "has_product_rating",
        "is_expensive",
        "user_consistency",
        "meta_review_rating_gap",
        "category_review_count",
        "is_popular_category"
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
        "category_rating_percentile"
    ]
    
    feature_map = {
        "baseline": baseline_features,
        "v1": v1_features,
        "v2": v2_features,
        "v3": v3_features,
        "full": full_features
    }
    
    selected = feature_map.get(feature_set, v2_features)
    
    # Filter only existing columns
    existing = set(df.columns)
    return [f for f in selected if f in existing]

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import datetime
    
    spark = SparkSession.builder.appName("MetadataFeaturesV2-Test").getOrCreate()
    
    # Test data with NULLs
    test_data = [
        ("u1", "p1", 5.0, 150, 10, 1, datetime(2023, 6, 15, 14, 30), "Electronics", 99.99, 4.5, 100),
        ("u1", "p2", 4.0, 80, 3, 1, datetime(2023, 6, 16, 9, 15), "Electronics", None, None, 0),  # NULL price/rating
        ("u2", "p1", 3.0, 200, 0, 0, datetime(2023, 6, 17, 20, 45), None, 99.99, 4.5, 100),  # NULL category
        ("u2", "p3", 5.0, 120, 5, 1, datetime(2023, 6, 18, 11, 0), "Books", 19.99, 4.0, 50),
        ("u3", "p1", 2.0, 50, 0, 0, datetime(2023, 6, 19, 15, 30), "Electronics", 99.99, 4.5, 100)
    ]
    
    df = spark.createDataFrame(
        test_data,
        ["user_id", "product_id", "star_rating", "review_length", "helpful_votes", 
         "is_helpful", "ts", "category", "price", "product_avg_rating_meta", "product_total_ratings"]
    )
    
    print("\n=== Testing Metadata Features V2 (NULL Handling) ===")
    
    # Add all features
    df_full = create_full_feature_set_v2(df)
    
    print("\n--- Basic Features + NULL Indicators ---")
    df_full.select(
        "star_rating", "review_length", "price", 
        "has_price", "has_product_rating", "has_metadata"
    ).show()
    
    print("\n--- User Features (NULL-safe) ---")
    df_full.select(
        "user_id", "user_review_count", "user_avg_rating",
        "user_consistency", "user_experience_score"
    ).show()
    
    print("\n--- Product Features (NULL-safe) ---")
    df_full.select(
        "product_id", "product_review_count", "product_avg_rating",
        "meta_review_rating_gap"
    ).show()
    
    print("\n--- Category Features ---")
    df_full.select(
        "category", "category_review_count", "is_popular_category",
        "category_price_percentile"
    ).show()
    
    print("\nAvailable feature sets:")
    for fs in ["baseline", "v1", "v2", "v3", "full"]:
        features = select_feature_columns_v2(df_full, fs)
        print(f"  {fs}: {len(features)} features")
    
    spark.stop()
