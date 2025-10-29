# code/features/metadata_features.py
# Day 2+ của Thanh: Đặc trưng metadata và aggregate features

from pyspark.sql import functions as F, Window

def add_basic_metadata_features(df):
    """
    Thêm các đặc trưng metadata cơ bản (đã có từ ETL của Tuấn)
    
    Features đã có trong Parquet:
    - star_rating (float): điểm rating 1.0-5.0
    - review_length (int): số từ trong review
    - year, month (int): từ timestamp
    
    Thêm mới:
    - review_length_log: log(review_length + 1)
    - is_long_review: review_length > 100
    - rating_deviation: |star_rating - 3.0| (độ lệch so với trung bình)
    """
    return (df
            .withColumn("review_length_log", F.log1p(F.col("review_length")))
            .withColumn("is_long_review", (F.col("review_length") > 100).cast("int"))
            .withColumn("rating_deviation", F.abs(F.col("star_rating") - 3.0)))

def add_user_aggregate_features(df):
    """
    Tính các đặc trưng aggregate theo user_id
    
    Features:
    - user_review_count: số lượng reviews của user
    - user_avg_rating: rating trung bình của user
    - user_helpful_ratio: tỷ lệ reviews helpful của user
    - user_avg_review_length: độ dài review trung bình
    """
    # Window cho mỗi user
    user_window = Window.partitionBy("user_id")
    
    return (df
            .withColumn("user_review_count", F.count("*").over(user_window))
            .withColumn("user_avg_rating", F.avg("star_rating").over(user_window))
            .withColumn("user_helpful_ratio", 
                       F.avg(F.col("is_helpful").cast("double")).over(user_window))
            .withColumn("user_avg_review_length", 
                       F.avg("review_length").over(user_window)))

def add_product_aggregate_features(df):
    """
    Tính các đặc trưng aggregate theo product_id
    
    Features:
    - product_review_count: số lượng reviews của sản phẩm
    - product_avg_rating: rating trung bình của sản phẩm
    - product_helpful_ratio: tỷ lệ reviews helpful của sản phẩm
    - product_avg_review_length: độ dài review trung bình
    """
    product_window = Window.partitionBy("product_id")
    
    return (df
            .withColumn("product_review_count", F.count("*").over(product_window))
            .withColumn("product_avg_rating", F.avg("star_rating").over(product_window))
            .withColumn("product_helpful_ratio", 
                       F.avg(F.col("is_helpful").cast("double")).over(product_window))
            .withColumn("product_avg_review_length", 
                       F.avg("review_length").over(product_window)))

def add_temporal_features(df):
    """
    Thêm đặc trưng thời gian (từ cột 'ts' - timestamp)
    
    Features:
    - day_of_week: thứ trong tuần (1=Monday, 7=Sunday)
    - hour_of_day: giờ trong ngày (0-23)
    - is_weekend: cuối tuần hay không
    - quarter: quý trong năm (1-4)
    """
    return (df
            .withColumn("day_of_week", F.dayofweek("ts"))
            .withColumn("hour_of_day", F.hour("ts"))
            .withColumn("is_weekend", F.when(F.dayofweek("ts").isin([1, 7]), 1).otherwise(0))
            .withColumn("quarter", F.quarter("ts")))

def add_interaction_features(df):
    """
    Thêm interaction features (tương tác giữa các đặc trưng)
    
    Features:
    - rating_x_length: star_rating * review_length (reviews dài + rating cao?)
    - user_product_activity: user_review_count * product_review_count
    - deviation_x_sentiment: rating_deviation * sentiment_compound
    """
    return (df
            .withColumn("rating_x_length", 
                       F.col("star_rating") * F.col("review_length"))
            .withColumn("user_product_activity",
                       F.col("user_review_count") * F.col("product_review_count"))
            # Interaction với sentiment (nếu có)
            .withColumn("deviation_x_sentiment",
                       F.when(F.col("sentiment_compound").isNotNull(),
                             F.col("rating_deviation") * F.col("sentiment_compound"))
                        .otherwise(0.0)))

def create_full_feature_set(df, include_user_agg=True, include_product_agg=True, 
                           include_temporal=True, include_interactions=True):
    """
    Tạo bộ đặc trưng đầy đủ cho training
    
    Args:
        df: DataFrame đầu vào (cần có: star_rating, review_length, is_helpful, user_id, product_id, ts)
        include_user_agg: bật/tắt user aggregate features
        include_product_agg: bật/tắt product aggregate features
        include_temporal: bật/tắt temporal features
        include_interactions: bật/tắt interaction features
    
    Returns:
        DataFrame với đầy đủ features
    """
    # Basic metadata (luôn thêm)
    df = add_basic_metadata_features(df)
    
    # Optional features
    if include_user_agg:
        df = add_user_aggregate_features(df)
    
    if include_product_agg:
        df = add_product_aggregate_features(df)
    
    if include_temporal:
        df = add_temporal_features(df)
    
    if include_interactions:
        df = add_interaction_features(df)
    
    return df

def select_feature_columns(df, feature_set="baseline"):
    """
    Chọn subset các cột features theo mức độ phức tạp
    
    Args:
        df: DataFrame với đầy đủ features
        feature_set: "baseline", "v1", "v2", "full"
    
    Returns:
        List tên các cột features
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
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neg"
    ]
    
    full_features = v2_features + [
        "user_avg_review_length",
        "product_avg_review_length",
        "day_of_week",
        "is_weekend",
        "quarter",
        "rating_x_length",
        "user_product_activity",
        "deviation_x_sentiment"
    ]
    
    feature_map = {
        "baseline": baseline_features,
        "v1": v1_features,
        "v2": v2_features,
        "full": full_features
    }
    
    selected = feature_map.get(feature_set, baseline_features)
    
    # Filter only existing columns
    existing = set(df.columns)
    return [f for f in selected if f in existing]

if __name__ == "__main__":
    # Test/Demo
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("MetadataFeatures-Test").getOrCreate()
    
    # Test data
    from datetime import datetime
    test_data = [
        ("u1", "p1", 5.0, 150, 10, 1, datetime(2023, 6, 15, 14, 30)),
        ("u1", "p2", 4.0, 80, 3, 1, datetime(2023, 6, 16, 9, 15)),
        ("u2", "p1", 3.0, 200, 0, 0, datetime(2023, 6, 17, 20, 45)),
        ("u2", "p2", 5.0, 120, 5, 1, datetime(2023, 6, 18, 11, 0)),
        ("u3", "p1", 2.0, 50, 0, 0, datetime(2023, 6, 19, 15, 30))
    ]
    
    df = spark.createDataFrame(test_data, 
                               ["user_id", "product_id", "star_rating", "review_length", 
                                "helpful_votes", "is_helpful", "ts"])
    
    print("\n=== Testing Metadata Features ===")
    
    # Add all features
    df_full = create_full_feature_set(df)
    
    print("\nBasic Metadata Features:")
    df_full.select("star_rating", "review_length", "review_length_log", 
                   "is_long_review", "rating_deviation").show()
    
    print("\nUser Aggregate Features:")
    df_full.select("user_id", "user_review_count", "user_avg_rating", 
                   "user_helpful_ratio").show()
    
    print("\nProduct Aggregate Features:")
    df_full.select("product_id", "product_review_count", "product_avg_rating", 
                   "product_helpful_ratio").show()
    
    print("\nTemporal Features:")
    df_full.select("ts", "day_of_week", "hour_of_day", "is_weekend", "quarter").show()
    
    print("\nAvailable feature sets:")
    for fs in ["baseline", "v1", "v2", "full"]:
        features = select_feature_columns(df_full, fs)
        print(f"  {fs}: {len(features)} features")
    
    spark.stop()
