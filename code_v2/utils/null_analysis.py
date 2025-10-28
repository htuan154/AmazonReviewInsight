# code_v2/utils/null_analysis.py
# Công cụ phân tích NULL trong dataset
#
# Features:
# - Đếm NULL per column
# - Phân tích pattern NULL (correlated nulls)
# - Đề xuất chiến lược imputation
# - So sánh before/after imputation

from pyspark.sql import DataFrame, functions as F

def analyze_null_patterns(df: DataFrame, output_path: str = None):
    """
    Phân tích pattern NULL trong DataFrame
    
    Args:
        df: Spark DataFrame
        output_path: Nếu cung cấp, lưu kết quả ra CSV
    
    Returns:
        dict với thống kê NULL
    """
    print("\n" + "="*80)
    print("NULL PATTERN ANALYSIS")
    print("="*80)
    
    total_rows = df.count()
    print(f"\nTotal rows: {total_rows:,}\n")
    
    # 1. NULL count per column
    print("--- NULL Count per Column ---")
    null_counts = []
    
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        
        null_counts.append({
            "column": col_name,
            "null_count": null_count,
            "null_percentage": null_pct,
            "non_null_count": total_rows - null_count
        })
        
        if null_count > 0:
            print(f"  {col_name:30s}: {null_count:10,} ({null_pct:6.2f}%)")
    
    # 2. Columns with no NULLs
    print("\n--- Columns with NO NULLs ---")
    no_null_cols = [x["column"] for x in null_counts if x["null_count"] == 0]
    print(f"  Count: {len(no_null_cols)}")
    print(f"  Columns: {', '.join(no_null_cols[:10])}{'...' if len(no_null_cols) > 10 else ''}")
    
    # 3. Columns with high NULL rate (>50%)
    print("\n--- Columns with HIGH NULL (>50%) ---")
    high_null_cols = [x for x in null_counts if x["null_percentage"] > 50]
    if high_null_cols:
        for x in high_null_cols:
            print(f"  {x['column']:30s}: {x['null_percentage']:6.2f}%")
    else:
        print("  None")
    
    # 4. Correlated NULL analysis (sample: price vs rating)
    print("\n--- Correlated NULL Patterns ---")
    
    # Check if key columns exist
    key_cols = ["price", "product_avg_rating_meta", "product_total_ratings", "category"]
    existing_key_cols = [c for c in key_cols if c in df.columns]
    
    if len(existing_key_cols) >= 2:
        # Pattern: price NULL và rating NULL cùng lúc
        if "price" in df.columns and "product_avg_rating_meta" in df.columns:
            both_null = df.filter(
                F.col("price").isNull() & F.col("product_avg_rating_meta").isNull()
            ).count()
            both_null_pct = (both_null / total_rows * 100) if total_rows > 0 else 0
            print(f"  Both price & rating NULL: {both_null:,} ({both_null_pct:.2f}%)")
        
        # Pattern: chỉ price NULL
        if "price" in df.columns and "product_avg_rating_meta" in df.columns:
            only_price_null = df.filter(
                F.col("price").isNull() & F.col("product_avg_rating_meta").isNotNull()
            ).count()
            only_price_pct = (only_price_null / total_rows * 100) if total_rows > 0 else 0
            print(f"  Only price NULL: {only_price_null:,} ({only_price_pct:.2f}%)")
        
        # Pattern: chỉ rating NULL
        if "price" in df.columns and "product_avg_rating_meta" in df.columns:
            only_rating_null = df.filter(
                F.col("price").isNotNull() & F.col("product_avg_rating_meta").isNull()
            ).count()
            only_rating_pct = (only_rating_null / total_rows * 100) if total_rows > 0 else 0
            print(f"  Only rating NULL: {only_rating_null:,} ({only_rating_pct:.2f}%)")
    
    # 5. NULL per category (if exists)
    if "category" in df.columns:
        print("\n--- NULL Rate per Category (price) ---")
        if "price" in df.columns:
            cat_null = (
                df.groupBy("category")
                .agg(
                    F.count("*").alias("total"),
                    F.sum(F.when(F.col("price").isNull(), 1).otherwise(0)).alias("null_count")
                )
                .withColumn("null_rate", F.col("null_count") / F.col("total") * 100)
                .orderBy(F.desc("null_rate"))
                .limit(10)
            )
            cat_null.show(truncate=False)
    
    # 6. Summary
    print("\n--- Summary ---")
    total_nulls = sum(x["null_count"] for x in null_counts)
    total_cells = total_rows * len(df.columns)
    overall_null_rate = (total_nulls / total_cells * 100) if total_cells > 0 else 0
    
    print(f"  Total NULL values: {total_nulls:,}")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Overall NULL rate: {overall_null_rate:.2f}%")
    
    # Save to CSV if requested
    if output_path:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        
        null_df = spark.createDataFrame(null_counts)
        null_df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
        print(f"\n✓ NULL analysis saved to: {output_path}")
    
    print("="*80 + "\n")
    
    return {
        "total_rows": total_rows,
        "null_counts": null_counts,
        "overall_null_rate": overall_null_rate
    }

def suggest_imputation_strategy(df: DataFrame, column: str):
    """
    Đề xuất chiến lược imputation cho một cột
    
    Args:
        df: Spark DataFrame
        column: Tên cột cần phân tích
    
    Returns:
        dict với chiến lược đề xuất
    """
    print(f"\n--- Imputation Strategy for '{column}' ---")
    
    # Check if column exists
    if column not in df.columns:
        print(f"  ERROR: Column '{column}' not found")
        return None
    
    # Get column type
    col_type = df.schema[column].dataType.typeName()
    print(f"  Type: {col_type}")
    
    # Count NULLs
    null_count = df.filter(F.col(column).isNull()).count()
    total = df.count()
    null_rate = (null_count / total * 100) if total > 0 else 0
    
    print(f"  NULL: {null_count:,} ({null_rate:.2f}%)")
    
    if null_count == 0:
        print("  ✓ No NULLs - no imputation needed")
        return {"strategy": "none", "reason": "no nulls"}
    
    # Strategy based on type and null rate
    strategy = {}
    
    if col_type in ["double", "float", "integer", "long"]:
        # Numeric column
        stats = df.select(
            F.mean(column).alias("mean"),
            F.expr(f"percentile_approx({column}, 0.5)").alias("median"),
            F.min(column).alias("min"),
            F.max(column).alias("max"),
            F.stddev(column).alias("stddev")
        ).first()
        
        print(f"  Mean: {stats['mean']:.2f}" if stats['mean'] else "  Mean: NULL")
        print(f"  Median: {stats['median']:.2f}" if stats['median'] else "  Median: NULL")
        print(f"  Min: {stats['min']:.2f}" if stats['min'] else "  Min: NULL")
        print(f"  Max: {stats['max']:.2f}" if stats['max'] else "  Max: NULL")
        
        if null_rate < 5:
            strategy = {
                "strategy": "drop",
                "reason": f"Low NULL rate ({null_rate:.2f}%), safe to drop"
            }
        elif stats['stddev'] and stats['stddev'] / stats['mean'] > 1.0:
            strategy = {
                "strategy": "median",
                "value": stats['median'],
                "reason": "High variance, use median for robustness"
            }
        else:
            strategy = {
                "strategy": "mean",
                "value": stats['mean'],
                "reason": "Low variance, use mean"
            }
    
    elif col_type == "string":
        # Categorical column
        mode_row = (
            df.groupBy(column)
            .count()
            .orderBy(F.desc("count"))
            .first()
        )
        
        if mode_row:
            mode_value = mode_row[column]
            mode_count = mode_row["count"]
            mode_pct = (mode_count / total * 100) if total > 0 else 0
            
            print(f"  Mode: '{mode_value}' ({mode_pct:.2f}%)")
            
            if mode_pct > 50:
                strategy = {
                    "strategy": "mode",
                    "value": mode_value,
                    "reason": f"Dominant mode ({mode_pct:.2f}%)"
                }
            else:
                strategy = {
                    "strategy": "constant",
                    "value": "Unknown",
                    "reason": "No dominant mode, use 'Unknown'"
                }
    
    else:
        strategy = {
            "strategy": "constant",
            "value": None,
            "reason": f"Unsupported type: {col_type}"
        }
    
    print(f"\n  Recommended: {strategy['strategy']}")
    print(f"  Reason: {strategy['reason']}")
    if "value" in strategy and strategy["value"] is not None:
        print(f"  Value: {strategy['value']}")
    
    return strategy

def compare_imputation_impact(df_before: DataFrame, df_after: DataFrame, 
                              columns: list, label_col: str = "is_helpful"):
    """
    So sánh impact của imputation trên distribution và model performance
    
    Args:
        df_before: DataFrame trước khi impute
        df_after: DataFrame sau khi impute
        columns: Danh sách cột đã impute
        label_col: Cột label để check class distribution
    
    Returns:
        dict với so sánh
    """
    print("\n" + "="*80)
    print("IMPUTATION IMPACT ANALYSIS")
    print("="*80)
    
    # 1. Row count comparison
    rows_before = df_before.count()
    rows_after = df_after.count()
    
    print(f"\nRows before: {rows_before:,}")
    print(f"Rows after: {rows_after:,}")
    print(f"Difference: {rows_after - rows_before:,} ({(rows_after/rows_before - 1)*100:+.2f}%)")
    
    # 2. NULL count comparison per column
    print("\n--- NULL Count Comparison ---")
    print(f"{'Column':<30} {'Before':<15} {'After':<15} {'Reduction':<15}")
    print("-"*75)
    
    for col in columns:
        if col not in df_before.columns or col not in df_after.columns:
            continue
        
        null_before = df_before.filter(F.col(col).isNull()).count()
        null_after = df_after.filter(F.col(col).isNull()).count()
        reduction = null_before - null_after
        
        print(f"{col:<30} {null_before:<15,} {null_after:<15,} {reduction:<15,}")
    
    # 3. Class distribution comparison
    if label_col in df_before.columns and label_col in df_after.columns:
        print("\n--- Class Distribution Comparison ---")
        
        dist_before = df_before.groupBy(label_col).count().orderBy(label_col).collect()
        dist_after = df_after.groupBy(label_col).count().orderBy(label_col).collect()
        
        print("Before imputation:")
        for row in dist_before:
            label = row[label_col]
            count = row["count"]
            pct = (count / rows_before * 100) if rows_before > 0 else 0
            print(f"  Class {label}: {count:,} ({pct:.2f}%)")
        
        print("\nAfter imputation:")
        for row in dist_after:
            label = row[label_col]
            count = row["count"]
            pct = (count / rows_after * 100) if rows_after > 0 else 0
            print(f"  Class {label}: {count:,} ({pct:.2f}%)")
    
    # 4. Statistical comparison for numeric columns
    print("\n--- Statistical Distribution Comparison ---")
    numeric_cols = [c for c in columns if c in df_before.columns and 
                   df_before.schema[c].dataType.typeName() in ["double", "float", "integer", "long"]]
    
    for col in numeric_cols[:3]:  # Limit to first 3 for brevity
        print(f"\n{col}:")
        
        stats_before = df_before.select(
            F.mean(col).alias("mean"),
            F.stddev(col).alias("stddev"),
            F.expr(f"percentile_approx({col}, 0.5)").alias("median")
        ).first()
        
        stats_after = df_after.select(
            F.mean(col).alias("mean"),
            F.stddev(col).alias("stddev"),
            F.expr(f"percentile_approx({col}, 0.5)").alias("median")
        ).first()
        
        print(f"  Mean: {stats_before['mean']:.2f} → {stats_after['mean']:.2f}")
        print(f"  Std: {stats_before['stddev']:.2f} → {stats_after['stddev']:.2f}")
        print(f"  Median: {stats_before['median']:.2f} → {stats_after['median']:.2f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("NullAnalysis-Test").getOrCreate()
    
    # Test data with NULLs
    test_data = [
        ("r1", "u1", "p1", 5.0, 100, 1, "Electronics", 99.99, 4.5, 100),
        ("r2", "u1", "p2", 4.0, 80, 1, "Electronics", None, None, 0),
        ("r3", "u2", "p1", 3.0, 200, 0, None, 99.99, 4.5, 100),
        ("r4", "u2", "p3", 5.0, 120, 1, "Books", 19.99, None, 50),
        ("r5", "u3", "p4", 2.0, 50, 0, "Electronics", None, 3.0, 10),
    ]
    
    df = spark.createDataFrame(
        test_data,
        ["review_id", "user_id", "product_id", "star_rating", "review_length", 
         "is_helpful", "category", "price", "product_avg_rating_meta", "product_total_ratings"]
    )
    
    print("\n=== Testing NULL Analysis Tools ===")
    
    # 1. Analyze NULL patterns
    analyze_null_patterns(df)
    
    # 2. Suggest imputation for specific columns
    suggest_imputation_strategy(df, "price")
    suggest_imputation_strategy(df, "product_avg_rating_meta")
    suggest_imputation_strategy(df, "category")
    
    # 3. Create imputed version
    df_imputed = df.fillna({
        "price": 50.0,
        "product_avg_rating_meta": 3.0,
        "product_total_ratings": 0,
        "category": "Unknown"
    })
    
    # 4. Compare impact
    compare_imputation_impact(
        df, df_imputed,
        ["price", "product_avg_rating_meta", "product_total_ratings", "category"]
    )
    
    spark.stop()
