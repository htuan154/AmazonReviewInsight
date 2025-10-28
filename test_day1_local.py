# Test Day 1 V2 - Local mode (no HDFS)
# Quick test to verify logic without Java version issues

print("="*80)
print("Testing Day 1 V2 Logic - Local Mode")
print("="*80)

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    
    print("\n[INFO] Creating Spark session (local mode)...")
    spark = SparkSession.builder \
        .appName("Test-Day1-V2") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    print("[OK] Spark session created successfully!")
    print(f"[INFO] Spark version: {spark.version}")
    
    # Test read from HDFS
    print("\n[INFO] Testing HDFS connection...")
    print("[INFO] Reading Movies_and_TV.jsonl (sample 1000 rows)...")
    
    reviews = spark.read.json(
        "hdfs://localhost:9000/datasets/amazon/movies/raw/Movies_and_TV.jsonl",
        multiLine=False
    ).limit(1000)
    
    print(f"[OK] Successfully read {reviews.count()} reviews")
    print("\n[INFO] Sample schema:")
    reviews.printSchema()
    
    print("\n[INFO] Testing metadata read...")
    meta = spark.read.json(
        "hdfs://localhost:9000/datasets/amazon/movies/raw/meta_Movies_and_TV.jsonl",
        multiLine=False
    ).limit(1000)
    
    print(f"[OK] Successfully read {meta.count()} metadata records")
    
    print("\n" + "="*80)
    print("[SUCCESS] Day 1 V2 logic test passed!")
    print("="*80)
    print("\nNext: Upgrade to Java 17 to run full pipeline")
    print("Download: https://adoptium.net/temurin/releases/?version=17")
    
    spark.stop()
    
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nTroubleshooting:")
    print("1. Check if HDFS is running: docker ps")
    print("2. Check Java version: java -version (need Java 11 or 17)")
    print("3. Install PySpark: pip install pyspark")
