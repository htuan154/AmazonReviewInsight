# code/features/day2_thanh_tfidf_test.py
# Day 2 - Thanh: Test TF-IDF voi Spark ML
"""
Usage:
    spark-submit code/features/day2_thanh_tfidf_test.py \
        --data hdfs://localhost:9000/datasets/amazon/movies/parquet/features_v1 \
        --sample 50000 \
        --numFeatures 10000
"""

import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml import Pipeline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--numFeatures", type=int, default=10000)
    return ap.parse_args()

def main():
    args = parse_args()
    
    spark = SparkSession.builder\
        .appName("Day2-Thanh-TFIDF-Test")\
        .config("spark.driver.memory", "4g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n{'='*60}")
    print("DAY 2 - THANH: TF-IDF TESTING")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"Loading data from: {args.data}")
    df = spark.read.parquet(args.data)
    
    total = df.count()
    print(f"Total records: {total:,}")
    
    # Sample
    if args.sample < total:
        print(f"Sampling {args.sample:,} records...")
        df = df.sample(fraction=args.sample/total, seed=42)
    
    df = df.select("review_id", "clean_text", "star_rating", "review_length", 
                   "sentiment_compound", "is_helpful").na.fill({
        "clean_text": "",
        "star_rating": 0.0,
        "review_length": 0,
        "sentiment_compound": 0.0
    })
    
    # ===== TF-IDF Pipeline =====
    print(f"\n{'='*60}")
    print("TF-IDF PIPELINE")
    print(f"{'='*60}\n")
    
    print(f"Building pipeline with numFeatures={args.numFeatures}...")
    
    # Tokenize
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="tokens")
    
    # Remove stopwords
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    
    # TF
    hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="tf", 
                          numFeatures=args.numFeatures)
    
    # IDF
    idf = IDF(inputCol="tf", outputCol="tfidf", minDocFreq=5)
    
    # Assemble with metadata features
    assembler = VectorAssembler(
        inputCols=["tfidf", "star_rating", "review_length", "sentiment_compound"],
        outputCol="features"
    )
    
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler])
    
    # Fit pipeline
    print("Fitting pipeline...")
    model = pipeline.fit(df)
    
    # Transform
    print("Transforming data...")
    result = model.transform(df)
    
    # Show feature vector info
    print(f"\nFeature vector analysis:")
    feature_sample = result.select("features").first()["features"]
    print(f"  Feature vector size: {feature_sample.size}")
    print(f"  Non-zero elements: {len(feature_sample.indices)}")
    print(f"  Sparsity: {1 - len(feature_sample.indices)/feature_sample.size:.4f}")
    
    # Check by class
    print(f"\n{'='*60}")
    print("FEATURE STATISTICS BY CLASS")
    print(f"{'='*60}\n")
    
    print("Average token count by helpfulness:")
    result.withColumn("token_count", F.size("filtered_tokens"))\
        .groupBy("is_helpful")\
        .agg(F.avg("token_count").alias("avg_tokens"),
             F.stddev("token_count").alias("std_tokens"))\
        .show()
    
    print("Average TF-IDF vector norm by helpfulness:")
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.types import DoubleType
    
    @F.udf(returnType=DoubleType())
    def vector_norm(v):
        if v is None:
            return 0.0
        return float(Vectors.norm(v, 2))
    
    result.withColumn("tfidf_norm", vector_norm("tfidf"))\
        .groupBy("is_helpful")\
        .agg(F.avg("tfidf_norm").alias("avg_norm"),
             F.stddev("tfidf_norm").alias("std_norm"))\
        .show()
    
    print(f"\n{'='*60}")
    print("DAY 2 - THANH TF-IDF TEST COMPLETED")
    print(f"{'='*60}")
    print("\nKey findings:")
    print("  1. TF-IDF pipeline successfully built")
    print(f"  2. Feature vector size: {feature_sample.size}")
    print(f"  3. Sparsity: {1 - len(feature_sample.indices)/feature_sample.size:.2%}")
    print("\nNext steps:")
    print("  - Ready for Day 3: Baseline Logistic Regression")
    print("  - Tuan will integrate this into full pipeline")
    
    spark.stop()

if __name__ == "__main__":
    main()
