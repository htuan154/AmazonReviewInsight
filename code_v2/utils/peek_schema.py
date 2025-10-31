
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Parquet path (HDFS or local)")
    ap.add_argument("--limit", type=int, default=5, help="Rows to preview")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("peek_schema").getOrCreate()
    try:
        df = spark.read.parquet(args.path)
    except Exception as e:
        print("!! ERROR reading parquet:", e)
        spark.stop()
        raise

    print("=== SCHEMA ===")
    df.printSchema()
    print("=== N ROWS (approx df.count()) ===")
    try:
        print(df.count())
    except Exception as e:
        print("count() failed:", e)

    print("=== HEAD ===")
    cols = df.columns[:25]
    if not cols:
        print("(no columns)")
    else:
        df.select([col(c) for c in cols]).show(args.limit, truncate=80)

    spark.stop()

if __name__ == "__main__":
    main()
