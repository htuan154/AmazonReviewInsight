# code/models/predict_spark_logreg.py
# Usage:
# spark-submit --master yarn --deploy-mode client \
#   code/models/predict_spark_logreg.py \
#   --data  hdfs:///datasets/amazon/movies/parquet/reviews \
#   --model hdfs:///datasets/amazon/movies/models/logreg_baseline \
#   --out   hdfs:///datasets/amazon/movies/output/submission.csv

import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.pipeline import PipelineModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)  # file csv đầu ra trên HDFS
    return ap.parse_args()

def main():
    args = parse_args()
    spark = SparkSession.builder.appName("Helpful-Predict-LogReg").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Load data & model
    df = (spark.read.parquet(args.data)
          .select("review_id","review_text","star_rating","review_length")
          .na.fill({"review_text":"", "star_rating":0.0, "review_length":0})
         )
    model = PipelineModel.load(args.model)

    # Predict
    pred = model.transform(df).select(
        "review_id",
        (F.col("probability")[1]).alias("probability_helpful")
    )

    # Write single CSV (coalesce để ra 1 file)
    # Nếu 'args.out' là đường dẫn file, ghi về thư mục tạm rồi rename
    out_dir = args.out.rsplit("/", 1)[0]
    tmp_dir = out_dir + "/_tmp_submission"
    (pred.coalesce(1)
         .write.mode("overwrite")
         .option("header", True)
         .csv(tmp_dir))

    # Tìm part-*.csv và đổi tên thành submission.csv
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    p = spark._jvm.org.apache.hadoop.fs.Path(tmp_dir)
    for f in fs.listStatus(p):
        name = f.getPath().getName()
        if name.startswith("part-") and name.endswith(".csv"):
            fs.rename(f.getPath(), spark._jvm.org.apache.hadoop.fs.Path(args.out))
    # dọn rác
    fs.delete(p, True)

    print(f"[DONE] Wrote {args.out}")
    spark.stop()

if __name__ == "__main__":
    main()
