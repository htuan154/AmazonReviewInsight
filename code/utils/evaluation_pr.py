# code/utils/evaluation_pr.py
# Usage:
# spark-submit --master yarn --deploy-mode client \
#   code/utils/evaluation_pr.py \
#   --data  hdfs:///datasets/amazon/movies/parquet/reviews \
#   --model hdfs:///datasets/amazon/movies/models/logreg_baseline \
#   --out   hdfs:///datasets/amazon/movies/output/eval_pr \
#   --pr_points 200
#
# Ghi ra: metrics.json (areaUnderPR, pos/neg) và (tuỳ chọn) pr_points.parquet

import argparse, json, time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pr_points", type=int, default=0)  # >0 để xuất thêm điểm P/R
    return ap.parse_args()

def main():
    args = parse_args()
    spark = SparkSession.builder.appName("Helpful-Eval-PR").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = (spark.read.parquet(args.data)
          .select("review_id","review_text","star_rating","review_length","is_helpful")
          .na.fill({"review_text":"", "star_rating":0.0, "review_length":0}))
    model = PipelineModel.load(args.model)

    # dùng 20% để đánh giá (random)
    _, val = df.randomSplit([0.8, 0.2], seed=42)
    pred = model.transform(val).select("is_helpful","rawPrediction","probability")

    evaluator = BinaryClassificationEvaluator(labelCol="is_helpful",
                                              rawPredictionCol="rawPrediction",
                                              metricName="areaUnderPR")
    ap = float(evaluator.evaluate(pred))

    cnt = val.groupBy("is_helpful").count().collect()
    stats = {int(r["is_helpful"]): int(r["count"]) for r in cnt}
    pos, neg = stats.get(1,0), stats.get(0,0)

    # Save metrics.json
    metrics_path = args.out + "/metrics.json"
    (spark.createDataFrame([(pos,neg,ap,time.time())], ["pos","neg","val_ap","ts"])
          .write.mode("overwrite").json(metrics_path))

    print(f"[RESULT] Validation AUC-PR = {ap:.6f} (pos={pos}, neg={neg})")

    # Optional: PR points
    if args.pr_points and args.pr_points > 0:
        # BinaryClassificationMetrics yêu cầu RDD[(score,label)]
        rdd = pred.select((F.col("probability")[1]).alias("score"), F.col("is_helpful").cast("double")).rdd.map(tuple)
        bcm = BinaryClassificationMetrics(rdd)
        pr = bcm.pr().toDF(["recall","precision"])  # chú ý thứ tự (recall, precision)
        pr.write.mode("overwrite").parquet(args.out + "/pr_points.parquet")
        print(f"[DONE] Wrote PR points -> {args.out}/pr_points.parquet")

    spark.stop()

if __name__ == "__main__":
    main()
