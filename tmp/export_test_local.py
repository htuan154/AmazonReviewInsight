from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName("Export-Test-Parquet")
         .getOrCreate())

src = "hdfs:///output_v2/features_test_v3"   # đường dẫn HDFS đã có
dst = "file:///{PWD}/output_v2/features_test_v3/test.parquet".replace("{PWD}", __import__("os").getcwd().replace("\\","/"))

df = spark.read.parquet(src)
(df.coalesce(1)
   .write.mode("overwrite")
   .parquet(dst))

spark.stop()
