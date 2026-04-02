from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder \
    .appName("News Clustering") \
    .getOrCreate()

df = spark.read.csv("data/featured_dataset.csv", header=True, inferSchema=True)

assembler = VectorAssembler(
    inputCols=["article_ctr", "popularity_score", "user_ctr"],
    outputCol="features"
)

feature_df = assembler.transform(df)

kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(feature_df)

cluster_df = model.transform(feature_df)

cluster_df.groupBy("prediction").count().show()

cluster_df.write.mode("overwrite").csv("cluster_output", header=True)

spark.stop()