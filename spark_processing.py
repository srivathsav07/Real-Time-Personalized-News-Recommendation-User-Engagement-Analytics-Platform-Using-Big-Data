"""
Phase 1: PySpark + Hadoop Data Processing
Real-Time Personalized News Recommendation Platform
Big Data Stack: Hadoop HDFS + Apache Spark

Pipeline:
  MIND Dataset (TSV) → PySpark → Feature Engineering → MongoDB
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Spark Setup ───────────────────────────────────────────────────
os.environ.setdefault('HADOOP_HOME', r'C:\hadoop')
os.environ.setdefault('SPARK_HOME',  r'C:\spark')

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField,
                                StringType, IntegerType, FloatType)
from pyspark.sql.window import Window
import pymongo

# ─── Config ──────────────────────────────────────────────────────
MIND_BASE    = "MINDlarge_train/MINDlarge_train"   # adjust if needed
MONGO_URI    = "mongodb://localhost:27017/"
MONGO_DB     = "news_platform"
HDFS_OUTPUT  = "hdfs://localhost:9000/news_platform/featured"  # or local path
USE_HDFS     = False   # set True if HDFS is running; False = local mode

LOCAL_OUTPUT = "data/spark_output"


def create_spark_session():
    builder = (
        SparkSession.builder
        .appName("NewsRecommendationPlatform")
        .config("spark.master", "local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    )
    if USE_HDFS:
        builder = builder.config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"✅ Spark Session created | Version: {spark.version}")
    return spark


# ─── Schema Definitions ──────────────────────────────────────────
NEWS_SCHEMA = StructType([
    StructField("NewsID",      StringType(), True),
    StructField("Category",    StringType(), True),
    StructField("SubCategory", StringType(), True),
    StructField("Title",       StringType(), True),
    StructField("Abstract",    StringType(), True),
    StructField("URL",         StringType(), True),
    StructField("TitleEnt",    StringType(), True),
    StructField("AbsEnt",      StringType(), True),
])

BEH_SCHEMA = StructType([
    StructField("ImpID",       StringType(), True),
    StructField("UserID",      StringType(), True),
    StructField("Time",        StringType(), True),
    StructField("History",     StringType(), True),
    StructField("Impressions", StringType(), True),
])


# ─── Phase 1A: Load Data ─────────────────────────────────────────
def load_data(spark):
    print("\n📂 Loading MIND Dataset with PySpark...")

    news_path = os.path.join(MIND_BASE, "news.tsv")
    beh_path  = os.path.join(MIND_BASE, "behaviors.tsv")

    if not os.path.exists(news_path):
        print(f"⚠️  MIND data not found at {news_path}")
        print("    Generating simulated data...")
        return simulate_spark_data(spark)

    news_df = (spark.read
               .option("sep", "\t")
               .option("header", "false")
               .schema(NEWS_SCHEMA)
               .csv(news_path))

    beh_df = (spark.read
              .option("sep", "\t")
              .option("header", "false")
              .schema(BEH_SCHEMA)
              .csv(beh_path))

    print(f"   News articles: {news_df.count():,}")
    print(f"   Behavior rows: {beh_df.count():,}")
    return news_df, beh_df


def simulate_spark_data(spark):
    """Generate simulated MIND-like data using Spark."""
    import random, string
    random.seed(42)
    categories  = ['sports','technology','politics','entertainment','business',
                   'health','science','travel','food','world']
    subcats_map = {'sports':['football','cricket','tennis'],'technology':['AI','gadgets','software'],
                   'politics':['india','us','elections'],'entertainment':['bollywood','music','gaming'],
                   'business':['stocks','startups','economy'],'health':['fitness','nutrition','medicine'],
                   'science':['space','environment','research'],'travel':['domestic','international','adventure'],
                   'food':['recipes','restaurants','nutrition'],'world':['asia','europe','africa']}

    news_rows = []
    for i in range(1000):
        nid = f"N{str(i).zfill(5)}"
        cat = random.choice(categories)
        sub = random.choice(subcats_map[cat])
        news_rows.append((nid, cat, sub, f"Sample {cat} news about {sub} article {i}",
                          f"Abstract about {cat}", f"https://news.example.com/{nid}", "", ""))

    beh_rows = []
    user_ids = [f"U{str(i).zfill(6)}" for i in range(500)]
    news_ids = [f"N{str(i).zfill(5)}" for i in range(1000)]
    for j in range(5000):
        uid  = random.choice(user_ids)
        imps = " ".join([f"{random.choice(news_ids)}-{random.choice([0,1,1])}" for _ in range(random.randint(3,8))])
        beh_rows.append((str(j), uid, "2024-01-01 10:00:00", "", imps))

    news_df = spark.createDataFrame(news_rows, schema=NEWS_SCHEMA)
    beh_df  = spark.createDataFrame(beh_rows,  schema=BEH_SCHEMA)
    print(f"   Simulated news: {news_df.count():,}")
    print(f"   Simulated behaviors: {beh_df.count():,}")
    return news_df, beh_df


# ─── Phase 1B: Parse Impressions ─────────────────────────────────
def parse_impressions(beh_df):
    """Explode Impressions column into (UserID, NewsID, Click) rows."""
    print("\n⚙️  Parsing impression logs...")

    inter_df = (beh_df
        .filter(F.col("Impressions").isNotNull())
        .withColumn("imp", F.explode(F.split("Impressions", " ")))
        .withColumn("NewsID", F.split("imp", "-")[0])
        .withColumn("Click",  F.split("imp", "-")[1].cast(IntegerType()))
        .filter(F.col("Click").isNotNull())
        .select("UserID", "NewsID", "Click", "Time"))

    print(f"   Total interactions: {inter_df.count():,}")
    click_rate = inter_df.agg(F.mean("Click")).collect()[0][0]
    print(f"   Overall CTR: {click_rate:.2%}")
    return inter_df


# ─── Phase 1C: Feature Engineering with Spark ────────────────────
def engineer_features(news_df, inter_df):
    print("\n🔧 Engineering features with PySpark...")

    # News metadata features
    news_clean = (news_df
        .select("NewsID","Category","SubCategory","Title")
        .withColumn("TitleWordCount", F.size(F.split("Title", " ")))
        .fillna({"Category":"unknown","SubCategory":"unknown","Title":""}))

    # Article-level popularity features
    art_stats = (inter_df
        .groupBy("NewsID")
        .agg(F.count("Click").alias("total_impressions"),
             F.sum("Click").alias("total_clicks"))
        .withColumn("article_ctr",
                    F.round(F.col("total_clicks") / F.col("total_impressions"), 4))
        .withColumn("popularity_score",
                    F.round(F.log1p(F.col("total_clicks")) * F.col("article_ctr"), 4)))

    # User-level features
    user_stats = (inter_df
        .groupBy("UserID")
        .agg(F.count("Click").alias("user_total_impressions"),
             F.sum("Click").alias("user_total_clicks"))
        .withColumn("user_ctr",
                    F.round(F.col("user_total_clicks") / F.col("user_total_impressions"), 4))
        .withColumn("activity_encoded",
                    F.when(F.col("user_total_impressions") <= 5,  0)
                     .when(F.col("user_total_impressions") <= 15, 1)
                     .when(F.col("user_total_impressions") <= 30, 2)
                     .otherwise(3)))

    # Category affinity per user
    with_cat = inter_df.join(news_clean.select("NewsID","Category"), "NewsID", "left")
    cat_clicks = (with_cat.filter(F.col("Click") == 1)
                  .groupBy("UserID","Category")
                  .agg(F.count("Click").alias("cat_click_count")))
    user_total_cat = (cat_clicks.groupBy("UserID")
                      .agg(F.sum("cat_click_count").alias("total_cat_clicks")))
    cat_affinity = (cat_clicks.join(user_total_cat, "UserID")
                    .withColumn("category_affinity",
                                F.round(F.col("cat_click_count") / F.col("total_cat_clicks"), 4)))

    # Top category per user
    w = Window.partitionBy("UserID").orderBy(F.col("category_affinity").desc())
    top_cat = (cat_affinity
               .withColumn("rank", F.rank().over(w))
               .filter(F.col("rank") == 1)
               .select(F.col("UserID"),
                       F.col("Category").alias("preferred_category"),
                       F.col("category_affinity").alias("top_category_affinity")))

    # Join everything
    dataset = (inter_df
        .join(news_clean,  "NewsID", "left")
        .join(art_stats,   "NewsID", "left")
        .join(user_stats,  "UserID", "left")
        .join(top_cat,     "UserID", "left")
        .withColumn("category_match",
                    (F.col("Category") == F.col("preferred_category")).cast(IntegerType()))
        .withColumn("title_len_encoded",
                    F.when(F.col("TitleWordCount") <= 6,  0)
                     .when(F.col("TitleWordCount") <= 10, 1)
                     .when(F.col("TitleWordCount") <= 15, 2)
                     .otherwise(3))
        .fillna({"article_ctr":0.0,"popularity_score":0.0,"user_ctr":0.0,
                 "top_category_affinity":0.0,"activity_encoded":1,"title_len_encoded":1,
                 "TitleWordCount":8,"category_match":0}))

    print(f"   Featured dataset rows: {dataset.count():,}")
    print(f"   Columns: {dataset.columns}")
    return dataset, news_clean


# ─── Phase 1D: Save to MongoDB ───────────────────────────────────
def save_to_mongodb(dataset, news_clean):
    print("\n💾 Saving to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db     = client[MONGO_DB]

    # Save featured interactions
    inter_cols = ["UserID","NewsID","Click","Category","Title",
                  "article_ctr","popularity_score","user_ctr",
                  "category_match","activity_encoded","title_len_encoded",
                  "top_category_affinity","TitleWordCount","preferred_category"]
    # Sample to avoid memory issues with large datasets
    sample = dataset.select([c for c in inter_cols if c in dataset.columns]).limit(50000)
    inter_records = [row.asDict() for row in sample.collect()]

    db.interactions.drop()
    if inter_records:
        db.interactions.insert_many(inter_records)
    print(f"   ✅ Inserted {len(inter_records):,} interaction records")

    # Save news metadata
    news_records = [row.asDict() for row in news_clean.collect()]
    db.news.drop()
    if news_records:
        db.news.insert_many(news_records)
    print(f"   ✅ Inserted {len(news_records):,} news articles")

    # Create indexes
    db.interactions.create_index("UserID")
    db.interactions.create_index("NewsID")
    db.news.create_index("NewsID")
    print("   ✅ MongoDB indexes created")

    client.close()


# ─── Phase 1E: Save to local / HDFS ─────────────────────────────
def save_to_parquet(dataset, news_clean):
    print("\n💾 Saving Parquet files...")
    os.makedirs(LOCAL_OUTPUT, exist_ok=True)
    out_path  = HDFS_OUTPUT if USE_HDFS else LOCAL_OUTPUT

    (dataset.write.mode("overwrite")
     .parquet(f"{out_path}/featured_dataset"))
    (news_clean.write.mode("overwrite")
     .parquet(f"{out_path}/news_clean"))
    print(f"   ✅ Parquet saved to: {out_path}")


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  PHASE 1: PySpark + Hadoop Data Processing")
    print("="*60)

    spark = create_spark_session()

    result = load_data(spark)
    if isinstance(result, tuple):
        news_df, beh_df = result
    else:
        print("Failed to load data.")
        sys.exit(1)

    inter_df             = parse_impressions(beh_df)
    dataset, news_clean  = engineer_features(news_df, inter_df)
    save_to_parquet(dataset, news_clean)

    try:
        save_to_mongodb(dataset, news_clean)
    except Exception as e:
        print(f"   ⚠️  MongoDB save failed: {e}")
        print("   (Make sure MongoDB is running: net start MongoDB)")

    spark.stop()
    print("\n✅ Phase 1 Complete!")
