"""
Phase 2: MongoDB Storage & Retrieval Layer
News Recommendation Platform

Collections:
  - interactions  : featured user-article interaction records
  - news          : article metadata
  - model_results : ML model metrics
  - recommendations : cached recommendation results
"""

import pymongo
import pandas as pd
import numpy as np
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB  = "news_platform"


def get_db():
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.server_info()   # raises if not connected
    return client[MONGO_DB], client


# ── Read helpers ─────────────────────────────────────────────────
def load_interactions_from_mongo(limit=50000):
    """Load featured interaction dataset from MongoDB into pandas."""
    db, client = get_db()
    cursor = db.interactions.find({}, {"_id": 0}).limit(limit)
    df = pd.DataFrame(list(cursor))
    client.close()
    return df


def load_news_from_mongo():
    """Load news metadata from MongoDB."""
    db, client = get_db()
    cursor = db.news.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    client.close()
    return df


def get_user_profile(user_id):
    """Fetch a single user's aggregated profile from MongoDB."""
    db, client = get_db()
    pipeline = [
        {"$match": {"UserID": user_id}},
        {"$group": {
            "_id": "$UserID",
            "total_impressions": {"$sum": 1},
            "total_clicks":      {"$sum": "$Click"},
            "avg_user_ctr":      {"$avg": "$user_ctr"},
            "preferred_category":{"$first": "$preferred_category"},
            "top_category_affinity": {"$first": "$top_category_affinity"},
            "activity_encoded":  {"$first": "$activity_encoded"},
        }}
    ]
    result = list(db.interactions.aggregate(pipeline))
    client.close()
    return result[0] if result else None


def get_platform_stats():
    """Aggregate platform-wide statistics."""
    db, client = get_db()
    stats = {
        "total_users":    db.interactions.distinct("UserID").__len__(),
        "total_news":     db.news.count_documents({}),
        "total_interactions": db.interactions.count_documents({}),
        "total_clicks":   list(db.interactions.aggregate([
            {"$group": {"_id": None, "s": {"$sum": "$Click"}}}
        ]))[0]["s"],
    }
    stats["overall_ctr"] = round(stats["total_clicks"] / max(stats["total_interactions"], 1), 4)

    # Category breakdown
    cat_pipeline = [
        {"$match": {"Click": 1}},
        {"$group": {"_id": "$Category", "clicks": {"$sum": 1}}},
        {"$sort": {"clicks": -1}}
    ]
    stats["category_clicks"] = list(db.interactions.aggregate(cat_pipeline))

    client.close()
    return stats


def get_top_news(limit=20):
    """Get most clicked news articles."""
    db, client = get_db()
    pipeline = [
        {"$group": {
            "_id": "$NewsID",
            "clicks":      {"$sum": "$Click"},
            "impressions": {"$sum": 1},
            "Category":    {"$first": "$Category"},
            "Title":       {"$first": "$Title"},
        }},
        {"$addFields": {"CTR": {"$divide": ["$clicks", "$impressions"]}}},
        {"$sort": {"clicks": -1}},
        {"$limit": limit}
    ]
    result = list(db.interactions.aggregate(pipeline))
    client.close()
    return pd.DataFrame(result).rename(columns={"_id": "NewsID"})


def get_top_users(limit=20):
    """Get most active users."""
    db, client = get_db()
    pipeline = [
        {"$group": {
            "_id":         "$UserID",
            "clicks":      {"$sum": "$Click"},
            "impressions": {"$sum": 1},
        }},
        {"$addFields": {"CTR": {"$divide": ["$clicks", "$impressions"]}}},
        {"$sort": {"impressions": -1}},
        {"$limit": limit}
    ]
    result = list(db.interactions.aggregate(pipeline))
    client.close()
    return pd.DataFrame(result).rename(columns={"_id": "UserID"})


# ── Write helpers ─────────────────────────────────────────────────
def save_model_results(results: dict):
    """Persist ML model evaluation results to MongoDB."""
    db, client = get_db()
    db.model_results.drop()
    records = [{"model": k, **v, "timestamp": datetime.utcnow()}
               for k, v in results.items()]
    db.model_results.insert_many(records)
    client.close()
    print("✅ Model results saved to MongoDB")


def save_recommendations(user_id, recs_df):
    """Cache recommendations for a user."""
    db, client = get_db()
    db.recommendations.delete_many({"UserID": user_id})
    records = recs_df.to_dict("records")
    for r in records:
        r["UserID"] = user_id
        r["timestamp"] = datetime.utcnow()
    if records:
        db.recommendations.insert_many(records)
    client.close()


def get_cached_recommendations(user_id):
    """Retrieve cached recommendations from MongoDB."""
    db, client = get_db()
    cursor = db.recommendations.find({"UserID": user_id}, {"_id": 0}).sort("click_probability", -1)
    df = pd.DataFrame(list(cursor))
    client.close()
    return df


def get_category_ctr():
    """Get CTR by category for analytics."""
    db, client = get_db()
    pipeline = [
        {"$group": {
            "_id":   "$Category",
            "clicks": {"$sum": "$Click"},
            "total":  {"$sum": 1}
        }},
        {"$addFields": {"CTR": {"$divide": ["$clicks", "$total"]}}},
        {"$sort": {"CTR": -1}}
    ]
    result = list(db.interactions.aggregate(pipeline))
    client.close()
    return pd.DataFrame(result).rename(columns={"_id": "Category"})


def list_all_users(limit=100):
    """Get a list of all UserIDs."""
    db, client = get_db()
    pipeline = [
        {"$group": {"_id": "$UserID", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    result = list(db.interactions.aggregate(pipeline))
    client.close()
    return [r["_id"] for r in result]


# ── Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MongoDB connection...")
    try:
        db, client = get_db()
        print(f"✅ Connected to MongoDB | Database: {MONGO_DB}")
        print(f"   Collections: {db.list_collection_names()}")
        stats = get_platform_stats()
        print(f"   Platform Stats: {stats}")
        client.close()
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("   Make sure MongoDB is running: net start MongoDB")
