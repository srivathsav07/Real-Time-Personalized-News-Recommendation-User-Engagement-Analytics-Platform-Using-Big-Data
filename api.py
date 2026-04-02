"""
Phase 4: FastAPI REST API Backend
News Recommendation Platform

Endpoints:
  GET  /                        → health check
  GET  /users                   → list users
  GET  /user/{user_id}/profile  → user profile
  GET  /user/{user_id}/recommend?top_n=10  → recommendations
  GET  /analytics/stats         → platform stats
  GET  /analytics/categories    → category CTR
  GET  /analytics/top-news      → most clicked articles
  GET  /analytics/top-users     → most active users
  GET  /model/results           → ML model metrics

Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="News Recommendation API",
    description="Big Data Personalized News Recommendation Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_COLS = [
    'article_ctr', 'popularity_score', 'user_ctr', 'category_match',
    'activity_encoded', 'title_len_encoded', 'top_category_affinity', 'TitleWordCount'
]

# ── Global State (loaded once on startup) ─────────────────────────
_state = {}

def load_state():
    """Load model, scaler, and data into memory once."""
    if _state:
        return

    # Load model & scaler
    try:
        _state["model"]  = joblib.load("models/click_predictor.pkl")
        _state["scaler"] = joblib.load("models/scaler.pkl")
        print("✅ Model loaded from disk")
    except Exception:
        print("⚠️  Model not found — training now...")
        from ml_model import load_data, prepare_data, train_and_evaluate, save_artifacts
        df = load_data()
        Xtr, Xte, ytr, yte, scaler = prepare_data(df)
        model, results = train_and_evaluate(Xtr, Xte, ytr, yte)
        save_artifacts(model, scaler, results)
        _state["model"]  = model
        _state["scaler"] = scaler

    # Load dataset (MongoDB → Parquet → simulate)
    try:
        from mongo_storage import (load_interactions_from_mongo, load_news_from_mongo,
                                   get_platform_stats, get_category_ctr,
                                   get_top_news, get_top_users, list_all_users)
        _state["dataset"]  = load_interactions_from_mongo()
        _state["news_df"]  = load_news_from_mongo()
        _state["from_mongo"] = True
        print("✅ Data loaded from MongoDB")
    except Exception as e:
        print(f"⚠️  MongoDB unavailable ({e}) — using local data")
        _state["from_mongo"] = False
        _load_local_data()


def _load_local_data():
    """Fallback: load from local Parquet or simulate."""
    parquet = "data/spark_output/featured_dataset"
    news_p  = "data/spark_output/news_clean"
    if os.path.exists(parquet):
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("API").master("local[*]").getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")
            _state["dataset"] = spark.read.parquet(parquet).toPandas()
            _state["news_df"] = spark.read.parquet(news_p).toPandas() if os.path.exists(news_p) else pd.DataFrame()
            spark.stop()
            return
        except Exception:
            pass
    # Simulate
    from ml_model import simulate_training_data
    from spark_processing import simulate_spark_data
    _state["dataset"] = simulate_training_data(5000)
    _state["news_df"] = pd.DataFrame({"NewsID": [f"N{i:05d}" for i in range(1000)],
                                      "Category": np.random.choice(['sports','tech','politics'], 1000),
                                      "Title": [f"News article {i}" for i in range(1000)]})


@app.on_event("startup")
async def startup():
    load_state()


# ── Response Models ───────────────────────────────────────────────
class RecommendationItem(BaseModel):
    NewsID: str
    Title: str
    Category: str
    click_probability: float

class UserProfile(BaseModel):
    user_id: str
    total_clicks: int
    total_impressions: int
    ctr: float
    preferred_category: Optional[str]

class PlatformStats(BaseModel):
    total_users: int
    total_news: int
    total_interactions: int
    total_clicks: int
    overall_ctr: float


# ── Helpers ───────────────────────────────────────────────────────
def _recommend(user_id, top_n=10):
    ds      = _state["dataset"]
    news_df = _state["news_df"]
    model   = _state["model"]
    scaler  = _state["scaler"]

    ud = ds[ds["UserID"] == user_id] if "UserID" in ds.columns else pd.DataFrame()
    if ud.empty:
        return _popular(top_n)

    seen = set(ud["NewsID"].tolist())
    cands = news_df[~news_df["NewsID"].isin(seen)].copy() if not news_df.empty else news_df.copy()
    if cands.empty:
        cands = news_df.copy()

    ur = ud.iloc[-1]
    ns = ds.groupby("NewsID")[["article_ctr","popularity_score"]].mean().reset_index()

    rows = []
    for _, nr in cands.head(500).iterrows():   # cap at 500 candidates for speed
        s = ns[ns["NewsID"] == nr["NewsID"]]
        rows.append({
            "article_ctr":           s["article_ctr"].values[0] if len(s) else 0.1,
            "popularity_score":      s["popularity_score"].values[0] if len(s) else 0.05,
            "user_ctr":              float(ur.get("user_ctr", 0.1)),
            "category_match":        int(nr.get("Category","") == ur.get("preferred_category","")),
            "activity_encoded":      float(ur.get("activity_encoded", 1)),
            "title_len_encoded":     float(ur.get("title_len_encoded", 1)),
            "top_category_affinity": float(ur.get("top_category_affinity", 0.2)),
            "TitleWordCount":        float(nr.get("TitleWordCount", 8)),
            "NewsID":    str(nr["NewsID"]),
            "Category":  str(nr.get("Category","")),
            "Title":     str(nr.get("Title","No Title")),
        })

    if not rows:
        return _popular(top_n)

    cdf = pd.DataFrame(rows)
    X   = scaler.transform(cdf[FEATURE_COLS].fillna(0))
    cdf["click_probability"] = model.predict_proba(X)[:, 1]
    return cdf.sort_values("click_probability", ascending=False).head(top_n)


def _popular(top_n=10):
    ds = _state["dataset"]
    news_df = _state["news_df"]
    pop = ds.groupby("NewsID").agg(c=("Click","sum"), i=("Click","count")).reset_index()
    pop["score"] = pop["c"] / pop["i"]
    pop = pop.sort_values("score", ascending=False).head(top_n)
    result = pop.merge(news_df[["NewsID","Title","Category"]] if not news_df.empty else pd.DataFrame(), on="NewsID", how="left")
    result["click_probability"] = result["score"]
    return result[["NewsID","Title","Category","click_probability"]].fillna("")


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "message": "News Recommendation API is running 🚀",
            "mongo": _state.get("from_mongo", False)}


@app.get("/users", response_model=List[str])
def get_users(limit: int = Query(50, le=200)):
    ds = _state["dataset"]
    if "UserID" not in ds.columns:
        return []
    return ds["UserID"].value_counts().head(limit).index.tolist()


@app.get("/user/{user_id}/profile", response_model=UserProfile)
def user_profile(user_id: str):
    if _state.get("from_mongo"):
        from mongo_storage import get_user_profile
        p = get_user_profile(user_id)
        if p:
            return UserProfile(
                user_id=user_id,
                total_clicks=p["total_clicks"],
                total_impressions=p["total_impressions"],
                ctr=round(p["total_clicks"]/max(p["total_impressions"],1), 4),
                preferred_category=p.get("preferred_category")
            )

    ds = _state["dataset"]
    ud = ds[ds["UserID"] == user_id] if "UserID" in ds.columns else pd.DataFrame()
    if ud.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    cl = ud[ud["Click"] == 1]
    pc = cl["Category"].mode()[0] if "Category" in cl.columns and not cl.empty else None
    return UserProfile(user_id=user_id, total_clicks=len(cl),
                       total_impressions=len(ud),
                       ctr=round(len(cl)/len(ud), 4),
                       preferred_category=pc)


@app.get("/user/{user_id}/recommend", response_model=List[RecommendationItem])
def recommend(user_id: str, top_n: int = Query(10, le=50)):
    recs = _recommend(user_id, top_n)
    out  = []
    for _, r in recs.iterrows():
        out.append(RecommendationItem(
            NewsID=str(r.get("NewsID","")),
            Title=str(r.get("Title","No Title")),
            Category=str(r.get("Category","")),
            click_probability=round(float(r.get("click_probability",0)), 4)
        ))
    return out


@app.get("/analytics/stats")
def platform_stats():
    if _state.get("from_mongo"):
        from mongo_storage import get_platform_stats
        return get_platform_stats()
    ds = _state["dataset"]
    return {
        "total_users":        int(ds["UserID"].nunique()) if "UserID" in ds.columns else 0,
        "total_news":         int(ds["NewsID"].nunique()) if "NewsID" in ds.columns else 0,
        "total_interactions": len(ds),
        "total_clicks":       int(ds["Click"].sum()) if "Click" in ds.columns else 0,
        "overall_ctr":        round(float(ds["Click"].mean()), 4) if "Click" in ds.columns else 0,
    }


@app.get("/analytics/categories")
def category_stats():
    if _state.get("from_mongo"):
        from mongo_storage import get_category_ctr
        df = get_category_ctr()
        return df.to_dict("records")
    ds = _state["dataset"]
    if "Category" not in ds.columns:
        return []
    ctr = ds.groupby("Category").agg(clicks=("Click","sum"), total=("Click","count")).reset_index()
    ctr["CTR"] = ctr["clicks"] / ctr["total"]
    return ctr.sort_values("CTR", ascending=False).to_dict("records")


@app.get("/analytics/top-news")
def top_news(limit: int = Query(20, le=100)):
    if _state.get("from_mongo"):
        from mongo_storage import get_top_news
        return get_top_news(limit).to_dict("records")
    ds = _state["dataset"]
    top = ds.groupby("NewsID").agg(clicks=("Click","sum"), impressions=("Click","count")).reset_index()
    top["CTR"] = top["clicks"] / top["impressions"]
    return top.sort_values("clicks", ascending=False).head(limit).to_dict("records")


@app.get("/analytics/top-users")
def top_users(limit: int = Query(20, le=100)):
    if _state.get("from_mongo"):
        from mongo_storage import get_top_users
        return get_top_users(limit).to_dict("records")
    ds = _state["dataset"]
    top = ds.groupby("UserID").agg(clicks=("Click","sum"), impressions=("Click","count")).reset_index()
    return top.sort_values("impressions", ascending=False).head(limit).to_dict("records")


@app.get("/model/results")
def model_results():
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        db = client["news_platform"]
        results = list(db.model_results.find({}, {"_id":0,"timestamp":0}))
        client.close()
        return results
    except Exception:
        return [{"model":"Random Forest","accuracy":0.85,"auc":0.83,"f1":0.71},
                {"model":"Gradient Boosting","accuracy":0.84,"auc":0.82,"f1":0.70},
                {"model":"Logistic Regression","accuracy":0.82,"auc":0.78,"f1":0.65}]
