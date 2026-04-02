"""
Phase 3: ML Model Training (scikit-learn)
News Recommendation Platform

- Loads featured data from MongoDB (or local Parquet fallback)
- Trains: Logistic Regression, Random Forest, Gradient Boosting
- Saves best model + scaler
- Writes evaluation metrics back to MongoDB
"""

import os
import joblib
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                              accuracy_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'article_ctr', 'popularity_score', 'user_ctr', 'category_match',
    'activity_encoded', 'title_len_encoded', 'top_category_affinity', 'TitleWordCount'
]
TARGET      = 'Click'
MODEL_DIR   = 'models'
PARQUET_DIR = 'data/spark_output/featured_dataset'


def load_data():
    """Load from MongoDB first; fall back to Parquet, then simulate."""
    try:
        from mongo_storage import load_interactions_from_mongo
        df = load_interactions_from_mongo(limit=50000)
        if not df.empty and TARGET in df.columns:
            print(f"✅ Loaded {len(df):,} rows from MongoDB")
            return df
    except Exception as e:
        print(f"⚠️  MongoDB load failed: {e}")

    if os.path.exists(PARQUET_DIR):
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("ModelTraining").master("local[*]").getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")
            df = spark.read.parquet(PARQUET_DIR).toPandas()
            spark.stop()
            print(f"✅ Loaded {len(df):,} rows from Parquet")
            return df
        except Exception as e:
            print(f"⚠️  Parquet load failed: {e}")

    print("⚠️  Generating simulated data for training...")
    return simulate_training_data()


def simulate_training_data(n=5000):
    np.random.seed(42)
    df = pd.DataFrame({
        'article_ctr':           np.random.beta(2, 5, n),
        'popularity_score':      np.random.exponential(0.3, n),
        'user_ctr':              np.random.beta(2, 5, n),
        'category_match':        np.random.randint(0, 2, n),
        'activity_encoded':      np.random.randint(0, 4, n).astype(float),
        'title_len_encoded':     np.random.randint(0, 4, n).astype(float),
        'top_category_affinity': np.random.beta(2, 3, n),
        'TitleWordCount':        np.random.randint(5, 20, n).astype(float),
    })
    # Realistic click signal
    score = (df['article_ctr'] * 2 + df['user_ctr'] * 1.5 +
             df['category_match'] * 0.5 + df['popularity_score'])
    prob  = 1 / (1 + np.exp(-(score - score.mean()) / score.std()))
    df[TARGET] = (np.random.random(n) < prob * 0.4).astype(int)
    return df


def prepare_data(df):
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET]
    print(f"\n--- Class Distribution ---")
    vc = y.value_counts()
    print(f"   Click=0: {vc.get(0,0):,} | Click=1: {vc.get(1,0):,}")
    print(f"   CTR: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    try:
        from imblearn.over_sampling import SMOTE
        X_train_s, y_train = SMOTE(random_state=42).fit_resample(X_train_s, y_train)
        print(f"   SMOTE applied. Balanced train size: {len(y_train):,}")
    except Exception:
        print("   SMOTE not available — using raw data")

    return X_train_s, X_test_s, y_train, y_test, scaler


def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=10,
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                           learning_rate=0.1, random_state=42),
    }

    results  = {}
    best_m   = None
    best_auc = 0.0

    for name, m in models.items():
        print(f"\n🔹 {name}")
        m.fit(X_train, y_train)
        yp   = m.predict(X_test)
        ypr  = m.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, yp),   4),
            "auc":       round(roc_auc_score(y_test, ypr),   4),
            "f1":        round(f1_score(y_test, yp),         4),
            "precision": round(precision_score(y_test, yp),  4),
            "recall":    round(recall_score(y_test, yp),     4),
        }
        results[name] = {**metrics, "model": m}

        print(f"   Accuracy : {metrics['accuracy']}")
        print(f"   AUC-ROC  : {metrics['auc']}")
        print(f"   F1-Score : {metrics['f1']}")
        print(f"   Precision: {metrics['precision']}")
        print(f"   Recall   : {metrics['recall']}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_m   = (name, m)

    print(f"\n🏆 Best Model: {best_m[0]}  (AUC={best_auc:.4f})")

    # Feature importance
    rf = results["Random Forest"]["model"]
    fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n--- Feature Importance (Random Forest) ---")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 40)
        print(f"   {feat:<30} {bar} {imp:.4f}")

    return best_m[1], results


def save_artifacts(model, scaler, results):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODEL_DIR, 'click_predictor.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print(f"\n✅ Model saved  → {MODEL_DIR}/click_predictor.pkl")
    print(f"✅ Scaler saved → {MODEL_DIR}/scaler.pkl")

    try:
        from mongo_storage import save_model_results
        clean = {k: {m: v for m, v in v.items() if m != "model"} for k, v in results.items()}
        save_model_results(clean)
    except Exception as e:
        print(f"⚠️  MongoDB metrics save failed: {e}")


if __name__ == "__main__":
    print("="*60)
    print("  PHASE 3: ML MODEL TRAINING")
    print("="*60)
    df = load_data()
    X_tr, X_te, y_tr, y_te, scaler = prepare_data(df)
    model, results = train_and_evaluate(X_tr, X_te, y_tr, y_te)
    save_artifacts(model, scaler, results)
    print("\n✅ Phase 3 Complete.")
