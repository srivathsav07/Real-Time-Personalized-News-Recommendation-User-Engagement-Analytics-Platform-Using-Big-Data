Real-Time Personalized News Recommendation & User Engagement Analytics Platform


> A production-grade Big Data pipeline that processes Microsoft MIND news interaction logs through Hadoop HDFS + Apache Spark (Master/Worker cluster), stores engineered features in MongoDB, trains click-prediction ML models, and delivers personalized recommendations through a FastAPI backend and live Streamlit analytics dashboard.

---

Table of Contents

- [Architecture](#architecture)
- [Spark Master/Worker Cluster](#spark-masterworker-cluster)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features Engineered](#features-engineered)
- [ML Models](#ml-models)
- [API Reference](#api-reference)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Real-Time Streaming](#real-time-streaming)
- [Clustering](#clustering)
- [Dashboard](#dashboard)
- [Fallback Behaviour](#fallback-behaviour)
- [Future Scope](#future-scope)

---

Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MIND Dataset (TSV)                        │
│              news.tsv + behaviors.tsv                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Apache Hadoop HDFS                             │
│         Distributed File Storage (Port 9000)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Apache Spark Cluster (spark_processing.py)          │
│   Master → distributes jobs across Workers                   │
│   Feature Engineering: CTR · Popularity · Affinity · ...    │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐   ┌──────────────────────┐
│      MongoDB         │   │    Parquet (HDFS /    │
│  (mongo_storage.py)  │   │     local fallback)   │
│  interactions · news │   └──────────────────────┘
│  model_results · recs│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              ML Model Training (ml_model.py)                 │
│   Logistic Regression · Random Forest · Gradient Boosting    │
│   SMOTE oversampling · AUC-ROC selection · joblib save       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (api.py)                     │
│              REST API — Port 8000                            │
│   /recommend · /profile · /analytics · /model/results        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Streamlit + Plotly Dashboard (app.py)              │
│     Recommendations · Analytics · Model Performance          │
└─────────────────────────────────────────────────────────────┘

Real-Time Layer:
kafka_producer.py ──► [news-topic] ──► kafka_consumer.py
                                   └──► spark_streaming.py (Structured Streaming)

Clustering Layer:
clustering.py ──► KMeans (PySpark MLlib) ──► cluster_output/
```

---

Spark Master/Worker Cluster

This project uses **Apache Spark in distributed cluster mode** with a dedicated Master and Worker node setup — not just local mode.

How It Works

```
┌─────────────────────────────────────────────────────┐
│                  Spark Master Node                   │
│            spark://localhost:7077                    │
│   - Accepts job submissions                          │
│   - Schedules tasks across workers                   │
│   - Master UI: http://localhost:8080                 │
└──────────────────────┬──────────────────────────────┘
                       │  distributes tasks
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Worker 1  │ │   Worker 2  │ │   Worker N  │
│  (CPU core) │ │  (CPU core) │ │  (CPU core) │
│  local[*]   │ │  local[*]   │ │  local[*]   │
└─────────────┘ └─────────────┘ └─────────────┘
```

Starting the Cluster

```bash
# Start Spark Master
%SPARK_HOME%\sbin\start-master.cmd
# Master UI → http://localhost:8080

# Start a Worker (connect it to the master)
%SPARK_HOME%\sbin\start-worker.cmd spark://localhost:7077
# Worker UI → http://localhost:8081
```

Submitting Jobs to the Cluster

```bash
# Submit spark_processing.py to the cluster
spark-submit --master spark://localhost:7077 \
             --driver-memory 4g \
             --executor-memory 2g \
             spark_processing.py

# Submit clustering job
spark-submit --master spark://localhost:7077 \
             clustering.py
```

Configuration Used

```python
SparkSession.builder
  .appName("NewsRecommendationPlatform")
  .config("spark.master", "local[*]")          # uses all CPU cores
  .config("spark.driver.memory", "4g")
  .config("spark.sql.shuffle.partitions", "8")
  .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")  # HDFS mode
```

> Switch `local[*]` to `spark://localhost:7077` to run on a full cluster.

---

Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Distributed Storage | Apache Hadoop HDFS 3.3.6 | Raw data + Parquet output storage |
| Big Data Processing | Apache Spark / PySpark 3.5.1 | Feature engineering at scale |
| Cluster Management | Spark Master + Worker nodes | Distributed job scheduling |
| Stream Processing | Spark Structured Streaming + Kafka | Real-time click event processing |
| NoSQL Database | MongoDB 7.0 | Interaction, news & model storage |
| Machine Learning | scikit-learn, imbalanced-learn | Click prediction models |
| Clustering | PySpark MLlib KMeans | Article/user segmentation |
| API Backend | FastAPI + Uvicorn | REST endpoints |
| Frontend | Streamlit + Plotly | Interactive analytics dashboard |
| Messaging | Apache Kafka | Real-time event streaming |
| Language | Python 3.9+ | End-to-end implementation |

---

Project Structure

```
BDA_MIND/
├── MINDlarge_train/
│   └── MINDlarge_train/
│       ├── news.tsv              ← Article metadata
│       └── behaviors.tsv         ← User impression logs
│
├── spark_processing.py           ← Phase 1: PySpark + Hadoop pipeline
├── mongo_storage.py              ← Phase 2: MongoDB read/write layer
├── ml_model.py                   ← Phase 3: ML training & evaluation
├── api.py                        ← Phase 4: FastAPI REST backend
├── app.py                        ← Phase 5: Streamlit dashboard
│
├── clustering.py                 ← KMeans article clustering (PySpark MLlib)
├── kafka_producer.py             ← Simulated real-time click event producer
├── kafka_consumer.py             ← Kafka event consumer
├── spark_streaming.py            ← Spark Structured Streaming from Kafka
│
├── models/                       ← Auto-created on training
│   ├── click_predictor.pkl
│   └── scaler.pkl
│
├── data/spark_output/            ← Auto-created by Phase 1
│   ├── featured_dataset/         ← Parquet: full interaction features
│   └── news_clean/               ← Parquet: cleaned news metadata
│
├── cluster_output/               ← KMeans cluster assignments
├── SETUP_GUIDE.md
└── README.md
```

---

Dataset

**Microsoft MIND** (Microsoft News Dataset)

| File | Columns | Description |
|------|---------|-------------|
| `news.tsv` | NewsID, Category, SubCategory, Title, Abstract, URL | Article metadata |
| `behaviors.tsv` | ImpID, UserID, Time, History, Impressions | User click logs |

> If the MIND dataset is not present, the pipeline auto-generates realistic simulated data (500 users, 1,000 articles, 5,000 interactions).

Download: [https://msnews.github.io/](https://msnews.github.io/)

---

Features Engineered (via PySpark)

| Feature | Description | How Computed |
|---------|-------------|--------------|
| `article_ctr` | Article click-through rate | `total_clicks / total_impressions` |
| `popularity_score` | Engagement-weighted score | `log1p(clicks) × article_ctr` |
| `user_ctr` | User's historical CTR | `user_clicks / user_impressions` |
| `category_match` | Category preference match | `1` if article matches user's top category |
| `activity_encoded` | User activity level (0–3) | Bucketed by impression count |
| `title_len_encoded` | Title length bucket (0–3) | Bucketed by word count |
| `top_category_affinity` | Affinity score for preferred category | `category_clicks / total_clicks` |
| `TitleWordCount` | Raw word count | `size(split(Title, " "))` |

---

ML Models

Three classifiers are trained and compared. Best model by AUC-ROC is saved automatically.

| Model | Accuracy | AUC-ROC | F1-Score |
|-------|----------|---------|---------|
| Logistic Regression | ~0.82 | ~0.78 | ~0.65 |
| Random Forest  | ~0.85 | ~0.83 | ~0.71 |
| Gradient Boosting | ~0.84 | ~0.82 | ~0.70 |

**Training pipeline:**
1. 80/20 stratified train-test split
2. `StandardScaler` normalization
3. SMOTE oversampling (if `imbalanced-learn` installed)
4. Best model saved → `models/click_predictor.pkl`
5. Metrics persisted → MongoDB `model_results` collection

---

API Reference

Base URL: `http://localhost:8000`  
Docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/users?limit=50` | List most active users |
| GET | `/user/{user_id}/profile` | User profile: CTR, preferred category |
| GET | `/user/{user_id}/recommend?top_n=10` | Personalized article recommendations |
| GET | `/analytics/stats` | Platform-wide statistics |
| GET | `/analytics/categories` | CTR breakdown by news category |
| GET | `/analytics/top-news?limit=20` | Most-clicked articles |
| GET | `/analytics/top-users?limit=20` | Most active users |
| GET | `/model/results` | ML model evaluation metrics |

Example:
```bash
curl http://localhost:8000/user/U000001/recommend?top_n=5
```

---

Installation

Prerequisites

| Tool | Version | Download |
|------|---------|---------|
| Java JDK | 11 | [Adoptium](https://adoptium.net/temurin/releases/?version=11) |
| Hadoop | 3.3.6 | [hadoop.apache.org](https://hadoop.apache.org/releases.html) |
| Apache Spark | 3.5.1 | [spark.apache.org](https://spark.apache.org/downloads.html) |
| MongoDB | 7.0 | [mongodb.com](https://www.mongodb.com/try/download/community) |
| Python | 3.9+ | [python.org](https://www.python.org/downloads/) |

Windows Environment Variables

```
JAVA_HOME  = C:\Program Files\Eclipse Adoptium\jdk-11
HADOOP_HOME = C:\hadoop
SPARK_HOME  = C:\spark

PATH += C:\hadoop\bin
PATH += C:\spark\bin
```

> For Windows, place `winutils.exe` and `hadoop.dll` in `C:\hadoop\bin\`  
> Download from: [github.com/cdarlint/winutils](https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin)

Python Dependencies

```bash
pip install pyspark==3.5.1 pymongo fastapi uvicorn streamlit plotly \
            pandas numpy scikit-learn imbalanced-learn \
            python-multipart requests kafka-python --user
```

---

How to Run

Run each phase in order across separate terminals.

Terminal 1 — Phase 1: Spark Processing
```bash
cd BDA_MIND
python spark_processing.py
```
Reads MIND TSV → engineers features → saves to MongoDB + Parquet

Terminal 2 — Phase 3: Train ML Model
```bash
python ml_model.py
```
Trains 3 models → saves best to `models/` → writes metrics to MongoDB

Terminal 3 — Phase 4: Start API
```bash
uvicorn api:app --reload --port 8000
```
Auto-trains model if `models/` is missing. Visit: http://localhost:8000/docs

Terminal 4 — Phase 5: Start Dashboard
```bash
streamlit run app.py
```
Visit: http://localhost:8501

---

Real-Time Streaming

Requires Apache Kafka on `localhost:9092`.

```bash
# Terminal A — produce simulated click events (1/sec)
python kafka_producer.py

# Terminal B — print consumed events
python kafka_consumer.py

# Terminal C — Spark Structured Streaming consumer
python spark_streaming.py
```

Event Schema:
```json
{
  "UserID": "U42",
  "NewsID": "N00123",
  "Category": "technology",
  "Click": 1
}
```

---

Clustering

Segments articles into 3 behavioral clusters using **KMeans (PySpark MLlib)** on CTR, popularity score, and user engagement:

```bash
python clustering.py
```

Output saved to `cluster_output/` as CSV.

```
Cluster 0 → Low engagement articles
Cluster 1 → Moderate engagement articles
Cluster 2 → High engagement / viral articles
```

---

 Dashboard Pages

| Page | Content |
|------|---------|
|  Home | Platform stats, data source indicator |
|  Recommendations | User selector, ranked article cards with P(Click) scores |
|  Analytics | Category CTR charts, user CTR distribution, top articles |
|  Model Performance | Comparison table, grouped bar chart, radar chart, feature importance |
|  About | Architecture overview, tech stack table |

---

Fallback Behaviour

All components degrade gracefully — the platform runs even without Hadoop or MongoDB:

| Infrastructure | Fallback |
|---------------|---------|
| MongoDB down | Reads from local Parquet files |
| Parquet missing | Generates 5,000-row simulated dataset |
| Model not trained | `api.py` auto-trains on startup |
| MIND dataset absent | Spark pipeline generates simulated MIND-like data |
| FastAPI not running | Streamlit loads data directly (standalone mode) |

---

 Future Scope

- Deep learning recommendation models: **NRMS**, **NAML**
- Interest drift detection over time
- User cold-start handling
- Docker + Kubernetes deployment
- A/B testing framework for recommendation strategies
- HDFS NameNode HA (High Availability) setup

---

Author

Built as a Big Data Analytics capstone project demonstrating an end-to-end production pipeline from raw data ingestion to personalized real-time recommendations.

---

 License

This project is for educational purposes. The MIND dataset is subject to Microsoft's [dataset license](https://msnews.github.io/assets/doc/ACM%20MIND%20News%20Dataset%20License%20Agreement.pdf).
