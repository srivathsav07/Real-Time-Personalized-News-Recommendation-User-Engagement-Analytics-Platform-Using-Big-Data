# 🛠️ Complete Installation Guide — Windows
# Big Data News Recommendation Platform
# Stack: Hadoop + PySpark + MongoDB + FastAPI + Streamlit + Plotly

---

## STEP 1 — Install Java (Required for Hadoop & Spark)

1. Download Java JDK 11:
   https://adoptium.net/temurin/releases/?version=11
2. Install → choose "Set JAVA_HOME" during install
3. Verify:
   ```
   java -version
   ```

---

## STEP 2 — Install Hadoop on Windows

1. Download Hadoop 3.3.6:
   https://hadoop.apache.org/releases.html
   → hadoop-3.3.6.tar.gz

2. Extract to: `C:\hadoop`

3. Download winutils.exe for Windows:
   https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin
   → Place winutils.exe and hadoop.dll into `C:\hadoop\bin\`

4. Set Environment Variables (System → Advanced → Environment Variables):
   ```
   HADOOP_HOME = C:\hadoop
   Add to PATH: C:\hadoop\bin
   ```

5. Verify:
   ```
   hadoop version
   ```

---

## STEP 3 — Install Apache Spark

1. Download Spark 3.5.1 (pre-built for Hadoop 3):
   https://spark.apache.org/downloads.html
   → spark-3.5.1-bin-hadoop3.tgz

2. Extract to: `C:\spark`

3. Set Environment Variables:
   ```
   SPARK_HOME = C:\spark
   HADOOP_HOME = C:\hadoop
   Add to PATH: C:\spark\bin
   ```

4. Verify:
   ```
   spark-shell --version
   ```

---

## STEP 4 — Install MongoDB

1. Download MongoDB Community Server:
   https://www.mongodb.com/try/download/community

2. Install → choose "Complete" installation
   → Also install MongoDB Compass (GUI)

3. MongoDB runs automatically as a Windows Service on port 27017

4. Verify:
   ```
   mongosh
   ```

---

## STEP 5 — Install Python Packages

```bash
pip install pyspark==3.5.1
pip install pymongo
pip install fastapi
pip install uvicorn
pip install streamlit
pip install plotly
pip install pandas
pip install numpy
pip install scikit-learn
pip install imbalanced-learn
pip install python-multipart
pip install requests
```

---

## STEP 6 — Verify Full Stack

```python
# test_setup.py
from pyspark.sql import SparkSession
import pymongo
import fastapi
import streamlit
import plotly

print("PySpark:", __import__('pyspark').__version__)
print("PyMongo:", pymongo.__version__)
print("FastAPI:", fastapi.__version__)
print("All OK!")
```

Run: `python test_setup.py`

---

## PROJECT FOLDER STRUCTURE

```
BDA_MIND/
├── MINDlarge_train/MINDlarge_train/
│   ├── news.tsv
│   └── behaviors.tsv
├── spark_processing.py     ← Phase 1: PySpark + Hadoop
├── mongo_storage.py        ← Phase 2: MongoDB
├── ml_model.py             ← Phase 3: scikit-learn
├── api.py                  ← Phase 4: FastAPI
├── app.py                  ← Phase 5: Streamlit + Plotly
└── requirements.txt
```

---

## HOW TO RUN (in order)

```bash
# Terminal 1 — Run Spark processing + MongoDB load
python spark_processing.py

# Terminal 2 — Start FastAPI backend
uvicorn api:app --reload --port 8000

# Terminal 3 — Start Streamlit frontend
streamlit run app.py
```
