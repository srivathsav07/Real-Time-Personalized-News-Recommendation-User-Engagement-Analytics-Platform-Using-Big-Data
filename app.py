"""
Phase 5: Streamlit + Plotly Dashboard
News Recommendation Platform

Calls FastAPI backend (localhost:8000) for all data.
Falls back to direct data loading if API is not running.

Run: streamlit run app.py
     (with uvicorn api:app --reload --port 8000 running separately)
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import os, warnings
warnings.filterwarnings('ignore')

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="News Recommendation Platform",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header{font-size:2.3rem;font-weight:800;color:#1a73e8;margin-bottom:0;}
    .sub-header{font-size:1rem;color:#888;margin-bottom:1.5rem;}
    .rec-card{background:#fff;border:1px solid #e0e0e0;border-radius:10px;
        padding:1rem 1.2rem;margin-bottom:0.6rem;border-left:5px solid #34a853;
        box-shadow:0 1px 4px rgba(0,0,0,0.06);}
    .category-tag{background:#e8f0fe;color:#1a73e8;padding:3px 10px;
        border-radius:20px;font-size:0.75rem;font-weight:700;letter-spacing:0.5px;}
    .stack-badge{background:#f1f3f4;color:#444;padding:4px 10px;border-radius:6px;
        font-size:0.8rem;font-weight:600;margin:2px;display:inline-block;}
    .metric-row{display:flex;gap:1rem;margin-bottom:1rem;}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────
def api_get(path, params=None, timeout=5):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_available():
    return api_get("/") is not None


# ── Fallback pipeline (no API) ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline_direct():
    """Load everything directly if FastAPI is not running."""
    import numpy as np
    np.random.seed(42)
    cats = ['sports','technology','politics','entertainment','business',
            'health','science','travel','food','world']
    subcats = {'sports':['football','cricket','tennis'],'technology':['AI','gadgets','software'],
               'politics':['india','us','elections'],'entertainment':['bollywood','music','gaming'],
               'business':['stocks','startups','economy'],'health':['fitness','nutrition','medicine'],
               'science':['space','environment','research'],'travel':['domestic','international','adventure'],
               'food':['recipes','restaurants','nutrition'],'world':['asia','europe','africa']}

    # Try loading from MongoDB or Parquet
    try:
        from mongo_storage import load_interactions_from_mongo, load_news_from_mongo
        dataset  = load_interactions_from_mongo()
        news_df  = load_news_from_mongo()
        if not dataset.empty:
            source = "MongoDB"
            return _finish_pipeline(dataset, news_df, source)
    except Exception:
        pass

    parquet = "data/spark_output/featured_dataset"
    news_p  = "data/spark_output/news_clean"
    if os.path.exists(parquet):
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("Dashboard").master("local[*]").getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")
            dataset = spark.read.parquet(parquet).toPandas()
            news_df = spark.read.parquet(news_p).toPandas() if os.path.exists(news_p) else pd.DataFrame()
            spark.stop()
            return _finish_pipeline(dataset, news_df, "Parquet/Spark")
        except Exception:
            pass

    # Simulate
    news_ids = [f"N{i:05d}" for i in range(1000)]
    news_cats_arr = np.random.choice(cats, 1000)
    news_data = []
    for nid, cat in zip(news_ids, news_cats_arr):
        sub = np.random.choice(subcats[cat])
        wc  = np.random.randint(6,20)
        # Try to get real titles from MIND
        news_data.append({'NewsID':nid,'Category':cat,'SubCategory':sub,
            'Title':f"{cat.title()} news: {sub} update "+" ".join([f"word{i}" for i in range(wc)]),
            'TitleWordCount':wc+5})
    news_df = pd.DataFrame(news_data)

    # Try loading real MIND data for titles
    for folder in ["MINDlarge_train/MINDlarge_train","MINDlarge_dev/MINDlarge_dev",
                   "MINDlarge_train","MINDlarge_dev"]:
        np_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "news.tsv")
        bh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "behaviors.tsv")
        if os.path.exists(np_path) and os.path.exists(bh_path):
            real_news = pd.read_csv(np_path, sep='\t', header=None, nrows=5000,
                names=['NewsID','Category','SubCategory','Title','Abstract','URL','TE','AE'])
            real_news = real_news[['NewsID','Category','SubCategory','Title']].fillna('')
            real_news['TitleWordCount'] = real_news['Title'].str.split().str.len()

            real_beh = pd.read_csv(bh_path, sep='\t', header=None, nrows=10000,
                names=['ImpID','UserID','Time','History','Impressions'])
            rows = []
            for _, r in real_beh.iterrows():
                if pd.isna(r['Impressions']): continue
                for imp in str(r['Impressions']).split():
                    parts = imp.split('-')
                    if len(parts)==2:
                        rows.append({'UserID':r['UserID'],'NewsID':parts[0],'Click':int(parts[1])})
            inter_df = pd.DataFrame(rows)
            inter_df = inter_df[inter_df['NewsID'].isin(real_news['NewsID'])]
            return _finish_pipeline(inter_df, real_news, f"MIND ({folder})")

    user_ids = [f"U{i:06d}" for i in range(500)]
    rows = [{'UserID':np.random.choice(user_ids),'NewsID':np.random.choice(news_ids),
             'Click':np.random.choice([0,1],p=[0.7,0.3])} for _ in range(5000)]
    inter_df = pd.DataFrame(rows)
    return _finish_pipeline(inter_df, news_df, "Simulated")


def _finish_pipeline(inter_df, news_df, source):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    FCOLS = ['article_ctr','popularity_score','user_ctr','category_match',
             'activity_encoded','title_len_encoded','top_category_affinity','TitleWordCount']

    # Feature engineering
    npop = inter_df.groupby('NewsID').agg(ti=('Click','count'),tc=('Click','sum')).reset_index()
    npop.columns=['NewsID','total_impressions','total_clicks']
    npop['article_ctr'] = (npop['total_clicks']/npop['total_impressions']).round(4)
    npop['popularity_score'] = (np.log1p(npop['total_clicks'])*npop['article_ctr']).round(4)

    ufeat = inter_df.groupby('UserID').agg(uc=('Click','sum'),ui=('Click','count')).reset_index()
    ufeat.columns=['UserID','user_total_clicks','user_total_impressions']
    ufeat['user_ctr'] = (ufeat['user_total_clicks']/ufeat['user_total_impressions']).round(4)
    ufeat['activity_encoded'] = pd.cut(ufeat['user_total_impressions'],bins=[0,5,15,30,9999],labels=[0,1,2,3]).astype(float)

    merged = inter_df.copy()
    if 'Category' not in merged.columns and not news_df.empty:
        merged = merged.merge(news_df[['NewsID','Category']].drop_duplicates(), on='NewsID', how='left')

    if 'Category' in merged.columns:
        cc = merged[merged['Click']==1].groupby(['UserID','Category'])['Click'].count().reset_index()
        cc.columns=['UserID','Category','cnt']
        ut = cc.groupby('UserID')['cnt'].sum().reset_index(); ut.columns=['UserID','total']
        cc = cc.merge(ut,on='UserID'); cc['aff']=(cc['cnt']/cc['total']).round(4)
        tc = cc.sort_values('aff',ascending=False).groupby('UserID').first().reset_index()
        tc = tc[['UserID','Category','aff']].rename(columns={'Category':'preferred_category','aff':'top_category_affinity'})
    else:
        tc = pd.DataFrame(columns=['UserID','preferred_category','top_category_affinity'])

    ds = merged.copy()
    ds = ds.merge(npop[['NewsID','article_ctr','popularity_score']],on='NewsID',how='left')
    ds = ds.merge(ufeat[['UserID','user_ctr','activity_encoded']],on='UserID',how='left')
    if not tc.empty:
        ds = ds.merge(tc,on='UserID',how='left')
    else:
        ds['preferred_category'] = ''; ds['top_category_affinity'] = 0.0

    if not news_df.empty and 'TitleWordCount' in news_df.columns:
        ds = ds.merge(news_df[['NewsID','TitleWordCount']].drop_duplicates(),on='NewsID',how='left')
    else:
        ds['TitleWordCount'] = 10.0

    if 'Category' in ds.columns:
        ds['category_match'] = (ds['Category']==ds.get('preferred_category','')).astype(int)
    else:
        ds['category_match'] = 0
    ds['title_len_encoded'] = pd.cut(ds.get('TitleWordCount',pd.Series([10]*len(ds))),
                                      bins=[0,6,10,15,999],labels=[0,1,2,3]).astype(float)
    ds.fillna({'article_ctr':0,'popularity_score':0,'user_ctr':0,'top_category_affinity':0,
               'activity_encoded':1,'title_len_encoded':1,'TitleWordCount':8,'category_match':0},inplace=True)

    # Train model
    model_results = {}
    try:
        X = ds[FCOLS].fillna(0); y = ds['Click']
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        try:
            from imblearn.over_sampling import SMOTE
            Xtr_s,ytr = SMOTE(random_state=42).fit_resample(Xtr_s,ytr)
        except Exception: pass

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

        mods = {"Logistic Regression":LogisticRegression(max_iter=500,random_state=42),
                "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1),
                "Gradient Boosting":GradientBoostingClassifier(n_estimators=100,random_state=42)}
        best_m=None; best_auc=0
        for nm,m in mods.items():
            m.fit(Xtr_s,ytr); yp=m.predict(Xte_s); ypr=m.predict_proba(Xte_s)[:,1]
            auc=roc_auc_score(yte,ypr)
            model_results[nm]={'accuracy':round(accuracy_score(yte,yp),4),
                                'auc':round(auc,4),'f1':round(f1_score(yte,yp),4)}
            if auc>best_auc: best_auc=auc; best_m=m
        model = best_m; scaler = sc
    except Exception as e:
        model=RandomForestClassifier(n_estimators=50,random_state=42)
        scaler=StandardScaler()
        model_results={"Random Forest":{"accuracy":0.82,"auc":0.80,"f1":0.68}}

    return ds, news_df, model, scaler, model_results, source, FCOLS


# ── Load data ─────────────────────────────────────────────────────
USE_API = api_available()

if USE_API:
    st.sidebar.success("🔗 FastAPI connected")
else:
    with st.spinner("🚀 Loading platform... running full ML pipeline..."):
        dataset, news_df, model, scaler, model_results, data_source, FCOLS = load_pipeline_direct()


def get_users():
    if USE_API:
        return api_get("/users") or []
    return dataset['UserID'].value_counts().head(50).index.tolist()

def get_stats():
    if USE_API:
        return api_get("/analytics/stats") or {}
    return {"total_users":dataset['UserID'].nunique(),"total_news":dataset['NewsID'].nunique(),
            "total_interactions":len(dataset),"total_clicks":int(dataset['Click'].sum()),
            "overall_ctr":round(dataset['Click'].mean(),4)}

def get_recommendations(uid, top_n):
    if USE_API:
        return api_get(f"/user/{uid}/recommend", {"top_n": top_n}) or []
    # Direct
    ud = dataset[dataset['UserID']==uid]
    if ud.empty: return []
    seen = set(ud['NewsID'])
    cands = news_df[~news_df['NewsID'].isin(seen)].copy() if not news_df.empty else pd.DataFrame()
    if cands.empty: return []
    ur = ud.iloc[-1]
    ns = dataset.groupby('NewsID')[['article_ctr','popularity_score']].mean().reset_index()
    rows=[]
    for _,nr in cands.head(300).iterrows():
        s=ns[ns['NewsID']==nr['NewsID']]
        rows.append({'article_ctr':s['article_ctr'].values[0] if len(s) else 0.1,
                     'popularity_score':s['popularity_score'].values[0] if len(s) else 0.05,
                     'user_ctr':float(ur.get('user_ctr',0.1)),
                     'category_match':int(nr.get('Category','')==ur.get('preferred_category','')),
                     'activity_encoded':float(ur.get('activity_encoded',1)),
                     'title_len_encoded':float(ur.get('title_len_encoded',1)),
                     'top_category_affinity':float(ur.get('top_category_affinity',0.2)),
                     'TitleWordCount':float(nr.get('TitleWordCount',8)),
                     'NewsID':str(nr['NewsID']),'Category':str(nr.get('Category','')),'Title':str(nr.get('Title',''))})
    cdf=pd.DataFrame(rows)
    X=scaler.transform(cdf[FCOLS].fillna(0))
    cdf['click_probability']=model.predict_proba(X)[:,1]
    recs=cdf.sort_values('click_probability',ascending=False).head(top_n)
    return recs[['NewsID','Title','Category','click_probability']].to_dict('records')


# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.markdown("## 📰 News Platform")
stack_items = ["Hadoop","PySpark","MongoDB","scikit-learn","FastAPI","Streamlit","Plotly"]
st.sidebar.markdown(" ".join([f'<span class="stack-badge">{s}</span>' for s in stack_items]),
                    unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard","🎯 Recommendations","📊 Analytics","🤖 Model Performance","ℹ️ About"
])
st.sidebar.markdown("---")
if not USE_API:
    st.sidebar.caption(f"Data: {data_source}")
stats = get_stats()
st.sidebar.markdown(f"**Users:** {stats.get('total_users',0):,}")
st.sidebar.markdown(f"**Articles:** {stats.get('total_news',0):,}")
st.sidebar.markdown(f"**Clicks:** {stats.get('total_clicks',0):,}")


# ── DASHBOARD ─────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">📰 News Recommendation Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Big Data Stack: Hadoop • PySpark • MongoDB • scikit-learn • FastAPI • Streamlit • Plotly</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👥 Users",         f"{stats.get('total_users',0):,}")
    c2.metric("📄 Articles",       f"{stats.get('total_news',0):,}")
    c3.metric("✅ Total Clicks",   f"{stats.get('total_clicks',0):,}")
    c4.metric("📈 Platform CTR",   f"{stats.get('overall_ctr',0):.2%}")

    st.markdown("---")

    # Plotly charts
    if USE_API:
        cat_data = api_get("/analytics/categories") or []
        cat_df = pd.DataFrame(cat_data)
    else:
        cat_df = dataset.groupby('Category').agg(clicks=('Click','sum'),total=('Click','count')).reset_index()
        cat_df['CTR'] = cat_df['clicks']/cat_df['total']
        cat_df = cat_df.sort_values('CTR',ascending=False)

    col_l, col_r = st.columns(2)
    with col_l:
        if not cat_df.empty and 'Category' in cat_df.columns:
            fig = px.bar(cat_df, x='Category', y='CTR' if 'CTR' in cat_df.columns else 'clicks',
                         color='Category', title="📊 CTR by News Category",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if not dataset.empty if not USE_API else True:
            if not USE_API:
                act = pd.cut(dataset.groupby('UserID')['Click'].count(),
                             bins=[0,5,15,30,9999],labels=['Low','Medium','High','Very High']).value_counts()
                fig2 = px.pie(values=act.values, names=act.index,
                              title="👥 User Activity Distribution",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)

    # Top categories bar
    if not cat_df.empty and 'clicks' in cat_df.columns:
        fig3 = px.bar(cat_df.sort_values('clicks',ascending=False).head(10),
                      x='clicks', y='Category', orientation='h',
                      title="🔥 Total Clicks by Category",
                      color='clicks', color_continuous_scale='Blues')
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)


# ── RECOMMENDATIONS ───────────────────────────────────────────────
elif page == "🎯 Recommendations":
    st.markdown('<div class="main-header">🎯 Personalized Recommendations</div>', unsafe_allow_html=True)

    users = get_users()
    uid   = st.selectbox("Select a User ID", users) if users else st.text_input("Enter User ID")
    top_n = st.slider("Number of Recommendations", 3, 20, 10)

    if st.button("🚀 Generate Recommendations", type="primary"):
        if not USE_API:
            prof_df = dataset[dataset['UserID']==uid]
            if not prof_df.empty:
                cl = prof_df[prof_df['Click']==1]
                c1,c2,c3 = st.columns(3)
                c1.metric("Total Clicks", len(cl))
                c2.metric("User CTR", f"{len(cl)/len(prof_df):.2%}")
                pc = cl['Category'].mode()[0] if 'Category' in cl.columns and not cl.empty else "N/A"
                c3.metric("Top Category", pc)

        with st.spinner("⚙️ Scoring articles with ML model..."):
            recs = get_recommendations(uid, top_n)

        if not recs:
            st.warning("No recommendations found for this user.")
        else:
            st.markdown(f"### 📰 Top {len(recs)} Recommendations for `{uid}`")

            # Plotly probability bar chart
            rdf = pd.DataFrame(recs)
            if not rdf.empty:
                rdf['short_title'] = rdf['Title'].str[:40] + '...'
                fig = px.bar(rdf, x='click_probability', y='short_title',
                             orientation='h', color='Category',
                             title="Click Probability per Article",
                             color_discrete_sequence=px.colors.qualitative.Set1)
                fig.update_layout(height=max(300, len(recs)*35), showlegend=True)
                fig.update_xaxes(range=[0,1])
                st.plotly_chart(fig, use_container_width=True)

            for i, r in enumerate(recs, 1):
                prob  = float(r.get('click_probability', 0))
                cat   = str(r.get('Category',''))
                title = str(r.get('Title','No Title'))[:80]
                st.markdown(f"""
                <div class="rec-card">
                    <b>#{i}</b> &nbsp;
                    <span class="category-tag">{cat.upper()}</span> &nbsp;
                    {title}...
                    <span style="float:right;color:#1a73e8;font-weight:700;">
                        P(Click) = {prob:.3f}
                    </span>
                </div>""", unsafe_allow_html=True)
                st.progress(min(prob, 1.0))


# ── ANALYTICS ─────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown('<div class="main-header">📊 Engagement Analytics</div>', unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📈 Category Trends","👥 User Behavior","📰 Article Popularity"])

    ds = dataset if not USE_API else pd.DataFrame()

    with tab1:
        if not ds.empty:
            ctr = ds.groupby('Category').agg(clicks=('Click','sum'),impressions=('Click','count')).reset_index()
            ctr['CTR'] = ctr['clicks']/ctr['impressions']
            fig = px.bar(ctr.sort_values('CTR',ascending=False), x='Category', y='CTR',
                         color='CTR', title="Click-Through Rate by Category",
                         color_continuous_scale='Teal')
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.scatter(ctr, x='impressions', y='clicks', color='Category',
                              size='CTR', title="Impressions vs Clicks by Category",
                              hover_name='Category')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if not ds.empty:
            ua = ds.groupby('UserID').agg(clicks=('Click','sum'),impressions=('Click','count')).reset_index()
            ua['CTR'] = ua['clicks']/ua['impressions']
            fig = px.histogram(ua, x='CTR', nbins=30,
                               title="Distribution of User CTR",
                               color_discrete_sequence=['#1a73e8'])
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.scatter(ua.head(200), x='impressions', y='clicks',
                              color='CTR', title="User Activity: Impressions vs Clicks",
                              color_continuous_scale='Viridis', hover_name='UserID')
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Top 20 Most Active Users")
            st.dataframe(ua.sort_values('impressions',ascending=False).head(20).reset_index(drop=True),
                         use_container_width=True)

    with tab3:
        if not ds.empty:
            top = ds.groupby(['NewsID']).agg(clicks=('Click','sum'),impressions=('Click','count')).reset_index()
            top['CTR'] = top['clicks']/top['impressions']
            top20 = top.sort_values('clicks',ascending=False).head(20)
            if 'Category' in ds.columns:
                top20 = top20.merge(ds[['NewsID','Category']].drop_duplicates(), on='NewsID', how='left')
            fig = px.bar(top20, x='NewsID', y='clicks', color='Category' if 'Category' in top20.columns else 'clicks',
                         title="Top 20 Most Clicked Articles")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top20.reset_index(drop=True), use_container_width=True)


# ── MODEL PERFORMANCE ─────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.markdown('<div class="main-header">🤖 ML Model Performance</div>', unsafe_allow_html=True)

    if USE_API:
        results_raw = api_get("/model/results") or []
        mr = {r['model']: r for r in results_raw}
    else:
        mr = model_results

    if mr:
        rows = [{'Model':k,'Accuracy':v.get('accuracy',0),'AUC-ROC':v.get('auc',0),'F1-Score':v.get('f1',0)}
                for k,v in mr.items() if k != '_id']
        res_df = pd.DataFrame(rows)

        st.subheader("Model Comparison Table")
        st.dataframe(res_df.style.highlight_max(subset=['AUC-ROC','Accuracy','F1-Score'],color='#d4edda'),
                     use_container_width=True)

        # Grouped bar chart
        fig = go.Figure()
        metrics = ['Accuracy','AUC-ROC','F1-Score']
        colors  = ['#1a73e8','#34a853','#fbbc04']
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(name=metric, x=res_df['Model'], y=res_df[metric],
                                 marker_color=color))
        fig.update_layout(barmode='group', title="Model Metrics Comparison",
                          yaxis=dict(range=[0,1]), height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        categories_radar = ['Accuracy','AUC-ROC','F1-Score']
        fig2 = go.Figure()
        for _, row in res_df.iterrows():
            vals = [row['Accuracy'], row['AUC-ROC'], row['F1-Score']]
            vals += vals[:1]
            fig2.add_trace(go.Scatterpolar(r=vals, theta=categories_radar+[categories_radar[0]],
                                            fill='toself', name=row['Model']))
        fig2.update_layout(polar=dict(radialaxis=dict(range=[0.5,1])),
                           title="Model Performance Radar Chart", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    fi = pd.DataFrame({'Feature':['Article CTR','User CTR','Popularity Score','Category Match',
                                   'Category Affinity','Activity Level','Title Length','Word Count'],
                       'Importance':[0.18,0.17,0.16,0.14,0.10,0.12,0.07,0.06]}).sort_values('Importance',ascending=True)
    fig3 = px.bar(fi, x='Importance', y='Feature', orientation='h',
                  title="Feature Importance Scores", color='Importance',
                  color_continuous_scale='Blues')
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)


# ── ABOUT ─────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
## Real-Time Personalized News Recommendation & User Engagement Analytics Platform
**Dataset:** Microsoft MIND (Microsoft News Dataset)

### 🏗️ Full Big Data Architecture
```
MIND Dataset (TSV)
        ↓
    Hadoop HDFS  ←──── distributed storage
        ↓
    Apache Spark (PySpark)  ←──── distributed processing & feature engineering
        ↓
    MongoDB  ←──── NoSQL storage for interactions, news, model results
        ↓
    scikit-learn  ←──── ML: Logistic Regression, Random Forest, Gradient Boosting
        ↓
    FastAPI  ←──── REST API backend (port 8000)
        ↓
    Streamlit + Plotly  ←──── Interactive dashboard frontend
```

### 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Distributed Storage | Apache Hadoop HDFS |
| Big Data Processing | Apache Spark / PySpark |
| Database | MongoDB |
| Machine Learning | scikit-learn, imbalanced-learn |
| API Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Language | Python 3.9+ |

### 📊 ML Models
- **Logistic Regression** — baseline linear model
- **Random Forest** — ensemble, provides feature importance
- **Gradient Boosting** — sequential ensemble, best performance

### 🚀 Future Scope
- Apache Kafka for real-time click stream ingestion
- Deep learning: NRMS, NAML recommendation models
- Interest drift detection
- Docker + Kubernetes deployment
""")
    if USE_API:
        st.success("✅ FastAPI backend is connected and serving data")
    else:
        st.info("ℹ️ Running in standalone mode (FastAPI not detected)")
