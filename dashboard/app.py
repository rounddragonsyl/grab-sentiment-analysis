"""Streamlit dashboard — Grab Sentiment Analysis."""
import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Grab Sentiment Dashboard",
    page_icon="🚗",
    layout="wide",
)

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'grab_reviews_topics.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf-grab', 'tfidf_lr.joblib')

SENTIMENT_COLORS = {'negative': '#e74c3c', 'neutral': '#f39c12', 'positive': '#2ecc71'}
LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    return df


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

st.title("Grab App Reviews — Sentiment Analysis Dashboard")
st.caption("Singapore market · Google Play + App Store · TF-IDF + Logistic Regression · LDA Topics")

if not data_loaded:
    st.warning("Data not found. Run notebooks 01 → 04 first to generate `data/grab_reviews_topics.csv`.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

all_topics = sorted(df['topic_label'].dropna().unique())
selected_topics = st.sidebar.multiselect("Topics", all_topics, default=all_topics)

all_sources = sorted(df['source'].unique())
selected_sources = st.sidebar.multiselect("Source", all_sources, default=all_sources)

date_min = df['date'].min().date()
date_max = df['date'].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(date_min, date_max),
    min_value=date_min, max_value=date_max,
)

df_f = df[
    df['topic_label'].isin(selected_topics)
    & df['source'].isin(selected_sources)
    & (df['date'].dt.date >= date_range[0])
    & (df['date'].dt.date <= date_range[1])
].copy()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews", f"{len(df_f):,}")
k2.metric("% Negative", f"{(df_f['model_pred'] == 'negative').mean() * 100:.1f}%")
k3.metric("% Positive", f"{(df_f['model_pred'] == 'positive').mean() * 100:.1f}%")
k4.metric("Avg Star Rating", f"{df_f['rating'].mean():.2f} ⭐")

st.divider()

# ── Row 1: sentiment by topic + pie ───────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Sentiment by Topic")
    topic_sent = (
        df_f.groupby(['topic_label', 'model_pred'])
        .size()
        .reset_index(name='count')
    )
    fig_bar = px.bar(
        topic_sent,
        x='count', y='topic_label', color='model_pred',
        orientation='h',
        color_discrete_map=SENTIMENT_COLORS,
        labels={'topic_label': 'Topic', 'count': 'Reviews', 'model_pred': 'Sentiment'},
        category_orders={'model_pred': ['negative', 'neutral', 'positive']},
    )
    fig_bar.update_layout(barmode='stack', height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("Overall Sentiment Split")
    sent_counts = df_f['model_pred'].value_counts().reset_index()
    sent_counts.columns = ['Sentiment', 'Count']
    fig_pie = px.pie(
        sent_counts, values='Count', names='Sentiment',
        color='Sentiment', color_discrete_map=SENTIMENT_COLORS,
        hole=0.4,
    )
    fig_pie.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── Row 2: rating distribution + volume over time ─────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.subheader("Rating Distribution")
    rating_counts = df_f['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    fig_rating = px.bar(
        rating_counts, x='Rating', y='Count',
        color='Rating',
        color_continuous_scale=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60'],
        labels={'Rating': 'Star Rating', 'Count': 'Reviews'},
    )
    fig_rating.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig_rating, use_container_width=True)

with c4:
    st.subheader("Review Volume Over Time")
    vol = (
        df_f.set_index('date')
        .resample('ME')
        .size()
        .reset_index(name='count')
    )
    fig_vol = px.line(vol, x='date', y='count', markers=True,
                      labels={'date': 'Month', 'count': 'Reviews'})
    fig_vol.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# ── Sentiment trend over time ─────────────────────────────────────────────────
st.subheader("Sentiment Trend Over Time (per Topic)")

score_map = {'negative': -1, 'neutral': 0, 'positive': 1}
df_f['sent_score'] = df_f['model_pred'].map(score_map)

top_topics_by_vol = df_f['topic_label'].value_counts().head(6).index.tolist()
trend_topics = st.multiselect("Topics to show", top_topics_by_vol, default=top_topics_by_vol[:4])

if trend_topics:
    fig_trend = go.Figure()
    for topic in trend_topics:
        ts = (
            df_f[df_f['topic_label'] == topic]
            .set_index('date')
            .resample('ME')['sent_score']
            .mean()
            .rolling(3, min_periods=1)
            .mean()
            .reset_index()
        )
        fig_trend.add_trace(go.Scatter(
            x=ts['date'], y=ts['sent_score'],
            mode='lines+markers', name=topic, line=dict(width=2),
        ))
    fig_trend.add_hline(y=0, line_dash='dash', line_color='grey', opacity=0.4)
    fig_trend.update_layout(
        yaxis=dict(range=[-1, 1], title='Avg Sentiment (-1 → +1)'),
        xaxis_title='Month', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ── Review explorer ───────────────────────────────────────────────────────────
st.subheader("Review Explorer")

col_s, col_t = st.columns(2)
with col_s:
    sent_filter = st.radio("Sentiment", ['All', 'negative', 'positive'], horizontal=True)
with col_t:
    topic_filter = st.selectbox("Topic", ['All'] + all_topics)

sample = df_f.copy()
if sent_filter != 'All':
    sample = sample[sample['model_pred'] == sent_filter]
if topic_filter != 'All':
    sample = sample[sample['topic_label'] == topic_filter]

search = st.text_input("Search reviews", placeholder="e.g. refund, driver, crash")
if search:
    sample = sample[sample['review_clean'].str.contains(search, case=False, na=False)]

st.dataframe(
    sample[['date', 'rating', 'source', 'model_pred', 'vader_pred', 'topic_label', 'review_clean']]
    .rename(columns={'review_clean': 'review', 'model_pred': 'model_sentiment', 'vader_pred': 'vader_sentiment'})
    .sort_values('date', ascending=False)
    .head(100),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Live classifier ───────────────────────────────────────────────────────────
st.subheader("Live Sentiment Classifier")
st.caption("Paste any review text and get an instant prediction from the TF-IDF + Logistic Regression model.")

user_text = st.text_area("Review text:", height=120,
                          placeholder="e.g. 'The driver was very rude and the app kept crashing…'")

if st.button("Classify", type="primary") and user_text.strip():
    model = load_model()
    if model is None:
        st.error("Model not found. Run notebook 03 to train and save the model.")
    else:
        pred_id = model.predict([user_text])[0]
        proba   = model.predict_proba([user_text])[0]
        label   = LABEL_MAP[pred_id]
        conf    = proba[pred_id]
        icon    = {'negative': '🔴', 'neutral': '🟡', 'positive': '🟢'}[label]
        st.markdown(f"### {icon} **{label.upper()}** &nbsp;&nbsp; *(confidence: {conf:.1%})*")
        c_n, c_u, c_p = st.columns(3)
        c_n.metric("Negative", f"{proba[0]:.1%}")
        c_u.metric("Neutral",  f"{proba[1]:.1%}")
        c_p.metric("Positive", f"{proba[2]:.1%}")
