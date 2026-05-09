"""Streamlit dashboard — Grab Sentiment Analysis."""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline as hf_pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grab Sentiment Dashboard",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Grab App Reviews — Sentiment Analysis Dashboard")
st.caption("Singapore market · Google Play + App Store · Built with DistilBERT & BERTopic")

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'grab_reviews_topics.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'distilbert-grab')

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    return df


@st.cache_resource
def load_classifier():
    if os.path.exists(MODEL_PATH):
        return hf_pipeline(
            'text-classification',
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=-1,
            truncation=True,
            max_length=128,
        )
    return None


try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.warning(
        "⚠️ Data not found. Run notebooks 01 → 04 first to generate "
        "`data/grab_reviews_topics.csv`, then refresh this page."
    )

# ── Sidebar filters ───────────────────────────────────────────────────────────
if data_loaded:
    st.sidebar.header("Filters")

    all_topics = sorted(df['topic_label'].dropna().unique())
    selected_topics = st.sidebar.multiselect(
        "Topics", all_topics, default=all_topics
    )

    all_sources = sorted(df['source'].unique())
    selected_sources = st.sidebar.multiselect(
        "Source", all_sources, default=all_sources
    )

    date_min = df['date'].min().date()
    date_max = df['date'].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(date_min, date_max),
        min_value=date_min, max_value=date_max
    )

    df_f = df[
        df['topic_label'].isin(selected_topics)
        & df['source'].isin(selected_sources)
        & (df['date'].dt.date >= date_range[0])
        & (df['date'].dt.date <= date_range[1])
    ]

    # ── KPI row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(df_f):,}")
    pct_neg = (df_f['bert_pred'] == 'negative').mean() * 100
    pct_pos = (df_f['bert_pred'] == 'positive').mean() * 100
    col2.metric("% Negative", f"{pct_neg:.1f}%")
    col3.metric("% Positive", f"{pct_pos:.1f}%")
    avg_star = df_f['rating'].mean()
    col4.metric("Avg Star Rating", f"{avg_star:.2f} ⭐")

    st.divider()

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Sentiment Breakdown by Topic")
        topic_sent = (
            df_f[df_f['topic_label'] != 'Noise / Uncategorised']
            .groupby(['topic_label', 'bert_pred'])
            .size()
            .reset_index(name='count')
        )
        color_map = {'negative': '#e74c3c', 'neutral': '#f39c12', 'positive': '#2ecc71'}
        fig_bar = px.bar(
            topic_sent,
            x='count', y='topic_label', color='bert_pred',
            orientation='h',
            color_discrete_map=color_map,
            labels={'topic_label': 'Topic', 'count': 'Reviews', 'bert_pred': 'Sentiment'},
            category_orders={'bert_pred': ['negative', 'neutral', 'positive']},
        )
        fig_bar.update_layout(barmode='stack', height=420, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.subheader("Overall Sentiment Split")
        sent_counts = df_f['bert_pred'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        fig_pie = px.pie(
            sent_counts, values='Count', names='Sentiment',
            color='Sentiment', color_discrete_map=color_map,
            hole=0.4,
        )
        fig_pie.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Trend chart ───────────────────────────────────────────────────────────
    st.subheader("Sentiment Trend Over Time (per Topic)")

    sentiment_score = {'negative': -1, 'neutral': 0, 'positive': 1}
    df_f = df_f.copy()
    df_f['sent_score'] = df_f['bert_pred'].map(sentiment_score)

    top5_topics = (
        df_f[df_f['topic_label'] != 'Noise / Uncategorised']['topic_label']
        .value_counts().head(6).index.tolist()
    )
    trend_topics = st.multiselect("Show topics", top5_topics, default=top5_topics[:4])

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
                x=ts['date'], y=ts['sent_score'], mode='lines+markers',
                name=topic, line=dict(width=2)
            ))
        fig_trend.add_hline(y=0, line_dash='dash', line_color='grey', opacity=0.4)
        fig_trend.update_layout(
            yaxis=dict(range=[-1, 1], title='Avg Sentiment Score'),
            xaxis_title='Month',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # ── Sample reviews table ──────────────────────────────────────────────────
    st.subheader("Sample Reviews")
    sent_filter = st.radio("Filter by sentiment", ['All', 'negative', 'neutral', 'positive'], horizontal=True)
    sample_df = df_f if sent_filter == 'All' else df_f[df_f['bert_pred'] == sent_filter]
    st.dataframe(
        sample_df[['date', 'rating', 'bert_pred', 'topic_label', 'review_clean']]
        .rename(columns={'review_clean': 'review', 'bert_pred': 'model_sentiment'})
        .sort_values('date', ascending=False)
        .head(50),
        use_container_width=True,
        hide_index=True,
    )

# ── Live classifier ───────────────────────────────────────────────────────────
st.divider()
st.subheader("🔬 Live Sentiment Classifier")
st.caption("Paste any review text and get an instant prediction from the fine-tuned DistilBERT model.")

user_text = st.text_area("Enter review text:", height=120, placeholder="e.g. 'The driver was very rude and the app kept crashing…'")

if st.button("Classify", type="primary") and user_text.strip():
    classifier = load_classifier()
    if classifier is None:
        st.error("Model not found. Run notebook 03 first to train and save the DistilBERT model.")
    else:
        with st.spinner("Classifying…"):
            result = classifier(user_text)[0]
        label = result['label']
        score = result['score']
        color = {'negative': '🔴', 'neutral': '🟡', 'positive': '🟢'}.get(label, '⚪')
        st.markdown(f"### {color} **{label.upper()}** &nbsp;&nbsp; *(confidence: {score:.1%})*")
