# Grab App Sentiment Analysis

> **"What are Grab customers complaining about — and is it getting better or worse?"**

End-to-end NLP pipeline on ~10,000 Grab app reviews from the Google Play Store and Apple App Store (Singapore market). Built as a portfolio project targeting BA / Data Analyst internships.

---

## Project Overview

| Stage | Notebook | Description |
|---|---|---|
| 1 | `01_scraping.ipynb` | Scrape reviews via `google-play-scraper` + `app-store-scraper` |
| 2 | `02_cleaning_eda.ipynb` | Clean text, detect language, EDA plots |
| 3 | `03_sentiment_modelling.ipynb` | VADER baseline → DistilBERT fine-tune, F1 evaluation |
| 4 | `04_topic_modelling.ipynb` | BERTopic themes, sentiment × topic cross-tab, time trends |
| — | `dashboard/app.py` | Streamlit dashboard with live classifier |

---

## Key Findings

_To be filled in after running the full pipeline._

- **Top complaint themes**: TBD
- **Sentiment trend**: TBD (improving / worsening since YYYY)
- **Worst-rated topic**: TBD
- **DistilBERT F1**: TBD vs VADER baseline TBD

---

## Business Recommendations

_To be filled in after analysis._

1. **[Theme]** — Recommended action and expected impact.
2. **[Theme]** — Recommended action and expected impact.
3. **[Theme]** — Recommended action and expected impact.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

```bash
jupyter notebook
```

Open and run:
1. `notebooks/01_scraping.ipynb` — produces `data/grab_reviews_raw.csv`
2. `notebooks/02_cleaning_eda.ipynb` — produces `data/grab_reviews_clean.csv`
3. `notebooks/03_sentiment_modelling.ipynb` — trains and saves model to `models/`
4. `notebooks/04_topic_modelling.ipynb` — produces `data/grab_reviews_topics.csv`

### 3. Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## Tech Stack

- **Scraping**: `google-play-scraper`, `app-store-scraper`
- **NLP**: `vaderSentiment`, HuggingFace `transformers` (DistilBERT), `BERTopic`
- **Data**: `pandas`, `scikit-learn`
- **Visualisation**: `matplotlib`, `seaborn`, `wordcloud`, `plotly`
- **Dashboard**: `Streamlit`

---

## Project Structure

```
grab-sentiment-analysis/
├── data/                        # Raw + cleaned CSVs (git-ignored)
├── notebooks/
│   ├── 01_scraping.ipynb
│   ├── 02_cleaning_eda.ipynb
│   ├── 03_sentiment_modelling.ipynb
│   └── 04_topic_modelling.ipynb
├── dashboard/
│   └── app.py
├── models/                      # Saved checkpoints (git-ignored)
├── requirements.txt
└── README.md
```
