from pathlib import Path
import json
import hashlib
import urllib.request
import html

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MovieMind", page_icon="", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "moviemind_models.joblib"
LINKS_FILE = ROOT / "data" / "ml-32m" / "links.csv"
POSTER_DIR = ROOT / "data" / "posters"
POSTER_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_engine():
    return joblib.load(MODEL_FILE)

@st.cache_data(show_spinner=False)
def load_links():
    if not LINKS_FILE.exists():
        return pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
    links = pd.read_csv(LINKS_FILE)
    for column in ["movieId", "imdbId", "tmdbId"]:
        if column not in links.columns:
            links[column] = pd.NA
    return links[["movieId", "imdbId", "tmdbId"]]

if not MODEL_FILE.exists():
    st.title("MovieMind")
    st.caption("Personalized Movie Discovery")
    st.warning("MovieMind is not ready yet.")
    st.stop()

try:
    saved = load_engine()
    movies = saved["movies"].copy()
    recommender = saved["hybrid_model"]
except Exception as exc:
    st.title("MovieMind")
    st.caption("Personalized Movie Discovery")
    st.error("MovieMind could not load the recommendation engine.")
    # Safe diagnostic: expose only the exception type and a short message.
    # Never display secrets, environment variables, or model contents.
    st.caption(f"Model loading error: {type(exc).__name__}")
    st.code(str(exc)[:500] if str(exc) else "No additional error message was provided.")
    st.stop()

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
if not isinstance(TMDB_API_KEY, str):
    TMDB_API_KEY = str(TMDB_API_KEY)
TMDB_API_KEY = TMDB_API_KEY.strip()

links = load_links()
movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce").astype("Int64")
links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
movies = movies.merge(links, on="movieId", how="left", suffixes=("", "_link"))
movies["tmdbId"] = pd.to_numeric(movies["tmdbId"], errors="coerce")

if "year" not in movies.columns:
    movies["year"] = pd.to_numeric(movies["title"].astype(str).str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
if "clean_title" not in movies.columns:
    movies["clean_title"] = movies["title"].astype(str).str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()

# The remainder of the existing MovieMind application follows unchanged.
