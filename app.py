from pathlib import Path
import json
import hashlib
import urllib.request
import html

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="MovieMind",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "moviemind_models.joblib"
LINKS_FILE = ROOT / "data" / "ml-32m" / "links.csv"
POSTER_DIR = ROOT / "data" / "posters"
POSTER_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

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
    st.caption(f"Model loading error: {type(exc).__name__}")
    st.stop()


# ============================================================
# TMDB SECURITY
# ============================================================

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
if not isinstance(TMDB_API_KEY, str):
    TMDB_API_KEY = str(TMDB_API_KEY)
TMDB_API_KEY = TMDB_API_KEY.strip()


# ============================================================
# MOVIE DATA
# ============================================================

links = load_links()
movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce").astype("Int64")
links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
movies = movies.merge(links, on="movieId", how="left", suffixes=("", "_link"))
movies["tmdbId"] = pd.to_numeric(movies["tmdbId"], errors="coerce")

if "year" not in movies.columns:
    movies["year"] = pd.to_numeric(
        movies["title"].astype(str).str.extract(r"\((\d{4})\)\s*$")[0],
        errors="coerce",
    )

if "clean_title" not in movies.columns:
    movies["clean_title"] = (
        movies["title"].astype(str)
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
    )


# ============================================================
# SESSION STATE
# ============================================================

for key, default in {
    "recommendations": None,
    "selected_movie": None,
    "movie_search": "",
    "selected_details": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# TMDB
# ============================================================

@st.cache_data(show_spinner=False, max_entries=2000)
def get_tmdb_details(tmdb_id):
    if not TMDB_API_KEY or tmdb_id is None or pd.isna(tmdb_id):
        return {}
    try:
        movie_id = int(tmdb_id)
        request = urllib.request.Request(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def get_tmdb_poster_url(tmdb_id, size="w780"):
    details = get_tmdb_details(tmdb_id)
    path = details.get("poster_path", "") if details else ""
    return f"https://image.tmdb.org/t/p/{size}{path}" if path else ""


@st.cache_data(show_spinner=False, max_entries=5000)
def download_poster(poster_url):
    if not poster_url:
        return ""
    try:
        filename = hashlib.sha256(poster_url.encode("utf-8")).hexdigest() + ".jpg"
        poster_file = POSTER_DIR / filename
        if poster_file.exists() and poster_file.stat().st_size > 1000:
            return str(poster_file)
        request = urllib.request.Request(
            poster_url,
            headers={"User-Agent": "MovieMind/1.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            image_data = response.read()
        if len(image_data) < 1000:
            return ""
        poster_file.write_bytes(image_data)
        return str(poster_file)
    except Exception:
        return ""


def get_poster_path(tmdb_id):
    return download_poster(get_tmdb_poster_url(tmdb_id, "w780"))


def why_movie(movie, selected):
    genres = str(movie.get("genres", "")).replace("|", ", ")
    content_score = float(movie.get("content_score", 0))
    if content_score >= 0.55:
        base = f"Strong content similarity to {selected}"
    elif content_score >= 0.35:
        base = f"Good content similarity to {selected}"
    else:
        base = "Selected from MovieMind's combined recommendation signals"
    return f"{base}, with {genres} as related genres." if genres else base + "."


# ============================================================
# MATCH SCORE
# ============================================================

def add_match_scores(df):
    if df is None or df.empty:
        return df
    result = df.copy()

    def normalize(series):
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        low, high = float(values.min()), float(values.max())
        if high <= low:
            return pd.Series(0.5, index=values.index, dtype=float)
        return ((values - low) / (high - low)).clip(0, 1)

    base = normalize(result["rank_score"] if "rank_score" in result.columns else result.get("hybrid_score", 0))
    content = normalize(result["content_score"] if "content_score" in result.columns else 0)
    years = pd.to_numeric(result.get("year", pd.Series(float("nan"), index=result.index)), errors="coerce")
    recency = (((years - 1980) / 46).clip(0, 1).fillna(0.45))
    match = 0.70 * base + 0.20 * content + 0.10 * recency
    result["match_score"] = (70 + match * 30).round().clip(70, 99)
    return result


def rerank_recommendations(recommendations, selected_movie, limit):
    if recommendations is None or recommendations.empty:
        return pd.DataFrame()
    df = recommendations.copy()
    if "movieId" in df.columns:
        df = df.drop_duplicates("movieId")
    if "title" in df.columns:
        selected_clean = str(selected_movie).strip().lower()
        df = df[df["title"].astype(str).str.strip().str.lower() != selected_clean]
    if df.empty:
        return df

    for column in ["predicted_rating", "hybrid_score", "content_score"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    def normalize(series):
        low, high = float(series.min()), float(series.max())
        if high <= low:
            return pd.Series(0.5, index=series.index, dtype=float)
        return ((series - low) / (high - low)).clip(0, 1)

    df["_hybrid"] = normalize(df["hybrid_score"])
    df["_content"] = normalize(df["content_score"])
    df["_rating"] = normalize(df["predicted_rating"])
    years = pd.to_numeric(df.get("year", pd.Series(float("nan"), index=df.index)), errors="coerce")
    df["_modernity"] = (1 - ((2026 - years).clip(0, 50) / 50)).fillna(0.35).clip(0, 1)
    df["_base_score"] = 0.52 * df["_hybrid"] + 0.25 * df["_content"] + 0.10 * df["_rating"] + 0.13 * df["_modernity"]

    def parse_genres(value):
        return {p.strip().lower() for p in str(value).split("|") if p.strip() and p.strip().lower() != "(no genres listed)"}

    df["_genre_set"] = df.get("genres", pd.Series("", index=df.index)).apply(parse_genres)
    remaining = df.copy()
    chosen, chosen_genres, chosen_titles = [], [], set()
    target = min(int(limit), len(remaining))

    while len(chosen) < target and not remaining.empty:
        best_index, best_score = None, float("-inf")
        for index, row in remaining.iterrows():
            title = str(row.get("title", "")).strip().lower()
            if not title or title in chosen_titles:
                continue
            movie_genres = row["_genre_set"]
            max_similarity = 0.0
            for previous in chosen_genres:
                union = movie_genres | previous
                similarity = len(movie_genres & previous) / len(union) if union else 0.0
                max_similarity = max(max_similarity, similarity)
            score = 0.82 * float(row["_base_score"]) + 0.18 * (1 - max_similarity)
            if score > best_score:
                best_score, best_index = score, index
        if best_index is None:
            break
        row = remaining.loc[best_index].copy()
        row["rank_score"] = best_score
        chosen.append(row)
        chosen_titles.add(str(row.get("title", "")).strip().lower())
        chosen_genres.append(row["_genre_set"])
        remaining = remaining.drop(index=best_index)

    result = pd.DataFrame(chosen)
    return result.drop(columns=["_hybrid", "_content", "_rating", "_modernity", "_base_score", "_genre_set"], errors="ignore").reset_index(drop=True)


# ============================================================
# DETAILS VIEW
# ============================================================

def show_movie_details(movie):
    tmdb_id = movie.get("tmdbId")
    details = get_tmdb_details(tmdb_id)
    if not details:
        st.warning("Movie details are temporarily unavailable.")
        return

    title = details.get("title") or movie.get("title", "Unknown movie")
    overview = details.get("overview") or "No overview is available for this movie."
    release_date = details.get("release_date", "")
    year = release_date[:4] if release_date else str(movie.get("year", ""))
    vote_average = details.get("vote_average")
    runtime = details.get("runtime")
    tagline = details.get("tagline") or ""
    poster_path = details.get("poster_path")
    homepage = details.get("homepage")
    genres = " · ".join(g.get("name", "") for g in details.get("genres", []) if g.get("name"))
    match_score = int(movie.get("match_score", 0))
    model_rating = float(movie.get("predicted_rating", 0))

    st.markdown('<div class="details-kicker">MOVIEMIND MOVIE PROFILE</div>', unsafe_allow_html=True)
    if st.button("Back to recommendations", key="close_movie_details"):
        st.session_state.selected_details = None
        st.rerun()

    left, right = st.columns([1, 1.85], gap="large")
    with left:
        poster_url = f"https://image.tmdb.org/t/p/w780{poster_path}" if poster_path else ""
        if poster_url:
            try:
                st.image(poster_url, width="stretch")
            except Exception:
                poster_url = ""
        if not poster_url:
            local = get_poster_path(tmdb_id)
            if local:
                st.image(local, width="stretch")
            else:
                st.markdown('<div class="poster-fallback">Poster unavailable</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f'<div class="details-title">{html.escape(str(title))}</div>', unsafe_allow_html=True)
        meta = [str(year) if year else "", html.escape(genres) if genres else "", f"{int(runtime)} min" if runtime else ""]
        meta = [x for x in meta if x]
        if meta:
            st.markdown(f'<div class="details-meta">{" · ".join(meta)}</div>', unsafe_allow_html=True)
        if tagline:
            st.markdown(f'<div class="details-tagline">{html.escape(str(tagline))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="details-score-grid">' + f'<div class="details-score-card primary"><div class="details-score-label">MOVIEMIND MATCH</div><div class="details-score-value">{match_score}%</div></div>' + f'<div class="details-score-card"><div class="details-score-label">MODEL ESTIMATE</div><div class="details-score-value">{model_rating:.2f} / 5</div></div></div>', unsafe_allow_html=True)
        tmdb_rating = f"{float(vote_average):.1f} / 10" if vote_average is not None else "Not available"
        runtime_text = f"{int(runtime)} min" if runtime else "Not available"
        st.markdown(f'<div class="details-facts"><div><span>TMDB RATING</span><strong>{tmdb_rating}</strong></div><div><span>RUNTIME</span><strong>{runtime_text}</strong></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="details-section-title">Why MovieMind recommended it</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-reason">{html.escape(why_movie(movie, st.session_state.selected_movie or "your selection"))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="details-section-title">Overview</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-overview">{html.escape(str(overview))}</div>', unsafe_allow_html=True)
        if homepage:
            st.link_button("Open official movie page", homepage, use_container_width=True)
    st.markdown('<div class="details-attribution">Movie metadata and ratings are supplied by TMDB. MovieMind recommendation scores are generated locally.</div>', unsafe_allow_html=True)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.stApp{background:#f7f9fc;color:#101828}.block-container{max-width:1180px;padding-top:24px;padding-bottom:50px}header[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer{display:none}.stApp h1,.stApp h2,.stApp h3{color:#081a36!important;font-weight:900!important}.stApp p{color:#344054!important;font-weight:550!important}
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{background:#0b3b91!important}section[data-testid="stSidebar"]{border-right:1px solid #082f73}section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#fff!important;font-weight:900!important}section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label{color:#e6efff!important;font-weight:650!important}.stButton>button{background:#1455c0!important;color:#fff!important;border:none!important;border-radius:10px!important;min-height:46px!important;font-weight:900!important}.stButton>button p{color:#fff!important;font-weight:900!important}
[data-testid="stImage"] img{width:100%!important;aspect-ratio:2/3!important;object-fit:cover!important;border-radius:12px!important}.poster-fallback{width:100%;aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;padding:25px;text-align:center;color:#667085;font-size:13px;font-weight:750;background:#edf2f8;border-radius:12px}.details-kicker{color:#155eef;font-size:11px;font-weight:900;letter-spacing:1px;margin:22px 0 10px}.details-title{color:#081a36;font-size:42px;font-weight:900;line-height:1.1;margin-bottom:8px}.details-meta{color:#155eef;font-size:13px;font-weight:800;margin-bottom:14px}.details-tagline{color:#475467;font-size:15px;font-style:italic;font-weight:650;margin:10px 0 16px}.details-score-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:18px 0}.details-score-card{background:#fff;border:1px solid #e1e7f0;border-radius:14px;padding:16px}.details-score-card.primary{background:#f0f6ff;border-color:#cfe0ff}.details-score-label{color:#667085;font-size:10px;font-weight:850;letter-spacing:.7px}.details-score-value{color:#155eef;font-size:29px;font-weight:900;margin-top:4px}.details-facts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0 18px}.details-facts>div{background:#fff;border:1px solid #e1e7f0;border-radius:12px;padding:13px 15px}.details-facts span{display:block;color:#667085;font-size:9px;font-weight:850;letter-spacing:.7px}.details-facts strong{display:block;color:#101828;font-size:17px;font-weight:900;margin-top:3px}.details-section-title{color:#081a36;font-size:13px;font-weight:900;margin-top:18px;margin-bottom:7px}.details-reason{background:#f0f6ff;border:1px solid #d6e5ff;border-radius:12px;padding:14px 16px;color:#344054;font-size:13px;font-weight:600;line-height:1.55}.details-overview{color:#344054;font-size:15px;font-weight:500;line-height:1.7}.details-attribution{color:#667085;font-size:11px;font-weight:600;margin-top:20px;padding-top:12px;border-top:1px solid #e1e7f0}
@media(max-width:700px){.details-title{font-size:31px}.details-score-grid,.details-facts{grid-template-columns:1fr}.block-container{padding-left:14px;padding-right:14px}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("MovieMind")
    st.caption("Personalized Movie Discovery")
    st.divider()
    st.subheader("Discover")
    if st.button("New Discovery", use_container_width=True):
        st.session_state.recommendations = None
        st.session_state.selected_movie = None
        st.session_state.movie_search = ""
        st.session_state.selected_details = None
        st.rerun()


# ============================================================
# DETAIL ROUTE
# ============================================================

if st.session_state.selected_details is not None:
    show_movie_details(st.session_state.selected_details)
    st.stop()


# ============================================================
# MAIN SEARCH
# ============================================================

st.markdown('<div class="details-kicker">AI-POWERED MOVIE DISCOVERY</div>', unsafe_allow_html=True)
st.markdown('<div class="details-title">Find movies you will love to watch.</div>', unsafe_allow_html=True)
st.write("Tell MovieMind what you enjoy and discover personalized recommendations built around your taste.")

movie_titles = movies["title"].astype(str).tolist()
search_value = st.session_state.movie_search
selected = st.selectbox(
    "Choose a movie",
    movie_titles,
    index=movie_titles.index(search_value) if search_value in movie_titles else 0,
)
st.session_state.movie_search = selected

recommendation_count = st.slider("Recommendations", min_value=5, max_value=20, value=10)

if st.button("Find Movies For Me", use_container_width=True):
    st.session_state.selected_movie = selected
    with st.spinner("Finding movies for you..."):
        raw_count = min(max(recommendation_count * 4, 40), 100)
        candidates = recommender.recommend(
            user_id=1,
            movie_title=selected,
            num_recommendations=raw_count,
        )
        recommendations = rerank_recommendations(candidates, selected, recommendation_count)
        st.session_state.recommendations = add_match_scores(recommendations)
    st.rerun()


# ============================================================
# RESULTS
# ============================================================

if st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations
    st.divider()
    st.subheader("Your recommendations")
    st.caption(f"Because you liked {st.session_state.selected_movie}")

    if recommendations.empty:
        st.warning("No recommendations were found. Try another movie.")
    else:
        for start in range(0, len(recommendations), 4):
            row = recommendations.iloc[start:start + 4]
            columns = st.columns(len(row), gap="medium")
            for position, (_, movie) in enumerate(row.iterrows()):
                with columns[position]:
                    title = str(movie.get("title", "Unknown movie"))
                    tmdb_id = movie.get("tmdbId")
                    poster = get_poster_path(tmdb_id)
                    if poster:
                        st.image(poster, width="stretch")
                    else:
                        st.markdown('<div class="poster-fallback">Poster unavailable</div>', unsafe_allow_html=True)
                    st.caption(f"RECOMMENDATION {start + position + 1:02d}")
                    st.markdown(f"**{title}**")
                    year = movie.get("year")
                    if pd.notna(year):
                        st.caption(str(int(year)))
                    genres = str(movie.get("genres", "")).replace("|", " · ")
                    if genres:
                        st.caption(genres)
                    st.markdown("**MOVIEMIND MATCH**")
                    st.markdown(f"### {int(movie.get('match_score', 0))}%")
                    st.caption(f"Model estimate: {float(movie.get('predicted_rating', 0)):.2f} / 5")
                    st.write(why_movie(movie, st.session_state.selected_movie))
                    if st.button("View details", key=f"details_{movie.get('movieId', start + position)}", use_container_width=True):
                        st.session_state.selected_details = movie.to_dict()
                        st.rerun()

st.divider()
st.caption("MovieMind · Personalized Movie Discovery")
st.caption("Created By: Nabin Chettri")
st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")
