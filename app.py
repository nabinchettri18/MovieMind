from pathlib import Path
import json
import hashlib
import urllib.request

import joblib
import pandas as pd
import streamlit as st

from src.tmdb_service import TMDBService


st.set_page_config(
    page_title="MovieMind",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    for col in ["movieId", "imdbId", "tmdbId"]:
        if col not in links.columns:
            links[col] = pd.NA
    return links[["movieId", "imdbId", "tmdbId"]]


if not MODEL_FILE.exists():
    st.title("MovieMind")
    st.caption("Personalized Movie Discovery")
    st.warning("MovieMind is not ready yet.")
    st.stop()

try:
    saved = load_engine()
    movies = saved["movies"].copy()
    ratings = saved["ratings"].copy()
    recommender = saved["hybrid_model"]
except Exception:
    st.title("MovieMind")
    st.caption("Personalized Movie Discovery")
    st.error("MovieMind could not load the recommendation engine.")
    st.stop()

TMDB_API_KEY = str(st.secrets.get("TMDB_API_KEY", "")).strip()
tmdb = TMDBService(TMDB_API_KEY, POSTER_DIR)

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


@st.cache_data(show_spinner=False, max_entries=5000)
def get_poster_url(tmdb_id, api_key):
    if not api_key or tmdb_id is None or pd.isna(tmdb_id):
        return ""
    details = tmdb.movie_details(int(tmdb_id))
    return tmdb.image_url(details.get("poster_path", ""), "w500")


@st.cache_data(show_spinner=False, max_entries=5000)
def get_local_poster(tmdb_id, api_key):
    if not api_key or tmdb_id is None or pd.isna(tmdb_id):
        return ""
    details = tmdb.movie_details(int(tmdb_id))
    return tmdb.local_poster(details.get("poster_path", ""))


if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "movie_search" not in st.session_state:
    st.session_state.movie_search = ""
if "details_movie_id" not in st.session_state:
    st.session_state.details_movie_id = None


st.markdown("""
<style>
.stApp{background:#f7f9fc;color:#101828}.block-container{max-width:1180px;padding-top:24px;padding-bottom:50px}
header[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer{display:none}
.stApp h1,.stApp h2,.stApp h3{color:#081a36!important;font-weight:900!important}.stApp p{color:#344054!important;font-weight:550!important}
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{background:#0b3b91!important}section[data-testid="stSidebar"]{border-right:1px solid #082f73}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#fff!important;font-weight:900!important}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label{color:#e6efff!important;font-weight:650!important}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.18)}
section[data-testid="stSidebar"] .stButton>button{background:rgba(255,255,255,.08)!important;color:#fff!important;border:1px solid rgba(255,255,255,.14)!important;border-radius:10px!important;min-height:42px!important;font-weight:800!important}
section[data-testid="stSidebar"] .stButton>button p{color:#fff!important;font-weight:800!important}
.hero-kicker{color:#155eef;font-size:12px;font-weight:900;letter-spacing:1.25px;margin-top:30px;margin-bottom:8px}.hero-title{color:#081a36;font-size:50px;font-weight:900;line-height:1.05;letter-spacing:-2px;max-width:850px;margin-bottom:12px}.hero-description{color:#344054;font-size:16px;font-weight:550;line-height:1.65;max-width:680px}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff!important;border:1px solid #e1e7f0!important;border-radius:16px!important;box-shadow:0 6px 22px rgba(23,43,77,.045)}
div[data-testid="stTextInput"] label,div[data-testid="stSelectbox"] label,div[data-testid="stNumberInput"] label,div[data-testid="stSlider"] label{color:#182230!important;font-weight:850!important}
div[data-testid="stTextInput"] input{background:#fff!important;color:#101828!important;border:1px solid #cbd5e1!important;border-radius:10px!important;min-height:48px!important;font-size:15px!important;font-weight:700!important}
div[data-baseweb="select"]>div{background:#fff!important;border:1px solid #cbd5e1!important;border-radius:10px!important}div[data-baseweb="select"] *{color:#172033!important;font-weight:700!important}
div[data-testid="stNumberInput"] input{background:#fff!important;color:#101828!important;font-weight:750!important}
.stButton>button{background:#1455c0!important;color:#fff!important;border:none!important;border-radius:10px!important;min-height:46px!important;font-weight:900!important}.stButton>button p{color:#fff!important;font-weight:900!important}.stButton>button:hover{background:#0b3b91!important}
[data-testid="stImage"] img{width:100%!important;aspect-ratio:2/3!important;object-fit:cover!important;object-position:center!important;border-radius:12px!important;display:block!important}.poster-fallback{width:100%;aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;padding:25px;text-align:center;color:#667085;font-size:13px;font-weight:750;line-height:1.45;background:#edf2f8;border-radius:12px}
.movie-number{color:#155eef;font-size:10px;font-weight:900;letter-spacing:.8px;text-transform:uppercase;margin-top:14px;margin-bottom:5px}.movie-title{color:#081a36;font-size:18px;font-weight:900;line-height:1.28;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;min-height:46px}.movie-year{color:#155eef;font-size:12px;font-weight:850;margin-top:6px}.movie-genres{color:#475467;font-size:12px;font-weight:600;line-height:1.45;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;min-height:35px;margin-top:10px}.movie-score-label{color:#667085;font-size:10px;font-weight:750;text-transform:uppercase;letter-spacing:.55px;margin-top:15px}.movie-score{color:#101828;font-size:19px;font-weight:900;margin-top:2px}.why-label{color:#155eef;font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:.6px;margin-top:14px}.why-text{color:#475467;font-size:11px;font-weight:550;line-height:1.45;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;margin-top:4px}
.details-title{color:#081a36;font-size:38px;font-weight:900;line-height:1.12;margin-bottom:6px}.details-meta{color:#155eef;font-size:13px;font-weight:800;margin-bottom:12px}.details-overview{color:#344054;font-size:15px;font-weight:500;line-height:1.7}.details-label{color:#667085;font-size:10px;font-weight:850;letter-spacing:.7px;text-transform:uppercase;margin-top:18px}.details-value{color:#101828;font-size:18px;font-weight:900;margin-top:3px}
@media(max-width:900px){.hero-title{font-size:42px}.details-title{font-size:32px}}@media(max-width:640px){.block-container{padding-left:14px;padding-right:14px}.hero-kicker{margin-top:18px}.hero-title{font-size:36px;letter-spacing:-1px}.hero-description{font-size:15px}.movie-title{font-size:20px}.movie-genres{font-size:13px}.movie-score{font-size:20px}.details-title{font-size:28px}}
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
        st.session_state.details_movie_id = None
    st.divider()
    st.subheader("Quick Picks")
    quick_picks = [
        "Toy Story (1995)",
        "Forrest Gump (1994)",
        "Inception (2010)",
        "Interstellar (2014)",
        "Dune (2021)",
    ]
    available_movies = set(movies["title"].astype(str))
    for title in quick_picks:
        if title in available_movies:
            if st.button(title, use_container_width=True):
                st.session_state.selected_movie = title
                st.session_state.movie_search = title
                st.session_state.recommendations = None
                st.session_state.details_movie_id = None
    st.divider()
    st.subheader("Explore")
    st.caption("Drama")
    st.caption("Comedy")
    st.caption("Action")
    st.caption("Romance")
    st.caption("Horror")
    st.caption("Science Fiction")
    st.caption("Animation")
    st.divider()
    st.caption("Personalized movie discovery.")
    st.divider()
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")


# ============================================================
# HERO
# ============================================================
left, right = st.columns([5, 1], vertical_alignment="center")
with left:
    st.caption("MovieMind")
with right:
    st.success("ONLINE")
st.markdown('<div class="hero-kicker">AI-POWERED MOVIE DISCOVERY</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Find movies you\'ll love to watch.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-description">Tell MovieMind what you enjoy and discover personalized recommendations built around your taste.</div>', unsafe_allow_html=True)
st.info("Personalized movie recommendations")


# ============================================================
# DETAIL VIEW
# ============================================================

detail_id = st.session_state.details_movie_id
if detail_id is not None:
    details = tmdb.movie_details(detail_id)
    local_poster = tmdb.local_poster(details.get("poster_path", ""))

    if st.button("Back to recommendations"):
        st.session_state.details_movie_id = None
        st.rerun()

    if details:
        detail_left, detail_right = st.columns([1, 2], gap="large")
        with detail_left:
            if local_poster:
                st.image(local_poster, width="stretch")
            else:
                st.markdown('<div class="poster-fallback">Poster unavailable</div>', unsafe_allow_html=True)
        with detail_right:
            title = details.get("title", "Movie")
            release = details.get("release_date", "")
            year = release[:4] if release else ""
            genres = " · ".join(details.get("genres", []))
            runtime = details.get("runtime")
            vote = details.get("vote_average")

            st.markdown(f'<div class="details-title">{title}</div>', unsafe_allow_html=True)
            if year:
                st.markdown(f'<div class="details-meta">{year}</div>', unsafe_allow_html=True)
            if genres:
                st.markdown(f'<div class="details-meta">{genres}</div>', unsafe_allow_html=True)
            if details.get("tagline"):
                st.markdown(f'**{details["tagline"]}**')

            metric_a, metric_b = st.columns(2)
            with metric_a:
                st.markdown('<div class="details-label">TMDB rating</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="details-value">{float(vote):.1f} / 10</div>' if vote is not None else '<div class="details-value">Not available</div>', unsafe_allow_html=True)
            with metric_b:
                st.markdown('<div class="details-label">Runtime</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="details-value">{int(runtime)} min</div>' if runtime else '<div class="details-value">Not available</div>', unsafe_allow_html=True)

            st.markdown('<div class="details-label">Overview</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="details-overview">{details.get("overview", "No overview available.")}</div>', unsafe_allow_html=True)

            if details.get("homepage"):
                st.link_button("Open official movie page", details["homepage"], use_container_width=True)

        st.divider()
        st.subheader("MovieMind")
        st.caption("This title is part of your selected recommendation context. Return to recommendations to explore more movies.")
    else:
        st.warning("Movie details are unavailable right now.")

    st.stop()


# ============================================================
# SEARCH PANEL
# ============================================================
with st.container(border=True):
    st.subheader("Tell MovieMind what you like")
    st.caption("Search the MovieMind catalog and choose a movie you already love.")
    search_col, year_col = st.columns([3, 1], gap="medium")
    with search_col:
        search_text = st.text_input("Search for a movie", value=st.session_state.movie_search, placeholder="Try Interstellar, Inception, Dune, Oppenheimer...")
        st.session_state.movie_search = search_text
    with year_col:
        year_filter = st.selectbox("Release year", ["All", "2020+", "2015+", "2010+", "2000+", "1990+"])

filtered_movies = movies.copy()
query = search_text.strip().lower()
if query:
    title_match = filtered_movies["title"].astype(str).str.lower().str.contains(query, regex=False, na=False)
    clean_match = filtered_movies["clean_title"].astype(str).str.lower().str.contains(query, regex=False, na=False)
    filtered_movies = filtered_movies[title_match | clean_match]
if year_filter != "All":
    minimum_year = int(year_filter.replace("+", ""))
    filtered_movies = filtered_movies[filtered_movies["year"].notna() & (filtered_movies["year"].astype(int) >= minimum_year)]
if query:
    filtered_movies = filtered_movies.sort_values(["year", "title"], ascending=[False, True], na_position="last").head(100)

selected_movie = None
if query:
    if filtered_movies.empty:
        st.warning("No movies found. Try another title or year.")
    else:
        st.caption(f"{len(filtered_movies):,} matching movies")
        options = filtered_movies["title"].astype(str).tolist()
        current = st.session_state.selected_movie
        if current not in options:
            current = options[0]
        selected_movie = st.selectbox("Movie you like", options, index=options.index(current))
        st.session_state.selected_movie = selected_movie
else:
    st.info("Start typing a movie title above to search the catalog.")

if selected_movie:
    user_col, count_col = st.columns([1, 1], gap="medium")
    with user_col:
        user_id = st.number_input("User ID", min_value=1, max_value=int(ratings["userId"].max()), value=1, step=1)
    with count_col:
        recommendation_count = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    find_movies = st.button("Find Movies For Me", use_container_width=True)
else:
    find_movies = False


# ============================================================
# SECOND-STAGE RANKING
# ============================================================

def rerank_recommendations(recommendations, selected_movie, limit):
    if recommendations is None or recommendations.empty:
        return pd.DataFrame()

    df = recommendations.copy()
    if "movieId" in df.columns:
        df = df.drop_duplicates(subset=["movieId"], keep="first")

    selected_clean = str(selected_movie).strip().lower()
    if "title" in df.columns:
        df = df[df["title"].astype(str).str.strip().str.lower() != selected_clean]

    for col in ["predicted_rating", "hybrid_score", "content_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "year" not in df.columns:
        df["year"] = pd.to_numeric(df["title"].astype(str).str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    def norm(series):
        low = float(series.min())
        high = float(series.max())
        if high <= low:
            return pd.Series(0.5, index=series.index)
        return ((series - low) / (high - low)).clip(0, 1)

    h = norm(df["hybrid_score"])
    c = norm(df["content_score"])
    r = norm(df["predicted_rating"])
    age = (2026 - df["year"]).clip(lower=0, upper=50)
    modern = (1 - age / 50).fillna(0.35).clip(0, 1)
    df["_base"] = 0.52 * h + 0.25 * c + 0.10 * r + 0.13 * modern

    def genres(value):
        return {x.strip().lower() for x in str(value).split("|") if x.strip() and x.strip().lower() != "(no genres listed)"}

    df["_genres"] = df.get("genres", pd.Series("", index=df.index)).apply(genres)

    chosen = []
    chosen_sets = []
    remaining = df.copy()

    while len(chosen) < min(limit, len(df)) and not remaining.empty:
        best_idx = None
        best_score = float("-inf")

        for idx, row in remaining.iterrows():
            current_genres = row["_genres"]
            max_overlap = 0.0
            for previous in chosen_sets:
                union = current_genres | previous
                inter = current_genres & previous
                similarity = len(inter) / len(union) if union else 0.0
                max_overlap = max(max_overlap, similarity)
            diversity = 1.0 - max_overlap
            score = 0.82 * float(row["_base"]) + 0.18 * diversity
            if score > best_score:
                best_score = score
                best_idx = idx

        row = remaining.loc[best_idx].copy()
        row["rank_score"] = best_score
        chosen.append(row)
        chosen_sets.append(row["_genres"])
        remaining = remaining.drop(index=best_idx)

    if not chosen:
        return pd.DataFrame()

    result = pd.DataFrame(chosen).drop(columns=["_base", "_genres"], errors="ignore")
    return result.reset_index(drop=True)


# ============================================================
# RECOMMENDATIONS
# ============================================================

if find_movies:
    try:
        with st.spinner("Finding movies for you..."):
            raw_count = min(max(int(recommendation_count) * 4, 40), 100)
            candidates = recommender.recommend(
                user_id=int(user_id),
                movie_title=selected_movie,
                num_recommendations=raw_count,
            )
            st.session_state.recommendations = rerank_recommendations(
                candidates,
                selected_movie,
                int(recommendation_count),
            )
    except Exception:
        st.session_state.recommendations = None
        st.warning("We couldn't find recommendations right now. Please try another movie.")


# ============================================================
# RESULTS
# ============================================================

if st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations.copy()

    if not recommendations.empty:
        recommendations["movieId"] = pd.to_numeric(recommendations["movieId"], errors="coerce").astype("Int64")
        recommendations = recommendations.merge(
            movies[["movieId", "tmdbId", "year"]],
            on="movieId",
            how="left",
            suffixes=("", "_catalog"),
        )

    st.divider()
    st.subheader("Your recommendations")
    st.caption(f"Because you liked {selected_movie}")

    if recommendations.empty:
        st.info("No recommendations were found.")
    else:
        for start in range(0, len(recommendations), 3):
            row = recommendations.iloc[start:start + 3]
            columns = st.columns(3, gap="medium")

            for position, (column, (_, movie)) in enumerate(zip(columns, row.iterrows())):
                with column:
                    title = str(movie["title"])
                    year = movie.get("year_catalog")
                    if year is None or pd.isna(year):
                        year = movie.get("year")
                    year_text = "" if year is None or pd.isna(year) else str(int(year))
                    genres_text = str(movie.get("genres", "")).replace("|", " · ")
                    rating = float(movie.get("predicted_rating", 0))
                    match_score = int(round(float(movie.get("rank_score", 0)) * 100))
                    match_score = max(1, min(99, match_score))
                    poster_path = get_local_poster(movie.get("tmdbId"), TMDB_API_KEY)

                    if poster_path:
                        try:
                            st.image(poster_path, width="stretch")
                        except Exception:
                            st.markdown('<div class="poster-fallback">Poster unavailable</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="poster-fallback">Poster unavailable</div>', unsafe_allow_html=True)

                    st.markdown(f'<div class="movie-number">RECOMMENDATION {start + position + 1:02d}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
                    if year_text:
                        st.markdown(f'<div class="movie-year">{year_text}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-genres">{genres_text}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="movie-score-label">MOVIEMIND MATCH</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-score">{match_score}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="movie-score-label">MODEL ESTIMATE</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-score" style="font-size:15px;">{rating:.2f} / 5</div>', unsafe_allow_html=True)
                    st.markdown('<div class="why-label">WHY THIS MOVIE</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="why-text">Recommended from content, collaborative and diversity signals related to {selected_movie}.</div>', unsafe_allow_html=True)
                    if st.button("View details", key=f"details_{int(movie['movieId'])}", use_container_width=True):
                        st.session_state.details_movie_id = movie.get("tmdbId")
                        st.rerun()


st.divider()
st.caption("MovieMind  •  Personalized Movie Discovery")
''