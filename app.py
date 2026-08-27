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
    ratings = saved.get("ratings", pd.DataFrame())
    recommender = saved["hybrid_model"]
except Exception as exc:
    st.title("MovieMind")
    st.caption("Personalized Movie Discovery")
    st.error("MovieMind could not load the recommendation engine.")
    st.caption(f"Model loading error: {type(exc).__name__}")
    st.code(str(exc)[:500] if str(exc) else "No additional error message was provided.")
    st.stop()

TMDB_API_KEY = str(st.secrets.get("TMDB_API_KEY", "")).strip()

links = load_links()
movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce").astype("Int64")
links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
movies = movies.merge(links, on="movieId", how="left", suffixes=("", "_link"))
movies["tmdbId"] = pd.to_numeric(movies["tmdbId"], errors="coerce")

if "year" not in movies.columns:
    movies["year"] = pd.to_numeric(movies["title"].astype(str).str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
if "clean_title" not in movies.columns:
    movies["clean_title"] = movies["title"].astype(str).str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()

@st.cache_data(show_spinner=False, max_entries=5000)
def get_tmdb_poster_url(tmdb_id):
    if not TMDB_API_KEY or tmdb_id is None or pd.isna(tmdb_id):
        return ""
    try:
        movie_id = int(tmdb_id)
        request = urllib.request.Request(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        poster_path = data.get("poster_path")
        return "https://image.tmdb.org/t/p/w500" + poster_path if poster_path else ""
    except Exception:
        return ""

@st.cache_data(show_spinner=False, max_entries=5000)
def download_poster(poster_url):
    if not poster_url:
        return ""
    try:
        filename = hashlib.sha256(poster_url.encode("utf-8")).hexdigest() + ".jpg"
        poster_file = POSTER_DIR / filename
        if poster_file.exists() and poster_file.stat().st_size > 1000:
            return str(poster_file)
        request = urllib.request.Request(poster_url, headers={"User-Agent": "MovieMind/1.0"})
        with urllib.request.urlopen(request, timeout=10) as response:
            image_data = response.read()
        if len(image_data) < 1000:
            return ""
        poster_file.write_bytes(image_data)
        return str(poster_file)
    except Exception:
        return ""

def get_poster_path(tmdb_id):
    return download_poster(get_tmdb_poster_url(tmdb_id))

@st.cache_data(show_spinner=False, max_entries=2000)
def get_tmdb_details(tmdb_id):
    if not TMDB_API_KEY or tmdb_id is None or pd.isna(tmdb_id):
        return {}
    try:
        movie_id = int(tmdb_id)
        request = urllib.request.Request(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}

def parse_genres(value):
    return {p.strip().lower() for p in str(value).split("|") if p.strip() and p.strip().lower() != "(no genres listed)"}

def why_movie(movie, selected):
    genres = str(movie.get("genres", "")).replace("|", ", ")
    content_score = float(movie.get("content_score", 0) or 0)
    year = movie.get("year_catalog")
    if year is None or pd.isna(year):
        year = movie.get("year")
    if content_score >= 0.55:
        base = f"Strong content similarity to {selected}"
    elif content_score >= 0.35:
        base = f"Good content similarity to {selected}"
    else:
        base = "Selected from MovieMind's combined recommendation signals"
    if genres:
        base += f", with {genres} as related genres."
    else:
        base += "."
    if year is not None and not pd.isna(year) and int(year) >= 2020:
        base += " Recent-release preference also helped."
    return base

def normalize(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return pd.Series(0.5, index=values.index, dtype=float)
    return ((values - low) / (high - low)).clip(0, 1)

def rerank_recommendations(recommendations, selected_movie, limit):
    if recommendations is None or recommendations.empty:
        return pd.DataFrame() if recommendations is None else recommendations.copy()
    df = recommendations.drop_duplicates(subset=["movieId"], keep="first").copy()
    selected_clean = str(selected_movie).strip().lower()
    df = df[df["title"].astype(str).str.strip().str.lower() != selected_clean].copy()
    if df.empty:
        return df
    for column in ["predicted_rating", "hybrid_score", "content_score"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if "year" not in df.columns:
        df["year"] = pd.to_numeric(df["title"].astype(str).str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["_hybrid"] = normalize(df["hybrid_score"])
    df["_content"] = normalize(df["content_score"])
    df["_rating"] = normalize(df["predicted_rating"])
    age = (2026 - df["year"]).clip(lower=0, upper=50)
    df["_modernity"] = (1.0 - age / 50.0).fillna(0.35).clip(0, 1)
    df["_base_score"] = 0.52 * df["_hybrid"] + 0.25 * df["_content"] + 0.10 * df["_rating"] + 0.13 * df["_modernity"]
    df["_genre_set"] = df.get("genres", pd.Series("", index=df.index)).apply(parse_genres)
    remaining, chosen_rows, chosen_genres, chosen_titles = df.copy(), [], [], set()
    target = min(int(limit), len(remaining))
    while len(chosen_rows) < target and not remaining.empty:
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
            score = 0.82 * float(row["_base_score"]) + 0.18 * (1.0 - max_similarity)
            if score > best_score:
                best_score, best_index = score, index
        if best_index is None:
            break
        selected_row = remaining.loc[best_index].copy()
        selected_row["rank_score"] = best_score
        chosen_rows.append(selected_row)
        chosen_titles.add(str(selected_row.get("title", "")).strip().lower())
        chosen_genres.append(selected_row["_genre_set"])
        remaining = remaining.drop(index=best_index)
    result = pd.DataFrame(chosen_rows) if chosen_rows else df.head(limit)
    return result.drop(columns=["_hybrid", "_content", "_rating", "_modernity", "_base_score", "_genre_set"], errors="ignore").reset_index(drop=True)

def add_match_scores(df):
    if df is None or df.empty:
        return df
    result = df.copy()
    base = normalize(result["rank_score"] if "rank_score" in result else result.get("hybrid_score", pd.Series(0.5, index=result.index)))
    content = normalize(result["content_score"] if "content_score" in result else pd.Series(0.5, index=result.index))
    years = pd.to_numeric(result.get("year_catalog", result.get("year", pd.Series(float("nan"), index=result.index))), errors="coerce")
    recency = ((years - 1980) / 46).clip(0, 1).fillna(0.45)
    result["match_score"] = (70 + (0.70 * base + 0.20 * content + 0.10 * recency) * 30).round().clip(70, 99)
    return result

def show_movie_details(movie):
    details = get_tmdb_details(movie.get("tmdbId"))
    if not details:
        st.warning("Movie details are temporarily unavailable.")
        return
    title = details.get("title") or movie.get("title", "Unknown movie")
    overview = details.get("overview") or "No overview is available for this movie."
    release_date = details.get("release_date", "")
    year = release_date[:4] if release_date else ""
    vote_average = details.get("vote_average")
    runtime = details.get("runtime")
    tagline = details.get("tagline") or ""
    poster_path = details.get("poster_path")
    homepage = details.get("homepage")
    poster_url = "https://image.tmdb.org/t/p/w780" + poster_path if poster_path else ""
    genres = " · ".join(g.get("name", "") for g in details.get("genres", []) if g.get("name"))
    match_score = int(movie.get("match_score", 0))
    model_rating = float(movie.get("predicted_rating", 0))
    st.markdown('<div class="details-kicker">MOVIEMIND MOVIE PROFILE</div>', unsafe_allow_html=True)
    if st.button("Back to recommendations", key="close_movie_details"):
        st.session_state.selected_details = None
        st.rerun()
    left, right = st.columns([1, 1.85], gap="large")
    with left:
        if poster_url:
            st.image(poster_url, width="stretch")
        else:
            local = get_poster_path(movie.get("tmdbId"))
            if local: st.image(local, width="stretch")
            else: st.markdown('<div class="poster-fallback details-poster-fallback">Poster unavailable</div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="details-title">{html.escape(str(title))}</div>', unsafe_allow_html=True)
        meta = [str(year) if year else "", html.escape(genres), f"{int(runtime)} min" if runtime else ""]
        meta = [x for x in meta if x]
        if meta: st.markdown(f'<div class="details-meta">{" · ".join(meta)}</div>', unsafe_allow_html=True)
        if tagline: st.markdown(f'<div class="details-tagline">{html.escape(tagline)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-score-grid"><div class="details-score-card primary"><div class="details-score-label">MOVIEMIND MATCH</div><div class="details-score-value">{match_score}%</div></div><div class="details-score-card"><div class="details-score-label">MODEL ESTIMATE</div><div class="details-score-value">{model_rating:.2f} / 5</div></div></div>', unsafe_allow_html=True)
        rating_text = f"{float(vote_average):.1f} / 10" if vote_average is not None else "Not available"
        runtime_text = f"{int(runtime)} min" if runtime else "Not available"
        st.markdown(f'<div class="details-facts"><div><span>TMDB RATING</span><strong>{rating_text}</strong></div><div><span>RUNTIME</span><strong>{runtime_text}</strong></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="details-section-title">Why MovieMind recommended it</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-reason">{html.escape(why_movie(movie, st.session_state.selected_movie))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="details-section-title">Overview</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="details-overview">{html.escape(str(overview))}</div>', unsafe_allow_html=True)
        if homepage: st.link_button("Open official movie page", homepage, use_container_width=True)
    st.markdown('<div class="details-attribution">Movie metadata and ratings are supplied by TMDB. MovieMind recommendation scores are generated locally.</div>', unsafe_allow_html=True)

if "recommendations" not in st.session_state: st.session_state.recommendations = None
if "selected_movie" not in st.session_state: st.session_state.selected_movie = None
if "movie_search" not in st.session_state: st.session_state.movie_search = ""
if "selected_details" not in st.session_state: st.session_state.selected_details = None

st.markdown("""
<style>
.stApp{background:#f7f9fc;color:#101828}.block-container{max-width:1180px;padding-top:24px;padding-bottom:50px}header[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer{display:none}.hero-kicker,.details-kicker{color:#155eef;font-size:11px;font-weight:900;letter-spacing:1.15px}.hero-title{color:#081a36;font-size:50px;font-weight:900;line-height:1.05;letter-spacing:-2px;max-width:850px;margin-bottom:12px}.hero-description{color:#344054;font-size:16px;line-height:1.65;max-width:680px}.details-title{color:#081a36;font-size:42px;font-weight:900;line-height:1.08;letter-spacing:-1.5px;margin:8px 0}.details-meta{color:#155eef;font-size:13px;font-weight:800;line-height:1.55;margin-bottom:16px}.details-tagline{color:#475467;font-size:15px;font-weight:700;line-height:1.55;padding:12px 14px;background:#f5f8ff;border:1px solid #dce8ff;border-radius:12px;margin-bottom:16px}.details-score-grid,.details-facts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}.details-score-card{background:#f8fafc;border:1px solid #e4eaf2;border-radius:14px;padding:14px 16px}.details-score-card.primary{background:#f0f6ff;border-color:#cfe0ff}.details-score-label,.details-facts span{color:#667085;font-size:9px;font-weight:900;letter-spacing:.7px}.details-score-value{color:#101828;font-size:24px;font-weight:900;margin-top:3px}.details-score-card.primary .details-score-value{color:#155eef}.details-facts>div{border-top:1px solid #e4e7ec;padding-top:10px}.details-facts strong{display:block;color:#101828;font-size:17px;font-weight:900;margin-top:3px}.details-section-title{color:#081a36;font-size:16px;font-weight:900;margin-top:20px;margin-bottom:8px}.details-reason{color:#344054;background:#f0f6ff;border:1px solid #d6e5ff;border-radius:12px;padding:13px 15px;font-size:13px;font-weight:600;line-height:1.6}.details-overview{color:#344054;font-size:14px;line-height:1.75;margin-bottom:18px}.details-attribution{color:#98a2b3;font-size:10px;margin-top:24px;padding-top:14px;border-top:1px solid #e4e7ec}.poster-fallback{width:100%;aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;text-align:center;color:#667085;background:#edf2f8;border-radius:12px}[data-testid="stImage"] img{width:100%!important;aspect-ratio:2/3!important;object-fit:cover!important;border-radius:12px!important}section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{background:#0b3b91!important}section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label{color:#fff!important}.stButton>button{background:#1455c0!important;color:#fff!important;border:0!important;border-radius:10px!important;min-height:46px!important;font-weight:900!important}@media(max-width:640px){.block-container{padding-left:14px;padding-right:14px}.hero-title{font-size:36px}.details-title{font-size:30px}.details-score-grid,.details-facts{grid-template-columns:1fr}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("MovieMind")
    st.caption("Personalized Movie Discovery")
    st.divider(); st.subheader("Discover")
    if st.button("New Discovery", use_container_width=True):
        st.session_state.recommendations = None; st.session_state.selected_movie = None; st.session_state.movie_search = ""; st.session_state.selected_details = None
    st.divider(); st.subheader("Quick Picks")
    for title in ["Toy Story (1995)", "Forrest Gump (1994)", "Inception (2010)", "Interstellar (2014)", "Dune (2021)"]:
        if title in set(movies["title"].astype(str)) and st.button(title, use_container_width=True):
            st.session_state.selected_movie = title; st.session_state.movie_search = title; st.session_state.recommendations = None
    st.divider(); st.subheader("Explore")
    for genre in ["Drama", "Comedy", "Action", "Romance", "Horror", "Science Fiction", "Animation"]: st.caption(genre)
    st.divider(); st.caption("Personalized movie discovery."); st.divider(); st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")

st.markdown('<div class="hero-kicker">AI-POWERED MOVIE DISCOVERY</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Find movies you\'ll love to watch.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-description">Tell MovieMind what you enjoy and discover personalized recommendations built around your taste.</div>', unsafe_allow_html=True)
st.info("Personalized movie recommendations")

with st.container(border=True):
    st.subheader("Tell MovieMind what you like")
    st.caption("Search the MovieMind catalog and choose a movie you already love.")
    search_col, year_col = st.columns([3,1], gap="medium")
    with search_col:
        search_text = st.text_input("Search for a movie", value=st.session_state.movie_search, placeholder="Try Interstellar, Inception, Dune, Oppenheimer...")
        st.session_state.movie_search = search_text
    with year_col:
        year_filter = st.selectbox("Release year", ["All", "2020+", "2015+", "2010+", "2000+", "1990+"])

filtered_movies = movies.copy(); query = search_text.strip().lower()
if query:
    filtered_movies = filtered_movies[(filtered_movies["title"].astype(str).str.lower().str.contains(query, regex=False, na=False)) | (filtered_movies["clean_title"].astype(str).str.lower().str.contains(query, regex=False, na=False))]
if year_filter != "All":
    minimum_year = int(year_filter.replace("+", "")); filtered_movies = filtered_movies[filtered_movies["year"].notna() & (filtered_movies["year"].astype(int) >= minimum_year)]
if query: filtered_movies = filtered_movies.sort_values(["year","title"], ascending=[False,True], na_position="last").head(100)
selected_movie = None
if query:
    if filtered_movies.empty: st.warning("No movies found. Try another title or year.")
    else:
        options = filtered_movies["title"].astype(str).tolist(); current = st.session_state.selected_movie if st.session_state.selected_movie in options else options[0]
        selected_movie = st.selectbox("Movie you like", options, index=options.index(current)); st.session_state.selected_movie = selected_movie
else: st.info("Start typing a movie title above to search the catalog.")

if selected_movie:
    user_col, count_col = st.columns(2)
    with user_col:
        if not ratings.empty and "userId" in ratings.columns:
            max_user_id = int(ratings["userId"].max())
        else:
            max_user_id = max(1, len(getattr(recommender.svd_model, "user_index", {})))
        user_id = st.number_input("User ID", min_value=1, max_value=max_user_id, value=1, step=1)
    with count_col: recommendation_count = st.slider("Number of recommendations", 5, 20, 10)
    find_movies = st.button("Find Movies For Me", use_container_width=True)
else: find_movies = False

if find_movies:
    try:
        with st.spinner("Finding movies for you..."):
            raw_count = min(max(recommendation_count * 4, 40), 100)
            candidates = recommender.recommend(user_id=int(user_id), movie_title=selected_movie, num_recommendations=raw_count)
            st.session_state.recommendations = rerank_recommendations(candidates, selected_movie, recommendation_count)
    except Exception:
        st.session_state.recommendations = None
        st.warning("We couldn't find recommendations right now. Please try another movie.")

if st.session_state.selected_details is not None:
    show_movie_details(st.session_state.selected_details)

if st.session_state.recommendations is not None:
    recommendations = add_match_scores(st.session_state.recommendations.copy())
    if not recommendations.empty:
        recommendations["movieId"] = pd.to_numeric(recommendations["movieId"], errors="coerce").astype("Int64")
        recommendations = recommendations.merge(movies[["movieId","tmdbId","year"]], on="movieId", how="left", suffixes=("","_catalog"))
    st.divider(); st.subheader("Your recommendations"); st.caption(f"Because you liked {selected_movie}")
    if recommendations.empty: st.info("No recommendations were found.")
    else:
        for start in range(0, len(recommendations), 3):
            row = recommendations.iloc[start:start+3]; columns = st.columns(3, gap="medium")
            for position, (column, (_, movie)) in enumerate(zip(columns, row.iterrows())):
                with column:
                    title = str(movie["title"]); year = movie.get("year_catalog")
                    if year is None or pd.isna(year): year = movie.get("year")
                    year_text = "" if year is None or pd.isna(year) else str(int(year))
                    genres = str(movie.get("genres", "")).replace("|", " · "); rating = float(movie.get("predicted_rating", 0)); poster_path = get_poster_path(movie.get("tmdbId"))
                    if poster_path: st.image(poster_path, width="stretch")
                    else: st.markdown('<div class="poster-fallback">Poster unavailable</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-number">RECOMMENDATION {start+position+1:02d}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-title">{html.escape(title)}</div>', unsafe_allow_html=True)
                    if year_text: st.markdown(f'<div class="movie-year">{year_text}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-genres">{html.escape(genres)}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="movie-score-label">MOVIEMIND MATCH</div>', unsafe_allow_html=True); st.markdown(f'<div class="movie-score">{int(movie.get("match_score",0))}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="movie-score-label">MODEL ESTIMATE</div>', unsafe_allow_html=True); st.markdown(f'<div class="movie-score" style="font-size:15px">{rating:.2f} / 5</div>', unsafe_allow_html=True)
                    st.markdown('<div class="why-label">WHY THIS MOVIE</div>', unsafe_allow_html=True); st.markdown(f'<div class="why-text">{html.escape(why_movie(movie, selected_movie))}</div>', unsafe_allow_html=True)
                    if st.button("View details", key=f"details_{int(movie['movieId'])}", use_container_width=True):
                        st.session_state.selected_details = movie.to_dict(); st.rerun()

st.divider(); st.caption("MovieMind  •  Personalized Movie Discovery"); st.caption("Created By : Nabin Chettri")
