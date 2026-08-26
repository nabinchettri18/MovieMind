from pathlib import Path
import json
import hashlib
import urllib.request

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

POSTER_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_engine():
    return joblib.load(MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_links():
    if not LINKS_FILE.exists():
        return pd.DataFrame(
            columns=["movieId", "imdbId", "tmdbId"]
        )

    links = pd.read_csv(LINKS_FILE)

    for column in ["movieId", "imdbId", "tmdbId"]:
        if column not in links.columns:
            links[column] = pd.NA

    return links[
        ["movieId", "imdbId", "tmdbId"]
    ]


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


# ============================================================
# TMDB SECURITY
# ============================================================

# The secret is read only on the Python/server side.
# It is never written into HTML, JavaScript, query strings,
# recommendation results, or browser-visible configuration.

TMDB_API_KEY = st.secrets.get(
    "TMDB_API_KEY",
    "",
)

if not isinstance(TMDB_API_KEY, str):
    TMDB_API_KEY = str(TMDB_API_KEY)

TMDB_API_KEY = TMDB_API_KEY.strip()


# ============================================================
# MOVIE DATA
# ============================================================

links = load_links()

movies["movieId"] = pd.to_numeric(
    movies["movieId"],
    errors="coerce",
).astype("Int64")

links["movieId"] = pd.to_numeric(
    links["movieId"],
    errors="coerce",
).astype("Int64")

movies = movies.merge(
    links,
    on="movieId",
    how="left",
    suffixes=("", "_link"),
)

movies["tmdbId"] = pd.to_numeric(
    movies["tmdbId"],
    errors="coerce",
)

if "year" not in movies.columns:

    movies["year"] = pd.to_numeric(
        movies["title"]
        .astype(str)
        .str.extract(
            r"\((\d{4})\)\s*$"
        )[0],
        errors="coerce",
    )

if "clean_title" not in movies.columns:

    movies["clean_title"] = (
        movies["title"]
        .astype(str)
        .str.replace(
            r"\s*\(\d{4}\)\s*$",
            "",
            regex=True,
        )
        .str.strip()
    )


# ============================================================
# SECURE TMDB POSTER RESOLUTION
# ============================================================

@st.cache_data(show_spinner=False, max_entries=5000)
def get_tmdb_poster_url(tmdb_id):
    if not TMDB_API_KEY:
        return ""

    if tmdb_id is None or pd.isna(tmdb_id):
        return ""

    try:
        movie_id = int(tmdb_id)

        # TMDB API key authentication.
        # This request runs only on the Streamlit server;
        # the key is never rendered into the page/browser.
        request = urllib.request.Request(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}",
            headers={
                "Accept": "application/json",
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=10,
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        poster_path = data.get("poster_path")

        if not poster_path:
            return ""

        return (
            "https://image.tmdb.org/t/p/w500"
            + poster_path
        )

    except Exception:
        return ""


@st.cache_data(show_spinner=False, max_entries=5000)
def download_poster(poster_url):

    if not poster_url:
        return ""

    try:

        filename = (
            hashlib.sha256(
                poster_url.encode("utf-8")
            ).hexdigest()
            + ".jpg"
        )

        poster_file = POSTER_DIR / filename

        if (
            poster_file.exists()
            and poster_file.stat().st_size > 1000
        ):
            return str(poster_file)

        request = urllib.request.Request(
            poster_url,
            headers={
                "User-Agent": "MovieMind/1.0",
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=10,
        ) as response:
            image_data = response.read()

        if len(image_data) < 1000:
            return ""

        poster_file.write_bytes(image_data)

        return str(poster_file)

    except Exception:
        return ""


def get_poster_path(tmdb_id):

    poster_url = get_tmdb_poster_url(
        tmdb_id
    )

    if not poster_url:
        return ""

    return download_poster(
        poster_url
    )


# ============================================================
# SESSION STATE
# ============================================================

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "movie_search" not in st.session_state:
    st.session_state.movie_search = ""


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f9fc;
        color: #101828;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 24px;
        padding-bottom: 50px;
    }

    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    #MainMenu,
    footer {
        display: none;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3 {
        color: #081a36 !important;
        font-weight: 900 !important;
    }

    .stApp p {
        color: #344054 !important;
        font-weight: 550 !important;
    }

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background: #0b3b91 !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #082f73;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #e6efff !important;
        font-weight: 650 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.18);
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,.14) !important;
        border-radius: 10px !important;
        min-height: 42px !important;
        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] .stButton > button p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    .hero-kicker {
        color: #155eef;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 1.25px;
        margin-top: 30px;
        margin-bottom: 8px;
    }

    .hero-title {
        color: #081a36;
        font-size: 50px;
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: -2px;
        max-width: 850px;
        margin-bottom: 12px;
    }

    .hero-description {
        color: #344054;
        font-size: 16px;
        font-weight: 550;
        line-height: 1.65;
        max-width: 680px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #e1e7f0 !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 22px rgba(23,43,77,.045);
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #182230 !important;
        font-weight: 850 !important;
    }

    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #101828 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        min-height: 48px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] * {
        color: #172033 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: #101828 !important;
        font-weight: 750 !important;
    }

    .stButton > button {
        background: #1455c0 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 46px !important;
        font-weight: 900 !important;
    }

    .stButton > button p {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    .stButton > button:hover {
        background: #0b3b91 !important;
    }

    [data-testid="stImage"] img {
        width: 100% !important;
        aspect-ratio: 2 / 3 !important;
        object-fit: cover !important;
        object-position: center !important;
        border-radius: 12px !important;
        display: block !important;
    }

    .poster-fallback {
        width: 100%;
        aspect-ratio: 2 / 3;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 25px;
        text-align: center;
        color: #667085;
        font-size: 13px;
        font-weight: 750;
        line-height: 1.45;
        background: #edf2f8;
        border-radius: 12px;
    }

    .movie-number {
        color: #155eef;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: .8px;
        text-transform: uppercase;
        margin-top: 14px;
        margin-bottom: 5px;
    }

    .movie-title {
        color: #081a36;
        font-size: 18px;
        font-weight: 900;
        line-height: 1.28;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 46px;
    }

    .movie-year {
        color: #155eef;
        font-size: 12px;
        font-weight: 850;
        margin-top: 6px;
    }

    .movie-genres {
        color: #475467;
        font-size: 12px;
        font-weight: 600;
        line-height: 1.45;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 35px;
        margin-top: 10px;
    }

    .movie-score-label {
        color: #667085;
        font-size: 10px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .55px;
        margin-top: 15px;
    }

    .movie-score {
        color: #101828;
        font-size: 19px;
        font-weight: 900;
        margin-top: 2px;
    }

    .why-label {
        color: #155eef;
        font-size: 10px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: .6px;
        margin-top: 14px;
    }

    .why-text {
        color: #475467;
        font-size: 11px;
        font-weight: 550;
        line-height: 1.45;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-top: 4px;
    }

    @media (max-width: 900px) {

        .hero-title {
            font-size: 42px;
        }
    }

    @media (max-width: 640px) {

        .block-container {
            padding-left: 14px;
            padding-right: 14px;
        }

        .hero-kicker {
            margin-top: 18px;
        }

        .hero-title {
            font-size: 36px;
            line-height: 1.08;
            letter-spacing: -1px;
        }

        .hero-description {
            font-size: 15px;
        }

        .movie-title {
            font-size: 20px;
        }

        .movie-genres {
            font-size: 13px;
        }

        .movie-score {
            font-size: 20px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("MovieMind")
    st.caption("Personalized Movie Discovery")

    st.divider()

    st.subheader("Discover")

    if st.button(
        "New Discovery",
        use_container_width=True,
    ):
        st.session_state.recommendations = None
        st.session_state.selected_movie = None
        st.session_state.movie_search = ""

    st.divider()

    st.subheader("Quick Picks")

    quick_picks = [
        "Toy Story (1995)",
        "Forrest Gump (1994)",
        "Inception (2010)",
        "Interstellar (2014)",
        "Dune (2021)",
    ]

    available_movies = set(
        movies["title"].astype(str)
    )

    for title in quick_picks:

        if title in available_movies:

            if st.button(
                title,
                use_container_width=True,
            ):
                st.session_state.selected_movie = title
                st.session_state.movie_search = title
                st.session_state.recommendations = None

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

    st.caption(
        "Personalized movie discovery."
    )

    st.divider()

    st.caption(
        "This product uses the TMDB API but is not "
        "endorsed or certified by TMDB."
    )


# ============================================================
# HERO
# ============================================================

left, right = st.columns(
    [5, 1],
    vertical_alignment="center",
)

with left:
    st.caption("MovieMind")

st.markdown(
    '<div class="hero-kicker">'
    'AI-POWERED MOVIE DISCOVERY'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">'
    "Find movies you'll love to watch."
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-description">'
    "Tell MovieMind what you enjoy and discover "
    "personalized recommendations built around your taste."
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Personalized movie recommendations"
)


# ============================================================
# SEARCH PANEL
# ============================================================

with st.container(border=True):

    st.subheader(
        "Tell MovieMind what you like"
    )

    st.caption(
        "Search the MovieMind catalog and choose a movie "
        "you already love."
    )

    search_col, year_col = st.columns(
        [3, 1],
        gap="medium",
    )

    with search_col:

        search_text = st.text_input(
            "Search for a movie",
            value=st.session_state.movie_search,
            placeholder=(
                "Try Interstellar, Inception, Dune, "
                "Oppenheimer..."
            ),
        )

        st.session_state.movie_search = search_text

    with year_col:

        year_filter = st.selectbox(
            "Release year",
            [
                "All",
                "2020+",
                "2015+",
                "2010+",
                "2000+",
                "1990+",
            ],
        )


# ============================================================
# SEARCH
# ============================================================

filtered_movies = movies.copy()

query = search_text.strip().lower()

if query:

    title_match = (
        filtered_movies["title"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False,
            na=False,
        )
    )

    clean_match = (
        filtered_movies["clean_title"]
        .astype(str)
        .str.lower()
        .str.contains(
            query,
            regex=False,
            na=False,
        )
    )

    filtered_movies = filtered_movies[
        title_match | clean_match
    ]


if year_filter != "All":

    minimum_year = int(
        year_filter.replace("+", "")
    )

    filtered_movies = filtered_movies[
        filtered_movies["year"].notna()
        & (
            filtered_movies["year"].astype(int)
            >= minimum_year
        )
    ]


if query:

    filtered_movies = (
        filtered_movies
        .sort_values(
            ["year", "title"],
            ascending=[False, True],
            na_position="last",
        )
        .head(100)
    )


# ============================================================
# MOVIE SELECTION
# ============================================================

selected_movie = None

if query:

    if filtered_movies.empty:

        st.warning(
            "No movies found. Try another title or year."
        )

    else:

        st.caption(
            f"{len(filtered_movies):,} matching movies"
        )

        options = (
            filtered_movies["title"]
            .astype(str)
            .tolist()
        )

        current = (
            st.session_state.selected_movie
        )

        if current not in options:
            current = options[0]

        selected_movie = st.selectbox(
            "Movie you like",
            options,
            index=options.index(current),
        )

        st.session_state.selected_movie = selected_movie

else:

    st.info(
        "Start typing a movie title above to search the catalog."
    )


# ============================================================
# USER SETTINGS
# ============================================================

if selected_movie:

    user_col, count_col = st.columns(
        [1, 1],
        gap="medium",
    )

    with user_col:

        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=int(
                ratings["userId"].max()
            ),
            value=1,
            step=1,
        )

    with count_col:

        recommendation_count = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
        )

    find_movies = st.button(
        "Find Movies For Me",
        use_container_width=True,
    )

else:

    find_movies = False


# ============================================================
# RERANK RECOMMENDATIONS
# ============================================================

def rerank_recommendations(
    recommendations,
    selected_movie,
    limit,
):
    """
    Second-stage ranking for MovieMind.

    The trained 32M hybrid model is not modified.
    We take a larger candidate pool and then apply:

    1. duplicate removal
    2. selected-movie removal
    3. safe score normalization
    4. moderate recency preference
    5. genre-aware diversity
    6. greedy relevance/diversity ranking

    The final score is intentionally dominated by the
    original model signals rather than recency alone.
    """

    if recommendations is None:
        return pd.DataFrame()

    if recommendations.empty:
        return recommendations.copy()

    df = recommendations.copy()

    # --------------------------------------------------------
    # Remove duplicate movie IDs
    # --------------------------------------------------------

    if "movieId" in df.columns:
        df = df.drop_duplicates(
            subset=["movieId"],
            keep="first",
        )

    # --------------------------------------------------------
    # Never recommend the selected movie
    # --------------------------------------------------------

    if "title" in df.columns:

        selected_clean = (
            str(selected_movie)
            .strip()
            .lower()
        )

        df = df[
            df["title"]
            .astype(str)
            .str.strip()
            .str.lower()
            != selected_clean
        ]

    if df.empty:
        return df

    # --------------------------------------------------------
    # Make sure scoring columns exist
    # --------------------------------------------------------

    score_columns = [
        "predicted_rating",
        "hybrid_score",
        "content_score",
    ]

    for column in score_columns:

        if column not in df.columns:
            df[column] = 0.0

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0.0)

    # --------------------------------------------------------
    # Year extraction
    # --------------------------------------------------------

    if "year" not in df.columns:

        df["year"] = pd.to_numeric(
            df["title"]
            .astype(str)
            .str.extract(
                r"\((\d{4})\)\s*$"
            )[0],
            errors="coerce",
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Robust min-max normalization
    #
    # If every model score is identical, return 0.5 instead
    # of producing meaningless zeros.
    # --------------------------------------------------------

    def normalize(series):

        series = pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0.0)

        minimum = float(series.min())
        maximum = float(series.max())

        if (
            not pd.notna(minimum)
            or not pd.notna(maximum)
            or maximum <= minimum
        ):
            return pd.Series(
                0.5,
                index=series.index,
                dtype=float,
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        ).clip(0.0, 1.0)

    df["_hybrid"] = normalize(
        df["hybrid_score"]
    )

    df["_content"] = normalize(
        df["content_score"]
    )

    # Predicted rating is only a secondary signal.
    # If the model gives the same rating to everything,
    # this safely becomes neutral instead of distorting rank.
    df["_rating"] = normalize(
        df["predicted_rating"]
    )

    # --------------------------------------------------------
    # Modernity
    #
    # Moderate boost only. Recent movies are preferred,
    # but classics can still rank highly.
    # --------------------------------------------------------

    current_year = 2026

    age = (
        current_year - df["year"]
    )

    age = age.clip(
        lower=0,
        upper=50,
    )

    df["_modernity"] = (
        1.0 - (age / 50.0)
    ).fillna(0.35)

    df["_modernity"] = df[
        "_modernity"
    ].clip(0.0, 1.0)

    # --------------------------------------------------------
    # Base relevance
    #
    # Hybrid/content dominate. Recency is deliberately modest.
    # --------------------------------------------------------

    df["_base_score"] = (
        0.52 * df["_hybrid"]
        + 0.25 * df["_content"]
        + 0.10 * df["_rating"]
        + 0.13 * df["_modernity"]
    )

    # --------------------------------------------------------
    # Genre parsing
    # --------------------------------------------------------

    def parse_genres(value):

        return {
            part.strip().lower()
            for part in str(value).split("|")
            if part.strip()
            and part.strip().lower() != "(no genres listed)"
        }

    df["_genre_set"] = df.get(
        "genres",
        pd.Series(
            "",
            index=df.index,
        ),
    ).apply(parse_genres)

    # --------------------------------------------------------
    # Greedy diversity-aware ranking
    #
    # At every step:
    #
    # relevance = original recommendation strength
    # diversity = preference for a movie whose genres
    #             are not excessively repetitive
    #
    # This is deliberately not a hard genre filter.
    # Relevant same-genre movies are still allowed.
    # --------------------------------------------------------

    remaining = df.copy()

    chosen_rows = []
    chosen_genres = []
    chosen_titles = set()

    target = min(
        int(limit),
        len(remaining),
    )

    while (
        len(chosen_rows) < target
        and not remaining.empty
    ):

        best_index = None
        best_score = float("-inf")

        for index, row in remaining.iterrows():

            title = str(
                row.get("title", "")
            ).strip().lower()

            if not title or title in chosen_titles:
                continue

            movie_genres = row["_genre_set"]

            # Maximum genre overlap with any selected movie.
            max_similarity = 0.0

            for previous_genres in chosen_genres:

                if not movie_genres and not previous_genres:
                    similarity = 1.0

                else:

                    union = (
                        movie_genres
                        | previous_genres
                    )

                    intersection = (
                        movie_genres
                        & previous_genres
                    )

                    similarity = (
                        len(intersection)
                        / len(union)
                        if union
                        else 0.0
                    )

                max_similarity = max(
                    max_similarity,
                    similarity,
                )

            # Diversity penalty grows with repetition.
            diversity = (
                1.0 - max_similarity
            )

            # Relevance remains dominant.
            candidate_score = (
                0.82 * float(row["_base_score"])
                + 0.18 * diversity
            )

            if candidate_score > best_score:

                best_score = candidate_score
                best_index = index

        if best_index is None:
            break

        selected_row = remaining.loc[
            best_index
        ].copy()

        selected_row["rank_score"] = best_score

        chosen_rows.append(
            selected_row
        )

        chosen_titles.add(
            str(
                selected_row.get(
                    "title",
                    "",
                )
            ).strip().lower()
        )

        chosen_genres.append(
            selected_row["_genre_set"]
        )

        remaining = remaining.drop(
            index=best_index
        )

    if not chosen_rows:

        return df.head(limit).drop(
            columns=[
                "_hybrid",
                "_content",
                "_rating",
                "_modernity",
                "_base_score",
                "_genre_set",
            ],
            errors="ignore",
        ).reset_index(drop=True)

    result = pd.DataFrame(
        chosen_rows
    )

    # --------------------------------------------------------
    # Clean internal columns
    # --------------------------------------------------------

    result = result.drop(
        columns=[
            "_hybrid",
            "_content",
            "_rating",
            "_modernity",
            "_base_score",
            "_genre_set",
        ],
        errors="ignore",
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# RECOMMEND
# ============================================================


if find_movies:

    try:

        with st.spinner(
            "Finding movies for you..."
        ):

            raw_count = min(
                max(
                    recommendation_count * 4,
                    40,
                ),
                100,
            )

            candidates = recommender.recommend(
                user_id=int(user_id),
                movie_title=selected_movie,
                num_recommendations=raw_count,
            )

            st.session_state.recommendations = rerank_recommendations(
                candidates,
                selected_movie,
                recommendation_count,
            )

    except Exception:

        st.session_state.recommendations = None

        st.warning(
            "We couldn't find recommendations right now. "
            "Please try another movie."
        )


# ============================================================
# WHY MOVIE
# ============================================================

def why_movie(movie, selected):

    genres = str(
        movie.get("genres", "")
    ).replace("|", ", ")

    content_score = float(
        movie.get(
            "content_score",
            0,
        )
    )

    year = movie.get("year_catalog")

    if year is None or pd.isna(year):
        year = movie.get("year")

    year_text = ""

    if year is not None and not pd.isna(year):
        year_text = str(int(year))

    if content_score >= 0.55:

        base = (
            f"Strong content similarity to {selected}"
        )

    elif content_score >= 0.35:

        base = (
            f"Good content similarity to {selected}"
        )

    else:

        base = (
            f"Selected from MovieMind's combined "
            f"recommendation signals"
        )

    if genres:
        base += f", with {genres} as related genres."

    else:
        base += "."

    if year_text and int(year_text) >= 2020:
        base += " Recent-release preference also helped."

    return base



# ============================================================
# MOVIEMIND MATCH SCORE
# ============================================================

def add_match_scores(df):
    """
    Creates a user-facing MovieMind Match score.

    This is intentionally separate from predicted_rating.
    The latter can be tied for many candidates; Match Score
    represents the ranking strength of the final recommendation.
    """

    if df is None or df.empty:
        return df

    result = df.copy()

    def normalize(series):
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0.0)

        low = float(values.min())
        high = float(values.max())

        if high <= low:
            return pd.Series(
                0.5,
                index=values.index,
                dtype=float,
            )

        return ((values - low) / (high - low)).clip(0, 1)

    if "rank_score" in result.columns:
        base = normalize(result["rank_score"])

    elif "hybrid_score" in result.columns:
        base = normalize(result["hybrid_score"])

    else:
        base = pd.Series(
            0.5,
            index=result.index,
            dtype=float,
        )

    if "content_score" in result.columns:
        content = normalize(
            result["content_score"]
        )
    else:
        content = pd.Series(
            0.5,
            index=result.index,
            dtype=float,
        )

    if "year_catalog" in result.columns:
        years = pd.to_numeric(
            result["year_catalog"],
            errors="coerce",
        )
    elif "year" in result.columns:
        years = pd.to_numeric(
            result["year"],
            errors="coerce",
        )
    else:
        years = pd.Series(
            float("nan"),
            index=result.index,
        )

    recency = (
        ((years - 1980) / 46)
        .clip(0, 1)
        .fillna(0.45)
    )

    # Keep relevance dominant while allowing newer movies
    # to receive a moderate preference.
    match = (
        0.70 * base
        + 0.20 * content
        + 0.10 * recency
    )

    result["match_score"] = (
        70 + (match * 30)
    ).round().clip(70, 99)

    return result


# ============================================================
# RESULTS
# ============================================================

if st.session_state.recommendations is not None:

    recommendations = (
        st.session_state.recommendations.copy()
    )

    recommendations = add_match_scores(
        recommendations
    )

    if not recommendations.empty:

        recommendations["movieId"] = pd.to_numeric(
            recommendations["movieId"],
            errors="coerce",
        ).astype("Int64")

        recommendations = recommendations.merge(
            movies[
                [
                    "movieId",
                    "tmdbId",
                    "year",
                ]
            ],
            on="movieId",
            how="left",
            suffixes=("", "_catalog"),
        )

    st.divider()

    st.subheader(
        "Your recommendations"
    )

    st.caption(
        f"Because you liked {selected_movie}"
    )

    if recommendations.empty:

        st.info(
            "No recommendations were found."
        )

    else:

        for start in range(
            0,
            len(recommendations),
            3,
        ):

            row = recommendations.iloc[
                start:start + 3
            ]

            columns = st.columns(
                3,
                gap="medium",
            )

            for position, (
                column,
                (_, movie),
            ) in enumerate(
                zip(
                    columns,
                    row.iterrows(),
                )
            ):

                with column:

                    title = str(
                        movie["title"]
                    )

                    year = movie.get(
                        "year_catalog"
                    )

                    if year is None or pd.isna(year):
                        year = movie.get("year")

                    year_text = (
                        ""
                        if year is None or pd.isna(year)
                        else str(int(year))
                    )

                    genres = str(
                        movie.get(
                            "genres",
                            "",
                        )
                    ).replace(
                        "|",
                        " · ",
                    )

                    rating = float(
                        movie.get(
                            "predicted_rating",
                            0,
                        )
                    )

                    tmdb_id = movie.get(
                        "tmdbId"
                    )

                    poster_path = get_poster_path(
                        tmdb_id
                    )

                    if poster_path:

                        try:

                            st.image(
                                poster_path,
                                width="stretch",
                            )

                        except Exception:

                            st.markdown(
                                """
                                <div class="poster-fallback">
                                    Poster unavailable
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    else:

                        st.markdown(
                            """
                            <div class="poster-fallback">
                                Poster unavailable
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div class="movie-number">'
                        f'RECOMMENDATION '
                        f'{start + position + 1:02d}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div class="movie-title">'
                        f'{title}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if year_text:

                        st.markdown(
                            f'<div class="movie-year">'
                            f'{year_text}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div class="movie-genres">'
                        f'{genres}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    match_score = int(
                        movie.get(
                            "match_score",
                            0,
                        )
                    )

                    st.markdown(
                        '<div class="movie-score-label">'
                        'MOVIEMIND MATCH'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div class="movie-score">'
                        f'{match_score}%'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<div class="movie-score-label">'
                        'MODEL ESTIMATE'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div class="movie-score" '
                        f'style="font-size:15px;">'
                        f'{rating:.2f} / 5'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<div class="why-label">'
                        'WHY THIS MOVIE'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div class="why-text">'
                        f'{why_movie(movie, selected_movie)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MovieMind  •  Personalized Movie Discovery"
)

st.caption(
    "Created By : Nabin Chettri"
)
