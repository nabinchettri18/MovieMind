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
    for c in ["movieId", "imdbId", "tmdbId"]:
        if c not in links.columns:
            links[c] = pd.NA
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
    st.code(str(exc)[:800] or "No additional error message was provided.")
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

@st.cache_data(show_spinner=False, max_entries=2000)
def tmdb_details(tmdb_id):
    if not TMDB_API_KEY or tmdb_id is None or pd.isna(tmdb_id): return {}
    try:
        req = urllib.request.Request(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}", headers={"Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=10) as r: return json.loads(r.read().decode())
    except Exception: return {}

@st.cache_data(show_spinner=False, max_entries=5000)
def poster_url(tmdb_id):
    d = tmdb_details(tmdb_id)
    p = d.get("poster_path", "") if d else ""
    return f"https://image.tmdb.org/t/p/w780{p}" if p else ""

def why_movie(movie, selected):
    genres = str(movie.get("genres", "")).replace("|", ", ")
    score = float(movie.get("content_score", 0) or 0)
    base = f"Strong content similarity to {selected}" if score >= .55 else f"Good content similarity to {selected}" if score >= .35 else "Selected from MovieMind's combined recommendation signals"
    return f"{base}, with {genres} as related genres." if genres else base + "."

def normalize(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.)
    lo, hi = float(s.min()), float(s.max())
    return pd.Series(.5, index=s.index) if hi <= lo else ((s-lo)/(hi-lo)).clip(0,1)

def rerank(df, selected, limit):
    if df is None or df.empty: return pd.DataFrame()
    df = df.drop_duplicates("movieId").copy()
    df = df[df["title"].astype(str).str.lower().str.strip() != str(selected).lower().strip()].copy()
    for c in ["predicted_rating","hybrid_score","content_score"]:
        if c not in df: df[c] = 0.
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.)
    if "year" not in df: df["year"] = pd.to_numeric(df["title"].astype(str).str.extract(r"\((\d{4})\)")[0], errors="coerce")
    df["_h"],df["_c"],df["_r"] = normalize(df["hybrid_score"]),normalize(df["content_score"]),normalize(df["predicted_rating"])
    years=pd.to_numeric(df["year"],errors="coerce")
    df["_m"]=(1-((2026-years).clip(0,50)/50)).fillna(.35).clip(0,1)
    df["_base"]=.52*df["_h"]+.25*df["_c"]+.10*df["_r"]+.13*df["_m"]
    def genres(x): return {p.strip().lower() for p in str(x).split("|") if p.strip() and p.strip().lower()!="(no genres listed)"}
    df["_g"]=df.get("genres",pd.Series("",index=df.index)).apply(genres)
    rem=df.copy(); chosen=[]; gs=[]; titles=set()
    while len(chosen)<min(int(limit),len(rem)) and not rem.empty:
        bi=None; bs=-1e9
        for i,row in rem.iterrows():
            t=str(row.get("title","")).strip().lower()
            if not t or t in titles: continue
            sim=max([(len(row["_g"]&g)/len(row["_g"]|g)) if (row["_g"]|g) else 0 for g in gs],default=0)
            sc=.82*float(row["_base"])+.18*(1-sim)
            if sc>bs: bs,bi=sc,i
        if bi is None: break
        row=rem.loc[bi].copy(); row["rank_score"]=bs; chosen.append(row); titles.add(str(row["title"]).strip().lower()); gs.append(row["_g"]); rem=rem.drop(index=bi)
    return pd.DataFrame(chosen).drop(columns=["_h","_c","_r","_m","_base","_g"],errors="ignore").reset_index(drop=True)

def add_match(df):
    if df.empty:return df
    x=df.copy(); base=normalize(x["rank_score"] if "rank_score" in x else x.get("hybrid_score",0)); content=normalize(x["content_score"] if "content_score" in x else 0); years=pd.to_numeric(x.get("year",pd.Series(float("nan"),index=x.index)),errors="coerce"); rec=((years-1980)/46).clip(0,1).fillna(.45); x["match_score"]=(70+(0.70*base+0.20*content+0.10*rec)*30).round().clip(70,99); return x

for k,v in {"recommendations":None,"selected_movie":None,"movie_search":"","selected_details":None}.items():
    if k not in st.session_state: st.session_state[k]=v

st.markdown("""
<style>
.stApp{background:#f7f9fc;color:#101828}.block-container{max-width:1180px;padding-top:24px;padding-bottom:50px}header[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer{display:none}.hero-kicker,.details-kicker{color:#155eef;font-size:11px;font-weight:900;letter-spacing:1.15px}.hero-title{color:#081a36;font-size:50px;font-weight:900;line-height:1.05;letter-spacing:-2px;max-width:850px;margin-bottom:12px}.hero-description{color:#344054;font-size:16px;line-height:1.65;max-width:680px}.movie-title{color:#081a36;font-size:18px;font-weight:900;line-height:1.28}.movie-year{color:#155eef;font-size:12px;font-weight:850;margin-top:6px}.movie-genres{color:#475467;font-size:12px;line-height:1.45;min-height:35px;margin-top:10px}.movie-score-label,.why-label{color:#667085;font-size:10px;font-weight:750;text-transform:uppercase;letter-spacing:.55px;margin-top:15px}.movie-score{color:#101828;font-size:19px;font-weight:900;margin-top:2px}.why-label{color:#155eef}.why-text{color:#475467;font-size:11px;line-height:1.45;margin-top:4px}.poster-fallback{width:100%;aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;text-align:center;color:#667085;background:#edf2f8;border-radius:12px}[data-testid="stImage"] img{width:100%!important;aspect-ratio:2/3!important;object-fit:cover!important;border-radius:12px!important}section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div{background:#0b3b91!important}section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label{color:#fff!important}.stButton>button{background:#1455c0!important;color:#fff!important;border:0!important;border-radius:10px!important;min-height:46px!important;font-weight:900!important}.details-title{color:#081a36;font-size:42px;font-weight:900;line-height:1.08}.details-meta{color:#155eef;font-size:13px;font-weight:800;margin:10px 0}.details-overview{color:#344054;font-size:14px;line-height:1.75}.details-section-title{color:#081a36;font-size:16px;font-weight:900;margin-top:20px;margin-bottom:8px}.details-reason{color:#344054;background:#f0f6ff;border:1px solid #d6e5ff;border-radius:12px;padding:13px 15px;font-size:13px;line-height:1.6}.details-score-grid,.details-facts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}.details-score-card{background:#f8fafc;border:1px solid #e4eaf2;border-radius:14px;padding:14px}.details-score-card.primary{background:#f0f6ff;border-color:#cfe0ff}.details-score-label,.details-facts span{color:#667085;font-size:9px;font-weight:900;letter-spacing:.7px}.details-score-value{color:#101828;font-size:24px;font-weight:900}.details-facts>div{border-top:1px solid #e4e7ec;padding-top:10px}.details-facts strong{display:block;color:#101828;font-size:17px;margin-top:3px}@media(max-width:640px){.hero-title{font-size:36px}.details-title{font-size:30px}.details-score-grid,.details-facts{grid-template-columns:1fr}}
</style>
""",unsafe_allow_html=True)

with st.sidebar:
    st.title("MovieMind"); st.caption("Personalized Movie Discovery"); st.divider(); st.subheader("Discover")
    if st.button("New Discovery",use_container_width=True): st.session_state.recommendations=None; st.session_state.selected_movie=None; st.session_state.movie_search=""; st.session_state.selected_details=None; st.rerun()
    st.divider(); st.subheader("Quick Picks")
    for t in ["Toy Story (1995)","Forrest Gump (1994)","Inception (2010)","Interstellar (2014)","Dune (2021)"]:
        if t in set(movies["title"].astype(str)) and st.button(t,use_container_width=True): st.session_state.selected_movie=t; st.session_state.movie_search=t; st.session_state.recommendations=None
    st.divider(); st.subheader("Explore")
    for g in ["Drama","Comedy","Action","Romance","Horror","Science Fiction","Animation"]: st.caption(g)
    st.divider(); st.caption("Personalized movie discovery."); st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")

st.markdown('<div class="hero-kicker">AI-POWERED MOVIE DISCOVERY</div>',unsafe_allow_html=True)
st.markdown('<div class="hero-title">Find movies you\'ll love to watch.</div>',unsafe_allow_html=True)
st.markdown('<div class="hero-description">Tell MovieMind what you enjoy and discover personalized recommendations built around your taste.</div>',unsafe_allow_html=True)
st.info("Personalized movie recommendations")

with st.container(border=True):
    st.subheader("Tell MovieMind what you like"); st.caption("Search the MovieMind catalog and choose a movie you already love.")
    a,b=st.columns([3,1]);
    with a: search_text=st.text_input("Search for a movie",value=st.session_state.movie_search,placeholder="Try Interstellar, Inception, Dune, Oppenheimer..."); st.session_state.movie_search=search_text
    with b: year_filter=st.selectbox("Release year",["All","2020+","2015+","2010+","2000+","1990+"])

filtered=movies.copy(); q=search_text.strip().lower()
if q: filtered=filtered[(filtered["title"].astype(str).str.lower().str.contains(q,regex=False,na=False))|(filtered["clean_title"].astype(str).str.lower().str.contains(q,regex=False,na=False))]
if year_filter!="All": filtered=filtered[filtered["year"].notna()&(filtered["year"].astype(int)>=int(year_filter[:-1]))]
selected_movie=None
if q and not filtered.empty:
    opts=filtered["title"].astype(str).tolist(); cur=st.session_state.selected_movie if st.session_state.selected_movie in opts else opts[0]; selected_movie=st.selectbox("Movie you like",opts,index=opts.index(cur)); st.session_state.selected_movie=selected_movie
elif q: st.warning("No movies found. Try another title or year.")
else: st.info("Start typing a movie title above to search the catalog.")

if selected_movie:
    c,d=st.columns(2)
    with c: user_id=st.number_input("User ID",min_value=1,max_value=max(1,len(getattr(recommender.svd_model,"user_index",{}))),value=1,step=1)
    with d: count=st.slider("Number of recommendations",5,20,10)
    go=st.button("Find Movies For Me",use_container_width=True)
else: go=False

if go:
    try:
        with st.spinner("Finding movies for you..."):
            raw=recommender.recommend(user_id=int(user_id),movie_title=selected_movie,num_recommendations=min(max(count*4,40),100))
            st.session_state.recommendations=rerank(raw,selected_movie,count)
    except Exception as exc:
        st.session_state.recommendations=None; st.error(f"Recommendation error: {type(exc).__name__}"); st.code(str(exc)[:500])

if st.session_state.recommendations is not None:
    recs=add_match(st.session_state.recommendations.copy())
    if not recs.empty:
        recs["movieId"]=pd.to_numeric(recs["movieId"],errors="coerce").astype("Int64"); recs=recs.merge(movies[["movieId","tmdbId","year"]],on="movieId",how="left",suffixes=("","_catalog"))
    st.divider(); st.subheader("Your recommendations"); st.caption(f"Because you liked {selected_movie}")
    for start in range(0,len(recs),3):
        cols=st.columns(3)
        for pos,(col,(_,movie)) in enumerate(zip(cols,recs.iloc[start:start+3].iterrows())):
            with col:
                title=str(movie["title"]); year=movie.get("year_catalog",movie.get("year")); genres=str(movie.get("genres","")).replace("|"," · "); rating=float(movie.get("predicted_rating",0)); p=poster_url(movie.get("tmdbId"))
                if p: st.image(p,width="stretch")
                else: st.markdown('<div class="poster-fallback">Poster unavailable</div>',unsafe_allow_html=True)
                st.markdown(f'<div class="movie-title">{html.escape(title)}</div>',unsafe_allow_html=True); st.markdown(f'<div class="movie-year">{int(year) if pd.notna(year) else ""}</div>',unsafe_allow_html=True); st.markdown(f'<div class="movie-genres">{html.escape(genres)}</div>',unsafe_allow_html=True); st.markdown('<div class="movie-score-label">MOVIEMIND MATCH</div>',unsafe_allow_html=True); st.markdown(f'<div class="movie-score">{int(movie.get("match_score",0))}%</div>',unsafe_allow_html=True); st.markdown('<div class="movie-score-label">MODEL ESTIMATE</div>',unsafe_allow_html=True); st.markdown(f'<div class="movie-score">{rating:.2f} / 5</div>',unsafe_allow_html=True); st.markdown('<div class="why-label">WHY THIS MOVIE</div>',unsafe_allow_html=True); st.markdown(f'<div class="why-text">{html.escape(why_movie(movie,selected_movie))}</div>',unsafe_allow_html=True)
                if st.button("View details",key=f"details_{int(movie["movieId"])}",use_container_width=True): st.session_state.selected_details=movie.to_dict(); st.rerun()

if st.session_state.selected_details:
    movie=st.session_state.selected_details; d=tmdb_details(movie.get("tmdbId"))
    if d:
        if st.button("Back to recommendations",key="back_details"): st.session_state.selected_details=None; st.rerun()
        left,right=st.columns([1,1.8])
        with left:
            p=d.get("poster_path"); st.image(f"https://image.tmdb.org/t/p/w780{p}",width="stretch") if p else st.markdown('<div class="poster-fallback">Poster unavailable</div>',unsafe_allow_html=True)
        with right:
            st.markdown(f'<div class="details-kicker">MOVIEMIND MOVIE PROFILE</div><div class="details-title">{html.escape(str(d.get("title",movie.get("title","Unknown movie"))))}</div>',unsafe_allow_html=True)
            genres=" · ".join(x.get("name","") for x in d.get("genres",[]) if x.get("name")); st.markdown(f'<div class="details-meta">{html.escape(genres)}</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="details-score-grid"><div class="details-score-card primary"><div class="details-score-label">MOVIEMIND MATCH</div><div class="details-score-value">{int(movie.get("match_score",0))}%</div></div><div class="details-score-card"><div class="details-score-label">MODEL ESTIMATE</div><div class="details-score-value">{float(movie.get("predicted_rating",0)):.2f} / 5</div></div></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="details-section-title">Why MovieMind recommended it</div><div class="details-reason">{html.escape(why_movie(movie,st.session_state.selected_movie or "your selection"))}</div>',unsafe_allow_html=True); st.markdown(f'<div class="details-section-title">Overview</div><div class="details-overview">{html.escape(str(d.get("overview") or "No overview is available."))}</div>',unsafe_allow_html=True)
    else: st.warning("Movie details are temporarily unavailable.")

st.divider(); st.caption("MovieMind  •  Personalized Movie Discovery"); st.caption("Created By : Nabin Chettri")
