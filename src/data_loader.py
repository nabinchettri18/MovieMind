from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ml-32m"


def _file(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing MovieLens file: {path}")
    return path


def load_movies():
    movies = pd.read_csv(
        _file("movies.csv"),
        usecols=["movieId", "title", "genres"],
    )

    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    movies["year"] = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)\s*$")[0],
        errors="coerce",
    ).astype("Int64")

    movies["clean_title"] = (
        movies["title"]
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
    )

    return movies


def load_ratings():
    return pd.read_csv(
        _file("ratings.csv"),
        usecols=["userId", "movieId", "rating", "timestamp"],
    )


def load_tags():
    tags = pd.read_csv(
        _file("tags.csv"),
        usecols=["userId", "movieId", "tag", "timestamp"],
    )
    tags["tag"] = tags["tag"].fillna("").astype(str)
    return tags


def load_links():
    return pd.read_csv(_file("links.csv"))