# MovieMind

AI-powered personalized movie discovery and recommendation system.

MovieMind combines content similarity with collaborative filtering over the MovieLens 32M dataset, with a second-stage ranking layer for relevance, recency, and genre diversity. TMDB is used for movie poster metadata.

## Features

- MovieLens 32M catalog with 87k+ movies and 32M+ ratings
- Content-based recommendations using genres and tags
- SVD collaborative filtering
- Hybrid recommendation engine
- Relevance and diversity-aware post-ranking
- Movie search and release-year filtering
- TMDB poster integration
- Mobile-friendly Streamlit interface
- Server-side TMDB credential handling

## Run locally

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Create `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "YOUR_TMDB_API_KEY"
```

3. Make sure the MovieLens 32M files are available locally under `data/ml-32m/`.

4. Build the model once:

```powershell
python train_models.py
```

5. Start MovieMind:

```powershell
python -m streamlit run app.py
```

## Project structure

```text
MovieMind/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── data_loader.py
│   ├── content_based.py
│   ├── svd_recommender.py
│   └── hybrid_recommender.py
├── data/
│   └── ml-32m/        # local, not committed
└── models/
    └── *.joblib       # local, not committed
```

## Data and attribution

MovieLens 32M is provided by GroupLens. This repository does not include the dataset; obtain it from the official MovieLens distribution.

Movie posters use TMDB metadata/images. This product uses the TMDB API but is not endorsed or certified by TMDB.

## Security

Do not commit `.streamlit/secrets.toml`, model files, datasets, or downloaded poster caches. They are excluded by `.gitignore`.
