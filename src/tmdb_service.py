from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


class TMDBService:
    """Small server-side TMDB client for MovieMind."""

    def __init__(self, api_key: str, cache_dir: Path):
        self.api_key = (api_key or "").strip()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_json(self, path: str) -> dict:
        if not self.api_key:
            return {}

        request = urllib.request.Request(
            f"https://api.themoviedb.org/3{path}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return {}

    def movie_details(self, tmdb_id: int | float | str) -> dict:
        try:
            movie_id = int(float(tmdb_id))
        except (TypeError, ValueError):
            return {}

        if movie_id <= 0:
            return {}

        data = self._get_json(f"/movie/{movie_id}?language=en-US")
        if not data:
            return {}

        return {
            "id": data.get("id"),
            "title": data.get("title") or data.get("original_title") or "",
            "overview": data.get("overview") or "No overview available.",
            "tagline": data.get("tagline") or "",
            "poster_path": data.get("poster_path") or "",
            "backdrop_path": data.get("backdrop_path") or "",
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "runtime": data.get("runtime"),
            "release_date": data.get("release_date") or "",
            "genres": [g.get("name") for g in data.get("genres", []) if g.get("name")],
            "homepage": data.get("homepage") or "",
        }

    @staticmethod
    def image_url(path: str, size: str = "w500") -> str:
        if not path:
            return ""
        return f"https://image.tmdb.org/t/p/{size}{path}"

    def local_poster(self, poster_path: str) -> str:
        if not poster_path:
            return ""

        url = self.image_url(poster_path, "w500")
        filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".jpg"
        destination = self.cache_dir / filename

        if destination.exists() and destination.stat().st_size > 1000:
            return str(destination)

        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "MovieMind/1.0"},
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                data = response.read()
            if len(data) < 1000:
                return ""
            destination.write_bytes(data)
            return str(destination)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return ""
