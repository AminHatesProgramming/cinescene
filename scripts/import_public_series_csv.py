from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOVIE_CATALOG = ROOT / "data" / "processed" / "movies_enriched.json"
SERIES_CATALOG = ROOT / "data" / "processed" / "tv_series_enriched.json"
OFFLINE_CATALOG = ROOT / "data" / "processed" / "offline_media_enriched.json"
COMBINED_CATALOG = ROOT / "data" / "processed" / "cinescene_catalog.json"


def clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_list(value) -> List[str]:
    return [item.strip() for item in clean(value).split(",") if item.strip()]


def load_list(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def save_list(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def mood_tags(genres: List[str]) -> List[str]:
    text = " ".join(genres).lower()
    mapping = {
        "action": ["exciting", "intense"],
        "anime": ["imaginative", "stylized"],
        "comedy": ["funny", "lighthearted"],
        "crime": ["dark", "suspenseful"],
        "documentary": ["informative", "realistic"],
        "drama": ["emotional", "serious"],
        "family": ["heartwarming", "uplifting"],
        "fantasy": ["magical", "imaginative"],
        "horror": ["dark", "frightening"],
        "mystery": ["mysterious", "suspenseful"],
        "romance": ["romantic", "emotional"],
        "sci-fi": ["futuristic", "mind-bending"],
        "science": ["futuristic", "mind-bending"],
        "thriller": ["tense", "suspenseful"],
        "war": ["intense", "somber"],
    }
    return list(dict.fromkeys(tag for token, tags in mapping.items() if token in text for tag in tags))[:10]


def stable_id(row: pd.Series) -> str:
    identity = "|".join([clean(row.get("platform")), clean(row.get("title")), clean(row.get("release_year"))])
    return f"public-series:{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:16]}"


def to_document(row: pd.Series) -> Dict:
    title = clean(row.get("title")) or "Unknown"
    genres = split_list(row.get("listed_in"))
    cast = split_list(row.get("cast"))[:12]
    overview = clean(row.get("description"))
    platform = clean(row.get("platform"))
    country = clean(row.get("country"))
    content_rating = clean(row.get("rating"))
    duration = clean(row.get("duration"))
    year = clean(row.get("release_year"))
    moods = mood_tags(genres)
    keywords = list(
        dict.fromkeys(
            value
            for value in [
                "television series",
                "streaming series",
                platform.lower(),
                country.lower(),
                content_rating.lower(),
                duration.lower(),
                *[genre.lower() for genre in genres],
            ]
            if value
        )
    )[:18]
    rich_text = " | ".join(
        value
        for value in [
            f"Title: {title}",
            "Type: series",
            f"Year: {year}" if year else "",
            f"Genres: {', '.join(genres)}" if genres else "",
            f"Platform: {platform}" if platform else "",
            f"Cast: {', '.join(cast[:8])}" if cast else "",
            f"Plot: {overview}" if overview else "",
            f"Mood: {', '.join(moods)}" if moods else "",
            f"Keywords: {', '.join(keywords)}" if keywords else "",
        ]
        if value
    )
    return {
        "id": stable_id(row),
        "title": title,
        "original_title": title,
        "media_type": "series",
        "release_year": year,
        "genres": genres,
        "director": clean(row.get("director")) or "Unknown",
        "cast": cast,
        "overview": overview,
        "vote_average": 0.0,
        "popularity": 0.0,
        "country": country,
        "content_rating": content_rating,
        "duration": duration,
        "platform": platform,
        "scene_descriptions": [],
        "scene_timeline": [],
        "scene_count": 0,
        "mood_tags": moods,
        "keywords": keywords,
        "visual_tags": [],
        "rich_text": rich_text,
        "source": "public_series_metadata",
        "source_attribution": "MarcoM003/TV-Shows-Netflix-Disney on Hugging Face",
    }


def main():
    parser = argparse.ArgumentParser(description="Import the downloaded public TV series CSV")
    parser.add_argument("--input", default="data/raw/hf_tv_series/tv-shows.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    frame = pd.read_csv(input_path)
    series_rows = frame[frame["type"].fillna("").str.casefold() == "tv show"]
    imported = [to_document(row) for _, row in series_rows.iterrows()]
    by_id = {str(item["id"]): item for item in imported if item.get("title") != "Unknown"}
    series = list(by_id.values())
    series.sort(key=lambda item: (str(item.get("title") or "").casefold(), str(item.get("release_year") or "")))
    save_list(SERIES_CATALOG, series)

    movies = load_list(MOVIE_CATALOG)
    offline = load_list(OFFLINE_CATALOG)
    save_list(COMBINED_CATALOG, movies + series + offline)
    print(
        json.dumps(
            {
                "series_imported": len(series),
                "movies": len(movies),
                "offline_titles": len(offline),
                "combined_catalog": len(movies) + len(series) + len(offline),
                "output": str(SERIES_CATALOG),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
