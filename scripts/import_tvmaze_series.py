from __future__ import annotations

import argparse
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOVIE_CATALOG = ROOT / "data" / "processed" / "movies_enriched.json"
SERIES_CATALOG = ROOT / "data" / "processed" / "tv_series_enriched.json"
OFFLINE_CATALOG = ROOT / "data" / "processed" / "offline_media_enriched.json"
COMBINED_CATALOG = ROOT / "data" / "processed" / "cinescene_catalog.json"
API_ROOT = "https://api.tvmaze.com"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str):
        self.parts.append(data)


def strip_html(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value or "")
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


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
    mapping = {
        "Action": ["exciting", "intense"],
        "Adventure": ["adventurous", "exciting"],
        "Comedy": ["funny", "lighthearted"],
        "Crime": ["dark", "suspenseful"],
        "Drama": ["emotional", "serious"],
        "Family": ["heartwarming", "uplifting"],
        "Fantasy": ["magical", "imaginative"],
        "Horror": ["dark", "frightening"],
        "Mystery": ["mysterious", "suspenseful"],
        "Romance": ["romantic", "emotional"],
        "Science-Fiction": ["futuristic", "mind-bending"],
        "Thriller": ["tense", "suspenseful"],
        "War": ["intense", "somber"],
    }
    return list(dict.fromkeys(tag for genre in genres for tag in mapping.get(genre, [])))[:10]


def to_document(show: Dict) -> Dict:
    title = str(show.get("name") or "Unknown").strip()
    genres = [str(value) for value in show.get("genres") or [] if value]
    network = show.get("network") or show.get("webChannel") or {}
    network_name = str(network.get("name") or "").strip()
    overview = strip_html(str(show.get("summary") or ""))
    premiered = str(show.get("premiered") or "")
    year = premiered[:4] if len(premiered) >= 4 else ""
    rating = float((show.get("rating") or {}).get("average") or 0.0)
    weight = float(show.get("weight") or 0.0)
    keywords = list(
        dict.fromkeys(
            [
                "television series",
                str(show.get("type") or "series").lower(),
                str(show.get("language") or "").lower(),
                str(show.get("status") or "").lower(),
                network_name.lower(),
                *[genre.lower() for genre in genres],
            ]
        )
    )
    keywords = [value for value in keywords if value]
    moods = mood_tags(genres)
    rich_text = " | ".join(
        value
        for value in [
            f"Title: {title}",
            "Type: series",
            f"Year: {year}" if year else "",
            f"Genres: {', '.join(genres)}" if genres else "",
            f"Network: {network_name}" if network_name else "",
            f"Plot: {overview}" if overview else "",
            f"Mood: {', '.join(moods)}" if moods else "",
            f"Keywords: {', '.join(keywords)}" if keywords else "",
        ]
        if value
    )
    return {
        "id": f"tvmaze:{show.get('id')}",
        "title": title,
        "original_title": title,
        "media_type": "series",
        "release_year": year,
        "genres": genres,
        "director": "Unknown",
        "cast": [],
        "overview": overview,
        "vote_average": rating,
        "popularity": weight,
        "runtime": show.get("averageRuntime") or show.get("runtime"),
        "language": show.get("language"),
        "status": show.get("status"),
        "network": network_name,
        "official_site": show.get("officialSite"),
        "external_imdb": (show.get("externals") or {}).get("imdb"),
        "poster_url": (show.get("image") or {}).get("medium"),
        "scene_descriptions": [],
        "scene_timeline": [],
        "scene_count": 0,
        "mood_tags": moods,
        "keywords": keywords,
        "visual_tags": [],
        "rich_text": rich_text,
        "source": "tvmaze_series",
        "source_url": str(show.get("url") or ""),
    }


def fetch_pages(start_page: int, pages: int, delay: float) -> List[Dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "CineScene-Academic-Project/3.0"})
    rows: List[Dict] = []
    for page in range(start_page, start_page + pages):
        response = session.get(f"{API_ROOT}/shows", params={"page": page}, timeout=45)
        if response.status_code == 404:
            break
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
        print(f"Fetched TVmaze page {page}: {len(payload)} shows")
        if delay:
            time.sleep(delay)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Import legal public series metadata from TVmaze")
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    fetched = fetch_pages(args.start_page, args.pages, args.delay)
    existing = {str(item.get("id")): item for item in load_list(SERIES_CATALOG) if item.get("id")}
    for show in fetched:
        document = to_document(show)
        existing[str(document["id"])] = document
    series = list(existing.values())
    series.sort(key=lambda item: int(str(item.get("id")).split(":")[-1]))
    save_list(SERIES_CATALOG, series)

    movies = load_list(MOVIE_CATALOG)
    offline = load_list(OFFLINE_CATALOG)
    save_list(COMBINED_CATALOG, movies + series + offline)
    print(
        json.dumps(
            {
                "fetched": len(fetched),
                "series_catalog": len(series),
                "movie_catalog": len(movies),
                "offline_catalog": len(offline),
                "combined_catalog": len(movies) + len(series) + len(offline),
                "output": str(SERIES_CATALOG),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
