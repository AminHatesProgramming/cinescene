from __future__ import annotations

import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def normalize_title(title: str) -> str:
    title = clean_text(title).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def extract_year(value) -> str:
    match = re.search(r"(19\d{2}|20\d{2})", clean_text(value))
    return match.group(1) if match else ""


def as_text_list(values: Iterable) -> List[str]:
    if not isinstance(values, list):
        return []
    return [clean_text(value) for value in values if clean_text(value)]


def load_tmdb_processed() -> List[Dict]:
    with open(PROCESSED_DIR / "tmdb_processed.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_wikipedia_plots() -> pd.DataFrame:
    print("Loading Wikipedia plots...")
    with zipfile.ZipFile(RAW_DIR / "archive (2).zip", "r") as z:
        with z.open("wiki_movie_plots_deduped.csv") as f:
            return pd.read_csv(f)


def load_cmu_summaries() -> pd.DataFrame:
    print("Loading CMU summaries...")
    with tarfile.open(RAW_DIR / "MovieSummaries.tar.gz", "r:gz") as tar:
        plot_file = tar.extractfile("MovieSummaries/plot_summaries.txt")
        plots = {}
        if plot_file:
            for line in plot_file:
                parts = line.decode("utf-8", errors="ignore").strip().split("\t", 1)
                if len(parts) == 2:
                    plots[parts[0]] = parts[1]

        meta_file = tar.extractfile("MovieSummaries/movie.metadata.tsv")
        rows = []
        if meta_file:
            for line in meta_file:
                parts = line.decode("utf-8", errors="ignore").strip().split("\t")
                if len(parts) >= 3:
                    rows.append(
                        {
                            "wikipedia_id": parts[0],
                            "title": parts[2],
                            "release_year": extract_year(parts[3] if len(parts) > 3 else ""),
                            "plot": plots.get(parts[0], ""),
                        }
                    )
    return pd.DataFrame(rows)


def build_lookup(df: pd.DataFrame, title_col: str, year_col: str) -> Tuple[Dict[Tuple[str, str], Dict], Dict[str, Dict]]:
    by_title_year: Dict[Tuple[str, str], Dict] = {}
    by_title: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        title_key = normalize_title(row.get(title_col))
        if not title_key:
            continue
        year = extract_year(row.get(year_col))
        payload = row.to_dict()
        if year:
            by_title_year.setdefault((title_key, year), payload)
        by_title.setdefault(title_key, payload)

    return by_title_year, by_title


def lookup_movie(
    title: str,
    year: str,
    by_title_year: Dict[Tuple[str, str], Dict],
    by_title: Dict[str, Dict],
) -> Optional[Dict]:
    title_key = normalize_title(title)
    if not title_key:
        return None
    if year and (title_key, year) in by_title_year:
        return by_title_year[(title_key, year)]
    return by_title.get(title_key)


def split_scene_sentences(*texts: str) -> List[str]:
    scenes = []
    seen = set()
    for text in texts:
        text = clean_text(text)
        if not text:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = clean_text(sentence)
            key = sentence.lower()
            if len(sentence) >= 50 and key not in seen:
                scenes.append(sentence)
                seen.add(key)
            if len(scenes) >= 20:
                return scenes
    return scenes


def infer_mood_tags(movie: Dict, wiki_plot: str, cmu_plot: str) -> List[str]:
    tags = {item.lower() for item in as_text_list(movie.get("genres", []))}
    tags.update(item.lower() for item in as_text_list(movie.get("keywords", []))[:12])

    mood_rules = {
        "dark": ["dark", "noir", "gritty", "violent", "murder", "death", "crime"],
        "romantic": ["love", "romance", "relationship", "couple", "wedding"],
        "suspenseful": ["suspense", "thriller", "mystery", "investigation", "escape"],
        "comedic": ["comedy", "funny", "humor", "laugh"],
        "dramatic": ["drama", "emotional", "tragic", "conflict", "loss"],
        "action": ["fight", "chase", "explosion", "battle", "mission"],
        "science fiction": ["space", "alien", "future", "robot", "planet", "time travel"],
        "lonely": ["alone", "lonely", "isolation", "isolated", "solitude"],
    }

    all_text = " ".join([clean_text(movie.get("overview")), wiki_plot, cmu_plot]).lower()
    for mood, keywords in mood_rules.items():
        if any(keyword in all_text for keyword in keywords):
            tags.add(mood)

    return sorted(tags)


def build_rich_text(movie: Dict, wiki_plot: str, cmu_plot: str, scenes: List[str], mood_tags: List[str]) -> str:
    title = clean_text(movie.get("title")) or clean_text(movie.get("original_title")) or "Unknown"
    parts = [
        f"Title: {title}",
        f"Year: {clean_text(movie.get('release_year'))}",
        f"Genres: {', '.join(as_text_list(movie.get('genres', [])))}",
        f"Director: {clean_text(movie.get('director'))}",
        f"Cast: {', '.join(as_text_list(movie.get('cast', []))[:8])}",
        f"Overview: {clean_text(movie.get('overview'))}",
    ]
    if clean_text(movie.get("tagline")):
        parts.append(f"Tagline: {clean_text(movie.get('tagline'))}")
    if wiki_plot:
        parts.append(f"Plot: {wiki_plot[:900]}")
    if cmu_plot:
        parts.append(f"Summary: {cmu_plot[:900]}")
    if scenes:
        parts.append(f"Scenes: {' '.join(scenes[:5])[:1000]}")
    if movie.get("keywords"):
        parts.append(f"Keywords: {', '.join(as_text_list(movie.get('keywords', []))[:15])}")
    if mood_tags:
        parts.append(f"Mood: {', '.join(mood_tags)}")
    return " | ".join(part for part in parts if not part.endswith(": "))


def enrich_movies(tmdb_movies: List[Dict]) -> List[Dict]:
    wiki_df = load_wikipedia_plots()
    cmu_df = load_cmu_summaries()
    wiki_by_title_year, wiki_by_title = build_lookup(wiki_df, "Title", "Release Year")
    cmu_by_title_year, cmu_by_title = build_lookup(cmu_df, "title", "release_year")

    enriched_movies = []
    for index, movie in enumerate(tmdb_movies):
        if index % 500 == 0:
            print(f"Processing {index}/{len(tmdb_movies)}...")

        enriched = movie.copy()
        title = clean_text(movie.get("title")) or clean_text(movie.get("original_title"))
        year = extract_year(movie.get("release_year"))
        enriched["title"] = title

        wiki_match = lookup_movie(title, year, wiki_by_title_year, wiki_by_title)
        cmu_match = lookup_movie(title, year, cmu_by_title_year, cmu_by_title)

        wiki_plot = clean_text(wiki_match.get("Plot")) if wiki_match else ""
        cmu_plot = clean_text(cmu_match.get("plot")) if cmu_match else ""
        scenes = split_scene_sentences(clean_text(movie.get("overview")), wiki_plot, cmu_plot)
        mood_tags = infer_mood_tags(movie, wiki_plot, cmu_plot)

        enriched["wiki_plot"] = wiki_plot
        enriched["wiki_origin"] = clean_text(wiki_match.get("Origin/Ethnicity")) if wiki_match else ""
        enriched["wiki_director"] = clean_text(wiki_match.get("Director")) if wiki_match else ""
        enriched["cmu_plot"] = cmu_plot
        enriched["scene_descriptions"] = scenes
        enriched["mood_tags"] = mood_tags
        enriched["rich_text"] = build_rich_text(movie, wiki_plot, cmu_plot, scenes, mood_tags)

        enriched_movies.append(enriched)

    return enriched_movies


def main():
    print("Starting dataset enrichment...")
    tmdb_movies = load_tmdb_processed()
    print(f"Loaded {len(tmdb_movies)} TMDB movies")
    enriched_movies = enrich_movies(tmdb_movies)

    output_path = PROCESSED_DIR / "movies_enriched.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_movies, f, ensure_ascii=False, indent=2)

    sample = enriched_movies[0]
    print("Enrichment complete")
    print(f"Saved {len(enriched_movies)} movies to {output_path}")
    print(f"Sample title: {sample['title']}")
    print(f"Sample scenes: {len(sample['scene_descriptions'])}")
    print(f"Sample moods: {sample['mood_tags'][:8]}")


if __name__ == "__main__":
    main()
