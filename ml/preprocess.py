import pandas as pd
import ast
import json
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)

def parse_json_field(val, key="name"):
    try:
        items = ast.literal_eval(val)
        return [i[key] for i in items if key in i]
    except:
        return []

def get_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for member in crew:
            if member.get("job") == "Director":
                return member.get("name", "")
    except:
        pass
    return ""

def get_top_cast(cast_str, n=5):
    try:
        cast = ast.literal_eval(cast_str)
        cast_sorted = sorted(cast, key=lambda x: x.get("order", 99))
        return [c["name"] for c in cast_sorted[:n]]
    except:
        return []

def build_rich_text(row):
    parts = []

    if row.get("title"):
        parts.append(f"Title: {row['title']}")

    if row.get("tagline"):
        parts.append(f"Tagline: {row['tagline']}")

    if row.get("overview"):
        parts.append(f"Overview: {row['overview']}")

    if row.get("genres"):
        parts.append(f"Genres: {', '.join(row['genres'])}")

    if row.get("keywords"):
        parts.append(f"Keywords: {', '.join(row['keywords'][:10])}")

    if row.get("cast"):
        parts.append(f"Cast: {', '.join(row['cast'])}")

    if row.get("director"):
        parts.append(f"Director: {row['director']}")

    return " | ".join(parts)

def main():
    print("Loading CSVs...")
    movies = pd.read_csv(RAW / "tmdb_5000_movies.csv")
    credits = pd.read_csv(RAW / "tmdb_5000_credits.csv")

    # join روی id
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    print(f"Total movies: {len(df)}")

    # فیلتر: فقط فیلم‌های Released با overview
    df = df[df["status"] == "Released"]
    df = df[df["overview"].notna() & (df["overview"].str.strip() != "")]
    df = df[df["vote_count"] >= 50]

    print(f"After filtering: {len(df)}")

    # parse فیلدهای JSON
    df["genres"]   = df["genres"].apply(parse_json_field)
    df["keywords"] = df["keywords"].apply(parse_json_field)
    df["cast"]     = df["cast"].apply(get_top_cast)
    df["director"] = df["crew"].apply(get_director)

    # ساخت متن غنی
    df["rich_text"] = df.apply(build_rich_text, axis=1)

    # ذخیره خروجی
    output_cols = ["id", "title", "overview", "tagline", "genres",
                   "keywords", "cast", "director", "vote_average",
                   "vote_count", "popularity", "rich_text"]

    result = df[output_cols].reset_index(drop=True)
    result.to_csv(PROCESSED / "movies_clean.csv", index=False)
    result.to_json(PROCESSED / "movies_clean.json", orient="records", force_ascii=False, indent=2)

    print(f"Saved {len(result)} movies to data/processed/")
    print("\nSample rich_text:")
    print(result["rich_text"].iloc[0])

if __name__ == "__main__":
    main()
