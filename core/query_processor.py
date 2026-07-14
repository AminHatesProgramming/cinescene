import re

# کلمات کلیدی برای تشخیص mood/tone
MOOD_MAP = {
    "funny":      ["comedy", "humor", "funny", "laugh"],
    "scary":      ["horror", "scary", "terrifying", "fear"],
    "romantic":   ["romance", "love", "relationship"],
    "action":     ["action", "fight", "explosion", "chase"],
    "sad":        ["sad", "emotional", "cry", "drama"],
    "inspiring":  ["inspiring", "motivational", "uplifting"],
    "thriller":   ["thriller", "suspense", "tension", "mystery"],
    "sci-fi":     ["sci-fi", "science fiction", "space", "future", "robot"],
    "animated":   ["animated", "animation", "cartoon", "pixar"],
    "documentary":["documentary", "real", "true story", "based on"],
}

def expand_query(query: str) -> str:
    query_lower = query.lower()
    expansions  = []

    for genre, keywords in MOOD_MAP.items():
        if any(kw in query_lower for kw in keywords):
            expansions.append(genre)

    if expansions:
        return f"{query} | Genres: {', '.join(expansions)}"
    return query

def clean_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    return query

def process_query(query: str) -> str:
    query = clean_query(query)
    query = expand_query(query)
    return query
