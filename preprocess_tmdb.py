# preprocess_tmdb.py
import pandas as pd
import json
import ast
from pathlib import Path

def first_non_empty(*values):
    """Return the first non-empty, non-NaN string-like value."""
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""

def safe_parse_json(text):
    """Safely parse JSON-like strings"""
    if pd.isna(text):
        return []
    try:
        return ast.literal_eval(text)
    except:
        return []

def extract_names(items):
    """Extract 'name' field from list of dicts"""
    if isinstance(items, list):
        return [item.get('name', '') for item in items if isinstance(item, dict)]
    return []

def preprocess_tmdb():
    print("Loading TMDB datasets...")
    
    # Load movies
    movies_df = pd.read_csv('data/raw/tmdb_5000_movies.csv')
    credits_df = pd.read_csv('data/raw/tmdb_5000_credits.csv')
    
    print(f"Loaded {len(movies_df)} movies and {len(credits_df)} credits")
    
    # Merge on movie_id/id
    merged_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')
    
    processed_movies = []
    
    for idx, row in merged_df.iterrows():
        if idx % 500 == 0:
            print(f"Processing {idx}/{len(merged_df)}...")
        
        # Parse JSON fields
        genres = extract_names(safe_parse_json(row.get('genres', '[]')))
        keywords = extract_names(safe_parse_json(row.get('keywords', '[]')))
        production_companies = extract_names(safe_parse_json(row.get('production_companies', '[]')))
        cast_list = safe_parse_json(row.get('cast', '[]'))
        crew_list = safe_parse_json(row.get('crew', '[]'))
        
        # Extract cast names (top 10)
        cast = [actor.get('name', '') for actor in cast_list[:10] if isinstance(actor, dict)]
        
        # Extract director
        director = ''
        for person in crew_list:
            if isinstance(person, dict) and person.get('job') == 'Director':
                director = person.get('name', '')
                break
        
        # Extract release year - handle NaN
        release_date = row.get('release_date', '')
        if pd.isna(release_date) or not release_date:
            release_date = ''
            release_year = None
        else:
            release_year = str(release_date).split('-')[0] if '-' in str(release_date) else None
        
        title = first_non_empty(row.get('title'), row.get('title_x'), row.get('title_y'), row.get('original_title'))
        original_title = first_non_empty(row.get('original_title'), row.get('title_x'), row.get('title_y'), title)

        movie = {
            'id': int(row['id']),
            'title': title,
            'original_title': original_title,
            'tagline': row.get('tagline', '') if pd.notna(row.get('tagline')) else '',
            'overview': row.get('overview', '') if pd.notna(row.get('overview')) else '',
            'release_date': release_date,
            'release_year': release_year,
            'runtime': int(row['runtime']) if pd.notna(row.get('runtime')) else None,
            'budget': int(row['budget']) if pd.notna(row.get('budget')) else 0,
            'revenue': int(row['revenue']) if pd.notna(row.get('revenue')) else 0,
            'vote_average': float(row['vote_average']) if pd.notna(row.get('vote_average')) else 0.0,
            'vote_count': int(row['vote_count']) if pd.notna(row.get('vote_count')) else 0,
            'popularity': float(row['popularity']) if pd.notna(row.get('popularity')) else 0.0,
            'genres': genres,
            'keywords': keywords,
            'production_companies': production_companies,
            'cast': cast,
            'director': director,
            'original_language': row.get('original_language', ''),
        }
        
        processed_movies.append(movie)
    
    # Save as JSON
    output_json = 'data/processed/tmdb_processed.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_movies, f, ensure_ascii=False, indent=2)
    
    # Save as CSV for easy viewing
    output_csv = 'data/processed/tmdb_processed.csv'
    pd.DataFrame(processed_movies).to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\nProcessing complete!")
    print(f"Saved {len(processed_movies)} movies to:")
    print(f"  - {output_json}")
    print(f"  - {output_csv}")
    
    # Print sample
    print("\n=== Sample movie ===")
    sample = processed_movies[0]
    print(f"Title: {sample['title']}")
    print(f"Year: {sample['release_year']}")
    print(f"Genres: {sample['genres']}")
    print(f"Director: {sample['director']}")
    print(f"Cast: {sample['cast'][:3]}")

if __name__ == '__main__':
    preprocess_tmdb()
