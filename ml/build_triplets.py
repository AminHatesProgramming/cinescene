# ml/build_triplets.py
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import random
from tqdm import tqdm

PROCESSED = Path("data/processed")
MODEL_PATH = Path("model/base")
OUTPUT = PROCESSED / "triplets.json"

def load_model():
    if not (MODEL_PATH / "config.json").exists():
        raise FileNotFoundError(
            f"Base model not found!\n"
            f"Please download 'all-MiniLM-L6-v2' manually and place it in:\n"
            f"  {MODEL_PATH.absolute()}\n"
            f"Required files: config.json, pytorch_model.bin, tokenizer files, etc."
        )
    
    print(f"Loading model from {MODEL_PATH}...")
    model = SentenceTransformer(str(MODEL_PATH), local_files_only=True)
    return model

def load_movies():
    print("Loading movies...")
    df = pd.read_json(PROCESSED / "movies_clean.json")
    return df

def build_triplets(df, target_count=8000):
    print(f"Building {target_count} triplets...")
    triplets = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if len(triplets) >= target_count:
            break
            
        anchor = row['rich_text']
        anchor_genres = set(row.get('genres', []))
        anchor_director = row.get('director', '')
        anchor_cast = set(row.get('cast', []))
        
        positives = df[
            (df.index != idx) & 
            (df['genres'].apply(lambda x: len(set(x) & anchor_genres) > 0) | 
             (df['director'] == anchor_director))
        ]
        
        negatives = df[
            (df.index != idx) & 
            (df['genres'].apply(lambda x: len(set(x) & anchor_genres) == 0)) &
            (df['director'] != anchor_director)
        ]
        
        if len(positives) > 0 and len(negatives) > 0:
            pos = positives.sample(1).iloc[0]['rich_text']
            neg = negatives.sample(1).iloc[0]['rich_text']
            
            triplets.append({
                "anchor": anchor,
                "positive": pos,
                "negative": neg
            })
    
    return triplets

def main():
    model = load_model()
    print(f"Model loaded successfully")
    
    df = load_movies()
    print(f"Loaded {len(df)} movies")
    
    triplets = build_triplets(df, target_count=8000)
    print(f"Generated {len(triplets)} triplets")
    
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    main()
