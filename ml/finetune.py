import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
import random
from tqdm import tqdm

PROCESSED = Path("data/processed")
MODEL_DIR = Path("model")
BASE_MODEL_DIR = Path("model/base")
MODEL_DIR.mkdir(exist_ok=True)

def load_triplets(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        t = self.triplets[idx]
        return t["anchor"], t["positive"], t["negative"]

def triplet_loss(anchor, positive, negative, margin=0.5):
    """محاسبه Triplet Loss با cosine distance"""
    pos_dist = 1 - torch.nn.functional.cosine_similarity(anchor, positive)
    neg_dist = 1 - torch.nn.functional.cosine_similarity(anchor, negative)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def encode_with_grad(model, texts, device):
    """Encode با حفظ gradient"""
    features = model.tokenize(texts)
    features = {k: v.to(device) for k, v in features.items()}
    embeddings = model(features)['sentence_embedding']
    return embeddings

def main():
    print("Loading triplets...")
    triplets = load_triplets(PROCESSED / "triplets.json")
    random.shuffle(triplets)

    split = int(len(triplets) * 0.9)
    train_raw = triplets[:split]
    eval_raw = triplets[split:]

    print(f"Train: {len(train_raw)} | Eval: {len(eval_raw)}")

    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Base model not found at {BASE_MODEL_DIR}. "
            "Please download all-MiniLM-L6-v2 files to model/base/"
        )
    
    print(f"Loading model from {BASE_MODEL_DIR}...")
    model = SentenceTransformer(str(BASE_MODEL_DIR), local_files_only=True)
    
    train_dataset = TripletDataset(train_raw)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            anchors, positives, negatives = batch
            
            # Encode با gradient
            anchor_emb = encode_with_grad(model, anchors, device)
            positive_emb = encode_with_grad(model, positives, device)
            negative_emb = encode_with_grad(model, negatives, device)
            
            # محاسبه loss
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb, margin=0.5)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        if eval_raw:
            model.eval()
            with torch.no_grad():
                eval_sample = eval_raw[:100]
                eval_anchors = [t["anchor"] for t in eval_sample]
                eval_positives = [t["positive"] for t in eval_sample]
                
                anchor_embs = model.encode(eval_anchors, convert_to_tensor=True, device=device, show_progress_bar=False)
                positive_embs = model.encode(eval_positives, convert_to_tensor=True, device=device, show_progress_bar=False)
                
                similarities = torch.nn.functional.cosine_similarity(anchor_embs, positive_embs)
                avg_sim = similarities.mean().item()
                print(f"Eval - Average Similarity: {avg_sim:.4f}")

    print(f"\nSaving model to {MODEL_DIR}...")
    model.save(str(MODEL_DIR))
    print(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
