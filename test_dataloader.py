# test_dataloader.py
import torch
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
import json

print("Loading model...")
model = SentenceTransformer('models/bge-large-en-v1.5')

print("Loading triplets...")
with open('data/processed/triplets_v2.json', encoding='utf-8') as f:
    data = json.load(f)[:100]  # فقط ۱۰۰ نمونه

print(f"Creating {len(data)} examples...")
examples = [InputExample(texts=[d['anchor'], d['positive'], d['negative']]) for d in data]

print("Creating DataLoader...")
loader = DataLoader(examples, batch_size=2, shuffle=False)

print(f'Testing DataLoader with {len(examples)} examples...')
for i, batch in enumerate(loader):
    print(f'Batch {i+1} loaded successfully')
    if i >= 2:
        break
print('DataLoader test passed!')
