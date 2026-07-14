"""
CineScene v2 fine-tuning.

Fine-tunes BGE embeddings on CineScene movie pairs using curriculum learning.
Hard negatives are kept in the triplet file for evaluation and future losses;
MultipleNegativesRankingLoss uses anchor-positive pairs and in-batch negatives.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, evaluation, losses
from torch.utils.data import DataLoader


LOCAL_BGE_PATH = Path("models/bge-large-en-v1.5")


def resolve_model_path(model_name: str) -> str:
    if model_name == "BAAI/bge-large-en-v1.5" and (LOCAL_BGE_PATH / "config.json").exists():
        return str(LOCAL_BGE_PATH)
    return model_name


class CineSceneFinetuner:
    def __init__(
        self,
        base_model: str = "BAAI/bge-large-en-v1.5",
        output_dir: str = "models/cinescene-v2",
        batch_size: int = 4,
        epochs: int = 3,
        seed: int = 42,
    ):
        self.base_model = resolve_model_path(base_model)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model = SentenceTransformer(self.base_model, device=self.device)

    def load_triplets(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            triplets = json.load(f)
        valid = [
            item
            for item in triplets
            if item.get("anchor") and item.get("positive") and item.get("negative")
        ]
        print(f"Loaded {len(triplets)} triplets; {len(valid)} valid")
        return valid

    def prepare_training_data(self, triplets: List[Dict]) -> List[InputExample]:
        return [InputExample(texts=[item["anchor"], item["positive"]]) for item in triplets]

    def create_curriculum_stages(self, examples: List[InputExample]) -> List[List[InputExample]]:
        n = len(examples)
        return [
            examples[: max(1, int(0.30 * n))],
            examples[: max(1, int(0.60 * n))],
            examples,
        ]

    def _create_evaluator(self, val_triplets: List[Dict]):
        anchors = [item["anchor"] for item in val_triplets]
        positives = [item["positive"] for item in val_triplets]
        negatives = [item["negative"] for item in val_triplets]
        return evaluation.TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            name="cinescene-val",
        )

    def train(self, triplets_path: str, use_curriculum: bool = True):
        triplets = self.load_triplets(triplets_path)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(triplets)

        split_idx = max(1, int(0.9 * len(triplets)))
        train_triplets = triplets[:split_idx]
        val_triplets = triplets[split_idx:] or triplets[-100:]
        train_examples = self.prepare_training_data(train_triplets)

        print(f"Train pairs: {len(train_examples)}, validation triplets: {len(val_triplets)}")
        evaluator = self._create_evaluator(val_triplets)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        if use_curriculum:
            stages = self.create_curriculum_stages(train_examples)
        else:
            stages = [train_examples]

        epochs_per_stage = max(1, self.epochs // len(stages))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for stage_idx, stage_data in enumerate(stages, start=1):
            stage_path = self.output_dir / f"stage_{stage_idx}"
            print(f"Stage {stage_idx}/{len(stages)} with {len(stage_data)} pairs")
            dataloader = DataLoader(stage_data, shuffle=True, batch_size=self.batch_size)
            warmup_steps = max(1, int(0.1 * len(dataloader)))

            self.model.fit(
                train_objectives=[(dataloader, train_loss)],
                evaluator=evaluator,
                epochs=epochs_per_stage,
                warmup_steps=warmup_steps,
                output_path=str(stage_path),
                save_best_model=True,
                show_progress_bar=True,
                use_amp=self.device == "cuda",
            )

        final_path = self.output_dir / "final"
        self.model.save(str(final_path))
        metadata = {
            "base_model": self.base_model,
            "triplets_path": triplets_path,
            "train_size": len(train_examples),
            "val_size": len(val_triplets),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "curriculum": use_curriculum,
            "device": self.device,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Model saved to {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CineScene embeddings")
    parser.add_argument("--triplets", default="data/processed/triplets_v2.json")
    parser.add_argument("--base-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--output-dir", default="models/cinescene-v2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--no-curriculum", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    finetuner = CineSceneFinetuner(
        base_model=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    finetuner.train(args.triplets, use_curriculum=not args.no_curriculum)


if __name__ == "__main__":
    main()
