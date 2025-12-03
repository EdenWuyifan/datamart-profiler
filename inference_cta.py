#!/usr/bin/env python3
"""
CTA Classifier Inference Script

Usage:
    python inference_cta.py --model_dir ./model --text "column_name: value1, value2, value3"
    python inference_cta.py --model_dir ./model_combined --text "latitude: 40.71, 40.72, 40.73"
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertModel,
    DistilBertTokenizerFast,
)
import torch.nn as nn


class CTAContrastiveModel(nn.Module):
    """DistilBERT with projection head for contrastive learning."""

    def __init__(
        self, model_name="distilbert-base-uncased", embed_dim=128, num_labels=None
    ):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )

        self.classifier = nn.Linear(hidden_size, num_labels) if num_labels else None

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        if return_embeddings:
            return F.normalize(self.projection(pooled), dim=1)

        if self.classifier:
            return self.classifier(pooled)
        return pooled

    def get_embeddings(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask, return_embeddings=True)


class CTAClassifier:
    """Unified interface for CTA classification."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

        # Load label encoder config
        with open(self.model_dir / "label_encoder.json") as f:
            config = json.load(f)

        self.classes = config["classes"]
        self.mode = config.get("mode", "classification")
        self.embed_dim = config.get("embed_dim", 128)

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load tokenizer
        if (self.model_dir / "tokenizer_config.json").exists():
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                str(self.model_dir)
            )
        else:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-uncased"
            )

        # Load model
        if self.mode == "classification":
            self.model = DistilBertForSequenceClassification.from_pretrained(
                str(self.model_dir), num_labels=len(self.classes)
            )
        else:  # contrastive or combined
            self.model = CTAContrastiveModel(
                embed_dim=self.embed_dim, num_labels=len(self.classes)
            )
            self.model.load_state_dict(
                torch.load(self.model_dir / "model.pt", map_location=self.device)
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, top_k: int = 3) -> list[dict]:
        """
        Predict column type for given text.

        Args:
            text: Format "column_name: value1, value2, value3"
            top_k: Number of top predictions to return

        Returns:
            List of dicts with 'label' and 'confidence'
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            if self.mode == "classification":
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:
                logits = self.model(input_ids, attention_mask)

            probs = F.softmax(logits, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, min(top_k, len(self.classes)))

        return [
            {"label": self.classes[idx], "confidence": prob.item()}
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
        ]

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text (contrastive/combined modes only)."""
        if self.mode == "classification":
            raise ValueError("Embeddings only available for contrastive/combined modes")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            emb = self.model.get_embeddings(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            )
        return emb[0].cpu().tolist()

    def predict_column(self, column_name: str, values: list) -> list[dict]:
        """Convenience method to predict from column name and values."""
        values_str = ", ".join(str(v) for v in values[:10])  # Limit to 10 values
        text = f"{column_name}: {values_str}"
        return self.predict(text)


def main():
    parser = argparse.ArgumentParser(description="CTA Classifier Inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory")
    parser.add_argument(
        "--text", type=str, help="Input text (format: 'column_name: val1, val2, val3')"
    )
    parser.add_argument("--column", type=str, help="Column name (use with --values)")
    parser.add_argument(
        "--values", type=str, help="Comma-separated values (use with --column)"
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of top predictions"
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Return embedding instead of prediction",
    )
    args = parser.parse_args()

    classifier = CTAClassifier(args.model_dir)
    print(f"Loaded model from {args.model_dir} (mode: {classifier.mode})")

    # Determine input
    if args.text:
        text = args.text
    elif args.column and args.values:
        text = f"{args.column}: {args.values}"
    else:
        print("Error: Provide --text OR (--column and --values)")
        return

    if args.embedding:
        emb = classifier.get_embedding(text)
        print(f"Embedding ({len(emb)} dims): {emb[:5]}...")
    else:
        predictions = classifier.predict(text, top_k=args.top_k)
        print(f"\nInput: {text}")
        print(f"\nTop {args.top_k} predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['label']}: {pred['confidence']:.4f}")


if __name__ == "__main__":
    main()
