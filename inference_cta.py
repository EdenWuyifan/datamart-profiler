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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class CTAClassificationModel(nn.Module):
    """Simple classification model using any transformer encoder."""

    def __init__(self, num_labels, config=None):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return type("Output", (), {"logits": logits})()


class CTAContrastiveModel(nn.Module):
    """Encoder with projection head for contrastive learning."""

    def __init__(self, embed_dim=128, num_labels=None, config=None):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
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
        self.model_name = config.get("model_name", "BAAI/bge-small-en-v1.5")
        self.embed_dim = config.get("embed_dim", 128)
        self.name_repeat = config.get("name_repeat", 3)  # Name emphasis
        self.special_tokens = config.get("special_tokens", {})  # [COL], [VAL] tokens

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load tokenizer (prefer saved, fallback to model_name)
        if (self.model_dir / "tokenizer_config.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load encoder config from saved directory
        encoder_config = AutoConfig.from_pretrained(str(self.model_dir))

        # Load model based on mode
        if self.mode == "classification":
            self.model = CTAClassificationModel(
                num_labels=len(self.classes),
                config=encoder_config,
            )
        else:  # contrastive or combined
            self.model = CTAContrastiveModel(
                embed_dim=self.embed_dim,
                num_labels=len(self.classes),
                config=encoder_config,
            )

        # Load saved weights
        self.model.load_state_dict(
            torch.load(self.model_dir / "model.pt", map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

    def _format_input(self, text: str) -> str:
        """Format input with special tokens: [COL] name [VAL] val1 [VAL] val2..."""
        # Parse "name: val1, val2, val3" format
        if ": " not in text:
            return text

        name, values = text.split(": ", 1)
        col_tok = self.special_tokens.get("col_token", "")
        val_tok = self.special_tokens.get("val_token", "")

        # Format column name with repetition
        if col_tok and name:
            if self.name_repeat > 1:
                col_parts = " ".join([f"{col_tok} {name}"] * self.name_repeat)
            else:
                col_parts = f"{col_tok} {name}"
        else:
            col_parts = name

        # Format values with [VAL] token
        if val_tok:
            val_list = [v.strip() for v in str(values).split(",")]
            val_parts = " ".join([f"{val_tok} {v}" for v in val_list[:10]])
        else:
            val_parts = values

        return f"{col_parts} {val_parts}".strip()

    def predict(
        self, text: str, top_k: int = 3, threshold: float | None = None
    ) -> list[dict]:
        """
        Predict column type for given text.

        Args:
            text: Format "column_name: value1, value2, value3"
            top_k: Number of top predictions to return
            threshold: If set, returns "non_spatial" when top confidence < threshold

        Returns:
            List of dicts with 'label' and 'confidence'
        """
        # Apply structured formatting: [COL] name [VAL] val1 [VAL] val2
        text = self._format_input(text)

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

        results = [
            {"label": self.classes[idx], "confidence": prob}
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
        ]

        # If threshold set and top prediction is below it, prepend non_spatial
        if threshold is not None and results[0]["confidence"] < threshold:
            results.insert(
                0,
                {"label": "non_spatial", "confidence": 1.0 - results[0]["confidence"]},
            )

        return results

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text (contrastive/combined modes only)."""
        if self.mode == "classification":
            raise ValueError("Embeddings only available for contrastive/combined modes")

        # Apply structured formatting
        text = self._format_input(text)

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
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold; below this returns 'non_spatial' (e.g., 0.5)",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Return embedding instead of prediction",
    )
    args = parser.parse_args()

    classifier = CTAClassifier(args.model_dir)
    print(f"Loaded model from {args.model_dir}")
    print(
        f"  Model: {classifier.model_name}, Mode: {classifier.mode}, "
        f"Name repeat: {classifier.name_repeat}"
    )

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
        predictions = classifier.predict(
            text, top_k=args.top_k, threshold=args.threshold
        )
        print(f"\nInput: {text}")
        if args.threshold:
            print(f"Threshold: {args.threshold}")
        print("\nTop predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['label']}: {pred['confidence']:.4f}")


if __name__ == "__main__":
    main()
