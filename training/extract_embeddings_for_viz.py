#!/usr/bin/env python3
"""
Extract embeddings from GeoClassifier model for visualization.

This script:
1. Loads data from CSV file (with name, values, label columns)
2. Extracts embeddings using GeoClassifier from profiler/spatial.py
3. Generates two TSV files:
   - embeddings.tsv: Vector embeddings (one row per data point, tab-separated)
   - metadata.tsv: Metadata with column names (name, label, etc.)
"""

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
import torch

from profiler.spatial import (
    GeoClassifier,
    CTAContrastiveModel,
    CTAClassificationModel,
    mean_pool,
)


def parse_values(values_str):
    """Parse values string (may be string representation of list)."""
    if isinstance(values_str, str):
        try:
            # Try to parse as Python literal (list)
            parsed = ast.literal_eval(values_str)
            if isinstance(parsed, list):
                return parsed
            else:
                return [str(parsed)]
        except (ValueError, SyntaxError):
            # If parsing fails, treat as single value or comma-separated
            return [v.strip() for v in values_str.split(",") if v.strip()]
    return [str(values_str)]


def extract_embeddings_batch(classifier, texts):
    """
    Extract embeddings for a batch of texts.

    Returns:
        numpy array of shape (batch_size, embedding_dim)
    """
    # Format all inputs
    formatted_texts = [classifier._format_input(t) for t in texts]

    # Batch tokenize
    encodings = classifier.tokenizer(
        formatted_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        input_ids = encodings["input_ids"].to(classifier.device)
        attention_mask = encodings["attention_mask"].to(classifier.device)

        # Extract embeddings based on model type
        if isinstance(classifier.model, CTAContrastiveModel):
            # Use projection head embeddings (normalized)
            embeddings = classifier.model.get_embeddings(input_ids, attention_mask)
        elif isinstance(classifier.model, CTAClassificationModel):
            # Use pooled encoder output (before classifier)
            outputs = classifier.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            embeddings = mean_pool(outputs, attention_mask)
        else:
            # Fallback: try to get pooled output
            outputs = classifier.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            embeddings = mean_pool(outputs, attention_mask)

    return embeddings.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from GeoClassifier for visualization"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file with name, values, label columns",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Model directory (default: profiler/model)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for TSV files (default: current directory)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)",
    )
    args = parser.parse_args()

    # Load classifier
    print(f"Loading GeoClassifier from {args.model_dir or 'profiler/model'}...")
    classifier = GeoClassifier(model_dir=args.model_dir, auto_download=True)
    print(f"Model loaded: {classifier.model_name}, Mode: {classifier.mode}")
    print(f"Using device: {classifier.device}")

    # Load CSV data
    print(f"\nLoading data from {args.input_csv}...")
    data_points = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "")
            values_str = row.get("values", "")
            label = row.get("label", "")

            # Parse values
            values = parse_values(values_str)

            # Format as "column_name: value1, value2, value3"
            values_display = ", ".join(
                str(v) for v in values[:5]
            )  # Limit to 5 for formatting
            text = f"{name}: {values_display}"

            data_points.append(
                {
                    "name": name,
                    "values": values,
                    "label": label,
                    "text": text,
                }
            )

    print(f"Loaded {len(data_points)} data points")

    # Extract embeddings in batches
    print(f"\nExtracting embeddings (batch size: {args.batch_size})...")
    all_embeddings = []

    for i in range(0, len(data_points), args.batch_size):
        batch = data_points[i : i + args.batch_size]
        texts = [dp["text"] for dp in batch]

        embeddings = extract_embeddings_batch(classifier, texts)
        all_embeddings.append(embeddings)

        if (i // args.batch_size + 1) % 10 == 0:
            print(
                f"  Processed {min(i + args.batch_size, len(data_points))}/{len(data_points)}..."
            )

    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    embedding_dim = all_embeddings.shape[1]
    print(
        f"Extracted embeddings: shape {all_embeddings.shape} (dimension: {embedding_dim})"
    )

    # Write embeddings TSV (vectors only, tab-separated)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_file = output_dir / "embeddings.tsv"
    print(f"\nWriting embeddings to {embeddings_file}...")
    with open(embeddings_file, "w", encoding="utf-8") as f:
        for emb in all_embeddings:
            # Write as tab-separated values
            f.write("\t".join(f"{val:.6f}" for val in emb) + "\n")
    print(f"✓ Wrote {len(all_embeddings)} vectors to {embeddings_file}")

    # Write metadata TSV (with column headers)
    metadata_file = output_dir / "metadata.tsv"
    print(f"\nWriting metadata to {metadata_file}...")
    with open(metadata_file, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "label", "sample_values"], delimiter="\t"
        )
        writer.writeheader()
        for dp in data_points:
            sample_values = ", ".join(str(v) for v in dp["values"][:3])  # Show first 3
            writer.writerow(
                {
                    "name": dp["name"],
                    "label": dp["label"],
                    "sample_values": sample_values,
                }
            )
    print(f"✓ Wrote {len(data_points)} metadata rows to {metadata_file}")

    print("\n✓ Complete! Generated files:")
    print(f"  - {embeddings_file}")
    print(f"  - {metadata_file}")
    print("\nYou can now load these files in your embedding visualization tool.")


if __name__ == "__main__":
    main()
