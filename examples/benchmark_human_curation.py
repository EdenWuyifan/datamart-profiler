#!/usr/bin/env python3
"""Benchmark GeoClassifier on the human curation sheet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiler.spatial import GeoClassifier, HybridGeoClassifier  # noqa: E402


SAMPLE_COLUMNS = ("sample_value_1", "sample_value_2", "sample_value_3")
LABEL_ALIASES = {
    "non-spatial": "non_spatial",
}


def normalize_label(value: str) -> str:
    label = str(value).strip().lower()
    return LABEL_ALIASES.get(label, label)


def sample_values(row: pd.Series) -> list[str]:
    return [str(row[col]).strip() for col in SAMPLE_COLUMNS if str(row[col]).strip()]


def accuracy(frame: pd.DataFrame, pred_col: str) -> float:
    return (frame[pred_col] == frame["gold_label"]).mean()


def per_label_accuracy(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return (
        frame.assign(correct=frame[pred_col] == frame["gold_label"])
        .groupby("gold_label", as_index=False)
        .agg(support=("gold_label", "size"), accuracy=("correct", "mean"))
        .sort_values(["support", "gold_label"], ascending=[False, True])
    )


def predict_batches(
    classifier,
    inputs: list[tuple[str, list[str]]],
    threshold: float,
    batch_size: int,
) -> list[dict]:
    results = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        results.extend(classifier.predict_batch(batch, threshold=threshold))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=Path(__file__).parent / "dataset" / "human_curation_sheet.csv",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).parent / "output" / "human_curation_geoclassifier_benchmark.csv",
        type=Path,
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, keep_default_na=False)
    frame["gold_label"] = frame["human_label"].map(normalize_label)
    frame = frame[frame["gold_label"] != ""].copy()

    inputs = [(row["column_name"], sample_values(row)) for _, row in frame.iterrows()]
    classifier = HybridGeoClassifier(
        GeoClassifier(
            model_dir=str(args.model_dir) if args.model_dir else None,
            auto_download=not args.no_download,
        )
    )
    predictions = predict_batches(classifier, inputs, args.threshold, args.batch_size)

    frame["predicted_label"] = [p["label"] for p in predictions]
    frame["confidence"] = [p["confidence"] for p in predictions]
    frame["source"] = [p.get("source", "") for p in predictions]
    frame["correct"] = frame["predicted_label"] == frame["gold_label"]
    frame["llm_label_normalized"] = frame["llm_label"].map(normalize_label)
    frame["llm_correct"] = frame["llm_label_normalized"] == frame["gold_label"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"Rows evaluated: {len(frame)}")
    print(f"GeoClassifier accuracy: {accuracy(frame, 'predicted_label'):.3f}")
    print(f"LLM label accuracy: {accuracy(frame, 'llm_label_normalized'):.3f}")
    print("\nGeoClassifier per-label accuracy:")
    print(per_label_accuracy(frame, "predicted_label").to_string(index=False))
    print(f"\nSaved predictions: {args.output}")


if __name__ == "__main__":
    main()
