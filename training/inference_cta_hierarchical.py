#!/usr/bin/env python3
"""
Hierarchical CTA inference (l1 + l2 + datatype lookup).

Examples:
    python training/inference_cta_hierarchical.py \
      --model_dir model_hierarchical \
      --text "zip_code: 10001, 11201, 10013"

    python training/inference_cta_hierarchical.py \
      --model_dir model_hierarchical \
      --column customer_id \
      --values "CUST-000123, CUST-000124, CUST-000125"

    python training/inference_cta_hierarchical.py \
      --model_dir model_hierarchical \
      --input_csv training/synthetic_df.csv \
      --name_col name \
      --values_col values \
      --output_path predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from training.atlas_types import TWO_LEVEL_ONTOLOGY
except ImportError:
    from atlas_types import TWO_LEVEL_ONTOLOGY

DEFAULT_SPECIAL_TOKENS = {"col_token": "[COL]", "val_token": "[VAL]"}
DATATYPE_VOCAB = {"string", "integer", "float", "boolean", "date", "datetime"}


def mean_pool(outputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    return torch.device(device_name)


def build_l2_to_l1_from_ontology() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for l1_label, l2_labels in TWO_LEVEL_ONTOLOGY.items():
        for l2_label in l2_labels:
            if l2_label in mapping:
                raise ValueError(f"Duplicate l2 label in ontology: {l2_label}")
            mapping[l2_label] = l1_label
    return mapping


def build_family_gate(
    l1_classes: list[str], l2_classes: list[str], l2_to_l1: dict[str, str]
) -> torch.Tensor:
    l1_index = {label: idx for idx, label in enumerate(l1_classes)}
    gate = torch.zeros((len(l1_classes), len(l2_classes)), dtype=torch.bool)
    for l2_idx, l2_label in enumerate(l2_classes):
        l1_label = l2_to_l1[l2_label]
        gate[l1_index[l1_label], l2_idx] = True
    return gate


def format_cta_text(
    name: str | None,
    values: str | None,
    name_repeat: int,
    special_tokens: dict[str, str],
) -> str:
    col_tok = special_tokens.get("col_token", DEFAULT_SPECIAL_TOKENS["col_token"])
    val_tok = special_tokens.get("val_token", DEFAULT_SPECIAL_TOKENS["val_token"])
    safe_name = "" if pd.isna(name) else str(name).strip()
    safe_values = "" if pd.isna(values) else str(values)

    if safe_name and name_repeat > 1:
        col_part = " ".join([f"{col_tok} {safe_name}"] * name_repeat)
    elif safe_name:
        col_part = f"{col_tok} {safe_name}"
    else:
        col_part = ""

    value_tokens = [v.strip() for v in safe_values.split(",") if v.strip()]
    val_part = " ".join([f"{val_tok} {v}" for v in value_tokens[:10]])
    return f"{col_part} {val_part}".strip()


def build_default_l2_to_datatype(l2_labels: list[str]) -> dict[str, str]:
    l2_to_datatype = {l2: "string" for l2 in l2_labels}

    for label in ("data_time", "iso8601"):
        if label in l2_to_datatype:
            l2_to_datatype[label] = "datetime"

    for label in ("date", "birth_date"):
        if label in l2_to_datatype:
            l2_to_datatype[label] = "date"

    for label in (
        "age",
        "year",
        "quarter",
        "week_of_year",
        "month_of_year",
        "day_of_month",
        "unix_time",
        "ean8",
        "ean13",
        "primary_key",
        "foreign_key",
        "http_status_code",
        "latency_ms",
        "bytes_transferred",
        "quantity",
    ):
        if label in l2_to_datatype:
            l2_to_datatype[label] = "integer"

    for label in (
        "latitude",
        "longitude",
        "x_coord",
        "y_coord",
        "unit_price",
        "discount_percent",
        "tax_percent",
    ):
        if label in l2_to_datatype:
            l2_to_datatype[label] = "float"

    for label in ("boolean", "flag"):
        if label in l2_to_datatype:
            l2_to_datatype[label] = "boolean"

    return l2_to_datatype


def load_l2_to_datatype_from_csv(
    csv_path: str, l2_column: str = "l2_label", dtype_column: str = "datatype"
) -> dict[str, str]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Datatype mapping CSV not found: {csv_path}")

    df = pd.read_csv(path)
    required = {l2_column, dtype_column}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must include columns {sorted(required)} "
            f"(found: {sorted(df.columns.tolist())})"
        )

    tmp = df[[l2_column, dtype_column]].dropna().copy()
    tmp[l2_column] = tmp[l2_column].astype(str).str.strip()
    tmp[dtype_column] = tmp[dtype_column].astype(str).str.strip().str.lower()
    tmp = tmp[(tmp[l2_column] != "") & (tmp[dtype_column] != "")]

    mapping: dict[str, str] = {}
    for l2_label, group in tmp.groupby(l2_column):
        most_common_dtype = group[dtype_column].value_counts().idxmax()
        if most_common_dtype not in DATATYPE_VOCAB:
            raise ValueError(
                f"Invalid datatype '{most_common_dtype}' for l2 '{l2_label}' in {csv_path}. "
                f"Allowed: {sorted(DATATYPE_VOCAB)}"
            )
        mapping[l2_label] = most_common_dtype

    return mapping


class CTAHierarchicalModel(nn.Module):
    def __init__(self, config: AutoConfig, num_l1: int, num_l2: int, embed_dim: int):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )
        self.family_head = nn.Linear(embed_dim, num_l1)
        self.subtype_head = nn.Linear(embed_dim, num_l2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs, attention_mask)
        embeddings = F.normalize(self.projection(pooled), dim=1)
        family_logits = self.family_head(embeddings)
        subtype_logits = self.subtype_head(embeddings)
        return embeddings, family_logits, subtype_logits


class CTAHierarchicalClassifier:
    def __init__(
        self,
        model_dir: str,
        max_length: int = 128,
        device: str = "auto",
        use_gated_subtype: bool = True,
        datatype_map_csv: str | None = None,
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        label_encoder_path = self.model_dir / "label_encoder.json"
        if not label_encoder_path.exists():
            raise FileNotFoundError(
                f"Missing label encoder metadata: {label_encoder_path}"
            )

        with open(label_encoder_path, "r") as f:
            metadata = json.load(f)

        self.mode = metadata.get("mode", "hierarchical_contrastive")
        self.model_name = metadata.get("model_name", "BAAI/bge-base-en-v1.5")
        self.embed_dim = int(metadata.get("embed_dim", 128))
        self.name_repeat = int(metadata.get("name_repeat", 3))
        self.special_tokens = metadata.get("special_tokens", DEFAULT_SPECIAL_TOKENS)

        self.l2_classes = metadata.get("l2_classes", metadata.get("classes", []))
        if not self.l2_classes:
            raise ValueError("No l2 classes found in label_encoder.json")
        self.l2_classes = [str(x) for x in self.l2_classes]

        l2_to_l1 = metadata.get("l2_to_l1", {})
        if l2_to_l1:
            self.l2_to_l1 = {str(k): str(v) for k, v in l2_to_l1.items()}
        else:
            ontology_map = build_l2_to_l1_from_ontology()
            missing = sorted(set(self.l2_classes) - set(ontology_map))
            if missing:
                raise ValueError(
                    "Could not infer l2->l1 mapping for labels not in ontology: "
                    f"{missing}"
                )
            self.l2_to_l1 = {l2: ontology_map[l2] for l2 in self.l2_classes}

        missing_l2 = sorted(set(self.l2_classes) - set(self.l2_to_l1))
        if missing_l2:
            raise ValueError(f"Missing l2->l1 mapping for labels: {missing_l2}")

        self.l1_classes = metadata.get("l1_classes")
        if self.l1_classes:
            self.l1_classes = [str(x) for x in self.l1_classes]
        else:
            self.l1_classes = list(
                dict.fromkeys([self.l2_to_l1[l2] for l2 in self.l2_classes])
            )

        missing_l1 = sorted(set(self.l2_to_l1.values()) - set(self.l1_classes))
        if missing_l1:
            raise ValueError(
                f"l1 classes missing from label_encoder.json: {missing_l1}"
            )

        self.family_gate = build_family_gate(
            l1_classes=self.l1_classes,
            l2_classes=self.l2_classes,
            l2_to_l1=self.l2_to_l1,
        )

        self.max_length = max_length
        self.use_gated_subtype_default = use_gated_subtype
        self.device = get_device(device)

        tokenizer_path = self.model_dir if (self.model_dir / "tokenizer_config.json").exists() else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        config = AutoConfig.from_pretrained(str(self.model_dir))

        self.model = CTAHierarchicalModel(
            config=config,
            num_l1=len(self.l1_classes),
            num_l2=len(self.l2_classes),
            embed_dim=self.embed_dim,
        )
        if len(self.tokenizer) != self.model.encoder.get_input_embeddings().weight.size(0):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer))

        checkpoint_path = self.model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            print(f"Warning: missing keys while loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: unexpected keys while loading checkpoint: {unexpected_keys}")

        self.model.to(self.device)
        self.model.eval()

        self.l2_to_datatype = build_default_l2_to_datatype(self.l2_classes)
        if "l2_to_datatype" in metadata and isinstance(metadata["l2_to_datatype"], dict):
            for k, v in metadata["l2_to_datatype"].items():
                k_str = str(k)
                v_str = str(v).lower()
                if k_str in self.l2_to_datatype and v_str in DATATYPE_VOCAB:
                    self.l2_to_datatype[k_str] = v_str

        if datatype_map_csv:
            csv_map = load_l2_to_datatype_from_csv(datatype_map_csv)
            self.l2_to_datatype.update(
                {k: v for k, v in csv_map.items() if k in self.l2_to_datatype}
            )

    def _normalize_input(self, text: str) -> str:
        text = str(text).strip()
        if not text:
            return ""

        col_tok = self.special_tokens.get("col_token", DEFAULT_SPECIAL_TOKENS["col_token"])
        val_tok = self.special_tokens.get("val_token", DEFAULT_SPECIAL_TOKENS["val_token"])
        if col_tok in text or val_tok in text:
            return text

        if ":" in text:
            name, values = text.split(":", 1)
            return format_cta_text(
                name=name,
                values=values,
                name_repeat=self.name_repeat,
                special_tokens=self.special_tokens,
            )

        return format_cta_text(
            name="",
            values=text,
            name_repeat=self.name_repeat,
            special_tokens=self.special_tokens,
        )

    def _topk(self, probs: torch.Tensor, labels: list[str], top_k: int) -> list[dict]:
        k = min(top_k, len(labels))
        top_probs, top_indices = torch.topk(probs, k=k)
        return [
            {"label": labels[idx], "confidence": float(prob)}
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
        ]

    def predict_text(
        self,
        text: str,
        top_k: int = 3,
        threshold: float | None = None,
        use_gated_subtype: bool | None = None,
    ) -> dict:
        use_gated = (
            self.use_gated_subtype_default
            if use_gated_subtype is None
            else bool(use_gated_subtype)
        )
        formatted_text = self._normalize_input(text)
        if not formatted_text:
            raise ValueError("Input text is empty after formatting.")

        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            _, family_logits, subtype_logits = self.model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            )

            family_logits = family_logits[0]
            subtype_logits = subtype_logits[0]
            family_probs = F.softmax(family_logits, dim=-1)
            family_pred_idx = int(family_probs.argmax().item())

            if use_gated:
                allowed = self.family_gate.to(self.device)[family_pred_idx]
                subtype_logits_for_pred = subtype_logits.masked_fill(~allowed, -1e9)
            else:
                subtype_logits_for_pred = subtype_logits

            subtype_probs = F.softmax(subtype_logits_for_pred, dim=-1)

        top_l1 = self._topk(family_probs, self.l1_classes, top_k=top_k)
        top_l2_base = self._topk(subtype_probs, self.l2_classes, top_k=top_k)
        top_l2 = []
        for item in top_l2_base:
            l2_label = item["label"]
            top_l2.append(
                {
                    "label": l2_label,
                    "confidence": item["confidence"],
                    "l1_label": self.l2_to_l1[l2_label],
                    "datatype": self.l2_to_datatype.get(l2_label, "string"),
                }
            )

        if threshold is not None:
            top_l2 = [item for item in top_l2 if item["confidence"] >= threshold]

        if top_l2:
            best_l2 = top_l2[0]
        else:
            best_l2 = {
                "label": "unknown",
                "confidence": 0.0,
                "l1_label": top_l1[0]["label"],
                "datatype": "string",
            }

        best_l1 = top_l1[0]
        return {
            "input_text": text,
            "formatted_text": formatted_text,
            "use_gated_subtype": use_gated,
            "l1_label": best_l1["label"],
            "l1_confidence": best_l1["confidence"],
            "l2_label": best_l2["label"],
            "l2_confidence": best_l2["confidence"],
            "datatype": best_l2["datatype"],
            "top_l1": top_l1,
            "top_l2": top_l2,
        }

    def predict_column(
        self,
        column_name: str,
        values: str | list[str],
        top_k: int = 3,
        threshold: float | None = None,
        use_gated_subtype: bool | None = None,
    ) -> dict:
        if isinstance(values, list):
            values_str = ", ".join(str(v) for v in values[:10])
        else:
            values_str = str(values)
        text = f"{column_name}: {values_str}"
        return self.predict_text(
            text=text,
            top_k=top_k,
            threshold=threshold,
            use_gated_subtype=use_gated_subtype,
        )


def run_batch_inference(
    classifier: CTAHierarchicalClassifier,
    input_csv: str,
    name_col: str,
    values_col: str,
    top_k: int,
    threshold: float | None,
    use_gated_subtype: bool,
) -> list[dict]:
    df = pd.read_csv(input_csv)
    required = {name_col, values_col}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{input_csv} must include columns {sorted(required)} "
            f"(found: {sorted(df.columns.tolist())})"
        )

    records: list[dict] = []
    for _, row in df.iterrows():
        pred = classifier.predict_column(
            column_name=row[name_col],
            values=row[values_col],
            top_k=top_k,
            threshold=threshold,
            use_gated_subtype=use_gated_subtype,
        )
        records.append(
            {
                name_col: row[name_col],
                values_col: row[values_col],
                "pred_l1": pred["l1_label"],
                "pred_l1_confidence": pred["l1_confidence"],
                "pred_l2": pred["l2_label"],
                "pred_l2_confidence": pred["l2_confidence"],
                "pred_datatype": pred["datatype"],
                "top_l1": json.dumps(pred["top_l1"]),
                "top_l2": json.dumps(pred["top_l2"]),
            }
        )
    return records


def write_records(records: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".jsonl":
        with open(path, "w") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")
    else:
        pd.DataFrame(records).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for hierarchical CTA model (l1/l2/datatype)."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained hierarchical model directory.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single input text, e.g. 'column_name: v1, v2, v3'.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Column name for single inference (use with --values).",
    )
    parser.add_argument(
        "--values",
        type=str,
        default=None,
        help="Comma-separated values for single inference (use with --column).",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="CSV path for batch inference.",
    )
    parser.add_argument(
        "--name_col",
        type=str,
        default="name",
        help="Column name column in --input_csv.",
    )
    parser.add_argument(
        "--values_col",
        type=str,
        default="values",
        help="Values column in --input_csv.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for batch results (.csv or .jsonl).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-k predictions to include.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional confidence filter for top_l2 entries.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Tokenizer max length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    parser.add_argument(
        "--no_gated_subtype",
        action="store_true",
        help="Disable family-gated subtype prediction.",
    )
    parser.add_argument(
        "--datatype_map_csv",
        type=str,
        default=None,
        help="Optional CSV providing l2->datatype mapping (columns: l2_label, datatype).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1.")
    if (args.column is None) != (args.values is None):
        raise ValueError("Use --column and --values together.")

    classifier = CTAHierarchicalClassifier(
        model_dir=args.model_dir,
        max_length=args.max_length,
        device=args.device,
        use_gated_subtype=not args.no_gated_subtype,
        datatype_map_csv=args.datatype_map_csv,
    )
    print(
        f"Loaded hierarchical model from {args.model_dir} | "
        f"mode={classifier.mode}, l1={len(classifier.l1_classes)}, l2={len(classifier.l2_classes)}"
    )

    single_text_mode = args.text is not None or (args.column is not None and args.values is not None)
    batch_mode = args.input_csv is not None
    if single_text_mode and batch_mode:
        raise ValueError("Use either single-input mode (--text or --column/--values) or --input_csv, not both.")
    if not single_text_mode and not batch_mode:
        raise ValueError("Provide --text, or (--column and --values), or --input_csv.")

    if single_text_mode:
        if args.text is not None:
            prediction = classifier.predict_text(
                text=args.text,
                top_k=args.top_k,
                threshold=args.threshold,
                use_gated_subtype=not args.no_gated_subtype,
            )
        else:
            prediction = classifier.predict_column(
                column_name=args.column,
                values=args.values,
                top_k=args.top_k,
                threshold=args.threshold,
                use_gated_subtype=not args.no_gated_subtype,
            )
        print(json.dumps(prediction, indent=2))
        return

    records = run_batch_inference(
        classifier=classifier,
        input_csv=args.input_csv,
        name_col=args.name_col,
        values_col=args.values_col,
        top_k=args.top_k,
        threshold=args.threshold,
        use_gated_subtype=not args.no_gated_subtype,
    )

    if args.output_path:
        write_records(records, args.output_path)
        print(f"Wrote {len(records)} predictions to {args.output_path}")
    else:
        for row in records:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
