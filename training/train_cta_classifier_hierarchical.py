#!/usr/bin/env python3
"""
Hierarchical CTA training (Option A)

Single encoder with:
- Hierarchical supervised contrastive loss
  - strong positives: same l2 subtype
  - weak positives: same l1 family, different l2 subtype
- Two classifiers on top of embeddings
  - family (l1)
  - subtype (l2)

Keeps hard-negative retrieval by mining cross-family negatives each epoch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from training.atlas_types import TWO_LEVEL_ONTOLOGY
except ImportError:
    from atlas_types import TWO_LEVEL_ONTOLOGY

MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPECIAL_TOKENS = {"col_token": "[COL]", "val_token": "[VAL]"}


# =========================================================================
# Utilities
# =========================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def mean_pool(outputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def build_l2_to_l1_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for l1_label, l2_labels in TWO_LEVEL_ONTOLOGY.items():
        for l2_label in l2_labels:
            if l2_label in mapping:
                raise ValueError(f"Duplicate l2 label in ontology: {l2_label}")
            mapping[l2_label] = l1_label
    return mapping


def format_cta_text(name: str, values: str, name_repeat: int) -> str:
    col_tok = SPECIAL_TOKENS["col_token"]
    val_tok = SPECIAL_TOKENS["val_token"]
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


def load_training_dataframe(
    csv_path: str, name_repeat: int, l2_to_l1: dict[str, str]
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(path)
    required = {"name", "values"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must include columns: {sorted(required)}")

    if {"l1_label", "l2_label"}.issubset(df.columns):
        df["l1_label"] = df["l1_label"].astype(str).str.strip()
        df["l2_label"] = df["l2_label"].astype(str).str.strip()
    elif "label" in df.columns:
        df["l2_label"] = df["label"].astype(str).str.strip()
        unknown_l2 = sorted(set(df["l2_label"]) - set(l2_to_l1))
        if unknown_l2:
            raise ValueError(
                f"`label` contains l2 classes not found in ontology: {unknown_l2}"
            )
        df["l1_label"] = df["l2_label"].map(l2_to_l1)
    else:
        raise ValueError(
            f"{csv_path} must include either (l1_label, l2_label) or label columns."
        )

    ontology_unknown_l2 = sorted(set(df["l2_label"]) - set(l2_to_l1))
    if ontology_unknown_l2:
        raise ValueError(
            f"Found l2 labels outside ontology in {csv_path}: {ontology_unknown_l2}"
        )

    df = df[["name", "values", "l1_label", "l2_label"]].copy()
    df["text"] = df.apply(
        lambda row: format_cta_text(row["name"], row["values"], name_repeat), axis=1
    )
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid rows left after text formatting.")

    return df


def enforce_min_subtype_count(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    counts = df["l2_label"].value_counts()
    keep_labels = counts[counts >= min_count].index
    dropped = counts[counts < min_count]
    if len(dropped) > 0:
        print(
            "Dropping l2 labels with insufficient rows for strong positives: "
            f"{dropped.to_dict()}"
        )
    kept_df = df[df["l2_label"].isin(keep_labels)].reset_index(drop=True)
    if len(kept_df) == 0:
        raise ValueError("No rows left after enforcing minimum subtype count.")
    return kept_df


def build_family_gate(l1_classes: list[str], l2_classes: list[str], l2_to_l1: dict[str, str]):
    l1_index = {label: idx for idx, label in enumerate(l1_classes)}
    gate = torch.zeros((len(l1_classes), len(l2_classes)), dtype=torch.bool)
    for l2_idx, l2_label in enumerate(l2_classes):
        l1_label = l2_to_l1[l2_label]
        gate[l1_index[l1_label], l2_idx] = True
    return gate


# =========================================================================
# Datasets
# =========================================================================


class CTAHierDataset(Dataset):
    def __init__(self, texts, l1_labels, l2_labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length
        )
        self.l1_labels = l1_labels
        self.l2_labels = l2_labels

    def __len__(self):
        return len(self.l2_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "l1_labels": torch.tensor(self.l1_labels[idx]),
            "l2_labels": torch.tensor(self.l2_labels[idx]),
            "idx": torch.tensor(idx),
        }


class EmbedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "idx": torch.tensor(idx),
        }


class HierarchicalBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        l1_labels,
        l2_labels,
        batch_size,
        num_negatives,
        seed=42,
        drop_last=True,
    ):
        self.l1_labels = list(l1_labels)
        self.l2_labels = list(l2_labels)
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.hard_negs: dict[int, list[int]] = {}

        if batch_size < 3:
            raise ValueError("batch_size must be >= 3 for hierarchical sampling.")

        self.group_size = 3 + max(0, num_negatives)
        self.anchors_per_batch = max(1, batch_size // self.group_size)

        l1_to_l2_to_idx = defaultdict(lambda: defaultdict(list))
        for i, (l1, l2) in enumerate(zip(self.l1_labels, self.l2_labels)):
            l1_to_l2_to_idx[int(l1)][int(l2)].append(i)
        self.l1_to_l2_to_idx = l1_to_l2_to_idx
        self.l1_to_indices = {
            l1: [idx for l2 in l2_map.values() for idx in l2]
            for l1, l2_map in l1_to_l2_to_idx.items()
        }

        self.l1_candidates = []
        self.l1_to_l2_any = {}
        self.l1_to_l2_strong = {}
        for l1, l2_map in l1_to_l2_to_idx.items():
            l2_any = [l2 for l2, idxs in l2_map.items() if len(idxs) >= 1]
            l2_strong = [l2 for l2, idxs in l2_map.items() if len(idxs) >= 2]
            if len(l2_any) >= 2 and len(l2_strong) >= 1:
                self.l1_candidates.append(l1)
                self.l1_to_l2_any[l1] = l2_any
                self.l1_to_l2_strong[l1] = l2_strong

        if not self.l1_candidates:
            raise ValueError(
                "No eligible l1 families found for hierarchical sampling. "
                "Ensure each family has >=2 l2 labels and >=2 samples in at least one l2."
            )

        self.num_batches = (
            len(self.l1_labels) // self.batch_size
            if drop_last
            else math.ceil(len(self.l1_labels) / self.batch_size)
        )

    def set_hard_negatives(self, hard_negs: dict[int, list[int]]) -> None:
        self.hard_negs = hard_negs

    def __len__(self):
        return self.num_batches

    def _choose_l1s(self):
        l1s = []
        while len(l1s) < self.anchors_per_batch:
            candidates = self.l1_candidates[:]
            self.rng.shuffle(candidates)
            l1s.extend(candidates)
        return l1s[: self.anchors_per_batch]

    def _sample_negatives(self, anchor_idx: int, family: int) -> list[int]:
        negatives = []
        if self.num_negatives <= 0:
            return negatives

        mined = [
            idx
            for idx in self.hard_negs.get(anchor_idx, [])
            if int(self.l1_labels[idx]) != family
        ]
        negatives.extend(mined[: self.num_negatives])

        if len(negatives) < self.num_negatives:
            pool = [i for i, l1 in enumerate(self.l1_labels) if int(l1) != family]
            needed = self.num_negatives - len(negatives)
            if pool:
                negatives.extend(self.rng.choices(pool, k=needed))
        return negatives

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            l1_choices = self._choose_l1s()
            used_l1 = set(l1_choices)

            for l1 in l1_choices:
                anchor_l2 = self.rng.choice(self.l1_to_l2_strong[l1])
                anchor_pool = self.l1_to_l2_to_idx[l1][anchor_l2]
                anchor_idx = self.rng.choice(anchor_pool)
                strong_pool = [i for i in anchor_pool if i != anchor_idx]
                strong_idx = self.rng.choice(strong_pool) if strong_pool else anchor_idx

                weak_l2_choices = [
                    l2 for l2 in self.l1_to_l2_any[l1] if l2 != anchor_l2
                ]
                weak_l2 = self.rng.choice(weak_l2_choices)
                weak_idx = self.rng.choice(self.l1_to_l2_to_idx[l1][weak_l2])

                negatives = self._sample_negatives(anchor_idx, l1)
                batch.extend([anchor_idx, strong_idx, weak_idx])
                batch.extend(negatives)

            extra = self.batch_size - len(batch)
            if extra > 0:
                neg_l1s = [l1 for l1 in self.l1_candidates if l1 not in used_l1]
                if not neg_l1s:
                    neg_l1s = self.l1_candidates
                for _ in range(extra):
                    l1 = self.rng.choice(neg_l1s)
                    idx = self.rng.choice(self.l1_to_indices[l1])
                    batch.append(idx)

            yield batch[: self.batch_size]


# =========================================================================
# Model & Loss
# =========================================================================


class CTAHierarchicalModel(nn.Module):
    def __init__(self, model_name: str, num_l1: int, num_l2: int, embed_dim: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
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
        return embeddings, self.family_head(embeddings), self.subtype_head(embeddings)


class HierarchicalSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, weak_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.weak_weight = weak_weight

    def forward(self, features, l1_labels, l2_labels):
        device = features.device
        batch_size = features.size(0)

        sim = torch.matmul(features, features.T) / self.temperature
        logits_mask = torch.ones_like(sim, device=device) - torch.eye(
            batch_size, device=device
        )
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        l1 = l1_labels.view(-1, 1)
        l2 = l2_labels.view(-1, 1)
        same_l2 = torch.eq(l2, l2.T).float()
        same_l1 = torch.eq(l1, l1.T).float()

        strong = same_l2 - torch.eye(batch_size, device=device)
        weak = same_l1 - same_l2
        weights = strong + self.weak_weight * weak
        weights = torch.clamp(weights, min=0.0)

        weight_sum = weights.sum(dim=1)
        loss = -(weights * log_prob).sum(dim=1) / torch.clamp(weight_sum, min=1.0)
        loss = loss * (weight_sum > 0).float()
        return loss.mean()


# =========================================================================
# Training
# =========================================================================


@torch.no_grad()
def encode_all_embeddings(model, loader, device):
    model.eval()
    all_emb = []
    all_idx = []
    for batch in loader:
        emb, _, _ = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        all_emb.append(emb.cpu())
        all_idx.append(batch["idx"].cpu())
    return torch.cat(all_emb), torch.cat(all_idx)


def mine_hard_negatives_cosine(embeddings, l1_ids, topk=64, hard_k=32):
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-1e9)
    nn_idx = torch.topk(sim, k=min(topk, embeddings.size(0) - 1), dim=1).indices
    nn_idx = nn_idx.cpu().numpy()

    hard_negs = {}
    for i in range(embeddings.size(0)):
        family = int(l1_ids[i])
        mined = [int(j) for j in nn_idx[i] if int(l1_ids[j]) != family]
        hard_negs[i] = mined[:hard_k]
    return hard_negs


def gated_subtype_predictions(family_logits, subtype_logits, family_gate):
    family_pred = family_logits.argmax(dim=1)
    allowed = family_gate.to(subtype_logits.device)[family_pred]
    masked_logits = subtype_logits.masked_fill(~allowed, -1e9)
    return masked_logits.argmax(dim=1)


def run_epoch(
    model,
    loader,
    l1_ids,
    l2_ids,
    device,
    supcon,
    lambda_contrastive,
    lambda_family,
    lambda_subtype,
    family_gate,
    optimizer=None,
    scheduler=None,
    grad_clip_norm=1.0,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_con = 0.0
    total_fce = 0.0
    total_sce = 0.0
    correct_fam = 0
    correct_sub = 0
    correct_sub_gated = 0
    total_samples = 0

    progress = tqdm(loader, desc="[Train]" if is_train else "[Val]")
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in progress:
            embeddings, family_logits, subtype_logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )

            l1_labels = batch["l1_labels"].to(device)
            l2_labels = batch["l2_labels"].to(device)

            loss_con = supcon(embeddings, l1_labels, l2_labels)
            loss_f = F.cross_entropy(family_logits, l1_labels)
            loss_s = F.cross_entropy(subtype_logits, l2_labels)
            loss = (
                lambda_contrastive * loss_con
                + lambda_family * loss_f
                + lambda_subtype * loss_s
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            family_pred = family_logits.argmax(dim=1)
            subtype_pred = subtype_logits.argmax(dim=1)
            subtype_pred_gated = gated_subtype_predictions(
                family_logits=family_logits,
                subtype_logits=subtype_logits,
                family_gate=family_gate,
            )

            batch_size = l1_labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_con += loss_con.item() * batch_size
            total_fce += loss_f.item() * batch_size
            total_sce += loss_s.item() * batch_size
            correct_fam += (family_pred == l1_labels).sum().item()
            correct_sub += (subtype_pred == l2_labels).sum().item()
            correct_sub_gated += (subtype_pred_gated == l2_labels).sum().item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                fam_acc=f"{(family_pred == l1_labels).float().mean().item():.2f}",
                sub_acc=f"{(subtype_pred == l2_labels).float().mean().item():.2f}",
            )

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / total_samples,
        total_con / total_samples,
        total_fce / total_samples,
        total_sce / total_samples,
        correct_fam / total_samples,
        correct_sub / total_samples,
        correct_sub_gated / total_samples,
    )


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CTA Hierarchical Contrastive Model (Option A)"
    )
    parser.add_argument(
        "--synthetic_path",
        type=str,
        default="training/synthetic_df_checkpoint.csv",
        help="Path to training CSV. Supports columns: name, values, (l1_label+l2_label) or label.",
    )
    parser.add_argument("--output_dir", type=str, default="training/model_hierarchical")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--name_repeat", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--weak_positive_weight", type=float, default=0.3)
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--hard_mining_topk", type=int, default=64)
    parser.add_argument("--hard_mining_k", type=int, default=32)
    parser.add_argument("--hard_mining_refresh", type=int, default=1)
    parser.add_argument(
        "--disable_hard_mining",
        action="store_true",
        help="Use random cross-family negatives only.",
    )
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_family", type=float, default=1.0)
    parser.add_argument("--lambda_subtype", type=float, default=1.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    l2_to_l1 = build_l2_to_l1_map()

    df = load_training_dataframe(
        csv_path=args.synthetic_path,
        name_repeat=args.name_repeat,
        l2_to_l1=l2_to_l1,
    )
    df = enforce_min_subtype_count(df, min_count=2)

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["l2_label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    l1_encoder = LabelEncoder()
    l2_encoder = LabelEncoder()
    train_df["l1_id"] = l1_encoder.fit_transform(train_df["l1_label"])
    train_df["l2_id"] = l2_encoder.fit_transform(train_df["l2_label"])

    val_df = val_df[
        val_df["l1_label"].isin(l1_encoder.classes_)
        & val_df["l2_label"].isin(l2_encoder.classes_)
    ].copy()
    if len(val_df) > 0:
        val_df["l1_id"] = l1_encoder.transform(val_df["l1_label"])
        val_df["l2_id"] = l2_encoder.transform(val_df["l2_label"])
    else:
        val_df["l1_id"] = pd.Series(dtype=int)
        val_df["l2_id"] = pd.Series(dtype=int)

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    print(f"Families (l1): {len(l1_encoder.classes_)} | Subtypes (l2): {len(l2_encoder.classes_)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
    )
    print(f"Tokenizer added {added_tokens} special tokens: {list(SPECIAL_TOKENS.values())}")

    train_dataset = CTAHierDataset(
        train_df["text"].tolist(),
        train_df["l1_id"].tolist(),
        train_df["l2_id"].tolist(),
        tokenizer,
        args.max_length,
    )
    val_dataset = CTAHierDataset(
        val_df["text"].tolist(),
        val_df["l1_id"].tolist(),
        val_df["l2_id"].tolist(),
        tokenizer,
        args.max_length,
    )

    train_sampler = HierarchicalBatchSampler(
        train_df["l1_id"].tolist(),
        train_df["l2_id"].tolist(),
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        seed=args.seed,
        drop_last=True,
    )
    val_sampler = HierarchicalBatchSampler(
        val_df["l1_id"].tolist() if len(val_df) > 0 else [0],
        val_df["l2_id"].tolist() if len(val_df) > 0 else [0],
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        seed=args.seed + 1,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers
    )

    embed_loader = DataLoader(
        EmbedDataset(train_df["text"].tolist(), tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CTAHierarchicalModel(
        model_name=args.model_name,
        num_l1=len(l1_encoder.classes_),
        num_l2=len(l2_encoder.classes_),
        embed_dim=args.embed_dim,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_train_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    supcon = HierarchicalSupConLoss(
        temperature=args.temperature, weak_weight=args.weak_positive_weight
    )
    family_gate = build_family_gate(
        l1_classes=l1_encoder.classes_.tolist(),
        l2_classes=l2_encoder.classes_.tolist(),
        l2_to_l1=l2_to_l1,
    )

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    best_val_score = -math.inf

    for epoch in range(1, args.epochs + 1):
        if not args.disable_hard_mining and (
            epoch == 1 or epoch % max(1, args.hard_mining_refresh) == 0
        ):
            embeddings, idxs = encode_all_embeddings(model, embed_loader, device)
            hard_negs = mine_hard_negatives_cosine(
                embeddings=embeddings,
                l1_ids=train_df.loc[idxs.numpy(), "l1_id"].to_numpy(),
                topk=args.hard_mining_topk,
                hard_k=args.hard_mining_k,
            )
            train_sampler.set_hard_negatives(hard_negs)
            print(
                f"[Epoch {epoch}] Refreshed hard negatives "
                f"(topk={args.hard_mining_topk}, hard_k={args.hard_mining_k})"
            )

        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            l1_ids=train_df["l1_id"].to_numpy(),
            l2_ids=train_df["l2_id"].to_numpy(),
            device=device,
            supcon=supcon,
            lambda_contrastive=args.lambda_contrastive,
            lambda_family=args.lambda_family,
            lambda_subtype=args.lambda_subtype,
            family_gate=family_gate,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip_norm=args.grad_clip_norm,
        )

        if len(val_df) > 0:
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                l1_ids=val_df["l1_id"].to_numpy(),
                l2_ids=val_df["l2_id"].to_numpy(),
                device=device,
                supcon=supcon,
                lambda_contrastive=args.lambda_contrastive,
                lambda_family=args.lambda_family,
                lambda_subtype=args.lambda_subtype,
                family_gate=family_gate,
                optimizer=None,
                scheduler=None,
                grad_clip_norm=args.grad_clip_norm,
            )
        else:
            val_stats = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        (
            train_total,
            train_con,
            train_fce,
            train_sce,
            train_facc,
            train_sacc,
            train_sacc_gated,
        ) = train_stats
        (
            val_total,
            val_con,
            val_fce,
            val_sce,
            val_facc,
            val_sacc,
            val_sacc_gated,
        ) = val_stats

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss={train_total:.4f}, "
            f"Train FamAcc={train_facc:.4f}, "
            f"Train SubAcc={train_sacc:.4f}, "
            f"Val Loss={val_total:.4f}, "
            f"Val FamAcc={val_facc:.4f}, "
            f"Val SubAcc={val_sacc:.4f}, "
            f"Val SubAcc(gated)={val_sacc_gated:.4f}"
        )

        current_score = val_sacc_gated if len(val_df) > 0 else train_sacc_gated
        if current_score > best_val_score:
            best_val_score = current_score
            torch.save(model.state_dict(), output_dir / "model.pt")
            print(f"  -> Saved best checkpoint (score={best_val_score:.4f})")

    tokenizer.save_pretrained(output_dir)
    model.encoder.config.save_pretrained(output_dir)

    l2_to_l1_label = {
        l2_label: l2_to_l1[l2_label] for l2_label in l2_encoder.classes_.tolist()
    }
    with open(output_dir / "label_encoder.json", "w") as f:
        json.dump(
            {
                "mode": "hierarchical_contrastive",
                "model_name": args.model_name,
                "embed_dim": args.embed_dim,
                "name_repeat": args.name_repeat,
                "special_tokens": SPECIAL_TOKENS,
                "classes": l2_encoder.classes_.tolist(),
                "l1_classes": l1_encoder.classes_.tolist(),
                "l2_classes": l2_encoder.classes_.tolist(),
                "l2_to_l1": l2_to_l1_label,
            },
            f,
            indent=2,
        )

    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training complete. Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
