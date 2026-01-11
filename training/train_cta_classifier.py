#!/usr/bin/env python3
"""
Column Type Annotation (CTA) Classifier Training Script

Supports Curriculum Learning Pipeline:
- Stage 1: Contrastive pre-training (encoder learns geometry)
- Stage 2: Classification fine-tuning (train classifier on fixed/frozen encoder)
- Stage 3: Optional combined multi-task polish (low alpha to avoid disruption)

Usage (Curriculum Learning - Recommended):
    # Stage 1: Pre-train encoder
    python train_cta_classifier.py --mode contrastive --epochs 20 --output_dir ./model_contrastive
    
    # Stage 2: Fine-tune classifier
    python train_cta_classifier.py --mode fine_tune \\
        --load_encoder_from ./model_contrastive/model.pt --epochs 10 \\
        --output_dir ./model_fine_tune
    
    # Stage 3 (optional): Multi-task polish
    python train_cta_classifier.py --mode combined \\
        --load_encoder_from ./model_fine_tune/model.pt --alpha 0.2 --epochs 5
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPECIAL_TOKENS = {"col_token": "[COL]", "val_token": "[VAL]"}


# ============================================================================
# Utilities
# ============================================================================


def mean_pool(outputs, attention_mask):
    """Mean pooling over sequence length."""
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_encoder_weights(model, checkpoint_path, strict=False):
    """Load encoder weights from checkpoint, handling different key formats."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Try encoder. prefix first
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    if not encoder_state:
        # Fallback: exclude classifier/projection
        encoder_state = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("classifier") and not k.startswith("projection")
        }
    model.encoder.load_state_dict(encoder_state, strict=strict)
    print(f"Loaded encoder weights from {checkpoint_path}")


def create_optimizer(model, lr, encoder_lr=None, freeze_encoder=False):
    """Create optimizer with optional differential learning rates."""
    if freeze_encoder:
        return torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    elif encoder_lr is not None and encoder_lr != lr:
        return torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.classifier.parameters(), "lr": lr},
            ]
        )
    return torch.optim.AdamW(model.parameters(), lr=lr)


def get_device():
    """Get available device."""
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


def encode_all(model, loader, device):
    model.eval()
    all_emb = []
    all_labels = []
    all_idx = []
    with torch.no_grad():
        for batch in loader:
            emb = model.get_embeddings(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )  # (b, d), normalized already
            all_emb.append(emb.cpu())
            all_labels.append(batch["labels"].cpu())
            all_idx.append(batch["idx"].cpu())
    return torch.cat(all_emb), torch.cat(all_labels), torch.cat(all_idx)


def mine_hard_negatives_cosine(embeddings, labels, topk=50, hard_k=20):
    """
    embeddings: (N, d) normalized
    labels: (N,)
    Return: dict i -> list of mined negative indices (global indices)
    """
    # cosine sim = dot product (since normalized)
    sim = embeddings @ embeddings.T  # (N, N)
    sim.fill_diagonal_(-1e9)

    # get topk nearest neighbors for each i
    nn_scores, nn_idx = torch.topk(sim, k=min(topk, embeddings.size(0) - 1), dim=1)

    hard_negs = {}
    labels_np = labels.numpy()

    for i in range(embeddings.size(0)):
        yi = labels_np[i]
        nbrs = nn_idx[i].tolist()
        # different label neighbors are hard negatives
        negs = [j for j in nbrs if labels_np[j] != yi]
        hard_negs[i] = negs[:hard_k]
    return hard_negs


# ============================================================================
# Data Loading
# ============================================================================


def load_training_data(synthetic_path="synthetic_df.csv", name_repeat=3):
    """Load and format synthetic training data."""
    if not Path(synthetic_path).exists():
        raise FileNotFoundError(
            f"Synthetic training data not found at {synthetic_path}!"
        )

    df = pd.read_csv(synthetic_path)[["name", "values", "label"]].copy()
    print(f"Loaded {len(df)} synthetic samples from {synthetic_path}")

    def make_text(row):
        col_tok, val_tok = SPECIAL_TOKENS["col_token"], SPECIAL_TOKENS["val_token"]
        name = row["name"]
        col_parts = (
            " ".join([f"{col_tok} {name}"] * name_repeat)
            if name_repeat > 1 and name
            else (f"{col_tok} {name}" if name else "")
        )
        val_list = [v.strip() for v in str(row["values"]).split(",")]
        val_parts = " ".join([f"{val_tok} {v}" for v in val_list[:10]])
        return f"{col_parts} {val_parts}".strip()

    df["text"] = df.apply(make_text, axis=1)
    print(f"Total training samples: {len(df)} (name_repeat={name_repeat})")
    return df


# ============================================================================
# Datasets
# ============================================================================


class CTADataset(Dataset):
    """Dataset for classification."""

    def __init__(self, texts, labels, tokenizer, non_spatial_id, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = labels
        self.non_spatial_id = non_spatial_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(label),
            "spatial_labels": torch.tensor(0 if label == self.non_spatial_id else 1),
        }


class EmbedDataset(Dataset):
    """Single-view dataset used to embed all samples for mining."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = np.array(labels)
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
            "labels": torch.tensor(int(self.labels[idx])),
            "idx": torch.tensor(idx),
        }


class TripletContrastiveDataset(Dataset):
    """
    Uses mined hard negatives.
    Each sample returns (anchor, positive, negative).
    """

    def __init__(self, texts, labels, hard_negs, tokenizer, max_length=128):
        self.texts = texts
        self.labels = np.array(labels)
        self.hard_negs = hard_negs  # dict: i -> list of neg indices
        self.tokenizer = tokenizer
        self.max_length = max_length

        # label -> indices for positive sampling
        self.label_to_indices = {}
        for i, y in enumerate(self.labels):
            self.label_to_indices.setdefault(int(y), []).append(i)

    def __len__(self):
        return len(self.texts)

    def _encode(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"])

    def __getitem__(self, idx):
        y = int(self.labels[idx])

        # positive
        pos_pool = self.label_to_indices[y]
        pos_idx = random.choice([i for i in pos_pool if i != idx] or [idx])

        # mined negative (fallback to random if missing)
        neg_pool = self.hard_negs.get(idx, [])
        if len(neg_pool) == 0:
            # random negative label
            other_labels = [k for k in self.label_to_indices.keys() if k != y]
            neg_label = random.choice(other_labels)
            neg_idx = random.choice(self.label_to_indices[neg_label])
        else:
            neg_idx = random.choice(neg_pool)

        a_ids, a_mask = self._encode(self.texts[idx])
        p_ids, p_mask = self._encode(self.texts[pos_idx])
        n_ids, n_mask = self._encode(self.texts[neg_idx])

        return {
            "a_input_ids": a_ids,
            "a_attention_mask": a_mask,
            "p_input_ids": p_ids,
            "p_attention_mask": p_mask,
            "n_input_ids": n_ids,
            "n_attention_mask": n_mask,
            "labels": torch.tensor(y),
        }


# ============================================================================
# Models
# ============================================================================


class BaseEncoder(nn.Module):
    """Base encoder wrapper with common functionality."""

    def __init__(self, model_name=None, config=None):
        super().__init__()
        model_name = model_name or MODEL_NAME
        if config is not None:
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


class CTAClassificationModel(nn.Module):
    """Classification model with optional spatial head."""

    def __init__(
        self, num_labels, model_name=None, config=None, use_spatial_head=False
    ):
        super().__init__()
        self.encoder = BaseEncoder(model_name, config).encoder
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.use_spatial_head = use_spatial_head
        if use_spatial_head:
            self.spatial_head = nn.Linear(hidden_size, 2)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None, spatial_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs, attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if self.use_spatial_head and spatial_labels is not None:
                loss = loss + 0.5 * F.cross_entropy(
                    self.spatial_head(pooled), spatial_labels
                )

        class Output:
            pass

        out = Output()
        out.loss = loss
        out.pooled = pooled
        out.logits = logits
        if self.use_spatial_head:
            out.spatial_logits = self.spatial_head(pooled)
        return out

    def load_encoder_weights(self, checkpoint_path, strict=False):
        load_encoder_weights(self, checkpoint_path, strict)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen - only classifier will be trained")


class CTAContrastiveModel(nn.Module):
    """Encoder with projection head for contrastive learning."""

    def __init__(self, embed_dim=128, num_labels=None, model_name=None, config=None):
        super().__init__()
        self.encoder = BaseEncoder(model_name, config).encoder
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )
        self.classifier = nn.Linear(hidden_size, num_labels) if num_labels else None

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs, attention_mask)
        if return_embeddings:
            return F.normalize(self.projection(pooled), dim=1)
        return self.classifier(pooled) if self.classifier else pooled

    def get_embeddings(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask, return_embeddings=True)


# ============================================================================
# Loss Functions
# ============================================================================


def info_nce_triplet(a, p, n, temperature=0.07):
    # a,p,n are normalized
    pos = (a * p).sum(dim=1) / temperature
    neg = (a * n).sum(dim=1) / temperature
    logits = torch.stack([pos, neg], dim=1)  # (b, 2)
    labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
    return F.cross_entropy(logits, labels)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device) - torch.eye(
            batch_size, device=device
        )

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) - exp_logits.diag().unsqueeze(1) + 1e-8
        )

        mask_sum = torch.clamp(mask.sum(dim=1), min=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        return -mean_log_prob_pos.mean()


class CombinedLoss(nn.Module):
    """Combined contrastive + classification loss."""

    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.supcon = SupConLoss(temperature)
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, embeddings, logits, labels):
        loss_con = self.supcon(embeddings, labels)
        loss_ce = self.ce(logits, labels)
        return self.alpha * loss_con + (1 - self.alpha) * loss_ce, loss_con, loss_ce


# ============================================================================
# Training Loops
# ============================================================================


def train_epoch(
    model, loader, device, optimizer=None, scheduler=None, criterion=None, mode="train"
):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"[{mode.capitalize()}]")

    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            if is_train:
                optimizer.zero_grad()

            # Classification
            spatial_labels = batch.get("spatial_labels")
            if spatial_labels is not None:
                spatial_labels = spatial_labels.to(device)

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                spatial_labels=spatial_labels,
            )

            labels = batch["labels"].to(device)
            logits = outputs.logits

            ce = F.cross_entropy(
                logits, labels, label_smoothing=getattr(model, "label_smoothing", 0.0)
            )

            loss = ce
            if getattr(model, "metric_alpha", 0.0) > 0:
                feats = F.normalize(outputs.pooled, dim=1)
                loss_con = SupConLoss(temperature=0.07)(feats, labels)
                loss = ce + model.metric_alpha * loss_con

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if not is_train:
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"].to(device)).sum().item()
                total += len(batch["labels"])

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    if is_train:
        return avg_loss
    return avg_loss, (correct / total if total > 0 else 0.0)


def train_classification(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    encoder_lr=None,
    freeze_encoder=False,
    output_dir="./model",
):
    """Classification training (Stage 2: fine-tuning)."""
    model.to(device)
    if freeze_encoder:
        model.freeze_encoder()

    optimizer = create_optimizer(model, lr, encoder_lr, freeze_encoder)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * len(train_loader) * epochs),
        num_training_steps=len(train_loader) * epochs,
    )

    best_val_acc = 0
    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, device, optimizer, scheduler, mode="train"
        )
        val_loss, val_acc = train_epoch(model, val_loader, device, mode="val")

        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/model.pt")
            print(f"  → Saved best model (acc={val_acc:.4f})")

    return model


def train_contrastive(
    model,
    tokenizer,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    device,
    epochs=10,
    lr=2e-5,
    temperature=0.07,
    max_length=128,
    batch_size=32,
    mine_topk=50,
    mine_hard_k=20,
    output_dir="./model_contrastive",
):
    """Contrastive learning training (Stage 1: pre-training)."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # loader for embedding all samples each epoch
    embed_ds = EmbedDataset(train_texts, train_labels, tokenizer, max_length)
    embed_loader = DataLoader(embed_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # 1) mine hard negatives using current model
        embs, labs, _ = encode_all(model, embed_loader, device)
        hard_negs = mine_hard_negatives_cosine(
            embs, labs, topk=mine_topk, hard_k=mine_hard_k
        )

        # 2) build triplet dataset for this epoch
        triplet_ds = TripletContrastiveDataset(
            train_texts, train_labels, hard_negs, tokenizer, max_length=max_length
        )
        # Validation uses empty hard_negs (falls back to random negatives)
        val_ds = TripletContrastiveDataset(
            val_texts, val_labels, {}, tokenizer, max_length=max_length
        )
        train_loader = DataLoader(triplet_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 3) train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()

            a = model.get_embeddings(
                batch["a_input_ids"].to(device), batch["a_attention_mask"].to(device)
            )
            p = model.get_embeddings(
                batch["p_input_ids"].to(device), batch["p_attention_mask"].to(device)
            )
            n = model.get_embeddings(
                batch["n_input_ids"].to(device), batch["n_attention_mask"].to(device)
            )

            loss = info_nce_triplet(a, p, n, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 4) validation (you can keep your current val loop; shown as loss-only)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                a = model.get_embeddings(
                    batch["a_input_ids"].to(device),
                    batch["a_attention_mask"].to(device),
                )
                p = model.get_embeddings(
                    batch["p_input_ids"].to(device),
                    batch["p_attention_mask"].to(device),
                )
                n = model.get_embeddings(
                    batch["n_input_ids"].to(device),
                    batch["n_attention_mask"].to(device),
                )

                loss = info_nce_triplet(a, p, n, temperature=temperature)
                val_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}: Val Loss={avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/model.pt")
            print(f"  → Saved best model (loss={best_val_loss:.4f})")

    return model


# ============================================================================
# Mode Handlers
# ============================================================================


def run_classification(
    args,
    train_texts,
    val_texts,
    train_labels,
    val_labels,
    tokenizer,
    num_labels,
    non_spatial_id,
    device,
):
    """Run standard classification from scratch."""
    train_ds = CTADataset(
        train_texts, train_labels, tokenizer, non_spatial_id, args.max_length
    )
    val_ds = CTADataset(
        val_texts, val_labels, tokenizer, non_spatial_id, args.max_length
    )

    model = CTAClassificationModel(num_labels, use_spatial_head=args.use_spatial_head)
    model.encoder.resize_token_embeddings(len(tokenizer))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    train_classification(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        output_dir=args.output_dir,
    )
    tokenizer.save_pretrained(args.output_dir)
    return model


def run_contrastive(
    args,
    train_texts,
    val_texts,
    train_labels,
    val_labels,
    tokenizer,
    num_labels,
    non_spatial_id,
    device,
):
    """Run Stage 1: Contrastive pre-training."""
    print("\n" + "=" * 60)
    print("STAGE 1: Contrastive Pre-training")
    print("=" * 60)

    model = CTAContrastiveModel(embed_dim=args.embed_dim, num_labels=None)
    model.encoder.resize_token_embeddings(len(tokenizer))

    train_contrastive(
        model,
        tokenizer,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        device,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        max_length=args.max_length,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Stage 1 complete! Encoder saved to {args.output_dir}/model.pt")
    print(
        f"   Next: Run --mode fine_tune --load_encoder_from {args.output_dir}/model.pt"
    )
    return model


def run_fine_tune(
    args,
    train_texts,
    val_texts,
    train_labels,
    val_labels,
    tokenizer,
    num_labels,
    non_spatial_id,
    device,
):
    """Run Stage 2: Classification fine-tuning."""
    print("\n" + "=" * 60)
    print("STAGE 2: Classification Fine-tuning")
    print("=" * 60)

    if not args.load_encoder_from or not Path(args.load_encoder_from).exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {args.load_encoder_from}"
        )

    checkpoint_dir = Path(args.load_encoder_from).parent
    config = (
        AutoConfig.from_pretrained(str(checkpoint_dir))
        if (checkpoint_dir / "config.json").exists()
        else None
    )

    model = CTAClassificationModel(
        num_labels, config=config, use_spatial_head=args.use_spatial_head
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_encoder_weights(args.load_encoder_from, strict=False)
    model.metric_alpha = args.metric_alpha
    model.label_smoothing = args.label_smoothing

    if args.freeze_encoder:
        model.freeze_encoder()

    train_ds = CTADataset(
        train_texts, train_labels, tokenizer, non_spatial_id, args.max_length
    )
    val_ds = CTADataset(
        val_texts, val_labels, tokenizer, non_spatial_id, args.max_length
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    train_classification(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        args.encoder_lr,
        args.freeze_encoder,
        args.output_dir,
    )
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Stage 2 complete! Model saved to {args.output_dir}/model.pt")
    print(
        f"   Optional: Run --mode combined --load_encoder_from {args.output_dir}/model.pt --alpha 0.2"
    )
    return model


def run_combined(
    args,
    train_texts,
    val_texts,
    train_labels,
    val_labels,
    tokenizer,
    num_labels,
    non_spatial_id,
    device,
):
    """Run Stage 3: Combined multi-task polish."""
    print("\n" + "=" * 60)
    print("STAGE 3: Combined Multi-task Fine-tuning")
    print("=" * 60)
    print(f"⚠️  Using alpha={args.alpha}. Should be low (0.1-0.3) after pre-training.")

    train_ds = CTADataset(
        train_texts, train_labels, tokenizer, non_spatial_id, args.max_length
    )
    val_ds = CTADataset(
        val_texts, val_labels, tokenizer, non_spatial_id, args.max_length
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    checkpoint_dir = (
        Path(args.load_encoder_from).parent if args.load_encoder_from else None
    )
    config = (
        AutoConfig.from_pretrained(str(checkpoint_dir))
        if checkpoint_dir and (checkpoint_dir / "config.json").exists()
        else None
    )

    # IMPORTANT: use CTAClassificationModel so train_epoch works (labels, pooled, logits)
    model = CTAClassificationModel(
        num_labels, config=config, use_spatial_head=args.use_spatial_head
    )
    model.encoder.resize_token_embeddings(len(tokenizer))

    # IMPORTANT: load the FULL fine-tune checkpoint (encoder + classifier)
    state = torch.load(args.load_encoder_from, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded full checkpoint for Stage 3")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Stage 3 = small supervised contrastive regularizer
    model.metric_alpha = args.alpha
    model.label_smoothing = args.label_smoothing

    # Use lower LR than stage2 (polish)
    lr = min(args.lr, 1e-5)
    encoder_lr = min(args.encoder_lr, lr) if args.encoder_lr is not None else None

    train_classification(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=lr,
        encoder_lr=encoder_lr,
        freeze_encoder=False,
        output_dir=args.output_dir,
    )

    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Stage 3 complete! Model saved to {args.output_dir}/model.pt")
    return model


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train CTA Classifier with Curriculum Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Curriculum Learning Pipeline:
  1. Stage 1 (contrastive): Pre-train encoder with contrastive learning
     python train_cta_classifier.py --mode contrastive --epochs 20 --output_dir ./model_contrastive
  
  2. Stage 2 (fine_tune): Load encoder, fine-tune classifier
     python train_cta_classifier.py --mode fine_tune \\
         --load_encoder_from ./model_contrastive/model.pt --epochs 10
  
  3. Stage 3 (combined, optional): Multi-task polish with low alpha
     python train_cta_classifier.py --mode combined \\
         --load_encoder_from ./model_fine_tune/model.pt --alpha 0.2 --epochs 5
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classification",
        choices=["classification", "contrastive", "fine_tune", "combined"],
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=None,
        help="Separate LR for encoder (fine_tune mode)",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder, only train classifier",
    )
    parser.add_argument(
        "--load_encoder_from",
        type=str,
        default=None,
        help="Path to pretrained encoder checkpoint",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Contrastive loss weight (combined mode, use 0.1-0.3)",
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./model")
    parser.add_argument("--synthetic_path", type=str, default="synthetic_df.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name_repeat", type=int, default=3)
    parser.add_argument("--use_spatial_head", action="store_true")
    parser.add_argument(
        "--metric_alpha",
        type=float,
        default=0.1,
        help="Aux supervised contrastive weight during fine-tune (try 0.05-0.2)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing (try 0.05-0.1)",
    )
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    # Load and prepare data
    df = load_training_data(args.synthetic_path, args.name_repeat)
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    non_spatial_id = label_encoder.transform(["non_spatial"])[0]
    num_labels = len(label_encoder.classes_)
    print(f"Number of classes: {num_labels}")
    print(f"Classes: {label_encoder.classes_.tolist()}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label_id"].tolist(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label_id"],
    )
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
    )
    print(f"Using model: {MODEL_NAME}")
    print(f"Added {num_added} special tokens: {list(SPECIAL_TOKENS.values())}")

    # Run training
    mode_handlers = {
        "classification": run_classification,
        "contrastive": run_contrastive,
        "fine_tune": run_fine_tune,
        "combined": run_combined,
    }

    model = mode_handlers[args.mode](
        args,
        train_texts,
        val_texts,
        train_labels,
        val_labels,
        tokenizer,
        num_labels,
        non_spatial_id,
        device,
    )

    # Save metadata
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/label_encoder.json", "w") as f:
        json.dump(
            {
                "classes": label_encoder.classes_.tolist(),
                "mode": args.mode,
                "model_name": MODEL_NAME,
                "embed_dim": args.embed_dim if args.mode != "classification" else None,
                "name_repeat": args.name_repeat,
                "special_tokens": SPECIAL_TOKENS,
            },
            f,
            indent=2,
        )

    model.encoder.config.save_pretrained(args.output_dir)
    print(f"\n✅ Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
