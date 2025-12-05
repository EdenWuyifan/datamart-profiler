#!/usr/bin/env python3
"""
Column Type Annotation (CTA) Classifier Training Script

Supports:
- Standard classification with DistilBERT
- Supervised Contrastive Learning for better embeddings
- GPU acceleration
- Combined training (contrastive + classification)

Usage:
    python train_cta_classifier.py --mode classification --epochs 10
    python train_cta_classifier.py --mode contrastive --epochs 20
    python train_cta_classifier.py --mode combined --epochs 15
"""

import argparse
import json
import os
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
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Model configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Strong embedding model, 33M params

# Special tokens for structured input
SPECIAL_TOKENS = {
    "col_token": "[COL]",  # Marks column name
    "val_token": "[VAL]",  # Marks values section
}


# ============================================================================
# Data Loading
# ============================================================================


def load_training_data(
    curated_path: str = "curated_spatial_cta.csv",
    synthetic_path: str = "synthetic_df.csv",
    name_repeat: int = 3,
) -> pd.DataFrame:
    """Load and combine curated + synthetic training data.

    Args:
        name_repeat: Number of times to repeat column name in text (emphasizes name).
    """
    dfs = []

    if Path(curated_path).exists():
        curated = pd.read_csv(curated_path)
        curated_df = pd.DataFrame(
            {
                "name": curated["Column"],
                "values": curated["Values"],
                "label": curated["Label"],
            }
        )
        dfs.append(curated_df)
        print(f"Loaded {len(curated_df)} samples from {curated_path}")

    if Path(synthetic_path).exists():
        synthetic = pd.read_csv(synthetic_path)
        dfs.append(synthetic[["name", "values", "label"]])
        print(f"Loaded {len(synthetic)} samples from {synthetic_path}")

    if not dfs:
        raise FileNotFoundError("No training data found!")

    df = pd.concat(dfs, ignore_index=True)

    # Create text with structured tokens
    # Format: [COL] Borough [COL] Borough [VAL] Manhattan [VAL] Brooklyn
    def make_text(row):
        name = row["name"]
        values = row["values"]
        col_tok = SPECIAL_TOKENS["col_token"]
        val_tok = SPECIAL_TOKENS["val_token"]

        # Repeat column name with [COL] token for emphasis
        if name_repeat > 1 and name:
            col_parts = " ".join([f"{col_tok} {name}"] * name_repeat)
        else:
            col_parts = f"{col_tok} {name}" if name else ""

        # Format values with [VAL] token
        val_list = [v.strip() for v in str(values).split(",")]
        val_parts = " ".join([f"{val_tok} {v}" for v in val_list[:10]])  # Limit values

        return f"{col_parts} {val_parts}".strip()

    df["text"] = df.apply(make_text, axis=1)
    print(f"Total training samples: {len(df)} (name_repeat={name_repeat})")
    return df


# ============================================================================
# Dataset Classes
# ============================================================================


class CTADataset(Dataset):
    """Dataset for classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx]),
        }


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with anchor-positive pairs."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = np.array(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Build label-to-indices mapping for positive sampling
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        anchor_text = self.texts[idx]
        anchor_label = self.labels[idx]

        # Sample a positive (same class, different sample)
        pos_indices = self.label_to_indices[anchor_label]
        pos_idx = np.random.choice([i for i in pos_indices if i != idx] or [idx])
        pos_text = self.texts[pos_idx]

        anchor_enc = self.tokenizer(
            anchor_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        pos_enc = self.tokenizer(
            pos_text, truncation=True, padding="max_length", max_length=self.max_length
        )

        return {
            "anchor_input_ids": torch.tensor(anchor_enc["input_ids"]),
            "anchor_attention_mask": torch.tensor(anchor_enc["attention_mask"]),
            "pos_input_ids": torch.tensor(pos_enc["input_ids"]),
            "pos_attention_mask": torch.tensor(pos_enc["attention_mask"]),
            "labels": torch.tensor(anchor_label),
        }


# ============================================================================
# Model Classes
# ============================================================================


class CTAClassificationModel(nn.Module):
    """Simple classification model using any transformer encoder."""

    def __init__(self, model_name=None, num_labels=None, config=None):
        super().__init__()
        model_name = model_name or MODEL_NAME

        if config is not None:
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        # Return object-like result for compatibility
        class Output:
            pass

        out = Output()
        out.loss = loss
        out.logits = logits
        return out


class CTAContrastiveModel(nn.Module):
    """Encoder with projection head for contrastive learning (supports BGE, BERT, etc.)."""

    def __init__(self, model_name=None, embed_dim=128, num_labels=None, config=None):
        super().__init__()
        model_name = model_name or MODEL_NAME

        # Load encoder from pretrained or config
        if config is not None:
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )

        # Classification head (optional)
        self.classifier = nn.Linear(hidden_size, num_labels) if num_labels else None

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]

        if return_embeddings:
            return F.normalize(self.projection(pooled), dim=1)

        if self.classifier:
            return self.classifier(pooled)
        return pooled

    def get_embeddings(self, input_ids, attention_mask):
        """Get normalized embeddings for contrastive learning."""
        return self.forward(input_ids, attention_mask, return_embeddings=True)


# ============================================================================
# Loss Functions
# ============================================================================


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Reference: https://arxiv.org/abs/2004.11362
    Pulls together samples of the same class, pushes apart different classes.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, embed_dim) normalized embeddings
            labels: (batch_size,) class labels
        """
        device = features.device
        batch_size = features.shape[0]

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same class, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask - torch.eye(batch_size, device=device)  # Exclude self

        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits)
        # Exclude self from denominator
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) - exp_logits.diag().unsqueeze(1) + 1e-8
        )

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1)  # Avoid division by zero
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss


class CombinedLoss(nn.Module):
    """Combined contrastive + classification loss."""

    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.supcon = SupConLoss(temperature)
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha  # Weight for contrastive loss

    def forward(self, embeddings, logits, labels):
        loss_con = self.supcon(embeddings, labels)
        loss_ce = self.ce(logits, labels)
        return self.alpha * loss_con + (1 - self.alpha) * loss_ce, loss_con, loss_ce


# ============================================================================
# Training Functions
# ============================================================================


def train_classification(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=2e-5,
    output_dir="./model",
):
    """Standard classification training."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"].to(device)).sum().item()
                total += len(batch["labels"])

        val_acc = correct / total
        print(
            f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
            f"Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/model.pt")
            print(f"  → Saved best model (acc={val_acc:.4f})")

    return model


def train_contrastive(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=2e-5,
    temperature=0.07,
    output_dir="./model_contrastive",
):
    """Contrastive learning training."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = SupConLoss(temperature=temperature)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()

            # Get embeddings for anchor and positive
            anchor_emb = model.get_embeddings(
                batch["anchor_input_ids"].to(device),
                batch["anchor_attention_mask"].to(device),
            )
            pos_emb = model.get_embeddings(
                batch["pos_input_ids"].to(device),
                batch["pos_attention_mask"].to(device),
            )

            # Combine embeddings and labels for SupCon loss
            embeddings = torch.cat([anchor_emb, pos_emb], dim=0)
            labels = batch["labels"].to(device)
            labels = torch.cat([labels, labels], dim=0)

            loss = criterion(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                anchor_emb = model.get_embeddings(
                    batch["anchor_input_ids"].to(device),
                    batch["anchor_attention_mask"].to(device),
                )
                pos_emb = model.get_embeddings(
                    batch["pos_input_ids"].to(device),
                    batch["pos_attention_mask"].to(device),
                )
                embeddings = torch.cat([anchor_emb, pos_emb], dim=0)
                labels = torch.cat(
                    [batch["labels"].to(device), batch["labels"].to(device)], dim=0
                )
                val_loss += criterion(embeddings, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
            f"Val Loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/model.pt")
            print(f"  → Saved best model (loss={avg_val_loss:.4f})")

    return model


def train_combined(
    model,
    train_loader,
    val_loader,
    device,
    epochs=15,
    lr=2e-5,
    temperature=0.07,
    alpha=0.5,
    output_dir="./model_combined",
):
    """Combined contrastive + classification training."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = CombinedLoss(temperature=temperature, alpha=alpha)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_con, train_ce = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            embeddings = model.get_embeddings(input_ids, attention_mask)
            logits = model(input_ids, attention_mask)

            loss, loss_con, loss_ce = criterion(embeddings, logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_con += loss_con.item()
            train_ce += loss_ce.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                val_loss += F.cross_entropy(logits, labels).item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_acc = correct / total
        print(
            f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f} "
            f"(Con={train_con/len(train_loader):.4f}, CE={train_ce/len(train_loader):.4f}), "
            f"Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/model.pt")
            print(f"  → Saved best model (acc={val_acc:.4f})")

    return model


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train CTA Classifier")
    parser.add_argument(
        "--mode",
        type=str,
        default="classification",
        choices=["classification", "contrastive", "combined"],
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="Contrastive temperature"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Contrastive loss weight (combined mode)",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension for contrastive"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Max sequence length"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./model", help="Output directory"
    )
    parser.add_argument("--curated_path", type=str, default="curated_spatial_cta.csv")
    parser.add_argument("--synthetic_path", type=str, default="synthetic_df.csv")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--name_repeat",
        type=int,
        default=3,
        help="Repeat column name N times to emphasize it (default: 3)",
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    df = load_training_data(
        args.curated_path,
        args.synthetic_path,
        name_repeat=args.name_repeat,
    )

    # Encode labels
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    num_labels = len(label_encoder.classes_)
    print(f"Number of classes: {num_labels}")
    print(f"Classes: {label_encoder.classes_.tolist()}")

    # Train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label_id"].tolist(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label_id"],
    )
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # Tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Add custom tokens for structured input: [COL] column_name [VAL] value1 [VAL] value2
    new_tokens = list(SPECIAL_TOKENS.values())
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Using model: {MODEL_NAME}")
    print(f"Added {num_added} special tokens: {new_tokens}")

    # Create datasets and loaders based on mode
    if args.mode == "classification":
        train_ds = CTADataset(train_texts, train_labels, tokenizer, args.max_length)
        val_ds = CTADataset(val_texts, val_labels, tokenizer, args.max_length)

        model = CTAClassificationModel(num_labels=num_labels)
        model.encoder.resize_token_embeddings(
            len(tokenizer)
        )  # Accommodate new special tokens

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = train_classification(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.output_dir,
        )

        # Save tokenizer and label encoder
        tokenizer.save_pretrained(args.output_dir)

    elif args.mode == "contrastive":
        train_ds = ContrastiveDataset(
            train_texts, train_labels, tokenizer, args.max_length
        )
        val_ds = ContrastiveDataset(val_texts, val_labels, tokenizer, args.max_length)

        model = CTAContrastiveModel(embed_dim=args.embed_dim)
        model.encoder.resize_token_embeddings(
            len(tokenizer)
        )  # Accommodate new special tokens

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = train_contrastive(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.temperature,
            args.output_dir,
        )

    else:  # combined
        train_ds = CTADataset(train_texts, train_labels, tokenizer, args.max_length)
        val_ds = CTADataset(val_texts, val_labels, tokenizer, args.max_length)

        model = CTAContrastiveModel(embed_dim=args.embed_dim, num_labels=num_labels)
        model.encoder.resize_token_embeddings(
            len(tokenizer)
        )  # Accommodate new special tokens

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = train_combined(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.temperature,
            args.alpha,
            args.output_dir,
        )

    # Save label encoder mapping and config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/label_encoder.json", "w") as f:
        json.dump(
            {
                "classes": label_encoder.classes_.tolist(),
                "mode": args.mode,
                "model_name": MODEL_NAME,
                "embed_dim": args.embed_dim if args.mode != "classification" else None,
                "name_repeat": args.name_repeat,
                "special_tokens": SPECIAL_TOKENS,  # Save token format
            },
            f,
            indent=2,
        )

    # Save encoder config for fast offline loading
    model.encoder.config.save_pretrained(args.output_dir)

    print(f"\n✅ Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
