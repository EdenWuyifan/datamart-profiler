# Hierarchical CTA Training (2-Level Model)

This document explains the training logic in `training/train_cta_classifier_hierarchical.py` in implementation detail.

## 1. What this trainer does

The script trains a single encoder model for two related tasks:

1. `l1` family classification (coarse class, e.g. `spatial`, `temporal`).
2. `l2` subtype classification (fine class, e.g. `latitude`, `zip5`, `month_name`).

It combines:

- Hierarchical supervised contrastive learning on embeddings.
- Cross-entropy losses for both family and subtype heads.
- Hard-negative mining refreshed during training.

## 2. Label ontology and class hierarchy

The hierarchy is defined by `TWO_LEVEL_ONTOLOGY` in `training/atlas_types.py`:

- Keys are `l1` family labels.
- Values are lists of `l2` subtype labels.

At startup, `build_l2_to_l1_map()` creates a reverse map `l2 -> l1` and fails if any `l2` appears under multiple families.

## 3. Expected training CSV schema

Input CSV (`--synthetic_path`) must include:

- `name`
- `values`

And either:

- `l1_label` and `l2_label`, or
- `label` (treated as `l2_label`, then mapped to `l1_label` via ontology).

Validation performed in `load_training_dataframe()`:

- Missing required columns -> error.
- Any `l2` not present in ontology -> error.
- Empty formatted text rows are dropped.
- If all rows drop out -> error.

## 4. Text construction

Each training row is converted to one text string:

```text
[COL] <name> [COL] <name> ... [VAL] <v1> [VAL] <v2> ... [VAL] <v10>
```

Behavior:

- Column name is repeated `--name_repeat` times (default `3`) when non-empty.
- Up to first 10 comma-separated values are used.
- Special tokens are `"[COL]"` and `"[VAL]"`.

Tokenizer flow:

- Base model default is `BAAI/bge-base-en-v1.5`.
- `"[COL]"` and `"[VAL]"` are added as additional special tokens.
- Encoder embedding table is resized after tokenizer extension.

## 5. Pre-filter before split

`enforce_min_subtype_count(min_count=2)` removes `l2` classes with fewer than 2 rows globally.

Reason: strong positives in contrastive learning require at least two examples of the same `l2`.

## 6. Train/validation split and encoding

Data split:

- `train_test_split(..., stratify=df["l2_label"], test_size=--test_size, random_state=--seed)`.

Label encoding:

- `LabelEncoder` for `l1` and `l2`, fit on train split only.
- Validation rows with unseen train classes are filtered out.
- Encoded columns: `l1_id`, `l2_id`.

## 7. Hierarchical batch sampler

`HierarchicalBatchSampler` builds batches with explicit structure:

- `group_size = 3 + num_negatives`
- `anchors_per_batch = max(1, batch_size // group_size)`

Per anchor group:

1. `anchor`: one row from a chosen family and subtype.
2. `strong positive`: same family, same subtype, different row when possible.
3. `weak positive`: same family, different subtype.
4. `num_negatives` negatives: cross-family, preferably hard-mined.

Family eligibility (`l1_candidates`) for anchor sampling:

- Family must contain at least 2 different `l2` subtypes.
- At least one subtype in that family must have >=2 rows.

If batch still has free slots after anchor groups, extra examples are filled from candidate families.

Important constraints:

- `batch_size` must be >= 3.
- If no eligible family exists, sampler raises an error.

## 8. Model architecture

`CTAHierarchicalModel`:

1. Transformer encoder (`AutoModel`).
2. Mean pooling over token embeddings (attention-mask weighted).
3. Projection head: `Linear -> ReLU -> Linear` to `embed_dim` (default `128`).
4. L2-normalization of projected embedding.
5. Two classification heads from embedding:
   - `family_head` (`num_l1` logits)
   - `subtype_head` (`num_l2` logits)

Forward returns:

- `embeddings`, `family_logits`, `subtype_logits`.

## 9. Losses

### 9.1 Hierarchical supervised contrastive loss

`HierarchicalSupConLoss(temperature, weak_weight)`:

- Similarity matrix: `sim(i,j) = emb_i dot emb_j / temperature`.
- Self-pairs excluded.
- Positive weighting:
  - Strong positives: same `l2` (weight `1.0`).
  - Weak positives: same `l1` but different `l2` (weight `weak_weight`, default `0.3`).

Loss per sample is weighted negative log-probability over its positives.

### 9.2 Classification losses

- Family CE: `cross_entropy(family_logits, l1_labels)`.
- Subtype CE: `cross_entropy(subtype_logits, l2_labels)`.

### 9.3 Total loss

```text
total =
  lambda_contrastive * contrastive_loss +
  lambda_family      * family_ce +
  lambda_subtype     * subtype_ce
```

All three lambda defaults are `1.0`.

## 10. Hard-negative mining

When enabled (`--disable_hard_mining` not set), at epoch 1 and every `--hard_mining_refresh` epochs:

1. Encode all train rows via `EmbedDataset` and current model.
2. Compute cosine-like similarity matrix (dot product of normalized embeddings).
3. For each sample:
   - Take top `--hard_mining_topk` nearest neighbors (excluding self).
   - Keep only cross-family neighbors.
   - Store first `--hard_mining_k` as hard negatives.

Sampler then uses these hard negatives first, with random cross-family negatives as fallback.

## 11. Gated subtype prediction metric

A family-to-subtype boolean mask (`family_gate`) is created from ontology and class encoders.

For evaluation metric `SubAcc(gated)`:

1. Predict family via `argmax(family_logits)`.
2. Mask subtype logits to only subtypes allowed for that predicted family.
3. Predict subtype on masked logits.

This reports hierarchical consistency-aware subtype accuracy.

## 12. Epoch execution details

`run_epoch(...)` does train or eval depending on whether optimizer is passed:

- Train mode:
  - zero grad
  - backward
  - gradient clipping (`--grad_clip_norm`, default `1.0`)
  - optimizer step
  - LR scheduler step
- Eval mode:
  - no grad, no optimization

Tracked outputs (averaged over samples):

- `total_loss`
- `contrastive_loss`
- `family_ce`
- `subtype_ce`
- `family_accuracy`
- `subtype_accuracy`
- `subtype_accuracy_gated`

Progress bar postfix shows per-batch `loss`, `fam_acc`, `sub_acc`.

## 13. Optimization setup

- Optimizer: `AdamW(lr=--lr, weight_decay=--weight_decay)`.
- Scheduler: linear warmup/decay with
  - `total_train_steps = len(train_loader) * epochs`
  - `warmup_steps = int(total_train_steps * warmup_ratio)`.

## 14. Checkpointing and model selection

Per epoch, trainer prints train/val aggregate stats.

Best checkpoint criterion:

- If validation exists: maximize `val_subtype_accuracy_gated`.
- Else: maximize `train_subtype_accuracy_gated`.

When improved, save `model.pt`.

## 15. Saved artifacts

In `--output_dir`:

- `model.pt`: full `state_dict` for hierarchical model.
- Hugging Face tokenizer files (`tokenizer.save_pretrained`).
- Encoder config (`model.encoder.config.save_pretrained`).
- `label_encoder.json` with:
  - mode, model name, embed dim, `name_repeat`
  - special tokens
  - `l1_classes`, `l2_classes`, `l2_to_l1` mapping
  - legacy `classes` key (same as `l2_classes`)
- `training_args.json`: full CLI arguments used.

## 16. Main CLI arguments

Core:

- `--synthetic_path`
- `--output_dir`
- `--model_name`
- `--epochs`
- `--batch_size`
- `--max_length`
- `--name_repeat`

Optimization:

- `--lr`
- `--weight_decay`
- `--warmup_ratio`
- `--grad_clip_norm`

Representation/loss:

- `--embed_dim`
- `--temperature`
- `--weak_positive_weight`
- `--lambda_contrastive`
- `--lambda_family`
- `--lambda_subtype`

Sampling/mining:

- `--num_negatives`
- `--hard_mining_topk`
- `--hard_mining_k`
- `--hard_mining_refresh`
- `--disable_hard_mining`

Data split/repro:

- `--test_size`
- `--seed`
- `--num_workers`

## 17. Example run

```bash
python training/train_cta_classifier_hierarchical.py \
  --synthetic_path training/synthetic_df.csv \
  --output_dir ./model_hierarchical \
  --epochs 20 \
  --weak_positive_weight 0.3 \
  --lambda_contrastive 1.0 \
  --lambda_family 1.0 \
  --lambda_subtype 1.0
```

## 18. Practical notes

- The sampler requires hierarchical diversity; tiny or imbalanced splits can fail eligibility checks.
- Because validation also uses `HierarchicalBatchSampler`, validation data must also satisfy sampler constraints.
- Larger `num_negatives` increases cross-family pressure but reduces anchors per batch for fixed `batch_size`.

## 19. Hierarchical Inference Script

Use `training/inference_cta_hierarchical.py` to run inference with:

- predicted `l1` family
- predicted `l2` subtype
- datatype lookup for predicted `l2`

Single input:

```bash
python training/inference_cta_hierarchical.py \
  --model_dir model_hierarchical \
  --text "zip_code: 10001, 11201, 10013"
```

Column + values input:

```bash
python training/inference_cta_hierarchical.py \
  --model_dir model_hierarchical \
  --column customer_id \
  --values "CUST-000123, CUST-000124, CUST-000125"
```

Batch CSV inference:

```bash
python training/inference_cta_hierarchical.py \
  --model_dir model_hierarchical \
  --input_csv training/synthetic_df.csv \
  --name_col name \
  --values_col values \
  --output_path predictions.csv
```

Notes:

- Subtype prediction is family-gated by default to enforce hierarchical consistency.
- Use `--no_gated_subtype` to disable gating.
- If you want datatype mapping from a custom CSV, pass `--datatype_map_csv`.
