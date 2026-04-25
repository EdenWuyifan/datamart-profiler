# CTA Training

This directory contains the synthetic data generation and model training scripts for the Atlas Profiler CTA model.

Run the commands below from this directory:

```bash
cd /scratch/yfw215/OSCUR/atlas-profiler/training
```

## Pipeline

1. Generate synthetic CTA data with `generate_synthetic_cta.py`.
2. Train the CTA classifier with `train_cta_classifier.py`.
3. Use the trained model from `model_combined/` or copy the desired checkpoint into the package model directory.

## Environment Variables

Synthetic data generation uses an LLM provider. Set these before running `generate_synthetic_cta.py`:

```bash
export PORTKEY_API_KEY="..."
```

## Generate Synthetic Training Data

`generate_synthetic_cta.py` augments `curated_spatial_cta.csv` with diverse column names and values, including hard-negative pairs for confusing spatial labels.

Default generation:

```bash
python generate_synthetic_cta.py
```

Equivalent explicit command:

```bash
python generate_synthetic_cta.py \
  --mode both \
  --target 120 \
  --curated-csv curated_spatial_cta.csv \
  --output synthetic_df.csv \
  --cache synthetic_df_checkpoint.csv
```

Useful generation options:

| Argument | Default | Description |
| --- | --- | --- |
| `--mode` | `both` | Generate single-label examples, hard-negative pairs, or both. |
| `--target` | `120` | Target number of single-mode samples per label. |
| `--n-values` | `3` | Number of sample values per synthetic column. |
| `--n-rows-per-call` | `5` | Rows requested per single-mode LLM call. |
| `--n-pairs-per-label-pair` | `5` | Rows requested for each hard-negative label pair. |
| `--reset-cache` | off | Ignore and overwrite the existing checkpoint cache. |

Inputs:

- `curated_spatial_cta.csv` - hand-labeled training examples.

Outputs:

- `synthetic_df.csv` - augmented training data.
- `synthetic_df_checkpoint.csv` - incremental checkpoint for resuming generation.

## Train the CTA Classifier

`train_cta_classifier.py` trains a transformer classifier using `BAAI/bge-base-en-v1.5` as the encoder.

The CTA model is the profiler's spatial column classifier. It does not replace the profiler's full structural and semantic type detector; instead, accepted spatial predictions are mapped into the runtime profiler's type system.

Current CTA labels include:

- Coordinate labels: `latitude`, `longitude`, `x_coord`, `y_coord`.
- Geometry labels: `point`, `line`, `polygon`, `multi-line`, `multi-polygon`.
- Address/admin labels: `zip_code`, `zip5`, `zip9`, `address`, `borough`, `borough_code`, `city`, `state`, `state_code`, `country`.
- NYC identifier labels: `bbl`, `bin`.
- Fallback label: `non_spatial`.

Training modes:

| Mode | Description | Best for |
| --- | --- | --- |
| `classification` | Standard cross-entropy training from scratch. | Fast baseline. |
| `contrastive` | Stage 1 contrastive encoder pre-training. | Better embedding geometry. |
| `fine_tune` | Stage 2 classifier fine-tuning from the contrastive encoder. | Main supervised model. |
| `combined` | Stage 3 classifier polish with supervised contrastive regularization. | Final optional polish. |

`combined` expects a full fine-tuned checkpoint from Stage 2, not only a contrastive encoder.

## Recommended Curriculum

The training defaults now match the current three-stage workflow, so each stage only needs `--mode`.

```bash
# Stage 1: contrastive pre-training
python train_cta_classifier.py --mode contrastive

# Stage 2: classification fine-tune from Stage 1
python train_cta_classifier.py --mode fine_tune

# Stage 3: combined polish from Stage 2
python train_cta_classifier.py --mode combined
```

Expanded Stage 1 defaults:

```bash
python train_cta_classifier.py \
  --mode contrastive \
  --synthetic_path synthetic_df.csv \
  --output_dir ./model_contrastive \
  --epochs 8 \
  --batch_size 32 \
  --lr 2e-5 \
  --temperature 0.07 \
  --embed_dim 128 \
  --max_length 128 \
  --name_repeat 3 \
  --name_dropout_prob 0.15 \
  --mine_topk 50 \
  --mine_hard_k 20 \
  --mine_interval 2 \
  --test_size 0.2 \
  --seed 42
```

Expanded Stage 2 defaults:

```bash
python train_cta_classifier.py \
  --mode fine_tune \
  --synthetic_path synthetic_df.csv \
  --load_encoder_from ./model_contrastive/model.pt \
  --output_dir ./model_fine_tune \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-5 \
  --encoder_lr 5e-6 \
  --freeze_warmup_epochs 1 \
  --metric_alpha 0.05 \
  --supcon_temperature 0.07 \
  --label_smoothing 0.05 \
  --metric_embed_dim 256 \
  --max_length 128 \
  --name_repeat 3 \
  --name_dropout_prob 0.15 \
  --test_size 0.2 \
  --seed 42 \
  --use_spatial_head
```

Expanded Stage 3 defaults:

```bash
python train_cta_classifier.py \
  --mode combined \
  --synthetic_path synthetic_df.csv \
  --load_encoder_from ./model_fine_tune/model.pt \
  --output_dir ./model_combined \
  --epochs 3 \
  --batch_size 32 \
  --lr 1e-5 \
  --encoder_lr 2e-6 \
  --alpha 0.10 \
  --supcon_temperature 0.07 \
  --label_smoothing 0.05 \
  --metric_embed_dim 256 \
  --max_length 128 \
  --name_repeat 3 \
  --name_dropout_prob 0.15 \
  --test_size 0.2 \
  --seed 42 \
  --use_spatial_head
```

The spatial head is enabled by default. Pass `--no-use_spatial_head` to disable it.

## Input Format

Training text uses special tokens to emphasize column names and values:

```text
[COL] column_name [COL] column_name [COL] column_name [VAL] val1 [VAL] val2 [VAL] val3
```

Column names are repeated 3 times by default, with a 0.15 probability of dropping the name during training data construction.

## Outputs

Each training stage writes the following files to its `--output_dir`:

- `model.pt` - trained model weights.
- `label_encoder.json` - class labels and training metadata.
- `config.json` - encoder configuration.
- `tokenizer_config.json` and tokenizer files - tokenizer with CTA special tokens.

## Standalone CTA Inference

The main package uses `profiler.spatial.GeoClassifier` and `HybridGeoClassifier` during `process_dataset`. For local model debugging, this directory also includes standalone inference utilities.

Pure ML inference:

```bash
# Text input
python inference_cta.py --model_dir ./model_combined --text "lat: 40.71, 40.72, 40.73"

# Column + values input
python inference_cta.py --model_dir ./model_combined --column "BOROUGH" --values "Manhattan, Brooklyn, Queens"

# With confidence threshold
python inference_cta.py --model_dir ./model_combined --text "col1: 123, 456" --threshold 0.5

# Get embeddings when the loaded model supports them
python inference_cta.py --model_dir ./model_combined --text "lat: 40.71" --embedding
```

Hybrid ML + rules validation:

```python
from inference_cta import CTAClassifier
from rules_cta import HybridCTAClassifier

ml_classifier = CTAClassifier("./model_combined")
hybrid = HybridCTAClassifier(ml_classifier)

result = hybrid.classify("BBL", [1001234567, 2005678901, 3012345678])
```

Rule validation is used for high-risk spatial labels such as BBL, BIN, latitude, longitude, projected coordinates, ZIP/postal codes, and WKT geometry. If the top ML prediction fails validation, the hybrid classifier returns `non_spatial` for that prediction.

Standalone rules-only checks are also available:

```python
from rules_cta import RuleBasedCTA

classifier = RuleBasedCTA()
result = classifier.classify("BBL", [1001234567, 2005678901])
```

## Runtime Profiler Integration

After training, copy or publish the chosen checkpoint so the runtime package can find:

- `model.pt`
- `config.json`
- `label_encoder.json`
- tokenizer files

At runtime, `process_dataset(..., geo_classifier=True)` creates a hybrid classifier, predicts spatial labels for all non-manual columns in one batch, validates sensitive predictions with rules, and maps accepted labels into `profiler.core.GEO_CLASSIFIER_SPATIAL_MAP`.
