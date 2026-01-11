# atlas-profiler - CTA (Column Type Annotation)

A machine learning pipeline for spatial column type classification with rule-based validation.

## Overview

This system classifies tabular columns into spatial types (latitude, longitude, BBL, BIN, zip codes, geometries, etc.) using a hybrid ML + rules approach.

**Supported Column Types:**

- `latitude`, `longitude` - Geographic coordinates
- `x_coord`, `y_coord` - Projected coordinates
- `bbl` - Borough-Block-Lot (NYC property identifier)
- `bin` - Building Identification Number
- `zip_code` - Postal codes (worldwide)
- `borough_code` - District/borough codes
- `city`, `state`, `address` - Location strings
- `point`, `line`, `polygon`, `multi-polygon`, `multi-line` - WKT geometries
- `non_spatial` - Non-spatial identifiers

---

## Pipeline Workflow

Training scripts and datasets live under `training/`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. DATA GENERATION                                                      │
│     training/generate_synthetic_cta.py                                   │
│     curated_spatial_cta.csv  ──►  LLM augmentation  ──►  synthetic_df.csv│
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. MODEL TRAINING                                                       │
│     training/train_cta_classifier.py                                     │
│     curated + synthetic data  ──►  BGE encoder + classifier  ──►  model/ │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. INFERENCE + VALIDATION                                               │
│     training/inference_cta.py + training/rules_cta.py                    │
│     ML prediction  ──►  rule-based validation  ──►  final classification │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Generate Synthetic Training Data

**Script:** `training/generate_synthetic_cta.py`

Uses an LLM to augment curated examples with diverse variations:

- **Naming styles:** snake_case, camelCase, abbreviations, short/ambiguous names
- **Value diversity:** Worldwide locations (not limited to NYC)
- **Short name samples:** Forces value-based learning (e.g., `x`, `lt`, `coord`)

```bash
# Generate synthetic data (default: 120 samples per class)
python training/generate_synthetic_cta.py \
    --target 120 \
    --curated-csv training/curated_spatial_cta.csv \
    --output training/synthetic_df.csv

# Custom settings
python training/generate_synthetic_cta.py \
    --target 150 \
    --max-stale 15 \
    --curated-csv training/curated_spatial_cta.csv \
    --output training/synthetic_df.csv
```

**Inputs:**

- `training/curated_spatial_cta.csv` - Hand-labeled training examples

**Outputs:**

- `training/synthetic_df.csv` - Augmented training data
- `training/synthetic_df_checkpoint.csv` - Incremental checkpoint (for resuming)

---

## Step 2: Train the CTA Classifier

**Script:** `training/train_cta_classifier.py`

Trains a transformer-based classifier using [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5) as the encoder.

### Training Modes

| Mode             | Description                              | Best For          |
| ---------------- | ---------------------------------------- | ----------------- |
| `classification` | Standard cross-entropy loss              | Fast baseline     |
| `contrastive`    | Supervised contrastive learning (SupCon) | Better embeddings |
| `combined`       | Contrastive + classification loss        | **Recommended**   |

### Input Format

Uses structured tokens for emphasis:

```
[COL] column_name [COL] column_name [COL] column_name [VAL] val1 [VAL] val2 [VAL] val3
```

Column name is repeated (default: 3×) to emphasize its importance.

### Usage

```bash
# Standard classification (fast)
python training/train_cta_classifier.py --mode classification --epochs 10

# Supervised contrastive learning (better embeddings)
python training/train_cta_classifier.py --mode contrastive --epochs 20 --temperature 0.07

# Combined training (recommended)
python training/train_cta_classifier.py --mode combined --epochs 15 --alpha 0.5

# Full configuration
python training/train_cta_classifier.py \
    --mode combined \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5 \
    --temperature 0.07 \
    --alpha 0.5 \
    --name_repeat 3 \
    --output_dir profiler/model \
    --curated_path training/curated_spatial_cta.csv \
    --synthetic_path training/synthetic_df.csv
```

### Key Arguments

| Argument        | Default          | Description                             |
| --------------- | ---------------- | --------------------------------------- |
| `--mode`        | `classification` | Training mode                           |
| `--epochs`      | `10`             | Number of training epochs               |
| `--batch_size`  | `16`             | Batch size                              |
| `--lr`          | `2e-5`           | Learning rate                           |
| `--temperature` | `0.07`           | Contrastive loss temperature            |
| `--alpha`       | `0.5`            | Contrastive loss weight (combined mode) |
| `--name_repeat` | `3`              | Column name repetition count            |
| `--output_dir`  | `./model`        | Model output directory                  |

**Outputs (in `--output_dir`, default `./model/`):**

- `model.pt` - Trained model weights
- `label_encoder.json` - Class labels and config
- `config.json` - Encoder configuration
- `tokenizer_config.json` - Tokenizer with special tokens

---

## Step 3: Inference with Rule-Based Validation

### Pure ML Inference

**Script:** `training/inference_cta.py`

```bash
# Text input
python training/inference_cta.py --model_dir profiler/model --text "lat: 40.71, 40.72, 40.73"

# Column + values input
python training/inference_cta.py --model_dir profiler/model --column "BOROUGH" --values "Manhattan, Brooklyn, Queens"

# With confidence threshold (returns non_spatial if below)
python training/inference_cta.py --model_dir profiler/model --text "col1: 123, 456" --threshold 0.5

# Get embeddings (contrastive/combined modes only)
python training/inference_cta.py --model_dir profiler/model --text "lat: 40.71" --embedding
```

### Hybrid Classification (ML + Rules)

**Script:** `training/rules_cta.py`

The `HybridCTAClassifier` combines ML predictions with rule-based validation:

Note: imports below assume `training/` is on your `PYTHONPATH` or you run from that directory.

```python
from rules_cta import HybridCTAClassifier
from inference_cta import CTAClassifier

# Initialize
ml_classifier = CTAClassifier("profiler/model")
hybrid = HybridCTAClassifier(ml_classifier)

# Classify
result = hybrid.classify("BBL", [1001234567, 2005678901, 3012345678])
# Returns: [{"label": "bbl", "confidence": 0.95, "source": "ml+validated"}]
```

### Validation Logic

For sensitive spatial types, ML predictions are validated against rules:

| Type                     | Validation Rule                                 |
| ------------------------ | ----------------------------------------------- |
| `bbl`                    | 10-digit number starting with 1-5 (NYC borough) |
| `bin`                    | 7-digit number starting with 1-5                |
| `latitude`               | Values in range [-90, 90]                       |
| `longitude`              | Values in range [-180, 180], some > 90          |
| `x_coord`, `y_coord`     | Projected coordinates > 10,000                  |
| `zip_code`               | Matches postal code patterns (US, UK, CA, etc.) |
| `point`, `polygon`, etc. | Valid WKT geometry format                       |

**Validation outcomes:**

- ✅ **Passed:** Return ML prediction with `source: "ml+validated"`
- ❌ **Failed:** Return `non_spatial` with `source: "ml:{type}→rule_rejected"`

### Standalone Rule-Based Classification

```python
from rules_cta import RuleBasedCTA

classifier = RuleBasedCTA()

# Single column
result = classifier.classify("BBL", [1001234567, 2005678901])
# {"label": "bbl", "confidence": 0.95, "rule": "bbl_name_and_pattern"}

# Entire DataFrame
results = classifier.classify_dataframe(df, sample_size=100)
```

---

## Project Structure

```
atlas-profiler/
├── profiler/                   # Library package
│   ├── core.py
│   ├── spatial.py
│   └── model/                  # Bundled model artifacts
│       ├── model.pt
│       ├── label_encoder.json
│       ├── config.json
│       └── tokenizer_config.json
├── training/                   # Training + inference scripts/data
│   ├── generate_synthetic_cta.py
│   ├── train_cta_classifier.py
│   ├── inference_cta.py
│   ├── rules_cta.py
│   ├── curated_spatial_cta.csv
│   └── synthetic_df.csv
├── output/
├── results/
├── README.md
└── pyproject.toml
```

---

## Quick Start

```bash
# 1. Generate synthetic training data
python training/generate_synthetic_cta.py \
    --target 120 \
    --curated-csv training/curated_spatial_cta.csv \
    --output training/synthetic_df.csv

# 2. Train the model (combined mode recommended)
python training/train_cta_classifier.py \
    --mode combined \
    --epochs 15 \
    --output_dir profiler/model \
    --curated_path training/curated_spatial_cta.csv \
    --synthetic_path training/synthetic_df.csv

# 3. Run inference
python training/inference_cta.py --model_dir profiler/model --column "latitude" --values "40.71, 40.72"
```

---

## Auctus Datamart Profiler Integration

This section describes how the CTA classifier integrates with the [Auctus datamart profiler](https://gitlab.com/ViDA-NYU/auctus/datamart-profiler).

### Original Profiler Workflow (Before Geo Classifier)

The original `process_dataset()` function in `core.py` followed this sequential workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  process_dataset(data)                                                       │
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ 1. LOAD DATA     │  load_data() → pandas DataFrame                        │
│  │    - Read CSV    │  - Handle file size limits (MAX_SIZE = 5MB)            │
│  │    - Sample rows │  - Random sampling if file > max size                  │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 2. PROCESS COLS  │  For each column (sequential):                         │
│  │    (sequential)  │                                                        │
│  │                  │  process_column(array, column_meta)                    │
│  │                  │    │                                                   │
│  │                  │    ├─► identify_types()  ← regex + heuristics          │
│  │                  │    │     - Structural: INTEGER, FLOAT, TEXT, GEO_*     │
│  │                  │    │     - Semantic: LATITUDE, LONGITUDE, DATE_TIME,   │
│  │                  │    │                 ADDRESS, ADMIN, CATEGORICAL       │
│  │                  │    │                                                   │
│  │                  │    ├─► Compute numerical ranges & histograms           │
│  │                  │    ├─► Resolve addresses via Nominatim (HTTP calls)    │
│  │                  │    └─► Resolve admin areas via datamart_geo            │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 3. POST-PROCESS  │                                                        │
│  │                  │  - Index textual data with Lazo                        │
│  │                  │  - Pair lat/long columns (name matching)               │
│  │                  │  - Determine dataset types (spatial/temporal/etc.)     │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 4. COVERAGE      │  Compute spatial/temporal coverage:                    │
│  │                  │  - Lat/long pairs → geohashes + bounding boxes         │
│  │                  │  - WKT points → spatial ranges                         │
│  │                  │  - Addresses → resolved coordinates                    │
│  │                  │  - Admin areas → merged bounding boxes                 │
│  │                  │  - Datetime columns → temporal ranges                  │
│  └──────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Original Type Detection (`identify_types`)

The original `identify_types()` function used **regex patterns and heuristics**:

| Detection Method     | Types Detected                                               | Limitations                                     |
| -------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| Column name patterns | `latitude`, `longitude` (via `LATITUDE`, `LONGITUDE` tuples) | Only exact matches like `lat`, `long`, `xcoord` |
| Value regex          | `DATE_TIME`, `GEO_POINT` (WKT)                               | Limited patterns                                |
| Statistical analysis | `INTEGER`, `FLOAT`, `CATEGORICAL`                            | No semantic understanding                       |
| Nominatim lookup     | `ADDRESS`                                                    | Slow (HTTP calls per address)                   |
| datamart_geo lookup  | `ADMIN` areas                                                | Requires local geo database                     |

**Key limitations:**

- ❌ Failed to detect borough codes, BBL, BIN
- ❌ No detection of projected coordinates (x_coord, y_coord)
- ❌ Missed WKT polygons and multi-polygons
- ❌ Sequential column processing (slow for large datasets)

### Enhanced Workflow (With Geo Classifier)

The geo classifier adds a **batch ML prediction phase** before column processing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  process_dataset(data, geo_classifier=HybridGeoClassifier)                   │
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ 1. LOAD DATA     │  (unchanged)                                           │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 2. BATCH ML      │  ★ NEW: Single forward pass for ALL columns            │
│  │    PREDICTION    │                                                        │
│  │                  │  geo_classifier.predict_batch([                        │
│  │                  │      (col_name, sample_values),                        │
│  │                  │      ...                                               │
│  │                  │  ])                                                    │
│  │                  │                                                        │
│  │                  │  → Returns: {col_idx: {"label", "confidence"}}         │
│  │                  │                                                        │
│  │                  │  Detected types:                                       │
│  │                  │    latitude, longitude, x_coord, y_coord,              │
│  │                  │    bbl, bin, zip_code, borough_code,                   │
│  │                  │    city, state, address, point, polygon, ...           │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 3. PROCESS COLS  │  ★ NOW PARALLEL (ThreadPoolExecutor)                   │
│  │    (parallel)    │                                                        │
│  │                  │  process_column(..., geo_prediction=pred)              │
│  │                  │    │                                                   │
│  │                  │    ├─► If geo_prediction exists & spatial type:        │
│  │                  │    │     Use ML result directly (skip identify_types)  │
│  │                  │    │                                                   │
│  │                  │    └─► Else: Fall back to identify_types()             │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 4-5. POST-PROC   │  (unchanged)                                           │
│  │    + COVERAGE    │                                                        │
│  └──────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Type Mapping (`GEO_CLASSIFIER_SPATIAL_MAP`)

The geo classifier maps ML labels to Auctus type system:

```python
GEO_CLASSIFIER_SPATIAL_MAP = {
    # Coordinates
    "latitude":      (types.FLOAT, [types.LATITUDE]),
    "longitude":     (types.FLOAT, [types.LONGITUDE]),
    "x_coord":       (types.FLOAT, []),
    "y_coord":       (types.FLOAT, []),

    # Geometries
    "point":         (types.GEO_POINT, []),
    "polygon":       (types.GEO_POLYGON, []),
    "multi-polygon": (types.GEO_POLYGON, []),
    "line":          (types.GEO_POLYGON, []),

    # Addresses
    "zip_code":      (types.TEXT, [types.ADDRESS]),
    "address":       (types.TEXT, [types.ADDRESS]),

    # Administrative
    "borough_code":  (types.TEXT, [types.ADMIN]),
    "city":          (types.TEXT, [types.ADMIN]),
    "state":         (types.TEXT, [types.ADMIN]),

    # NYC-specific
    "bbl":           (types.INTEGER, [types.ID]),
    "bin":           (types.INTEGER, [types.ID]),
}
```

### Performance Improvements

| Aspect            | Original                    | With Geo Classifier                          |
| ----------------- | --------------------------- | -------------------------------------------- |
| Type detection    | Sequential regex per column | **Single batch forward pass**                |
| Column processing | Sequential                  | **Parallel (ThreadPoolExecutor)**            |
| Spatial types     | Limited (lat/lon only)      | **15+ types** including BBL, BIN, geometries |
| Accuracy          | Heuristic-based             | **ML + rule validation**                     |

### Usage in Auctus

```python
from profiler import process_dataset
from profiler.spatial import GeoClassifier, HybridGeoClassifier

# Initialize classifier (auto-downloads model from NYU Box)
geo_clf = HybridGeoClassifier(GeoClassifier())

# Profile dataset with geo classifier
metadata = process_dataset(
    "data.csv",
    geo_classifier=geo_clf,  # Enable ML-based type detection
    coverage=True,
    plots=True,
)

# Results include geo_classifier metadata
for col in metadata["columns"]:
    if "geo_classifier" in col:
        print(f"{col['name']}: {col['geo_classifier']}")
        # {'label': 'latitude', 'confidence': 0.97, 'source': 'ml+validated'}
```

### Model Auto-Download

The `GeoClassifier` automatically downloads model files from NYU Box on first use:

```python
GEO_MODEL_FILES = {
    "model.pt":           "https://nyu.box.com/shared/static/...",
    "config.json":        "https://nyu.box.com/shared/static/...",
    "label_encoder.json": "https://nyu.box.com/shared/static/...",
}
```

---

## Environment Variables

For synthetic data generation with LLM:

```bash
export PORTKEY_API_KEY="..."
export PROVIDER_API_KEY="..."
```
