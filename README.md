# atlas-profiler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/atlas-profiler.svg)](https://pypi.org/project/atlas-profiler/)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://atlas-profiler.readthedocs.io/en/latest/)
[![GitHub](https://img.shields.io/badge/github-VIDA--NYU%2Fatlas--profiler-brightgreen.svg)](https://github.com/VIDA-NYU/atlas-profiler)

**Atlas Profiler** is a comprehensive dataset profiling library that automatically detects and annotates data types, including spatial and temporal features. Given a CSV/TSV, file-like object, or pandas DataFrame, it returns rich JSON-style metadata about your dataset, its columns, detected types, value ranges, optional plots, spatial/temporal coverage, and profiling runtime.

## Quick Start

### Installation

Install from PyPI:

```bash
pip install atlas-profiler
```

Or install from source for development:

```bash
git clone https://github.com/VIDA-NYU/atlas-profiler.git
cd atlas-profiler
pip install -e .
```

### Basic Usage

```python
from atlas_profiler import process_dataset

# Profile a CSV file
metadata = process_dataset("data.csv")

# Or profile a pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
metadata = process_dataset(
    df,
    geo_classifier=True,
    geo_classifier_threshold=0.5,
    coverage=True,
)
```

## Documentation

For comprehensive guides, API reference, examples, and advanced configuration, visit the **[Complete Documentation](https://atlas-profiler.readthedocs.io/en/latest/)**.

## Table of Contents

- [Features](#features)
- [What It Produces](#what-it-produces)
- [Type System](#type-system)
- [Architecture](#architecture)
  - [Pipeline](#pipeline)
  - [Spatial ML Classifier](#spatial-ml-classifier)
- [Advanced Usage](#advanced-usage)
  - [Configuration Parameters](#configuration-parameters)
  - [Manual Annotations](#manual-annotations)
  - [Model Files](#model-files)
- [Project Structure](#project-structure)
- [Related Projects](#related-projects)

## Features

✨ **Automatic Type Detection**: Identifies structural types (Integer, Float, Text, Boolean, GeoCoordinates, GeoShape) and semantic types (DateTime, Address, URL, ID, etc.)

🌍 **Spatial Intelligence**: ML-powered spatial column classifier trained on synthetic data, recognizing coordinates, addresses, geospatial identifiers, and administrative areas

⏰ **Temporal Analysis**: Detects and analyzes temporal columns with coverage and resolution information

📊 **Rich Metadata**: Comprehensive dataset profiling including:
- Column-level statistics and distinct value counts
- Dataset-level type summaries
- Spatial and temporal coverage information
- Optional histograms and sample data
- Profiling performance metrics

## What It Produces

`process_dataset(...)` returns a metadata dictionary with:

- **Dataset metrics**: row count, column count, profiled row count
- **Per-column analysis**: structural type, semantic types, missing value ratios, distinct counts, sample values
- **Dataset summary**: numerical, categorical, spatial, and temporal type counts
- **Coverage information**: spatial bounding boxes, temporal ranges, geohash coverage
- **Attribute keywords**: automatically extracted from column names
- **Performance metrics**: per-step profiling timings

## Type System

### Structural Types

The profiler recognizes these broad structural types:

| Type | Meaning |
| --- | --- |
| `Integer` | Integer-like values |
| `Float` | Floating point values |
| `Text` | String/text values |
| `Boolean` | Boolean-like values (true/false, yes/no, 0/1) |
| `GeoCoordinates` | Point geometry or coordinate-pair strings |
| `GeoShape` | Polygon-like geometry |
| `MissingData` | Empty column |

### Semantic Types

The profiler also annotates semantic meaning when evidence is available:

| Type | Examples |
| --- | --- |
| `DateTime` | Dates, timestamps, year columns |
| `latitude`, `longitude` | Coordinate columns (paired after profiling) |
| `address`, `AdministrativeArea` | Address text or admin areas (optionally resolved via Nominatim or `datamart_geo`) |
| `URL`, `FileName`, `identifier`, `Enumeration` | URLs, file paths, IDs, categorical values |

## Architecture

### Pipeline

`process_dataset` executes a consistent workflow for every dataset:

1. **Load data** from path, file object, or DataFrame
2. **Compute statistics** on full data and collect sample values per column
3. **Predict spatial labels** (optional) using batch ML inference
4. **Process columns** with geo predictions or rule-based type detection
5. **Pair lat/long columns** and compute dataset-level type summaries
6. **Compute coverage** (optional) for numerical, spatial, and temporal ranges

### Spatial ML Classifier

When `geo_classifier=True`, Atlas Profiler uses a `HybridGeoClassifier` that:

- Samples values from each column
- Predicts spatial labels in a single batch
- Validates predictions using rule-based checks
- Maps predictions to the profiler's type system

**Supported spatial labels:**

| Label Family | Mapped Type |
| --- | --- |
| `latitude`, `longitude` | Float + semantic types |
| `x_coord`, `y_coord` | Projected coordinates |
| `point`, `polygon`, `line` | Geometry types |
| `address`, `zip5`, `zip9` | Address/postal codes |
| `borough`, `city`, `state`, `country` | Administrative areas |
| `bbl`, `bin` | NYC spatial identifiers |
| `non_spatial` | Falls back to standard detection |

Manual annotations take precedence over ML predictions. Low-confidence or rejected predictions fall back to rule-based detection.

## Advanced Usage

### Configuration Parameters

Key parameters for `process_dataset()`:

| Parameter | Default | Description |
| --- | --- | --- |
| `data` | required | Path, file-like object, or pandas DataFrame |
| `geo_classifier` | `True` | Enable spatial ML classifier |
| `geo_classifier_threshold` | `0.5` | Confidence cutoff for predictions |
| `coverage` | `True` | Compute numerical ranges and spatial/temporal coverage |
| `plots` | `False` | Include histogram-style plot data |
| `include_sample` | `False` | Include sample rows in output |
| `indexes` | `True` | Preserve DataFrame indexes as columns |
| `load_max_size` | `5000000` | Target bytes to profile (larger inputs are sampled) |
| `metadata` | `None` | Optional seed metadata with manual annotations |
| `nominatim` | `None` | Nominatim endpoint for address resolution |
| `datamart_geo_data` | `None` | GeoData instance for admin-area resolution |

### Manual Annotations

Supply manual type annotations through the `metadata` argument. Useful when upstream processes or domain knowledge already identifies column types:

```python
metadata = {
    "columns": [
        {
            "name": "latitude",
            "semantic_types": ["http://schema.org/latitude"]
        },
        {
            "name": "longitude", 
            "semantic_types": ["http://schema.org/longitude"]
        }
    ]
}

result = process_dataset(df, metadata=metadata)
```

Manually annotated columns skip the spatial ML classifier and are reconciled with observed values during processing.

### Model Files

The spatial ML classifier uses these model files (automatically downloaded if missing):

- `model.pt` — PyTorch model weights
- `config.json` — Model configuration
- `label_encoder.json` — Label encoding

Files are cached locally and `auto_download=True` enables automatic retrieval.

For model training details, see [`training/README.md`](training/README.md).

## Project Structure

```
atlas-profiler/
├── atlas_profiler/          # Public API: from atlas_profiler import process_dataset
├── profiler/                # Core profiling package
│   ├── core.py              # process_dataset(), data loading, column pipeline
│   ├── profile_types.py     # Rule-based type detection
│   ├── spatial.py           # Spatial coverage & GeoClassifier
│   ├── temporal.py          # Temporal analysis
│   ├── numerical.py         # Numerical profiling
│   └── types.py             # Type constants
├── training/                # Model training & synthetic data generation
├── tests/                   # Unit tests
├── examples/                # Example notebooks
├── docs/                    # Sphinx documentation
└── pyproject.toml           # Project configuration
```

## Related Projects

This project builds upon and extends [Datamart Profiler](https://gitlab.com/ViDA-NYU/auctus/auctus) with additional spatial intelligence via ML-assisted column type classification.

- **Datamart Profiler**: https://pypi.org/project/datamart-profiler/
- **Research Background**: Developed by the [NYU Visualization and Data Analytics Lab](https://vida-nyu.github.io/)

## License

Atlas Profiler is released under the [MIT License](LICENSE).
