# atlas-profiler

Atlas Profiler is a dataset profiling library. Given a CSV/TSV, file-like object, or pandas DataFrame, it returns JSON-style metadata about the dataset, its columns, detected types, value ranges, optional plots, spatial/temporal coverage, and profiling runtime.

The package builds on the Datamart Profiler workflow and adds an ML-assisted spatial column classifier. That classifier is only one part of the profiler: non-spatial columns still go through the core rule-based type detection, statistics, plots, coverage, and dataset-summary pipeline.

## What It Produces

`process_dataset(...)` returns a metadata dictionary with fields such as:

- Dataset size, row count, profiled row count, and column count.
- Per-column structural type, semantic types, missing/unclean value ratios, distinct counts, and optional plots.
- Dataset-level type summary: numerical, categorical, spatial, and temporal.
- Spatial coverage from lat/long pairs, WKT points, resolved addresses, and administrative areas.
- Temporal coverage and temporal resolution for datetime columns.
- Attribute keywords derived from column names.
- Optional random sample rows and per-step profiling timings.

## Core Type System

The profiler detects broad structural types for all columns:

| Structural type | Meaning |
| --- | --- |
| `MissingData` | Empty column. |
| `Integer` | Integer-like values. |
| `Float` | Floating point values. |
| `Text` | String/text values. |
| `Boolean` | Boolean-like values such as true/false, yes/no, 0/1. |
| `GeoCoordinates` | Point geometry or coordinate-pair strings. |
| `GeoShape` | Polygon-like geometry. |

It also annotates semantic types when evidence is available:

| Semantic type | Examples |
| --- | --- |
| `DateTime` | Dates, timestamps, and year columns. |
| `latitude`, `longitude` | Coordinate columns, paired after profiling. |
| `address`, `AdministrativeArea` | Address-like and admin-area text, optionally resolved with Nominatim or `datamart_geo`. |
| `URL`, `FileName`, `identifier`, `Enumeration` | URLs, file paths, IDs, and categorical columns. |

## Spatial ML Classifier

When `geo_classifier=True`, Atlas Profiler creates a `HybridGeoClassifier(GeoClassifier())`. It samples values from each column, predicts spatial labels in one batch, validates sensitive predictions with rules, and passes accepted labels into the normal profiler type system.

The classifier labels are not the full profiler type system. They are a spatial CTA layer mapped into profiler structural and semantic types:

| Classifier label family | Mapped profiler behavior |
| --- | --- |
| `latitude`, `longitude` | Float columns with latitude/longitude semantic types, then paired for coverage. |
| `x_coord`, `y_coord` | Projected coordinate-like float columns. |
| `point`, `line`, `polygon`, `multi-line`, `multi-polygon` | Geometry columns mapped to point or shape structural types. |
| `zip5`, `zip9`, `address` | Text columns with address semantics. |
| `borough`, `borough_code`, `city`, `state`, `state_code`, `country` | Text columns with administrative-area semantics. |
| `bbl`, `bin` | NYC spatial identifiers mapped as integer identifiers. |
| `non_spatial` | Falls back to the core profiler's normal type detection. |

Manual column annotations take precedence over ML predictions. Low-confidence or rule-rejected ML predictions also fall back to the regular profiler workflow.

## Pipeline

`process_dataset` runs the same high-level workflow for every dataset:

1. Load data from a path, file object, or DataFrame.
2. Compute cheap full-data stats and sample values for each column.
3. Optionally run a single batch spatial ML prediction for all non-manual columns.
4. Process every column with either an accepted geo prediction or the regular profiler type detector.
5. Pair latitude/longitude columns and compute dataset-level type counts.
6. Optionally compute numerical ranges, histograms, spatial coverage, temporal coverage, keywords, samples, and timing metadata.

The regular type detector recognizes integers, floats, text, booleans, URLs, file paths, WKT points/polygons, categorical values, IDs, datetimes, latitude/longitude name patterns, and optional administrative areas.

## Installation

```bash
pip install atlas-profiler
```

For source development:

```bash
git clone https://github.com/VIDA-NYU/atlas-profiler.git
cd atlas-profiler
pip install -e .
```

## Basic Usage

```python
from atlas_profiler import process_dataset

metadata = process_dataset("data.csv")
```

`process_dataset` also accepts a pandas DataFrame:

```python
metadata = process_dataset(
    df,
    geo_classifier=True,
    geo_classifier_threshold=0.5,
    coverage=True,
    plots=False,
    include_sample=False,
)
```

Key parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `data` | required | Path, file-like object, or pandas DataFrame. |
| `geo_classifier` | `True` | Enable the default hybrid spatial classifier, disable with `False`, or pass a classifier instance. |
| `geo_classifier_threshold` | `0.5` | Confidence cutoff for spatial ML predictions. |
| `coverage` | `True` | Compute numerical ranges plus spatial/temporal coverage. |
| `plots` | `False` | Add compact histogram-style plot data to column metadata. |
| `include_sample` | `False` | Include a small deterministic CSV sample in the output. |
| `indexes` | `True` | Preserve non-default DataFrame indexes as columns. |
| `load_max_size` | `5000000` | Target bytes to profile; larger inputs are sampled. |
| `metadata` | `None` | Optional seed metadata, including manual annotations. |
| `nominatim` | `None` | Optional Nominatim endpoint for resolving address strings. |
| `datamart_geo_data` | `None` | `True` or a `datamart_geo.GeoData` instance for administrative-area resolution. |

## Manual Annotations

Manual annotations can be supplied through the `metadata` argument. They are useful when a user or upstream discovery step already knows a column's type. Manually annotated columns skip the spatial ML classifier and are reconciled with observed values during normal column processing.

## Model Files

`GeoClassifier()` first looks for bundled model files under `profiler/model/`. If they are not present, it uses a user cache directory and downloads missing files when `auto_download=True`.

Required model files:

- `model.pt`
- `config.json`
- `label_encoder.json`

CTA model training, synthetic data generation, and standalone CTA inference are documented in [`training/README.md`](training/README.md).

## Project Structure

```text
atlas-profiler/
├── atlas_profiler/          # Public import shim: from atlas_profiler import process_dataset
├── profiler/                # Runtime profiling package
│   ├── core.py              # process_dataset, loading, column pipeline, coverage
│   ├── profile_types.py     # Rule-based structural/semantic type detection
│   ├── spatial.py           # Spatial coverage, geohashing, GeoClassifier integration
│   ├── temporal.py          # Date parsing and temporal resolution
│   ├── numerical.py         # Numeric summaries and ranges
│   └── types.py             # Type constants
├── training/                # CTA data generation, model training, standalone inference
├── tests/                   # Unit tests
├── examples/                # Example notebooks
├── README.md
└── pyproject.toml
```

## Relationship To Datamart Profiler

This project reuses the structure and main profiling logic of Datamart Profiler, with additional spatial CTA model integration.

Credits:

- Datamart Profiler codebase: https://gitlab.com/ViDA-NYU/auctus/auctus
- Datamart Profiler on PyPI: https://pypi.org/project/datamart-profiler/
