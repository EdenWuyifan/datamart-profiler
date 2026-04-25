#!/usr/bin/env python3
"""
Generate synthetic CTA training data with two modes:

1. single:
   Generate normal positive examples per label.

2. pairs:
   Generate paired hard negatives:
   - curated spatial-vs-spatial pairs
   - every spatial label vs non_spatial

Output CSV columns:
    name, values, label

Each generated row represents one synthetic column.
"""

import argparse
import itertools
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders
from tqdm import tqdm

load_dotenv()


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "@vertexai/gemini-2.5-pro"
DEFAULT_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"

CACHE_FILE = "synthetic_df_checkpoint.csv"
DEFAULT_CURATED_CTA_FILE = "curated_spatial_cta.csv"
DEFAULT_OUTPUT_FILE = "synthetic_df.csv"

SPATIAL_LABELS = [
    "latitude",
    "longitude",
    "x_coord",
    "y_coord",
    "point",
    "line",
    "multi-line",
    "polygon",
    "multi-polygon",
    "borough",
    "borough_code",
    "city",
    "state",
    "state_code",
    "country",
    "zip5",
    "zip9",
    "address",
    "bbl",
    "bin",
]

NON_SPATIAL_LABEL = "non_spatial"

# Focused spatial hard-negative pairs.
# These are more useful than all-vs-all for improving confusing classes.
DEFAULT_SPATIAL_PAIRS = [
    ("x_coord", "y_coord"),
    ("x_coord", "longitude"),
    ("y_coord", "latitude"),
    ("latitude", "longitude"),
    ("point", "latitude"),
    ("point", "longitude"),
    ("point", "polygon"),
    ("line", "multi-line"),
    ("polygon", "multi-polygon"),
    ("polygon", "line"),
    ("city", "state"),
    ("state", "state_code"),
    ("zip5", "zip9"),
    ("zip5", "borough_code"),
    ("zip5", "bin"),
    ("bbl", "bin"),
    ("bbl", "address"),
    ("borough", "borough_code"),
]


# ============================================================
# LLM
# ============================================================

def get_llm(model: str, temperature: float):
    portkey_headers = createHeaders(
        api_key=os.getenv("PORTKEY_API_KEY"),
        metadata={"_user": "yfw215"},
    )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=DEFAULT_BASE_URL,
        default_headers=portkey_headers,
        timeout=1000,
        max_retries=3,
    )


# ============================================================
# Compact rules
# ============================================================

def compact_label_rules() -> str:
    return """
LABEL RULES, US DATA ONLY:

latitude:
- Decimal degrees, positive, usually 24.5 to 49.5.
- Names should signal lat/latitude.
- Never projected-coordinate magnitudes.

longitude:
- Decimal degrees, negative, usually -125 to -66.
- Names should signal lon/lng/longitude.
- Never projected-coordinate magnitudes.

x_coord:
- Projected horizontal coordinate, not longitude.
- Names must signal x/east/easting/map_x/projected_x/gis_x.
- Values should be large projected coordinates:
  either negative Web Mercator X around -7000000 to -14000000,
  or positive State Plane/local easting around 200000 to 3500000.
- Never degree-scale values like -74 or 40.
- Never generic names like coord/coordinate/location_coord.

y_coord:
- Projected vertical coordinate, not latitude.
- Names must signal y/north/northing/map_y/projected_y/gis_y.
- Values should be large positive projected coordinates:
  usually 100000 to 6500000.
- Never degree-scale values like 40 or -74.
- Never generic names like coord/coordinate/location_coord.

point:
- WKT POINT(lon lat), lon first, no comma inside POINT.
- Example shape: POINT(-74.0060 40.7128).

line:
- WKT LINESTRING(lon lat,lon lat,...).

multi-line:
- WKT MULTILINESTRING((lon lat,lon lat),(lon lat,lon lat)).

polygon:
- WKT POLYGON((lon lat,lon lat,...)).
- Ring must be closed: first coordinate equals last coordinate.

multi-polygon:
- WKT MULTIPOLYGON with 2 or more closed polygons.

borough:
- Realistic US borough names, especially NYC or Alaska boroughs.

borough_code:
- Borough identifiers, usually NYC codes 1 to 5.
- Avoid generic numeric category codes.

city:
- US city/town names only. No state, no ZIP.

state:
- Full US state names only.

state_code:
- Two-letter USPS codes only.

country:
- United States, United States of America, USA, or U.S.A. only.

zip5:
- Exactly 5 digits.
- Same generated column should use ZIPs from one plausible state/region.

zip9:
- ZIP+4 format: 5 digits, dash, 4 digits.
- Same generated column should use one plausible state/region.

address:
- US street or PO Box address.
- Must start with street number or PO Box.
- May include city/state/ZIP.

bbl:
- NYC borough-block-lot.
- Format: borough-block-lot, exactly two dashes, numeric segments.

bin:
- NYC building identification number.
- Exactly 7 digits.

non_spatial:
- Must not encode location.
- Good values: person names, status labels, booleans, dates, non-location IDs, counts, categories.
- Avoid ZIP-like 5 digits, BIN-like 7 digits, BBL-like patterns, WKT, cities, states, countries, addresses, lat/lon, projected coordinates.
"""


def naming_style_rules() -> str:
    return """
For each generated column name:
- Use realistic open-data naming styles.
- Mix canonical, abbreviated, and messy names.
- Do not make names too clean or repetitive.
- For x_coord/y_coord, the name must explicitly reveal x/east or y/north semantics.
"""


# ============================================================
# Prompts
# ============================================================

def single_prompt(label: str, n_rows: int, n_values: int, seed_name: str = "", seed_values: str = "") -> str:
    example_hint = ""
    if seed_name:
        example_hint = f"""
Seed example, for style only:
- seed column name: {seed_name}
- seed values: {seed_values}
Do not copy the seed values.
"""

    return f"""
You generate synthetic tabular columns for CTA training.

Generate {n_rows} synthetic columns for label: {label}
Each column must have exactly {n_values} sample values.

{example_hint}

{naming_style_rules()}

{compact_label_rules()}

Return JSONL only.
Each line must be one JSON object:
{{"name": "...", "values": ["...", "...", "..."], "label": "{label}"}}

No markdown. No explanation. No extra text.
"""


def pair_prompt(label_a: str, label_b: str, n_pairs: int, n_values: int) -> str:
    return f"""
You generate HARD-NEGATIVE synthetic CTA training data.

Generate {n_pairs} paired examples for labels:
- {label_a}
- {label_b}

Each pair should look like two columns that could realistically appear in the same dataset.
The two columns should be similar enough to be confusing, but each must clearly obey its own label.

Each generated column must have exactly {n_values} sample values.

Important:
- If one label is non_spatial, make it a strong non-location distractor.
- non_spatial must not contain city/state/ZIP/address/WKT/lat/lon/projected coordinates.
- Do not use generic ambiguous coordinate names like coord, coordinate, location_coord.
- For x_coord, names must reveal x/east/easting semantics.
- For y_coord, names must reveal y/north/northing semantics.

{naming_style_rules()}

{compact_label_rules()}

Return JSONL only.
For each pair, output one row for {label_a} and one row for {label_b}.
Total lines: {n_pairs * 2}

Each line must be one JSON object:
{{"name": "...", "values": ["...", "...", "..."], "label": "..."}}

No markdown. No explanation. No extra text.
"""


# ============================================================
# Parsing and validation
# ============================================================

def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|jsonl)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def parse_jsonl_response(response) -> list[dict]:
    content = response.content if hasattr(response, "content") else str(response)
    content = strip_code_fences(content)

    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        name = str(obj.get("name", "")).strip()
        label = str(obj.get("label", "")).strip()
        values = obj.get("values", [])

        if not name or not label or not isinstance(values, list) or len(values) == 0:
            continue

        values = [str(v).strip() for v in values if str(v).strip()]
        if not values:
            continue

        rows.append(
            {
                "name": name,
                "values": ", ".join(values),
                "label": label,
            }
        )

    return rows


def basic_row_filter(row: dict, allowed_labels: set[str]) -> bool:
    name = str(row.get("name", "")).strip()
    values = str(row.get("values", "")).strip()
    label = str(row.get("label", "")).strip()

    if not name or not values or label not in allowed_labels:
        return False

    # Basic hard filters for common failure modes.
    vals = [v.strip() for v in values.split(",")]

    if label == "zip5":
        return all(re.fullmatch(r"\d{5}", v) for v in vals)

    if label == "zip9":
        return all(re.fullmatch(r"\d{5}-\d{4}", v) for v in vals)

    if label == "bin":
        return all(re.fullmatch(r"\d{7}", v) for v in vals)

    if label == "bbl":
        return all(re.fullmatch(r"\d{1,5}-\d{1,5}-\d{1,5}", v) for v in vals)

    if label == "state_code":
        return all(re.fullmatch(r"[A-Z]{2}", v) for v in vals)

    if label == "country":
        allowed = {"United States", "United States of America", "USA", "U.S.A."}
        return all(v in allowed for v in vals)

    if label == "latitude":
        try:
            nums = [float(v.replace("+", "")) for v in vals]
            return all(18.0 <= x <= 71.5 for x in nums)
        except ValueError:
            return False

    if label == "longitude":
        try:
            nums = [float(v) for v in vals]
            return all(-180.0 <= x <= -60.0 for x in nums)
        except ValueError:
            return False

    if label == "x_coord":
        lowered = name.lower()
        if not any(k in lowered for k in ["x", "east", "easting"]):
            return False
        if any(k in lowered for k in ["northing", "latitude", " lat", "y_coord"]):
            return False
        return True

    if label == "y_coord":
        lowered = name.lower()
        if not any(k in lowered for k in ["y", "north", "northing"]):
            return False
        if any(k in lowered for k in ["easting", "longitude", " lon", "lng", "x_coord"]):
            return False
        return True

    if label == "non_spatial":
        text = f"{name} {values}".lower()
        forbidden = [
            "point(",
            "linestring(",
            "polygon(",
            "multipolygon(",
            "multilinestring(",
            "latitude",
            "longitude",
            "address",
            "zipcode",
            "zip_code",
            "postal",
            "city",
            "state",
            "country",
            "borough",
        ]
        return not any(x in text for x in forbidden)

    return True


def append_and_save(rows: list[dict], output_rows: list[dict], cache_path: str):
    output_rows.extend(rows)
    pd.DataFrame(output_rows).drop_duplicates().to_csv(cache_path, index=False)


# ============================================================
# Pair construction
# ============================================================

def build_pairs(include_all_spatial_pairs: bool) -> list[tuple[str, str]]:
    pairs = set()

    if include_all_spatial_pairs:
        for a, b in itertools.combinations(SPATIAL_LABELS, 2):
            pairs.add((a, b))
    else:
        for pair in DEFAULT_SPATIAL_PAIRS:
            pairs.add(tuple(pair))

    # Add every spatial label vs non_spatial.
    for spatial_label in SPATIAL_LABELS:
        pairs.add((spatial_label, NON_SPATIAL_LABEL))

    return sorted(pairs)


# ============================================================
# Generation loops
# ============================================================

def generate_single_mode(
    llm,
    curated_df: pd.DataFrame,
    labels: list[str],
    target_per_label: int,
    n_rows_per_call: int,
    n_values: int,
    all_rows: list[dict],
    cache_path: str,
):
    existing = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=["label"])
    existing_counts = existing["label"].value_counts().to_dict() if len(existing) else {}

    allowed_labels = set(labels)

    for label in labels:
        current = int(existing_counts.get(label, 0))
        needed = max(0, target_per_label - current)

        if needed == 0:
            print(f"[single] {label}: already {current}/{target_per_label}")
            continue

        examples = curated_df[curated_df["Label"] == label].to_dict("records")
        if not examples:
            examples = [{"Column": "", "Values": ""}]

        print(f"[single] {label}: need {needed}")

        pbar = tqdm(total=needed)
        attempts = 0
        generated = 0

        while generated < needed:
            attempts += 1
            seed = random.choice(examples)
            prompt = single_prompt(
                label=label,
                n_rows=min(n_rows_per_call, needed - generated),
                n_values=n_values,
                seed_name=str(seed.get("Column", "")),
                seed_values=str(seed.get("Values", "")),
            )

            try:
                response = llm.invoke(prompt)
                rows = parse_jsonl_response(response)
                rows = [r for r in rows if basic_row_filter(r, allowed_labels)]
            except Exception as e:
                print(f"  error for {label}: {e}")
                continue

            if not rows:
                if attempts % 5 == 0:
                    print(f"  warning: {attempts} attempts, no valid rows recently")
                continue

            rows = rows[: needed - generated]
            append_and_save(rows, all_rows, cache_path)
            generated += len(rows)
            pbar.update(len(rows))

        pbar.close()


def generate_pair_mode(
    llm,
    pairs: list[tuple[str, str]],
    n_pairs_per_label_pair: int,
    n_values: int,
    all_rows: list[dict],
    cache_path: str,
):
    allowed_labels = set(SPATIAL_LABELS + [NON_SPATIAL_LABEL])

    print(f"[pairs] total label pairs: {len(pairs)}")
    for label_a, label_b in tqdm(pairs, desc="[pairs]"):
        prompt = pair_prompt(
            label_a=label_a,
            label_b=label_b,
            n_pairs=n_pairs_per_label_pair,
            n_values=n_values,
        )

        try:
            response = llm.invoke(prompt)
            rows = parse_jsonl_response(response)
            rows = [r for r in rows if basic_row_filter(r, allowed_labels)]
        except Exception as e:
            print(f"  error for pair ({label_a}, {label_b}): {e}")
            continue

        if rows:
            append_and_save(rows, all_rows, cache_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CTA data.")

    parser.add_argument("--curated-csv", default=DEFAULT_CURATED_CTA_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--cache", default=CACHE_FILE)

    parser.add_argument(
        "--mode",
        choices=["single", "pairs", "both"],
        default="both",
        help="single = per-label positives; pairs = hard negatives; both = both.",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=120,
        help="Target number of single-mode samples per label.",
    )

    parser.add_argument(
        "--target-labels",
        type=str,
        default=None,
        help="Comma-separated labels for single-mode generation. Defaults to curated CSV labels.",
    )

    parser.add_argument(
        "--n-values",
        type=int,
        default=3,
        help="Number of sample values per synthetic column.",
    )

    parser.add_argument(
        "--n-rows-per-call",
        type=int,
        default=5,
        help="Number of synthetic columns requested per single-mode LLM call.",
    )

    parser.add_argument(
        "--n-pairs-per-label-pair",
        type=int,
        default=5,
        help="Number of paired examples requested for each hard-negative label pair.",
    )

    parser.add_argument(
        "--all-spatial-pairs",
        action="store_true",
        help="Use all spatial-vs-spatial combinations instead of curated spatial hard-negative pairs.",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--temperature-single",
        type=float,
        default=0.8,
        help="Temperature for normal positive generation.",
    )
    parser.add_argument(
        "--temperature-pairs",
        type=float,
        default=0.4,
        help="Temperature for hard-negative pair generation.",
    )

    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Ignore and overwrite the existing cache.",
    )

    args = parser.parse_args()

    curated_path = Path(args.curated_csv)
    if not curated_path.exists():
        raise FileNotFoundError(f"Curated CSV not found: {args.curated_csv}")

    curated_df = pd.read_csv(curated_path)

    required = {"Column", "Values", "Label"}
    missing = required.difference(curated_df.columns)
    if missing:
        raise ValueError(f"Curated CSV missing columns: {missing}")

    if args.reset_cache or not Path(args.cache).exists():
        all_rows = []
    else:
        cache_df = pd.read_csv(args.cache)
        all_rows = cache_df.to_dict("records")
        print(f"Loaded cache: {len(all_rows)} rows from {args.cache}")

    if args.target_labels:
        labels = [x.strip() for x in args.target_labels.split(",") if x.strip()]
    else:
        labels = sorted(curated_df["Label"].dropna().unique().tolist())

    if args.mode in {"single", "both"}:
        llm_single = get_llm(args.model, args.temperature_single)
        generate_single_mode(
            llm=llm_single,
            curated_df=curated_df,
            labels=labels,
            target_per_label=args.target,
            n_rows_per_call=args.n_rows_per_call,
            n_values=args.n_values,
            all_rows=all_rows,
            cache_path=args.cache,
        )

    if args.mode in {"pairs", "both"}:
        llm_pairs = get_llm(args.model, args.temperature_pairs)
        pairs = build_pairs(include_all_spatial_pairs=args.all_spatial_pairs)
        generate_pair_mode(
            llm=llm_pairs,
            pairs=pairs,
            n_pairs_per_label_pair=args.n_pairs_per_label_pair,
            n_values=args.n_values,
            all_rows=all_rows,
            cache_path=args.cache,
        )

    result_df = pd.DataFrame(all_rows).drop_duplicates()
    result_df.to_csv(args.output, index=False)

    print(f"\nSaved {len(result_df)} rows to {args.output}")
    print("\nFinal label distribution:")
    print(result_df["label"].value_counts())


if __name__ == "__main__":
    main()