#!/usr/bin/env python3
"""Generate synthetic CTA (Column Type Annotation) training data using LLM."""

import os
import random as _rand
import time
import argparse

import pandas as pd
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# === Configuration ===

SYNTHETIC_CACHE_FILE = "synthetic_df_checkpoint.csv"
CURATED_CTA_FILE = "curated_spatial_cta.csv"

NAMING_STYLES = [
    "snake_case with underscores",
    "camelCase",
    "abbreviations and acronyms",
    "descriptive full names",
    "short abbreviated forms",
]

# Numeric ranges for programmatic value generation (worldwide)
VALUE_RANGES = {
    "latitude": (-60.0, 70.0),
    "longitude": (-180.0, 180.0),
    "x_coord": (100000, 900000),
    "y_coord": (100000, 900000),
}

# Label-specific constraints for valid value generation (worldwide diversity)
LABEL_CONSTRAINTS = {
    "borough_code": (
        "Values MUST be district/borough/ward codes from cities WORLDWIDE "
        "(e.g., London boroughs, Paris arrondissements, Tokyo wards, Sydney councils). "
        "MIX different cities and formats."
    ),
    "bbl": (
        "Values MUST be property/parcel identifiers from various countries. "
        "Use formats like US APN, UK UPRN, Canadian PID, Australian lot numbers. "
        "VARY formats significantly."
    ),
    "bin": (
        "Values MUST be building identification numbers from various cities worldwide. "
        "VARY formats (numeric, alphanumeric) across different countries."
    ),
    "zip_code": (
        "Values MUST be postal codes from WORLDWIDE locations. Include US ZIP, "
        "UK postcodes (SW1A 1AA), Canadian (K1A 0B1), German (10115), "
        "Japanese (100-0001), Australian (2000). MIX countries."
    ),
    "latitude": (
        "Values MUST be valid latitudes from WORLDWIDE locations. Include places from "
        "ALL continents: Europe, Asia, Americas, Africa, Oceania. "
        "Range from -60 to 70. Use 2-6 decimal places."
    ),
    "longitude": (
        "Values MUST be valid longitudes from WORLDWIDE locations. Include places from "
        "ALL continents. Full range -180 to 180. Use 2-6 decimal places."
    ),
    "x_coord": (
        "Values MUST be projected X coordinates. These can be from various projection "
        "systems (UTM, State Plane, national grids). VARY the ranges based on different regions."
    ),
    "y_coord": (
        "Values MUST be projected Y coordinates. These can be from various projection "
        "systems (UTM, State Plane, national grids). VARY the ranges based on different regions."
    ),
    "point": (
        "Values MUST be valid WKT POINT format. VARY coordinates across ALL continents "
        "worldwide, not limited to any single region."
    ),
    "polygon": (
        "Values MUST be valid WKT POLYGON format with 4+ vertices. "
        "VARY sizes and locations across different countries and continents."
    ),
    "multi-polygon": (
        "Values MUST be valid WKT MULTIPOLYGON format. Include polygons from DIFFERENT countries."
    ),
    "line": (
        "Values MUST be valid WKT LINESTRING format. VARY lengths and locations "
        "across different continents."
    ),
    "multi-line": (
        "Values MUST be valid WKT MULTILINESTRING format. VARY across different "
        "countries and regions."
    ),
    "state": (
        "Values MUST be states/provinces/regions from WORLDWIDE: US states, "
        "Canadian provinces, UK counties, German Länder, Japanese prefectures, "
        "Australian states, etc."
    ),
    "city": (
        "Values MUST be city names from WORLDWIDE: major cities from all continents "
        "(Tokyo, London, Paris, São Paulo, Sydney, Cairo, Mumbai, etc.)."
    ),
}


def get_llm():
    """Initialize the LLM client."""
    portkey_headers = createHeaders(
        api_key=os.getenv("PORTKEY_API_KEY"),
        virtual_key=os.getenv("PROVIDER_API_KEY"),
        metadata={"_user": "yfw215"},
    )
    return ChatOpenAI(
        model="gemini-2.5-pro",
        temperature=0.95,
        base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
        default_headers=portkey_headers,
        timeout=1000,
        max_retries=3,
    )


def parse_training_df(curated_cta: pd.DataFrame) -> pd.DataFrame:
    """Parse curated CTA CSV into training dataframe."""
    training_df = {"name": [], "values": [], "label": []}
    for _, row in curated_cta.iterrows():
        training_df["name"].append(row["Column"])
        training_df["values"].append(row["Values"])
        training_df["label"].append(row["Label"])
    return pd.DataFrame(training_df)


def generate_random_values(label: str, num_values: int = 4) -> str:
    """Generate random values within valid ranges for numeric types."""
    if label not in VALUE_RANGES:
        return None
    min_val, max_val = VALUE_RANGES[label]
    values = []
    for _ in range(num_values):
        if label in ("x_coord", "y_coord"):
            val = _rand.randint(int(min_val), int(max_val))
        else:
            val = round(_rand.uniform(min_val, max_val), _rand.randint(2, 6))
        values.append(str(val))
    return ", ".join(values)


def generate_synthetic_prompt(column_name: str, column_values: str, label: str) -> str:
    """Generate prompt for LLM to create synthetic training samples."""
    style = _rand.choice(NAMING_STYLES)
    num_values = _rand.randint(3, 5)
    constraint = LABEL_CONSTRAINTS.get(label, "")
    constraint_line = f"\nCONSTRAINT: {constraint}" if constraint else ""
    example_values = generate_random_values(label, num_values) or column_values

    return f"""Given the table column '{column_name}' which represents a '{label}' type,
generate three UNIQUE alternative column names using {style} naming convention.
Example values for reference (GENERATE DIFFERENT values in similar ranges): [{example_values}]{constraint_line}
Each alternative should have {num_values} NEWLY GENERATED values (do NOT copy the examples).
Format your output EXACTLY as:
alt_name_1, val1, val2, val3; alt_name_2, val1, val2, val3; alt_name_3, val1, val2, val3
Output ONLY the formatted result, no explanations or quotes."""


def parse_llm_response(response: str, label: str) -> list:
    """Parse LLM response into training samples."""
    samples = []
    try:
        for part in response.strip().split(";"):
            part = part.strip()
            if not part:
                continue
            tokens = [t.strip() for t in part.split(",")]
            if len(tokens) >= 2:
                samples.append(
                    {
                        "name": tokens[0],
                        "values": ", ".join(tokens[1:]),
                        "label": label,
                    }
                )
    except Exception as e:
        print(f"Failed to parse response: {e}")
    return samples


def generate_synthetic_data_llm(
    training_df: pd.DataFrame,
    label_counts: pd.Series,
    target_per_class: int = 50,
    max_retries: int = 3,
    max_stale_rounds: int = 10,
) -> pd.DataFrame:
    """Generate synthetic training data using LLM for undersampled labels."""
    llm = get_llm()

    # Load existing checkpoint
    if os.path.exists(SYNTHETIC_CACHE_FILE):
        existing_df = pd.read_csv(SYNTHETIC_CACHE_FILE)
        synthetic_samples = existing_df.to_dict("records")
        existing_counts = existing_df["label"].value_counts().to_dict()
        print(f"Loaded {len(synthetic_samples)} existing samples from checkpoint")
    else:
        synthetic_samples = []
        existing_counts = {}

    seen = {(s["name"], s["values"], s["label"]) for s in synthetic_samples}
    label_generated = {
        label: existing_counts.get(label, 0) for label in label_counts.index
    }

    for label, orig_count in tqdm(
        label_counts.items(), desc="Generating synthetic data"
    ):
        current_synthetic = label_generated.get(label, 0)
        total_current = orig_count + current_synthetic
        needed = max(0, target_per_class - total_current)

        if needed == 0:
            print(f"'{label}': already at target ({total_current}/{target_per_class})")
            continue

        label_examples = training_df[training_df["label"] == label].to_dict("records")
        generated_this_run = 0
        stale_rounds = 0
        round_idx = 0

        while generated_this_run < needed and stale_rounds < max_stale_rounds:
            row = label_examples[round_idx % len(label_examples)]
            prompt = generate_synthetic_prompt(row["name"], row["values"], label)
            round_idx += 1

            for attempt in range(max_retries):
                try:
                    response = llm.invoke(prompt)
                    new_samples = parse_llm_response(response.content, label)

                    unique_samples = []
                    for s in new_samples:
                        key = (s["name"], s["values"], s["label"])
                        if key not in seen:
                            seen.add(key)
                            unique_samples.append(s)

                    if unique_samples:
                        synthetic_samples.extend(unique_samples)
                        generated_this_run += len(unique_samples)
                        label_generated[label] = label_generated.get(label, 0) + len(
                            unique_samples
                        )
                        stale_rounds = 0
                        pd.DataFrame(synthetic_samples).to_csv(
                            SYNTHETIC_CACHE_FILE, index=False
                        )
                    else:
                        stale_rounds += 1
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {label}: {e}")
                    time.sleep(2)
            else:
                stale_rounds += 1

        final_total = orig_count + label_generated.get(label, 0)
        status = (
            "✓"
            if final_total >= target_per_class
            else f"⚠ SHORT by {target_per_class - final_total}"
        )
        print(
            f"'{label}': {status} - generated {generated_this_run} this run, total={final_total}/{target_per_class}"
        )

    return pd.DataFrame(synthetic_samples)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CTA training data")
    parser.add_argument(
        "--target", type=int, default=120, help="Target samples per class"
    )
    parser.add_argument(
        "--max-stale", type=int, default=10, help="Max stale rounds before giving up"
    )
    parser.add_argument(
        "--curated-csv",
        type=str,
        default=CURATED_CTA_FILE,
        help="Path to curated CTA CSV",
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_df.csv", help="Output file path"
    )
    args = parser.parse_args()

    # Load curated data
    print(f"Loading curated CTA from {args.curated_csv}")
    curated_cta = pd.read_csv(args.curated_csv)
    training_df = parse_training_df(curated_cta)

    # Analyze label distribution
    label_counts = training_df["label"].value_counts()
    print(f"\nLabel distribution:\n{label_counts}")
    print(f"\nTotal samples: {len(training_df)}, Unique labels: {len(label_counts)}")

    # Generate synthetic data
    synthetic_df = generate_synthetic_data_llm(
        training_df,
        label_counts,
        target_per_class=args.target,
        max_stale_rounds=args.max_stale,
    )

    # Save final output
    synthetic_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(synthetic_df)} synthetic samples to {args.output}")

    # Print final distribution
    final_counts = synthetic_df["label"].value_counts()
    print(f"\nFinal synthetic distribution:\n{final_counts}")


if __name__ == "__main__":
    main()
