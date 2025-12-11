#!/usr/bin/env python3
"""Generate synthetic CTA (Column Type Annotation) training data using LLM."""

import os
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
OUTPUT_FILE = "synthetic_df.csv"


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


def generate_synthetic_prompt(column_name: str, column_values: str, label: str) -> str:
    """Generate prompt for LLM to create synthetic training samples."""
    num_values = 3

    return f"""You are a synthetic tabular data generator used from a Python program.

Your job is: given a table column name, its semantic label (one of a fixed set), a naming style, and a few example values, you must:

1. Propose THREE UNIQUE alternative column names that follow the requested naming convention.
2. For EACH alternative column name, generate a list of NEW sample values that:
   - Match the semantic label,
   - Are realistic for UNITED STATES data only,
   - DO NOT copy or trivially paraphrase the provided example values,
   - Obey the strict constraints below for that label.


  Given the table column: 
  - Column name: '{column_name}'
  - Column values (sample): {column_values}
  - Label: '{label}'

  Each alternative should have {num_values} NEWLY GENERATED values (do NOT copy the examples).
  
  Format output EXACTLY as:
  alt_name_1, val1, val2, val3; alt_name_2, val1, val2, val3; alt_name_3, val1, val2, val3
  Output ONLY the formatted result, no explanations or quotes.

You MUST:
- Make all THREE alternative names distinct in wording.
- Generate exactly {num_values} values per alternative name.
- NEVER reuse the exact example values; generate new ones in similar ranges or domains.
- Do NOT add quotes, brackets, extra text, or newlines.

All data must be US-focused. Use the following label-specific rules:

──────────────── LABEL-SPECIFIC GENERATION RULES (US-ONLY) ────────────────

1) label = "latitude"  (spatial)
- Decimal latitude in degrees (single numeric per value).
- Range (contiguous US): 24.5 ≤ lat ≤ 49.5 (you MAY occasionally use AK/HI: 18.0–71.5).
- Always POSITIVE; MAY include a leading "+"; NEVER include a minus sign.
- No commas or brackets; just a float-like string (e.g. 40.7128, 34.05, +47.60621).
- Use 3–7 decimal places; optional surrounding spaces are allowed but not required.

2) label = "longitude"  (spatial)
- Decimal longitude in degrees (single numeric per value).
- Range (US): -125.0 ≤ lon ≤ -66.0.
- Always NEGATIVE; no positive longitudes.
- No commas or brackets; just a float-like string (e.g. -74.0060, -118.2437).
- Use 3–7 decimal places.

3) label = "x_coord"  (spatial)
- Projected X coordinate, large magnitude (e.g., Web Mercator / State Plane).
- Numeric values with magnitude ~1,000,000–15,000,000 in ABSOLUTE value.
- Use integers or floats with few decimals: -9735653, -8234567.12.
- No commas; single number per value.
- Should typically be NEGATIVE for US if you mimic Web Mercator (but not strictly required; keep large magnitude).

4) label = "y_coord"  (spatial)
- Projected Y coordinate, large magnitude.
- Numeric values ~2,000,000–7,000,000 for US.
- Integers or floats, e.g. 3123456, 4567890.75.
- No commas; single number per value.

5) label = "point"  (spatial)
- Represent a geographic point ONLY as WKT POINT, in (lon lat) order.
- Format: POINT(<lon> <lat>)
  - Example shape: POINT(-74.0060 40.7128)
- <lon> in [-125, -66], <lat> in [24.5, 49.5] (or extended US ranges).
- No commas inside the parentheses; exactly two numbers.
- Always start with "POINT(" and end with ")".

6) label = "line"  (spatial)
- Represent as WKT LINESTRING.
- Format: LINESTRING(x1 y1,x2 y2,...)
  - x = lon, y = lat, all in US ranges.
- Use 2–10 coordinate pairs.
- No extra parentheses; do NOT use MULTILINESTRING here.

7) label = "multi-line"  (spatial)
- Represent as WKT MULTILINESTRING.
- Format: MULTILINESTRING((x1 y1,x2 y2,...),(x'1 y'1,x'2 y'2,...))
- Use 2–5 lines; each line has 2–10 points.
- Always start with "MULTILINESTRING(" and use nested parentheses.

8) label = "polygon"  (spatial)
- Represent as WKT POLYGON with a single outer ring.
- Format: POLYGON((x1 y1,x2 y2,...,xN yN))
- First and last coordinate pair must be IDENTICAL to close the ring.
- Use 4–15 vertices.
- lon/lat ranges follow US bounds.

9) label = "multi-polygon"  (spatial)
- Represent as WKT MULTIPOLYGON.
- Format: MULTIPOLYGON(((x1 y1,...,xN yN)),((x'1 y'1,...,x'M y'M)))
- 2–3 polygons; each polygon ring is closed (first = last).
- Use 4–15 vertices per polygon.

10) label = "borough"  (spatial)
- Represents US borough-level or similar sub-city administrative units.
- Values may include:
  - Named boroughs in large US cities (e.g., "Brooklyn", "Queens", "Staten Island"),
  - Alaska boroughs (e.g., "Anchorage Borough", "Matanuska-Susitna Borough"),
  - Other “borough”-style or district names used in US municipalities (e.g., "North Borough", "Downtown Borough").
- Mostly Title Case; you may occasionally use ALL CAPS.
- Do NOT generate raw numeric codes here (those belong to "borough_code").

11) label = "borough_code"  (spatial)
- Represents codes for boroughs or similar sub-city units in the US.
- Values can be:
  - Small integers (e.g., 1, 2, 3, 10, 12),
  - Or short alphanumeric codes (e.g., B1, Q2, AB-03) if plausible as district/borough codes.
- Use plain tokens with digits and optional letters; avoid full names.
- NEVER include full borough names or words like "Borough" in this label.

12) label = "city"  (spatial)
- US city/town names ONLY; no state, no zip code.
- Examples of shape (NOT to be copied directly):
  - New York, Los Angeles, Chicago, Houston, Miami, Suffield, Ellington, Union, Boulder, Madison, Raleigh, Phoenix, Omaha.
- Mostly Title Case; some ALL CAPS or lowercase variations are ok.
- No ", ST" or zip here; just the city/town name string.
- NOT restricted to New York City; use cities from across the US.

13) label = "state"  (spatial)
- Full US state names ONLY (50 states; DC optional).
- Examples of shape:
  - New York, California, Texas, Florida, Connecticut, Massachusetts, Washington, Colorado, Georgia, Ohio, Arizona.
- Mostly Title Case; occasional ALL CAPS is ok.
- Do NOT generate city names here.
- NOT restricted to any single region; may be any US state.

14) label = "state_code"  (spatial)
- 2-letter USPS state codes ONLY, uppercase.
- Examples (shape only): NY, CA, TX, FL, IL, MA, CT, WA, CO, GA, NC, AZ, OH.
- Exactly 2 characters; no dots or extra symbols.
- Any valid US state code (nationwide), not just codes around NYC.

15) label = "country"  (spatial)
- Country names for the United States ONLY.
- Allowed forms:
  - United States
  - United States of America
  - USA
  - U.S.A.
- Do NOT output any other countries.

16) label = "zip5"  (spatial)
- 5-digit US ZIP codes as strings.
- Pattern: exactly 5 digits, leading zeros allowed: "02115", "10001", "60616", "94110".
- NEVER add a dash or extra digits.
- Use ZIP patterns from any US state (not NYC-specific).

17) label = "zip9"  (spatial)
- ZIP+4 codes.
- Pattern: 5 digits, dash, 4 digits (exactly 10 characters).
  - e.g. "02115-1234", "10001-0001".
- Do NOT omit the dash or shorten the groups.
- Use plausible US ZIP+4 combinations from any US region.

18) label = "address"  (spatial)
- US street/residence addresses. Each value MUST look like a real postal-style address fragment.
- All values MUST start with:
  - A street number (integer, or integer+letter), or
  - A PO Box indicator (e.g., "PO Box 123").
- Include a street name and optional suffix; may also include city/state/zip.
- Examples of shape (NOT to be copied directly):
  - 1600 Pennsylvania Ave NW
  - 25 Broadway Apt 12B
  - 742 Evergreen Terrace
  - 123 Main St, Suffield, CT 06078
  - PO Box 123, Boston, MA 02115
- Addresses may be anywhere in the US (not just NYC).
- Do NOT generate pure "City, ST" without a number for this label.

19) label = "bbl"  (spatial, NYC Borough–Block–Lot)
- NYC-specific identifier; this ONE label remains NYC-focused.
- Format: "<borough>-<block>-<lot>"
  - borough: integer 1–5.
  - block: 1–5 digits, optionally zero-padded.
  - lot: 1–4 digits, optionally zero-padded.
- Examples of shape:
  - 3-14151-2922
  - 1-00023-0005
  - 5-1234-12
- Exactly two dashes and three numeric segments.

20) label = "bin"  (spatial, NYC Building ID)
- NYC-specific identifier; this ONE label remains NYC-focused.
- 7-digit numeric string ONLY.
- Pattern: exactly 7 digits, e.g. 1087654, 5852956, 3001234.
- No dashes, no letters.

21) label = "non_spatial"  (NON-spatial)
- Values MUST NOT encode recognizable spatial/location information.
- Good options:
  - Person names: Alice, Bob, Charlie Brown, John Doe, etc.
  - Generic IDs: A12345, USER_00123, ID-98765 (avoid pure 5-digit or 7-digit numerics).
  - Categorical labels: High, Medium, Low; Yes, No; Pending, Approved, Rejected.
  - Numeric measures: 0, 1, 12, 250, 3.14, 0.75 (avoid values that look exactly like lat/lon or x/y ranges).
  - Dates/times: 2024-01-15, 2023-12-31 10:30:00.
  - Generic notes: Patient reported mild symptoms; Order shipped on Monday.
- Avoid:
  - Anything that matches ZIP, BIN, BBL, POINT/LINESTRING/POLYGON/etc., city/state/country names, or address formats.

──────────────── OUTPUT FORMAT (VERY IMPORTANT) ────────────────

After applying all the rules above, respond with EXACTLY:

alt_name_1, val1, val2, ..., valN; alt_name_2, val1, val2, ..., valN; alt_name_3, val1, val2, ..., valN

- Replace N with {num_values}.
- Use commas and semicolons exactly as shown.
- Do NOT wrap values or names in quotes.
- Do NOT prepend or append any explanation, label, or extra text.
"""


def parse_response(response: str, label: str) -> list:
    """Parse LLM response into training samples."""
    samples = []
    try:
        content = response.content if hasattr(response, "content") else str(response)
        for part in content.strip().split(";"):
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


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CTA training data")
    parser.add_argument(
        "--curated-csv",
        type=str,
        default=CURATED_CTA_FILE,
        help="Path to curated CTA CSV",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE, help="Output file path"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=120,
        help="Target number of synthetic samples per class",
    )
    args = parser.parse_args()

    # Load curated data
    print(f"Loading curated CTA from {args.curated_csv}")
    curated_df = pd.read_csv(args.curated_csv)

    # Initialize LLM
    llm = get_llm()

    # Load existing cache if it exists
    if os.path.exists(SYNTHETIC_CACHE_FILE):
        cache_df = pd.read_csv(SYNTHETIC_CACHE_FILE)
        all_samples = cache_df.to_dict("records")
        existing_count = len(all_samples)
        print(f"Loaded {existing_count} existing samples from {SYNTHETIC_CACHE_FILE}")
    else:
        all_samples = []
        existing_count = 0

    # Count existing samples per label
    existing_counts = {}
    if all_samples:
        result_df = pd.DataFrame(all_samples)
        existing_counts = result_df["label"].value_counts().to_dict()
        print(f"\nExisting samples per label: {existing_counts}")

    # Get unique labels from curated data
    unique_labels = curated_df["Label"].unique()
    print(f"\nTarget: {args.target} samples per class")
    print(f"Unique labels: {len(unique_labels)}")

    # Generate synthetic data for each label until target is reached
    for label in unique_labels:
        current_count = existing_counts.get(label, 0)
        needed = max(0, args.target - current_count)

        if needed == 0:
            print(f"\n'{label}': Already at target ({current_count}/{args.target})")
            continue

        print(
            f"\n'{label}': Need {needed} more samples (current: {current_count}/{args.target})"
        )

        # Get all curated examples for this label
        label_examples = curated_df[curated_df["Label"] == label].to_dict("records")
        generated_this_label = 0
        example_idx = 0

        while generated_this_label < needed:
            row = label_examples[example_idx % len(label_examples)]
            example_idx += 1

            prompt = generate_synthetic_prompt(
                row["Column"], row["Values"], row["Label"]
            )
            try:
                response = llm.invoke(prompt)
                samples = parse_response(response, row["Label"])

                if samples:
                    all_samples.extend(samples)
                    generated_this_label += len(samples)
                    # Cache after each successful generation
                    pd.DataFrame(all_samples).to_csv(SYNTHETIC_CACHE_FILE, index=False)
                    print(
                        f"  Generated {len(samples)} samples (total for '{label}': {current_count + generated_this_label}/{args.target})"
                    )
            except Exception as e:
                print(f"  Error processing {row['Column']}: {e}")
                continue

        final_count = current_count + generated_this_label
        status = (
            "✓"
            if final_count >= args.target
            else f"⚠ SHORT by {args.target - final_count}"
        )
        print(
            f"'{label}': {status} - generated {generated_this_label} this run, total={final_count}/{args.target}"
        )

    # Save final output
    result_df = pd.DataFrame(all_samples)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(result_df)} total samples to {args.output}")
    print(f"Added {len(result_df) - existing_count} new samples")

    # Print final distribution
    if len(result_df) > 0:
        final_counts = result_df["label"].value_counts()
        print(f"\nFinal distribution:\n{final_counts}")


if __name__ == "__main__":
    main()
