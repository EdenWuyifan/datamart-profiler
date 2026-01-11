#!/usr/bin/env python3
"""
Rule-based Column Type Annotation (CTA) classifier.

This module provides pattern-based classification for common spatial and non-spatial
column types. Use as a pre-filter before ML classification, or as a standalone classifier.

Usage:
    from rules_cta import RuleBasedCTA

    classifier = RuleBasedCTA()
    result = classifier.classify("BBL", ["1001234567", "2005678901", "3012345678"])
    # Returns: {"label": "bbl", "confidence": 0.95, "rule": "bbl_pattern"}
"""

import re
from typing import Any

import numpy as np
import pandas as pd


class RuleBasedCTA:
    """Rule-based column type classifier using patterns and heuristics."""

    # Column name patterns → type mappings
    NAME_PATTERNS = {
        # Non-spatial ID patterns (high priority)
        "non_spatial_id": {
            "patterns": [
                r"(?i)^(physical_?id|physicalid)$",
                r"(?i)^(object_?id|objectid|oid)$",
                r"(?i)^(feature_?id|featureid)$",
                r"(?i)^(segment_?id[t]?|segmentid[t]?)$",
                r"(?i)^(record_?id|recordid)$",
                r"(?i)^(row_?id|rowid)$",
                r"(?i)^(pk_?id|primary_?key)$",
                r"(?i)^(ref_?num|reference_?number)$",
                r"(?i)^(fid|gid|uid|oid)$",
                r"(?i)^index(_?(right|left))?$",
                r"(?i)^(seq|sequence)(_?num)?$",
            ],
            "label": "non_spatial",
        },
        # BBL (Borough-Block-Lot)
        "bbl": {
            "patterns": [r"(?i)^bbl$", r"(?i)^borough_?block_?lot$"],
            "label": "bbl",
        },
        # BIN (Building Identification Number)
        "bin": {
            "patterns": [r"(?i)^bin$", r"(?i)^building_?id(entification)?_?num(ber)?$"],
            "label": "bin",
        },
        # Latitude
        "latitude": {
            "patterns": [
                r"(?i)^lat(itude)?$",
                r"(?i)^y_?coord(inate)?$",
                r"(?i)^(geo_?)?lat(itude)?$",
                r"(?i)^(point_?)?lat$",
                r"(?i)^lat_?(dd|deg|decimal)$",
            ],
            "label": "latitude",
        },
        # Longitude
        "longitude": {
            "patterns": [
                r"(?i)^lon(g|gitude)?$",
                r"(?i)^lng$",
                r"(?i)^x_?coord(inate)?$",
                r"(?i)^(geo_?)?lon(gitude)?$",
                r"(?i)^(point_?)?lon$",
                r"(?i)^lon_?(dd|deg|decimal)$",
            ],
            "label": "longitude",
        },
        # X coordinate (projected)
        "x_coord": {
            "patterns": [
                r"(?i)^x_?coord$",
                r"(?i)^xcoord$",
                r"(?i)^easting$",
                r"(?i)^x_?pos(ition)?$",
            ],
            "label": "x_coord",
        },
        # Y coordinate (projected)
        "y_coord": {
            "patterns": [
                r"(?i)^y_?coord$",
                r"(?i)^ycoord$",
                r"(?i)^northing$",
                r"(?i)^y_?pos(ition)?$",
            ],
            "label": "y_coord",
        },
        # Zip code
        "zip_code": {
            "patterns": [
                r"(?i)^zip(_?code)?$",
                r"(?i)^postal(_?code)?$",
                r"(?i)^postcode$",
            ],
            "label": "zip_code",
        },
        # Borough
        "borough": {
            "patterns": [
                r"(?i)^borough(_?(name|code))?$",
                r"(?i)^boro(_?(name|code))?$",
                r"(?i)^district(_?(name|code))?$",
                r"(?i)^ward(_?(name|code))?$",
            ],
            "label": "borough_code",
        },
        # City
        "city": {
            "patterns": [r"(?i)^city(_?name)?$", r"(?i)^municipality$", r"(?i)^town$"],
            "label": "city",
        },
        # State
        "state": {
            "patterns": [
                r"(?i)^state(_?(name|code))?$",
                r"(?i)^province$",
                r"(?i)^region$",
            ],
            "label": "state",
        },
        # Address
        "address": {
            "patterns": [
                r"(?i)^address$",
                r"(?i)^street_?address$",
                r"(?i)^full_?address$",
                r"(?i)^location_?address$",
            ],
            "label": "address",
        },
        # Geometry columns
        "geometry": {
            "patterns": [
                r"(?i)^(the_?)?geom(etry)?$",
                r"(?i)^geometry_?wkt$",
                r"(?i)^wkt$",
                r"(?i)^shape$",
            ],
            "label": None,  # Determined by value pattern
        },
    }

    # NYC Borough names/codes (for backward compatibility)
    # Note: Borough codes are now extended to support US-wide districts/wards
    NYC_BOROUGHS = {
        "manhattan",
        "bronx",
        "brooklyn",
        "queens",
        "staten island",
        "mn",
        "bx",
        "bk",
        "qn",
        "si",
        "new york",
        "kings",
        "richmond",
        "1",
        "2",
        "3",
        "4",
        "5",
    }

    def __init__(self, strict: bool = False):
        """
        Initialize the rule-based classifier.

        Args:
            strict: If True, only return results with high confidence.
        """
        self.strict = strict
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._compiled = {}
        for name, config in self.NAME_PATTERNS.items():
            self._compiled[name] = {
                "regexes": [re.compile(p) for p in config["patterns"]],
                "label": config["label"],
            }

    def _match_name_pattern(self, column_name: str) -> tuple[str | None, str | None]:
        """Match column name against known patterns."""
        for pattern_name, config in self._compiled.items():
            for regex in config["regexes"]:
                if regex.match(column_name):
                    return config["label"], pattern_name
        return None, None

    def _extract_numeric_values(self, values: list) -> list[float]:
        """Extract numeric values from a list, handling strings."""
        nums = []
        for v in values:
            if pd.isna(v):
                continue
            try:
                nums.append(float(v))
            except (ValueError, TypeError):
                continue
        return nums

    def _check_bbl_pattern(self, values: list) -> bool:
        """Check if values match NYC BBL pattern (10 digits, starts with 1-5)."""
        for v in values[:10]:  # Check first 10 values
            if pd.isna(v):
                continue
            try:
                s = str(int(float(v)))
                if len(s) != 10 or s[0] not in "12345":
                    return False
            except (ValueError, TypeError):
                return False
        return len(values) > 0

    def _check_bin_pattern(self, values: list) -> bool:
        """Check if values match NYC BIN pattern (7 digits, starts with 1-5)."""
        for v in values[:10]:
            if pd.isna(v):
                continue
            try:
                s = str(int(float(v)))
                if len(s) != 7 or s[0] not in "12345":
                    return False
            except (ValueError, TypeError):
                return False
        return len(values) > 0

    def _check_latitude_range(self, values: list) -> bool:
        """Check if numeric values are in valid US latitude range (24.5-49.5, or extended 18.0-71.5)."""
        nums = self._extract_numeric_values(values)
        if not nums:
            return False
        # US latitudes: contiguous US (24.5-49.5) or extended for AK/HI (18.0-71.5)
        # Should be POSITIVE (no negative values)
        return all(
            (18.0 <= n <= 71.5) and n >= 0  # Positive latitudes in US range
            for n in nums
        )

    def _check_longitude_range(self, values: list) -> bool:
        """Check if numeric values are in valid US longitude range (-125.0 to -66.0)."""
        nums = self._extract_numeric_values(values)
        if not nums:
            return False
        # US longitudes: always NEGATIVE, range -125.0 to -66.0
        return all(-125.0 <= n <= -66.0 and n < 0 for n in nums)

    def _check_projected_coord_range(self, values: list) -> bool:
        """Check if values look like projected coordinates (large magnitude)."""
        nums = self._extract_numeric_values(values)
        if not nums:
            return False
        # X_coord: magnitude ~1,000,000-15,000,000
        # Y_coord: magnitude ~2,000,000-7,000,000
        # Accept either range (X or Y)
        return all(
            (1_000_000 <= abs(n) <= 15_000_000) or (2_000_000 <= abs(n) <= 7_000_000)
            for n in nums
        )

    def _check_wkt_geometry(self, values: list) -> str | None:
        """Check if values are WKT geometry strings and return type."""
        wkt_patterns = {
            "point": r"^\s*POINT\s*\(",
            "multi-point": r"^\s*MULTIPOINT\s*\(",
            "line": r"^\s*LINESTRING\s*\(",
            "multi-line": r"^\s*MULTILINESTRING\s*\(",
            "polygon": r"^\s*POLYGON\s*\(",
            "multi-polygon": r"^\s*MULTIPOLYGON\s*\(",
        }
        for v in values[:5]:
            if pd.isna(v) or not isinstance(v, str):
                continue
            v_upper = v.upper()
            for geom_type, pattern in wkt_patterns.items():
                if re.match(pattern, v_upper, re.IGNORECASE):
                    return geom_type
        return None

    def _check_borough_values(self, values: list) -> bool:
        """Check if values are borough/district/ward codes from US cities."""
        matches = 0
        valid_values = [v for v in values[:10] if not pd.isna(v)]

        for v in valid_values:
            v_str = str(v).strip()
            v_lower = v_str.lower()

            # Check NYC boroughs (still supported)
            if v_lower in self.NYC_BOROUGHS:
                matches += 1
                continue

            # Check numeric codes (1-50 for wards/districts)
            if re.match(r"^\d+$", v_str):
                try:
                    code = int(v_str)
                    if 1 <= code <= 50:  # Common range for US wards/districts
                        matches += 1
                        continue
                except ValueError:
                    pass

            # Check short alphanumeric codes (e.g., B1, Q2, AB-03)
            if re.match(r"^[A-Za-z]{1,3}[-]?\d{1,3}$", v_str):
                matches += 1
                continue

        return matches >= len(valid_values) * 0.5

    def _check_zip_pattern(self, values: list) -> bool:
        """Check if values match common postal code patterns."""
        zip_patterns = [
            r"^\d{5}$",  # US ZIP
            r"^\d{5}-\d{4}$",  # US ZIP+4
            r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$",  # Canadian
            r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$",  # UK
            r"^\d{3}-\d{4}$",  # Japan
        ]
        matches = 0
        valid_values = [v for v in values[:10] if not pd.isna(v)]

        for v in valid_values:
            v_str = str(v).strip()
            if any(re.match(p, v_str) for p in zip_patterns):
                matches += 1
        return matches >= len([v for v in values[:10] if not pd.isna(v)]) * 0.5

    def _is_generic_numeric_id(self, column_name: str, values: list) -> bool:
        """Check if column looks like a generic numeric ID (not BBL/BIN)."""
        name_lower = column_name.lower()

        # Strong ID indicators in name
        id_keywords = [
            "id",
            "index",
            "segment",
            "physical",
            "object",
            "fid",
            "pk",
            "key",
            "num",
            "seq",
        ]
        has_id_keyword = any(kw in name_lower for kw in id_keywords)

        # Check if values are integers (IDs are usually integers)
        nums = self._extract_numeric_values(values)
        if not nums:
            return False

        all_integers = all(n == int(n) for n in nums)

        # High variance suggests IDs (random assignment)
        if len(nums) > 1:
            variance = np.var(nums)
            mean = np.mean(nums)
            cv = np.sqrt(variance) / mean if mean != 0 else 0
            high_variance = cv > 0.3
        else:
            high_variance = True

        return has_id_keyword and all_integers and high_variance

    def classify(self, column_name: str, values: list[Any]) -> dict[str, Any] | None:
        """
        Classify a column based on its name and values.

        Args:
            column_name: The column header/name
            values: Sample values from the column (list of any type)

        Returns:
            Dict with 'label', 'confidence', 'rule' if a rule matches, else None
        """
        # Filter out None/NaN values for analysis
        clean_values = [v for v in values if not pd.isna(v)]
        if not clean_values:
            return None

        # 1. Check WKT geometry patterns first (value-based)
        wkt_type = self._check_wkt_geometry(clean_values)
        if wkt_type:
            return {"label": wkt_type, "confidence": 0.98, "rule": "wkt_geometry"}

        # 2. Check column name patterns
        name_label, pattern_name = self._match_name_pattern(column_name)

        # 3. Apply value-based validation for certain types
        if name_label == "bbl":
            if self._check_bbl_pattern(clean_values):
                return {
                    "label": "bbl",
                    "confidence": 0.95,
                    "rule": "bbl_name_and_pattern",
                }
            # Name says BBL but values don't match pattern
            return {"label": "bbl", "confidence": 0.6, "rule": "bbl_name_only"}

        if name_label == "bin":
            if self._check_bin_pattern(clean_values):
                return {
                    "label": "bin",
                    "confidence": 0.95,
                    "rule": "bin_name_and_pattern",
                }
            return {"label": "bin", "confidence": 0.6, "rule": "bin_name_only"}

        if name_label == "non_spatial":
            return {"label": "non_spatial", "confidence": 0.9, "rule": pattern_name}

        if name_label == "latitude":
            if self._check_latitude_range(clean_values):
                return {
                    "label": "latitude",
                    "confidence": 0.95,
                    "rule": "lat_name_and_range",
                }
            return {"label": "latitude", "confidence": 0.7, "rule": "lat_name_only"}

        if name_label == "longitude":
            if self._check_longitude_range(clean_values):
                return {
                    "label": "longitude",
                    "confidence": 0.95,
                    "rule": "lon_name_and_range",
                }
            return {"label": "longitude", "confidence": 0.7, "rule": "lon_name_only"}

        if name_label == "x_coord":
            if self._check_projected_coord_range(clean_values):
                return {
                    "label": "x_coord",
                    "confidence": 0.9,
                    "rule": "x_coord_name_and_range",
                }
            return {"label": "x_coord", "confidence": 0.7, "rule": "x_coord_name_only"}

        if name_label == "y_coord":
            if self._check_projected_coord_range(clean_values):
                return {
                    "label": "y_coord",
                    "confidence": 0.9,
                    "rule": "y_coord_name_and_range",
                }
            return {"label": "y_coord", "confidence": 0.7, "rule": "y_coord_name_only"}

        if name_label == "borough_code":
            if self._check_borough_values(clean_values):
                return {
                    "label": "borough_code",
                    "confidence": 0.95,
                    "rule": "borough_name_and_values",
                }
            return {
                "label": "borough_code",
                "confidence": 0.7,
                "rule": "borough_name_only",
            }

        if name_label == "zip_code":
            if self._check_zip_pattern(clean_values):
                return {
                    "label": "zip_code",
                    "confidence": 0.95,
                    "rule": "zip_name_and_pattern",
                }
            return {"label": "zip_code", "confidence": 0.7, "rule": "zip_name_only"}

        if name_label in ("city", "state", "address"):
            return {
                "label": name_label,
                "confidence": 0.8,
                "rule": f"{name_label}_name",
            }

        # 4. Check for generic numeric IDs (fallback)
        if self._is_generic_numeric_id(column_name, clean_values):
            return {
                "label": "non_spatial",
                "confidence": 0.75,
                "rule": "generic_numeric_id",
            }

        # 5. Value-only checks (no name match)
        # Check BBL pattern without name hint
        if self._check_bbl_pattern(clean_values):
            nums = self._extract_numeric_values(clean_values)
            if nums and all(len(str(int(n))) == 10 for n in nums[:5]):
                return {"label": "bbl", "confidence": 0.6, "rule": "bbl_pattern_only"}

        # Check BIN pattern without name hint
        if self._check_bin_pattern(clean_values):
            nums = self._extract_numeric_values(clean_values)
            if nums and all(len(str(int(n))) == 7 for n in nums[:5]):
                return {"label": "bin", "confidence": 0.6, "rule": "bin_pattern_only"}

        # No rule matched
        return None

    def classify_dataframe(
        self, df: pd.DataFrame, sample_size: int = 100
    ) -> dict[str, dict[str, Any]]:
        """
        Classify all columns in a DataFrame.

        Args:
            df: Input DataFrame
            sample_size: Number of values to sample per column

        Returns:
            Dict mapping column names to classification results
        """
        results = {}
        for col in df.columns:
            col_data = df[col].dropna()
            sample = col_data.iloc[: min(sample_size, len(col_data))].tolist()
            result = self.classify(col, sample)
            results[col] = result
        return results


class HybridCTAClassifier:
    """
    Hybrid classifier: ML prediction first, then rule-based validation for specific types.

    If ML predicts BBL, BIN, latitude, longitude, zip_code, or geometry types,
    validates with rules. If validation fails, returns non_spatial.
    """

    # Types that require rule-based validation
    VALIDATE_TYPES = {
        "bbl",
        "bin",
        "latitude",
        "longitude",
        "zip_code",
        "point",
        "polygon",
        "multi-polygon",
        "line",
        "multi-line",
        "x_coord",
        "y_coord",
    }

    def __init__(self, ml_classifier):
        """
        Initialize hybrid classifier.

        Args:
            ml_classifier: ML-based classifier (required)
        """
        self.rules = RuleBasedCTA()
        self.ml = ml_classifier

    def _validate_with_rules(self, label: str, column_name: str, values: list) -> bool:
        """Validate ML prediction using rule-based checks."""
        clean_values = [v for v in values if not pd.isna(v)]

        if label == "bbl":
            # Must match BBL pattern OR have BBL in name
            name_match = bool(re.match(r"(?i)^bbl$", column_name))
            pattern_match = self.rules._check_bbl_pattern(clean_values)
            return name_match or pattern_match

        if label == "bin":
            name_match = bool(re.match(r"(?i)^bin$", column_name))
            pattern_match = self.rules._check_bin_pattern(clean_values)
            return name_match or pattern_match

        if label == "latitude":
            name_match = any(
                re.match(p, column_name)
                for p in [r"(?i)^lat(itude)?$", r"(?i)^y_?coord(inate)?$"]
            )
            range_match = self.rules._check_latitude_range(clean_values)
            return name_match or range_match

        if label == "longitude":
            name_match = any(
                re.match(p, column_name)
                for p in [
                    r"(?i)^lon(g|gitude)?$",
                    r"(?i)^lng$",
                    r"(?i)^x_?coord(inate)?$",
                ]
            )
            range_match = self.rules._check_longitude_range(clean_values)
            return name_match or range_match

        if label == "x_coord":
            name_match = bool(re.match(r"(?i)^x_?coord$", column_name))
            range_match = self.rules._check_projected_coord_range(clean_values)
            return name_match or range_match

        if label == "y_coord":
            name_match = bool(re.match(r"(?i)^y_?coord$", column_name))
            range_match = self.rules._check_projected_coord_range(clean_values)
            return name_match or range_match

        if label == "zip_code":
            name_match = bool(
                re.match(r"(?i)^zip(_?code)?$|^postal(_?code)?$", column_name)
            )
            pattern_match = self.rules._check_zip_pattern(clean_values)
            return name_match or pattern_match

        if label in ("point", "polygon", "multi-polygon", "line", "multi-line"):
            # Check if values contain WKT geometry
            wkt_type = self.rules._check_wkt_geometry(clean_values)
            return wkt_type is not None

        # Unknown type, assume valid
        return True

    def classify(
        self, column_name: str, values: list[Any], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Classify a column: ML first, then validate sensitive types with rules.

        Returns:
            List of predictions with 'label', 'confidence', and 'source'
        """
        # Step 1: Get ML prediction
        values_str = ", ".join(str(v) for v in values[:10])
        text = f"{column_name}: {values_str}"
        ml_results = self.ml.predict(text, top_k=top_k)

        if not ml_results:
            return [{"label": "non_spatial", "confidence": 0.5, "source": "default"}]

        top_prediction = ml_results[0]
        top_label = top_prediction["label"]
        top_confidence = top_prediction["confidence"]

        # Step 2: If top prediction needs validation, check with rules
        if top_label in self.VALIDATE_TYPES:
            is_valid = self._validate_with_rules(top_label, column_name, values)

            if is_valid:
                # Validation passed
                return [
                    {
                        "label": top_label,
                        "confidence": top_confidence,
                        "source": "ml+validated",
                    }
                ] + [
                    {"label": r["label"], "confidence": r["confidence"], "source": "ml"}
                    for r in ml_results[1:top_k]
                ]
            else:
                # Validation failed → return non_spatial
                return [
                    {
                        "label": "non_spatial",
                        "confidence": 0.9,
                        "source": f"ml:{top_label}→rule_rejected",
                    }
                ] + [
                    {"label": r["label"], "confidence": r["confidence"], "source": "ml"}
                    for r in ml_results[1:top_k]
                    if r["label"] != "non_spatial"
                ]

        # Step 3: No validation needed, return ML results as-is
        return [
            {"label": r["label"], "confidence": r["confidence"], "source": "ml"}
            for r in ml_results[:top_k]
        ]


# === Demo / Test ===
if __name__ == "__main__":
    classifier = RuleBasedCTA()

    test_cases = [
        ("segmentidt", [2920.0, 8171.0, 8177.0]),
        ("PHYSICALID", [30644, 15071, 168005]),
        ("index_right", [100766, 48271, 26106]),
        ("BBL", [1001234567, 2005678901, 3012345678]),
        ("BIN", [1012345, 2023456, 3034567]),
        ("lat", [40.7128, 40.7580, 40.6892]),
        ("longitude", [-74.0060, -73.9855, -73.9442]),
        ("Borough Name", ["Manhattan", "Brooklyn", "Queens"]),
        ("zip_code", ["10001", "11201", "10451"]),
        ("geometry_wkt", ["POINT(-73.99 40.71)", "POINT(-73.98 40.72)"]),
        (
            "the_geom",
            ["MULTIPOLYGON(((-73.9 40.7, -73.8 40.7, -73.8 40.8, -73.9 40.7)))"],
        ),
        ("random_col", [1, 2, 3]),  # Should return None (no match)
    ]

    print("Rule-Based CTA Classifier Demo\n" + "=" * 50)
    for col_name, values in test_cases:
        result = classifier.classify(col_name, values)
        if result:
            print(f"\n{col_name}: {values[:3]}")
            print(
                f"  → {result['label']} (confidence: {result['confidence']:.2f}, rule: {result['rule']})"
            )
        else:
            print(f"\n{col_name}: {values[:3]}")
            print("  → No rule matched (use ML fallback)")
