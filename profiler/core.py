import collections
from datetime import datetime
import logging
import numpy
import os
import pandas
from pandas.errors import EmptyDataError
import string
import time
import re
import warnings

from .numerical import mean_stddev, get_numerical_ranges
from .profile_types import identify_types, determine_dataset_type
from .spatial import (
    LatLongColumn,
    Geohasher,
    nominatim_resolve_all,
    pair_latlong_columns,
    get_spatial_ranges,
    parse_wkt_column,
    GeoClassifier,
    HybridGeoClassifier,
)
from .temporal import get_temporal_resolution
from . import types


logger = logging.getLogger(__name__)


# Mapping from GeoClassifier labels to (structural_type, [semantic_types])
# Only spatial types are included - other labels fall through to regular workflow
GEO_CLASSIFIER_SPATIAL_MAP = {
    # Lat/Long coordinates
    "latitude": (types.FLOAT, [types.LATITUDE]),
    "longitude": (types.FLOAT, [types.LONGITUDE]),
    # Projected coordinates (e.g., State Plane, UTM)
    "x_coord": (types.FLOAT, []),
    "y_coord": (types.FLOAT, []),
    # WKT geometry types
    "point": (types.GEO_POINT, []),
    "polygon": (types.GEO_POLYGON, []),
    "multi-polygon": (types.GEO_POLYGON, []),
    "line": (types.GEO_POLYGON, []),
    "multi-line": (types.GEO_POLYGON, []),
    # Address-related
    "zip5": (types.TEXT, [types.ADDRESS]),
    "zip9": (types.TEXT, [types.ADDRESS]),
    "zip_code": (types.TEXT, [types.ADDRESS]),
    "address": (types.TEXT, [types.ADDRESS]),
    # Administrative areas
    "borough": (
        types.TEXT,
        [types.ADMIN],
    ),  # Named boroughs (e.g., "Brooklyn", "Queens")
    "borough_code": (types.TEXT, [types.ADMIN]),  # Borough codes (numeric/alphanumeric)
    "city": (types.TEXT, [types.ADMIN]),
    "state": (types.TEXT, [types.ADMIN]),
    "state_code": (types.TEXT, [types.ADMIN]),
    "country": (types.TEXT, [types.ADMIN]),
    # NYC-specific identifiers (spatial context)
    "bbl": (types.INTEGER, [types.ID]),
    "bin": (types.INTEGER, [types.ID]),
}

CLASSIFIER_SEMANTIC_MAP = {
    # Spatial
    "latitude": [types.LATITUDE],
    "longitude": [types.LONGITUDE],
    "point": [types.GEO_POINT],
    "polygon": [types.GEO_POLYGON],
    "multi-polygon": [types.GEO_POLYGON],
    "line": [types.GEO_POLYGON],
    "multi-line": [types.GEO_POLYGON],
    "zip5": [types.POSTAL_CODE, types.ADDRESS],
    "zip9": [types.POSTAL_CODE, types.ADDRESS],
    "zip_code": [types.POSTAL_CODE, types.ADDRESS],
    "address": [types.ADDRESS],
    "city": [types.ADDRESS_LOCALITY, types.ADMIN],
    "state": [types.ADDRESS_REGION, types.ADMIN],
    "state_code": [types.ADDRESS_REGION, types.ADMIN],
    "country": [types.ADDRESS_COUNTRY, types.ADMIN],
    "borough": [types.ADMIN],
    "borough_code": [types.ADMIN],
    "bbl": [types.ID],
    "bin": [types.ID],
    # Non-spatial
    "identifier": [types.ID],
    "ean8": [types.GTIN8],
    "ean13": [types.GTIN13],
    "hex_color": [types.COLOR],
    "rgb_color": [types.COLOR],
    "color": [types.COLOR],
    "company": [types.ORGANIZATION],
    "credit_card_number": [types.CREDIT_CARD],
    "currency_code": [types.PRICE_CURRENCY],
    "money": [types.PRICE],
    "rating": [types.RATING_VALUE],
    "score": [types.QUANTITATIVE_VALUE],
    "percent": [types.QUANTITATIVE_VALUE],
    "flag": [types.BOOLEAN],
    "status": [types.CATEGORICAL],
    "priority": [types.CATEGORICAL],
    "severity": [types.CATEGORICAL],
    "size": [types.SIZE],
    "height": [types.HEIGHT],
    "weight": [types.WEIGHT],
    "temperature": [types.QUANTITATIVE_VALUE],
    "distance": [types.QUANTITATIVE_VALUE],
    "speed": [types.QUANTITATIVE_VALUE],
    "area": [types.QUANTITATIVE_VALUE],
    "volume": [types.QUANTITATIVE_VALUE],
    "pressure": [types.QUANTITATIVE_VALUE],
    "energy": [types.QUANTITATIVE_VALUE],
    "duration": [types.DURATION],
    "age": [types.SUGGESTED_AGE],
    "payment_method": [types.PAYMENT_METHOD],
    "shipping_method": [types.DELIVERY_METHOD],
    "platform": [
        types.OPERATING_SYSTEM,
        types.AVAILABLE_ON_DEVICE,
        types.BROWSER_REQUIREMENTS,
    ],
    "locale": [types.IN_LANGUAGE, types.SCHEDULE_TIMEZONE],
    "version": [types.VERSION],
    "hash": [types.ID],
    "email": [types.EMAIL],
    "url": [types.URL],
    "ipv4": [types.ID],
    "ipv6": [types.ID],
    "mac_address": [types.ID],
    "job": [types.JOB_TITLE],
    "name": [types.NAME],
    "first_name": [types.GIVEN_NAME],
    "last_name": [types.FAMILY_NAME],
    "prefix": [types.HONORIFIC_PREFIX],
    "phone_number": [types.TELEPHONE],
    "ssn": [types.TAX_ID],
    "file_extension": [types.ENCODING_FORMAT],
    "file_name": [types.FILE_PATH, types.NAME],
    "date_time": [types.DATE_TIME],
    "iso8601": [types.DATE_TIME],
    "unix_time": [types.DATE_TIME],
    "year": [types.DATE],
    "month_name": [types.DATE],
    "month": [types.DATE],
    "day_of_week": [types.DAY_OF_WEEK],
    "day_of_month": [types.DATE],
    "grade": [types.EDUCATIONAL_LEVEL],
}


RANDOM_SEED = 89

MAX_SIZE = 5000000  # 5 MB
SAMPLE_ROWS = 20

MAX_UNCLEAN_ADDRESSES = 0.20  # 20%


MAX_SKIPPED_ROWS = 6
"""Maximum number of rows to discard at the top of the file"""

HEADER_CONSISTENT_ROWS = 4
"""Stop throwing out lines when that many in a row have same number of columns
"""

MAX_GEOHASHES = 100


_re_word_split = re.compile(r"\W+")


def truncate_string(s, limit=140):
    """Truncate a string, replacing characters over the limit with "..."."""
    if len(s) <= limit:
        return s
    else:
        # Try to find a space
        space = s.rfind(" ", limit - 20, limit - 3)
        if space == -1:
            return s[: limit - 3] + "..."
        else:
            return s[:space] + "..."


DELIMITERS = set(string.punctuation) | set(string.whitespace)
UPPER = set(string.ascii_uppercase)
LOWER = set(string.ascii_lowercase)


def expand_attribute_name(name):
    """Expand an attribute names to keywords derived from it."""
    name = name.replace("_", " ").replace("-", " ")

    word = []
    for c in name:
        if c in DELIMITERS:
            if word:
                yield "".join(word)
                word = []
            continue

        if word:
            if (word[-1] in string.digits) != (c in string.digits) or (
                word[-1] in LOWER and c in UPPER
            ):
                yield "".join(word)
                word = []

        word.append(c)

    yield "".join(word)


# Simple no-op context manager for compatibility
class NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def load_data(data, load_max_size=None, indexes=True):
    """Load data from file path, file object, or DataFrame.

    Returns sampled data for profiling along with full-data statistics.
    """
    if load_max_size is None:
        load_max_size = MAX_SIZE

    metadata = {}
    full_data_stats = {}

    # Step 1: Convert file path/file object to DataFrame
    if isinstance(data, (str, bytes)):
        path = str(data)
        if not os.path.exists(path):
            raise ValueError("data file does not exist")

        metadata["size"] = os.path.getsize(path)
        logger.info("File size: %r bytes", metadata["size"])

        # Detect separator from extension
        sep = "\t" if path.endswith(".tsv") else ","

        # For large files, estimate rows to load based on target size
        nrows = None
        if metadata["size"] > load_max_size:
            # Sample first 1000 rows to estimate average row size
            sample_df = pandas.read_csv(
                path, dtype=str, na_filter=False, sep=sep, nrows=1000
            )
            avg_row_size = (
                metadata["size"] / (len(sample_df) + 1) if len(sample_df) > 0 else 100
            )
            nrows = int(load_max_size / avg_row_size)
            logger.info("Large file, loading ~%d rows (estimated)", nrows)

        data = pandas.read_csv(path, dtype=str, na_filter=False, sep=sep, nrows=nrows)

    elif hasattr(data, "read"):
        # File object
        data.seek(0, 2)
        metadata["size"] = data.tell()
        data.seek(0, 0)
        data = pandas.read_csv(data, dtype=str, na_filter=False)

    elif not isinstance(data, pandas.DataFrame):
        raise TypeError(
            "data should be a filename, a file object, or a pandas.DataFrame"
        )

    # Step 2: Process DataFrame (common path for all input types)
    if indexes and (
        data.index.dtype != numpy.int64
        or not pandas.Index(numpy.arange(len(data))).equals(data.index)
    ):
        data = data.reset_index()

    metadata["nb_rows"] = len(data)
    logger.info("DataFrame: %d rows, %d columns", data.shape[0], data.shape[1])

    # Compute stats on data before sampling (cheap operations)
    # Also extract 3 non-null sample values for geo classifier
    for col in data.columns:
        # Count distinct values
        full_data_stats[col] = {"num_distinct_values": data[col].nunique()}

        # Extract 3 unique non-null sample values from full dataset
        sample_values = []
        seen = set()
        col_series = data[col]
        for v in col_series:
            v_str = str(v).strip()
            if v_str and v_str not in ("", "nan", "None") and v_str not in seen:
                seen.add(v_str)
                sample_values.append(v_str)
                if len(sample_values) >= 3:
                    break
        full_data_stats[col]["sample_values"] = sample_values

    # Sample if DataFrame exceeds target size
    avg_row_size = data.memory_usage(deep=True).sum() / max(len(data), 1)
    max_rows = int(load_max_size / avg_row_size)
    if len(data) > max_rows:
        logger.info(
            "Sampling %d rows for profiling (target size: %d bytes)",
            max_rows,
            load_max_size,
        )
        data = data.sample(n=max_rows, random_state=RANDOM_SEED)

    metadata["nb_profiled_rows"] = len(data)

    # Convert to string dtype (workaround for pandas nan-as-'nan' bug)
    data = data.astype(object).fillna("").astype(str)

    return data, metadata, data.columns, full_data_stats


def process_column(
    array,
    column_meta,
    *,
    manual=None,
    plots=True,
    coverage=True,
    datamart_geo_data=None,
    nominatim=None,
    geo_prediction=None,  # Pre-computed from batch prediction
):
    structural_type = None
    semantic_types_dict = {}
    additional_meta = {}
    used_geo_prediction = False
    classifier_meta = None

    # Use pre-computed geo_prediction if available and no manual override
    if geo_prediction is not None:
        label = geo_prediction.get("label")
        if label:
            classifier_meta = {
                "label": label,
                "confidence": geo_prediction.get("confidence", 0.0),
                "source": geo_prediction.get("source", "ml"),
            }
            if geo_prediction.get("validated") is not None:
                classifier_meta["validated"] = geo_prediction.get("validated")
            if geo_prediction.get("filtered"):
                classifier_meta["filtered"] = True
            if geo_prediction.get("rejected"):
                classifier_meta["rejected"] = True

    if classifier_meta is not None and manual is None:
        label = classifier_meta["label"]
        if (
            label in GEO_CLASSIFIER_SPATIAL_MAP
            and not classifier_meta.get("filtered")
            and not classifier_meta.get("rejected")
        ):
            # Geo classifier identified a spatial type
            used_geo_prediction = True
            structural_type, semantic_list = GEO_CLASSIFIER_SPATIAL_MAP[label]
            semantic_types_dict = {st: None for st in semantic_list}
            additional_meta["geo_classifier"] = classifier_meta

    # If geo prediction wasn't used, use regular identify_types
    if not used_geo_prediction:
        structural_type, semantic_types_dict, additional_meta = identify_types(
            array, column_meta["name"], datamart_geo_data, manual
        )
        if classifier_meta is not None and "geo_classifier" not in additional_meta:
            additional_meta["geo_classifier"] = classifier_meta

    if (
        classifier_meta is not None
        and not classifier_meta.get("filtered")
        and not classifier_meta.get("rejected")
    ):
        label = classifier_meta.get("label")
        if label in CLASSIFIER_SEMANTIC_MAP:
            for sem_type in CLASSIFIER_SEMANTIC_MAP[label]:
                semantic_types_dict.setdefault(sem_type, None)

    # Log column type with source information
    if used_geo_prediction:
        geo_info = additional_meta.get("geo_classifier", {})
        label = geo_info.get("label", "unknown")
        confidence = geo_info.get("confidence", 0.0)
        source = geo_info.get("source", "ml")

        # Get sample values for logging (use stored samples if available)
        column_name = column_meta.get("name", "unknown")
        samples_str = ""
        if geo_prediction and "sample_values" in geo_prediction:
            sample_values = geo_prediction["sample_values"]
            samples_str = ", ".join([str(v) for v in sample_values])

        logger.info(
            "Column type %s [%s] (from geo_classifier: column=%r, label=%s, confidence=%.4f, source=%s, samples=[%s])",
            structural_type,
            ", ".join(semantic_types_dict),
            column_name,
            label,
            confidence,
            source,
            samples_str,
        )
    else:
        logger.info(
            "Column type %s [%s]",
            structural_type,
            ", ".join(semantic_types_dict),
        )

    # Set structural type
    column_meta["structural_type"] = structural_type
    # Add semantic types to the ones already present
    sem_types = column_meta.setdefault("semantic_types", [])
    for sem_type in semantic_types_dict:
        if sem_type not in sem_types:
            sem_types.append(sem_type)
    # Insert additional metadata
    column_meta.update(additional_meta)

    # Resolved values are returned so they can be used again to compute spatial
    # coverage
    resolved = {}

    # Compute ranges for numerical data
    if structural_type in (types.INTEGER, types.FLOAT) and (coverage or plots):
        # Get numerical values needed for either ranges or plot
        numerical_values = []
        for e in array:
            try:
                e = float(e)
            except ValueError:
                pass
            else:
                if -3.4e38 < e < 3.4e38:  # Overflows in ES
                    numerical_values.append(e)

        # Compute ranges from numerical values
        if coverage:
            column_meta["mean"], column_meta["stddev"] = mean_stddev(numerical_values)

            ranges = get_numerical_ranges(numerical_values)
            if ranges:
                column_meta["coverage"] = ranges

        # Compute histogram from numerical values
        if plots:
            counts, edges = numpy.histogram(
                numerical_values,
                bins=10,
            )
            counts = [int(i) for i in counts]
            edges = [float(f) for f in edges]
            column_meta["plot"] = {
                "type": "histogram_numerical",
                "data": [
                    {
                        "count": count,
                        "bin_start": edges[i],
                        "bin_end": edges[i + 1],
                    }
                    for i, count in enumerate(counts)
                ],
            }

    if types.DATE_TIME in semantic_types_dict:
        datetimes = semantic_types_dict[types.DATE_TIME]
        resolved["datetimes"] = datetimes
        timestamps = numpy.empty(
            len(datetimes),
            dtype="float32",
        )
        for j, dt in enumerate(datetimes):
            timestamps[j] = dt.timestamp()
        resolved["timestamps"] = timestamps

        # Compute histogram from temporal values
        if plots and "plot" not in column_meta:
            counts, edges = numpy.histogram(timestamps, bins=10)
            counts = [int(i) for i in counts]
            column_meta["plot"] = {
                "type": "histogram_temporal",
                "data": [
                    {
                        "count": count,
                        "date_start": datetime.utcfromtimestamp(
                            float(edges[i]),
                        ).isoformat(),
                        "date_end": datetime.utcfromtimestamp(
                            float(edges[i + 1]),
                        ).isoformat(),
                    }
                    for i, count in enumerate(counts)
                ],
            }

    # Compute histogram from categorical values
    if plots and types.CATEGORICAL in semantic_types_dict:
        counter = collections.Counter()
        for value in array:
            if not value:
                continue
            counter[value] += 1
        counts = counter.most_common(5)
        counts = sorted(counts)
        column_meta["plot"] = {
            "type": "histogram_categorical",
            "data": [
                {
                    "bin": value,
                    "count": count,
                }
                for value, count in counts
            ],
        }

    # Compute histogram from textual values
    if plots and types.TEXT in semantic_types_dict and "plot" not in column_meta:
        counter = collections.Counter()
        for value in array:
            for word in _re_word_split.split(value):
                word = word.lower()
                if word:
                    counter[word] += 1
        counts = counter.most_common(5)
        column_meta["plot"] = {
            "type": "histogram_text",
            "data": [
                {
                    "bin": value,
                    "count": count,
                }
                for value, count in counts
            ],
        }

    # Resolve addresses into coordinates
    if (
        nominatim is not None
        and structural_type == types.TEXT
        and types.TEXT in semantic_types_dict
        and types.ADMIN not in semantic_types_dict
    ):
        locations, non_empty = nominatim_resolve_all(
            nominatim,
            array,
        )
        if non_empty > 0:
            unclean_ratio = 1.0 - len(locations) / non_empty
            if unclean_ratio <= MAX_UNCLEAN_ADDRESSES:
                resolved["addresses"] = locations
                if types.ADDRESS not in column_meta["semantic_types"]:
                    column_meta["semantic_types"].append(types.ADDRESS)

    # Set level of administrative areas
    if types.ADMIN in semantic_types_dict:
        admin_value = semantic_types_dict[types.ADMIN]
        if admin_value is not None:
            # disambiguate_admin_areas returns (level, areas) tuple or just areas
            if isinstance(admin_value, tuple) and len(admin_value) == 2:
                level, areas = admin_value
            else:
                # Fallback: admin_value is just the areas list
                level, areas = None, admin_value
            if level is not None:
                column_meta["admin_area_level"] = level
            resolved["admin_areas"] = areas

    return resolved


def process_dataset(
    data,
    geo_classifier=True,
    geo_classifier_threshold=0.5,
    geo_classifier_model_dir=None,
    include_sample=False,
    coverage=True,
    plots=False,
    indexes=True,
    load_max_size=None,
    metadata=None,
    nominatim=None,
    datamart_geo_data=None,
    **kwargs,
):
    """Compute all metafeatures from a dataset.

    :param data: path to dataset, or file object, or DataFrame
    :param geo_classifier: ``True`` to enable geo_classifier
    :param geo_classifier_threshold: Confidence threshold for geo_classifier
        predictions (default: 0.85). Predictions below this threshold are
        flagged as low-confidence and do not override heuristics.
    :param geo_classifier_model_dir: Optional model directory to load CTA model
        files from when geo_classifier is True.
    :param include_sample: Set to True to include a few random rows to the
        result. Useful to present to a user.
    :param coverage: Whether to compute data ranges
    :param plots: Whether to compute plots
    :param indexes: Whether to include indexes. If True (the default), the
        input is a DataFrame, and it has index(es) different from the default
        range, they will appear in the result with the columns.
    :param load_max_size (bytes): Target size of the data to be analyzed. The data will
        be randomly sampled if it is bigger. Defaults to `MAX_SIZE`, currently
        5 MB (5000000). This is different from the sample data included in the result.
    :param metadata: The metadata provided by the discovery plugin (might be
        very limited).
    :param nominatim: URL of the Nominatim server
    :param datamart_geo_data: ``True`` or a datamart_geo.GeoData instance to use to
        resolve named administrative territorial entities
    :return: JSON structure (dict)
    """
    # Track runtime for each pipeline step
    pipeline_start = time.perf_counter()
    step_times = {}

    if "sample_size" in kwargs:
        warnings.warn(
            "Argument 'sample_size' is deprecated, use 'load_max_size'",
            DeprecationWarning,
        )
        load_max_size = kwargs.pop("sample_size")
    if kwargs:
        raise TypeError(
            "process_dataset() got unexpected keyword argument %r" % next(iter(kwargs))
        )

    if datamart_geo_data is True:
        from datamart_geo import GeoData

        datamart_geo_data = GeoData.from_local_cache()

    if geo_classifier is True:
        geo_classifier = HybridGeoClassifier(
            GeoClassifier(model_dir=geo_classifier_model_dir)
        )

    if metadata is None:
        metadata = {}

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    step_start = time.perf_counter()
    logger.info("[STEP 1/6] Loading data...")

    try:
        data, file_metadata, column_names, full_data_stats = load_data(
            data,
            load_max_size=load_max_size,
            indexes=indexes,
        )
    except EmptyDataError:
        logger.warning("Dataframe is empty!")
        metadata["nb_rows"] = 0
        metadata["nb_profiled_rows"] = 0
        metadata["columns"] = []
        metadata["types"] = []
        return metadata

    step_times["1_load_data"] = time.perf_counter() - step_start
    logger.info(
        "[STEP 1/6] Data loaded in %.3fs (%d rows, %d columns)",
        step_times["1_load_data"],
        data.shape[0],
        data.shape[1],
    )

    metadata.update(file_metadata)
    metadata["nb_profiled_rows"] = data.shape[0]
    metadata["nb_columns"] = data.shape[1]

    if "columns" in metadata:
        columns = metadata["columns"]
        logger.info("Using provided columns info")
        if len(columns) != len(data.columns):
            raise ValueError("Column metadata doesn't match number of columns")
        for column_meta, name in zip(columns, column_names):
            if "name" in column_meta and column_meta["name"] != name:
                raise ValueError("Column names don't match")
            column_meta["name"] = name
    else:
        logger.info("Setting column names from header")
        columns = [{"name": name} for name in column_names]
        metadata["columns"] = columns

    if data.shape[0] == 0:
        logger.info("0 rows, returning early")
        metadata["types"] = []
        return metadata

    # Get manual updates from the user
    manual_columns = {}
    if "manual_annotations" in metadata:
        if "columns" in metadata["manual_annotations"]:
            manual_columns = {
                col["name"]: col for col in metadata["manual_annotations"]["columns"]
            }

    # Cache some values that have been resolved for type identification but are
    # useful for spatial coverage computation: admin areas and addresses
    # Having to resolve them once to see if they're valid and a second time to
    # build coverage information would be too slow
    resolved_columns = {}

    # =========================================================================
    # STEP 2: Batch ML prediction for ALL columns (single forward pass!)
    # =========================================================================
    step_start = time.perf_counter()
    logger.info("[STEP 2/6] Geo classifier batch prediction...")

    geo_predictions = {}  # column_idx -> prediction dict

    if geo_classifier:
        # Collect samples from all columns (no manual override)
        # Use pre-extracted sample values from full dataset (before sampling)
        columns_for_batch = []
        for column_idx, column_meta in enumerate(columns):
            name = column_meta["name"]
            if name in manual_columns:
                continue  # Skip columns with manual annotations

            # Use pre-extracted sample values from full dataset
            if name in full_data_stats and "sample_values" in full_data_stats[name]:
                sample_values = full_data_stats[name]["sample_values"]
                if sample_values:
                    columns_for_batch.append((column_idx, name, sample_values))

        # BATCH PREDICTION - single forward pass for ALL columns!
        if columns_for_batch:
            batch_inputs = [(name, vals) for _, name, vals in columns_for_batch]
            batch_results = geo_classifier.predict_batch(
                batch_inputs, threshold=geo_classifier_threshold
            )

            for (column_idx, name, sample_values), prediction in zip(
                columns_for_batch, batch_results
            ):
                # Store prediction with sample values used
                geo_predictions[column_idx] = prediction
                geo_predictions[column_idx]["sample_values"] = sample_values

    step_times["2_geo_batch_predict"] = time.perf_counter() - step_start

    # Incrementally save geo attributes to CSV
    if geo_predictions:
        geo_results = []
        for col_idx, prediction in geo_predictions.items():
            column_meta = columns[col_idx]
            prediction = geo_predictions[col_idx]
            label = prediction.get("label")
            if (
                not label
                or label not in GEO_CLASSIFIER_SPATIAL_MAP
                or prediction.get("filtered")
                or prediction.get("rejected")
            ):
                continue
            col_name = column_meta["name"]
            geo_results.append(
                {
                    "name": col_name,
                    "values": prediction["sample_values"],
                    "label": label,
                }
            )
        if geo_results:
            geo_df = pandas.DataFrame(geo_results)
            if os.path.exists("output/geo_classifier_results.csv"):
                mode = "a"
                header = False
            else:
                os.makedirs("output", exist_ok=True)
                mode = "w"
                header = True
            geo_df.to_csv(
                "output/geo_classifier_results.csv",
                index=False,
                mode=mode,
                header=header,
            )
            logger.info(
                "Saved %d geo classifier results to output/geo_classifier_results.csv",
                len(geo_results),
            )

    logger.info(
        "[STEP 2/6] Geo batch prediction completed in %.3fs (%d columns)",
        step_times["2_geo_batch_predict"],
        len(geo_predictions),
    )

    # =========================================================================
    # STEP 3: Process columns (apply predictions + compute features)
    # =========================================================================
    step_start = time.perf_counter()
    logger.info("[STEP 3/6] Processing %d columns...", len(columns))

    for col_idx, column_meta in enumerate(columns):
        resolved = process_column(
            data[column_meta["name"]],
            column_meta,
            manual=manual_columns.get(column_meta["name"]),
            plots=plots,
            coverage=coverage,
            datamart_geo_data=datamart_geo_data,
            nominatim=nominatim,
            geo_prediction=geo_predictions.get(col_idx),
        )
        resolved_columns[col_idx] = resolved

        # Override with exact stats from full data (computed before sampling)
        col_name = column_meta["name"]
        if col_name in full_data_stats:
            stats = full_data_stats[col_name]
            if "num_distinct_values" in column_meta:
                column_meta["num_distinct_values"] = stats["num_distinct_values"]

    step_times["3_process_columns"] = time.perf_counter() - step_start
    logger.info(
        "[STEP 3/6] Column processing completed in %.3fs",
        step_times["3_process_columns"],
    )

    # =========================================================================
    # STEP 4: Pair lat/long columns and determine dataset types
    # =========================================================================
    step_start = time.perf_counter()
    logger.info("[STEP 4/6] Pairing lat/long columns and determining dataset types...")

    # Pair lat & long columns
    columns_lat = [
        LatLongColumn(
            index=col_idx,
            name=col["name"],
            annot_pair=manual_columns.get(col["name"], {}).get("latlong_pair"),
        )
        for col_idx, col in enumerate(columns)
        if types.LATITUDE in col["semantic_types"]
    ]
    columns_long = [
        LatLongColumn(
            index=col_idx,
            name=col["name"],
            annot_pair=manual_columns.get(col["name"], {}).get("latlong_pair"),
        )
        for col_idx, col in enumerate(columns)
        if types.LONGITUDE in col["semantic_types"]
    ]
    latlong_pairs, (missed_lat, missed_long) = pair_latlong_columns(
        columns_lat, columns_long
    )

    # Log missed columns
    if missed_lat:
        logger.warning("Unmatched latitude columns: %r", missed_lat)
    if missed_long:
        logger.warning("Unmatched longitude columns: %r", missed_long)

    # Remove semantic type from unpaired columns
    for col in columns:
        if col["name"] in missed_lat:
            col["semantic_types"].remove(types.LATITUDE)
        if col["name"] in missed_long:
            col["semantic_types"].remove(types.LONGITUDE)

    # Identify the overall dataset types (numerical, categorical, spatial, or temporal)
    dataset_types = collections.Counter()
    for column_meta in columns:
        dataset_type = determine_dataset_type(
            column_meta["structural_type"],
            column_meta["semantic_types"],
        )
        if dataset_type:
            dataset_types[dataset_type] += 1
    for key, d_type in [
        ("nb_spatial_columns", types.DATASET_SPATIAL),
        ("nb_temporal_columns", types.DATASET_TEMPORAL),
        ("nb_categorical_columns", types.DATASET_CATEGORICAL),
        ("nb_numerical_columns", types.DATASET_NUMERICAL),
    ]:
        if dataset_types[d_type]:
            metadata[key] = dataset_types[d_type]
    metadata["types"] = sorted(set(dataset_types))

    step_times["4_latlong_pairing"] = time.perf_counter() - step_start
    logger.info(
        "[STEP 4/6] Lat/long pairing completed in %.3fs (%d pairs found)",
        step_times["4_latlong_pairing"],
        len(latlong_pairs),
    )

    # =========================================================================
    # STEP 5: Compute spatial coverage
    # =========================================================================
    step_start = time.perf_counter()

    if coverage:
        logger.info("[STEP 5/6] Computing spatial coverage...")
        spatial_coverage = []
        with NoOpContext():
            # Compute sketches from lat/long pairs
            for col_lat, col_long in latlong_pairs:
                lat_values = data.iloc[:, col_lat.index]
                lat_values = pandas.to_numeric(lat_values, errors="coerce")
                long_values = data.iloc[:, col_long.index]
                long_values = pandas.to_numeric(long_values, errors="coerce")
                mask = (
                    ~numpy.isnan(lat_values)
                    & ~numpy.isnan(long_values)
                    & (-90.0 < lat_values)
                    & (lat_values < 90.0)
                    & (-180.0 < long_values)
                    & (long_values < 180.0)
                )

                if mask.any():
                    lat_values = lat_values[mask]
                    long_values = long_values[mask]
                    values = numpy.array([lat_values, long_values]).T
                    logger.info(
                        "Computing spatial sketch lat=%r long=%r (%d rows)",
                        col_lat.name,
                        col_long.name,
                        len(values),
                    )
                    # Ranges
                    spatial_ranges = get_spatial_ranges(values)
                    # Geohashes
                    builder = Geohasher(number=MAX_GEOHASHES)
                    builder.add_points(values)
                    hashes = builder.get_hashes_json()

                    spatial_coverage.append(
                        {
                            "type": "latlong",
                            "column_names": [col_lat.name, col_long.name],
                            "column_indexes": [
                                col_lat.index,
                                col_long.index,
                            ],
                            "geohashes4": hashes,
                            "ranges": spatial_ranges,
                            "number": len(values),
                        }
                    )

            # Compute sketches from WKT points
            for i, col in enumerate(columns):
                if col["structural_type"] != types.GEO_POINT:
                    continue
                latlong = col.get("point_format") == "lat,long"
                name = col["name"]
                values = parse_wkt_column(
                    data.iloc[:, i],
                    latlong=latlong,
                )
                total = numpy.sum(data.iloc[:, i].apply(lambda x: bool(x)))
                if len(values) < 0.5 * total:
                    logger.warning(
                        "Most data points did not parse correctly as "
                        "point (%s) col=%d %r",
                        "lat,long" if latlong else "long,lat",
                        i,
                        col,
                    )
                if values:
                    logger.info(
                        "Computing spatial sketches point=%r (%d rows)",
                        name,
                        len(values),
                    )
                    # Ranges
                    spatial_ranges = get_spatial_ranges(values)
                    # Geohashes
                    builder = Geohasher(number=MAX_GEOHASHES)
                    builder.add_points(values)
                    hashes = builder.get_hashes_json()

                    spatial_coverage.append(
                        {
                            "type": "point_latlong" if latlong else "point",
                            "column_names": [name],
                            "column_indexes": [i],
                            "geohashes4": hashes,
                            "ranges": spatial_ranges,
                            "number": len(values),
                        }
                    )

            for idx, resolved in resolved_columns.items():
                # Compute sketches from addresses
                if "addresses" in resolved:
                    locations = resolved["addresses"]

                    name = columns[idx]["name"]
                    logger.info(
                        "Computing spatial sketches address=%r (%d rows)",
                        name,
                        len(locations),
                    )
                    # Ranges
                    spatial_ranges = get_spatial_ranges(locations)
                    # Geohashes
                    builder = Geohasher(number=MAX_GEOHASHES)
                    builder.add_points(locations)
                    hashes = builder.get_hashes_json()

                    spatial_coverage.append(
                        {
                            "type": "address",
                            "column_names": [name],
                            "column_indexes": [idx],
                            "geohashes4": hashes,
                            "ranges": spatial_ranges,
                            "number": len(locations),
                        }
                    )

                # Compute sketches from administrative areas
                if "admin_areas" in resolved:
                    areas = resolved["admin_areas"]

                    name = columns[idx]["name"]
                    logger.info(
                        "Computing spatial sketches admin_areas=%r (%d rows)",
                        name,
                        len(areas),
                    )
                    cov = {
                        "type": "admin",
                        "column_names": [name],
                        "column_indexes": [idx],
                    }

                    # Merge into a single range
                    merged = None
                    for area in areas:
                        if area is None:
                            continue
                        new = area.bounds
                        if new:
                            if merged is None:
                                merged = new
                            else:
                                merged = (
                                    min(merged[0], new[0]),
                                    max(merged[1], new[1]),
                                    min(merged[2], new[2]),
                                    max(merged[3], new[3]),
                                )
                    if (
                        merged is not None
                        and merged[1] - merged[0] > 0.01
                        and merged[3] - merged[2] > 0.01
                    ):
                        logger.info("Computed bounding box")
                        cov["ranges"] = [
                            {
                                "range": {
                                    "type": "envelope",
                                    "coordinates": [
                                        [merged[0], merged[3]],
                                        [merged[1], merged[2]],
                                    ],
                                },
                            },
                        ]
                    else:
                        logger.info("Couldn't build a bounding box")

                    # Compute geohashes
                    builder = Geohasher(number=MAX_GEOHASHES)
                    for area in areas:
                        if area is None or not area.bounds:
                            continue
                        builder.add_aab(area.bounds)
                    hashes = builder.get_hashes_json()
                    if hashes:
                        cov["geohashes4"] = hashes

                    # Count
                    cov["number"] = builder.total

                    if "ranges" in cov or "geohashes4" in cov:
                        spatial_coverage.append(cov)

        if spatial_coverage:
            metadata["spatial_coverage"] = spatial_coverage

        step_times["5_spatial_coverage"] = time.perf_counter() - step_start
        logger.info(
            "[STEP 5/6] Spatial coverage completed in %.3fs (%d coverage entries)",
            step_times["5_spatial_coverage"],
            len(spatial_coverage),
        )

        # =====================================================================
        # STEP 6: Compute temporal coverage
        # =====================================================================
        step_start = time.perf_counter()
        logger.info("[STEP 6/6] Computing temporal coverage...")
        temporal_coverage = []

        with NoOpContext():
            # Datetime columns
            for idx, col in enumerate(columns):
                if types.DATE_TIME not in col["semantic_types"]:
                    continue
                datetimes = resolved_columns[idx]["datetimes"]
                timestamps = resolved_columns[idx]["timestamps"]
                logger.info(
                    "Computing temporal ranges datetime=%r (%d rows)",
                    col["name"],
                    len(datetimes),
                )

                # Get temporal ranges
                ranges = get_numerical_ranges(timestamps)
                if not ranges:
                    continue

                # Get temporal resolution
                resolution = get_temporal_resolution(datetimes)

                temporal_coverage.append(
                    {
                        "type": "datetime",
                        "column_names": [col["name"]],
                        "column_indexes": [idx],
                        "column_types": [types.DATE_TIME],
                        "ranges": ranges,
                        "temporal_resolution": resolution,
                    }
                )

            # TODO: Times split over multiple columns

        if temporal_coverage:
            metadata["temporal_coverage"] = temporal_coverage

        step_times["6_temporal_coverage"] = time.perf_counter() - step_start
        logger.info(
            "[STEP 6/6] Temporal coverage completed in %.3fs (%d coverage entries)",
            step_times["6_temporal_coverage"],
            len(temporal_coverage),
        )
    else:
        step_times["5_spatial_coverage"] = 0
        step_times["6_temporal_coverage"] = 0
        logger.info("[STEP 5-6/6] Coverage computation skipped (coverage=False)")

    # Attribute names
    attribute_keywords = []
    for col in columns:
        attribute_keywords.append(col["name"])
        kw = list(expand_attribute_name(col["name"]))
        if kw != [col["name"]]:
            attribute_keywords.extend(kw)
    metadata["attribute_keywords"] = attribute_keywords

    # Sample data
    if include_sample:
        step_start = time.perf_counter()
        rand = numpy.random.RandomState(RANDOM_SEED)
        choose_rows = rand.choice(
            len(data),
            min(SAMPLE_ROWS, len(data)),
            replace=False,
        )
        choose_rows.sort()  # Keep it in order
        sample = data.iloc[choose_rows]
        sample = sample.map(truncate_string)  # Truncate long values
        metadata["sample"] = sample.to_csv(index=False, lineterminator="\r\n")
        step_times["sample_extraction"] = time.perf_counter() - step_start

    # =========================================================================
    # PIPELINE SUMMARY
    # =========================================================================
    total_time = time.perf_counter() - pipeline_start

    # Build summary string
    summary_parts = [f"\n{'='*60}"]
    summary_parts.append("PROFILING PIPELINE SUMMARY")
    summary_parts.append(f"{'='*60}")
    summary_parts.append(f"Dataset: {data.shape[0]} rows × {data.shape[1]} columns")
    summary_parts.append(f"{'-'*60}")

    for step_name, step_time in step_times.items():
        pct = (step_time / total_time * 100) if total_time > 0 else 0
        bar_len = int(pct / 2)  # 50 chars = 100%
        bar = "█" * bar_len + "░" * (50 - bar_len)
        summary_parts.append(f"{step_name:25s} {step_time:7.3f}s ({pct:5.1f}%) {bar}")

    summary_parts.append(f"{'-'*60}")
    summary_parts.append(f"{'TOTAL':25s} {total_time:7.3f}s")
    summary_parts.append(f"{'='*60}\n")

    logger.info("\n".join(summary_parts))

    # Also store timing in metadata for analysis
    metadata["_profiling_times"] = {
        "steps": step_times,
        "total": total_time,
    }

    return metadata
