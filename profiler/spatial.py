import collections
from dataclasses import dataclass
import json
import logging
import math
import os
import numpy
import numpy.random
import re
import requests
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors._kd_tree import KDTree
import time
import typing
from urllib.parse import urlencode

from .warning_tools import ignore_warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from pathlib import Path


logger = logging.getLogger(__name__)


# Model download configuration - individual file URLs from NYU Box
GEO_MODEL_FILES = {
    "model.pt": "https://nyu.box.com/shared/static/2x0rnwhte4e8fbxkf0jyyd48cid232ci.pt",
    "config.json": "https://nyu.box.com/shared/static/eojx465mrfu1b5vasjkilt3rh9uvz19p.json",
    "label_encoder.json": "https://nyu.box.com/shared/static/hl347mei57rxe1flgap19j2lxst3xpa9.json",
}


N_RANGES = 3
MIN_RANGE_SIZE = 0.10  # 10%

SPATIAL_RANGE_DELTA_LONG = 0.0001
SPATIAL_RANGE_DELTA_LAT = 0.0001

MAX_ADDRESS_LENGTH = 90  # 90 characters
MAX_NOMINATIM_REQUESTS = 200
NOMINATIM_BATCH_SIZE = 20
NOMINATIM_MIN_SPLIT_BATCH_SIZE = 2  # Batches >=this are divided on failure

LATITUDE = ("latitude", "lat", "ycoord", "y_coord")
LONGITUDE = ("longitude", "long", "lon", "lng", "xcoord", "x_coord")

MAX_WRONG_LEVEL_ADMIN = 0.10  # 10%


def get_spatial_ranges(values):
    """Build a small number (3) of bounding boxes from lat/long points.

    This performs K-Means clustering, returning a maximum of 3 clusters as
    bounding boxes.
    """

    clustering = KMeans(n_clusters=min(N_RANGES, len(values)), random_state=0)
    with ignore_warnings(ConvergenceWarning):
        clustering.fit(values)
    logger.info("K-Means clusters: %r", list(clustering.cluster_centers_))

    # Compute confidence intervals for each range
    ranges = []
    sizes = []
    for rg in range(N_RANGES):
        cluster = [values[i] for i in range(len(values)) if clustering.labels_[i] == rg]
        if not cluster:
            continue

        # Eliminate clusters of outliers
        if len(cluster) < MIN_RANGE_SIZE * len(values):
            continue

        cluster.sort(key=lambda p: p[0])
        min_idx = int(0.05 * len(cluster))
        max_idx = int(0.95 * len(cluster))
        min_lat = cluster[min_idx][0]
        max_lat = cluster[max_idx][0]
        cluster.sort(key=lambda p: p[1])
        min_long = cluster[min_idx][1]
        max_long = cluster[max_idx][1]
        ranges.append(
            [
                [min_long, max_lat],
                [max_long, min_lat],
            ]
        )
        sizes.append(len(cluster))
    ranges.sort()
    logger.info("Ranges: %r", ranges)
    logger.info("Sizes: %r", sizes)

    # Lucene needs shapes to have an area for tessellation (no point or line)
    for rg in ranges:
        if rg[0][0] == rg[1][0]:
            rg[0][0] -= SPATIAL_RANGE_DELTA_LONG
            rg[1][0] += SPATIAL_RANGE_DELTA_LONG
        if rg[0][1] == rg[1][1]:
            rg[0][1] += SPATIAL_RANGE_DELTA_LAT
            rg[1][1] -= SPATIAL_RANGE_DELTA_LAT

    # Convert to Elasticsearch syntax
    ranges = [
        {"range": {"type": "envelope", "coordinates": coords}} for coords in ranges
    ]
    return ranges


def normalize_latlong_column_name(name, substrings):
    """Find the remainder of the column name after removing a substring.

    This goes over the substrings in order and removes the first it finds from
    the name. You should therefore put the more specific substrings first and
    the shorter ones last. For example, this is used to turn both
    ``"cab_latitude_from"`` and ``"cab_longitude_from"`` into ``"cab__from"``
    which can then be matched.
    """
    name = name.strip().lower()
    for substr in substrings:
        idx = name.find(substr)
        if idx >= 0:
            name = name[:idx] + name[idx + len(substr) :]
            break
    return name


@dataclass
class LatLongColumn(object):
    index: int
    name: str
    annot_pair: typing.Optional[str]


def pair_latlong_columns(columns_lat, columns_long):
    """Go through likely latitude and longitude columns and finds pairs.

    This tries to find columns that match apart from latitude and longitude
    keywords (e.g. `longitude`, `long`, `lon`).
    """
    # Normalize latitude column names
    normalized_lat = {}
    for i, col in enumerate(columns_lat):
        # check if a pair was defined by the user (human-in-the-loop)
        name = col.annot_pair
        if name is None:
            # Use normalized column name
            name = normalize_latlong_column_name(col.name, LATITUDE)
        normalized_lat[name] = i

    # Go over normalized longitude column names and try to match
    pairs = []
    missed_long = []
    for col in columns_long:
        # check if a pair was defined by the user (human-in-the-loop)
        name = col.annot_pair
        if name is None:
            # Use normalized column name
            name = normalize_latlong_column_name(col.name, LONGITUDE)
        if name in normalized_lat:
            pairs.append(
                (
                    columns_lat[normalized_lat.pop(name)],
                    col,
                )
            )
        else:
            missed_long.append(col.name)

    # Gather missed columns
    missed_lat = [columns_lat[i].name for i in sorted(normalized_lat.values())]

    return pairs, (missed_lat, missed_long)


_re_loc = re.compile(
    r"\("
    r"(-?[0-9]{1,3}\.[0-9]{1,15})"
    r"(?:,| |(?:, ))"
    r"(-?[0-9]{1,3}\.[0-9]{1,15})"
    r"\)$"
)


def _parse_point(value, latlong):
    m = _re_loc.search(value)
    if m is not None:
        try:
            x = float(m.group(1))
            y = float(m.group(2))
        except ValueError:
            return None
        if latlong:
            x, y = y, x
        if -180.0 < x < 180.0 and -90.0 < y < 90.0:
            return y, x


def parse_wkt_column(values, latlong=False):
    """Parse a pandas.Series of points in WKT format or similar "(long, lat)".

    :param latlong: If False (the default), read ``(long, lat)`` format. If
        True, read ``(lat, long)``.
    :returns: A list of ``(lat, long)`` pairs
    """
    # Parse points
    values = values.apply(_parse_point, latlong=latlong)
    # Drop NaN values
    values = values.dropna(axis=0)

    return list(values)


_nominatim_session = requests.Session()


def nominatim_query(url, *, q):
    url = url.rstrip("/")
    res = None  # Avoids warnings
    for i in range(5):
        if i > 0:
            time.sleep(1)
        if isinstance(q, (tuple, list)):
            # Batch query
            res = _nominatim_session.get(
                url
                + "/search?"
                + urlencode(
                    {
                        "batch": json.dumps([{"q": qe} for qe in q]),
                        "format": "jsonv2",
                    }
                ),
            )
        else:
            # Normal query
            res = _nominatim_session.get(
                url + "/search?" + urlencode({"q": q, "format": "jsonv2"}),
            )
        if res.status_code not in (502, 503, 504):
            break
    res.raise_for_status()
    if not res.headers["Content-Type"].startswith("application/json"):
        raise requests.HTTPError(
            "Response is not JSON for URL: %s" % res.url,
            response=res,
        )
    if isinstance(q, (tuple, list)):
        return res.json()["batch"]
    else:
        return res.json()


def _nominatim_batch(url, batch, locations, cache):
    try:
        locs = nominatim_query(url, q=list(batch.keys()))
    except requests.HTTPError as e:
        if e.response.status_code in (500, 414) and len(batch) >= max(
            2, NOMINATIM_MIN_SPLIT_BATCH_SIZE
        ):
            # Try smaller batch size
            batch_list = list(batch.items())
            mid = len(batch) // 2
            return _nominatim_batch(
                url, dict(batch_list[:mid]), locations, cache
            ) + _nominatim_batch(url, dict(batch_list[mid:]), locations, cache)
        raise e from None

    not_found = 0
    for location, (value, count) in zip(locs, batch.items()):
        if location:
            loc = (
                float(location[0]["lat"]),
                float(location[0]["lon"]),
            )
            cache[value] = loc
            locations.extend([loc] * count)
        else:
            cache[value] = None
            not_found += count
    batch.clear()
    return not_found


def nominatim_resolve_all(url, array, max_requests=MAX_NOMINATIM_REQUESTS):
    cache = {}
    locations = []
    not_found = 0  # Unique locations not found
    non_empty = 0
    processed = 0
    batch = {}

    for processed, value in enumerate(array):
        value = value.strip()
        if not value:
            continue
        non_empty += 1

        if len(value) > MAX_ADDRESS_LENGTH:
            continue
        elif value in cache:
            if cache[value] is not None:
                locations.append(cache[value])
        elif value in batch:
            batch[value] += 1
        else:
            batch[value] = 1
            if len(batch) == NOMINATIM_BATCH_SIZE:
                not_found += _nominatim_batch(url, batch, locations, cache)
                if len(cache) >= max_requests:
                    break

    if batch and len(cache) < max_requests:
        not_found += _nominatim_batch(url, batch, locations, cache)

    logger.info(
        "Performed %d Nominatim queries (%d hits). Found %d/%d",
        len(cache),
        len(cache) - not_found,
        len(locations),
        processed,
    )
    return locations, non_empty


def disambiguate_admin_areas(admin_areas):
    """This takes admin areas resolved from names and tries to disambiguate.

    Each name in the input will have been resolved to multiple possible areas,
    making the input a list of list. We want to build a simple list, where
    each name has been resolved to the most likely area.

    We choose so that all the areas are of the same level (e.g. all countries,
    or all states, but not a mix of counties and states), and if possible all
    in the same parent area (for example, states of the same country, or
    counties in states of the same country).
    """
    # Count possible options
    options = collections.Counter()
    for candidates in admin_areas:
        # Count options from the same list of candidates only once
        options_for_entry = set()
        for area in candidates:
            level = area.type.value
            area = area.get_parent_area()
            while area:
                options_for_entry.add((level, area))
                area = area.get_parent_area()
            options_for_entry.add((level, None))
        options.update(options_for_entry)

    # Throw out options with too few matches
    threshold = (1.0 - MAX_WRONG_LEVEL_ADMIN) * len(admin_areas)
    threshold = max(3, threshold)
    options = [
        (option, count) for (option, count) in options.items() if count >= threshold
    ]
    if not options:
        return None

    # Find best option
    (level, common_parent), _ = min(
        options,
        # Order:
        key=lambda entry: (  # lambda ((level, parent_area), count):
            # - by ascending level (prefer recognizing as a list of countries
            #   over a list of states), then
            entry[0][0],
            # - by descending level of the common parent (prefer a list of
            #   counties in the same state over counties merely in the same
            #   country over counties in different countries), then
            -(entry[0][1].type.value if entry[0][1] is not None else -1),
            # - by descending count
            -entry[1],
        ),
    )
    if common_parent is None:
        common_admin = None
    else:
        common_admin = common_parent.levels[common_parent.type.value]

    # Build the result
    result = []
    for candidates in admin_areas:
        for area in candidates:
            if area.type.value == level and (
                common_parent is None
                or area.levels[common_parent.type.value] == common_admin
            ):
                result.append(area)
                break

    return level, result


GEOHASH_CHARS = "0123456789bcdefghjkmnpqrstuvwxyz"
assert len(GEOHASH_CHARS) == 32
GEOHASH_CHAR_VALUES = {c: i for i, c in enumerate(GEOHASH_CHARS)}


def bits_to_chars(bits, base_bits):
    result = []
    i = 0
    while i + base_bits <= len(bits):
        char = 0
        for j in range(base_bits):
            char = (char << 1) | bits[i + j]
        result.append(GEOHASH_CHARS[char])
        i += base_bits

    return "".join(result)


def chars_to_bits(chars, base_bits):
    for char in chars:
        char = GEOHASH_CHAR_VALUES[char]
        for i in reversed(range(base_bits)):
            yield (char >> i) & 1


def location_to_bits(point, base, precision):
    latitude, longitude = point

    # Compute the number of bits we need
    base_bits = base.bit_length() - 1
    if 2**base_bits != base:
        raise ValueError("Base is not a power of 2")
    precision_bits = base_bits * precision

    lat_range = -90.0, 90.0
    long_range = -180.0, 180.0
    bits = []
    while len(bits) < precision_bits:
        mid = (long_range[0] + long_range[1]) / 2.0
        if longitude > mid:
            bits.append(1)
            long_range = mid, long_range[1]
        else:
            bits.append(0)
            long_range = long_range[0], mid

        mid = (lat_range[0] + lat_range[1]) / 2.0
        if latitude > mid:
            bits.append(1)
            lat_range = mid, lat_range[1]
        else:
            bits.append(0)
            lat_range = lat_range[0], mid
    return bits


def hash_location(point, base=32, precision=16):
    """Hash coordinates into short strings usable for prefix search.

    If base=32, this gives Geohash strings (each level splits cells into 32).

    If base=4, this makes a quadtree (each level split cells into 4 quadrants).
    """
    base_bits = base.bit_length() - 1

    # Build the bitstring
    bits = location_to_bits(point, base, precision)

    # Encode the bitstring
    return bits_to_chars(bits, base_bits)


def decode_hash(hash, base=32):
    """Turn a hash back into a rectangle.

    :returns: ``(min_lat, max_lat, min_long, max_long)``
    """
    base_bits = base.bit_length() - 1
    if 2**base_bits != base:
        raise ValueError("Base is not a power of 2")

    lat_range = -90.0, 90.0
    long_range = -180.0, 180.0
    next_long = True
    for bit in chars_to_bits(hash, base_bits):
        if next_long:
            mid = (long_range[0] + long_range[1]) / 2.0
            if bit:
                long_range = mid, long_range[1]
            else:
                long_range = long_range[0], mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2.0
            if bit:
                lat_range = mid, lat_range[1]
            else:
                lat_range = lat_range[0], mid
        next_long = not next_long

    return (
        lat_range[0],
        lat_range[1],
        long_range[0],
        long_range[1],
    )


def bitrange(from_bits, to_bits):
    bits = list(from_bits)
    while bits != to_bits:
        yield bits
        for i in range(len(bits) - 1, -1, -1):
            if bits[i] == 0:
                bits[i] = 1
                break
            else:
                bits[i] = 0
    yield bits


class Geohasher(object):
    def __init__(self, *, number, base=4, precision=16):
        self.number = number
        self.base = base
        self.precision = precision

        self.tree_root = [0, {}]
        self.number_at_level = [0] * (precision)

    def add_points(self, points):
        for point in points:
            geohash = hash_location(point, self.base, self.precision)
            # Add this hash to the tree
            node = self.tree_root
            for level, key in enumerate(geohash):
                node[0] += 1
                try:
                    node = node[1][key]
                except KeyError:
                    new_node = [0, {}]
                    node[1][key] = new_node
                    node = new_node
                    self.number_at_level[level] += 1

                    # If this level has too many nodes, stop building it
                    if self.number_at_level[level] > self.number:
                        self.precision = level
                        break
            node[0] += 1

    def add_aab(self, box):
        base_bits = self.base.bit_length() - 1

        min_long, max_long, min_lat, max_lat = box
        min_bits = location_to_bits(
            (min_lat, min_long),
            self.base,
            self.precision,
        )
        min_long_bits = min_bits[0::2]
        min_lat_bits = min_bits[1::2]
        max_bits = location_to_bits(
            (max_lat, max_long),
            self.base,
            self.precision,
        )
        max_long_bits = max_bits[0::2]
        max_lat_bits = max_bits[1::2]

        self.tree_root[0] += 1
        level = 1
        while level <= self.precision:
            n_long_bits = math.ceil(level * base_bits / 2)
            n_lat_bits = math.floor(level * base_bits / 2)
            for long_bits in bitrange(
                min_long_bits[:n_long_bits],
                max_long_bits[:n_long_bits],
            ):
                for lat_bits in bitrange(
                    min_lat_bits[:n_lat_bits],
                    max_lat_bits[:n_lat_bits],
                ):
                    bits = [0] * (n_long_bits + n_lat_bits)
                    bits[0::2] = long_bits
                    bits[1::2] = lat_bits
                    geohash = bits_to_chars(bits, base_bits)

                    # Add this hash to the tree
                    node = self.tree_root
                    for lvl, key in enumerate(geohash):
                        try:
                            node = node[1][key]
                        except KeyError:
                            new_node = [0, {}]
                            node[1][key] = new_node
                            node = new_node
                            self.number_at_level[lvl] += 1
                    node[0] += 1

                    if self.number_at_level[level - 1] > self.number:
                        self.precision = level - 1
                        break

            level += 1

    def get_hashes(self):
        # Reconstruct the hashes at this level
        hashes = []

        def add_node(prefix, node, level):
            if level == self.precision:
                hashes.append((prefix, node[0]))
                return
            for k, n in node[1].items():
                add_node(prefix + k, n, level + 1)

        add_node("", self.tree_root, 0)
        return hashes

    def get_hashes_json(self):
        hashes = self.get_hashes()
        return [
            {
                "hash": h,
                "number": n,
            }
            for h, n in hashes
        ]

    @property
    def total(self):
        return self.tree_root[0]


def median_smallest_distance(points, tree=None):
    """Median over all points of the distance to their closest neighbor.

    This gives an idea of the "grid size" of a point dataset.
    """
    points = numpy.array(points)
    if tree is None:
        # points = numpy.unique(points, axis=0)  # Too slow
        points = numpy.array(list(set(tuple(p) for p in points)))
        tree = KDTree(points)

    # Get the minimum distances to neighbors for a sample of points
    rnd = numpy.random.RandomState(89)
    sample_size = min(len(points), 100)
    sample_idx = rnd.choice(len(points), sample_size, replace=False)
    sample = points[sample_idx]
    distances, _ = tree.query(sample, k=2, return_distance=True)

    # Return the median of that
    return numpy.median(distances[:, 1])


# ============================================================================
# Geo Classifier
# author: Eden Wu
# ============================================================================


def mean_pool(outputs, attention_mask):
    """Mean pooling over sequence length (matches train_cta_classifier.py)."""
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class CTAClassificationModel(nn.Module):
    """Classification model with mean pooling (matches train_cta_classifier.py)."""

    def __init__(self, num_labels, config=None, use_spatial_head=False):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.use_spatial_head = use_spatial_head
        if use_spatial_head:
            self.spatial_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None, spatial_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs, attention_mask)
        logits = self.classifier(pooled)

        class Output:
            pass

        out = Output()
        out.logits = logits
        out.pooled = pooled
        if self.use_spatial_head and spatial_labels is not None:
            out.spatial_logits = self.spatial_head(pooled)
        return out


class CTAContrastiveModel(nn.Module):
    """Encoder with projection head for contrastive learning (matches train_cta_classifier.py)."""

    def __init__(self, embed_dim=128, num_labels=None, config=None):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )
        self.classifier = nn.Linear(hidden_size, num_labels) if num_labels else None

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs, attention_mask)
        if return_embeddings:
            return F.normalize(self.projection(pooled), dim=1)
        return self.classifier(pooled) if self.classifier else pooled

    def get_embeddings(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask, return_embeddings=True)


def download_geo_model(model_dir: str, files: dict = None) -> None:
    """
    Download GeoClassifier model files from NYU Box.

    Args:
        model_dir: Directory to save the model files
        files: Dict mapping filename to URL (default: GEO_MODEL_FILES)
    """
    if files is None:
        files = GEO_MODEL_FILES

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading GeoClassifier model to {model_path}")

    for filename, url in files.items():
        file_path = model_path / filename
        if file_path.exists():
            logger.info(f"  {filename}: already exists, skipping")
            continue

        logger.info(f"  Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            downloaded = 0
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            if downloaded > 1024 * 1024:
                size_str = f"{downloaded / 1024 / 1024:.1f}MB"
            else:
                size_str = f"{downloaded / 1024:.1f}KB"
            logger.info(f"  {filename}: downloaded ({size_str})")

        except requests.RequestException as e:
            logger.error(f"  Failed to download {filename}: {e}")
            raise

    logger.info("Model download complete")


class GeoClassifier:
    """Unified interface for CTA classification with automatic model download."""

    def __init__(self, model_dir: str | None = None, auto_download: bool = True):
        """
        Initialize the GeoClassifier.

        Args:
            model_dir: Path to the model directory (default: bundled model if present,
                otherwise a user cache directory)
            auto_download: If True, automatically download model if not found
        """
        required_files = list(GEO_MODEL_FILES.keys())
        if model_dir is None:
            # Default to profiler/model relative to this module's location
            profiler_dir = Path(__file__).parent
            package_model_dir = profiler_dir / "model"
            if all((package_model_dir / f).exists() for f in required_files):
                model_dir = str(package_model_dir)
            else:
                cache_root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
                model_dir = str(cache_root / "atlas_profiler" / "model")

        self.model_dir = Path(model_dir)

        # Check if model files exist, download if needed
        missing_files = [f for f in required_files if not (self.model_dir / f).exists()]

        if missing_files:
            if auto_download:
                logger.info(f"Model files missing: {missing_files}. Downloading...")
                download_geo_model(str(self.model_dir))
            else:
                raise FileNotFoundError(
                    f"Model files not found in {self.model_dir}: {missing_files}. "
                    "Set auto_download=True or download from NYU Box."
                )

        # Load label encoder config
        with open(self.model_dir / "label_encoder.json") as f:
            config = json.load(f)

        self.classes = config["classes"]
        self.mode = config.get("mode", "classification")
        self.model_name = config.get("model_name", "BAAI/bge-small-en-v1.5")
        self.embed_dim = config.get("embed_dim", 128)
        self.name_repeat = config.get("name_repeat", 3)  # Name emphasis
        self.special_tokens = config.get("special_tokens", {})  # [COL], [VAL] tokens

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load tokenizer (prefer saved, fallback to model_name)
        if (self.model_dir / "tokenizer_config.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load encoder config from saved directory
        encoder_config = AutoConfig.from_pretrained(str(self.model_dir))

        # Load checkpoint to inspect keys and determine model architecture
        checkpoint_path = self.model_dir / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_keys = set(checkpoint.keys())

        # Detect model type from checkpoint keys (prioritize checkpoint structure over mode)
        has_projection = any("projection" in k for k in checkpoint_keys)
        has_spatial_head = any("spatial_head" in k for k in checkpoint_keys)

        # Determine which model to use based on checkpoint structure
        # If checkpoint has projection head, use contrastive model
        # Otherwise, use classification model (works for classification, fine_tune, combined modes)
        if has_projection:
            # Contrastive model (has projection head)
            self.model = CTAContrastiveModel(
                embed_dim=self.embed_dim,
                num_labels=len(self.classes),
                config=encoder_config,
            )
            logger.info(
                "Using CTAContrastiveModel (detected projection head in checkpoint, mode=%s)",
                self.mode,
            )
        else:
            # Classification model (no projection head)
            # This works for classification, fine_tune, and combined modes
            # since combined/fine_tune models are saved as classification models
            self.model = CTAClassificationModel(
                num_labels=len(self.classes),
                config=encoder_config,
                use_spatial_head=has_spatial_head,
            )
            logger.info(
                "Using CTAClassificationModel (no projection head, mode=%s, has_spatial_head=%s)",
                self.mode,
                has_spatial_head,
            )

        # Load saved weights (use strict=False to handle missing/extra keys gracefully)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint, strict=False
        )
        if missing_keys:
            logger.warning(f"Missing keys when loading model: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

        self.model.to(self.device)
        self.model.eval()

    def _format_input(self, text: str) -> str:
        """Format input with special tokens: [COL] name [VAL] val1 [VAL] val2..."""
        # Parse "name: val1, val2, val3" format
        if ": " not in text:
            return text

        name, values = text.split(": ", 1)
        col_tok = self.special_tokens.get("col_token", "")
        val_tok = self.special_tokens.get("val_token", "")

        # Format column name with repetition
        if col_tok and name:
            if self.name_repeat > 1:
                col_parts = " ".join([f"{col_tok} {name}"] * self.name_repeat)
            else:
                col_parts = f"{col_tok} {name}"
        else:
            col_parts = name

        # Format values with [VAL] token
        if val_tok:
            val_list = [v.strip() for v in str(values).split(",")]
            val_parts = " ".join([f"{val_tok} {v}" for v in val_list[:10]])
        else:
            val_parts = values

        return f"{col_parts} {val_parts}".strip()

    def predict_batch(
        self,
        texts: list[str],
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Batch prediction for multiple columns in ONE forward pass.

        Args:
            texts: List of texts in format "column_name: val1, val2, val3"
            threshold: Confidence threshold; predictions below this are flagged

        Returns:
            List of dicts with 'label' and 'confidence' for each input
        """
        if not texts:
            return []

        # Format all inputs
        formatted_texts = [self._format_input(t) for t in texts]

        # Batch tokenize (much faster than tokenizing one by one)
        encodings = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,  # Pad to longest in batch (more efficient)
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # SINGLE forward pass for ALL columns!
            # Handle both model types
            if isinstance(self.model, CTAClassificationModel):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:  # CTAContrastiveModel
                logits = self.model(input_ids, attention_mask)
                # If contrastive model doesn't have classifier, this won't work
                # But based on training code, it should have classifier for inference

            probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]

        # Process results for each column
        results = []
        for i in range(len(texts)):
            top_prob, top_idx = probs[i].max(dim=0)
            label = self.classes[top_idx.item()]
            confidence = top_prob.item()

            if threshold is not None and confidence < threshold:
                results.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "filtered": True,
                    }
                )
            else:
                results.append(
                    {
                        "label": label,
                        "confidence": confidence,
                    }
                )

        return results


class HybridGeoClassifier:
    """
    Hybrid classifier: ML prediction first, then rule-based validation.

    For sensitive spatial types (BBL, BIN, lat/lon, zip, geometry), validates
    the ML prediction with pattern/range checks. If validation fails, marks the
    prediction as rejected so heuristics can take over.
    """

    # Types requiring rule-based validation after ML prediction
    VALIDATE_TYPES = {
        "bbl",
        "bin",
        "latitude",
        "longitude",
        "x_coord",
        "y_coord",
        "zip5",  # 5-digit ZIP code
        "zip9",  # 9-digit ZIP+4 code
        "zip_code",  # Generic ZIP code label
        "state_code",  # State codes (e.g., "NY", "CA")
        "borough",  # Named boroughs (e.g., "Brooklyn", "Queens")
        "borough_code",  # Borough codes (numeric/alphanumeric)
        "point",
        "polygon",
        "multi-polygon",
        "line",
        "multi-line",
    }

    # Pre-compiled regex patterns for zip codes (speedup)
    _ZIP_PATTERNS = [
        re.compile(r"^\d{5}$"),  # US ZIP
        re.compile(r"^\d{5}-\d{4}$"),  # US ZIP+4
        re.compile(r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$"),  # Canadian
        re.compile(r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$"),  # UK
        re.compile(r"^\d{3}-\d{4}$"),  # Japan
    ]

    # WKT geometry prefixes (tuple for faster startswith)
    _WKT_PREFIXES = (
        "POINT",
        "MULTIPOINT",
        "LINESTRING",
        "MULTILINESTRING",
        "POLYGON",
        "MULTIPOLYGON",
    )

    def __init__(self, ml_classifier):
        """
        Initialize hybrid classifier.

        Args:
            ml_classifier: GeoClassifier instance for ML predictions
        """
        self.ml = ml_classifier

    def _check_bbl_pattern(self, values):
        """Check if values match NYC BBL pattern (10 digits, starts with 1-5)."""
        valid_count = 0
        for v in values:
            if v is None:
                continue
            try:
                s = str(int(float(v)))
                if len(s) != 10 or s[0] not in "12345":
                    return False  # Early fail
                valid_count += 1
            except (ValueError, TypeError):
                return False
        return valid_count > 0

    def _check_bin_pattern(self, values):
        """Check if values match NYC BIN pattern (7 digits, starts with 1-5)."""
        valid_count = 0
        for v in values:
            if v is None:
                continue
            try:
                s = str(int(float(v)))
                if len(s) != 7 or s[0] not in "12345":
                    return False  # Early fail
                valid_count += 1
            except (ValueError, TypeError):
                return False
        return valid_count > 0

    def _check_latitude_range(self, values):
        """Check if numeric values are in valid US latitude range (18.0-71.5, positive)."""
        valid_count = 0
        for v in values:
            if v is None:
                continue
            try:
                n = float(v)
                # US latitudes: contiguous US (24.5-49.5) or extended for AK/HI (18.0-71.5)
                # Should be POSITIVE (no negative values)
                if not ((18.0 <= n <= 71.5) and n >= 0):
                    return False  # Early fail
                valid_count += 1
            except (ValueError, TypeError):
                continue
        return valid_count > 0

    def _check_longitude_range(self, values):
        """Check if numeric values are in valid US longitude range (-125.0 to -66.0, negative)."""
        valid_count = 0
        for v in values:
            if v is None:
                continue
            try:
                n = float(v)
                # US longitudes: always NEGATIVE, range -125.0 to -66.0
                if not (-125.0 <= n <= -66.0 and n < 0):
                    return False  # Early fail
                valid_count += 1
            except (ValueError, TypeError):
                continue
        return valid_count > 0

    def _check_projected_coord(self, values):
        """Check if values look like projected coordinates (specific magnitude ranges)."""
        valid_count = 0
        for v in values:
            if v is None:
                continue
            try:
                n = float(v)
                # X_coord: magnitude ~1,000,000-15,000,000
                # Y_coord: magnitude ~2,000,000-7,000,000
                # Accept either range (X or Y)
                if not (
                    (1_000_000 <= abs(n) <= 15_000_000)
                    or (2_000_000 <= abs(n) <= 7_000_000)
                ):
                    return False  # Early fail - not in expected projected coord range
                valid_count += 1
            except (ValueError, TypeError):
                continue
        return valid_count > 0

    def _check_zip_pattern(self, values):
        """Check if values match common postal code patterns (uses pre-compiled regex)."""
        for v in values:
            if v is None:
                continue
            s = str(v).strip().upper()
            if any(p.match(s) for p in self._ZIP_PATTERNS):
                return True  # Early return on first match
        return False

    def _check_state_code_pattern(self, values):
        """Check if values match US state code pattern (2 uppercase letters)."""
        us_states = {
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
            "DC",
            "AS",
            "GU",
            "MP",
            "PR",
            "VI",  # Territories
        }
        for v in values:
            if v is None:
                continue
            s = str(v).strip().upper()
            if s in us_states:
                return True
        return False

    def _check_wkt_geometry(self, values):
        """Check if values are WKT geometry strings."""
        for v in values:
            if not isinstance(v, str):
                continue
            v_upper = v.strip().upper()
            if v_upper.startswith(self._WKT_PREFIXES):  # Tuple for faster check
                return True  # Early return on first match
        return False

    def _validate_prediction(self, label, column_name, values):
        """Validate ML prediction using rule-based checks."""
        name_lower = column_name.lower()

        if label == "bbl":
            name_ok = "bbl" in name_lower
            pattern_ok = self._check_bbl_pattern(values)
            return name_ok or pattern_ok

        if label == "bin":
            name_ok = "bin" in name_lower and "binary" not in name_lower
            pattern_ok = self._check_bin_pattern(values)
            return name_ok or pattern_ok

        if label == "latitude":
            name_ok = any(k in name_lower for k in ("lat", "latitude"))
            range_ok = self._check_latitude_range(values)
            return name_ok or range_ok

        if label == "longitude":
            name_ok = any(k in name_lower for k in ("lon", "lng", "longitude"))
            range_ok = self._check_longitude_range(values)
            return name_ok or range_ok

        if label == "x_coord":
            name_ok = "x_coord" in name_lower or "xcoord" in name_lower
            range_ok = self._check_projected_coord(values)
            return name_ok or range_ok

        if label == "y_coord":
            name_ok = "y_coord" in name_lower or "ycoord" in name_lower
            range_ok = self._check_projected_coord(values)
            return name_ok or range_ok

        if label == "zip_code" or label == "zip5" or label == "zip9":
            name_ok = any(k in name_lower for k in ("zip", "postal"))
            pattern_ok = self._check_zip_pattern(values)
            return name_ok or pattern_ok

        if label == "state_code":
            name_ok = any(k in name_lower for k in ("state", "st", "state_code"))
            pattern_ok = self._check_state_code_pattern(values)
            return name_ok or pattern_ok

        if label in ("point", "polygon", "multi-polygon", "line", "multi-line"):
            return self._check_wkt_geometry(values)

        return True  # Unknown type, assume valid

    def predict_batch(
        self,
        columns_data: list[tuple[str, list]],
        threshold: float | None = 0.85,
    ) -> list[dict]:
        """
        Batch prediction with rule validation for multiple columns.

        Uses single forward pass for ML prediction, then validates each result.

        Args:
            columns_data: List of (column_name, sample_values) tuples
            threshold: Confidence threshold

        Returns:
            List of prediction dicts for each column
        """
        if not columns_data:
            return []

        # Format all inputs for batch ML prediction
        texts = []
        for column_name, values in columns_data:
            values_str = ", ".join(str(v) for v in values[:5])
            texts.append(f"{column_name}: {values_str}")

        # BATCH ML prediction - single forward pass!
        ml_results = self.ml.predict_batch(texts, threshold=threshold)

        # Validate each result with rules
        final_results = []
        for (column_name, values), ml_pred in zip(columns_data, ml_results):
            label = ml_pred.get("label")
            confidence = ml_pred.get("confidence", 0.0)

            # If below threshold, keep prediction but mark as filtered
            if ml_pred.get("filtered"):
                final_results.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "source": "ml_low_conf",
                        "filtered": True,
                    }
                )
                continue

            # Validate spatial types with rules
            if label in self.VALIDATE_TYPES:
                is_valid = self._validate_prediction(label, column_name, values)

                if is_valid:
                    final_results.append(
                        {
                            "label": label,
                            "confidence": confidence,
                            "source": "ml+validated",
                            "validated": True,
                        }
                    )
                else:
                    logger.debug(
                        "HybridGeo batch: %s -> %s rejected", column_name, label
                    )
                    final_results.append(
                        {
                            "label": label,
                            "confidence": confidence,
                            "source": f"ml:{label}->rejected",
                            "validated": False,
                            "rejected": True,
                        }
                    )
            else:
                # No validation needed
                final_results.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "source": "ml",
                    }
                )

        return final_results
