import os
import random
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pandas as pd
from faker import Faker
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders

try:
    from training.atlas_types import TWO_LEVEL_ONTOLOGY
except ImportError:
    from atlas_types import TWO_LEVEL_ONTOLOGY

fake = Faker()


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


def parse_response(response: str) -> list[str]:
    """Parse LLM response into names."""
    names: list[str] = []
    try:
        content = response.content if hasattr(response, "content") else str(response)
        tokens = [t.strip() for t in re.split(r"[,;\n|\u2022]+", content.strip())]
        if len(tokens) == 1 and re.search(r"\d+[\.\)]\s+", content):
            tokens = [t.strip() for t in re.split(r"\s*\d+[\.\)]\s+", content.strip())]
        names = [
            re.sub(r"^[\s\-\*\u2022\d\.\)\:]+", "", token).strip()
            for token in tokens
            if token.strip()
        ]
    except Exception as e:
        print(f"Failed to parse response: {e}")
    return names


def _choice(options: list[str]) -> str:
    return random.choice(options)


def _float_range(low: float, high: float, decimals: int = 1) -> str:
    return f"{random.uniform(low, high):.{decimals}f}"


def _int_range(low: int, high: int) -> str:
    return str(random.randint(low, high))


def _with_unit(value: str, unit: str) -> str:
    return f"{value} {unit}"


def _format_number(
    value: float,
    decimals: int,
    thousands: bool = False,
    decimal_sep: str = ".",
    thousands_sep: str = ",",
) -> str:
    if decimals <= 0:
        formatted = f"{value:,.0f}" if thousands else f"{value:.0f}"
    else:
        formatted = f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"

    if thousands and thousands_sep != ",":
        formatted = formatted.replace(",", "TMP")
        formatted = formatted.replace(".", decimal_sep)
        formatted = formatted.replace("TMP", thousands_sep)
    elif decimal_sep != ".":
        formatted = formatted.replace(".", decimal_sep)

    return formatted


def _measure_value(options: list[tuple[str, float, float, int]]) -> str:
    unit, low, high, decimals = random.choice(options)
    value = random.uniform(low, high)
    value_str = _format_number(value, decimals=decimals, thousands=False)
    return f"{value_str} {unit}"


def _money_value() -> str:
    currencies = [
        {
            "code": "USD",
            "symbol": "$",
            "decimal_sep": ".",
            "thousands_sep": ",",
            "min": 0.5,
            "max": 250000,
        },
        {
            "code": "EUR",
            "symbol": "€",
            "decimal_sep": ",",
            "thousands_sep": ".",
            "min": 0.5,
            "max": 200000,
        },
        {
            "code": "GBP",
            "symbol": "£",
            "decimal_sep": ".",
            "thousands_sep": ",",
            "min": 0.5,
            "max": 200000,
        },
    ]
    currency = random.choice(currencies)
    amount = random.uniform(currency["min"], currency["max"])
    decimals = random.choice([0, 2])
    thousands = random.random() < 0.7
    amount_str = _format_number(
        amount,
        decimals=decimals,
        thousands=thousands,
        decimal_sep=currency["decimal_sep"],
        thousands_sep=currency["thousands_sep"],
    )
    style = random.choice(["symbol", "code_prefix", "code_suffix"])
    if style == "symbol":
        return f"{currency['symbol']}{amount_str}"
    if style == "code_prefix":
        return f"{currency['code']} {amount_str}"
    return f"{amount_str} {currency['code']}"


def _height_value() -> str:
    if random.random() < 0.5:
        return _with_unit(_int_range(140, 210), "cm")
    feet = random.randint(4, 7)
    inches = random.randint(0, 11)
    return f"{feet} ft {inches} in"


def _rating_value() -> str:
    scale = random.choice([5, 10])
    decimals = random.choice([0, 1])
    value = _format_number(random.uniform(1, scale), decimals=decimals, thousands=False)
    style = random.choice(["plain", "slash", "stars"])
    if style == "plain":
        return value
    if style == "slash":
        return f"{value}/{scale}"
    return f"{value} stars"


def _score_value() -> str:
    scale = random.choice([100, 1000])
    value = random.randint(0, scale)
    style = random.choice(["plain", "slash"])
    if style == "plain":
        return str(value)
    return f"{value}/{scale}"


def _version_value() -> str:
    major = random.randint(0, 10)
    minor = random.randint(0, 20)
    patch = random.randint(0, 50)
    if random.random() < 0.3:
        return f"{major}.{minor}"
    return f"{major}.{minor}.{patch}"


def _normalize_value(value: object) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(x, int) for x in value
    ):
        return f"rgb({value[0]}, {value[1]}, {value[2]})"
    return str(value)


@dataclass(frozen=True)
class TypeSpec:
    kind: str
    description: str
    faker_methods: tuple[str, ...] = ()
    generators: tuple[Callable[[], str], ...] = ()


AREA_OPTIONS = [
    ("sq ft", 50, 10000, 0),
    ("sq m", 5, 1000, 1),
    ("acre", 0.01, 50, 2),
    ("hectare", 0.01, 20, 2),
]
DISTANCE_OPTIONS = [
    ("mi", 0.1, 3000, 1),
    ("km", 0.1, 5000, 1),
    ("m", 1, 100000, 0),
    ("ft", 1, 100000, 0),
]
DURATION_OPTIONS = [
    ("ms", 1, 5000, 0),
    ("s", 1, 86400, 0),
    ("min", 0.1, 600, 1),
    ("hr", 0.1, 72, 1),
    ("day", 1, 365, 0),
]
ENERGY_OPTIONS = [
    ("kWh", 0.1, 5000, 2),
    ("Wh", 1, 100000, 0),
    ("kJ", 10, 50000, 0),
    ("MJ", 0.1, 500, 2),
    ("kcal", 10, 4000, 0),
]
PRESSURE_OPTIONS = [
    ("psi", 5, 150, 1),
    ("kPa", 30, 110, 1),
    ("hPa", 900, 1100, 0),
    ("bar", 0.8, 1.2, 2),
    ("inHg", 28, 31, 2),
]
SPEED_OPTIONS = [
    ("mph", 0, 120, 1),
    ("km/h", 0, 200, 1),
    ("m/s", 0, 60, 1),
]
TEMPERATURE_OPTIONS = [
    ("C", -20, 45, 1),
    ("F", -4, 115, 1),
    ("K", 250, 330, 1),
]
VOLUME_OPTIONS = [
    ("L", 0.1, 500, 1),
    ("mL", 50, 200000, 0),
    ("gal", 0.1, 200, 1),
    ("fl oz", 1, 256, 0),
    ("m3", 0.1, 50, 2),
]
WEIGHT_OPTIONS = [
    ("kg", 0.5, 200, 1),
    ("lb", 1, 400, 0),
    ("g", 1, 50000, 0),
    ("oz", 1, 200, 0),
    ("t", 0.1, 50, 2),
]

GRADE_LETTERS = ["A", "B", "C", "D", "F", "A+", "B+", "C+", "D+", "A-", "B-", "C-", "D-", "P"]
SIZE_LABELS = ["XS", "S", "M", "L", "XL", "XXL"]
FLAG_LABELS = ["Yes", "No", "Y", "N"]


TYPE_SPECS: dict[str, TypeSpec] = {
    "address": TypeSpec("faker", "street addresses", faker_methods=("address",)),
    "city": TypeSpec("faker", "city names", faker_methods=("city",)),
    "state": TypeSpec("faker", "US state names", faker_methods=("state",)),
    "state_code": TypeSpec(
        "faker", "US state abbreviations", faker_methods=("state_abbr",)
    ),
    "country": TypeSpec("faker", "country names", faker_methods=("country",)),
    "country_code": TypeSpec(
        "faker", "country codes", faker_methods=("country_code",)
    ),
    "zip5": TypeSpec(
        "faker",
        "5-digit ZIP codes",
        faker_methods=("zipcode", "postalcode",),
    ),
    "zip9": TypeSpec(
        "faker",
        "ZIP+4 codes",
        faker_methods=("zipcode_plus4", "postalcode_plus4",),
    ),
    "borough": TypeSpec("curated+synthetic", "US borough names"),
    "borough_code": TypeSpec("curated+synthetic", "borough codes"),
    "bin": TypeSpec("curated+synthetic", "building identification numbers"),
    "bbl": TypeSpec("curated+synthetic", "borough-block-lot numbers"),
    "point": TypeSpec("curated+synthetic", "WKT POINT geometry"),
    "line": TypeSpec("curated+synthetic", "WKT LINESTRING geometry"),
    "polygon": TypeSpec("curated+synthetic", "WKT POLYGON geometry"),
    "multi-line": TypeSpec("curated+synthetic", "WKT MULTILINESTRING geometry"),
    "multi-polygon": TypeSpec("curated+synthetic", "WKT MULTIPOLYGON geometry"),
    "latitude": TypeSpec("faker", "latitude coordinates", faker_methods=("latitude",)),
    "longitude": TypeSpec(
        "faker", "longitude coordinates", faker_methods=("longitude",)
    ),
    "x_coord": TypeSpec("curated+synthetic", "projected x coordinates"),
    "y_coord": TypeSpec("curated+synthetic", "projected y coordinates"),
    "data_time": TypeSpec(
        "faker", "timestamps", faker_methods=("date_time",)
    ),
    "iso8601": TypeSpec("faker", "ISO-8601 timestamps", faker_methods=("iso8601",)),
    "unix_time": TypeSpec("faker", "unix timestamps", faker_methods=("unix_time",)),
    "year": TypeSpec("faker", "year values", faker_methods=("year",)),
    "month_name": TypeSpec("faker", "month names", faker_methods=("month_name",)),
    "month_of_year": TypeSpec("faker", "month numbers", faker_methods=("month",)),
    "day_of_month": TypeSpec(
        "faker", "day-of-month values", faker_methods=("day_of_month",)
    ),
    "day_of_week": TypeSpec(
        "faker", "weekday names", faker_methods=("day_of_week",)
    ),
    "first_name": TypeSpec("faker", "first names", faker_methods=("first_name",)),
    "last_name": TypeSpec("faker", "last names", faker_methods=("last_name",)),
    "name": TypeSpec("faker", "full names", faker_methods=("name",)),
    "prefix": TypeSpec("faker", "name prefixes", faker_methods=("prefix",)),
    "age": TypeSpec(
        "custom",
        "age values",
        generators=(lambda: _int_range(0, 100), lambda: _int_range(18, 90)),
    ),
    "email": TypeSpec("faker", "email addresses", faker_methods=("email",)),
    "phone_number": TypeSpec(
        "faker", "phone numbers", faker_methods=("phone_number",)
    ),
    "ssn": TypeSpec("faker", "social security numbers", faker_methods=("ssn",)),
    "company": TypeSpec("faker", "company names", faker_methods=("company",)),
    "job": TypeSpec("faker", "job titles", faker_methods=("job",)),
    "money": TypeSpec("custom", "monetary amounts", generators=(_money_value,)),
    "currency_code": TypeSpec(
        "faker", "currency codes", faker_methods=("currency_code",)
    ),
    "credit_card_number": TypeSpec(
        "faker", "credit card numbers", faker_methods=("credit_card_number",)
    ),
    "credit_card_security_code": TypeSpec(
        "faker",
        "credit card security codes",
        faker_methods=("credit_card_security_code",),
    ),
    "credit_card_expire": TypeSpec(
        "faker", "credit card expirations", faker_methods=("credit_card_expire",)
    ),
    "credit_card_provider": TypeSpec(
        "faker", "credit card providers", faker_methods=("credit_card_provider",)
    ),
    "credit_card_full": TypeSpec(
        "faker", "full credit card info", faker_methods=("credit_card_full",)
    ),
    "url": TypeSpec("faker", "URLs", faker_methods=("url",)),
    "ipv4": TypeSpec("faker", "IPv4 addresses", faker_methods=("ipv4",)),
    "ipv6": TypeSpec("faker", "IPv6 addresses", faker_methods=("ipv6",)),
    "mac_address": TypeSpec(
        "faker", "MAC addresses", faker_methods=("mac_address",)
    ),
    "platform": TypeSpec(
        "faker",
        "platform tokens",
        faker_methods=(
            "mac_platform_token",
            "windows_platform_token",
            "linux_platform_token",
        ),
    ),
    "file_name": TypeSpec("faker", "file names", faker_methods=("file_name",)),
    "file_extension": TypeSpec(
        "faker", "file extensions", faker_methods=("file_extension",)
    ),
    "uuid": TypeSpec("custom", "UUID values", generators=(lambda: str(uuid.uuid4()),)),
    "ean8": TypeSpec("faker", "EAN-8 codes", faker_methods=("ean8",)),
    "ean13": TypeSpec("faker", "EAN-13 codes", faker_methods=("ean13",)),
    "color_name": TypeSpec(
        "faker", "color names", faker_methods=("color_name",)
    ),
    "hex_color": TypeSpec("faker", "hex color codes", faker_methods=("hex_color",)),
    "rgb_color": TypeSpec(
        "faker",
        "RGB color values",
        faker_methods=("rgb_css_color", "rgb_color_list", "rgb_color"),
    ),
    "area": TypeSpec(
        "custom", "area measurements", generators=(lambda: _measure_value(AREA_OPTIONS),)
    ),
    "distance": TypeSpec(
        "custom",
        "distance measurements",
        generators=(lambda: _measure_value(DISTANCE_OPTIONS),),
    ),
    "duration": TypeSpec(
        "custom",
        "duration measurements",
        generators=(lambda: _measure_value(DURATION_OPTIONS),),
    ),
    "energy": TypeSpec(
        "custom", "energy measurements", generators=(lambda: _measure_value(ENERGY_OPTIONS),)
    ),
    "height": TypeSpec("custom", "height measurements", generators=(_height_value,)),
    "pressure": TypeSpec(
        "custom",
        "pressure measurements",
        generators=(lambda: _measure_value(PRESSURE_OPTIONS),),
    ),
    "speed": TypeSpec(
        "custom", "speed measurements", generators=(lambda: _measure_value(SPEED_OPTIONS),)
    ),
    "temperature": TypeSpec(
        "custom",
        "temperature measurements",
        generators=(lambda: _measure_value(TEMPERATURE_OPTIONS),),
    ),
    "volume": TypeSpec(
        "custom", "volume measurements", generators=(lambda: _measure_value(VOLUME_OPTIONS),)
    ),
    "weight": TypeSpec(
        "custom", "weight measurements", generators=(lambda: _measure_value(WEIGHT_OPTIONS),)
    ),
    "rating": TypeSpec("custom", "rating values", generators=(_rating_value,)),
    "score": TypeSpec("custom", "score values", generators=(_score_value,)),
    "grade": TypeSpec(
        "custom", "grade letters", generators=(lambda: _choice(GRADE_LETTERS),)
    ),
    "percent": TypeSpec(
        "custom",
        "percent values",
        generators=(lambda: f"{_float_range(0, 100, 1)}%", lambda: f"{_float_range(0, 100, 2)}%"),
    ),
    "boolean": TypeSpec(
        "faker", "boolean values", faker_methods=("boolean", "pybool")
    ),
    "flag": TypeSpec(
        "custom", "yes/no flags", generators=(lambda: _choice(FLAG_LABELS),)
    ),
    "size": TypeSpec(
        "custom", "size labels", generators=(lambda: _choice(SIZE_LABELS),)
    ),
    "version": TypeSpec("custom", "version strings", generators=(_version_value,)),
}


ALLOWED_KINDS = {"faker", "custom", "curated+synthetic"}


def _build_l2_to_l1() -> dict[str, str]:
    l2_to_l1: dict[str, str] = {}
    for l1_label, l2_labels in TWO_LEVEL_ONTOLOGY.items():
        for l2_label in l2_labels:
            if l2_label in l2_to_l1:
                raise ValueError(f"Duplicate l2 label found: {l2_label}")
            l2_to_l1[l2_label] = l1_label
    return l2_to_l1


def _validate_specs(l2_to_l1: dict[str, str]) -> None:
    missing = sorted(set(l2_to_l1) - set(TYPE_SPECS))
    extra = sorted(set(TYPE_SPECS) - set(l2_to_l1))
    if missing:
        raise ValueError(f"Missing TYPE_SPECS for l2 labels: {missing}")
    if extra:
        raise ValueError(f"TYPE_SPECS includes unknown l2 labels: {extra}")

    for label, spec in TYPE_SPECS.items():
        if spec.kind not in ALLOWED_KINDS:
            raise ValueError(f"Invalid kind for {label}: {spec.kind}")
        if spec.kind == "faker" and not spec.faker_methods:
            raise ValueError(f"Faker spec missing methods for {label}")
        if spec.kind == "custom" and not spec.generators:
            raise ValueError(f"Custom spec missing generators for {label}")


def _resolve_faker_methods(methods: tuple[str, ...]) -> list[str]:
    available = [method for method in methods if hasattr(fake, method)]
    if not available:
        raise ValueError(f"None of the faker methods exist: {methods}")
    return available


def _build_value_generator(spec: TypeSpec) -> Callable[[], str]:
    if spec.kind == "faker":
        methods = _resolve_faker_methods(spec.faker_methods)
        method = random.choice(methods)

        def _gen() -> str:
            return _normalize_value(getattr(fake, method)())

        return _gen

    if spec.kind == "custom":
        generator = random.choice(spec.generators)
        return generator

    raise ValueError("Curated+synthetic types should be skipped")


def _needs_header(path: str, columns: list[str]) -> bool:
    if not os.path.exists(path):
        return True
    with open(path, "r", newline="") as handle:
        existing = [col.strip() for col in handle.readline().strip().split(",")]
    if existing == columns:
        return False
    raise ValueError(
        f"{path} has header {existing}; expected {columns}. "
        "Remove or rename the file to regenerate."
    )


def iter_synthetic_name_batches(
    num_names: int,
    class_name: str,
    description: str,
    batch_size: int = 10,
    l1_label: str | None = None,
) -> Iterator[list[str]]:
    """Yield synthetic names per LLM batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    llm = get_llm()
    max_attempts = max(1, ((num_names + batch_size - 1) // batch_size) * 3)
    attempts = 0
    total_names = 0
    class_display = f"{class_name} (l1: {l1_label})" if l1_label else class_name

    while total_names < num_names and attempts < max_attempts:
        remaining = num_names - total_names
        request_size = min(batch_size, remaining)
        prompt = f"""
        Generate a list of {request_size} unique synthetic attribute names (use as the column names) for the following class:

        Class: {class_display}
        Description: {description}

        Guidelines (keep it flexible for training diversity):
        - Names should clearly relate to the class/description and make semantic sense.
        - Small stylistic variations are fine (capitalization, camelCase vs snake_case, abbreviations, suffixes, light noise like hashes), but do not drift to unrelated concepts.
        - Prefer domain-relevant tokens and common dataset naming patterns.
        - For barcode-related classes (e.g., ean8/ean13), it is helpful (not required) to include signals such as: ean, ean8, ean13, barcode, gtin, upc, code, scan, symbology, check_digit.

        Return ONLY the names as a comma-separated string (e.g. name1, name2, name3).
        """
        response = llm.invoke(prompt)
        batch_names = parse_response(response)
        cleaned_batch = [name.strip() for name in batch_names if name.strip()]
        if cleaned_batch:
            if len(cleaned_batch) > remaining:
                cleaned_batch = cleaned_batch[:remaining]
            total_names += len(cleaned_batch)
            yield cleaned_batch
        attempts += 1

    if total_names < num_names:
        request_size = min(batch_size, num_names - total_names)
        prompt = f"""
        Generate a list of {request_size} unique synthetic attribute names (use as the column names) for the following class:

        Class: {class_display}
        Description: {description}

        Guidelines (keep it flexible for training diversity):
        - Names should clearly relate to the class/description and make semantic sense.
        - Small stylistic variations are fine (capitalization, camelCase vs snake_case, abbreviations, suffixes, light noise like hashes), but do not drift to unrelated concepts.
        - Prefer domain-relevant tokens and common dataset naming patterns.
        - For barcode-related classes (e.g., ean8/ean13), it is helpful (not required) to include signals such as: ean, ean8, ean13, barcode, gtin, upc, code, scan, symbology, check_digit.

        Return ONLY the names as a comma-separated string (e.g. name1, name2, name3).
        """
        response = llm.invoke(prompt)
        fallback_names = [name.strip() for name in parse_response(response) if name.strip()]
        if fallback_names:
            needed = num_names - total_names
            fallback_slice = fallback_names[:needed]
            if len(fallback_slice) < needed:
                raise ValueError(f"Failed to generate enough names for class {class_name}")
            yield fallback_slice
        else:
            raise ValueError(f"Failed to generate enough names for class {class_name}")


def generate_synthetic_names(
    num_names: int,
    class_name: str,
    description: str,
    batch_size: int = 10,
    l1_label: str | None = None,
) -> list[str]:
    """Generate synthetic names using LLM in batches.
    If the number of names unmatches the requested number, retry.
    """
    names = []
    for batch in iter_synthetic_name_batches(
        num_names=num_names,
        class_name=class_name,
        description=description,
        batch_size=batch_size,
        l1_label=l1_label,
    ):
        names.extend(batch)
    return names


num_synthetic_per_class = 200
num_of_values_per_class = 3
output_columns = ["name", "values", "l2_label", "l1_label"]

l2_to_l1 = _build_l2_to_l1()
_validate_specs(l2_to_l1)

header_needed = _needs_header("synthetic_df_checkpoint.csv", output_columns)

for l1_label, l2_labels in TWO_LEVEL_ONTOLOGY.items():
    for l2_label in l2_labels:
        spec = TYPE_SPECS[l2_label]
        if spec.kind == "curated+synthetic":
            continue

        print(f"- `{l2_label}` ({l1_label}) - {spec.description} [{spec.kind}]")
        for name_batch in iter_synthetic_name_batches(
            num_names=num_synthetic_per_class,
            class_name=l2_label,
            description=spec.description,
            batch_size=10,
            l1_label=l1_label,
        ):
            rows = []
            for name in name_batch:
                generator = _build_value_generator(spec)
                values = [str(generator()) for _ in range(num_of_values_per_class)]
                rows.append(
                    {
                        "name": name,
                        "values": ", ".join(values),
                        "l2_label": l2_label,
                        "l1_label": l1_label,
                    }
                )

            synthetic_df = pd.DataFrame(rows, columns=output_columns)
            synthetic_df.to_csv(
                "synthetic_df_checkpoint.csv",
                mode="a",
                header=header_needed,
                index=False,
            )
            header_needed = False
