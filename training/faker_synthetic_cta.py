import argparse
import os
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from faker import Faker
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders

try:
    from mimesis import Generic
    from mimesis.locales import Locale
except ImportError:
    Generic = None
    Locale = None

try:
    from training.atlas_types import TWO_LEVEL_ONTOLOGY
except ImportError:
    from atlas_types import TWO_LEVEL_ONTOLOGY

fake = Faker()
MIMESIS_AVAILABLE = Generic is not None and Locale is not None
mimesis = Generic(locale=Locale.EN) if MIMESIS_AVAILABLE else None

DATATYPE_VOCAB = {"string", "integer", "float", "boolean", "date", "datetime"}
LEGACY_OUTPUT_COLUMNS = ["name", "values", "l2_label", "l1_label"]
OUTPUT_COLUMNS = ["name", "values", "l2_label", "l1_label", "datatype"]


def get_llm():
    """Initialize the LLM client."""
    portkey_headers = createHeaders(
        api_key=os.getenv("PORTKEY_API_KEY"),
        metadata={"_user": "yfw215"},
    )
    return ChatOpenAI(
        model="@vertexai/gemini-3-flash-preview",
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


def _random_date_iso(start_year: int = 1990, end_year: int = 2026) -> str:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    span_days = (end - start).days
    chosen = start + timedelta(days=random.randint(0, span_days))
    return chosen.isoformat()


def _random_time_hms() -> str:
    return f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"


def _rand_alnum(length: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(alphabet) for _ in range(length))


def _prefixed_numeric_id(prefix: str, width: int = 6) -> str:
    return f"{prefix}-{random.randint(0, 10**width - 1):0{width}d}"


def _upc12_value() -> str:
    return f"{random.randint(0, 10**12 - 1):012d}"


def _checksum_code_value() -> str:
    base = random.randint(10000, 99999)
    checksum = sum(int(d) for d in str(base)) % 10
    return f"ID-{base}-{checksum}"


def _endpoint_path_value() -> str:
    resources = ["users", "orders", "products", "sessions", "accounts", "events"]
    version = random.choice(["v1", "v2", "v3"])
    if random.random() < 0.7:
        return f"/api/{version}/{random.choice(resources)}/{random.randint(1, 999999)}"
    return f"/{random.choice(resources)}/{random.randint(1, 999999)}"


def _domain_name_value() -> str:
    domains = ["example", "acme", "contoso", "globex", "northwind", "atlasdata"]
    tlds = ["com", "org", "net", "io", "co"]
    return f"{random.choice(domains)}.{random.choice(tlds)}"


def _hostname_value() -> str:
    hosts = ["api", "app", "cdn", "www", "auth", "gateway", "m"]
    return f"{random.choice(hosts)}.{_domain_name_value()}"


def _referrer_url_value() -> str:
    paths = ["/", "/search", "/products", "/collections", "/checkout", "/blog"]
    path = random.choice(paths)
    if path == "/search":
        query = random.choice(["shoes", "laptop", "table", "headphones", "camera"])
        return f"https://{_domain_name_value()}{path}?q={query}"
    return f"https://{_domain_name_value()}{path}"


def _amount_range_value() -> str:
    ranges = ["0-50", "50-100", "100-250", "250-500", "500-1000", "1000+"]
    return random.choice(ranges)


def _age_bucket_value() -> str:
    buckets = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return random.choice(buckets)


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
    mimesis_methods: tuple[str, ...] = ()
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

BUSINESS_STATUS = ["active", "inactive", "pending", "closed", "archived", "on_hold"]
LIFECYCLE_STAGES = ["lead", "prospect", "active", "churned", "reactivated"]
PRIORITY_LEVELS = ["low", "medium", "high", "critical"]
CHANNELS = ["web", "mobile_app", "store", "phone", "marketplace", "partner"]
SOURCE_SYSTEMS = ["salesforce", "netsuite", "sap", "stripe", "shopify", "zendesk"]
PAYMENT_STATUS_VALUES = ["paid", "pending", "failed", "refunded", "partially_refunded"]
ORDER_STATUS_VALUES = ["created", "confirmed", "processing", "shipped", "delivered", "cancelled"]
SHIPMENT_STATUS_VALUES = ["pending", "label_created", "in_transit", "delivered", "failed"]
RETURN_STATUS_VALUES = ["requested", "approved", "rejected", "received", "refunded"]
FULFILLMENT_STATUS_VALUES = ["pending", "picking", "packed", "shipped", "completed"]
HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
HTTP_STATUS_CODES = ["200", "201", "204", "301", "302", "400", "401", "403", "404", "409", "422", "429", "500", "502", "503"]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1",
]
PRODUCT_NAMES = ["Wireless Mouse", "Bluetooth Speaker", "Running Shoes", "Ceramic Mug", "Desk Lamp", "USB-C Cable"]
BRAND_NAMES = ["Acme", "Contoso", "Globex", "Northwind", "Initech", "Umbrella"]
CATEGORY_NAMES = ["electronics", "apparel", "home_goods", "sports", "beauty", "office"]
AVAILABILITY_STATUS_VALUES = ["in_stock", "out_of_stock", "backorder", "preorder", "discontinued"]
GENDERS = ["male", "female", "non_binary", "other", "prefer_not_to_say"]
MARITAL_STATUS_VALUES = ["single", "married", "divorced", "widowed", "separated"]
EDUCATION_LEVEL_VALUES = ["high_school", "associate", "bachelor", "master", "doctorate"]
PAYMENT_METHOD_VALUES = ["card", "ach", "cash", "wire", "paypal", "apple_pay", "google_pay"]


TYPE_SPECS: dict[str, TypeSpec] = {
    "address": TypeSpec(
        "hybrid",
        "street addresses",
        faker_methods=("address",),
        mimesis_methods=("address.address",),
    ),
    "city": TypeSpec(
        "hybrid",
        "city names",
        faker_methods=("city",),
        mimesis_methods=("address.city",),
    ),
    "state": TypeSpec(
        "hybrid",
        "US state names",
        faker_methods=("state",),
        mimesis_methods=("address.state",),
    ),
    "state_code": TypeSpec(
        "faker", "US state abbreviations", faker_methods=("state_abbr",)
    ),
    "country": TypeSpec(
        "hybrid",
        "country names",
        faker_methods=("country",),
        mimesis_methods=("address.country",),
    ),
    "country_code": TypeSpec(
        "hybrid",
        "country codes",
        faker_methods=("country_code",),
        mimesis_methods=("address.country_code",),
    ),
    "zip5": TypeSpec(
        "hybrid",
        "5-digit ZIP codes",
        faker_methods=("zipcode", "postalcode",),
        mimesis_methods=("address.zip_code", "address.postal_code"),
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
    "latitude": TypeSpec(
        "hybrid",
        "latitude coordinates",
        faker_methods=("latitude",),
        mimesis_methods=("address.latitude",),
    ),
    "longitude": TypeSpec(
        "hybrid",
        "longitude coordinates",
        faker_methods=("longitude",),
        mimesis_methods=("address.longitude",),
    ),
    "x_coord": TypeSpec("curated+synthetic", "projected x coordinates"),
    "y_coord": TypeSpec("curated+synthetic", "projected y coordinates"),
    "data_time": TypeSpec(
        "hybrid",
        "timestamps",
        faker_methods=("date_time",),
        mimesis_methods=("datetime.datetime",),
    ),
    "iso8601": TypeSpec("faker", "ISO-8601 timestamps", faker_methods=("iso8601",)),
    "unix_time": TypeSpec(
        "hybrid",
        "unix timestamps",
        faker_methods=("unix_time",),
        mimesis_methods=("datetime.timestamp",),
    ),
    "date": TypeSpec(
        "hybrid",
        "date values",
        faker_methods=("date",),
        mimesis_methods=("datetime.date",),
    ),
    "time": TypeSpec(
        "hybrid",
        "time values",
        faker_methods=("time",),
        mimesis_methods=("datetime.time",),
    ),
    "year": TypeSpec(
        "hybrid",
        "year values",
        faker_methods=("year",),
        mimesis_methods=("datetime.year",),
    ),
    "quarter": TypeSpec(
        "custom", "quarter values", generators=(lambda: _int_range(1, 4),)
    ),
    "week_of_year": TypeSpec(
        "custom", "week-of-year values", generators=(lambda: _int_range(1, 53),)
    ),
    "month_name": TypeSpec(
        "hybrid",
        "month names",
        faker_methods=("month_name",),
        mimesis_methods=("datetime.month",),
    ),
    "month_of_year": TypeSpec("faker", "month numbers", faker_methods=("month",)),
    "day_of_month": TypeSpec(
        "hybrid",
        "day-of-month values",
        faker_methods=("day_of_month",),
        mimesis_methods=("datetime.day_of_month",),
    ),
    "day_of_week": TypeSpec(
        "hybrid",
        "weekday names",
        faker_methods=("day_of_week",),
        mimesis_methods=("datetime.day_of_week",),
    ),
    "first_name": TypeSpec(
        "hybrid",
        "first names",
        faker_methods=("first_name",),
        mimesis_methods=("person.first_name",),
    ),
    "last_name": TypeSpec(
        "hybrid",
        "last names",
        faker_methods=("last_name",),
        mimesis_methods=("person.last_name",),
    ),
    "name": TypeSpec(
        "hybrid",
        "full names",
        faker_methods=("name",),
        mimesis_methods=("person.full_name", "person.name"),
    ),
    "prefix": TypeSpec(
        "hybrid",
        "name prefixes",
        faker_methods=("prefix",),
        mimesis_methods=("person.title",),
    ),
    "age": TypeSpec(
        "custom",
        "age values",
        generators=(lambda: _int_range(0, 100), lambda: _int_range(18, 90)),
    ),
    "email": TypeSpec(
        "hybrid",
        "email addresses",
        faker_methods=("email",),
        mimesis_methods=("person.email",),
    ),
    "phone_number": TypeSpec(
        "hybrid",
        "phone numbers",
        faker_methods=("phone_number",),
        mimesis_methods=("person.telephone",),
    ),
    "ssn": TypeSpec("faker", "social security numbers", faker_methods=("ssn",)),
    "company": TypeSpec(
        "hybrid",
        "company names",
        faker_methods=("company",),
        mimesis_methods=("finance.company",),
    ),
    "job": TypeSpec(
        "hybrid",
        "job titles",
        faker_methods=("job",),
        mimesis_methods=("person.occupation",),
    ),
    "money": TypeSpec("custom", "monetary amounts", generators=(_money_value,)),
    "currency_code": TypeSpec(
        "hybrid",
        "currency codes",
        faker_methods=("currency_code",),
        mimesis_methods=("finance.currency_iso_code",),
    ),
    "payment_method": TypeSpec(
        "custom",
        "payment method values",
        generators=(lambda: _choice(PAYMENT_METHOD_VALUES),),
    ),
    "amount_range": TypeSpec(
        "custom",
        "amount range values",
        generators=(_amount_range_value,),
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
    "url": TypeSpec(
        "hybrid",
        "URLs",
        faker_methods=("url",),
        mimesis_methods=("internet.url",),
    ),
    "ipv4": TypeSpec(
        "hybrid",
        "IPv4 addresses",
        faker_methods=("ipv4",),
        mimesis_methods=("internet.ipv4", "internet.ip_v4"),
    ),
    "ipv6": TypeSpec(
        "hybrid",
        "IPv6 addresses",
        faker_methods=("ipv6",),
        mimesis_methods=("internet.ipv6", "internet.ip_v6"),
    ),
    "mac_address": TypeSpec(
        "hybrid",
        "MAC addresses",
        faker_methods=("mac_address",),
        mimesis_methods=("internet.mac_address",),
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
    "file_name": TypeSpec(
        "hybrid",
        "file names",
        faker_methods=("file_name",),
        mimesis_methods=("file.file_name",),
    ),
    "file_extension": TypeSpec(
        "hybrid",
        "file extensions",
        faker_methods=("file_extension",),
        mimesis_methods=("file.extension",),
    ),
    "uuid": TypeSpec(
        "hybrid",
        "UUID values",
        faker_methods=("uuid4",),
        mimesis_methods=("cryptographic.uuid",),
    ),
    "ean8": TypeSpec("faker", "EAN-8 codes", faker_methods=("ean8",)),
    "ean13": TypeSpec("faker", "EAN-13 codes", faker_methods=("ean13",)),
    "alphanumeric_code": TypeSpec(
        "custom", "alphanumeric codes", generators=(lambda: _rand_alnum(8), lambda: _rand_alnum(10))
    ),
    "checksum_code": TypeSpec(
        "custom", "codes with checksum suffix", generators=(_checksum_code_value,)
    ),
    "primary_key": TypeSpec(
        "custom",
        "primary key identifiers",
        generators=(lambda: _int_range(1, 10000000),),
    ),
    "foreign_key": TypeSpec(
        "custom",
        "foreign key identifiers",
        generators=(lambda: _int_range(1, 10000000),),
    ),
    "user_id": TypeSpec(
        "custom", "user identifiers", generators=(lambda: _prefixed_numeric_id("USR", 6),)
    ),
    "customer_id": TypeSpec(
        "custom",
        "customer identifiers",
        generators=(lambda: _prefixed_numeric_id("CUST", 6),),
    ),
    "account_id": TypeSpec(
        "custom", "account identifiers", generators=(lambda: _prefixed_numeric_id("ACC", 6),)
    ),
    "order_id": TypeSpec(
        "custom", "order identifiers", generators=(lambda: _prefixed_numeric_id("ORD", 6),)
    ),
    "transaction_id": TypeSpec(
        "custom",
        "transaction identifiers",
        generators=(lambda: _prefixed_numeric_id("TXN", 8),),
    ),
    "session_id": TypeSpec(
        "custom",
        "session identifiers",
        generators=(lambda: f"SES-{_rand_alnum(10)}",),
    ),
    "device_id": TypeSpec(
        "custom",
        "device identifiers",
        generators=(lambda: f"DEV-{_rand_alnum(8)}",),
    ),
    "request_id": TypeSpec(
        "custom",
        "request identifiers",
        generators=(lambda: f"REQ-{date.today().strftime('%Y%m%d')}-{random.randint(1, 999999):06d}",),
    ),
    "trace_id": TypeSpec(
        "custom", "trace identifiers", generators=(lambda: f"TRACE-{_rand_alnum(12)}",)
    ),
    "sku": TypeSpec(
        "custom",
        "stock keeping unit identifiers",
        generators=(lambda: f"SKU-{_rand_alnum(4)}-{random.randint(100, 999)}",),
    ),
    "upc12": TypeSpec("custom", "UPC-12 identifiers", generators=(_upc12_value,)),
    "invoice_id": TypeSpec(
        "custom",
        "invoice identifiers",
        generators=(lambda: f"INV-{date.today().year}-{random.randint(1, 999999):06d}",),
    ),
    "status": TypeSpec(
        "custom", "generic status values", generators=(lambda: _choice(BUSINESS_STATUS),)
    ),
    "lifecycle_stage": TypeSpec(
        "custom",
        "lifecycle stage values",
        generators=(lambda: _choice(LIFECYCLE_STAGES),),
    ),
    "priority": TypeSpec(
        "custom", "priority values", generators=(lambda: _choice(PRIORITY_LEVELS),)
    ),
    "channel": TypeSpec(
        "custom", "channel values", generators=(lambda: _choice(CHANNELS),)
    ),
    "source_system": TypeSpec(
        "custom",
        "source system values",
        generators=(lambda: _choice(SOURCE_SYSTEMS),),
    ),
    "payment_status": TypeSpec(
        "custom",
        "payment status values",
        generators=(lambda: _choice(PAYMENT_STATUS_VALUES),),
    ),
    "order_status": TypeSpec(
        "custom",
        "order status values",
        generators=(lambda: _choice(ORDER_STATUS_VALUES),),
    ),
    "shipment_status": TypeSpec(
        "custom",
        "shipment status values",
        generators=(lambda: _choice(SHIPMENT_STATUS_VALUES),),
    ),
    "return_status": TypeSpec(
        "custom",
        "return status values",
        generators=(lambda: _choice(RETURN_STATUS_VALUES),),
    ),
    "fulfillment_status": TypeSpec(
        "custom",
        "fulfillment status values",
        generators=(lambda: _choice(FULFILLMENT_STATUS_VALUES),),
    ),
    "http_method": TypeSpec(
        "hybrid",
        "HTTP method values",
        faker_methods=("http_method",),
        mimesis_methods=("internet.http_method",),
    ),
    "http_status_code": TypeSpec(
        "hybrid",
        "HTTP status code values",
        faker_methods=("http_status_code",),
        mimesis_methods=("internet.http_status_code",),
    ),
    "endpoint_path": TypeSpec(
        "hybrid",
        "HTTP endpoint path values",
        faker_methods=("uri_path",),
        mimesis_methods=("internet.path",),
    ),
    "referrer_url": TypeSpec(
        "hybrid",
        "referrer URL values",
        faker_methods=("url",),
        mimesis_methods=("internet.url",),
    ),
    "user_agent": TypeSpec(
        "hybrid",
        "user-agent values",
        faker_methods=("user_agent",),
        mimesis_methods=("internet.user_agent",),
    ),
    "hostname": TypeSpec(
        "hybrid",
        "hostname values",
        faker_methods=("hostname",),
        mimesis_methods=("internet.hostname",),
    ),
    "domain_name": TypeSpec(
        "hybrid",
        "domain name values",
        faker_methods=("domain_name",),
        mimesis_methods=("internet.hostname",),
    ),
    "latency_ms": TypeSpec(
        "custom", "latency in milliseconds", generators=(lambda: _int_range(1, 5000),)
    ),
    "bytes_transferred": TypeSpec(
        "custom",
        "bytes transferred values",
        generators=(lambda: _int_range(128, 50000000),),
    ),
    "product_name": TypeSpec(
        "custom", "product name values", generators=(lambda: _choice(PRODUCT_NAMES),)
    ),
    "brand_name": TypeSpec(
        "custom", "brand name values", generators=(lambda: _choice(BRAND_NAMES),)
    ),
    "category_name": TypeSpec(
        "custom", "category name values", generators=(lambda: _choice(CATEGORY_NAMES),)
    ),
    "quantity": TypeSpec(
        "custom", "quantity values", generators=(lambda: _int_range(0, 10000),)
    ),
    "unit_price": TypeSpec(
        "custom",
        "unit price values",
        generators=(lambda: _float_range(0.5, 5000, 2),),
    ),
    "discount_percent": TypeSpec(
        "custom",
        "discount percent values",
        generators=(lambda: _float_range(0, 80, 2),),
    ),
    "tax_percent": TypeSpec(
        "custom",
        "tax percent values",
        generators=(lambda: _float_range(0, 20, 3),),
    ),
    "availability_status": TypeSpec(
        "custom",
        "availability status values",
        generators=(lambda: _choice(AVAILABILITY_STATUS_VALUES),),
    ),
    "birth_date": TypeSpec(
        "hybrid",
        "birth date values",
        faker_methods=("date",),
        mimesis_methods=("datetime.date",),
    ),
    "age_bucket": TypeSpec("custom", "age bucket values", generators=(_age_bucket_value,)),
    "gender": TypeSpec("custom", "gender values", generators=(lambda: _choice(GENDERS),)),
    "marital_status": TypeSpec(
        "custom",
        "marital status values",
        generators=(lambda: _choice(MARITAL_STATUS_VALUES),),
    ),
    "education_level": TypeSpec(
        "custom",
        "education level values",
        generators=(lambda: _choice(EDUCATION_LEVEL_VALUES),),
    ),
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
        "hybrid",
        "boolean values",
        faker_methods=("boolean", "pybool"),
        mimesis_methods=("development.boolean",),
    ),
    "flag": TypeSpec(
        "custom", "yes/no flags", generators=(lambda: _choice(FLAG_LABELS),)
    ),
    "size": TypeSpec(
        "custom", "size labels", generators=(lambda: _choice(SIZE_LABELS),)
    ),
    "version": TypeSpec(
        "mimesis",
        "version strings",
        mimesis_methods=("development.version",),
    ),
}


ALLOWED_KINDS = {"faker", "mimesis", "hybrid", "custom", "curated+synthetic"}


def _build_l2_to_l1() -> dict[str, str]:
    l2_to_l1: dict[str, str] = {}
    for l1_label, l2_labels in TWO_LEVEL_ONTOLOGY.items():
        for l2_label in l2_labels:
            if l2_label in l2_to_l1:
                raise ValueError(f"Duplicate l2 label found: {l2_label}")
            l2_to_l1[l2_label] = l1_label
    return l2_to_l1


def _build_l2_to_datatype(l2_to_l1: dict[str, str]) -> dict[str, str]:
    # Default to string unless explicitly typed by policy.
    l2_to_datatype = {l2_label: "string" for l2_label in l2_to_l1}

    for label in ("data_time", "iso8601"):
        l2_to_datatype[label] = "datetime"

    for label in ("date", "birth_date"):
        l2_to_datatype[label] = "date"

    for label in (
        "age",
        "year",
        "quarter",
        "week_of_year",
        "month_of_year",
        "day_of_month",
        "unix_time",
        "ean8",
        "ean13",
        "primary_key",
        "foreign_key",
        "http_status_code",
        "latency_ms",
        "bytes_transferred",
        "quantity",
    ):
        l2_to_datatype[label] = "integer"

    for label in (
        "latitude",
        "longitude",
        "x_coord",
        "y_coord",
        "unit_price",
        "discount_percent",
        "tax_percent",
    ):
        l2_to_datatype[label] = "float"

    for label in ("boolean", "flag"):
        l2_to_datatype[label] = "boolean"

    return l2_to_datatype


def _validate_datatype_mapping(
    l2_to_l1: dict[str, str], l2_to_datatype: dict[str, str]
) -> None:
    missing = sorted(set(l2_to_l1) - set(l2_to_datatype))
    extra = sorted(set(l2_to_datatype) - set(l2_to_l1))
    if missing:
        raise ValueError(f"Missing datatype mappings for l2 labels: {missing}")
    if extra:
        raise ValueError(f"Datatype mapping includes unknown l2 labels: {extra}")

    invalid = sorted({dtype for dtype in l2_to_datatype.values() if dtype not in DATATYPE_VOCAB})
    if invalid:
        raise ValueError(
            f"Invalid datatype values in mapping: {invalid}. "
            f"Allowed values: {sorted(DATATYPE_VOCAB)}"
        )


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
        if spec.kind == "mimesis":
            if not spec.mimesis_methods:
                raise ValueError(f"Mimesis spec missing methods for {label}")
            if not MIMESIS_AVAILABLE:
                raise ImportError(
                    "mimesis is required for 'mimesis' TypeSpec entries. "
                    "Install it with: pip install mimesis"
                )
        if spec.kind == "hybrid":
            if not spec.faker_methods or not spec.mimesis_methods:
                raise ValueError(
                    f"Hybrid spec must define both faker_methods and mimesis_methods for {label}"
                )
            if not MIMESIS_AVAILABLE:
                raise ImportError(
                    "mimesis is required for 'hybrid' TypeSpec entries. "
                    "Install it with: pip install mimesis"
                )
        if spec.kind == "custom" and not spec.generators:
            raise ValueError(f"Custom spec missing generators for {label}")


def _resolve_faker_methods(methods: tuple[str, ...]) -> list[str]:
    available = [method for method in methods if hasattr(fake, method)]
    if not available:
        raise ValueError(f"None of the faker methods exist: {methods}")
    return available


def _build_faker_generators(methods: tuple[str, ...]) -> list[Callable[[], str]]:
    available = _resolve_faker_methods(methods)
    return [
        (lambda method_name=method_name: _normalize_value(getattr(fake, method_name)()))
        for method_name in available
    ]


def _resolve_mimesis_method(method_path: str) -> Callable[[], object] | None:
    if not MIMESIS_AVAILABLE or mimesis is None:
        return None

    target: object = mimesis
    for attr in method_path.split("."):
        if not hasattr(target, attr):
            return None
        target = getattr(target, attr)

    if not callable(target):
        return None

    return target


def _resolve_mimesis_methods(methods: tuple[str, ...]) -> list[Callable[[], object]]:
    if not MIMESIS_AVAILABLE:
        raise ImportError(
            "mimesis is not installed. Install it with: pip install mimesis"
        )

    available = [
        method
        for method in (_resolve_mimesis_method(path) for path in methods)
        if method is not None
    ]
    if not available:
        raise ValueError(f"None of the mimesis methods exist: {methods}")
    return available


def _build_mimesis_generators(methods: tuple[str, ...]) -> list[Callable[[], str]]:
    available = _resolve_mimesis_methods(methods)
    generators: list[Callable[[], str]] = []

    for method in available:

        def _gen(method_callable: Callable[[], object] = method) -> str:
            try:
                return _normalize_value(method_callable())
            except TypeError as exc:
                raise ValueError(
                    "Configured mimesis method requires arguments. "
                    f"Use a no-arg method path. Got: {method_callable}"
                ) from exc

        generators.append(_gen)

    return generators


def _build_value_generator(spec: TypeSpec) -> Callable[[], str]:
    if spec.kind == "faker":
        generators = _build_faker_generators(spec.faker_methods)
        return lambda: random.choice(generators)()

    if spec.kind == "mimesis":
        generators = _build_mimesis_generators(spec.mimesis_methods)
        return lambda: random.choice(generators)()

    if spec.kind == "hybrid":
        generators = _build_faker_generators(spec.faker_methods) + _build_mimesis_generators(
            spec.mimesis_methods
        )
        return lambda: random.choice(generators)()

    if spec.kind == "custom":
        generator = random.choice(spec.generators)
        return generator

    raise ValueError("Curated+synthetic types should be skipped")


def _get_csv_header(path: str) -> list[str]:
    with open(path, "r", newline="") as handle:
        return [col.strip() for col in handle.readline().strip().split(",")]


def _ensure_output_schema(
    path: str, l2_to_datatype: dict[str, str], output_columns: list[str]
) -> bool:
    """Ensure CSV header matches output schema; migrate legacy schema if needed."""
    if not os.path.exists(path):
        return True

    existing_header = _get_csv_header(path)
    if existing_header == output_columns:
        return False

    if existing_header == LEGACY_OUTPUT_COLUMNS:
        legacy_df = pd.read_csv(path)
        missing_legacy_cols = [col for col in LEGACY_OUTPUT_COLUMNS if col not in legacy_df.columns]
        if missing_legacy_cols:
            raise ValueError(
                f"{path} is missing required legacy columns for migration: {missing_legacy_cols}"
            )

        unknown_l2 = sorted(set(legacy_df["l2_label"].astype(str)) - set(l2_to_datatype))
        if unknown_l2:
            raise ValueError(
                f"Cannot migrate {path}; unknown l2 labels found: {unknown_l2}"
            )

        legacy_df["l2_label"] = legacy_df["l2_label"].astype(str)
        legacy_df["datatype"] = legacy_df["l2_label"].map(l2_to_datatype)
        legacy_df.to_csv(path, index=False, columns=output_columns)
        print(f"Migrated legacy checkpoint schema to include datatype: {path}")
        return False

    raise ValueError(
        f"{path} has unsupported header {existing_header}. "
        f"Expected one of: {LEGACY_OUTPUT_COLUMNS} or {output_columns}"
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


def _iter_l2_labels_from(
    resume_from_l2: str | None = None,
) -> Iterator[tuple[str, str]]:
    started = resume_from_l2 is None
    for l1_label, l2_labels in TWO_LEVEL_ONTOLOGY.items():
        for l2_label in l2_labels:
            if not started:
                if l2_label != resume_from_l2:
                    continue
                started = True
            yield l1_label, l2_label


def generate_synthetic_checkpoint(
    output_path: str = "synthetic_df_checkpoint.csv",
    num_synthetic_per_class: int = 200,
    num_of_values_per_class: int = 3,
    batch_size: int = 10,
    resume_from_l2: str | None = None,
) -> None:
    if num_synthetic_per_class < 1:
        raise ValueError("num_synthetic_per_class must be >= 1")
    if num_of_values_per_class < 1:
        raise ValueError("num_of_values_per_class must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    output_columns = OUTPUT_COLUMNS
    l2_to_l1 = _build_l2_to_l1()
    l2_to_datatype = _build_l2_to_datatype(l2_to_l1)
    _validate_specs(l2_to_l1)
    _validate_datatype_mapping(l2_to_l1, l2_to_datatype)

    if resume_from_l2 is not None and resume_from_l2 not in l2_to_l1:
        valid_labels = ", ".join(sorted(l2_to_l1))
        raise ValueError(
            f"Unknown resume_from_l2 label '{resume_from_l2}'. "
            f"Valid l2 labels: {valid_labels}"
        )

    header_needed = _ensure_output_schema(
        output_path,
        l2_to_datatype=l2_to_datatype,
        output_columns=output_columns,
    )

    for l1_label, l2_label in _iter_l2_labels_from(resume_from_l2=resume_from_l2):
        spec = TYPE_SPECS[l2_label]
        if spec.kind == "curated+synthetic":
            continue

        print(f"- `{l2_label}` ({l1_label}) - {spec.description} [{spec.kind}]")
        for name_batch in iter_synthetic_name_batches(
            num_names=num_synthetic_per_class,
            class_name=l2_label,
            description=spec.description,
            batch_size=batch_size,
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
                        "datatype": l2_to_datatype[l2_label],
                    }
                )

            synthetic_df = pd.DataFrame(rows, columns=output_columns)
            synthetic_df.to_csv(
                output_path,
                mode="a",
                header=header_needed,
                index=False,
            )
            header_needed = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CTA checkpoint data with Faker/Mimesis + LLM names."
    )
    parser.add_argument(
        "--resume-from-l2",
        type=str,
        default=None,
        help="Start processing from this l2 class label (inclusive). Example: month_of_year",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="synthetic_df_checkpoint.csv",
        help="CSV checkpoint path.",
    )
    parser.add_argument(
        "--num-synthetic-per-class",
        type=int,
        default=200,
        help="Number of synthetic column names to generate per class.",
    )
    parser.add_argument(
        "--num-values-per-class",
        type=int,
        default=3,
        help="Number of synthetic example values per generated name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="LLM name generation batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_synthetic_checkpoint(
        output_path=args.output_path,
        num_synthetic_per_class=args.num_synthetic_per_class,
        num_of_values_per_class=args.num_values_per_class,
        batch_size=args.batch_size,
        resume_from_l2=args.resume_from_l2,
    )


if __name__ == "__main__":
    main()
