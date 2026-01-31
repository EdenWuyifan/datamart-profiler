import os
import random
import re
import string
import uuid
from collections.abc import Iterator

import pandas as pd
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders
from faker import Faker

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

def parse_response(response: str) -> list:
    """Parse LLM response into names."""
    names = []
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


def _rand_digits(length: int) -> str:
    return "".join(random.choices(string.digits, k=length))


def _rand_upper(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=length))


def _rand_alnum_upper(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _rand_hex(length: int) -> str:
    return "".join(random.choices("0123456789abcdef", k=length))


def _float_range(low: float, high: float, decimals: int = 1) -> str:
    return f"{random.uniform(low, high):.{decimals}f}"


def _int_range(low: int, high: int) -> str:
    return str(random.randint(low, high))


def _with_unit(value: str, unit: str) -> str:
    return f"{value} {unit}"


def _usd_amount(low: float, high: float, decimals: int = 2, style: str = "symbol") -> str:
    amount = f"{random.uniform(low, high):.{decimals}f}"
    if style == "symbol":
        return f"${amount}"
    if style == "code":
        return f"USD {amount}"
    if style == "suffix":
        return f"{amount} USD"
    return amount


def _gen_license_plate() -> str:
    if random.random() < 0.5:
        return f"{_rand_upper(3)}-{_rand_digits(4)}"
    return f"{_rand_upper(2)}{_rand_digits(3)}{_rand_upper(2)}"


def _gen_vin() -> str:
    vin_chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    return "".join(random.choices(vin_chars, k=17))


def _gen_blood_pressure() -> str:
    systolic = random.randint(90, 160)
    diastolic = random.randint(60, 100)
    return f"{systolic}/{diastolic}"

def iter_synthetic_name_batches(
    num_names: int,
    class_name: str,
    description: str,
    batch_size: int = 10,
) -> Iterator[list[str]]:
    """Yield synthetic names per LLM batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    llm = get_llm()
    max_attempts = max(1, ((num_names + batch_size - 1) // batch_size) * 3)
    attempts = 0
    total_names = 0
    while total_names < num_names and attempts < max_attempts:
        remaining = num_names - total_names
        request_size = min(batch_size, remaining)
        prompt = f"""
        Generate a list of {request_size} unique synthetic attribute names (use as the column names) for the following class:

        Class: {class_name}
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

        Class: {class_name}
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
    ):
        names.extend(batch)
    return names

non_spatial_classes = {
    "ean8": "barcode",
    "ean13": "barcode",
    "hex_color": "color codes",
    "rgb_color": "color codes",
    "company": "company names",
    "credit_card_number": "credit card numbers",
    "currency_code": "currency codes",
    "unix_time": "timestamps",
    "iso8601": "timestamps",
    "date_time": "timestamps",
    "year": "date parts",
    "month_name": "date parts",
    "month": "date parts",
    "day_of_week": "date parts",
    "day_of_month": "date parts",
    "file_extension": "file metadata",
    "file_name": "file metadata",
    "email": "email addresses",
    "url": "website URLs",
    "ipv6": "IP addresses",
    "ipv4": "IP addresses",
    "mac_address": "MAC addresses",
    "job": "job titles",
    "name": "personal names",
    "last_name": "personal names",
    "first_name": "personal names",
    "prefix": "personal names",
    "phone_number": "phone numbers",
    "ssn": "social security numbers",
}

# Custom categories not covered by Faker
color_names = [
    "Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Pink", "Brown",
    "Black", "White", "Gray", "Cyan", "Magenta", "Teal", "Navy", "Maroon",
    "Olive", "Lime", "Coral", "Turquoise", "Lavender", "Gold", "Silver",
]

status_values = ["active", "inactive", "pending", "closed", "archived", "suspended"]
priority_values = ["low", "medium", "high", "urgent"]
severity_values = ["minor", "major", "critical", "blocker"]
status_levels = status_values + priority_values + severity_values
yes_no_values = ["Yes", "No"]
boolean_values = ["True", "False"]
tshirt_sizes = ["XS", "S", "M", "L", "XL", "XXL"]
payment_methods = ["credit_card", "debit_card", "cash", "paypal", "ach", "apple_pay", "google_pay"]
shipping_methods = ["ground", "2-day", "overnight", "same-day", "pickup"]
device_types = ["mobile", "desktop", "tablet", "wearable", "iot"]
operating_systems = ["Windows 11", "macOS 14", "Ubuntu 22.04", "iOS 17", "Android 14", "ChromeOS"]
browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera", "Brave"]
language_codes = ["en", "es", "fr", "de", "pt", "it", "zh", "ja", "ko", "ar"]
time_zones = [
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Phoenix",
    "America/Anchorage",
    "Pacific/Honolulu",
    "UTC",
]
mime_types = [
    "application/json",
    "text/csv",
    "text/plain",
    "application/pdf",
    "image/png",
    "image/jpeg",
    "application/zip",
    "application/vnd.ms-excel",
]
fuel_types = ["gasoline", "diesel", "electric", "hybrid", "ethanol"]
vehicle_types = ["sedan", "suv", "truck", "van", "coupe", "hatchback", "motorcycle"]
blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
grade_letters = ["A", "B", "C", "D", "F"]

age_ranges = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
shoe_sizes_us = [str(x) for x in [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13]]

custom_classes = {
    "age": ("age values", [
        lambda: _int_range(0, 100),
        lambda: _choice(age_ranges),
    ]),
    "color": ("color names", lambda: _choice(color_names)),
    "money": ("monetary amounts in USD", [
        lambda: _usd_amount(0.5, 5000, style="symbol"),
        lambda: _usd_amount(1, 10000, style="code"),
        lambda: _usd_amount(25, 500, style="suffix"),
        lambda: _usd_amount(25000, 200000, decimals=0, style="symbol"),
    ]),
    "percent": ("percentage values", [
        lambda: f"{_float_range(0, 90, 1)}%",
        lambda: f"{_float_range(0, 12, 2)}%",
        lambda: f"{_float_range(0, 100, 1)}%",
    ]),
    "rating": ("rating values", [
        lambda: _float_range(1, 5, 1),
        lambda: _float_range(1, 10, 1),
    ]),
    "score": ("scores or points", [
        lambda: _int_range(0, 100),
        lambda: _int_range(0, 1000),
    ]),
    "flag": ("boolean or yes/no flags", [
        lambda: _choice(yes_no_values),
        lambda: _choice(boolean_values),
    ]),
    "status": ("status or priority labels", lambda: _choice(status_levels)),
    "size": ("size labels or sizes", [
        lambda: _choice(tshirt_sizes),
        lambda: _choice(shoe_sizes_us),
    ]),
    "height": ("height measurements", [
        lambda: _with_unit(_int_range(140, 210), "cm"),
        lambda: _with_unit(_int_range(55, 83), "in"),
    ]),
    "weight": ("weight measurements", [
        lambda: _with_unit(_float_range(40, 130, 1), "kg"),
        lambda: _with_unit(_int_range(90, 300), "lb"),
    ]),
    "temperature": ("temperature measurements", [
        lambda: _with_unit(_float_range(-10, 40, 1), "C"),
        lambda: _with_unit(_float_range(14, 104, 1), "F"),
    ]),
    "distance": ("distance measurements", [
        lambda: _with_unit(_float_range(0.1, 500, 1), "mi"),
        lambda: _with_unit(_float_range(1, 3000, 0), "ft"),
    ]),
    "speed": ("speed measurements", [
        lambda: _with_unit(_float_range(0, 120, 1), "mph"),
        lambda: _with_unit(_float_range(0, 200, 1), "km/h"),
    ]),
    "area": ("area measurements", [
        lambda: _with_unit(_int_range(200, 5000), "sqft"),
        lambda: _with_unit(_float_range(0.05, 5, 2), "acres"),
    ]),
    "volume": ("volume measurements", [
        lambda: _with_unit(_float_range(0.5, 500, 1), "gal"),
        lambda: _with_unit(_float_range(1, 2000, 0), "ml"),
    ]),
    "pressure": ("pressure measurements", lambda: _with_unit(_float_range(5, 100, 1), "psi")),
    "energy": ("energy usage values", lambda: _with_unit(_float_range(0.1, 100, 2), "kWh")),
    "duration": ("duration or latency values", [
        lambda: _with_unit(_int_range(1, 5000), "ms"),
        lambda: _with_unit(_int_range(1, 86400), "s"),
        lambda: _with_unit(_float_range(0.5, 600, 1), "min"),
        lambda: _with_unit(_float_range(0.1, 72, 1), "hr"),
    ]),
    "identifier": ("generic identifiers", [
        lambda: str(uuid.uuid4()),
        lambda: _rand_hex(16),
        lambda: f"ORD-{_rand_digits(6)}",
        lambda: f"INV-{_rand_digits(6)}",
        lambda: f"TXN-{_rand_digits(8)}",
        lambda: f"ID-{_rand_alnum_upper(8)}",
    ]),
    "payment_method": ("payment methods", lambda: _choice(payment_methods)),
    "shipping_method": ("shipping methods", lambda: _choice(shipping_methods)),
    "platform": ("device, OS, or browser names", [
        lambda: _choice(device_types),
        lambda: _choice(operating_systems),
        lambda: _choice(browsers),
    ]),
    "locale": ("language codes or time zones", [
        lambda: _choice(language_codes),
        lambda: _choice(time_zones),
    ]),
    "version": ("software version strings", lambda: f"v{random.randint(0, 9)}.{random.randint(0, 9)}.{random.randint(0, 20)}"),
    "hash": ("hash strings", [
        lambda: _rand_hex(random.randint(7, 10)),
        lambda: _rand_hex(32),
        lambda: _rand_hex(40),
    ]),
    "health": ("health-related values", [
        lambda: _choice(blood_types),
        lambda: _with_unit(_int_range(50, 140), "bpm"),
        _gen_blood_pressure,
        lambda: _float_range(15, 40, 1),
    ]),
    "grade": ("academic grades", [
        lambda: _choice(grade_letters),
        lambda: _float_range(0, 4, 2),
    ]),
}

overlapping_labels = set(non_spatial_classes) & set(custom_classes)
if overlapping_labels:
    raise ValueError(f"Custom labels overlap with Faker labels: {sorted(overlapping_labels)}")

num_synthetic_per_class = 200
num_of_values_per_class = 3

class_generators = {}
for function_name, description in non_spatial_classes.items():
    class_generators[function_name] = (
        description,
        lambda fn=function_name: str(getattr(fake, fn)()),
    )
class_generators.update(custom_classes)

for label, (description, generator) in class_generators.items():
    print(f"- `{label}` - {description}")
    if isinstance(generator, (list, tuple)):
        value_generator = random.choice(generator)
    else:
        value_generator = generator
    for name_batch in iter_synthetic_name_batches(
        num_names=num_synthetic_per_class,
        class_name=label,
        description=description,
        batch_size=10,
    ):
        batch_count = len(name_batch)
        values = [str(value_generator()) for _ in range(batch_count * num_of_values_per_class)]

        # cache generated data in synthetic_df_checkpoint.csv, as name,values,label - ZIP_Plus_4,"90210-1234, 75201-5555, 98101-2000",zip9
        synthetic_df = pd.DataFrame({
            "name": name_batch,
            "values": [", ".join(values[i*num_of_values_per_class:(i+1)*num_of_values_per_class]) for i in range(batch_count)],
            "label": label,
        })
        synthetic_df.to_csv("synthetic_df_checkpoint.csv", mode="a", header=not os.path.exists("synthetic_df_checkpoint.csv"), index=False)
