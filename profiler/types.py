# Column types

MISSING_DATA = "https://metadata.datadrivendiscovery.org/types/MissingData"
"""No data (whole column is missing)"""

INTEGER = "http://schema.org/Integer"
"""Integer (numbers without a decimal point)"""

FLOAT = "http://schema.org/Float"
"""Floating-point numbers"""

TEXT = "http://schema.org/Text"
"""Text, better represented as strings"""

BOOLEAN = "http://schema.org/Boolean"
"""Booleans, e.g. only the two values \"true\" and \"false\""""

LATITUDE = "http://schema.org/latitude"
"""Numerical values representing latitude coordinates"""

LONGITUDE = "http://schema.org/longitude"
"""Numerical values representing longitude coordinates"""

DATE_TIME = "http://schema.org/DateTime"
"""A specific instant in time (not partial ones such as "July 4" or "12am")"""

ADDRESS = "http://schema.org/address"
"""The street address of a location"""

ADMIN = "http://schema.org/AdministrativeArea"
"""A named administrative area, such as a country, state, or city"""

URL = "http://schema.org/URL"
"""A URL"""

EMAIL = "http://schema.org/email"
"""An email address"""

TELEPHONE = "http://schema.org/telephone"
"""A telephone number"""

NAME = "http://schema.org/name"
"""A name"""

GIVEN_NAME = "http://schema.org/givenName"
"""A given (first) name"""

FAMILY_NAME = "http://schema.org/familyName"
"""A family (last) name"""

HONORIFIC_PREFIX = "http://schema.org/honorificPrefix"
"""An honorific prefix (e.g., Dr, Mr, Ms)"""

JOB_TITLE = "http://schema.org/jobTitle"
"""A job title"""

ORGANIZATION = "http://schema.org/Organization"
"""An organization"""

GTIN8 = "http://schema.org/gtin8"
"""A GTIN-8 / EAN-8 code"""

GTIN13 = "http://schema.org/gtin13"
"""A GTIN-13 / EAN-13 code"""

COLOR = "http://schema.org/color"
"""A color"""

PRICE = "http://schema.org/price"
"""A price or monetary amount"""

PRICE_CURRENCY = "http://schema.org/priceCurrency"
"""A currency code for a price"""

DATE = "http://schema.org/Date"
"""A calendar date"""

DURATION = "http://schema.org/duration"
"""A duration"""

SIZE = "http://schema.org/size"
"""A standardized size"""

HEIGHT = "http://schema.org/height"
"""A height measurement"""

WEIGHT = "http://schema.org/weight"
"""A weight measurement"""

RATING_VALUE = "http://schema.org/ratingValue"
"""A rating value"""

QUANTITATIVE_VALUE = "http://schema.org/QuantitativeValue"
"""A quantitative value"""

PAYMENT_METHOD = "http://schema.org/paymentMethod"
"""A payment method"""

DELIVERY_METHOD = "http://schema.org/deliveryMethod"
"""A delivery/shipping method"""

VERSION = "http://schema.org/version"
"""A version string"""

IN_LANGUAGE = "http://schema.org/inLanguage"
"""A language code"""

SCHEDULE_TIMEZONE = "http://schema.org/scheduleTimezone"
"""A time zone"""

OPERATING_SYSTEM = "http://schema.org/operatingSystem"
"""An operating system"""

AVAILABLE_ON_DEVICE = "http://schema.org/availableOnDevice"
"""A device required to run a software application"""

BROWSER_REQUIREMENTS = "http://schema.org/browserRequirements"
"""Browser requirements"""

ENCODING_FORMAT = "http://schema.org/encodingFormat"
"""A media/file encoding format (MIME type)"""

POSTAL_CODE = "http://schema.org/postalCode"
"""A postal/ZIP code"""

ADDRESS_LOCALITY = "http://schema.org/addressLocality"
"""A city/locality component of an address"""

ADDRESS_REGION = "http://schema.org/addressRegion"
"""A state/region component of an address"""

ADDRESS_COUNTRY = "http://schema.org/addressCountry"
"""A country component of an address"""

DAY_OF_WEEK = "http://schema.org/dayOfWeek"
"""A day of the week"""

SUGGESTED_AGE = "http://schema.org/suggestedAge"
"""A suggested age or age range"""

EDUCATIONAL_LEVEL = "http://schema.org/educationalLevel"
"""An educational level"""

TAX_ID = "http://schema.org/taxID"
"""A tax identifier (e.g., SSN/TIN)"""

CREDIT_CARD = "http://schema.org/CreditCard"
"""A credit card (type)"""

FILE_PATH = "https://metadata.datadrivendiscovery.org/types/FileName"
"""A filename"""

ID = "http://schema.org/identifier"
"""An identifier"""

CATEGORICAL = "http://schema.org/Enumeration"
"""Categorical values, i.e. drawn from a limited number of options"""

GEO_POINT = "http://schema.org/GeoCoordinates"
"""A geographic location (latitude+longitude coordinates)"""

GEO_POLYGON = "http://schema.org/GeoShape"
"""A geographic shape described by its coordinates"""


# Dataset types

DATASET_NUMERICAL = "numerical"
DATASET_CATEGORICAL = "categorical"
DATASET_SPATIAL = "spatial"
DATASET_TEMPORAL = "temporal"
