# ======================= Atlas Profiler Classifications Types =========================
# Author: Eden Wu
# This module defines types and constants used for classifying Atlas profilers.


TWO_LEVEL_ONTOLOGY = {
    "spatial": [
        "address", # [faker] 7147 Roberts Plaza New Justinview, OH 08892
        "city", # [faker] Shawnbury, Lake Amychester, etc.
        "state", # [faker] Ohio, California, New York, etc.
        "state_code", # [faker (state_abbr)] OH, CA, NY, etc.
        "country", # [faker] Jordan, Canada, United States, etc.
        "country_code", # [faker] US, CA, GB, etc.
        "zip5", # [faker] 5-digit zip code
        "zip9", # [faker] 9-digit zip code
        "borough", # [curated+synthetic] e.g., Manhattan, Brooklyn, etc.
        "borough_code", # [curated+synthetic] e.g., MN, BK, 1, 3 etc.
        "bin", # [curated+synthetic] e.g., NYC Building Identification Number
        "bbl" # [curated+synthetic] e.g., NYC Borough, Block and Lot Number 
    ],
    "spatial_geom": [
        "point", # [curated+synthetic] POINT (1030236.5248402834 202340.1234234)
        "line", # [curated+synthetic] LINESTRING (998100 201500, 998200 202000, 998300 203000)
        "polygon", # [curated+synthetic] POLYGON ((998100 201500, 998200 202000, 998300 203000, 998100 201500))
        "multi-line", # [curated+synthetic] MULTILINESTRING ((998100 201500, 998200 202000), (998300 203000, 998400 20400))
        "multi-polygon", # [curated+synthetic] MULTIPOLYGON (((998100 20150, 9982444444444444444444444
    ],
    "spatial_coord": [
        "latitude", # [faker] 34.5678, -118.5678, etc.
        "longitude", # [faker] -118.5678, -117.5678, etc.
        "x_coord", # [curated+synthetic] e.g., 1337.5
        "y_coord", # [curated+synthetic] e.g., -765.3
    ],
    "temporal": [
        "data_time", # [faker] 1996-04-24 02:32:59
        "iso8601", # [faker] 1971-07-17T01:53:59
        "unix_time", # [faker] 17739602
        "date", # [custom_classes] 2026-02-23
        "time", # [custom_classes] 13:45:22
        "year", # [faker] 1975, 2024, etc.
        "quarter", # [custom_classes] 1, 2, 3, 4
        "week_of_year", # [custom_classes] 1-53
        "month_name", # [faker] January, February, ...
        "month_of_year", # [faker (month)] 1, 01, etc.
        "day_of_month", # [faker] 1-31, 01-31
        "day_of_week", # [faker] Monday, Tuesday, ...
    ],
    "person": [
        "first_name", # [faker] John, Jane, etc.
        "last_name", # [faker] Smith, Johnson, etc.
        "name", # [faker] John Smith, Jane Johnson, etc.
        "prefix", # [faker] Mr., Mrs., Dr., etc.
        "age", # [custom_classes] 25, 30, 35, etc.
        "email", # [faker] john.smith@example.com, jane.johnson@example.com, etc.
        "phone_number", # [faker] (555) 123-4567, +1-555-123-4567, etc.
        "ssn", # [faker] 123-45-6789
        "company", # [faker] Google, Microsoft, etc.
        "job", # [faker] Software Engineer, Data Scientist, etc.
    ],
    "financial_commerce": [
        "money", # [custom_classes] $1,234.56, €1.234,56, etc.
        "currency_code", # [faker] USD, EUR, etc.
        "payment_method", # [custom_classes] card, ach, cash, wire, etc.
        "amount_range", # [custom_classes] 0-50, 50-100, etc.
        "credit_card_number", # [faker] 4111 1111 1111 1111, etc.
        "credit_card_security_code", # [faker] 123, 456, etc.
        "credit_card_expire", # [faker] 12/24, 01/25, etc.
        "credit_card_provider", # [faker] Visa, MasterCard, etc.
        "credit_card_full", # [faker] Full credit card info
    ],
    "system": [
        "url", # [faker] http://example.com, https://example.com, etc.
        "ipv4", # [faker] 220.56.71.177, etc.
        "ipv6", # [faker] 2001:0db8:85a3:0000:0000:8a2e:0370:7334, etc.
        "mac_address", # [faker] 00:1A:2B:3C:4D:5E, etc.
        "platform", # [faker (mac_platform_token, windows_platform_token, linux_platform_token)] Windows 10, macOS 11, Ubuntu 20.04, etc.
        "file_name", # [faker] document.pdf, image.png, etc.
        "file_extension" # [faker] pdf, png, docx, etc.
    ],
    "identifier_code": [
        "uuid", # [custom_classes] 123e4567-e89b-12d3-a456-426614174000
        "ean8", # [faker] 12345670
        "ean13", # [faker] 1234567890128
        "alphanumeric_code", # [custom_classes] AB12CD34, X9Q7Z2
        "checksum_code", # [custom_classes] ID-12345-7
    ],
    "entity_identifier": [
        "primary_key", # [custom_classes] 1001, 1002, ...
        "foreign_key", # [custom_classes] 42, 77, ...
        "user_id", # [custom_classes] USR-000123
        "customer_id", # [custom_classes] CUST-000123
        "account_id", # [custom_classes] ACC-000123
        "order_id", # [custom_classes] ORD-000123
        "transaction_id", # [custom_classes] TXN-00012345
        "session_id", # [custom_classes] SES-9F2A7C1D
        "device_id", # [custom_classes] DEV-A1B2C3D4
        "request_id", # [custom_classes] REQ-20260223-000001
        "trace_id", # [custom_classes] TRACE-8F4B2A1C
        "sku", # [custom_classes] SKU-AB12-340
        "upc12", # [custom_classes] 036000291452
        "invoice_id", # [custom_classes] INV-2026-000123
    ],
    "business_process": [
        "status", # [custom_classes] active, pending, closed
        "lifecycle_stage", # [custom_classes] lead, active, churned
        "priority", # [custom_classes] low, medium, high, critical
        "channel", # [custom_classes] web, app, store, phone
        "source_system", # [custom_classes] salesforce, netsuite, sap
        "payment_status", # [custom_classes] paid, pending, failed
        "order_status", # [custom_classes] created, shipped, delivered
        "shipment_status", # [custom_classes] in_transit, delivered
        "return_status", # [custom_classes] requested, approved, refunded
        "fulfillment_status", # [custom_classes] picking, packed, shipped
    ],
    "web_telemetry": [
        "http_method", # [custom_classes] GET, POST, PUT, DELETE
        "http_status_code", # [custom_classes] 200, 404, 500
        "endpoint_path", # [custom_classes] /api/v1/orders/123
        "referrer_url", # [custom_classes] https://example.com/search?q=shoes
        "user_agent", # [custom_classes] Mozilla/5.0 ...
        "hostname", # [custom_classes] api.example.com
        "domain_name", # [custom_classes] example.com
        "latency_ms", # [custom_classes] 25, 120, 950
        "bytes_transferred", # [custom_classes] 512, 2048, 65536
    ],
    "catalog_inventory": [
        "product_name", # [custom_classes] Wireless Mouse
        "brand_name", # [custom_classes] Acme, Contoso
        "category_name", # [custom_classes] electronics, apparel
        "quantity", # [custom_classes] 0, 5, 120
        "unit_price", # [custom_classes] 19.99, 49.50
        "discount_percent", # [custom_classes] 5.0, 10.0, 25.5
        "tax_percent", # [custom_classes] 0.0, 6.5, 8.875
        "availability_status", # [custom_classes] in_stock, out_of_stock
    ],
    "demographic": [
        "birth_date", # [custom_classes] 1990-05-14
        "age_bucket", # [custom_classes] 18-24, 25-34, 35-44
        "gender", # [custom_classes] male, female, non_binary
        "marital_status", # [custom_classes] single, married, divorced
        "education_level", # [custom_classes] high_school, bachelor, master
    ],
    "color": [
        "color_name", # [faker] red, blue, green, etc.
        "hex_color", # [faker] #FF5733, #33FF57, etc.
        "rgb_color", # [faker (rgb_css_color, rgb_color_list, rgb_color)] rgb(255, 87, 51), rgb(51, 255, 87), etc.
    ],
    "measure_physical": [
        "area", # [custom_classes] 100 sq ft, 50 m2, etc.
        "distance", # [custom_classes] 10 miles, 5 km, etc.
        "duration", # [custom_classes] 2 hours, 30 minutes, etc.
        "energy", # [custom_classes] 500 kcal, 2000 kJ, etc.
        "height", # [custom_classes] 6 ft, 180 cm, etc.
        "pressure", # [custom_classes] 1013 hPa, 30 inHg, etc.
        "speed", # [custom_classes] 60 mph, 100 km/h, etc.
        "temperature", # [custom_classes] 98.6 F, 37 C, etc.
        "volume", # [custom_classes] 1 gallon, 3.5 L, etc.
        "weight" # [custom_classes] 150 lbs, 70 kg, etc.
    ],
    "categorical_meta": [
        "rating", # [custom_classes] 1-5 stars, 4/5 etc.
        "score", # [custom_classes] 0-100, 85/100 etc.
        "grade", # [custom_classes] A, B, C, D, F
        "percent", # [custom_classes] 0-100%, 75%, etc.
    ],
    "other": [
        "boolean", # [faker] True, False; T, F
        "flag", # [custom_classes] Yes, No; Y, N
        "size", # [custom_classes] XS, S, M, L,
        "version", # [custom_classes] 1.0.0, 2.1.3, etc.
    ]
}
