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
        "year", # [faker] 1975, 2024, etc.
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