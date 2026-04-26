Examples
========

Basic Usage
-----------

.. code-block:: python

   import pandas as pd
   from atlas_profiler import process_dataset

   # Load data into DataFrame
   df = pd.read_csv("geospatial_data.csv")

   # Process with all features enabled
   metadata = process_dataset(
       df,
       geo_classifier=True,
       geo_classifier_threshold=0.5,
       coverage=True,
       plots=True,
       include_sample=True,
   )

   # Access spatial coverage
   spatial_coverage = metadata.get('spatial_coverage', {})
   print(f"Bounding box: {spatial_coverage.get('bbox')}")

Getting Started with NYC Traffic Data
-------------------------------------

.. code-block:: python

   from atlas_profiler import process_dataset
   import pandas as pd

   # Load sample NYC traffic volume data
   nyc_traffic_volume_sample_df = pd.read_csv(
       "hf://datasets/oscur/automated-traffic-volume-counts-sample/sample_traffic.csv"
   )

   nyc_traffic_volume_sample_df.head()

.. code-block:: python

   # Profile the dataset with geo classifier enabled
   profiled_columns = process_dataset(
       nyc_traffic_volume_sample_df,
       coverage=False,
       indexes=False,
       geo_classifier=True,
       load_max_size=1000000,
   )

   profiled_columns

Output Structure
================

The ``process_dataset`` function returns a dictionary containing comprehensive metadata about the profiled dataset. Here's a breakdown of the main fields based on the actual output:

**Dataset-level Information:**

- ``nb_rows``: Total number of rows in the dataset
- ``nb_profiled_rows``: Number of rows actually profiled (may be less if sampling was used)
- ``nb_columns``: Total number of columns
- ``nb_spatial_columns``: Number of columns identified as spatial
- ``nb_categorical_columns``: Number of columns identified as categorical
- ``nb_numerical_columns``: Number of columns identified as numerical
- ``types``: List of dataset-level type categories present (e.g., ``["categorical", "numerical", "spatial"]``)

**Column-level Information:**

- ``columns``: List of column metadata dictionaries, each containing:
  - ``name``: Column name
  - ``structural_type``: Detected structural type using schema.org URLs (e.g., ``"http://schema.org/Integer"``, ``"http://schema.org/Text"``, ``"http://schema.org/GeoCoordinates"``)
  - ``semantic_types``: List of semantic type annotations using schema.org URLs (e.g., ``["http://schema.org/identifier"]``, ``["http://schema.org/AdministrativeArea"]``, ``["http://schema.org/address"]``)
  - ``unclean_values_ratio``: Ratio of unclean/missing values (0.0 to 1.0)
  - ``num_distinct_values``: Number of unique values in the column
  - ``geo_classifier`` (optional): Geo classifier results for spatial columns, containing:

    - ``label``: Predicted spatial label (e.g., ``"borough"``, ``"point"``, ``"address"``)
    - ``confidence``: Confidence score (0.0 to 1.0)
    - ``source``: Source of the classification (e.g., ``"ml+validated"``, ``"ml"``)

**Additional Metadata:**

- ``attribute_keywords``: List of keywords extracted from all column names
- ``_profiling_times``: Performance timing information:
  - ``steps``: Dictionary of timing for each profiling step (in seconds)
  - ``total``: Total profiling time (in seconds)