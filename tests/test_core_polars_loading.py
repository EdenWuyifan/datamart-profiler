import pytest
from datetime import datetime, timezone

from profiler.core import process_dataset


pytest.importorskip("polars")


def test_csv_path_profiles_full_numeric_stats_before_sampling(tmp_path):
    path = tmp_path / "values.csv"
    path.write_text(
        "value,latitude,longitude,observed_at\n"
        + "\n".join(
            f"{i},{i / 10 + 0.01},{-100 + (i / 10) + 0.01},"
            f"2020-01-{(i % 28) + 1:02d}"
            for i in range(200)
        )
        + "\n"
    )

    metadata = process_dataset(
        str(path),
        geo_classifier=False,
        coverage=True,
        load_max_size=1000,
    )

    value = metadata["columns"][0]
    assert metadata["nb_rows"] == 200
    assert metadata["nb_profiled_rows"] < metadata["nb_rows"]
    assert value["min"] == 0.0
    assert value["max"] == 199.0
    assert value["mean"] == pytest.approx(99.5)
    assert value["num_distinct_values_is_approximate"] is True

    spatial = next(c for c in metadata["spatial_coverage"] if c["type"] == "latlong")
    coordinates = spatial["ranges"][0]["range"]["coordinates"]
    assert spatial["number"] == 200
    assert coordinates[0][0] == pytest.approx(-99.99)
    assert coordinates[0][1] == pytest.approx(19.91)
    assert coordinates[1][0] == pytest.approx(-80.09)
    assert coordinates[1][1] == pytest.approx(0.01)

    temporal = metadata["temporal_coverage"][0]["ranges"][0]["range"]
    assert temporal["gte"] == datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    assert temporal["lte"] == datetime(2020, 1, 28, tzinfo=timezone.utc).timestamp()
