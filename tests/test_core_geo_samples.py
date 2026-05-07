import pandas

from profiler.core import load_data


def test_load_data_formats_float_samples_for_geo_classifier():
    data = pandas.DataFrame(
        {
            "lat": [
                40.67768952967763,
                40.69306007321615,
                40.69342487421583,
            ]
        }
    )

    _, _, _, stats = load_data(data, indexes=False)

    assert stats["lat"]["sample_values"] == ["40.67769", "40.69306", "40.693425"]
