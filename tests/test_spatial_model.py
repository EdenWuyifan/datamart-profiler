import pandas as pd
import pytest

import profiler.spatial as spatial


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        yield self._payload


def test_download_geo_model_writes_files(tmp_path, monkeypatch):
    files = {
        "model.pt": "https://example.com/model.pt",
        "config.json": "https://example.com/config.json",
        "label_encoder.json": "https://example.com/label_encoder.json",
    }
    payloads = {
        "https://example.com/model.pt": b"model-bytes",
        "https://example.com/config.json": b'{"model":"test"}',
        "https://example.com/label_encoder.json": b'{"classes":["a"]}',
    }

    def fake_get(url, **_kwargs):
        return DummyResponse(payloads[url])

    monkeypatch.setattr(spatial.requests, "get", fake_get)

    spatial.download_geo_model(str(tmp_path), files=files)

    for filename, url in files.items():
        assert (tmp_path / filename).read_bytes() == payloads[url]


def test_geo_classifier_uses_cache_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    expected_dir = tmp_path / "atlas_profiler" / "model"

    with pytest.raises(FileNotFoundError) as excinfo:
        spatial.GeoClassifier(model_dir=None, auto_download=False)

    assert str(expected_dir) in str(excinfo.value)


def test_pair_latlong_columns_matches_names():
    columns_lat = [
        spatial.LatLongColumn(index=0, name="cab_latitude_from", annot_pair=None)
    ]
    columns_long = [
        spatial.LatLongColumn(index=1, name="cab_longitude_from", annot_pair=None)
    ]
    pairs, (missed_lat, missed_long) = spatial.pair_latlong_columns(
        columns_lat, columns_long
    )

    assert len(pairs) == 1
    assert missed_lat == []
    assert missed_long == []


def test_parse_wkt_column_returns_lat_long_pairs():
    series = pd.Series(["(10.0, 20.0)", "(bad)"])
    points = spatial.parse_wkt_column(series, latlong=False)
    assert points == [(20.0, 10.0)]
