from datetime import datetime

from profiler.temporal import get_temporal_resolution, parse_date


def test_parse_date_full_date_adds_timezone():
    dt = parse_date("2020-06-15")
    assert dt is not None
    assert dt.tzinfo is not None
    assert (dt.year, dt.month, dt.day) == (2020, 6, 15)


def test_parse_date_time_only_rejected():
    assert parse_date("11:00") is None


def test_get_temporal_resolution_day():
    values = {datetime(2020, 1, 1), datetime(2020, 1, 2)}
    assert get_temporal_resolution(values) == "day"


def test_get_temporal_resolution_second():
    values = {datetime(2020, 1, 1, 0, 0, 1)}
    assert get_temporal_resolution(values) == "second"
