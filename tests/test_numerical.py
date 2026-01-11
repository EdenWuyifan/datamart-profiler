import pytest

from profiler.numerical import get_numerical_ranges, mean_stddev


def test_mean_stddev_basic():
    mean, stddev = mean_stddev([1, 2, 3])
    assert mean == 2
    assert stddev == pytest.approx((2 / 3) ** 0.5)


def test_get_numerical_ranges_basic():
    values = list(range(100))
    ranges = get_numerical_ranges(values)

    assert len(ranges) == 3
    for rg in ranges:
        assert "range" in rg
        assert rg["range"]["gte"] <= rg["range"]["lte"]
        assert values[0] <= rg["range"]["gte"] <= values[-1]
        assert values[0] <= rg["range"]["lte"] <= values[-1]
