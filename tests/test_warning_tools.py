import warnings

import pytest

from profiler.warning_tools import ignore_warnings, raise_warnings


class CustomWarning(UserWarning):
    pass


def test_ignore_warnings_filters_specific_category():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with ignore_warnings(CustomWarning):
            warnings.warn("ignore me", CustomWarning)
            warnings.warn("keep me", RuntimeWarning)

    assert len(recorded) == 1
    assert issubclass(recorded[0].category, RuntimeWarning)


def test_raise_warnings_escalates():
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.raises(CustomWarning):
            with raise_warnings(CustomWarning):
                warnings.warn("boom", CustomWarning)
