import math

import pytest

from mbc.reliable_change import (Change, classify_change, rci_threshold,
                                 reliable_change_index, standard_error_of_measurement)


def test_threshold_matches_hand_calculation():
    # SD = 5, r = 0.92: SEM = 5 * sqrt(0.08) ≈ 1.414; S_diff = 2.0; порог ≈ 3.92
    assert math.isclose(standard_error_of_measurement(5, 0.92), 1.4142, rel_tol=1e-3)
    assert math.isclose(rci_threshold(5, 0.92), 3.92, rel_tol=1e-3)


def test_choice_of_reliability_moves_threshold():
    alpha_based = rci_threshold(5, 0.92)
    retest_based = rci_threshold(5, 0.83)
    assert retest_based > alpha_based * 1.4


def test_rci_sign():
    assert reliable_change_index(14, 7, 5, 0.92) == pytest.approx(-3.5)


@pytest.mark.parametrize("x1,x2,expected", [
    (14, 10, Change.IMPROVED),
    (14, 11, Change.NONE),
    (10, 14, Change.DETERIORATED),
])
def test_classify_change(x1, x2, expected):
    assert classify_change(x1, x2, threshold=4) is expected


def test_reliability_one_is_rejected():
    with pytest.raises(ValueError):
        rci_threshold(5, 1.0)
