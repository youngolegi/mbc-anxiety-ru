import pytest

from mbc.measures import GAD7, OASIS, get_measure
from mbc.scoring import ScoringError, score


def test_full_responses_sum():
    result = score(GAD7, [1, 2, 3, 0, 1, 2, 1])
    assert result.total == 10
    assert result.severity == "умеренная"
    assert not result.prorated


def test_missing_item_strict_by_default():
    result = score(GAD7, [1, 2, None, 0, 1, 2, 1])
    assert result.total is None


def test_prorating_when_allowed():
    result = score(GAD7, [2, 2, None, 2, 2, 2, 2], max_missing=1)
    assert result.total == 14
    assert result.prorated


def test_wrong_length_raises():
    with pytest.raises(ScoringError):
        score(GAD7, [1, 2, 3])


def test_out_of_range_raises():
    with pytest.raises(ScoringError):
        score(GAD7, [4, 0, 0, 0, 0, 0, 0])


def test_oasis_range_is_0_to_4():
    assert score(OASIS, [4, 4, 4, 4, 4]).total == 20
    assert OASIS.max_total == 20


def test_bool_is_not_a_valid_answer():
    with pytest.raises(ScoringError):
        score(GAD7, [True, 0, 0, 0, 0, 0, 0])


def test_get_measure_accepts_dash():
    assert get_measure("gad-7") is GAD7


@pytest.mark.parametrize("answers,expected", [
    ([0, 1, 2, 2, 2, 2, None], 11),   # 9 / 6 × 7 = 10.5 → 11, а не 10 (round к чётному)
    ([3, 3, 3, 2, 2, 2, None], 18),   # 15 / 6 × 7 = 17.5 → 18
    ([1, 1, 1, 0, 0, 0, None], 4),    # 3 / 6 × 7 = 3.5 → 4
    ([2, 2, 2, 2, 1, 1, None], 12),   # 10 / 6 × 7 ≈ 11.67 → 12
    ([2, 2, 2, 1, 1, 1, None], 11),   # 9 / 6 × 7 = 10.5 → 11 (другая раскладка той же суммы)
])
def test_prorating_rounds_half_up(answers, expected):
    result = score(GAD7, answers, max_missing=1)
    assert result.total == expected and result.prorated
