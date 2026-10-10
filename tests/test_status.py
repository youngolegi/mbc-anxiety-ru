import pytest

from mbc.measures import GAD7, OASIS
from mbc.status import Outcome, outcome


@pytest.mark.parametrize("x1,x2,expected", [
    (14, 7, Outcome.RECOVERED),        # надёжное улучшение и ниже порога 8
    (18, 12, Outcome.IMPROVED),        # надёжное улучшение, но всё ещё >= 8
    (14, 12, Outcome.NO_CHANGE),
    (10, 15, Outcome.DETERIORATED),
    (5, 2, Outcome.NOT_CASE_AT_START),
    (5, 10, Outcome.DETERIORATED),     # ухудшение важно даже ниже порога
])
def test_gad7_outcomes(x1, x2, expected):
    assert outcome(GAD7, x1, x2) is expected


def test_recovery_requires_dropping_below_caseness():
    assert outcome(GAD7, 12, 8) is Outcome.IMPROVED   # 8 — ещё «случай»


def test_measure_without_threshold_requires_explicit_one():
    with pytest.raises(ValueError, match="rci_threshold"):
        outcome(OASIS, 12, 6)
    assert outcome(OASIS, 12, 6, threshold=5) is Outcome.RECOVERED


def test_recovered_label_is_reliable_recovery():
    # RECOVERED = надёжное улучшение + ниже порога: в терминах NHS это reliable recovery
    assert Outcome.RECOVERED.value == "надёжное выздоровление"
