import pytest

from mbc.hypotheses import Hypothesis, HypothesisStatus, evidence_for_pattern
from mbc.triggers import find_patterns

from test_triggers import ENTRIES


def make(**overrides):
    data = dict(statement="Тревога выше после звонков матери",
                evidence_ids=["e1", "e2", "e4"], support=3, total=5,
                alternative="Звонки чаще в дни недосыпа",
                check_question="Что обычно обсуждается в этих разговорах?")
    data.update(overrides)
    return Hypothesis(**data)


def test_valid_hypothesis_renders():
    text = make().render()
    assert "3 из 5" in text and "Альтернатива" in text


@pytest.mark.parametrize("field", ["statement", "alternative", "check_question"])
def test_required_text_fields(field):
    with pytest.raises(ValueError):
        make(**{field: "  "})


def test_support_must_match_evidence():
    with pytest.raises(ValueError):
        make(support=2)


def test_decision_only_once():
    h = make()
    h.reject("контекст не подтвердился")
    assert h.status is HypothesisStatus.REJECTED
    with pytest.raises(ValueError):
        h.confirm()


def test_evidence_for_pattern():
    pattern = find_patterns(ENTRIES, ("before",), min_episodes=3)[0]
    high, total = evidence_for_pattern(ENTRIES, pattern, high_intensity=7)
    assert high == ["e1", "e2", "e4"] and total == 5
