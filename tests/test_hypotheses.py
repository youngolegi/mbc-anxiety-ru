import pytest

from mbc.hypotheses import Hypothesis, HypothesisStatus, evidence_for_pattern, verify_evidence
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


# --- сверка гипотезы с записями ---------------------------------------------------

def call_pattern():
    return find_patterns(ENTRIES, ("before",), min_episodes=3)[0]


def test_hypothesis_built_from_records_passes_verification():
    pattern = call_pattern()
    high, total = evidence_for_pattern(ENTRIES, pattern, high_intensity=7)
    h = make(evidence_ids=high, support=len(high), total=total)
    verify_evidence(h, ENTRIES)
    verify_evidence(h, ENTRIES, pattern)


def test_form_check_alone_accepts_nonexistent_record():
    # Hypothesis проверяет форму: несуществующая запись проходит, её ловит только сверка
    h = make(evidence_ids=["e99"], support=1, total=5)
    with pytest.raises(ValueError, match="нет таких записей: e99"):
        verify_evidence(h, ENTRIES)


def test_repeated_evidence_rejected():
    h = make(evidence_ids=["e1", "e1", "e2"], support=2, total=5)
    with pytest.raises(ValueError, match="записи указаны повторно: e1"):
        verify_evidence(h, ENTRIES)


def test_total_cannot_exceed_records_in_period():
    h = make(evidence_ids=["e1"], support=1, total=len(ENTRIES) + 1)
    with pytest.raises(ValueError, match=f"total = 8, а записей за период — {len(ENTRIES)}"):
        verify_evidence(h, ENTRIES)


def test_evidence_outside_pattern_rejected():
    h = make(evidence_ids=["e1", "e3"], support=2, total=5)  # e3 — «лента», не «звонок матери»
    with pytest.raises(ValueError, match="записи вне контекста «before=звонок матери»: e3"):
        verify_evidence(h, ENTRIES, call_pattern())


def test_total_must_match_pattern_size():
    h = make(evidence_ids=["e1", "e2"], support=2, total=4)
    with pytest.raises(ValueError, match="total = 4, а записей в контексте «before=звонок матери» — 5"):
        verify_evidence(h, ENTRIES, call_pattern())
    verify_evidence(h, ENTRIES)  # без контекста total = 4 допустим


def test_all_problems_reported_at_once():
    h = make(evidence_ids=["e1", "e99"], support=2, total=4)
    with pytest.raises(ValueError) as exc:
        verify_evidence(h, ENTRIES, call_pattern())
    assert "нет таких записей: e99" in str(exc.value)
    assert "записей в контексте" in str(exc.value)


def test_verification_does_not_change_hypothesis():
    h = make()
    verify_evidence(h, ENTRIES)
    assert h.status is HypothesisStatus.PROPOSED and h.evidence_ids == ["e1", "e2", "e4"]
