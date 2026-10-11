"""Объединённый статус по двум шкалам — флаги NHS Talking Therapies
(Clark, 2018, Annu Rev Clin Psychol 14:159–183). Все баллы синтетические.

GAD-7: «случай» ≥ 8, надёжное изменение 4. PHQ-9: «случай» ≥ 10, надёжное изменение 6.
"""
import itertools

import pytest

from mbc.measures import GAD7, PHQ8, PHQ9, WSAS
from mbc.reliable_change import Change
from mbc.status import ScalePair, combined_outcome

FLAGS = ("at_caseness_start", "recovery", "any_down", "any_up", "reliable_improvement",
         "reliable_deterioration", "mixed", "reliable_recovery")
T, F, N = True, False, None


def outcome_of(gad, phq, finished_course=True):
    return combined_outcome(ScalePair(GAD7, *gad), ScalePair(PHQ9, *phq),
                            finished_course=finished_course)


def flags(gad, phq):
    result = outcome_of(gad, phq)
    return tuple(getattr(result, name) for name in FLAGS)


# порядок флагов: старт-«случай», recovery, any_down, any_up, RI, RD, mixed, reliable_recovery
CASES = [
    ("обязательный", (9, 7), (8, 7), (T, T, F, F, F, F, F, F)),
    ("надёжное-по-обеим", (12, 5), (14, 6), (T, T, T, F, T, F, F, T)),
    ("хватает-одной-шкалы", (12, 5), (8, 6), (T, T, T, F, T, F, F, T)),
    ("улучшение-без-выздоровления", (18, 12), (20, 15), (T, F, T, F, T, F, F, F)),
    ("выздоровление-требует-обеих", (15, 6), (16, 12), (T, F, T, F, T, F, F, F)),
    ("нет-изменения", (12, 10), (14, 12), (T, F, F, F, F, F, F, F)),
    ("ухудшение", (8, 14), (10, 11), (T, F, F, T, F, T, F, F)),
    ("ухудшение-ниже-порога-не-маскируется", (2, 7), (3, 4), (F, N, F, T, F, T, F, N)),
    ("разнонаправленно", (15, 9), (8, 15), (T, F, T, T, F, F, T, F)),
    ("разнонаправленно-с-выздоровлением", (2, 6), (14, 7), (T, T, T, T, F, F, T, F)),
    ("не-случай-на-старте", (6, 3), (8, 4), (F, N, F, F, F, F, F, N)),
    ("улучшение-без-случая-на-старте", (7, 2), (9, 3), (F, N, T, F, T, F, F, N)),
    ("граница-старта-GAD-8", (8, 4), (9, 5), (T, T, T, F, T, F, F, T)),
    ("граница-старта-PHQ-10", (5, 4), (10, 4), (T, T, T, F, T, F, F, T)),
    ("граница-конца-GAD-8", (14, 8), (12, 6), (T, F, T, F, T, F, F, F)),
    ("граница-конца-PHQ-10", (12, 4), (16, 10), (T, F, T, F, T, F, F, F)),
]


@pytest.mark.parametrize("gad,phq,expected", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_flags(gad, phq, expected):
    assert flags(gad, phq) == expected


def expected_by_formula(g1, g2, p1, p2):
    """Формулы автора, записанные прямо на баллах — независимо от реализации."""
    start = g1 >= 8 or p1 >= 10
    recovery = (g2 < 8 and p2 < 10) if start else None
    down = g1 - g2 >= 4 or p1 - p2 >= 6
    up = g2 - g1 >= 4 or p2 - p1 >= 6
    improvement = down and not up
    reliable_recovery = (recovery and improvement) if start else None
    return (start, recovery, down, up, improvement, up and not down, down and up,
            reliable_recovery)


def test_flags_match_formulas_on_grid():
    # каждые 3 балла по обеим шкалам на старте и в конце: 8 × 8 × 10 × 10 = 6400 комбинаций
    for g1, g2, p1, p2 in itertools.product(range(0, 22, 3), range(0, 22, 3),
                                            range(0, 28, 3), range(0, 28, 3)):
        assert flags((g1, g2), (p1, p2)) == expected_by_formula(g1, g2, p1, p2), (g1, g2, p1, p2)


def test_changes_per_scale_are_exposed():
    assert outcome_of((15, 9), (8, 15)).changes == {"GAD7": Change.IMPROVED,
                                                     "PHQ9": Change.DETERIORATED}


def test_interim_when_course_not_finished():
    final = outcome_of((12, 5), (14, 6), finished_course=True)
    interim = outcome_of((12, 5), (14, 6), finished_course=False)
    assert not final.interim and interim.interim
    assert [getattr(interim, f) for f in FLAGS] == [getattr(final, f) for f in FLAGS]


def test_finished_course_is_required():
    with pytest.raises(TypeError):
        combined_outcome(ScalePair(GAD7, 12, 5), ScalePair(PHQ9, 14, 6))


@pytest.mark.parametrize("value", [1, 0, None, "да"])
def test_finished_course_must_be_bool(value):
    with pytest.raises(TypeError):
        outcome_of((12, 5), (14, 6), finished_course=value)


def test_phq8_requires_explicit_threshold():
    # правило 3: у PHQ-8 нет порога надёжного изменения — чужой не подставляется молча
    with pytest.raises(ValueError, match="rci_threshold"):
        combined_outcome(ScalePair(GAD7, 12, 5), ScalePair(PHQ8, 14, 6), finished_course=True)
    result = combined_outcome(ScalePair(GAD7, 12, 5), ScalePair(PHQ8, 14, 6, threshold=6),
                              finished_course=True)
    assert result.reliable_recovery is True


def test_measure_without_caseness_rejected():
    with pytest.raises(ValueError, match="порог «случая»"):
        combined_outcome(ScalePair(GAD7, 12, 5), ScalePair(WSAS, 30, 10, threshold=8),
                         finished_course=True)


def test_same_measure_twice_rejected():
    with pytest.raises(ValueError, match="разные шкалы"):
        combined_outcome(ScalePair(GAD7, 12, 5), ScalePair(GAD7, 14, 6), finished_course=True)


@pytest.mark.parametrize("gad", [(22, 5), (12, -1), (12.5, 5), (True, 5)])
def test_invalid_scores_rejected(gad):
    with pytest.raises(ValueError):
        outcome_of(gad, (14, 6))
