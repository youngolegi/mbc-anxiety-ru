"""Порог надёжного изменения по выборке. Все выборки синтетические.

Ручной расчёт для GAD-7 [0, 5, 10]: среднее 5, SD (n − 1) = 5;
ω = 0,92 → SEM = 5·√0,08 ≈ 1,414; S_diff = √2·SEM = 2,0; порог = 1,96·2,0 = 3,92 → 4 балла.
"""
import itertools
import math

import pytest

from mbc.measures import GAD7, OASIS
from mbc.reliable_change import (Change, ReliabilityKind, classify_change, min_change_points,
                                 rci_threshold, threshold_from_sample)
from mbc.report import build_report, render_text
from mbc.scoring import Score, score

OMEGA, ALPHA, RETEST = ReliabilityKind.OMEGA, ReliabilityKind.ALPHA, ReliabilityKind.RETEST


def full(total, measure=GAD7):
    """Полный замер (без пропусков) с заданным суммарным баллом."""
    return Score(measure.code, total, measure.n_items, False, measure.severity(total))


SAMPLE = [full(0), full(5), full(10)]


# --- округление порога до баллов ---------------------------------------------------

@pytest.mark.parametrize("threshold,points", [
    (3.92, 4), (4.0, 4), (4.01, 5), (4.000000000000001, 4),
])
def test_min_change_points(threshold, points):
    assert min_change_points(threshold) == points


@pytest.mark.parametrize("threshold", [0, -1])
def test_min_change_points_rejects_non_positive(threshold):
    with pytest.raises(ValueError):
        min_change_points(threshold)


NOISY = [4.000000000000001, 3.999999999999999, 6.000000000000001, 5.999999999999999]


@pytest.mark.parametrize("higher_is_worse", [True, False])
def test_fractional_and_rounded_thresholds_classify_integer_changes_alike(higher_is_worse):
    # свойство: для целых изменений дробный порог и порог в баллах дают одно и то же
    thresholds = [k / 100 for k in range(1, 1001)] + NOISY
    for thr, delta in itertools.product(thresholds, range(-30, 31)):
        x1, x2 = 30, 30 - delta
        assert classify_change(x1, x2, thr, higher_is_worse) is \
            classify_change(x1, x2, min_change_points(thr), higher_is_worse), (thr, delta)


def test_classify_change_tolerates_float_noise():
    # порог «ровно 6» из расчёта приходит как 6.000000000000001 — изменение на 6 надёжно
    assert classify_change(10, 4, 6.000000000000001) is Change.IMPROVED
    assert classify_change(4, 10, 6.000000000000001) is Change.DETERIORATED


# --- расчёт по выборке ---------------------------------------------------------------

def test_hand_calculation():
    t = threshold_from_sample(SAMPLE, 0.92, kind=OMEGA, reliability_n=3)
    assert (t.measure, t.n, t.excluded, t.mean, t.sd) == ("GAD7", 3, 0, 5.0, 5.0)
    assert t.sem == pytest.approx(1.4142, rel=1e-3)
    assert t.s_diff == pytest.approx(2.0)
    assert t.threshold == pytest.approx(3.92)
    assert t.threshold == pytest.approx(rci_threshold(5, 0.92))
    assert t.min_change_points == 4


def test_sample_sd_uses_n_minus_1():
    # выборочное SD (n − 1): 5,0; «генеральное» (n) дало бы ≈ 4,08 и порог меньше
    assert threshold_from_sample(SAMPLE, 0.92, kind=OMEGA, reliability_n=3).sd == 5.0


def test_n_is_a_numeric_field():
    t = threshold_from_sample(SAMPLE, 0.92, kind=OMEGA, reliability_n=3)
    assert isinstance(t.n, int) and t.n == 3


def test_retest_gives_larger_threshold_than_omega():
    omega = threshold_from_sample(SAMPLE, 0.92, kind=OMEGA, reliability_n=3)
    retest = threshold_from_sample(SAMPLE, 0.83, kind=RETEST, reliability_n=3)
    assert retest.threshold > omega.threshold * 1.4
    assert (omega.min_change_points, retest.min_change_points) == (4, 6)


def test_computed_exact_threshold_is_not_inflated_by_noise():
    # r = 0,82, z = 2: порог ровно 6 (SEM = 5·√0,18, S_diff = 3), в плавающей точке — чуть больше
    t = threshold_from_sample(SAMPLE, 0.82, kind=OMEGA, reliability_n=3, z=2)
    assert t.threshold == pytest.approx(6.0) and t.threshold > 6
    assert t.min_change_points == 6
    assert classify_change(14, 8, t.threshold) is Change.IMPROVED


def test_incomplete_and_prorated_scores_are_excluded_and_counted():
    prorated = score(GAD7, [2, 2, None, 2, 2, 2, 2], max_missing=1)   # 14, пересчитан
    missing = score(GAD7, [2, 2, None, 2, 2, 2, 2])                    # не засчитан
    t = threshold_from_sample(SAMPLE + [prorated, missing], 0.92, kind=OMEGA, reliability_n=3)
    assert (t.n, t.excluded, t.sd) == (3, 2, 5.0)
    assert "исключено неполных: 2" in t.source


def test_source_records_every_choice():
    source = threshold_from_sample(SAMPLE, 0.92, kind=OMEGA, reliability_n=3).source
    for part in ("по выборке", "n = 3", "SD = 5", "ω наблюдаемой суммы = 0.92", "z = 1.96"):
        assert part in source
    assert "исключено" not in source


def test_iterator_input_is_not_lost():
    assert threshold_from_sample(iter(SAMPLE), 0.92, kind=OMEGA, reliability_n=3) == \
        threshold_from_sample(SAMPLE, 0.92, kind=OMEGA, reliability_n=3)


def test_kind_is_required():
    with pytest.raises(TypeError):
        threshold_from_sample(SAMPLE, 0.92, reliability_n=3)


@pytest.mark.parametrize("kind", ["omega", None, 1])
def test_kind_must_be_reliability_kind(kind):
    with pytest.raises(TypeError):
        threshold_from_sample(SAMPLE, 0.92, kind=kind, reliability_n=3)


# --- одна выборка для SD и надёжности -------------------------------------------------

def test_reliability_n_is_required():
    with pytest.raises(TypeError):
        threshold_from_sample(SAMPLE, 0.92, kind=OMEGA)


@pytest.mark.parametrize("kind", [OMEGA, ALPHA])
def test_internal_consistency_must_come_from_the_same_sample(kind):
    with pytest.raises(ValueError, match="на другой выборке"):
        threshold_from_sample(SAMPLE, 0.92, kind=kind, reliability_n=120)


def test_same_sample_means_complete_scores_after_exclusion():
    # 3 полных + 2 неполных: коэффициент считается на тех же 3 полных, а не на 5
    sample = SAMPLE + [score(GAD7, [None] * 7), score(GAD7, [2, 2, None, 2, 2, 2, 2], max_missing=1)]
    with pytest.raises(ValueError, match="на другой выборке"):
        threshold_from_sample(sample, 0.92, kind=OMEGA, reliability_n=5)
    assert threshold_from_sample(sample, 0.92, kind=OMEGA, reliability_n=3).same_sample


def test_retest_from_another_sample_is_allowed_and_marked():
    t = threshold_from_sample(SAMPLE, 0.83, kind=RETEST, reliability_n=50)
    assert not t.same_sample and t.reliability_n == 50
    assert "ICC ретеста = 0.83 на другой выборке (n = 50)" in t.source


def test_retest_from_the_same_sample_is_not_marked():
    t = threshold_from_sample(SAMPLE, 0.83, kind=RETEST, reliability_n=3)
    assert t.same_sample and "другой выборке" not in t.source


@pytest.mark.parametrize("value,error", [
    (1, ValueError), (0, ValueError), (True, TypeError), (2.5, TypeError), ("3", TypeError),
])
def test_invalid_reliability_n_rejected(value, error):
    with pytest.raises(error):
        threshold_from_sample(SAMPLE, 0.92, kind=RETEST, reliability_n=value)


def test_ordinal_omega_is_not_offered():
    # ординальный ω описывает не наблюдаемую сумму — для порога не используется (Chalmers, 2018)
    assert {k.name for k in ReliabilityKind} == {"OMEGA", "ALPHA", "RETEST"}


@pytest.mark.parametrize("sample", [
    [],
    [full(5)],                                                          # SD не определено
    [full(5), full(5), full(5)],                                        # SD = 0
    [full(5), score(GAD7, [None] * 7), score(GAD7, [None] * 7)],        # полный только один
], ids=["пусто", "один", "без-разброса", "один-полный"])
def test_too_few_or_degenerate_samples_rejected(sample):
    with pytest.raises(ValueError):
        threshold_from_sample(sample, 0.92, kind=OMEGA, reliability_n=3)


@pytest.mark.parametrize("bad", [
    5,                                        # не Score
    None,
    Score("GAD7", 22, 7, False, None),        # вне диапазона GAD-7
    Score("GAD7", 4.5, 7, False, None),       # не целое
    Score("GAD7", True, 7, False, None),      # bool
], ids=["число", "None", "вне-диапазона", "дробный", "bool"])
def test_invalid_items_rejected(bad):
    with pytest.raises(ValueError):
        threshold_from_sample(SAMPLE + [bad], 0.92, kind=OMEGA, reliability_n=3)


def test_mixed_measures_rejected():
    with pytest.raises(ValueError, match="разные шкалы"):
        threshold_from_sample(SAMPLE + [full(10, OASIS)], 0.92, kind=OMEGA, reliability_n=3)


@pytest.mark.parametrize("reliability", [1.0, -0.1, math.nan, False])
def test_invalid_reliability_rejected(reliability):
    with pytest.raises(ValueError):
        threshold_from_sample(SAMPLE, reliability, kind=OMEGA, reliability_n=3)


@pytest.mark.parametrize("z", [0, -1.96, math.nan, True])
def test_invalid_z_rejected(z):
    with pytest.raises(ValueError):
        threshold_from_sample(SAMPLE, 0.92, kind=OMEGA, reliability_n=3, z=z)


# --- связка с отчётом -----------------------------------------------------------------

OASIS_SAMPLE = [full(t, OASIS) for t in (6, 9, 12, 15, 18)]
OASIS_CLIENT = [(1, score(OASIS, [3, 3, 2, 2, 2])), (4, score(OASIS, [1] * 5))]


def test_report_prints_sample_size_and_source():
    t = threshold_from_sample(OASIS_SAMPLE, 0.85, kind=OMEGA, reliability_n=5)
    lines = render_text(build_report(OASIS, OASIS_CLIENT, threshold=t)).splitlines()
    assert any(line.startswith(f"  порог надёжного изменения: {t.threshold:g} — {t.source}")
               for line in lines)
    assert "  порог рассчитан на выборке n = 5" in lines


def test_report_rejects_second_source_for_computed_threshold():
    t = threshold_from_sample(OASIS_SAMPLE, 0.85, kind=OMEGA, reliability_n=5)
    with pytest.raises(ValueError, match="источник уже есть"):
        build_report(OASIS, OASIS_CLIENT, threshold=t, threshold_source="другой")


def test_report_rejects_threshold_of_another_measure():
    t = threshold_from_sample(SAMPLE, 0.92, kind=OMEGA, reliability_n=3)   # GAD-7
    with pytest.raises(ValueError, match="рассчитан для GAD7"):
        build_report(OASIS, OASIS_CLIENT, threshold=t)


def test_plain_threshold_has_no_sample_line():
    text = render_text(build_report(OASIS, OASIS_CLIENT, threshold=5, threshold_source="синтетика"))
    assert "рассчитан на выборке" not in text
