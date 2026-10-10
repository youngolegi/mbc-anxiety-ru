"""Индекс надёжного изменения (Jacobson & Truax, 1991).

SEM    = SD × √(1 − r)
S_diff = √2 × SEM
порог  = z × S_diff          (z = 1,96 для 95%)
RCI    = (X2 − X1) / S_diff

Выбор r — методологическое решение: внутренняя согласованность (α, ω) даёт
меньший порог, ретестовая надёжность — больший. Фиксируйте выбор и применяйте
его одинаково ко всем клиентам.

Порог по своей выборке — threshold_from_sample(). Правила расчёта:
- SD и надёжность считаются на одной и той же выборке, иначе SEM неверна;
- в выборку идут только полные замеры, пересчитанные из-за пропусков исключаются;
- SD выборочное (n − 1);
- для OASIS основной коэффициент — ω наблюдаемой суммы баллов (по ковариациям сырых
  баллов), ICC ретеста — анализ чувствительности (пререгистрация, раздел 6).
  Ординальный ω по полихорическим корреляциям описывает не наблюдаемую сумму и для
  порога не используется (Chalmers, 2018) — поэтому его нет среди ReliabilityKind.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import mean, stdev
from typing import Iterable

from .measures import get_measure
from .scoring import Score

_EPS = 1e-9
"""Допуск на шум плавающей точки: порог 6.000000000000001 — это 6."""


class Change(Enum):
    IMPROVED = "надёжное улучшение"
    DETERIORATED = "надёжное ухудшение"
    NONE = "нет надёжного изменения"


class ReliabilityKind(Enum):
    OMEGA = "ω наблюдаемой суммы"
    ALPHA = "α Кронбаха"
    RETEST = "ICC ретеста"


def _check(sd: float, reliability: float) -> None:
    if sd <= 0:
        raise ValueError("SD должно быть положительным")
    if isinstance(reliability, bool) or not 0 <= reliability < 1:
        raise ValueError("Надёжность должна быть в диапазоне [0, 1)")


def standard_error_of_measurement(sd: float, reliability: float) -> float:
    _check(sd, reliability)
    return sd * math.sqrt(1 - reliability)


def standard_error_of_difference(sd: float, reliability: float) -> float:
    return math.sqrt(2) * standard_error_of_measurement(sd, reliability)


def rci_threshold(sd: float, reliability: float, z: float = 1.96) -> float:
    """Минимальное изменение в баллах, которое больше ошибки измерения."""
    return z * standard_error_of_difference(sd, reliability)


def reliable_change_index(x1: float, x2: float, sd: float, reliability: float) -> float:
    return (x2 - x1) / standard_error_of_difference(sd, reliability)


def classify_change(x1: float, x2: float, threshold: float,
                    higher_is_worse: bool = True) -> Change:
    """Классификация изменения относительно порога в баллах.

    Сравнение с допуском 1e-9: для целых изменений результат совпадает
    с классификацией по min_change_points(threshold).
    """
    if threshold <= 0:
        raise ValueError("Порог надёжного изменения должен быть положительным")
    improvement = (x1 - x2) if higher_is_worse else (x2 - x1)
    if improvement >= threshold - _EPS:
        return Change.IMPROVED
    if improvement <= -(threshold - _EPS):
        return Change.DETERIORATED
    return Change.NONE


def min_change_points(threshold: float) -> int:
    """Минимальное целое изменение, которое считается надёжным: порог, округлённый вверх.

    ceil(порог − 1e-9): 3.92 → 4, 4.0 → 4, 4.01 → 5, а шум 4.000000000000001 → 4.
    """
    if threshold <= 0:
        raise ValueError("Порог надёжного изменения должен быть положительным")
    return math.ceil(threshold - _EPS)


@dataclass(frozen=True)
class SampleThreshold:
    """Порог надёжного изменения, рассчитанный по выборке, со всеми исходными числами."""
    measure: str
    n: int
    """Число полных замеров, по которым посчитано SD."""
    excluded: int
    """Исключено неполных замеров (пропуски, пересчёт)."""
    mean: float
    sd: float
    reliability: float
    kind: ReliabilityKind
    z: float
    sem: float
    s_diff: float
    threshold: float

    @property
    def min_change_points(self) -> int:
        return min_change_points(self.threshold)

    @property
    def source(self) -> str:
        excluded = f", исключено неполных: {self.excluded}" if self.excluded else ""
        return (f"рассчитан по выборке: n = {self.n}{excluded}, SD = {self.sd:.3g}, "
                f"{self.kind.value} = {self.reliability:g}, z = {self.z:g}")


def threshold_from_sample(scores: Iterable[Score], reliability: float, *,
                          kind: ReliabilityKind, z: float = 1.96) -> SampleThreshold:
    """Порог надёжного изменения по выборке первых замеров.

    scores — первые замеры одной шкалы, по одному на человека. Неполные (total=None)
    и пересчитанные из-за пропусков (prorated) исключаются и считаются в excluded.
    reliability — коэффициент, посчитанный на той же выборке полных замеров;
    kind — его вид, обязателен: выбор коэффициента не должен быть неявным.
    Жёсткого минимума n нет (кроме n ≥ 2), но n всегда виден в результате.
    """
    if not isinstance(kind, ReliabilityKind):
        raise TypeError("kind — значение ReliabilityKind")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0:
        raise ValueError("z должно быть положительным конечным числом")
    scores = tuple(scores)
    for sc in scores:
        if not isinstance(sc, Score):
            raise ValueError(f"Выборка — замеры Score, получено {sc!r}")
    codes = {sc.measure for sc in scores}
    if len(codes) > 1:
        raise ValueError(f"В выборке разные шкалы: {', '.join(sorted(codes))}")
    complete = [sc.total for sc in scores if sc.total is not None and not sc.prorated]
    excluded = len(scores) - len(complete)
    if len(complete) < 2:
        raise ValueError(f"Нужно минимум два полных замера, есть {len(complete)} "
                         f"(исключено неполных: {excluded})")
    measure = get_measure(codes.pop())
    for total in complete:
        if isinstance(total, bool) or not isinstance(total, int) \
                or not measure.min_total <= total <= measure.max_total:
            raise ValueError(f"{measure.code}: балл {total!r} — не целое число "
                             f"в диапазоне {measure.min_total}–{measure.max_total}")
    sd = stdev(complete)
    sem = standard_error_of_measurement(sd, reliability)
    s_diff = math.sqrt(2) * sem
    return SampleThreshold(measure.code, len(complete), excluded, mean(complete), sd,
                           reliability, kind, z, sem, s_diff, z * s_diff)
