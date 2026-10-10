"""Клинический статус по логике NHS Talking Therapies (на уровне одной шкалы).

outcome() — статус по одной шкале. RECOVERED — «надёжное выздоровление» (reliable
recovery в терминах NHS): надёжное улучшение и балл ниже порога «случая».

combined_outcome() — статус по двум шкалам сразу (GAD-7 и PHQ-9), как в NHS:
независимые флаги, а не одна категория (Clark, 2018, Annu Rev Clin Psychol 14:159–183).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .measures import Measure
from .reliable_change import Change, classify_change


class Outcome(Enum):
    RECOVERED = "надёжное выздоровление"
    IMPROVED = "надёжное улучшение"
    NO_CHANGE = "нет надёжного изменения"
    DETERIORATED = "надёжное ухудшение"
    NOT_CASE_AT_START = "на старте ниже клинического порога"


def resolve_threshold(measure: Measure, threshold: float | None) -> float:
    thr = threshold if threshold is not None else measure.reliable_change
    if thr is None:
        raise ValueError(
            f"{measure.code}: нет порога надёжного изменения. "
            f"Рассчитайте его функцией rci_threshold() на своих данных. "
            f"Справка: {measure.reliable_change_source}"
        )
    return thr


def outcome(measure: Measure, x1: int, x2: int, threshold: float | None = None) -> Outcome:
    thr = resolve_threshold(measure, threshold)
    change = classify_change(x1, x2, thr, measure.higher_is_worse)

    if change is Change.DETERIORATED:
        return Outcome.DETERIORATED
    if measure.caseness is not None and x1 < measure.caseness:
        return Outcome.NOT_CASE_AT_START
    if change is Change.IMPROVED:
        if measure.caseness is not None and x2 < measure.caseness:
            return Outcome.RECOVERED
        return Outcome.IMPROVED
    return Outcome.NO_CHANGE


@dataclass(frozen=True)
class ScalePair:
    """Баллы по одной шкале на старте и в конце курса."""
    measure: Measure
    start: int
    end: int
    threshold: float | None = None
    """Порог надёжного изменения; None — порог шкалы, а если его нет — ошибка."""


@dataclass(frozen=True)
class CombinedOutcome:
    """Флаги NHS Talking Therapies по двум шкалам.

    Хранятся только факты; флаги вычисляются из них по формулам и разойтись не могут.
    """
    changes: dict[str, Change]
    """Надёжное изменение по каждой шкале: код шкалы → Change."""
    at_caseness_start: bool
    """На старте «случай» хотя бы по одной шкале."""
    below_caseness_end: bool
    """В конце обе шкалы ниже порога «случая»."""
    interim: bool
    """Курс не завершён: статус промежуточный, не сопоставим со статистикой NHS."""

    @property
    def any_down(self) -> bool:
        return Change.IMPROVED in self.changes.values()

    @property
    def any_up(self) -> bool:
        return Change.DETERIORATED in self.changes.values()

    @property
    def reliable_improvement(self) -> bool:
        return self.any_down and not self.any_up

    @property
    def reliable_deterioration(self) -> bool:
        return self.any_up and not self.any_down

    @property
    def mixed(self) -> bool:
        return self.any_down and self.any_up

    @property
    def recovery(self) -> bool | None:
        """None — на старте не «случай»: выздоровление не определяется."""
        return self.below_caseness_end if self.at_caseness_start else None

    @property
    def reliable_recovery(self) -> bool | None:
        if self.recovery is None:
            return None
        return self.recovery and self.reliable_improvement


def combined_outcome(anxiety: ScalePair, depression: ScalePair, *,
                     finished_course: bool) -> CombinedOutcome:
    """Статус по двум шкалам — независимые флаги, как в NHS Talking Therapies.

    Сначала надёжное изменение по каждой шкале, потом флаги по формулам. Ни один флаг
    не перекрывает другой: в отличие от outcome(), ухудшение не проверяется первым,
    а «на старте ниже порога» не маскирует остальные флаги.

    finished_course обязателен: NHS считает статус по завершённым курсам, поэтому
    при False результат промежуточный (interim=True) и с их статистикой не сопоставим.
    """
    if not isinstance(finished_course, bool):
        raise TypeError("finished_course должен быть True или False")
    if anxiety.measure.code == depression.measure.code:
        raise ValueError(f"Нужны разные шкалы, а обе — {anxiety.measure.code}")
    pairs = (anxiety, depression)
    for pair in pairs:
        _check_pair(pair)
    changes = {
        p.measure.code: classify_change(p.start, p.end, resolve_threshold(p.measure, p.threshold),
                                        p.measure.higher_is_worse)
        for p in pairs
    }
    return CombinedOutcome(
        changes=changes,
        at_caseness_start=any(p.start >= p.measure.caseness for p in pairs),
        below_caseness_end=all(p.end < p.measure.caseness for p in pairs),
        interim=not finished_course,
    )


def _check_pair(pair: ScalePair) -> None:
    m = pair.measure
    if m.caseness is None:
        raise ValueError(f"{m.code}: для статуса по двум шкалам нужен порог «случая». "
                         f"Справка: {m.caseness_source}")
    for value in (pair.start, pair.end):
        if isinstance(value, bool) or not isinstance(value, int) \
                or not m.min_total <= value <= m.max_total:
            raise ValueError(f"{m.code}: балл {value!r} — не целое число "
                             f"в диапазоне {m.min_total}–{m.max_total}")
