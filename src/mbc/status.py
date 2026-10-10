"""Клинический статус по логике NHS Talking Therapies (на уровне одной шкалы).

В NHS «выздоровление» засчитывается по двум шкалам сразу (GAD-7 и PHQ-9);
здесь статус считается отдельно по каждой — объединение делает вызывающий код.
"""
from __future__ import annotations

from enum import Enum

from .measures import Measure
from .reliable_change import Change, classify_change


class Outcome(Enum):
    RECOVERED = "выздоровление"
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
