"""Подсчёт суммарного балла с проверкой ответов."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .measures import Measure


class ScoringError(ValueError):
    """Некорректные ответы: не та длина, значения вне диапазона, не целые числа."""


@dataclass(frozen=True)
class Score:
    measure: str
    total: int | None
    """None — если пропусков больше допустимого."""
    n_answered: int
    prorated: bool
    severity: str | None


def score(measure: Measure, responses: Sequence[int | None], max_missing: int = 0) -> Score:
    """Суммарный балл.

    max_missing — сколько пропусков допустимо. По умолчанию 0: любой пропуск даёт total=None.
    При 0 < пропусков <= max_missing балл пересчитывается пропорционально
    (среднее по ответам × число пунктов). Это допущение, а не стандарт:
    включайте его осознанно и фиксируйте в отчёте.

    Округление арифметическое: ровно .5 — вверх, как ROUND в Excel (10.5 → 11).
    Встроенный round() округляет .5 к чётному: 10.5 → 10, но 17.5 → 18.
    """
    if len(responses) != measure.n_items:
        raise ScoringError(
            f"{measure.code}: ожидается {measure.n_items} ответов, получено {len(responses)}"
        )
    answered: list[int] = []
    for i, value in enumerate(responses, start=1):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ScoringError(f"{measure.code}, пункт {i}: ответ должен быть целым числом")
        if not measure.item_min <= value <= measure.item_max:
            raise ScoringError(
                f"{measure.code}, пункт {i}: {value} вне диапазона "
                f"{measure.item_min}–{measure.item_max}"
            )
        answered.append(value)

    missing = measure.n_items - len(answered)
    if missing > max_missing or not answered:
        return Score(measure.code, None, len(answered), False, None)

    if missing == 0:
        total = sum(answered)
        prorated = False
    else:
        # floor(сумма × пункты / ответы + 1/2) в целых числах — без ошибок плавающей точки
        total = (2 * sum(answered) * measure.n_items + len(answered)) // (2 * len(answered))
        prorated = True
    return Score(measure.code, total, len(answered), prorated, measure.severity(total))
