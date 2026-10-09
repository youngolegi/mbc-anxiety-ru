"""Сигнал траектории — уровень 4а (правило, работает с первого клиента).

Это заглушка, а не модель: правило не валидировано. По мере накопления
завершённых случаев его заменяют кривые ожидаемого ответа по стартовой тяжести
(уровень 4б), а затем персональный прогноз по похожим клиентам (уровень 4в).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .measures import Measure
from .reliable_change import Change, classify_change
from .status import resolve_threshold


class Signal(Enum):
    GREEN = "на траектории"
    AMBER = "отстаёт от ожидаемого"
    RED = "надёжное ухудшение"
    INSUFFICIENT = "недостаточно данных"


@dataclass(frozen=True)
class TrajectoryFlag:
    signal: Signal
    reason: str
    session: int | None


def rule_based_flag(
    measure: Measure,
    scores: Sequence[tuple[int, int]],
    threshold: float | None = None,
    check_by_session: int = 4,
) -> TrajectoryFlag:
    """scores — пары (номер сессии, балл). Первая по номеру сессия — базовая линия."""
    if len(scores) < 2:
        return TrajectoryFlag(Signal.INSUFFICIENT, "нужно минимум два замера", None)

    ordered = sorted(scores)
    (_, baseline), (last_session, latest) = ordered[0], ordered[-1]
    thr = resolve_threshold(measure, threshold)
    change = classify_change(baseline, latest, thr, measure.higher_is_worse)

    if change is Change.DETERIORATED:
        return TrajectoryFlag(
            Signal.RED,
            f"балл {latest} против {baseline} на старте: ухудшение не меньше порога {thr:g}",
            last_session,
        )

    started_as_case = measure.caseness is None or baseline >= measure.caseness
    if last_session >= check_by_session and started_as_case and change is not Change.IMPROVED:
        return TrajectoryFlag(
            Signal.AMBER,
            f"к сессии {last_session} нет надёжного улучшения (порог {thr:g})",
            last_session,
        )

    return TrajectoryFlag(Signal.GREEN, "отклонений от правила нет", last_session)
