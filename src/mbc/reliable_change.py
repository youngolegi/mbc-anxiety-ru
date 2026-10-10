"""Индекс надёжного изменения (Jacobson & Truax, 1991).

SEM    = SD × √(1 − r)
S_diff = √2 × SEM
порог  = z × S_diff          (z = 1,96 для 95%)
RCI    = (X2 − X1) / S_diff

Выбор r — методологическое решение: внутренняя согласованность (α) даёт
меньший порог, ретестовая надёжность — больший. Фиксируйте выбор и применяйте
его одинаково ко всем клиентам.
"""
from __future__ import annotations

import math
from enum import Enum


class Change(Enum):
    IMPROVED = "надёжное улучшение"
    DETERIORATED = "надёжное ухудшение"
    NONE = "нет надёжного изменения"


def _check(sd: float, reliability: float) -> None:
    if sd <= 0:
        raise ValueError("SD должно быть положительным")
    if not 0 <= reliability < 1:
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
    """Классификация изменения относительно порога в баллах."""
    if threshold <= 0:
        raise ValueError("Порог надёжного изменения должен быть положительным")
    improvement = (x1 - x2) if higher_is_worse else (x2 - x1)
    if improvement >= threshold:
        return Change.IMPROVED
    if improvement <= -threshold:
        return Change.DETERIORATED
    return Change.NONE
