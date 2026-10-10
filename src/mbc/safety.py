"""Проверка ответов, требующих протокола безопасности.

Модуль только поднимает сигнал. Он не оценивает риск и не заменяет клиническую
оценку: при сигнале терапевт действует по своему протоколу, а клиенту сразу
показываются ресурсы экстренной помощи (их содержание задаётся конфигурацией
сервиса, а не этим модулем).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .measures import Measure


@dataclass(frozen=True)
class SafetyAlert:
    measure: str
    item: int
    value: int | None
    message: str


def check_safety(measure: Measure, responses: Sequence[int | None]) -> list[SafetyAlert]:
    alerts: list[SafetyAlert] = []
    for item in measure.safety_items:
        value = responses[item - 1]
        if value is None:
            alerts.append(SafetyAlert(
                measure.code, item, None,
                "Пункт о мыслях о смерти или самоповреждении пропущен — уточнить на сессии.",
            ))
        elif value > 0:
            alerts.append(SafetyAlert(
                measure.code, item, value,
                "Положительный ответ на пункт о мыслях о смерти или самоповреждении: "
                "требуется оценка риска терапевтом по протоколу.",
            ))
    return alerts
