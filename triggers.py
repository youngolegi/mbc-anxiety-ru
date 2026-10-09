"""Карта триггеров: записи эпизодов тревоги по схеме ABC и метрика прогресса паттерна.

Ключевое решение: прогресс паттерна — это снижение интенсивности и доли избегания
в одном и том же контексте, а НЕ снижение частоты. В хорошей терапии (экспозиция)
клиент перестаёт избегать, и встреч с триггером может стать больше.

Осторожно с интерпретацией: записи делаются, когда тревога заметна, поэтому
«частота контекста среди записей» — не риск этого контекста. И чем больше
измерений комбинируется в паттерн, тем выше шанс найти случайное совпадение
(проблема множественных сравнений). Паттерн — гипотеза, а не вывод.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from statistics import mean
from typing import Iterable, Sequence


class Reaction(Enum):
    STAYED = "остался в ситуации"
    LEFT = "ушёл (бегство)"
    AVOIDED = "избежал"
    SAFETY_BEHAVIOUR = "охранительное поведение"
    REASSURANCE = "искал перестраховки"
    SUPPORT = "обратился за поддержкой"


AVOIDANT_REACTIONS = frozenset({
    Reaction.LEFT, Reaction.AVOIDED, Reaction.SAFETY_BEHAVIOUR, Reaction.REASSURANCE,
})
"""Обращение за поддержкой к избеганию не относится; поиск перестраховки — относится."""

DIMENSIONS = ("where", "with_whom", "before")


@dataclass(frozen=True)
class TriggerEntry:
    id: str
    timestamp: datetime
    intensity: int
    """Интенсивность тревоги 0–10."""
    where: str | None = None
    with_whom: str | None = None
    before: str | None = None
    """Что было за час до: лента, новости, разговор, работа..."""
    thought: str = ""
    reaction: Reaction | None = None
    duration_min: int | None = None
    sleep_hours: float | None = None
    caffeine: bool | None = None
    alcohol_prev_day: bool | None = None
    unwell: bool | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.intensity <= 10:
            raise ValueError(f"Запись {self.id}: интенсивность должна быть 0–10")

    @property
    def avoidant(self) -> bool | None:
        if self.reaction is None:
            return None
        return self.reaction in AVOIDANT_REACTIONS


def somatic_confounders(entry: TriggerEntry, min_sleep_hours: float = 6.0) -> list[str]:
    """Телесные факторы, которые могли поднять тревогу независимо от ситуации.

    Порог сна 6 часов — эвристика, а не норма; настраивается.
    """
    found = []
    if entry.sleep_hours is not None and entry.sleep_hours < min_sleep_hours:
        found.append(f"сон {entry.sleep_hours:g} ч")
    if entry.caffeine:
        found.append("кофеин/энергетик")
    if entry.alcohol_prev_day:
        found.append("алкоголь накануне")
    if entry.unwell:
        found.append("недомогание")
    return found


@dataclass
class Pattern:
    dimensions: tuple[str, ...]
    values: tuple[str, ...]
    entry_ids: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return ", ".join(f"{d}={v}" for d, v in zip(self.dimensions, self.values))

    @property
    def count(self) -> int:
        return len(self.entry_ids)


def find_patterns(
    entries: Iterable[TriggerEntry],
    dimensions: Sequence[str] = DIMENSIONS,
    min_episodes: int = 3,
    combine: bool = False,
) -> list[Pattern]:
    """Повторяющиеся контексты.

    combine=False — каждое измерение отдельно (например, before=звонок матери).
    combine=True — точное совпадение по всем измерениям сразу; требует больше данных.
    Паттерн объявляется только при min_episodes эпизодах и больше.
    """
    for dim in dimensions:
        if dim not in DIMENSIONS:
            raise ValueError(f"Неизвестное измерение: {dim}. Доступны: {DIMENSIONS}")

    groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = defaultdict(list)
    for entry in entries:
        if combine:
            values = tuple(getattr(entry, d) for d in dimensions)
            if all(v is not None for v in values):
                groups[(tuple(dimensions), values)].append(entry.id)
        else:
            for d in dimensions:
                value = getattr(entry, d)
                if value is not None:
                    groups[((d,), (value,))].append(entry.id)

    patterns = [Pattern(dims, vals, ids) for (dims, vals), ids in groups.items()
                if len(ids) >= min_episodes]
    return sorted(patterns, key=lambda p: (-p.count, p.label))


@dataclass(frozen=True)
class WeekStat:
    week: str
    """ISO-неделя в виде '2026-W41'."""
    episodes: int
    mean_intensity: float
    avoidance_rate: float | None
    """Доля избегания среди записей, где реакция отмечена; None — если реакций нет."""


def weekly_progress(entries: Iterable[TriggerEntry], pattern: Pattern) -> list[WeekStat]:
    ids = set(pattern.entry_ids)
    by_week: dict[str, list[TriggerEntry]] = defaultdict(list)
    for entry in entries:
        if entry.id in ids:
            year, week, _ = entry.timestamp.isocalendar()
            by_week[f"{year}-W{week:02d}"].append(entry)

    stats = []
    for week in sorted(by_week):
        items = by_week[week]
        with_reaction = [e for e in items if e.avoidant is not None]
        rate = (sum(e.avoidant for e in with_reaction) / len(with_reaction)
                if with_reaction else None)
        stats.append(WeekStat(week, len(items), round(mean(e.intensity for e in items), 2), rate))
    return stats


def pattern_change(stats: Sequence[WeekStat]) -> dict[str, float | None]:
    """Разница между последней и первой неделей. Отрицательные значения — улучшение.

    Описательная величина без статистического вывода: для проверки значимости
    данных одного клиента обычно недостаточно.
    """
    if len(stats) < 2:
        return {"intensity_delta": None, "avoidance_delta": None}
    first, last = stats[0], stats[-1]
    avoidance_delta = (None if first.avoidance_rate is None or last.avoidance_rate is None
                       else round(last.avoidance_rate - first.avoidance_rate, 2))
    return {
        "intensity_delta": round(last.mean_intensity - first.mean_intensity, 2),
        "avoidance_delta": avoidance_delta,
    }
