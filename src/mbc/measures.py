"""Реестр шкал, с которыми работает движок.

Принципы:
1. В коде нет текстов пунктов — только структура и пороги. GAD-7 и PHQ-9
   свободны для использования, но для OASIS и WSAS разрешение правообладателя
   ещё не получено (этап 0 по ITC), поэтому репозиторий не хранит формулировки
   ни одной шкалы.
2. Каждый порог снабжён источником. Пороги из англоязычных выборок — временные:
   для русских версий их нужно пересчитать на русских данных
   (флаг verified_ru_thresholds).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Measure:
    code: str
    name_ru: str
    n_items: int
    item_min: int
    item_max: int
    higher_is_worse: bool = True
    caseness: int | None = None
    """Клинический порог: балл >= caseness считается «случаем»."""
    caseness_source: str = ""
    reliable_change: int | None = None
    """Изменение в баллах, которое считается надёжным (больше ошибки измерения)."""
    reliable_change_source: str = ""
    severity_bands: tuple[tuple[int, str], ...] = ()
    """Пары (нижняя граница, метка), по возрастанию."""
    severity_source: str = ""
    """Источник границ тяжести: это тоже пороги."""
    safety_items: tuple[int, ...] = ()
    """Номера пунктов (с 1), положительный ответ на которые требует протокола безопасности."""
    russian_version: str = ""
    verified_ru_thresholds: bool = False
    rater: str = "client"
    """Кто заполняет: «client» — самоотчёт, «clinician» — оценка терапевта."""

    @property
    def min_total(self) -> int:
        return self.n_items * self.item_min

    @property
    def max_total(self) -> int:
        return self.n_items * self.item_max

    def severity(self, total: int) -> str | None:
        """Метка тяжести по диапазонам; None, если диапазоны не заданы."""
        label = None
        for lower, name in self.severity_bands:
            if total >= lower:
                label = name
        return label


GAD7 = Measure(
    code="GAD7",
    name_ru="Шкала генерализованного тревожного расстройства (GAD-7)",
    n_items=7, item_min=0, item_max=3,
    caseness=8,
    caseness_source=("NHS Talking Therapies (сервисный порог; "
                     "классический скрининговый — 10, Spitzer et al., 2006)"),
    reliable_change=4,
    reliable_change_source="NHS Talking Therapies",
    severity_bands=((0, "минимальная"), (5, "лёгкая"), (10, "умеренная"), (15, "тяжёлая")),
    severity_source="Spitzer et al., 2006",
    russian_version="Золотарева А.А., 2023, Консультативная психология и психотерапия 31(4):31–46",
)

PHQ9 = Measure(
    code="PHQ9",
    name_ru="Опросник здоровья пациента-9 (PHQ-9)",
    n_items=9, item_min=0, item_max=3,
    caseness=10,
    caseness_source="NHS Talking Therapies",
    reliable_change=6,
    reliable_change_source="NHS Talking Therapies",
    severity_bands=((0, "минимальная"), (5, "лёгкая"), (10, "умеренная"),
                    (15, "умеренно тяжёлая"), (20, "тяжёлая")),
    severity_source="Kroenke et al., 2001",
    safety_items=(9,),
    russian_version="используется; конкретную адаптацию выбрать",
)

PHQ8 = Measure(
    code="PHQ8",
    name_ru="Опросник здоровья пациента-8 (PHQ-8, без пункта о суицидальных мыслях)",
    n_items=8, item_min=0, item_max=3,
    caseness=10,
    caseness_source="Kroenke et al., 2009 — сверить с первоисточником",
    reliable_change=None,
    reliable_change_source=("не установлен — рассчитать threshold_from_sample() "
                            "по своей выборке"),
    russian_version="проверить",
)

OASIS = Measure(
    code="OASIS",
    name_ru="Шкала общей тяжести тревоги и нарушений (OASIS)",
    n_items=5, item_min=0, item_max=4,
    caseness=8,
    caseness_source="Campbell-Sills et al., 2009 (N = 1036, первичное звено, США)",
    reliable_change=None,
    reliable_change_source=("ориентиры: 4 балла (специализированная клиника, США), "
                            "от 5 (немецкая версия OASIS-D); для русской версии — "
                            "рассчитать (трек Б, этап 8)"),
    russian_version="нет — адаптация в работе (трек Б)",
)

WSAS = Measure(
    code="WSAS",
    name_ru="Шкала работы и социальной адаптации (WSAS)",
    n_items=5, item_min=0, item_max=8,
    caseness=None,
    caseness_source="не задан — сверить с Mundt et al., 2002",
    reliable_change=None,
    reliable_change_source="не установлен",
    russian_version="нет — кандидат на адаптацию (вторая волна)",
)

# Оценка терапевта для слоя расхождений (docs/06). Guy, 1976 (ECDEU Assessment Manual) —
# сверить с первоисточником. Порядковая шкала 1–7, 7 — тяжелее всего; тексты якорей
# не хранятся (правило 2). Код 0 («не оценивалось») переводится в пропуск при приёме.
CGIS = Measure(
    code="CGIS",
    name_ru="Общее клиническое впечатление — тяжесть (CGI-S), оценка терапевта",
    n_items=1, item_min=1, item_max=7,
    caseness=None,
    caseness_source="не задан — уровни с самоотчётом не сравниваются",
    reliable_change=None,
    reliable_change_source=("не установлен для одного пункта — порог изменения передаётся "
                            "явно с источником (эвристика)"),
    russian_version="проверенной нет — проверить",
    rater="clinician",
)

MEASURES: dict[str, Measure] = {m.code: m for m in (GAD7, PHQ9, PHQ8, OASIS, WSAS, CGIS)}


def get_measure(code: str) -> Measure:
    key = code.upper().replace("-", "")
    try:
        return MEASURES[key]
    except KeyError as exc:
        raise KeyError(f"Неизвестная шкала: {code}. Доступны: {', '.join(MEASURES)}") from exc


def require_client_measure(measure: Measure) -> None:
    """Статус, траектория и отчёт считаются только по самоотчёту клиента."""
    if measure.rater != "client":
        raise ValueError(f"{measure.code} — оценка клинициста; здесь нужна шкала самоотчёта")
