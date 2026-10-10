"""Формат гипотез ИИ-слоя.

ИИ — генератор гипотез для терапевта, а не советчик клиенту. Поэтому каждая
гипотеза обязана содержать опору на конкретные записи, альтернативное объяснение
тех же данных и вопрос для проверки на сессии. Гипотеза без альтернативы
не создаётся — это защита от гладкого текста, который звучит убедительнее,
чем позволяют данные.

Модуль не вызывает языковые модели: он задаёт контракт, которому должен
соответствовать их вывод (см. prompts/hypothesis_prompt.md). Проверка двухступенчатая:
класс Hypothesis проверяет форму, verify_evidence() — что опора существует в записях.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .triggers import Pattern, TriggerEntry


class HypothesisStatus(Enum):
    PROPOSED = "предложена"
    CONFIRMED = "подтверждена терапевтом"
    REJECTED = "отклонена терапевтом"


@dataclass
class Hypothesis:
    statement: str
    evidence_ids: list[str]
    """Записи, которые поддерживают гипотезу."""
    support: int
    total: int
    """Сколько записей в контексте всего (support из total)."""
    alternative: str
    check_question: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    therapist_note: str = ""

    def __post_init__(self) -> None:
        for name in ("statement", "alternative", "check_question"):
            if not getattr(self, name).strip():
                raise ValueError(f"Гипотеза: поле «{name}» обязательно")
        if not self.evidence_ids:
            raise ValueError("Гипотеза без опоры на записи не создаётся")
        if len(set(self.evidence_ids)) != self.support:
            raise ValueError("support должен совпадать с числом уникальных записей-доказательств")
        if not 0 < self.support <= self.total:
            raise ValueError("Должно выполняться 0 < support <= total")

    def confirm(self, note: str = "") -> None:
        self._resolve(HypothesisStatus.CONFIRMED, note)

    def reject(self, note: str = "") -> None:
        self._resolve(HypothesisStatus.REJECTED, note)

    def _resolve(self, status: HypothesisStatus, note: str) -> None:
        if self.status is not HypothesisStatus.PROPOSED:
            raise ValueError("Решение по гипотезе уже принято")
        self.status = status
        self.therapist_note = note

    def render(self) -> str:
        return "\n".join([
            f"Гипотеза: {self.statement}",
            f"Опора: {self.support} из {self.total} записей ({', '.join(self.evidence_ids)})",
            f"Альтернатива: {self.alternative}",
            f"Что проверить на сессии: {self.check_question}",
            f"Статус: {self.status.value}"
            + (f" — {self.therapist_note}" if self.therapist_note else ""),
        ])


def evidence_for_pattern(
    entries: Sequence[TriggerEntry], pattern: Pattern, high_intensity: int = 7,
) -> tuple[list[str], int]:
    """Записи паттерна с высокой интенсивностью и общее число записей паттерна.

    Порог 7 из 10 — настраиваемая эвристика.
    """
    ids = set(pattern.entry_ids)
    in_pattern = [e for e in entries if e.id in ids]
    high = [e.id for e in in_pattern if e.intensity >= high_intensity]
    return high, len(in_pattern)


def verify_evidence(
    hypothesis: Hypothesis, entries: Sequence[TriggerEntry], pattern: Pattern | None = None,
) -> None:
    """Сверка гипотезы с записями: опора должна существовать, а не только быть оформлена.

    Hypothesis проверяет форму, но не знает, есть ли такие записи на самом деле:
    модель может сослаться на несуществующую запись или завысить total. Проверяется:
    - каждый id из evidence_ids есть среди entries, без повторов;
    - total не больше числа переданных записей;
    - если передан pattern (контекст гипотезы) — доказательства принадлежат ему,
      а total равен числу его записей среди entries (как в evidence_for_pattern).
    Несоответствия собираются все сразу в одну ошибку ValueError: гипотеза
    с любым из них терапевту не показывается.
    """
    problems = []
    cited = list(dict.fromkeys(hypothesis.evidence_ids))
    repeated = [i for i in cited if hypothesis.evidence_ids.count(i) > 1]
    if repeated:
        problems.append(f"записи указаны повторно: {', '.join(repeated)}")
    known = {e.id for e in entries}
    missing = [i for i in cited if i not in known]
    if missing:
        problems.append(f"нет таких записей: {', '.join(missing)}")
    if hypothesis.total > len(entries):
        problems.append(f"total = {hypothesis.total}, а записей за период — {len(entries)}")
    if pattern is not None:
        pattern_ids = set(pattern.entry_ids)
        outside = [i for i in cited if i in known and i not in pattern_ids]
        if outside:
            problems.append(f"записи вне контекста «{pattern.label}»: {', '.join(outside)}")
        in_pattern = sum(e.id in pattern_ids for e in entries)
        if hypothesis.total != in_pattern:
            problems.append(f"total = {hypothesis.total}, а записей в контексте "
                            f"«{pattern.label}» — {in_pattern}")
    if problems:
        raise ValueError("Гипотеза не сверяется с записями: " + "; ".join(problems))
