"""Демонстрация движка на СИНТЕТИЧЕСКИХ данных. Реальных клиентов здесь нет.

Запуск из корня репозитория:  python examples/demo.py
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbc.hypotheses import Hypothesis, evidence_for_pattern, verify_evidence  # noqa: E402
from mbc.measures import GAD7, PHQ8  # noqa: E402
from mbc.report import build_report, render_text  # noqa: E402
from mbc.scoring import score  # noqa: E402
from mbc.status import outcome  # noqa: E402
from mbc.trajectory import rule_based_flag  # noqa: E402
from mbc.triggers import (Reaction, TriggerEntry, find_patterns, pattern_change,  # noqa: E402
                          somatic_confounders, weekly_progress)

print("=" * 64)
print("СИНТЕТИЧЕСКИЙ КЛИЕНТ — данные выдуманы для демонстрации")
print("=" * 64)

# --- 1. GAD-7 по сессиям -------------------------------------------------------
gad7_by_session = {
    1: [2, 2, 3, 2, 2, 2, 2],
    2: [2, 2, 2, 2, 2, 2, 2],
    3: [2, 2, 2, 2, 1, 2, 2],
    4: [2, 1, 2, 2, 1, 2, 2],
    5: [1, 1, 2, 1, 1, 2, 1],
    6: [1, 1, 1, 1, 1, 1, 1],
}
history: list[tuple[int, int]] = []
gad7_scores = []
print("\nGAD-7 по сессиям:")
for session, answers in gad7_by_session.items():
    s = score(GAD7, answers)
    history.append((session, s.total))
    gad7_scores.append((session, s))
    flag = rule_based_flag(GAD7, history)
    print(f"  сессия {session}: {s.total:>2} ({s.severity:<11}) → {flag.signal.value}: {flag.reason}")

baseline, latest = history[0][1], history[-1][1]
print(f"\nИтог по GAD-7: {baseline} → {latest} — {outcome(GAD7, baseline, latest).value}")

phq8 = score(PHQ8, [1, 1, 0, 1, 1, 0, 1, 0])
print(f"PHQ-8 на последней сессии: {phq8.total} ({phq8.severity or 'диапазоны не заданы'})")

# --- 2. Карта триггеров ---------------------------------------------------------
start = datetime(2026, 9, 7, 20, 0)


def e(i, day, intensity, **kw):
    return TriggerEntry(f"e{i:02d}", start + timedelta(days=day), intensity, **kw)


entries = [
    e(1, 0, 8, before="звонок матери", where="дом", reaction=Reaction.AVOIDED, sleep_hours=7),
    e(2, 2, 9, before="звонок матери", where="дом", reaction=Reaction.REASSURANCE, sleep_hours=5),
    e(3, 3, 6, before="лента", where="дом", reaction=Reaction.STAYED, caffeine=True),
    e(4, 8, 8, before="звонок матери", where="работа", reaction=Reaction.AVOIDED),
    e(5, 10, 5, before="совещание", where="работа", reaction=Reaction.STAYED),
    e(6, 15, 6, before="звонок матери", where="дом", reaction=Reaction.STAYED),
    e(7, 17, 5, before="лента", where="дом", reaction=Reaction.SUPPORT),
    e(8, 22, 4, before="звонок матери", where="дом", reaction=Reaction.STAYED),
    e(9, 24, 5, before="звонок матери", where="транспорт", reaction=Reaction.STAYED),
]

print("\nПовторяющиеся контексты (минимум 3 эпизода):")
patterns = find_patterns(entries, dimensions=("before", "where"), min_episodes=3)
for p in patterns:
    print(f"  {p.label}: {p.count} эпизодов")

top = patterns[0]
print(f"\nДинамика паттерна «{top.label}» по неделям:")
stats = weekly_progress(entries, top)
for st in stats:
    rate = "—" if st.avoidance_rate is None else f"{st.avoidance_rate:.0%}"
    print(f"  {st.week}: эпизодов {st.episodes}, средняя интенсивность {st.mean_intensity}, избегание {rate}")
print(f"  Изменение (последняя − первая неделя): {pattern_change(stats)}")

print("\nЗаписи с возможными телесными факторами:")
for entry in entries:
    factors = somatic_confounders(entry)
    if factors:
        print(f"  {entry.id}: {', '.join(factors)}")

# --- 3. Гипотеза в формате ИИ-слоя (здесь составлена вручную) -----------------
high_ids, total = evidence_for_pattern(entries, top, high_intensity=7)
hypothesis = Hypothesis(
    statement="В начале терапии сильная тревога возникала после звонков матери",
    evidence_ids=high_ids, support=len(high_ids), total=total,
    alternative="Совпадение с недосыпом и вечерним временем, а не содержание разговора",
    check_question="Что именно в этих разговорах запускает тревогу — тема, тон, ожидание звонка?",
)
verify_evidence(hypothesis, entries, top)  # опора существует в записях, иначе ValueError
print("\n" + hypothesis.render())

# --- 4. Отчёт для терапевта ------------------------------------------------------
# PHQ-8 не содержит пункта безопасности: проверка не проводилась → safety_alerts=None
print("\n" + "=" * 64)
print(render_text(build_report(GAD7, gad7_scores, entries)))
