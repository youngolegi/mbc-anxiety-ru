"""Текстовый отчёт для терапевта: замеры шкалы, светофор, статус, паттерны, телесные факторы.

Отчёт только для терапевта: не для передачи клиенту и не диагноз. Он ничего
не интерпретирует и не рекомендует — только собирает в одном месте то, что уже
посчитали другие модули движка, и подписывает, где эвристика и откуда взяты пороги.

Замеры передаются парами (сессия, Score) одной шкалы. Отчёт — адаптер к trajectory
и status, их API не меняется:
- незасчитанный замер (total=None) в траекторию и статус не попадает, в отчёте
  помечен «замер не засчитан»; если не засчитана первая сессия, отчёт явно пишет,
  какая сессия стала базовой линией;
- замер, пересчитанный пропорционально (prorated), попадает в расчёты, но помечается;
  отдельное предупреждение — если пересчитан базовый замер.

Безопасность. Отчёт НЕ вызывает check_safety: проверка безопасности выполняется
в момент приёма ответов, а сюда передаются готовые сигналы — пары (сессия, SafetyAlert).
Отчёт выводит их в шапке как есть, без оценки риска и рекомендаций. Три состояния шапки:
- safety_alerts=None — проверка не проводилась;
- пустая последовательность — проверено, сигналов нет;
- непустая — есть сигналы (последние сессии сверху).
Пустую последовательность передавайте только после проверки по шкале с пунктом
безопасности (PHQ-9). У GAD-7 и PHQ-8 такого пункта нет: check_safety для них всегда
возвращает пустой список, и это не «проверено» — передавайте None.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

from .measures import Measure
from .safety import SafetyAlert
from .scoring import Score
from .status import Outcome, outcome, resolve_threshold
from .trajectory import Signal, TrajectoryFlag, rule_based_flag
from .triggers import (DIMENSIONS, Pattern, TriggerEntry, WeekStat, find_patterns,
                       pattern_change, somatic_confounders, weekly_progress)

SIGNAL_LABELS = {
    Signal.GREEN: "ЗЕЛЁНЫЙ",
    Signal.AMBER: "ЖЁЛТЫЙ",
    Signal.RED: "КРАСНЫЙ",
    Signal.INSUFFICIENT: "НЕДОСТАТОЧНО ДАННЫХ",
}

DIMENSION_LABELS = {"where": "где", "with_whom": "с кем", "before": "за час до"}

SOMATIC_FIELDS = ("sleep_hours", "caffeine", "alcohol_prev_day", "unwell")


@dataclass(frozen=True)
class SessionScore:
    """Строка истории замеров."""
    session: int
    total: int | None
    severity: str | None
    n_answered: int
    prorated: bool

    @property
    def counted(self) -> bool:
        return self.total is not None


@dataclass(frozen=True)
class PatternSummary:
    dimensions: tuple[str, ...]
    values: tuple[str, ...]
    count: int
    weeks: tuple[WeekStat, ...]
    """Только недели, в которых были эпизоды паттерна."""
    reactions_marked: tuple[int, ...]
    """По неделям: записи с отмеченной реакцией — знаменатель доли избегания."""
    intensity_delta: float | None
    avoidance_delta: float | None
    overlaps: tuple[tuple[int, int], ...]
    """Пары (номер паттерна выше в списке, число общих записей)."""

    @property
    def label(self) -> str:
        return "; ".join(f"{DIMENSION_LABELS[d]}: {v}" for d, v in zip(self.dimensions, self.values))


@dataclass(frozen=True)
class ClientReport:
    measure: Measure
    history: tuple[SessionScore, ...]
    """Все переданные замеры по возрастанию номера сессии, включая незасчитанные."""
    threshold: float
    threshold_source: str
    threshold_explicit: bool
    check_by_session: int
    flag: TrajectoryFlag
    outcome: Outcome | None
    """None — засчитанных замеров меньше двух."""
    patterns: tuple[PatternSummary, ...]
    patterns_found: int
    """Сколько паттернов найдено всего; показаны первые len(patterns)."""
    dimensions: tuple[str, ...]
    min_episodes: int
    entries_total: int
    entries_with_somatic_fields: int
    """Записи, где заполнено хотя бы одно поле телесных факторов."""
    somatic: tuple[tuple[str, tuple[str, ...]], ...]
    """Пары (id записи, телесные факторы) в порядке времени записи."""
    min_sleep_hours: float
    safety_alerts: tuple[tuple[int, SafetyAlert], ...] | None
    """None — проверка не проводилась; () — проверено, сигналов нет. Последние сессии сверху."""

    @property
    def counted(self) -> tuple[SessionScore, ...]:
        return tuple(h for h in self.history if h.counted)

    @property
    def not_counted(self) -> tuple[SessionScore, ...]:
        return tuple(h for h in self.history if not h.counted)

    @property
    def baseline(self) -> SessionScore | None:
        counted = self.counted
        return counted[0] if counted else None

    @property
    def latest(self) -> SessionScore | None:
        counted = self.counted
        return counted[-1] if counted else None

    @property
    def baseline_shifted(self) -> bool:
        """Базовая линия — не первая переданная сессия (ранние замеры не засчитаны)."""
        return self.baseline is not None and self.baseline.session != self.history[0].session


def build_report(
    measure: Measure,
    scores: Iterable[tuple[int, Score]],
    entries: Iterable[TriggerEntry] = (),
    safety_alerts: Iterable[tuple[int, SafetyAlert]] | None = None,
    *,
    threshold: float | None = None,
    threshold_source: str | None = None,
    check_by_session: int = 4,
    top_n: int = 3,
    min_episodes: int = 3,
    dimensions: Sequence[str] = DIMENSIONS,
    min_sleep_hours: float = 6.0,
) -> ClientReport:
    """Собирает данные отчёта. Ничего не печатает — для текста есть render_text().

    threshold — порог надёжного изменения в баллах. Если у шкалы его нет, передать
    обязательно (молча чужой порог не подставляется). Переданный порог требует
    threshold_source: каждый порог в отчёте — с источником.
    """
    if top_n < 1:
        raise ValueError("top_n должен быть не меньше 1")
    if min_episodes < 2:
        raise ValueError("min_episodes должен быть не меньше 2: паттерн — повторяющийся контекст")
    if not dimensions:
        raise ValueError("Нужно хотя бы одно измерение для поиска паттернов")
    if threshold is not None and not (threshold_source or "").strip():
        raise ValueError("Переданному порогу нужен источник: укажите threshold_source")
    if threshold is None and threshold_source is not None:
        raise ValueError("threshold_source без threshold: источник не к чему отнести")
    thr = resolve_threshold(measure, threshold)
    if isinstance(thr, bool) or not math.isfinite(thr) or thr <= 0:
        raise ValueError("Порог надёжного изменения должен быть положительным числом")

    history = _history(measure, tuple(scores))
    counted = [h for h in history if h.counted]
    flag = rule_based_flag(measure, [(h.session, h.total) for h in counted],
                           threshold=thr, check_by_session=check_by_session)
    result = (outcome(measure, counted[0].total, counted[-1].total, threshold=thr)
              if len(counted) >= 2 else None)

    entries = tuple(entries)
    ids = [e.id for e in entries]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"Записи карты триггеров с повторяющимся id: {', '.join(repeated)}")

    found = find_patterns(entries, dimensions, min_episodes)
    shown = found[:top_n]
    patterns = tuple(_summarise(entries, p, shown[:i]) for i, p in enumerate(shown))

    somatic = tuple(
        (e.id, tuple(factors))
        for e in sorted(entries, key=lambda e: (e.timestamp, e.id))
        if (factors := somatic_confounders(e, min_sleep_hours))
    )
    with_fields = sum(any(getattr(e, f) is not None for f in SOMATIC_FIELDS) for e in entries)

    return ClientReport(
        measure=measure, history=history, threshold=thr,
        threshold_source=threshold_source if threshold is not None
        else measure.reliable_change_source,
        threshold_explicit=threshold is not None, check_by_session=check_by_session,
        flag=flag, outcome=result, patterns=patterns, patterns_found=len(found),
        dimensions=tuple(dimensions), min_episodes=min_episodes, entries_total=len(entries),
        entries_with_somatic_fields=with_fields, somatic=somatic,
        min_sleep_hours=min_sleep_hours, safety_alerts=_safety(safety_alerts),
    )


def _check_session(session: object) -> None:
    if isinstance(session, bool) or not isinstance(session, int) or session < 1:
        raise ValueError(f"Номер сессии должен быть целым числом от 1, получено {session!r}")


def _history(measure: Measure, scores: tuple[tuple[int, Score], ...]) -> tuple[SessionScore, ...]:
    """Проверка входа: пары (сессия, Score), одна шкала, одна сессия — один замер."""
    for pair in scores:
        if not (isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], Score)):
            raise ValueError(f"Замеры передаются парами (сессия, Score), получено {pair!r}")
        _check_session(pair[0])
    foreign = [f"сессия {s}: {sc.measure}" for s, sc in scores if sc.measure != measure.code]
    if foreign:
        raise ValueError(f"Отчёт строится по шкале {measure.code}, но переданы замеры "
                         f"другой шкалы ({'; '.join(foreign)}). Смешивать шкалы нельзя.")
    sessions = [s for s, _ in scores]
    repeated = sorted({s for s in sessions if sessions.count(s) > 1})
    if repeated:
        raise ValueError(f"Больше одного замера в сессии: {', '.join(map(str, repeated))}")
    rows = [SessionScore(s, sc.total, sc.severity, sc.n_answered, sc.prorated) for s, sc in scores]
    return tuple(sorted(rows, key=lambda r: r.session))


def _safety(
    alerts: Iterable[tuple[int, SafetyAlert]] | None,
) -> tuple[tuple[int, SafetyAlert], ...] | None:
    if alerts is None:
        return None
    alerts = tuple(alerts)
    for pair in alerts:
        if not (isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], SafetyAlert)):
            raise ValueError("Сигналы безопасности передаются парами (сессия, SafetyAlert), "
                             f"получено {pair!r}")
        _check_session(pair[0])
    # sorted устойчив: сигналы одной сессии сохраняют исходный порядок
    return tuple(sorted(alerts, key=lambda pair: -pair[0]))


def _summarise(entries: tuple[TriggerEntry, ...], pattern: Pattern,
               above: Sequence[Pattern]) -> PatternSummary:
    """Сводка паттерна: недельная динамика и пересечения с паттернами выше в списке."""
    weeks = tuple(weekly_progress(entries, pattern))
    change = pattern_change(weeks)
    ids = set(pattern.entry_ids)
    marked: dict[str, int] = {}
    for e in entries:
        if e.id in ids and e.reaction is not None:
            year, week, _ = e.timestamp.isocalendar()
            key = f"{year}-W{week:02d}"
            marked[key] = marked.get(key, 0) + 1
    overlaps = tuple((j + 1, shared) for j, other in enumerate(above)
                     if (shared := len(ids & set(other.entry_ids))))
    return PatternSummary(pattern.dimensions, pattern.values, pattern.count, weeks,
                          tuple(marked.get(w.week, 0) for w in weeks),
                          change["intensity_delta"], change["avoidance_delta"], overlaps)


# --- текст ----------------------------------------------------------------------

def render_text(report: ClientReport) -> str:
    """Текст отчёта для терапевта. Порядок блоков: безопасность — первой."""
    period = (f"сессии {report.history[0].session}–{report.history[-1].session}"
              if report.history else "замеров нет")
    lines = [
        f"ОТЧЁТ ДЛЯ ТЕРАПЕВТА · {report.measure.code} · {period}",
        "Только для терапевта: не для передачи клиенту, не диагноз.",
        "",
        *_render_safety(report),
        "",
        *_render_history(report),
        "",
        *_render_flag(report),
        "",
        *_render_status(report),
        "",
        *_render_patterns(report),
        "",
        *_render_somatic(report),
    ]
    return "\n".join(lines)


def _render_safety(report: ClientReport) -> list[str]:
    alerts = report.safety_alerts
    if alerts is None:
        return ["БЕЗОПАСНОСТЬ: проверка не проводилась — сигналы в отчёт не переданы"]
    if not alerts:
        return ["БЕЗОПАСНОСТЬ: проверено — сигналов нет"]
    lines = ["БЕЗОПАСНОСТЬ: ЕСТЬ СИГНАЛЫ (последние сверху)"]
    for session, alert in alerts:
        answer = "пропущен" if alert.value is None else str(alert.value)
        lines.append(f"  ! сессия {session} · {alert.measure}, пункт {alert.item}, "
                     f"ответ {answer}: {alert.message}")
    return lines


def _render_history(report: ClientReport) -> list[str]:
    if not report.history:
        return ["ЗАМЕРЫ: не переданы"]
    n_items = report.measure.n_items
    cells = [f"с{h.session}: {h.total}{'*' if h.prorated else ''}" if h.counted
             else f"с{h.session}: не засчитан (ответов {h.n_answered} из {n_items})"
             for h in report.history]
    lines = ["ЗАМЕРЫ", "  " + " · ".join(cells)]
    if any(h.prorated for h in report.counted):
        lines.append("  * пересчитан пропорционально по неполным ответам — допущение, не стандарт")
    base = report.baseline
    if base is None:
        lines.append("  ! ни один замер не засчитан")
        return lines
    if report.baseline_shifted:
        skipped = [h.session for h in report.history if h.session < base.session]
        lines.append(f"  ! базовая линия — сессия {base.session}: {_not_counted(skipped)}")
    if base.prorated:
        lines.append(f"  ! базовый замер (сессия {base.session}) пересчитан пропорционально "
                     f"(ответов {base.n_answered} из {n_items}): "
                     f"все сравнения со стартом опираются на пересчитанный балл")
    return lines


def _render_flag(report: ClientReport) -> list[str]:
    flag, m = report.flag, report.measure
    base, latest = report.baseline, report.latest
    where = f" (сессия {flag.session})" if flag.session is not None else ""
    lines = [f"СВЕТОФОР{where}: [{SIGNAL_LABELS[flag.signal]}] {flag.signal.value} — "
             f"{flag.reason}"]

    if flag.signal is Signal.INSUFFICIENT:
        lines.append(f"  засчитано замеров: {len(report.counted)} из {len(report.history)}")
    else:
        lines.append(f"  базовая линия: сессия {base.session}, балл {base.total}")
        later = [h.session for h in report.not_counted if h.session > latest.session]
        if later:
            lines.append(f"  ! {_not_counted(later)} — сигнал по последнему засчитанному "
                         f"(сессия {latest.session})")
        prorated = [h.session for h in (base, latest) if h.prorated]
        if prorated:
            lines.append(f"  ! сигнал опирается на пересчитанный пропорционально балл: "
                         f"сессия {', '.join(map(str, sorted(set(prorated))))}")
        if flag.signal is Signal.GREEN:
            if m.caseness is not None and base.total < m.caseness:
                lines.append(f"  жёлтый не применяется: на старте балл ниже порога «случая» "
                             f"({m.caseness})")
            elif latest.session < report.check_by_session:
                lines.append(f"  жёлтый ещё не проверяется: проверка с сессии "
                             f"{report.check_by_session}")

    case = ("" if m.caseness is None
            else f", если на старте балл не ниже порога «случая» ({m.caseness})")
    lines.append(f"  правило (эвристика, не валидировано): красный — ухудшение от базовой линии "
                 f"не меньше порога надёжного изменения; жёлтый — с сессии "
                 f"{report.check_by_session} нет надёжного улучшения от базовой линии{case}; "
                 f"номер сессии — номер сессии терапии, а не число замеров")
    return lines


def _render_status(report: ClientReport) -> list[str]:
    m = report.measure
    base, latest = report.baseline, report.latest
    if report.outcome is None or base is None or latest is None:
        lines = ["СТАТУС: недостаточно данных — нужно минимум два засчитанных замера"]
    else:
        prorated = [h.session for h in (base, latest) if h.prorated]
        mark = (f" (пересчитан пропорционально: сессия {', '.join(map(str, prorated))})"
                if prorated else "")
        lines = [f"СТАТУС: {_with_severity(base)} → {_with_severity(latest)}, "
                 f"сессии {base.session} → {latest.session}: {report.outcome.value}{mark}"]
    lines.append(f"  логика статуса — практика NHS Talking Therapies, по одной шкале "
                 f"({m.code}); в NHS «выздоровление» засчитывают по двум шкалам сразу — "
                 f"здесь шкалы не объединяются")
    lines.append(f"  порог надёжного изменения: {report.threshold:g} — {report.threshold_source}"
                 + _threshold_note(report))
    if m.caseness is None:
        lines.append(f"  порог «случая»: {m.caseness_source} — «выздоровление» не определяется")
    else:
        lines.append(f"  порог «случая»: {m.caseness} — {m.caseness_source}")
    if not m.severity_bands:
        lines.append("  диапазоны тяжести для шкалы не заданы")
    if not m.verified_ru_thresholds:
        lines.append("  ! пороги шкалы не пересчитаны на русских данных")
    mark = "! " if m.russian_version.startswith("нет") else ""
    lines.append(f"  {mark}русская версия: {m.russian_version}")
    return lines


def _with_severity(h: SessionScore) -> str:
    return f"{h.total} ({h.severity})" if h.severity else str(h.total)


def _threshold_note(report: ClientReport) -> str:
    m = report.measure
    if not report.threshold_explicit:
        return ""
    if m.reliable_change is None:
        return f" (справка по шкале: {m.reliable_change_source})"
    return f" (вместо {m.reliable_change:g} — {m.reliable_change_source})"


def _render_patterns(report: ClientReport) -> list[str]:
    head = "ПАТТЕРНЫ КАРТЫ ТРИГГЕРОВ"
    if report.entries_total == 0:
        return [f"{head}: записей нет"]
    dims = ", ".join(DIMENSION_LABELS[d] for d in report.dimensions)
    params = (f"измерения: {dims}; минимум эпизодов: {report.min_episodes}; "
              f"записей: {report.entries_total}")
    if not report.patterns:
        return [f"{head}: нет — ни один контекст не набрал минимума эпизодов ({params})"]
    lines = [
        f"{head}: показано {len(report.patterns)} из {report.patterns_found} ({params})",
        "  частота среди записей ≠ риск; паттерн — гипотеза, не вывод; "
        "динамика описательная, без статистического вывода",
    ]
    for i, p in enumerate(report.patterns, start=1):
        first, last = p.weeks[0], p.weeks[-1]
        lines.append(f"  {i}. {p.label} — эпизодов: {p.count}, недель с эпизодами: "
                     f"{len(p.weeks)} ({_week_span(first.week, last.week)})")
        cells = " · ".join(
            f"{_monday(w.week):%d.%m} {w.mean_intensity:g}/{_rate(w.avoidance_rate)}/{w.episodes}"
            for w in p.weeks)
        lines.append(f"     по неделям (с понедельника; интенсивность/избегание/эпизодов): {cells}")
        if len(p.weeks) == 1:
            lines.append("     одна неделя с эпизодами — динамику сравнивать не с чем")
        else:
            lines.append(f"     первая → последняя неделя: интенсивность {first.mean_intensity:g} → "
                         f"{last.mean_intensity:g} ({p.intensity_delta:+g}; "
                         f"эпизодов {first.episodes} и {last.episodes})")
            if p.avoidance_delta is None:
                lines.append("     избегание: в первой или последней неделе реакции не отмечены — "
                             "изменение не считается")
            else:
                lines.append(f"     избегание {_rate(first.avoidance_rate)} → "
                             f"{_rate(last.avoidance_rate)} ({p.avoidance_delta * 100:+.0f} п.п.; "
                             f"реакция отмечена в {p.reactions_marked[0]} и "
                             f"{p.reactions_marked[-1]} записях)")
        for j, shared in p.overlaps:
            lines.append(f"     ↳ общих записей с паттерном {j}: {shared}")
    lines.append("  избегание — доля среди записей с отмеченной реакцией; «—» — реакции не отмечены")
    return lines


def _render_somatic(report: ClientReport) -> list[str]:
    head = "ТЕЛЕСНЫЕ ФАКТОРЫ (могли поднять тревогу независимо от ситуации)"
    total, filled = report.entries_total, report.entries_with_somatic_fields
    if total == 0:
        return [f"{head}: записей нет"]
    if filled == 0:
        return [f"{head}: поля не заполнены ни в одной записи (записей: {total}) — "
                f"отсутствие факторов не установлено"]
    if not report.somatic:
        return [f"{head}: не отмечены (поля заполнены в {filled} из {total} {_records(total)})"]
    lines = [head]
    lines += [f"  {entry_id}: {', '.join(factors)}" for entry_id, factors in report.somatic]
    lines.append(f"  итого: {len(report.somatic)} из {filled} {_records(filled)} с заполненными "
                 f"полями (всего записей: {total}); порог сна {report.min_sleep_hours:g} ч — "
                 f"эвристика")
    return lines


def _not_counted(sessions: Sequence[int]) -> str:
    if len(sessions) == 1:
        return f"замер сессии {sessions[0]} не засчитан"
    return f"замеры сессий {', '.join(map(str, sessions))} не засчитаны"


def _records(n: int) -> str:
    """Форма после «из N»: из 1 записи, из 5 записей."""
    return "записи" if n % 10 == 1 and n % 100 != 11 else "записей"


def _monday(week: str) -> date:
    """'2026-W37' → понедельник этой ISO-недели."""
    year, number = week.split("-W")
    return date.fromisocalendar(int(year), int(number), 1)


def _week_span(first: str, last: str) -> str:
    start, end = _monday(first), _monday(last) + timedelta(days=6)
    if start.year == end.year:
        return f"{start:%d.%m}–{end:%d.%m.%Y}"
    return f"{start:%d.%m.%Y}–{end:%d.%m.%Y}"


def _rate(rate: float | None) -> str:
    return "—" if rate is None else f"{rate:.0%}"
