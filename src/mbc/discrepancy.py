"""Слой расхождений: самоотчёт клиента × оценка терапевта (CGI-S). Спецификация — docs/06.

Расхождение — тема для разговора, а не вердикт. Модуль не решает, кто прав,
не исправляет ни один балл и не ставит диагнозов. Вывод только для терапевта,
поставившего оценку: к клиенту модуль не обращается, агрегаты по терапевту
(«сколько раз не совпал с клиентом») не считаются — иначе блок становится оценкой
работы терапевта. На светофор и шапку безопасности блок не влияет (правило 5).

Сравниваются направления изменений, каждый источник — сам с собой: проверенной
таблицы соответствия уровней GAD-7 ↔ CGI-S нет, поэтому устойчивое расхождение
уровней («клиент всегда пишет "минимальная", терапевт видит тяжёлое») здесь не видно.

Клиентская сторона берётся из готового ClientReport: та же базовая линия, тот же
порог и тот же источник порога, что у светофора и статуса. Сторона терапевта —
Direction по порогу cgi_threshold (эвристика): слово «надёжное» к оценке из одного
пункта не применяется.

Слепота оценки вычисляется по отметкам времени, а не вводится терапевтом: оценка
слепая, если поставлена до просмотра самоотчёта этой сессии. Прошлые баллы терапевт
уже видел, поэтому независимой оценка не бывает в любом случае.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable

from .measures import CGIS, Measure
from .reliable_change import (Change, ReliabilityKind, SampleThreshold, classify_change,
                              min_change_points)
from .report import ClientReport, PatternSummary, SessionScore, _check_session
from .scoring import Score, score


class Direction(Enum):
    """Направление изменения оценки терапевта — без слова «надёжное»: пункт один."""
    BETTER = "лучше"
    NONE = "сдвиг меньше порога"
    WORSE = "хуже"


class Kind(Enum):
    """Клетка матрицы «клиент × терапевт», где направления расходятся."""
    OPPOSITE = "направления противоположны"
    CLIENT_WORSE_UNSEEN = "самоотчёт — хуже, оценка терапевта — без сдвига на порог"
    CLIENT_BETTER_UNSEEN = "самоотчёт — лучше, оценка терапевта — без сдвига на порог"
    THERAPIST_ONLY = "сдвиг только в оценке терапевта"


ALWAYS_SHOWN = frozenset({Kind.OPPOSITE, Kind.CLIENT_WORSE_UNSEEN})
"""Остальные типы свёрнуты, пока не держатся хотя бы двух сравнений подряд."""


class Window(Enum):
    BASE = "база → текущая"
    ADJACENT = "предыдущая пара → текущая"


class BlockState(Enum):
    """Состояния блока взаимоисключающие — как три состояния шапки безопасности."""
    NOT_PROVIDED = "оценки терапевта не переданы"
    INSUFFICIENT = "недостаточно пар"
    NONE_FOUND = "расхождений нет"
    FOUND = "есть расхождения"


POINTER = ("Клиентская сторона: надёжное ухудшение — см. светофор и шапку безопасности; "
           "объяснения ниже сигнал не отменяют")
"""Первая строка записи при надёжном ухудшении клиента: указатель, а не оценка риска."""

QUESTIONS = {
    Kind.OPPOSITE: "Что за эти недели стало легче, а что — тяжелее?",
    Kind.CLIENT_WORSE_UNSEEN: "Как бы вы сами описали, что изменилось за последние недели?",
    Kind.CLIENT_BETTER_UNSEEN: "Что за последние недели стало даваться легче — и как вы это заметили?",
    Kind.THERAPIST_ONLY: "Было ли на этой неделе что-то, о чём мы ещё не говорили?",
}
"""Один вопрос для сессии на клетку матрицы. Тексты — черновик, правит автор."""

_AS_DIRECTION = {Change.IMPROVED: Direction.BETTER, Change.NONE: Direction.NONE,
                 Change.DETERIORATED: Direction.WORSE}
_FAVOURABLE = {Direction.WORSE: 0, Direction.NONE: 1, Direction.BETTER: 2}


def cgis_score(code: int | None) -> Score:
    """Score CGI-S по коду оценки: 1–7; 0 — «не оценивалось» → пропуск, total=None.

    Код 0 по Guy, 1976 — сверить с первоисточником.
    """
    return score(CGIS, [None if type(code) is int and code == 0 else code])


@dataclass(frozen=True)
class TherapistRating:
    """Оценка терапевта (CGI-S) за сессию.

    rated_at — когда поставлена оценка; report_viewed_at — когда терапевт впервые открыл
    самоотчёт этой сессии (None — до оценки не открывал). Обе отметки ставит система.
    """
    session: int
    score: Score
    rated_at: datetime
    report_viewed_at: datetime | None

    @property
    def blind(self) -> bool:
        """Поставлена до просмотра самоотчёта этой сессии; совпадение отметок — не слепая."""
        return self.report_viewed_at is None or self.rated_at < self.report_viewed_at

    @property
    def counted(self) -> bool:
        return self.score.total is not None


@dataclass(frozen=True)
class OtherScale:
    """Другой самоотчёт за то же окно — данные к объяснению «стороны измеряют разное»."""
    measure: str
    start: int | None
    end: int | None
    change: Change | None
    """None — в одной из сессий окна нет засчитанного замера."""
    threshold: float
    threshold_source: str


@dataclass(frozen=True)
class Discrepancy:
    kind: Kind
    window: Window
    from_session: int
    to_session: int
    client_start: int
    client_end: int
    client_change: Change
    client_prorated: tuple[int, ...]
    """Сессии окна, где самоотчёт пересчитан пропорционально."""
    therapist_start: int
    therapist_end: int
    therapist_direction: Direction
    non_blind: tuple[int, ...]
    """Сессии окна, где оценка поставлена после просмотра самоотчёта."""
    persisted: int
    """Сравнений подряд (в этом окне) с тем же типом и теми же направлениями обеих сторон."""
    alternatives: tuple[str, ...]
    question: str
    others: tuple[OtherScale, ...] = ()

    def __post_init__(self) -> None:
        if not self.alternatives:
            raise ValueError("У расхождения должна быть хотя бы одна альтернатива")
        if not self.question.strip():
            raise ValueError("У расхождения должен быть вопрос для сессии")
        if self.persisted < 1:
            raise ValueError("persisted не может быть меньше 1")

    @property
    def blind(self) -> bool:
        """Сравнение слепое, только если слепы обе оценки."""
        return not self.non_blind

    @property
    def collapsed(self) -> bool:
        return self.kind not in ALWAYS_SHOWN and self.persisted < 2


@dataclass(frozen=True)
class DiscrepancyBlock:
    report: ClientReport
    """Отчёт, по которому построен блок: клиентская сторона — из него."""
    state: BlockState
    pairs: tuple[int, ...]
    """Сессии, где засчитаны и самоотчёт, и оценка терапевта."""
    sessions_total: int
    """Сессий, где есть самоотчёт или оценка терапевта."""
    client_only: tuple[int, ...]
    """Засчитанный самоотчёт, оценки терапевта нет."""
    therapist_only: tuple[int, ...]
    """Засчитанная оценка терапевта, засчитанного самоотчёта нет."""
    ratings_not_counted: tuple[int, ...]
    non_blind: int
    """Оценок после просмотра самоотчёта среди пар."""
    cgi_threshold: int
    cgi_threshold_source: str
    discrepancies: tuple[Discrepancy, ...]
    notes: tuple[str, ...]
    """Какие направления не определяются (пол и потолок шкалы) и почему."""
    other_measures: tuple[str, ...]

    @property
    def shown(self) -> tuple[Discrepancy, ...]:
        return tuple(d for d in self.discrepancies if not d.collapsed)

    @property
    def collapsed(self) -> tuple[Discrepancy, ...]:
        return tuple(d for d in self.discrepancies if d.collapsed)

    def lines(self) -> list[str]:
        """Текст блока для отчёта."""
        return _render(self)


def classify_cell(client: Direction, client_worse: int, therapist: Direction,
                  therapist_worse: int) -> Kind | None:
    """Клетка матрицы; None — согласие, в том числе согласие ниже порога.

    client_worse и therapist_worse — сырые сдвиги в сторону ухудшения (плюс — хуже).
    Если одна сторона изменилась на порог, а другая — с тем же знаком, но ниже порога,
    это согласие ниже порога: «нет надёжного изменения» ≠ «без изменений».
    """
    if client is therapist:
        return None
    if Direction.NONE not in (client, therapist):
        return Kind.OPPOSITE
    if therapist is Direction.NONE:
        if _same_sign(client, therapist_worse):
            return None
        return (Kind.CLIENT_WORSE_UNSEEN if client is Direction.WORSE
                else Kind.CLIENT_BETTER_UNSEEN)
    if _same_sign(therapist, client_worse):
        return None
    return Kind.THERAPIST_ONLY


def _same_sign(direction: Direction, worse: int) -> bool:
    return worse > 0 if direction is Direction.WORSE else worse < 0


def build_discrepancies(report: ClientReport, ratings: Iterable[TherapistRating] | None, *,
                        cgi_threshold: int | SampleThreshold,
                        cgi_threshold_source: str | None,
                        other_reports: Iterable[ClientReport] = ()) -> DiscrepancyBlock:
    """Блок расхождений по готовому отчёту и оценкам терапевта.

    ratings=None — оценки не переданы; пустая последовательность — переданы, но их нет.
    cgi_threshold — сдвиг CGI-S в баллах (целое 1–6) с обязательным источником или
    SampleThreshold вида ICC ретеста (источник берётся из расчёта). Значения по умолчанию
    нет: порог CGI-S — эвристика, и выбор должен быть явным.
    other_reports — отчёты по другим самоотчётам (PHQ-8/9, WSAS): их направление за то же
    окно показывается как данные к объяснению «стороны измеряют разное».
    """
    if not isinstance(report, ClientReport):
        raise ValueError("Нужен ClientReport: клиентская сторона берётся из отчёта")
    cgi, cgi_source = _cgi_threshold(cgi_threshold, cgi_threshold_source)
    others = _others(report, other_reports)
    codes = tuple(o.measure.code for o in others)
    if ratings is None:
        return DiscrepancyBlock(report, BlockState.NOT_PROVIDED, (), len(report.history), (), (),
                                (), 0, cgi, cgi_source, (), (), codes)

    ratings = _ratings(tuple(ratings))
    client = {h.session: h for h in report.counted}
    by_session = {r.session: r for r in ratings}
    counted = {s: r for s, r in by_session.items() if r.counted}
    pairs = [_Pair(s, client[s], counted[s]) for s in sorted(client.keys() & counted.keys())]
    sessions = {h.session for h in report.history} | by_session.keys()
    common = dict(
        report=report, pairs=tuple(p.session for p in pairs), sessions_total=len(sessions),
        client_only=tuple(sorted(client.keys() - by_session.keys())),
        therapist_only=tuple(sorted(counted.keys() - client.keys())),
        ratings_not_counted=tuple(sorted(s for s, r in by_session.items() if not r.counted)),
        non_blind=sum(not p.rating.blind for p in pairs),
        cgi_threshold=cgi, cgi_threshold_source=cgi_source, other_measures=codes,
    )
    if len(pairs) < 2:
        return DiscrepancyBlock(state=BlockState.INSUFFICIENT, discrepancies=(), notes=(),
                                **common)

    base_cmp = [_compare(report, pairs[0], p, cgi) for p in pairs[1:]]
    found = []
    notes = _unreachable_notes(report, pairs[0], cgi)
    current = base_cmp[-1]
    if current.kind is not None:
        found.append(_discrepancy(report, others, Window.BASE, pairs[0], pairs[-1], current,
                                  _persisted(base_cmp)))
    if len(pairs) >= 3 and current.client is not Direction.WORSE:
        # окно 2 — только для ухудшения клиента: рецидив после ранних успехов от базы не виден
        adjacent = [_compare(report, a, b, cgi) for a, b in zip(pairs, pairs[1:])]
        last = adjacent[-1]
        if last.client is Direction.WORSE:
            if last.kind is not None:
                found.append(_discrepancy(report, others, Window.ADJACENT, pairs[-2], pairs[-1],
                                          last, _persisted(adjacent)))
            elif last.suppressed:
                notes += (last.suppressed,)
    state = BlockState.FOUND if found else BlockState.NONE_FOUND
    return DiscrepancyBlock(state=state, discrepancies=tuple(found), notes=notes, **common)


# --- подготовка входа --------------------------------------------------------------

def _cgi_threshold(threshold: object, source: object) -> tuple[int, str]:
    if isinstance(threshold, SampleThreshold):
        if source is not None:
            raise ValueError("У рассчитанного порога источник уже есть — "
                             "cgi_threshold_source не нужен")
        if threshold.measure != CGIS.code:
            raise ValueError(f"Порог рассчитан для {threshold.measure}, а нужен порог {CGIS.code}")
        if threshold.kind is not ReliabilityKind.RETEST:
            raise ValueError(f"{CGIS.code}: у одного пункта {threshold.kind.value} не определён — "
                             f"только ICC ретеста")
        points, source = threshold.min_change_points, threshold.source
    else:
        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise ValueError(f"cgi_threshold — целое число баллов или SampleThreshold, "
                             f"получено {threshold!r}")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Порогу CGI-S нужен источник: укажите cgi_threshold_source")
        points = threshold
    span = CGIS.max_total - CGIS.min_total
    if not 1 <= points <= span:
        raise ValueError(f"Порог CGI-S — от 1 до {span} баллов, получено {points}")
    return points, source


def _others(report: ClientReport, other_reports: Iterable[ClientReport]) -> tuple[ClientReport, ...]:
    others = tuple(other_reports)
    codes = [report.measure.code]
    for other in others:
        if not isinstance(other, ClientReport):
            raise ValueError(f"Другие самоотчёты передаются как ClientReport, получено {other!r}")
        if other.measure.code in codes:
            raise ValueError(f"Шкала {other.measure.code} уже есть: каждая шкала — один раз, "
                             f"и не шкала самого отчёта")
        codes.append(other.measure.code)
    return others


def _ratings(ratings: tuple[object, ...]) -> tuple[TherapistRating, ...]:
    """Проверка входа: как у замеров в отчёте — сессия от 1, одна оценка на сессию."""
    for r in ratings:
        if not isinstance(r, TherapistRating):
            raise ValueError(f"Оценки терапевта передаются как TherapistRating, получено {r!r}")
        _check_session(r.session)
        if not isinstance(r.score, Score) or r.score.measure != CGIS.code:
            raise ValueError(f"Сессия {r.session}: оценка терапевта — Score шкалы {CGIS.code}, "
                             f"получено {r.score!r}")
        total = r.score.total
        if total is not None and (isinstance(total, bool) or not isinstance(total, int)
                                  or not CGIS.min_total <= total <= CGIS.max_total):
            raise ValueError(f"Сессия {r.session}: оценка {total!r} — не целое число "
                             f"{CGIS.min_total}–{CGIS.max_total}; код 0 — через cgis_score()")
        if not isinstance(r.rated_at, datetime) or not (
                r.report_viewed_at is None or isinstance(r.report_viewed_at, datetime)):
            raise ValueError(f"Сессия {r.session}: отметки времени — datetime")
        try:
            r.blind
        except TypeError as exc:
            raise ValueError(f"Сессия {r.session}: отметки времени несравнимы — "
                             f"одна с часовым поясом, другая без") from exc
    sessions = [r.session for r in ratings]
    repeated = sorted({s for s in sessions if sessions.count(s) > 1})
    if repeated:
        raise ValueError(f"Больше одной оценки терапевта в сессии: {', '.join(map(str, repeated))}")
    return ratings  # type: ignore[return-value]


# --- сравнение ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Pair:
    session: int
    client: SessionScore
    rating: TherapistRating

    @property
    def cgi(self) -> int:
        return self.rating.score.total  # type: ignore[return-value]


@dataclass(frozen=True)
class _Comparison:
    client_change: Change
    client: Direction
    therapist: Direction
    kind: Kind | None
    """None — согласие или расхождение, которое недостижимо по построению."""
    suppressed: str | None
    """Почему расхождение не выводится: направление недостижимо у границы шкалы."""

    @property
    def signature(self) -> tuple[Kind, Direction, Direction] | None:
        return None if self.kind is None else (self.kind, self.client, self.therapist)


def _compare(report: ClientReport, a: _Pair, b: _Pair, cgi: int) -> _Comparison:
    m = report.measure
    change = classify_change(a.client.total, b.client.total, report.threshold, m.higher_is_worse)
    client = _AS_DIRECTION[change]
    t_worse = b.cgi - a.cgi
    therapist = (Direction.WORSE if t_worse >= cgi
                 else Direction.BETTER if t_worse <= -cgi else Direction.NONE)
    kind = classify_cell(client, _worse(m, a.client.total, b.client.total), therapist, t_worse)
    suppressed = None
    # расхождение «одна сторона сдвинулась, другая — нет» не выводится, если второй стороне
    # сдвинуться в ту же сторону было некуда: иначе мягкие и тяжёлые клиенты давали бы
    # расхождения по построению
    if kind in (Kind.CLIENT_WORSE_UNSEEN, Kind.CLIENT_BETTER_UNSEEN) \
            and not _reachable(CGIS, a.cgi, cgi, client):
        suppressed = _unreachable_note(client, _THERAPIST_SIDE, a.session, a.cgi, cgi)
    elif kind is Kind.THERAPIST_ONLY and not _reachable(
            m, a.client.total, min_change_points(report.threshold), therapist):
        suppressed = _unreachable_note(therapist, f"самоотчёту {m.code}", a.session,
                                       a.client.total, min_change_points(report.threshold))
    if suppressed:
        kind = None
    return _Comparison(change, client, therapist, kind, suppressed)


_THERAPIST_SIDE = "оценке терапевта (CGI-S)"


def _worse(measure: Measure, start: int, end: int) -> int:
    return end - start if measure.higher_is_worse else start - end


def _reachable(measure: Measure, start: int, points: int, direction: Direction) -> bool:
    to_min, to_max = start - measure.min_total, measure.max_total - start
    if direction is Direction.BETTER:
        room = to_min if measure.higher_is_worse else to_max
    else:
        room = to_max if measure.higher_is_worse else to_min
    return room >= points


def _unreachable_note(direction: Direction, side: str, session: int, value: int,
                      points: int) -> str:
    what = "улучшение" if direction is Direction.BETTER else "ухудшение"
    return (f"{what} по {side} не определяется: в сессии {session} — {value}, "
            f"до границы шкалы меньше порога {points}")


def _unreachable_notes(report: ClientReport, base: _Pair, cgi: int) -> tuple[str, ...]:
    """Пол и потолок на базе: какие направления не определяются — у обеих сторон."""
    m, points = report.measure, min_change_points(report.threshold)
    notes = []
    for direction in (Direction.BETTER, Direction.WORSE):
        if not _reachable(m, base.client.total, points, direction):
            notes.append(_unreachable_note(direction, f"самоотчёту {m.code}", base.session,
                                           base.client.total, points))
        if not _reachable(CGIS, base.cgi, cgi, direction):
            notes.append(_unreachable_note(direction, _THERAPIST_SIDE, base.session, base.cgi, cgi))
    return tuple(notes)


def _persisted(comparisons: list[_Comparison]) -> int:
    """Число последних сравнений подряд с той же подписью, что у текущего."""
    current = comparisons[-1].signature
    n = 0
    for c in reversed(comparisons):
        if c.signature != current:
            break
        n += 1
    return n


def _discrepancy(report: ClientReport, others: tuple[ClientReport, ...], window: Window,
                 a: _Pair, b: _Pair, c: _Comparison, persisted: int) -> Discrepancy:
    assert c.kind is not None
    return Discrepancy(
        kind=c.kind, window=window, from_session=a.session, to_session=b.session,
        client_start=a.client.total, client_end=b.client.total, client_change=c.client_change,  # type: ignore[arg-type]
        client_prorated=tuple(p.session for p in (a, b) if p.client.prorated),
        therapist_start=a.cgi, therapist_end=b.cgi, therapist_direction=c.therapist,
        non_blind=tuple(p.session for p in (a, b) if not p.rating.blind),
        persisted=persisted,
        alternatives=_alternatives(report, c.kind, c.client, c.therapist, a.session),
        question=QUESTIONS[c.kind],
        others=tuple(_other_scale(o, a.session, b.session) for o in others),
    )


def _other_scale(other: ClientReport, start: int, end: int) -> OtherScale:
    totals = {h.session: h.total for h in other.counted}
    x1, x2 = totals.get(start), totals.get(end)
    change = (classify_change(x1, x2, other.threshold, other.measure.higher_is_worse)
              if x1 is not None and x2 is not None else None)
    return OtherScale(other.measure.code, x1, x2, change, other.threshold, other.threshold_source)


# --- альтернативы ------------------------------------------------------------------
# Черновик клинического текста — правит автор. Каждая альтернатива в модальной форме.
# Первая — буквальное прочтение: изменение, о котором сообщает одна сторона, реально.

def _alternatives(report: ClientReport, kind: Kind, client: Direction, therapist: Direction,
                  from_session: int) -> tuple[str, ...]:
    alts = [_literal(kind, client, therapist),
            "Возможно, дело в разных окнах: самоотчёт охватывает последние две недели, "
            "оценка терапевта — состояние на сессии.",
            f"Возможно, стороны измеряют разное: самоотчёт — симптомы по шкале "
            f"{report.measure.code}, CGI-S — общее клиническое впечатление о тяжести."]
    if therapist is Direction.NONE:
        alts.append("Может быть, шаг однопунктовой оценки слишком крупный, чтобы отразить "
                    "такое изменение.")
    else:
        alts.append("Может быть, сдвиг однопунктовой оценки на порог случаен: у одного пункта "
                    "велик разброс.")
    if _FAVOURABLE[therapist] > _FAVOURABLE[client]:
        alts.append("Возможно, о трудностях легче написать в опроснике, чем сказать на сессии, — "
                    "например, если в отношениях с терапевтом есть напряжение.")
        alts.append("Может быть, впечатление от сессии оптимистичнее, чем динамика между "
                    "сессиями (на такую возможность указывают данные Hannan et al., 2005 — сверить).")
    if client is Direction.BETTER and from_session <= 2:
        alts.append("Возможно, часть снижения по самоотчёту — регрессия к среднему: к терапии "
                    "обычно приходят на пике, и следующий замер ниже даже без изменения состояния.")
    if therapist is Direction.WORSE:
        if from_session == 1:
            alts.append("Возможно, к этой сессии картина проблем клиента у терапевта стала полнее, "
                        "чем на первой встрече, и оценка выросла не из-за изменения состояния.")
        alts.append("Если сессия была тяжёлой по содержанию, оценка могла сдвинуться "
                    "по впечатлению от неё.")
    if client is Direction.WORSE:
        alts += [_exposure(p) for p in report.patterns if _approaching(p)]
    return tuple(alts)


def _literal(kind: Kind, client: Direction, therapist: Direction) -> str:
    if kind is Kind.CLIENT_WORSE_UNSEEN:
        return "Возможно, клиенту действительно стало хуже, а на сессии это пока не видно."
    if kind is Kind.CLIENT_BETTER_UNSEEN:
        return "Возможно, клиенту действительно стало лучше, а на сессии это пока не видно."
    if kind is Kind.THERAPIST_ONLY:
        what = "улучшение" if therapist is Direction.BETTER else "ухудшение"
        return (f"Возможно, {what}, которое видно на сессии, реально, а в самоотчёте "
                f"оно пока не отразилось.")
    return (f"Возможно, обе стороны видят реальные изменения в разных областях: "
            f"по самоотчёту — {client.value}, по оценке терапевта — {therapist.value}.")


def _approaching(p: PatternSummary) -> bool:
    """Доля избегания снизилась, а интенсивность не выросла — с данными по обоим концам."""
    return (p.avoidance_delta is not None and p.avoidance_delta < 0
            and p.intensity_delta is not None and p.intensity_delta <= 0)


def _exposure(p: PatternSummary) -> str:
    first, last = p.weeks[0], p.weeks[-1]
    return (f"Если в эти недели шла экспозиция, дистресс мог временно вырасти — проверить "
            f"по карте триггеров: «{p.label}», избегание {first.avoidance_rate:.0%} → "
            f"{last.avoidance_rate:.0%} (реакция отмечена в {p.reactions_marked[0]} и "
            f"{p.reactions_marked[-1]} записях), интенсивность {first.mean_intensity:g} → "
            f"{last.mean_intensity:g}.")


# --- текст -------------------------------------------------------------------------

HEAD = "РАСХОЖДЕНИЯ — темы для разговора"


def _render(block: DiscrepancyBlock) -> list[str]:
    r, pairs = block.report, block.pairs
    if block.state is BlockState.NOT_PROVIDED:
        return [f"{HEAD}: оценки терапевта не переданы", _blind_line(block)]
    span = f"сессии {pairs[0]} → {pairs[-1]}" if len(pairs) >= 2 else ""
    if block.state is BlockState.INSUFFICIENT:
        head = f"{HEAD}: недостаточно пар — нужно минимум две, есть {len(pairs)}"
    elif block.state is BlockState.NONE_FOUND:
        head = f"{HEAD}: расхождений нет · сравнено пар: {len(pairs)}, {span}"
    else:
        head = (f"{HEAD}: найдено {len(block.discrepancies)} (развёрнуто {len(block.shown)}, "
                f"свёрнуто {len(block.collapsed)}) · сравнено пар: {len(pairs)}, {span}")
    lines = [head,
             "  расхождение — тема для разговора, не вердикт; сравниваются направления изменений, "
             "каждая шкала — сама с собой; уровни не сравниваются, поэтому устойчивое расхождение "
             "уровней здесь не видно",
             f"  пары (засчитаны и самоотчёт, и оценка терапевта): {len(pairs)} из "
             f"{block.sessions_total} сессий; пропуски могут быть неслучайными"]
    if block.client_only:
        lines.append(f"  самоотчёт без оценки терапевта: {_sessions(block.client_only)}")
    if block.ratings_not_counted:
        lines.append(f"  оценка терапевта не засчитана: {_sessions(block.ratings_not_counted)}")
    if block.therapist_only:
        lines.append(f"  оценка терапевта без засчитанного самоотчёта: "
                     f"{_sessions(block.therapist_only)}")
    latest = r.latest
    if pairs and latest is not None and latest.session > pairs[-1] \
            and latest.session in block.client_only:
        lines.append(f"  самоотчёт сессии {latest.session} ещё без оценки терапевта — "
                     f"текущая пара: сессия {pairs[-1]}")
    if block.state in (BlockState.NONE_FOUND, BlockState.FOUND):
        lines += _comparison_context(block)
    lines.append(_blind_line(block))
    for i, d in enumerate(block.discrepancies, start=1):
        lines += _render_one(i, d, block)
    return lines


def _comparison_context(block: DiscrepancyBlock) -> list[str]:
    r, pairs = block.report, block.pairs
    lines = []
    status_base = r.baseline.session  # type: ignore[union-attr]
    if pairs[0] != status_base:
        reason = ("оценка терапевта не засчитана" if status_base in block.ratings_not_counted
                  else "нет оценки терапевта")
        lines.append(f"  сравнение по сессиям {pairs[0]} → {pairs[-1]}: в сессии {status_base} "
                     f"{reason}; база статуса и светофора — сессия {status_base}")
    if pairs[0] == 1:
        lines.append("  база — сессия 1: оценка терапевта на неполной картине")
    lines.append(f"  порог по самоотчёту — как у статуса: {r.threshold:g} — {r.threshold_source}")
    sensitivity = "; 1 балл — только анализ чувствительности" if block.cgi_threshold == 1 else ""
    lines.append(f"  порог сдвига оценки терапевта (CGI-S): {block.cgi_threshold} — "
                 f"{block.cgi_threshold_source}; эвристика для одного пункта{sensitivity}")
    lines += [f"  {note}" for note in block.notes]
    if not block.other_measures:
        lines.append(f"  другие самоотчёты не переданы — сравнение только со шкалой отчёта "
                     f"({r.measure.code})")
    return lines


def _blind_line(block: DiscrepancyBlock) -> str:
    head = "  оценок терапевта после просмотра самоотчёта"
    if not block.pairs:
        return f"{head}: пар нет"
    return (f"{head}: {block.non_blind} из {len(block.pairs)} в парах; совпадение направлений "
            f"по ним менее информативно")


def _render_one(i: int, d: Discrepancy, block: DiscrepancyBlock) -> list[str]:
    title = f"{d.kind.value} · сессии {d.from_session} → {d.to_session} ({d.window.value})"
    if d.collapsed:
        return [f"  {i}. свёрнуто: {title} — впервые, разворачивается при повторе"]
    code = block.report.measure.code
    lines = ([f"  {i}. {POINTER}", f"     {title}"] if d.client_change is Change.DETERIORATED
             else [f"  {i}. {title}"])
    prorated = (f"; пересчитан пропорционально: сессия {', '.join(map(str, d.client_prorated))}"
                if d.client_prorated else "")
    base = block.report.baseline
    relative = (f"; светофор и статус считают от базовой линии — сессия {base.session}"
                if d.window is Window.ADJACENT and base is not None else "")
    lines.append(f"     самоотчёт {code}: {d.client_start} → {d.client_end} "
                 f"({_delta(d.client_end - d.client_start)}), {d.client_change.value} "
                 f"(порог {block.report.threshold:g}){prorated}{relative}")
    thr = block.cgi_threshold
    direction = (f"сдвиг меньше порога {thr}" if d.therapist_direction is Direction.NONE
                 else f"{d.therapist_direction.value} (порог {thr})")
    blind = ("обе оценки — до просмотра самоотчёта" if d.blind
             else f"после просмотра самоотчёта: сессия {', '.join(map(str, d.non_blind))}")
    lines.append(f"     оценка терапевта CGI-S: {d.therapist_start} → {d.therapist_end} "
                 f"({_delta(d.therapist_end - d.therapist_start)}), {direction}; {blind}")
    caveat = ("от одной базы" if d.window is Window.BASE else "соседние окна делят сессии")
    lines.append(f"     держится сравнений подряд: {d.persisted} ({caveat} — "
                 f"не независимые подтверждения)")
    lines.append("     возможные объяснения (не вердикт):")
    lines += [f"       – {a}" for a in d.alternatives]
    if d.others:
        lines.append("     другие самоотчёты за то же окно (к объяснению «стороны измеряют разное»):")
        lines += [f"       {_render_other(o, d)}" for o in d.others]
    lines.append(f"     вопрос для сессии: «{d.question}»")
    return lines


def _render_other(o: OtherScale, d: Discrepancy) -> str:
    if o.change is None:
        missing = [s for s, x in ((d.from_session, o.start), (d.to_session, o.end)) if x is None]
        where = (f"в сессии {missing[0]}" if len(missing) == 1
                 else f"в сессиях {', '.join(map(str, missing))}")
        return f"{o.measure}: нет засчитанного замера {where}"
    return (f"{o.measure}: {o.start} → {o.end} ({_delta(o.end - o.start)}), {o.change.value} "  # type: ignore[operator]
            f"(порог {o.threshold:g} — {o.threshold_source})")


def _delta(value: int) -> str:
    return "0" if value == 0 else f"{value:+d}"


def _sessions(sessions: tuple[int, ...]) -> str:
    return ("сессия " if len(sessions) == 1 else "сессии ") + ", ".join(map(str, sessions))
