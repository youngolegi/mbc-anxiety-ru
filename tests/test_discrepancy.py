"""Слой расхождений «самоотчёт × оценка терапевта». Все данные синтетические.

Базовый сценарий матрицы: GAD-7 с 12 (до обеих границ не меньше порога 4),
CGI-S с 4 (до обеих границ не меньше порога 2).
"""
import math
from datetime import datetime, timedelta, timezone

import pytest

from mbc.discrepancy import (ALWAYS_SHOWN, POINTER, QUESTIONS, BlockState, Direction, Discrepancy,
                             Kind, TherapistRating, Window, build_discrepancies, cgis_score,
                             classify_cell)
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mbc.measures import CGIS, GAD7, MEASURES, OASIS, PHQ8, PHQ9, WSAS
from mbc.reliable_change import (Change, ReliabilityKind, SampleThreshold, classify_change,
                                 threshold_from_sample)
from mbc.report import build_report, render_text
from mbc.safety import check_safety
from mbc.scoring import Score, ScoringError, score
from mbc.status import ScalePair, combined_outcome, outcome
from mbc.trajectory import rule_based_flag
from mbc.triggers import Reaction, TriggerEntry

B, N, W = Direction.BETTER, Direction.NONE, Direction.WORSE
T0 = datetime(2026, 10, 5, 19, 0)
SOURCE = "синтетический порог для теста"


def full(total, measure=GAD7):
    """Полный замер (без пропусков) с заданным суммарным баллом."""
    return Score(measure.code, total, measure.n_items, False, measure.severity(total))


def rating(session, code, blind=True):
    """Оценка CGI-S. blind=False — терапевт открыл самоотчёт за 50 минут до оценки."""
    at = T0 + timedelta(days=7 * session)
    return TherapistRating(session, cgis_score(code), at,
                           None if blind else at - timedelta(minutes=50))


def report_of(gad, entries=(), **kwargs):
    return build_report(GAD7, [(s, full(t)) for s, t in gad.items()], entries, **kwargs)


def block_of(gad, cgi, *, blind=True, cgi_threshold=2, entries=(), others=(), **kwargs):
    rep = report_of(gad, entries, **kwargs)
    ratings = [rating(s, v, blind) for s, v in cgi.items()]
    return build_discrepancies(rep, ratings, cgi_threshold=cgi_threshold,
                               cgi_threshold_source=SOURCE, other_reports=others)


def kinds(block):
    return [d.kind for d in block.discrepancies]


CLIENT = {B: 8, N: 12, W: 16}      # GAD-7 от 12: −4 / 0 / +4 — ровно на границе порога 4
THERAPIST = {B: 2, N: 4, W: 6}     # CGI-S от 4: −2 / 0 / +2 — ровно на границе порога 2
MATRIX = {
    (B, B): None, (B, N): Kind.CLIENT_BETTER_UNSEEN, (B, W): Kind.OPPOSITE,
    (N, B): Kind.THERAPIST_ONLY, (N, N): None, (N, W): Kind.THERAPIST_ONLY,
    (W, B): Kind.OPPOSITE, (W, N): Kind.CLIENT_WORSE_UNSEEN, (W, W): None,
}


def cell_block(c, t, **kwargs):
    return block_of({1: 12, 2: CLIENT[c]}, {1: 4, 2: THERAPIST[t]}, **kwargs)


# --- матрица -------------------------------------------------------------------------

@pytest.mark.parametrize("cell", MATRIX, ids=lambda c: f"{c[0].name}-{c[1].name}")
def test_all_nine_cells_at_threshold_boundaries(cell):
    block = cell_block(*cell)
    expected = MATRIX[cell]
    assert kinds(block) == ([] if expected is None else [expected])
    assert block.state is (BlockState.NONE_FOUND if expected is None else BlockState.FOUND)


@pytest.mark.parametrize("cell", MATRIX, ids=lambda c: f"{c[0].name}-{c[1].name}")
def test_classify_cell_matches_matrix(cell):
    c, t = cell
    c_worse = {B: -4, N: 0, W: 4}[c]
    t_worse = {B: -2, N: 0, W: 2}[t]
    assert classify_cell(c, c_worse, t, t_worse) is MATRIX[cell]


def test_just_below_thresholds_is_no_change():
    # GAD-7 −3 при пороге 4 и CGI-S −1 при пороге 2: обе стороны ниже порога — согласие
    assert block_of({1: 12, 2: 9}, {1: 4, 2: 3}).state is BlockState.NONE_FOUND


@pytest.mark.parametrize("gad_end,cgi_end", [
    (16, 5),    # клиент надёжно хуже, CGI-S +1 — тот же знак ниже порога
    (8, 3),     # клиент надёжно лучше, CGI-S −1
    (13, 6),    # CGI-S хуже, GAD-7 +1
    (9, 2),     # CGI-S лучше, GAD-7 −3
], ids=["клиент-хуже", "клиент-лучше", "терапевт-хуже", "терапевт-лучше"])
def test_agreement_below_threshold_is_not_a_discrepancy(gad_end, cgi_end):
    block = block_of({1: 12, 2: gad_end}, {1: 4, 2: cgi_end})
    assert block.state is BlockState.NONE_FOUND and not block.discrepancies


@pytest.mark.parametrize("gad_end,cgi_end,kind", [
    (16, 3, Kind.CLIENT_WORSE_UNSEEN),     # CGI-S −1 — противоположный знак
    (8, 5, Kind.CLIENT_BETTER_UNSEEN),
    (11, 6, Kind.THERAPIST_ONLY),          # GAD-7 −1 при CGI-S хуже
    (13, 2, Kind.THERAPIST_ONLY),
])
def test_opposite_sign_below_threshold_is_a_discrepancy(gad_end, cgi_end, kind):
    assert kinds(block_of({1: 12, 2: gad_end}, {1: 4, 2: cgi_end})) == [kind]


def test_therapist_side_uses_direction_not_change():
    d = cell_block(W, N).discrepancies[0]
    assert d.client_change is Change.DETERIORATED and d.therapist_direction is Direction.NONE
    assert isinstance(d.therapist_direction, Direction)
    assert not any("надежн" in direction.value.lower().replace("ё", "е") for direction in Direction)


# --- клиентская сторона — та же, что у статуса --------------------------------------

def test_client_side_uses_report_threshold_and_source():
    # порог 5 вместо 4: изменение на 4 — уже не надёжное, расхождения нет
    block = block_of({1: 12, 2: 16}, {1: 4, 2: 4}, threshold=5, threshold_source=SOURCE)
    assert block.state is BlockState.NONE_FOUND
    assert f"  порог по самоотчёту — как у статуса: 5 — {SOURCE}" in block.lines()


def test_client_change_equals_status_change():
    rep = report_of({1: 12, 2: 10, 3: 16})
    block = build_discrepancies(rep, [rating(s, 4) for s in (1, 2, 3)], cgi_threshold=2,
                                cgi_threshold_source=SOURCE)
    d = block.discrepancies[0]
    assert d.client_change is classify_change(12, 16, rep.threshold)
    assert rep.outcome.value == d.client_change.value == "надёжное ухудшение"


def test_client_side_accepts_sample_threshold_of_report():
    sample = threshold_from_sample([full(t) for t in (0, 5, 10)], 0.92,
                                   kind=ReliabilityKind.OMEGA, reliability_n=3)
    rep = report_of({1: 12, 2: 16}, threshold=sample)
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=2,
                                cgi_threshold_source=SOURCE)
    assert kinds(block) == [Kind.CLIENT_WORSE_UNSEEN]
    assert any(line.endswith(sample.source) and "как у статуса" in line for line in block.lines())


def test_prorated_self_report_is_marked_in_the_record():
    prorated = score(GAD7, [3, 3, None, 2, 2, 2, 2], max_missing=1)      # 14 · 7/6 → 16
    rep = build_report(GAD7, [(1, full(12)), (2, prorated)])
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=2,
                                cgi_threshold_source=SOURCE)
    (d,) = block.discrepancies
    assert d.client_prorated == (2,) and d.client_end == 16
    assert any(line.startswith("     самоотчёт GAD7: 12 → 16 (+4)")
               and line.endswith("; пересчитан пропорционально: сессия 2") for line in block.lines())


@pytest.mark.parametrize("cgi,reason", [
    ({2: 4, 3: 4}, "нет оценки терапевта"),
    ({1: 0, 2: 4, 3: 4}, "оценка терапевта не засчитана"),
], ids=["нет оценки", "код 0"])
def test_block_base_differs_from_status_base(cgi, reason):
    block = block_of({1: 14, 2: 12, 3: 16}, cgi)
    assert block.pairs == (2, 3)
    assert (f"  сравнение по сессиям 2 → 3: в сессии 1 {reason}; база статуса и светофора — "
            f"сессия 1") in block.lines()


def test_same_base_has_no_base_line():
    assert not any("база статуса" in line for line in cell_block(W, N).lines())


def test_lag_line_when_latest_self_report_has_no_rating():
    block = block_of({1: 12, 2: 13, 3: 16}, {1: 4, 2: 4})
    assert block.state is BlockState.NONE_FOUND
    assert "  самоотчёт сессии 3 ещё без оценки терапевта — текущая пара: сессия 2" in block.lines()


def test_no_lag_line_when_latest_is_paired():
    assert not any("ещё без оценки" in line for line in cell_block(W, N).lines())


# --- пол и потолок ---------------------------------------------------------------------

@pytest.mark.parametrize("gad,cgi,note", [
    ({1: 16, 2: 10}, {1: 2, 2: 2},
     "улучшение по оценке терапевта (CGI-S) не определяется: в сессии 1 — 2, "
     "до границы шкалы меньше порога 2"),
    ({1: 10, 2: 16}, {1: 6, 2: 6},
     "ухудшение по оценке терапевта (CGI-S) не определяется: в сессии 1 — 6, "
     "до границы шкалы меньше порога 2"),
    ({1: 2, 2: 3}, {1: 5, 2: 3},
     "улучшение по самоотчёту GAD7 не определяется: в сессии 1 — 2, "
     "до границы шкалы меньше порога 4"),
    ({1: 19, 2: 18}, {1: 3, 2: 5},
     "ухудшение по самоотчёту GAD7 не определяется: в сессии 1 — 19, "
     "до границы шкалы меньше порога 4"),
], ids=["CGI-пол", "CGI-потолок", "GAD-пол", "GAD-потолок"])
def test_unreachable_direction_gives_no_discrepancy_and_a_reason(gad, cgi, note):
    block = block_of(gad, cgi)
    assert block.state is BlockState.NONE_FOUND
    assert note in block.notes and f"  {note}" in block.lines()


def test_reachable_at_boundary_still_counts():
    # CGI-S с 3: до 1 ровно 2 балла — улучшение достижимо, расхождение выводится
    assert kinds(block_of({1: 16, 2: 10}, {1: 3, 2: 3})) == [Kind.CLIENT_BETTER_UNSEEN]


def test_unreachable_note_shown_even_without_discrepancy():
    block = block_of({1: 12, 2: 12}, {1: 1, 2: 1})
    assert block.state is BlockState.NONE_FOUND
    assert any(n.startswith("улучшение по оценке терапевта") for n in block.notes)


# --- окно «предыдущая пара → текущая» -------------------------------------------------

RELAPSE_GAD = {1: 16, 2: 10, 3: 6, 4: 11}
RELAPSE_CGI = {1: 5, 2: 4, 3: 3, 4: 2}


def test_relapse_after_early_gains_is_seen_by_adjacent_window():
    # от базы обе стороны «лучше» — согласие; между последними сессиями самоотчёт +5
    block = block_of(RELAPSE_GAD, RELAPSE_CGI)
    (d,) = block.discrepancies
    assert (d.kind, d.window, d.from_session, d.to_session) == \
        (Kind.CLIENT_WORSE_UNSEEN, Window.ADJACENT, 3, 4)
    assert d.client_change is Change.DETERIORATED


def test_adjacent_window_says_traffic_light_counts_from_baseline():
    lines = block_of(RELAPSE_GAD, RELAPSE_CGI).lines()
    assert any(line.endswith("; светофор и статус считают по сессиям 1 → 4") for line in lines)


def test_adjacent_window_only_for_client_deterioration():
    # между последними сессиями клиенту лучше, терапевт без сдвига — окно 2 молчит
    block = block_of({1: 12, 2: 16, 3: 12}, {1: 4, 2: 4, 3: 4})
    assert block.state is BlockState.NONE_FOUND


def test_adjacent_window_skipped_when_window_one_already_shows_client_deterioration():
    # окно 1 уже вывело «самоотчёт хуже, терапевт без сдвига» — окно 2 его не дублирует
    block = block_of({1: 10, 2: 12, 3: 16}, {1: 3, 2: 3, 3: 3})
    assert [(d.kind, d.window) for d in block.discrepancies] == \
        [(Kind.CLIENT_WORSE_UNSEEN, Window.BASE)]


def test_agreement_in_window_one_does_not_hide_the_last_step():
    # от базы оба «хуже» — согласие; на последнем шаге самоотчёт +4, CGI-S без сдвига
    block = block_of({1: 10, 2: 12, 3: 16}, {1: 3, 2: 5, 3: 5})
    assert [(d.kind, d.window) for d in block.discrepancies] == \
        [(Kind.CLIENT_WORSE_UNSEEN, Window.ADJACENT)]


def test_shifted_block_base_does_not_hide_always_shown_kind():
    # CGI-S с сессии 2: от базы блока (2 → 4) оба «хуже» — согласие, светофор зелёный
    # (от сессии 1: 20 → 12); на последнем шаге самоотчёт +6, а CGI-S −2 — противоположны
    block = block_of({1: 20, 2: 4, 3: 6, 4: 12}, {2: 2, 3: 6, 4: 4})
    assert block.report.flag.signal.name == "GREEN"
    assert [(d.kind, d.window, d.from_session) for d in block.discrepancies] == \
        [(Kind.OPPOSITE, Window.ADJACENT, 3)]


def test_adjacent_window_needs_three_pairs():
    block = block_of({1: 6, 2: 11}, {1: 3, 2: 2})
    assert [d.window for d in block.discrepancies] == [Window.BASE]


def test_adjacent_unreachable_gives_reason():
    # от базы 16 → 10 — клиенту лучше (своё расхождение); 4 → 10 — хуже, но CGI-S на 6
    block = block_of({1: 16, 2: 8, 3: 4, 4: 10}, {1: 6, 2: 6, 3: 6, 4: 6})
    assert [d.window for d in block.discrepancies] == [Window.BASE]
    i = block.notes.index("предыдущая пара → текущая (сессии 3 → 4): самоотчёт GAD7 4 → 10, "
                          "надёжное ухудшение")
    assert block.notes[i + 1] == ("↳ ухудшение по оценке терапевта (CGI-S) не определяется: "
                                  "в сессии 3 — 6, до границы шкалы меньше порога 2")


# --- устойчивость persisted -----------------------------------------------------------

def test_persisted_is_one_at_first_appearance():
    assert cell_block(W, N).discrepancies[0].persisted == 1


def test_persisted_counts_consecutive_comparisons():
    block = block_of({1: 12, 2: 13, 3: 16, 4: 17, 5: 18}, {k: 4 for k in range(1, 6)})
    assert block.discrepancies[0].persisted == 3


def test_persisted_reset_by_agreement():
    # сессия 4: CGI-S +2 — согласие; затем снова расхождение
    block = block_of({1: 12, 2: 16, 3: 16, 4: 16, 5: 16}, {1: 4, 2: 4, 3: 4, 4: 6, 5: 4})
    assert block.discrepancies[0].persisted == 1


def test_persisted_reset_by_direction_change():
    # OPPOSITE с клиентом «хуже» → OPPOSITE с клиентом «лучше»: подпись другая
    block = block_of({1: 12, 2: 16, 3: 8}, {1: 4, 2: 2, 3: 6})
    d = block.discrepancies[0]
    assert d.kind is Kind.OPPOSITE and d.client_change is Change.IMPROVED and d.persisted == 1


def test_unpaired_sessions_do_not_break_persisted():
    block = block_of({1: 12, 2: 16, 3: 16, 4: 17}, {1: 4, 2: 4, 4: 4})   # в сессии 3 нет оценки
    assert block.pairs == (1, 2, 4) and block.discrepancies[0].persisted == 2
    assert "  самоотчёт без оценки терапевта: сессия 3" in block.lines()


@pytest.mark.parametrize("kind", [Kind.CLIENT_BETTER_UNSEEN, Kind.THERAPIST_ONLY])
def test_minor_kinds_collapsed_until_they_persist(kind):
    gad, cgi = (({1: 12, 2: 8, 3: 7}, {1: 4, 2: 4, 3: 4}) if kind is Kind.CLIENT_BETTER_UNSEEN
                else ({1: 12, 2: 12, 3: 12}, {1: 4, 2: 6, 3: 6}))
    once = block_of({1: gad[1], 2: gad[2]}, {1: cgi[1], 2: cgi[2]})
    twice = block_of(gad, cgi)
    assert kinds(once) == kinds(twice) == [kind]
    assert once.discrepancies[0].collapsed and not twice.discrepancies[0].collapsed
    collapsed_line = [line for line in once.lines() if "свёрнуто:" in line]
    assert len(collapsed_line) == 1
    assert collapsed_line[0].endswith("— не держится двух сравнений подряд; развернётся, если "
                                      "повторится в следующем сравнении")
    assert not any("вопрос для сессии" in line for line in once.lines())
    assert any("вопрос для сессии" in line for line in twice.lines())


@pytest.mark.parametrize("cell", [(W, N), (B, W), (W, B)], ids=["CLIENT_WORSE", "OPP-1", "OPP-2"])
def test_always_shown_kinds_expanded_at_first_appearance(cell):
    d = cell_block(*cell).discrepancies[0]
    assert d.kind in ALWAYS_SHOWN and d.persisted == 1 and not d.collapsed


# --- слепота оценок -----------------------------------------------------------------

def test_blind_is_computed_from_timestamps():
    at = T0
    assert TherapistRating(1, cgis_score(4), at, None).blind
    assert TherapistRating(1, cgis_score(4), at, at + timedelta(minutes=1)).blind
    assert not TherapistRating(1, cgis_score(4), at, at - timedelta(minutes=1)).blind
    assert not TherapistRating(1, cgis_score(4), at, at).blind


def test_comparison_blind_only_if_both_ratings_blind():
    rep = report_of({1: 12, 2: 16})
    cases = [((True, True), ()), ((True, False), (2,)), ((False, False), (1, 2))]
    for (b1, b2), non_blind in cases:
        block = build_discrepancies(rep, [rating(1, 4, b1), rating(2, 4, b2)], cgi_threshold=2,
                                    cgi_threshold_source=SOURCE)
        d = block.discrepancies[0]
        assert d.non_blind == non_blind and d.blind is (not non_blind)


BLIND_LINE = "  оценок терапевта после просмотра самоотчёта"


def test_blind_line_counts_pairs():
    rep = report_of({1: 12, 2: 13, 3: 12})
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 4, False), rating(3, 4, False)],
                                cgi_threshold=2, cgi_threshold_source=SOURCE)
    assert block.state is BlockState.NONE_FOUND
    assert (f"{BLIND_LINE}: 2 из 3 в парах; совпадение направлений по ним менее информативно"
            in block.lines())


# --- четыре состояния блока ---------------------------------------------------------

HEADS = {
    BlockState.NOT_PROVIDED: "РАСХОЖДЕНИЯ — темы для разговора: оценки терапевта не переданы",
    BlockState.INSUFFICIENT: "РАСХОЖДЕНИЯ — темы для разговора: недостаточно пар",
    BlockState.NONE_FOUND: "РАСХОЖДЕНИЯ — темы для разговора: расхождений нет",
    BlockState.FOUND: "РАСХОЖДЕНИЯ — темы для разговора: найдено",
}


def blocks_in_every_state():
    rep = report_of({1: 12, 2: 16})
    build = lambda ratings: build_discrepancies(rep, ratings, cgi_threshold=2,  # noqa: E731
                                                cgi_threshold_source=SOURCE)
    return {
        BlockState.NOT_PROVIDED: build(None),
        BlockState.INSUFFICIENT: build([rating(1, 4)]),
        BlockState.NONE_FOUND: build([rating(1, 4), rating(2, 6)]),
        BlockState.FOUND: build([rating(1, 4), rating(2, 4)]),
    }


@pytest.mark.parametrize("state", list(BlockState), ids=lambda s: s.name)
def test_block_states_are_exclusive(state):
    block = blocks_in_every_state()[state]
    lines = block.lines()
    assert block.state is state
    assert lines[0].startswith(HEADS[state])
    assert [s for s, head in HEADS.items() if any(line.startswith(head) for line in lines)] == [state]


@pytest.mark.parametrize("state", list(BlockState), ids=lambda s: s.name)
def test_blind_line_in_every_state(state):
    assert sum(line.startswith(BLIND_LINE) for line in blocks_in_every_state()[state].lines()) == 1


def test_empty_ratings_are_insufficient_not_missing():
    rep = report_of({1: 12, 2: 16})
    block = build_discrepancies(rep, [], cgi_threshold=2, cgi_threshold_source=SOURCE)
    assert block.state is BlockState.INSUFFICIENT
    assert block.lines()[0] == ("РАСХОЖДЕНИЯ — темы для разговора: недостаточно пар — "
                                "нужно минимум две, есть 0")


def test_insufficient_lists_unpaired_sessions():
    block = block_of({1: 12, 2: 16}, {1: 4, 3: 4})
    assert block.state is BlockState.INSUFFICIENT
    lines = block.lines()
    assert "  самоотчёт без оценки терапевта: сессия 2" in lines
    assert "  оценка терапевта без засчитанного самоотчёта: сессия 3" in lines
    assert "  пары (засчитаны и самоотчёт, и оценка терапевта): 1 из 3 сессий; пропуски могут " \
           "быть неслучайными" in lines


# --- вход ----------------------------------------------------------------------------

def test_cgis_code_zero_means_not_rated():
    s = cgis_score(0)
    assert (s.measure, s.total, s.n_answered) == ("CGIS", None, 0)
    assert cgis_score(None).total is None and cgis_score(5).total == 5


@pytest.mark.parametrize("code", [False, True, 8, -1, 0.0, 4.0, "4"])
def test_cgis_score_rejects_non_codes(code):
    with pytest.raises(ScoringError):
        cgis_score(code)


def test_not_counted_rating_is_listed_separately_from_unpaired():
    block = block_of({1: 12, 2: 13, 3: 16}, {1: 4, 2: 0, 3: 4})
    assert block.ratings_not_counted == (2,) and block.client_only == () and block.pairs == (1, 3)
    assert "  оценка терапевта не засчитана: сессия 2" in block.lines()


def test_two_ratings_in_one_session_rejected():
    rep = report_of({1: 12, 2: 16})
    with pytest.raises(ValueError, match="Больше одной оценки"):
        build_discrepancies(rep, [rating(1, 4), rating(2, 4), rating(2, 5)], cgi_threshold=2,
                            cgi_threshold_source=SOURCE)


@pytest.mark.parametrize("bad", [
    TherapistRating(1, full(5), T0, None),                          # чужая шкала, балл в 1–7
    TherapistRating(1, Score("CGIS", 8, 1, False, None), T0, None),  # вне 1–7
    TherapistRating(1, Score("CGIS", 0, 1, False, None), T0, None),  # код 0 мимо cgis_score
    TherapistRating(1, Score("CGIS", True, 1, False, None), T0, None),
    TherapistRating(0, cgis_score(4), T0, None),                    # сессия 0
    TherapistRating(True, cgis_score(4), T0, None),
    TherapistRating(1, cgis_score(4), "2026-10-05", None),          # не datetime
    TherapistRating(1, cgis_score(4), T0, T0.replace(tzinfo=timezone.utc)),  # несравнимы
    (1, cgis_score(4)),                                              # не TherapistRating
], ids=["шкала", "диапазон", "код-0", "bool", "сессия-0", "сессия-bool", "время", "пояс", "кортеж"])
def test_invalid_ratings_rejected(bad):
    rep = report_of({1: 12, 2: 16})
    with pytest.raises(ValueError):
        build_discrepancies(rep, [bad, rating(2, 4)], cgi_threshold=2, cgi_threshold_source=SOURCE)


def test_iterator_ratings_are_not_lost():
    rep = report_of({1: 12, 2: 16})
    ratings = [rating(1, 4), rating(2, 4)]
    kw = dict(cgi_threshold=2, cgi_threshold_source=SOURCE)
    assert build_discrepancies(rep, iter(ratings), **kw) == build_discrepancies(rep, ratings, **kw)


def test_report_is_required():
    with pytest.raises(ValueError, match="ClientReport"):
        build_discrepancies([(1, full(12))], [], cgi_threshold=2, cgi_threshold_source=SOURCE)


# --- порог CGI-S ----------------------------------------------------------------------

@pytest.mark.parametrize("value", [True, 0, 7, 1.5, math.nan, "2", None, -2])
def test_invalid_cgi_threshold_rejected(value):
    with pytest.raises(ValueError):
        block_of({1: 12, 2: 16}, {1: 4, 2: 4}, cgi_threshold=value)


@pytest.mark.parametrize("source", [None, "", "   ", 5])
def test_cgi_threshold_needs_source(source):
    rep = report_of({1: 12, 2: 16})
    with pytest.raises(ValueError, match="источник"):
        build_discrepancies(rep, [], cgi_threshold=2, cgi_threshold_source=source)


def test_cgi_threshold_and_source_have_no_defaults():
    rep = report_of({1: 12, 2: 16})
    with pytest.raises(TypeError):
        build_discrepancies(rep, [], cgi_threshold=2)
    with pytest.raises(TypeError):
        build_discrepancies(rep, [], cgi_threshold_source=SOURCE)


CGIS_SAMPLE = [full(t, CGIS) for t in (2, 3, 4, 5, 6)]   # SD = 1,58


def test_cgi_threshold_from_retest_sample():
    t = threshold_from_sample(CGIS_SAMPLE, 0.6, kind=ReliabilityKind.RETEST, reliability_n=40)
    assert t.min_change_points == 3                         # 1,96·√2·1,58·√0,4 ≈ 2,77 → 3
    rep = report_of({1: 12, 2: 16})
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 2)], cgi_threshold=t,
                                cgi_threshold_source=None)
    assert (block.cgi_threshold, block.cgi_threshold_source) == (3, t.source)
    # CGI-S −2: при пороге 2 это «лучше» (противоположны), при пороге 3 — сдвиг меньше порога
    assert kinds(block) == [Kind.CLIENT_WORSE_UNSEEN]


def test_sample_threshold_rejects_second_source():
    t = threshold_from_sample(CGIS_SAMPLE, 0.6, kind=ReliabilityKind.RETEST, reliability_n=40)
    rep = report_of({1: 12, 2: 16})
    with pytest.raises(ValueError, match="источник уже есть"):
        build_discrepancies(rep, [], cgi_threshold=t, cgi_threshold_source=SOURCE)


@pytest.mark.parametrize("measure,kind", [("CGIS", ReliabilityKind.OMEGA),
                                          ("CGIS", ReliabilityKind.ALPHA),
                                          ("GAD7", ReliabilityKind.RETEST)])
def test_sample_threshold_must_be_cgis_retest(measure, kind):
    t = SampleThreshold(measure, 5, 0, 4.0, 1.58, 0.6, kind, 5, 1.96, 1.0, 1.41, 2.77)
    rep = report_of({1: 12, 2: 16})
    with pytest.raises(ValueError):
        build_discrepancies(rep, [], cgi_threshold=t, cgi_threshold_source=None)


def test_sample_threshold_beyond_scale_rejected():
    t = SampleThreshold("CGIS", 5, 0, 4.0, 5.0, 0.1, ReliabilityKind.RETEST, 5, 1.96, 4.7, 6.7, 13.1)
    rep = report_of({1: 12, 2: 16})
    with pytest.raises(ValueError, match="от 1 до 6"):
        build_discrepancies(rep, [], cgi_threshold=t, cgi_threshold_source=None)


def test_threshold_one_is_marked_as_sensitivity_analysis():
    lines = cell_block(W, N, cgi_threshold=1).lines()
    assert any("1 балл — только анализ чувствительности" in line for line in lines)
    assert not any("анализ чувствительности" in line for line in cell_block(W, N).lines())


def test_threshold_one_changes_the_cell():
    # CGI-S −1: при пороге 2 — сдвиг меньше порога, при пороге 1 — «лучше» → противоположны
    assert kinds(block_of({1: 12, 2: 16}, {1: 4, 2: 3})) == [Kind.CLIENT_WORSE_UNSEEN]
    assert kinds(block_of({1: 12, 2: 16}, {1: 4, 2: 3}, cgi_threshold=1)) == [Kind.OPPOSITE]


# --- шкала клинициста не проходит в API самоотчёта ----------------------------------

def test_cgis_registered_as_clinician_measure():
    assert MEASURES["CGIS"] is CGIS and CGIS.rater == "clinician"
    assert {m.rater for code, m in MEASURES.items() if code != "CGIS"} == {"client"}
    assert (CGIS.n_items, CGIS.item_min, CGIS.item_max) == (1, 1, 7)
    assert CGIS.caseness is None and CGIS.reliable_change is None
    assert CGIS.caseness_source and CGIS.reliable_change_source and CGIS.russian_version


def test_clinician_measure_rejected_by_report():
    with pytest.raises(ValueError, match="оценка клинициста"):
        build_report(CGIS, [(1, cgis_score(4)), (2, cgis_score(3))], threshold=2,
                     threshold_source=SOURCE)


def test_clinician_measure_rejected_by_outcome():
    with pytest.raises(ValueError, match="оценка клинициста"):
        outcome(CGIS, 5, 3, threshold=2)


def test_clinician_measure_rejected_by_trajectory():
    with pytest.raises(ValueError, match="оценка клинициста"):
        rule_based_flag(CGIS, [(1, 5), (2, 3)], threshold=2)


def test_clinician_measure_rejected_by_combined_outcome():
    with pytest.raises(ValueError, match="оценка клинициста"):
        combined_outcome(ScalePair(GAD7, 12, 6), ScalePair(CGIS, 5, 3, threshold=2),
                         finished_course=True)


@pytest.mark.parametrize("kind", [ReliabilityKind.OMEGA, ReliabilityKind.ALPHA])
def test_single_item_threshold_only_from_retest(kind):
    with pytest.raises(ValueError, match="только ретестовая"):
        threshold_from_sample(CGIS_SAMPLE, 0.6, kind=kind, reliability_n=5)


# --- указатель при ухудшении клиента ------------------------------------------------

def record_lines(block, i=1):
    """Строки записи i: от «  i. » до следующей записи."""
    lines = block.lines()
    start = next(k for k, line in enumerate(lines) if line.startswith(f"  {i}. "))
    end = next((k for k, line in enumerate(lines[start + 1:], start + 1)
                if line.startswith("  ") and line[2:3].isdigit()), len(lines))
    return lines[start:end]


@pytest.mark.parametrize("cell", [(W, N), (W, B)], ids=["CLIENT_WORSE", "OPPOSITE"])
def test_pointer_is_first_line_on_client_deterioration(cell):
    lines = record_lines(cell_block(*cell))
    assert lines[0] == f"  1. {POINTER}"
    assert sum(POINTER in line for line in lines) == 1


def test_pointer_in_adjacent_window():
    assert record_lines(block_of(RELAPSE_GAD, RELAPSE_CGI))[0] == f"  1. {POINTER}"


@pytest.mark.parametrize("cell", [(B, W), (N, W), (N, B)])
def test_no_pointer_without_client_deterioration(cell):
    block = block_of({1: 12, 2: CLIENT[cell[0]], 3: CLIENT[cell[0]]},
                     {1: 4, 2: THERAPIST[cell[1]], 3: THERAPIST[cell[1]]})
    assert block.discrepancies and not any(POINTER in line for line in block.lines())


# --- альтернативы и вопросы ---------------------------------------------------------

DISCREPANT = [cell for cell, kind in MATRIX.items() if kind is not None]


def expanded(c, t, start=1):
    """Блок, где расхождение развёрнуто: клетка держится два сравнения подряд."""
    return block_of({start: 12, start + 1: CLIENT[c], start + 2: CLIENT[c]},
                    {start: 4, start + 1: THERAPIST[t], start + 2: THERAPIST[t]})


@pytest.mark.parametrize("cell", DISCREPANT, ids=lambda c: f"{c[0].name}-{c[1].name}")
def test_literal_reading_first_and_question_by_cell(cell):
    d = expanded(*cell).discrepancies[0]
    first = d.alternatives[0]
    assert first.startswith("Возможно,") and ("реальн" in first or "действительно" in first)
    assert d.question == QUESTIONS[d.kind]
    assert any("разных окнах" in a for a in d.alternatives)
    assert any("измеряют разное" in a for a in d.alternatives)


def test_regression_to_mean_only_for_early_client_improvement():
    early = expanded(B, N).discrepancies[0]
    late = expanded(B, N, start=3).discrepancies[0]
    worse = expanded(W, N).discrepancies[0]
    assert any("регрессия к среднему" in a for a in early.alternatives)
    assert not any("регрессия к среднему" in a for a in late.alternatives + worse.alternatives)


def test_fuller_picture_only_from_session_one_with_therapist_worse():
    first = expanded(N, W).discrepancies[0]
    later = expanded(N, W, start=2).discrepancies[0]
    better = expanded(N, B).discrepancies[0]
    assert any("картина проблем" in a for a in first.alternatives)
    assert not any("картина проблем" in a for a in later.alternatives + better.alternatives)


def test_optimism_and_alliance_only_when_therapist_view_is_more_favourable():
    for cell, expected in [((W, N), True), ((W, B), True), ((N, B), True),
                           ((B, N), False), ((B, W), False), ((N, W), False)]:
        alts = expanded(*cell).discrepancies[0].alternatives
        assert any("Hannan" in a for a in alts) is expected, cell
        assert any("в отношениях с терапевтом" in a for a in alts) is expected, cell


START = datetime(2026, 9, 7, 20, 0)   # понедельник


def entry(i, day, intensity, reaction):
    return TriggerEntry(f"t{i}", START + timedelta(days=day), intensity, before="звонок",
                        reaction=reaction)


APPROACH = [entry(1, 0, 6, Reaction.AVOIDED), entry(2, 1, 6, Reaction.AVOIDED),
            entry(3, 7, 6, Reaction.STAYED), entry(4, 8, 5, Reaction.STAYED)]
MORE_INTENSE = [entry(1, 0, 4, Reaction.AVOIDED), entry(2, 1, 4, Reaction.AVOIDED),
                entry(3, 7, 7, Reaction.STAYED), entry(4, 8, 7, Reaction.STAYED)]
MORE_AVOIDANT = [entry(1, 0, 6, Reaction.STAYED), entry(2, 1, 6, Reaction.STAYED),
                 entry(3, 7, 5, Reaction.AVOIDED), entry(4, 8, 5, Reaction.AVOIDED)]


def exposure_alts(entries, cell=(W, N)):
    block = block_of({1: 12, 2: CLIENT[cell[0]]}, {1: 4, 2: THERAPIST[cell[1]]},
                     entries=entries, min_episodes=3)
    return [a for d in block.discrepancies for a in d.alternatives if "экспозиция" in a]


def test_exposure_alternative_needs_supporting_trigger_data():
    (alt,) = exposure_alts(APPROACH)
    assert alt.startswith("Если в недели паттерна «за час до: звонок» (07.09–20.09.2026) шла "
                          "экспозиция, дистресс мог временно вырасти: по карте триггеров")
    assert alt.endswith("— сверить, попадают ли эти недели в окно сравнения.")
    assert "избегание 100% → 0% (реакция отмечена в 2 и 2 записях)" in alt
    assert "интенсивность 6 → 5.5" in alt


@pytest.mark.parametrize("entries", [(), MORE_INTENSE, MORE_AVOIDANT],
                         ids=["без-записей", "интенсивность-выросла", "избегание-выросло"])
def test_no_exposure_alternative_without_support(entries):
    assert exposure_alts(entries) == []


def test_no_exposure_alternative_when_client_not_worse():
    assert exposure_alts(APPROACH, cell=(N, W)) == []


def test_other_scales_shown_for_the_same_window():
    phq = build_report(PHQ8, [(1, full(14, PHQ8)), (2, full(8, PHQ8))], threshold=5,
                       threshold_source=SOURCE)
    block = block_of({1: 12, 2: 16}, {1: 4, 2: 4}, others=[phq])
    (other,) = block.discrepancies[0].others
    assert (other.measure, other.start, other.end, other.change) == ("PHQ8", 14, 8, Change.IMPROVED)
    assert f"       PHQ8: 14 → 8 (-6), надёжное улучшение (порог 5 — {SOURCE})" in block.lines()
    assert not any("сравнение только со шкалой отчёта" in line for line in block.lines())


def test_other_scale_without_scores_in_window():
    phq = build_report(PHQ9, [(1, full(14, PHQ9))])
    block = block_of({1: 12, 2: 16}, {1: 4, 2: 4}, others=[phq])
    assert block.discrepancies[0].others[0].change is None
    assert "       PHQ9: нет засчитанного замера в сессии 2" in block.lines()


def test_without_other_scales_block_says_so():
    assert ("  другие самоотчёты не переданы — сравнение только со шкалой отчёта (GAD7)"
            in cell_block(W, N).lines())


@pytest.mark.parametrize("others", [
    lambda: [report_of({1: 12})],                                          # шкала отчёта
    lambda: [build_report(PHQ9, []), build_report(PHQ9, [])],              # повтор
    lambda: [[(1, full(14, PHQ9))]],                                       # не ClientReport
], ids=["шкала-отчёта", "повтор", "не-отчёт"])
def test_invalid_other_reports_rejected(others):
    with pytest.raises(ValueError):
        block_of({1: 12, 2: 16}, {1: 4, 2: 4}, others=others())


def test_discrepancy_contract():
    base = dict(kind=Kind.OPPOSITE, window=Window.BASE, from_session=1, to_session=2,
                client_start=12, client_end=16, client_change=Change.DETERIORATED,
                client_prorated=(), therapist_start=4, therapist_end=2,
                therapist_direction=Direction.BETTER, non_blind=(), persisted=1,
                alternatives=("Возможно, x.",), question="Вопрос?")
    Discrepancy(**base)
    for bad in (dict(alternatives=()), dict(question="  "), dict(persisted=0)):
        with pytest.raises(ValueError):
            Discrepancy(**{**base, **bad})


# --- язык -----------------------------------------------------------------------------

BANNED = ("преувелич", "преуменьш", "занижа", "завыша", "минимизир", "агграв", "симул",
          "диссимул", "скрыва", "скрыл", "сопротивл", "недостовер", "неправ", "ошиба", "ошибк",
          "упустил", "не заметил", "на самом деле")
MODAL = ("может", "возможно", "если")


def every_rendered_block():
    blocks = [expanded(*cell) for cell in DISCREPANT]
    blocks += [expanded(*cell, start=2) for cell in DISCREPANT]
    blocks += [block_of(RELAPSE_GAD, RELAPSE_CGI),
               block_of({1: 12, 2: 16}, {1: 4, 2: 4}, entries=APPROACH, min_episodes=3)]
    blocks += [cell_block(*cell) for cell in MATRIX]
    blocks += list(blocks_in_every_state().values())
    blocks += extra_branch_blocks()
    return blocks


def test_no_banned_stems_anywhere_in_the_block():
    for block in every_rendered_block():
        text = "\n".join(block.lines()).lower()
        assert [s for s in BANNED if s in text] == []


def test_therapist_lines_never_say_reliable():
    for block in every_rendered_block():
        for line in block.lines():
            text = line.lower().replace("ё", "е")
            if "терапевт" in text:
                assert "надежн" not in text, line


def test_every_alternative_is_modal():
    for block in every_rendered_block():
        for d in block.discrepancies:
            for alt in d.alternatives:
                assert any(m in alt.lower() for m in MODAL), alt


def test_all_matrix_kinds_were_rendered_expanded():
    seen = {d.kind for b in every_rendered_block() for d in b.discrepancies if not d.collapsed}
    assert seen == set(Kind)


# --- место в отчёте -------------------------------------------------------------------

def test_report_without_block_is_unchanged():
    rep = report_of({1: 12, 2: 16})
    assert "РАСХОЖДЕНИЯ" not in render_text(rep)


def test_block_goes_after_status_and_changes_nothing_above():
    rep = report_of({1: 12, 2: 16})
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=2,
                                cgi_threshold_source=SOURCE)
    plain = render_text(rep).splitlines()
    with_block = render_text(rep, block).splitlines()
    head = with_block.index(block.lines()[0])
    assert with_block[:head - 1] == plain[:head - 1]            # шапка, светофор, статус — те же
    assert with_block[head - 1] == ""
    assert with_block[head:head + len(block.lines())] == block.lines()
    assert with_block[head + len(block.lines()) + 1].startswith("ПАТТЕРНЫ")
    assert with_block[3] == plain[3] and with_block[3].startswith("БЕЗОПАСНОСТЬ")


def test_block_of_another_report_rejected():
    block = build_discrepancies(report_of({1: 12, 2: 16}), None, cgi_threshold=2,
                                cgi_threshold_source=SOURCE)
    with pytest.raises(ValueError, match="другому отчёту"):
        render_text(report_of({1: 12, 2: 15}), block)


# --- находки независимого ревью ----------------------------------------------------------

def test_collapsed_line_does_not_claim_first_appearance_after_reset():
    # 1 → 2: только терапевт; 1 → 3: согласие; 1 → 4: снова только терапевт
    block = block_of({1: 12, 2: 12, 3: 12, 4: 12}, {1: 4, 2: 6, 3: 5, 4: 6})
    (d,) = block.discrepancies
    assert d.collapsed and d.persisted == 1
    assert not any("впервые" in line for line in block.lines())


@pytest.mark.parametrize("ratings,points", [
    ((3, 4, 4, 5), 1),     # ICC 0,9 → порог 0,72 → 1 балл
    ((2, 3, 4, 5, 6), 2),  # ICC 0,9 → порог 1,39 → 2 балла
])
def test_sample_cgi_threshold_is_not_labelled_heuristic(ratings, points):
    t = threshold_from_sample([full(v, CGIS) for v in ratings], 0.9, kind=ReliabilityKind.RETEST,
                              reliability_n=60)
    assert t.min_change_points == points
    rep = report_of({1: 12, 2: 16})
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=t,
                                cgi_threshold_source=None)
    assert block.cgi_threshold_from_sample
    line = next(line for line in block.lines() if "порог сдвига оценки терапевта" in line)
    assert line == (f"  порог сдвига оценки терапевта (CGI-S): {points} (расчёт "
                    f"{t.threshold:.3g}; в баллах — вверх до целого) — {t.source}")
    assert block.cgi_threshold_raw == t.threshold


def test_manual_cgi_threshold_line_has_source_and_heuristic_label():
    lines = cell_block(W, N).lines()
    assert f"  порог сдвига оценки терапевта (CGI-S): 2 — {SOURCE}; эвристика для одного пункта" \
        in lines
    assert (f"  порог сдвига оценки терапевта (CGI-S): 1 — {SOURCE}; эвристика для одного пункта; "
            f"1 балл — только анализ чувствительности") in cell_block(W, N, cgi_threshold=1).lines()


@pytest.mark.parametrize("value", [math.inf, math.nan, 0.0, -1.0])
def test_non_finite_sample_threshold_rejected_with_value_error(value):
    t = SampleThreshold("CGIS", 5, 0, 4.0, 1.0, 0.5, ReliabilityKind.RETEST, 5, 1.96, 1, 1, value)
    with pytest.raises(ValueError, match="положительным конечным"):
        build_discrepancies(report_of({1: 12, 2: 16}), [], cgi_threshold=t,
                            cgi_threshold_source=None)


PHQ9_S1 = [2, 2, 2, 2, 2, 2, 1, 1, 0]
PHQ9_S2 = [1, 1, 1, 1, 1, 1, 1, 0, 2]   # пункт 9 положительный — синтетический сигнал


def phq9_with_alert():
    scores = [(1, score(PHQ9, PHQ9_S1)), (2, score(PHQ9, PHQ9_S2))]
    alerts = [(2, a) for a in check_safety(PHQ9, PHQ9_S2)]
    assert alerts
    return build_report(PHQ9, scores, safety_alerts=alerts), alerts


def test_safety_alerts_of_other_report_must_reach_the_main_header():
    phq, alerts = phq9_with_alert()
    rep = report_of({1: 12, 2: 16})                       # шапка: проверка не проводилась
    with pytest.raises(ValueError, match="сигналы безопасности"):
        build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=2,
                            cgi_threshold_source=SOURCE, other_reports=[phq])
    rep = report_of({1: 12, 2: 16}, safety_alerts=alerts)  # сигналы — в шапке основного отчёта
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=2,
                                cgi_threshold_source=SOURCE, other_reports=[phq])
    assert block.discrepancies[0].others[0].measure == "PHQ9"
    assert alerts[0][1].message in render_text(rep, block)


def test_other_report_checked_without_signals_is_accepted():
    phq = build_report(PHQ9, [(1, score(PHQ9, PHQ9_S1))], safety_alerts=[])
    block = block_of({1: 12, 2: 16}, {1: 4, 2: 4}, others=[phq])
    assert block.other_measures == ("PHQ9",)


@pytest.mark.parametrize("scores", [[], [(1, 5)], [(1, 5), (2, 3)]])
def test_trajectory_rejects_clinician_measure_at_any_length(scores):
    with pytest.raises(ValueError, match="оценка клинициста"):
        rule_based_flag(CGIS, scores, threshold=2)


def test_blind_compares_aware_timestamps_in_utc():
    try:
        berlin = ZoneInfo("Europe/Berlin")
    except ZoneInfoNotFoundError:
        pytest.skip("нет системной базы часовых поясов")
    viewed = datetime(2026, 10, 25, 2, 30, tzinfo=berlin, fold=0)   # 00:30 UTC
    rated = datetime(2026, 10, 25, 2, 10, tzinfo=berlin, fold=1)    # 01:10 UTC — на 40 мин позже
    assert not TherapistRating(1, cgis_score(4), rated, viewed).blind
    assert TherapistRating(1, cgis_score(4), viewed, rated).blind


@pytest.mark.parametrize("gad,cgi,window", [
    ({1: 20, 2: 12, 3: 16}, {2: 4, 3: 4}, "1 → 3"),      # база блока сдвинута
    ({1: 12, 2: 16, 3: 10}, {1: 4, 2: 4}, "1 → 3"),      # самоотчёт сессии 3 ещё без оценки
], ids=["сдвиг-базы", "отставание"])
def test_record_says_which_sessions_the_traffic_light_uses(gad, cgi, window):
    block = block_of(gad, cgi)
    lines = record_lines(block)
    assert lines[0] == f"  1. {POINTER}"
    assert any(line.endswith(f"; светофор и статус считают по сессиям {window}") for line in lines)


def test_record_in_status_window_has_no_traffic_light_note():
    assert not any("светофор и статус считают" in line for line in cell_block(W, N).lines())


@pytest.mark.parametrize("measure,gad,threshold", [(OASIS, {1: 10, 2: 15}, 4),
                                                   (WSAS, {1: 10, 2: 20}, 8)])
def test_structural_alternatives_do_not_assume_gad7(measure, gad, threshold):
    rep = build_report(measure, [(s, full(t, measure)) for s, t in gad.items()],
                       threshold=threshold, threshold_source=SOURCE)
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=2,
                                cgi_threshold_source=SOURCE)
    alts = block.discrepancies[0].alternatives
    assert not any("две недели" in a or "симптомы" in a or "о котором спрашивает" in a
                   for a in alts)
    assert any(f"самоотчёт — шкала {measure.code}, оценка терапевта — общее клиническое "
               f"впечатление о тяжести (CGI-S)." in a for a in alts)
    assert any(f"по инструкции {measure.code} (период опроса у шкал разный)" in a for a in alts)


def test_plural_sessions_in_record_lines():
    prorated = score(GAD7, [3, 3, None, 2, 2, 2, 2], max_missing=1)      # 16, пересчитан
    base = score(GAD7, [2, 2, None, 2, 2, 1, 1], max_missing=1)          # 12, пересчитан
    rep = build_report(GAD7, [(1, base), (2, full(14)), (3, prorated)])
    block = build_discrepancies(rep, [rating(s, 4, blind=False) for s in (1, 2, 3)],
                                cgi_threshold=2, cgi_threshold_source=SOURCE)
    text = "\n".join(block.lines())
    assert "пересчитан пропорционально: сессии 1, 3" in text
    assert "после просмотра самоотчёта: сессии 1, 3" in text


def test_of_sessions_agrees_with_numeral():
    block = block_of({1: 12}, {1: 4})
    assert any("1 из 1 сессии;" in line for line in block.lines())
    assert any("2 из 2 сессий;" in line for line in block_of({1: 12, 2: 16}, {1: 4, 2: 4}).lines())


def test_other_scale_uses_the_window_of_the_record():
    phq = build_report(PHQ8, [(s, full(t, PHQ8)) for s, t in {1: 20, 2: 14, 3: 6, 4: 12}.items()],
                       threshold=5, threshold_source=SOURCE)
    block = block_of(RELAPSE_GAD, RELAPSE_CGI, others=[phq])
    (d,) = block.discrepancies
    (other,) = d.others
    assert d.window is Window.ADJACENT
    assert (other.measure, other.start, other.end, other.change) == \
        ("PHQ8", 6, 12, Change.DETERIORATED)


@pytest.mark.parametrize("gad,cgi,note", [
    ({1: 12, 2: 8}, {1: 2, 2: 3},
     "улучшение по оценке терапевта (CGI-S) не определяется: в сессии 1 — 2, "
     "до границы шкалы меньше порога 2"),
    ({1: 18, 2: 17}, {1: 3, 2: 5},
     "ухудшение по самоотчёту GAD7 не определяется: в сессии 1 — 18, "
     "до границы шкалы меньше порога 4"),
], ids=["CGI-от-базы", "GAD-от-базы"])
def test_reachability_is_measured_from_the_base(gad, cgi, note):
    # на конце окна направление достижимо, на базе — нет: решает база
    block = block_of(gad, cgi)
    assert block.state is BlockState.NONE_FOUND and note in block.notes


LEVELS_LINE = ("  расхождение — тема для разговора, не вердикт; сравниваются направления изменений, "
               "каждая шкала — сама с собой; уровни не сравниваются, поэтому устойчивое "
               "расхождение уровней здесь не видно")
SESSION_ONE = "  база — сессия 1: оценка терапевта на неполной картине"


def test_decision_lines_are_printed():
    for block in (cell_block(N, N), cell_block(W, N)):
        assert LEVELS_LINE in block.lines() and SESSION_ONE in block.lines()
    assert SESSION_ONE not in expanded(W, N, start=2).lines()
    assert "     держится сравнений подряд: 1 (от одной базы — не независимые подтверждения)" \
        in cell_block(W, N).lines()
    assert ("     держится сравнений подряд: 1 (соседние окна делят сессии — не независимые "
            "подтверждения)") in block_of(RELAPSE_GAD, RELAPSE_CGI).lines()


def test_blind_line_with_a_single_pair():
    block = block_of({1: 12, 2: 16}, {1: 4})
    assert block.state is BlockState.INSUFFICIENT
    assert (f"{BLIND_LINE}: 0 из 1 в парах; совпадение направлений по ним менее информативно"
            in block.lines())


def test_blind_count_is_over_pairs_only():
    rep = report_of({1: 12, 2: 13})
    ratings = [rating(1, 4), rating(2, 4), rating(3, 4, blind=False)]   # сессия 3 без самоотчёта
    block = build_discrepancies(rep, ratings, cgi_threshold=2, cgi_threshold_source=SOURCE)
    assert block.non_blind == 0
    assert any(line.startswith(f"{BLIND_LINE}: 0 из 2 в парах") for line in block.lines())


def test_not_counted_rating_without_self_report_is_not_therapist_only():
    block = block_of({1: 12, 2: 13}, {1: 4, 2: 4, 3: 0})
    assert block.ratings_not_counted == (3,) and block.therapist_only == ()


def test_no_lag_line_when_latest_rating_exists_but_not_counted():
    block = block_of({1: 12, 2: 13, 3: 16}, {1: 4, 2: 4, 3: 0})
    assert not any("ещё без оценки" in line for line in block.lines())
    assert "  оценка терапевта не засчитана: сессия 3" in block.lines()


def test_prorated_base_is_marked():
    base = score(GAD7, [2, 2, None, 2, 2, 1, 1], max_missing=1)          # 10 · 7/6 → 12
    rep = build_report(GAD7, [(1, base), (2, full(16))])
    block = build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=2,
                                cgi_threshold_source=SOURCE)
    (d,) = block.discrepancies
    assert d.client_prorated == (1,)
    assert any(line.endswith("; пересчитан пропорционально: сессия 1") for line in block.lines())


FLAT = [entry(1, 0, 6, Reaction.AVOIDED), entry(2, 1, 6, Reaction.AVOIDED),
        entry(3, 7, 6, Reaction.STAYED), entry(4, 8, 6, Reaction.STAYED)]


def test_exposure_alternative_when_intensity_unchanged():
    assert len(exposure_alts(FLAT)) == 1


def test_regression_to_mean_from_session_two():
    d = expanded(B, N, start=2).discrepancies[0]
    assert d.from_session == 2 and any("регрессия к среднему" in a for a in d.alternatives)


def extra_branch_blocks():
    """Блоки с ветками вывода, которых нет в матрице: для лексических тестов."""
    phq = build_report(PHQ8, [(s, full(t, PHQ8)) for s, t in {1: 20, 2: 14, 3: 6, 4: 12}.items()],
                       threshold=5, threshold_source=SOURCE)
    phq_partial = build_report(PHQ9, [(1, full(14, PHQ9))])
    sample = threshold_from_sample([full(v, CGIS) for v in (3, 4, 4, 5)], 0.9,
                                   kind=ReliabilityKind.RETEST, reliability_n=60)
    rep = report_of({1: 12, 2: 16})
    return [
        block_of(RELAPSE_GAD, RELAPSE_CGI, others=[phq, phq_partial], blind=False),
        block_of({1: 12, 2: 13, 3: 16}, {1: 4, 2: 0, 3: 4}),                 # код 0
        block_of({1: 12, 2: 16}, {1: 4, 2: 3}, cgi_threshold=1),
        build_discrepancies(rep, [rating(1, 4), rating(2, 4)], cgi_threshold=sample,
                            cgi_threshold_source=None),
        block_of({1: 2, 2: 3}, {1: 1, 2: 1}), block_of({1: 19, 2: 18}, {1: 7, 2: 7}),  # пол, потолок
        block_of({1: 14, 2: 12, 3: 16}, {2: 4, 3: 4}),                       # база сдвинута
        block_of({1: 12, 2: 16, 3: 10}, {1: 4, 2: 4}),                       # отставание
        block_of({1: 20, 2: 4, 3: 6, 4: 12}, {2: 2, 3: 6, 4: 4}),            # окно 2 при сдвиге
        block_of({1: 12, 2: 12, 3: 12, 4: 12}, {1: 4, 2: 6, 3: 5, 4: 6}),    # свёрнуто
        block_of({1: 12, 2: 16}, {1: 4, 3: 4}),                              # без пары
        block_of({1: 12, 2: 12, 3: 16}, {1: 0, 2: 4, 3: 4}),                 # база: не засчитана
        block_of({1: 12, 2: 8, 3: 4, 4: 10}, {1: 4, 2: 6, 3: 6, 4: 6}),      # окно 2: потолок
        block_of({1: 12, 2: 16}, {1: 4, 2: 4}, others=[phq_partial]),        # нет замера в сессии
        build_discrepancies(build_report(GAD7, [(1, score(GAD7, [2, 2, None, 2, 2, 1, 1],
                                                          max_missing=1)), (2, full(16))]),
                            [rating(1, 4), rating(2, 4)], cgi_threshold=2,
                            cgi_threshold_source=SOURCE),                    # пересчёт базы
        block_of({1: 10, 2: 12, 3: 16}, {1: 4, 2: 5, 3: 3}),                 # окно 2: противоположны
    ]


# --- повторная проверка правок -------------------------------------------------------------

def test_adjacent_window_shows_another_kind_after_client_worse_in_window_one():
    # окно 1: самоотчёт +6, CGI-S −1 — «клиент хуже»; последний шаг: +4 при CGI-S −2 — противоположны
    block = block_of({1: 10, 2: 12, 3: 16}, {1: 4, 2: 5, 3: 3})
    assert [(d.kind, d.window) for d in block.discrepancies] == \
        [(Kind.CLIENT_WORSE_UNSEEN, Window.BASE), (Kind.OPPOSITE, Window.ADJACENT)]


def test_two_pairs_at_ceiling_give_one_note():
    block = block_of({1: 12, 2: 16}, {1: 6, 2: 6})
    assert block.notes == ("ухудшение по оценке терапевта (CGI-S) не определяется: в сессии 1 — 6, "
                           "до границы шкалы меньше порога 2",)
    assert len(block.lines()) == len(set(block.lines()))


def test_window_two_note_names_the_client_change():
    block = block_of({1: 12, 2: 8, 3: 13}, {1: 4, 2: 6, 3: 6})
    assert ("предыдущая пара → текущая (сессии 2 → 3): самоотчёт GAD7 8 → 13, надёжное ухудшение"
            in block.notes)
    assert "  ↳ ухудшение по оценке терапевта (CGI-S) не определяется: в сессии 2 — 6, до " \
           "границы шкалы меньше порога 2" in block.lines()


def test_timestamps_out_of_range_rejected_with_value_error():
    plus5 = timezone(timedelta(hours=5))
    early = TherapistRating(1, cgis_score(4), datetime(1, 1, 1, 1, 0, tzinfo=plus5),
                            datetime(1, 1, 1, 2, 0, tzinfo=plus5))
    with pytest.raises(ValueError, match="вне диапазона"):
        build_discrepancies(report_of({1: 12, 2: 16}), [early, rating(2, 4)], cgi_threshold=2,
                            cgi_threshold_source=SOURCE)


def test_manual_cgi_threshold_has_no_raw_value():
    block = cell_block(W, N)
    assert block.cgi_threshold_raw is None and not block.cgi_threshold_from_sample
    assert not any("расчёт" in line for line in block.lines())
