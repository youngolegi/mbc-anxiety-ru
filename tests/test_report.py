"""Отчёт для терапевта. Все данные синтетические."""
from datetime import datetime, timedelta

import pytest

from mbc.measures import GAD7, OASIS, PHQ9, WSAS
from mbc.report import build_report, render_text
from mbc.safety import SafetyAlert, check_safety
from mbc.scoring import score
from mbc.status import Outcome, outcome
from mbc.trajectory import Signal, rule_based_flag
from mbc.triggers import Reaction, TriggerEntry

from test_triggers import ENTRIES

FULL_15 = [3, 2, 2, 2, 2, 2, 2]
FULL_14 = [2] * 7
FULL_13 = [2, 2, 2, 2, 2, 2, 1]
FULL_7 = [1] * 7
MISSING = [2, 2, None, 2, 2, 2, 2]


def gad(answers, max_missing=0):
    return score(GAD7, answers, max_missing=max_missing)


def text_of(*args, **kwargs):
    return render_text(build_report(*args, **kwargs))


def lines_of(text):
    return text.splitlines()


START = datetime(2026, 10, 5, 19, 0)  # понедельник


def entry(i, day, intensity, **kw):
    return TriggerEntry(f"x{i}", START + timedelta(days=day), intensity, **kw)


ALERT = SafetyAlert("PHQ9", 9, 1, "сообщение сигнала как есть")
SAFETY_STATES = (
    "БЕЗОПАСНОСТЬ: проверка не проводилась — сигналы в отчёт не переданы",
    "БЕЗОПАСНОСТЬ: проверено — сигналов нет",
    "БЕЗОПАСНОСТЬ: ЕСТЬ СИГНАЛЫ (последние сверху)",
)


# --- шапка безопасности: три состояния ---------------------------------------------

@pytest.mark.parametrize("alerts,state", [(None, 0), ([], 1), ([(1, ALERT)], 2)],
                         ids=["не проводилась", "проверено", "есть сигналы"])
def test_safety_header_states_are_exclusive(alerts, state):
    lines = lines_of(text_of(GAD7, [(1, gad(FULL_15))], safety_alerts=alerts))
    assert lines[3] == SAFETY_STATES[state]
    assert [s for s in SAFETY_STATES if s in lines] == [SAFETY_STATES[state]]


def test_safety_alerts_verbatim_latest_first_stable_within_session():
    a = SafetyAlert("PHQ9", 9, 1, "первый")
    b = SafetyAlert("PHQ9", 9, None, "второй")
    c = SafetyAlert("PHQ9", 9, 2, "третий")
    report = build_report(GAD7, [(1, gad(FULL_15))],
                          safety_alerts=iter([(3, a), (5, c), (3, b)]))
    assert [alert for _, alert in report.safety_alerts] == [c, a, b]
    assert lines_of(render_text(report))[4:7] == [
        "  ! сессия 5 · PHQ9, пункт 9, ответ 2: третий",
        "  ! сессия 3 · PHQ9, пункт 9, ответ 1: первый",
        "  ! сессия 3 · PHQ9, пункт 9, ответ пропущен: второй",
    ]


def test_alerts_checked_at_intake_pass_through_unchanged():
    # check_safety вызывается при приёме ответов PHQ-9, отчёт получает готовый результат
    alerts = [(4, alert) for alert in check_safety(PHQ9, [0] * 8 + [2])]
    text = text_of(GAD7, [(1, gad(FULL_15))], safety_alerts=alerts)
    assert alerts[0][1].message in text


@pytest.mark.parametrize("alerts", [[ALERT], [(0, ALERT)], [(True, ALERT)], [("3", ALERT)],
                                    [(1, "текст")]])
def test_invalid_safety_input_rejected(alerts):
    with pytest.raises(ValueError):
        build_report(GAD7, [], safety_alerts=alerts)


# --- замеры: незасчитанные, базовая линия, пересчёт ----------------------------------

def test_baseline_shifts_when_first_session_not_counted():
    scores = [(1, gad(MISSING)), (2, gad(FULL_15)), (3, gad(FULL_14)), (4, gad(FULL_7))]
    report = build_report(GAD7, scores)
    assert report.baseline.session == 2 and report.baseline_shifted
    assert report.outcome is outcome(GAD7, 15, 7)
    lines = lines_of(render_text(report))
    assert "  с1: не засчитан (ответов 6 из 7) · с2: 15 · с3: 14 · с4: 7" in lines
    assert "  ! базовая линия — сессия 2: замер сессии 1 не засчитан" in lines
    assert "  базовая линия: сессия 2, балл 15" in lines


def test_baseline_shift_lists_only_sessions_before_baseline():
    scores = [(1, gad(MISSING)), (2, gad(FULL_15)), (3, gad(MISSING)), (4, gad(FULL_7))]
    text = text_of(GAD7, scores)
    assert "  ! базовая линия — сессия 2: замер сессии 1 не засчитан" in lines_of(text)
    assert "сессий 1, 3" not in text and "сигнал по последнему засчитанному" not in text


def test_baseline_shift_over_several_sessions_uses_plural():
    scores = [(1, gad(MISSING)), (2, gad(MISSING)), (3, gad(FULL_15)), (4, gad(FULL_7))]
    assert "  ! базовая линия — сессия 3: замеры сессий 1, 2 не засчитаны" in lines_of(
        text_of(GAD7, scores))


def test_no_shift_when_first_passed_session_is_counted():
    report = build_report(GAD7, [(3, gad(FULL_15)), (5, gad(FULL_7))])
    assert not report.baseline_shifted
    assert "! базовая линия —" not in render_text(report)


def test_not_counted_scores_excluded_from_trajectory():
    scores = [(1, gad(FULL_15)), (2, gad(MISSING)), (3, gad(FULL_14)), (4, gad(FULL_13))]
    report = build_report(GAD7, scores)
    assert report.flag == rule_based_flag(GAD7, [(1, 15), (3, 14), (4, 13)])
    assert report.flag.signal is Signal.AMBER


def test_flag_uses_last_counted_session_and_says_so():
    scores = [(1, gad(FULL_15)), (2, gad(FULL_14)), (3, gad(FULL_13)), (4, gad(MISSING))]
    report = build_report(GAD7, scores)
    assert report.flag.session == 3 and report.flag.signal is Signal.GREEN
    lines = lines_of(render_text(report))
    assert lines[lines.index("СВЕТОФОР (сессия 3): [ЗЕЛЁНЫЙ] на траектории — "
                             "отклонений от правила нет") + 2] == (
        "  ! замер сессии 4 не засчитан — сигнал по последнему засчитанному (сессия 3)")


def test_prorated_baseline_is_used_and_flagged():
    scores = [(1, gad(MISSING, max_missing=1)), (2, gad(FULL_7))]
    report = build_report(GAD7, scores)
    assert report.baseline.prorated and report.baseline.total == 14
    assert report.flag == rule_based_flag(GAD7, [(1, 14), (2, 7)])
    lines = lines_of(render_text(report))
    assert "  с1: 14* · с2: 7" in lines
    assert "  * пересчитан пропорционально по неполным ответам — допущение, не стандарт" in lines
    assert ("  ! базовый замер (сессия 1) пересчитан пропорционально (ответов 6 из 7): "
            "все сравнения со стартом опираются на пересчитанный балл") in lines
    assert "  ! сигнал опирается на пересчитанный пропорционально балл: сессия 1" in lines


def test_prorated_latest_marked_in_flag_and_status_without_baseline_warning():
    text = text_of(GAD7, [(1, gad(FULL_15)), (2, gad(MISSING, max_missing=1))])
    assert "с2: 14*" in text and "базовый замер" not in text
    assert "сигнал опирается на пересчитанный пропорционально балл: сессия 2" in text
    assert "нет надёжного изменения (пересчитан пропорционально: сессия 2)" in text


def test_single_counted_score_gives_no_status():
    report = build_report(GAD7, [(1, gad(FULL_15)), (2, gad(MISSING))])
    assert report.outcome is None and report.flag.signal is Signal.INSUFFICIENT
    text = render_text(report)
    assert "СТАТУС: недостаточно данных — нужно минимум два засчитанных замера" in text
    assert "  засчитано замеров: 1 из 2" in lines_of(text)
    assert "сессии 1 → 1" not in text and "сигнал по последнему засчитанному" not in text


def test_no_counted_scores():
    report = build_report(GAD7, [(1, gad(MISSING)), (2, gad(MISSING))])
    assert report.baseline is None and report.outcome is None
    lines = lines_of(render_text(report))
    assert "  ! ни один замер не засчитан" in lines
    assert "  засчитано замеров: 0 из 2" in lines


def test_empty_input_renders():
    lines = lines_of(text_of(GAD7, []))
    assert "ЗАМЕРЫ: не переданы" in lines
    assert "ПАТТЕРНЫ КАРТЫ ТРИГГЕРОВ: записей нет" in lines
    assert "ТЕЛЕСНЫЕ ФАКТОРЫ (могли поднять тревогу независимо от ситуации): записей нет" in lines


def test_scores_from_iterator_are_not_lost():
    as_list = [(1, gad(FULL_15)), (2, gad(FULL_7))]
    assert text_of(GAD7, iter(as_list)) == text_of(GAD7, as_list)


# --- проверка входа ------------------------------------------------------------------

def test_mixed_measures_rejected():
    phq = score(PHQ9, [1] * 9)
    with pytest.raises(ValueError, match="сессия 2: PHQ9"):
        build_report(GAD7, [(1, gad(FULL_15)), (2, phq)])


def test_duplicate_session_rejected():
    with pytest.raises(ValueError, match="Больше одного замера в сессии: 1"):
        build_report(GAD7, [(1, gad(FULL_15)), (1, gad(FULL_14))])


@pytest.mark.parametrize("session", [0, -1, True, 1.0, "1"])
def test_invalid_session_number_rejected(session):
    with pytest.raises(ValueError, match="Номер сессии"):
        build_report(GAD7, [(session, gad(FULL_15))])


@pytest.mark.parametrize("pair", [(1, None), (1, 15), [1, score(GAD7, FULL_15)]])
def test_scores_must_be_session_score_pairs(pair):
    with pytest.raises(ValueError, match="парами"):
        build_report(GAD7, [pair])


def test_duplicate_entry_ids_rejected():
    e = entry(1, 0, 5, before="a")
    with pytest.raises(ValueError, match="x1"):
        build_report(GAD7, [], [e, e])


@pytest.mark.parametrize("kwargs,match", [
    ({"top_n": 0}, "top_n"),
    ({"min_episodes": 1}, "min_episodes"),
    ({"dimensions": ()}, "измерение"),
    ({"threshold": 4}, "укажите threshold_source"),
    ({"threshold_source": "синтетика"}, "без threshold"),
    ({"threshold": 0, "threshold_source": "синтетика"}, "положительным"),
    ({"threshold": float("nan"), "threshold_source": "синтетика"}, "положительным"),
    ({"threshold": True, "threshold_source": "синтетика"}, "положительным"),
])
def test_invalid_parameters_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_report(GAD7, [(1, gad(FULL_15))], **kwargs)


# --- статус, пороги, источники -------------------------------------------------------

def test_status_and_sources_in_text():
    report = build_report(GAD7, [(1, gad(FULL_15)), (6, gad(FULL_7))])
    assert report.outcome is Outcome.RECOVERED
    lines = lines_of(render_text(report))
    assert lines[0] == "ОТЧЁТ ДЛЯ ТЕРАПЕВТА · GAD7 · сессии 1–6"
    assert "СТАТУС: 15 (тяжёлая) → 7 (лёгкая), сессии 1 → 6: выздоровление" in lines
    assert any(line.startswith("  логика статуса — практика NHS Talking Therapies, "
                               "по одной шкале (GAD7)") for line in lines)
    assert f"  порог надёжного изменения: 4 — {GAD7.reliable_change_source}" in lines
    assert f"  порог «случая»: 8 — {GAD7.caseness_source}" in lines
    assert "  ! пороги шкалы не пересчитаны на русских данных" in lines
    assert f"  русская версия: {GAD7.russian_version}" in lines


def test_measure_without_threshold_requires_explicit_one():
    oasis = [(1, score(OASIS, [3, 3, 2, 2, 2])), (4, score(OASIS, [1] * 5))]
    with pytest.raises(ValueError, match="rci_threshold"):
        build_report(OASIS, oasis)
    report = build_report(OASIS, oasis, threshold=5, threshold_source="синтетический порог")
    assert report.outcome is Outcome.RECOVERED
    lines = lines_of(render_text(report))
    assert "СТАТУС: 12 → 5, сессии 1 → 4: выздоровление" in lines
    assert (f"  порог надёжного изменения: 5 — синтетический порог "
            f"(справка по шкале: {OASIS.reliable_change_source})") in lines
    assert "  диапазоны тяжести для шкалы не заданы" in lines
    assert f"  ! русская версия: {OASIS.russian_version}" in lines


def test_explicit_threshold_shows_what_it_replaces():
    text = text_of(GAD7, [(1, gad(FULL_15)), (2, gad(FULL_7))],
                   threshold=3, threshold_source="синтетика")
    assert (f"порог надёжного изменения: 3 — синтетика "
            f"(вместо 4 — {GAD7.reliable_change_source})") in text


def test_measure_without_caseness():
    wsas = [(1, score(WSAS, [6] * 5)), (2, score(WSAS, [2] * 5))]
    lines = lines_of(text_of(WSAS, wsas, threshold=8, threshold_source="синтетика"))
    assert "СТАТУС: 30 → 10, сессии 1 → 2: надёжное улучшение" in lines
    assert f"  порог «случая»: {WSAS.caseness_source} — «выздоровление» не определяется" in lines


# --- светофор: когда правило применимо --------------------------------------------------

def test_green_before_check_session_says_amber_not_checked_yet():
    lines = lines_of(text_of(GAD7, [(1, gad(FULL_15)), (2, gad(FULL_15))]))
    assert "СВЕТОФОР (сессия 2): [ЗЕЛЁНЫЙ] на траектории — отклонений от правила нет" in lines
    assert "  жёлтый ещё не проверяется: проверка с сессии 4" in lines


def test_green_below_caseness_says_amber_not_applicable():
    low = [1, 1, 1, 1, 1, 1, 0]
    lines = lines_of(text_of(GAD7, [(1, gad(low)), (5, gad(low))]))
    assert "  жёлтый не применяется: на старте балл ниже порога «случая» (8)" in lines


def test_check_by_session_is_passed_and_shown():
    scores = [(1, gad(FULL_15)), (2, gad(FULL_14)), (4, gad(FULL_13))]
    assert build_report(GAD7, scores).flag.signal is Signal.AMBER
    report = build_report(GAD7, scores, check_by_session=6)
    assert report.flag.signal is Signal.GREEN
    assert "жёлтый — с сессии 6 нет надёжного улучшения" in render_text(report)


# --- паттерны ---------------------------------------------------------------------------

def test_top_n_shows_hidden_count():
    entries = [entry(n * 10 + k, k, 5, before=ctx)
               for n, ctx in enumerate(["a", "b", "c", "d"]) for k in range(3)]
    report = build_report(GAD7, [], entries)
    assert report.patterns_found == 4 and len(report.patterns) == 3
    assert all(p.overlaps == () for p in report.patterns)
    text = render_text(report)
    assert "показано 3 из 4" in text and "↳" not in text
    assert "показано 1 из 4" in text_of(GAD7, [], entries, top_n=1)


def test_pattern_dynamics_overlap_and_avoidance_denominator():
    entries = [
        entry(1, 0, 8, before="звонок", where="дом", reaction=Reaction.AVOIDED),
        entry(2, 1, 8, before="звонок", where="дом"),  # реакция не отмечена
        entry(3, 8, 4, before="звонок", where="дом", reaction=Reaction.STAYED),
        entry(4, 9, 6, before="лента", where="работа"),
    ]
    report = build_report(GAD7, [], entries, dimensions=("before", "where"))
    first, second = report.patterns
    assert (first.label, second.label) == ("за час до: звонок", "где: дом")
    assert first.reactions_marked == (1, 1)
    assert first.overlaps == () and second.overlaps == ((1, 3),)
    lines = lines_of(render_text(report))
    assert "  1. за час до: звонок — эпизодов: 3, недель с эпизодами: 2 (05.10–18.10.2026)" in lines
    assert ("     по неделям (с понедельника; интенсивность/избегание/эпизодов): "
            "05.10 8/100%/2 · 12.10 4/0%/1") in lines
    assert "     первая → последняя неделя: интенсивность 8 → 4 (-4; эпизодов 2 и 1)" in lines
    assert "     избегание 100% → 0% (-100 п.п.; реакция отмечена в 1 и 1 записях)" in lines
    assert "     ↳ общих записей с паттерном 1: 3" in lines
    assert "паттерн — гипотеза, не вывод" in "\n".join(lines)


def test_middle_weeks_are_shown_and_missing_reactions_not_counted_as_zero():
    entries = [entry(1, 0, 8, before="a"), entry(2, 7, 3, before="a"),
               entry(3, 8, 3, before="a"), entry(4, 14, 8, before="a")]
    text = text_of(GAD7, [], entries)
    assert "05.10 8/—/1 · 12.10 3/—/2 · 19.10 8/—/1" in text
    assert "избегание: в первой или последней неделе реакции не отмечены — " \
           "изменение не считается" in text
    assert "п.п." not in text


def test_dimensions_and_min_episodes_are_passed_and_shown():
    entries = [entry(k, k, 5, before="a", with_whom="мать") for k in range(2)]
    assert build_report(GAD7, [], entries).patterns == ()
    report = build_report(GAD7, [], entries, min_episodes=2, dimensions=("with_whom",))
    assert [p.label for p in report.patterns] == ["с кем: мать"]
    assert "(измерения: с кем; минимум эпизодов: 2; записей: 2)" in render_text(report)


def test_single_week_pattern_has_no_dynamics():
    entries = [entry(k, k, 5, before="лента") for k in range(3)]
    report = build_report(GAD7, [], entries)
    assert report.patterns[0].intensity_delta is None
    text = render_text(report)
    assert "недель с эпизодами: 1 (05.10–11.10.2026)" in text
    assert "одна неделя с эпизодами — динамику сравнивать не с чем" in text


def test_no_patterns_message():
    entries = [entry(1, 0, 5, before="a"), entry(2, 1, 5, before="b")]
    assert ("ПАТТЕРНЫ КАРТЫ ТРИГГЕРОВ: нет — ни один контекст не набрал минимума эпизодов "
            "(измерения: где, с кем, за час до; минимум эпизодов: 3; записей: 2)") in lines_of(
        text_of(GAD7, [], entries))


# --- телесные факторы ----------------------------------------------------------------------

def test_somatic_factors_in_time_order_with_filled_denominator():
    entries = [entry(9, 0, 6, caffeine=True), entry(1, 1, 7, sleep_hours=5),
               entry(5, 2, 5, sleep_hours=8), entry(3, 3, 5)]
    report = build_report(GAD7, [], entries)
    assert report.somatic == (("x9", ("кофеин/энергетик",)), ("x1", ("сон 5 ч",)))
    assert report.entries_with_somatic_fields == 3
    assert ("  итого: 2 из 3 записей с заполненными полями (всего записей: 4); "
            "порог сна 6 ч — эвристика") in lines_of(render_text(report))


def test_min_sleep_hours_is_passed():
    entries = [entry(1, 0, 5, sleep_hours=7)]
    assert build_report(GAD7, [], entries).somatic == ()
    assert build_report(GAD7, [], entries, min_sleep_hours=8).somatic == (("x1", ("сон 7 ч",)),)


def test_somatic_absent_differs_from_not_filled():
    filled = [entry(1, 0, 5, sleep_hours=8, caffeine=False)]
    empty = [entry(1, 0, 5)]
    assert "не отмечены (поля заполнены в 1 из 1 записи)" in text_of(GAD7, [], filled)
    assert ("поля не заполнены ни в одной записи (записей: 1) — "
            "отсутствие факторов не установлено") in text_of(GAD7, [], empty)


# --- целиком ------------------------------------------------------------------------------

def test_report_is_deterministic_and_marked_for_therapist():
    scores = [(1, gad(FULL_15)), (2, gad(FULL_14)), (4, gad(FULL_13))]
    a = text_of(GAD7, scores, ENTRIES, safety_alerts=[])
    b = text_of(GAD7, list(reversed(scores)), list(reversed(ENTRIES)), safety_alerts=[])
    assert a == b
    assert lines_of(a)[1] == "Только для терапевта: не для передачи клиенту, не диагноз."
    order = [a.index(h) for h in
             ("БЕЗОПАСНОСТЬ", "ЗАМЕРЫ", "СВЕТОФОР", "СТАТУС", "ПАТТЕРНЫ", "ТЕЛЕСНЫЕ")]
    assert order == sorted(order)
