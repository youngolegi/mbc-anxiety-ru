"""Мутационная проверка слоя расхождений: ловят ли тесты намеренно внесённые ошибки.

Запуск из корня репозитория:  python tools/mutants_discrepancy.py

Каждый мутант — одна точечная замена в копии кода (во временной папке; сам репозиторий
не меняется). Копируются только src/, tests/ и pyproject.toml: данные и секреты, даже
игнорируемые git, в копию не попадают. Сначала тесты прогоняются без мутаций — если они
не зелёные, проверка останавливается. Мутант «убит», только если pytest завершился кодом 1
(тесты упали); любой другой код — ошибка окружения, проверка останавливается.
Эквивалентные мутанты (поведение не меняется) помечены и в итог не идут.
Код возврата: 0 — все неэквивалентные убиты; 1 — кто-то выжил; 2 — ошибка проверки.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = "src/mbc/discrepancy.py"

# (название, файл, что заменить, на что заменить, эквивалентен ли)
MUTANTS = [
    # матрица и согласие ниже порога
    ("согласие ниже порога не проверяется (терапевт)", D,
     "        if _same_sign(client, therapist_worse):\n", "        if False:\n", False),
    ("согласие ниже порога не проверяется (клиент)", D,
     "    if _same_sign(therapist, client_worse):\n", "    if False:\n", False),
    ("тот же знак включает ноль", D,
     "return worse > 0 if direction", "return worse >= 0 if direction", False),
    ("противоположные — как «клиент хуже»", D,
     "        return Kind.OPPOSITE\n", "        return Kind.CLIENT_WORSE_UNSEEN\n", False),
    ("клиент хуже/лучше перепутаны", D,
     "return (Kind.CLIENT_WORSE_UNSEEN if client is Direction.WORSE",
     "return (Kind.CLIENT_BETTER_UNSEEN if client is Direction.WORSE", False),
    ("порог терапевта: >= → >", D,
     "Direction.WORSE if t_worse >= cgi", "Direction.WORSE if t_worse > cgi", False),
    ("порог терапевта: <= → <", D,
     "Direction.BETTER if t_worse <= -cgi", "Direction.BETTER if t_worse < -cgi", False),
    ("порог клиента — из шкалы, а не из отчёта", D,
     "classify_change(a.client.total, b.client.total, report.threshold,",
     "classify_change(a.client.total, b.client.total, m.reliable_change or 4,", False),
    # пол и потолок
    ("достижимость у терапевта не проверяется", D,
     "            and not _reachable(CGIS, a.cgi, cgi, client):", "            and False:", False),
    ("достижимость у клиента не проверяется", D,
     "    elif kind is Kind.THERAPIST_ONLY and not _reachable(",
     "    elif False and not _reachable(", False),
    ("достижимость: >= → >", D, "    return room >= points", "    return room > points", False),
    ("достижимость терапевта — от конца окна", D,
     "and not _reachable(CGIS, a.cgi, cgi, client):",
     "and not _reachable(CGIS, b.cgi, cgi, client):", False),
    ("достижимость клиента — от конца окна", D,
     "            m, a.client.total, min_change_points(report.threshold), therapist):",
     "            m, b.client.total, min_change_points(report.threshold), therapist):", False),
    ("пометки о поле и потолке на базе пропали", D,
     "    notes = _unreachable_notes(report, pairs[0], cgi)", "    notes = ()", False),
    ("пометка о поле и потолке в окне 2 пропала", D,
     "            elif last.suppressed:\n", "            elif False:\n", False),
    # окна
    ("окно 2 выключено", D, "    if len(pairs) >= 3:\n", "    if False:\n", False),
    ("окно 2 для любого направления клиента", D,
     "        if last.client is Direction.WORSE:", "        if True:", False),
    ("окно 2 дублирует запись окна 1", D,
     "if last.kind is not None and last.signature != current.signature:",
     "if last.kind is not None:", False),
    ("окно 2 скрыто при любой записи окна 1", D,
     "if last.kind is not None and last.signature != current.signature:",
     "if last.kind is not None and current.kind is None:", False),
    ("окно 2 при двух парах", D, "    if len(pairs) >= 3:\n", "    if len(pairs) >= 2:\n", False),
    ("пометка окна 2 без стороны клиента", D,
     "                          f\"{last.client_change.value}\",\n                          f\"↳ {last.suppressed}\")",
     "                          f\"\",\n                          f\"↳ {last.suppressed}\")", False),
    # persisted и свёртка
    ("persisted всегда 1", D, "        n += 1\n", "        n = 1\n", False),
    ("подпись без направлений", D,
     "else (self.kind, self.client, self.therapist)", "else (self.kind,)", False),
    ("ничего не сворачивается", D,
     "return self.kind not in ALWAYS_SHOWN and self.persisted < 2", "return False", False),
    ("сворачивается до трёх сравнений", D,
     "and self.persisted < 2", "and self.persisted < 3", False),
    ("«клиент хуже» сворачивается", D,
     "ALWAYS_SHOWN = frozenset({Kind.OPPOSITE, Kind.CLIENT_WORSE_UNSEEN})",
     "ALWAYS_SHOWN = frozenset({Kind.OPPOSITE})", False),
    ("свёрнутая запись снова «впервые»", D,
     "— не держится двух сравнений подряд; развернётся, \"\n"
     "                f\"если повторится в следующем сравнении",
     "— впервые, разворачивается \"\n                f\"при повторе", False),
    # слепота
    ("слепая при равных отметках", D, "        return rated < viewed", "        return rated <= viewed",
     False),
    ("отметки с поясом — не в UTC", D,
     "            rated, viewed = rated.astimezone(timezone.utc), viewed.astimezone(timezone.utc)",
     "            pass", False),
    ("сравнение слепо, если слепа последняя", D,
     "non_blind=tuple(p.session for p in (a, b) if not p.rating.blind)",
     "non_blind=tuple(p.session for p in (b,) if not p.rating.blind)", False),
    ("строка слепоты пропала при «не переданы»", D,
     '        return [f"{HEAD}: оценки терапевта не переданы", _blind_line(block)]',
     '        return [f"{HEAD}: оценки терапевта не переданы"]', False),
    ("строка слепоты: «пар нет» при одной паре", D,
     "    if not block.pairs:\n        return f\"{head}: пар нет\"",
     "    if len(block.pairs) < 2:\n        return f\"{head}: пар нет\"", False),
    ("неслепые считаются по всем оценкам, не по парам", D,
     "non_blind=sum(not p.rating.blind for p in pairs),",
     "non_blind=sum(not r.blind for r in counted.values()),", False),
    # вход
    ("код 0 не переводится в пропуск", D,
     "[None if type(code) is int and code == 0 else code]", "[code]", False),
    ("False принимается как код 0", D, "type(code) is int and code == 0", "code == 0", False),
    ("две оценки в сессии допустимы", D,
     "    if repeated:\n        raise ValueError(f\"Больше одной оценки",
     "    if False:\n        raise ValueError(f\"Больше одной оценки", False),
    ("чужая шкала у оценки допустима", D,
     "if not isinstance(r.score, Score) or r.score.measure != CGIS.code:",
     "if not isinstance(r.score, Score):", False),
    ("диапазон оценки не проверяется", D,
     "or not CGIS.min_total <= total <= CGIS.max_total):", "or False):", False),
    ("незасчитанная оценка попадает в «без оценки»", D,
     "client_only=tuple(sorted(client.keys() - by_session.keys())),",
     "client_only=tuple(sorted(client.keys() - counted.keys())),", False),
    ("незасчитанная оценка попадает в «без самоотчёта»", D,
     "therapist_only=tuple(sorted(counted.keys() - client.keys())),",
     "therapist_only=tuple(sorted(by_session.keys() - client.keys())),", False),
    # порог CGI-S
    ("источник порога необязателен", D,
     "        if not isinstance(source, str) or not source.strip():\n", "        if False:\n", False),
    ("bool как порог", D,
     "        if isinstance(threshold, bool) or not isinstance(threshold, int):",
     "        if not isinstance(threshold, int):", False),
    ("порог от 0", D, "    if not 1 <= points <= span:", "    if not 0 <= points <= span:", False),
    ("порог до 7", D, "    if not 1 <= points <= span:", "    if not 1 <= points <= span + 1:", False),
    ("порог из выборки не только ретест", D,
     "        if threshold.kind is not ReliabilityKind.RETEST:\n", "        if False:\n", False),
    ("второй источник у порога из выборки", D,
     "        if source is not None:\n            raise", "        if False:\n            raise", False),
    ("бесконечный порог из выборки", D,
     "or not math.isfinite(value) or value <= 0:", "or False:", False),
    ("нет пометки «анализ чувствительности»", D,
     "if block.cgi_threshold == 1 else \"\")", "if False else \"\")", False),
    ("порог из выборки помечен эвристикой", D,
     "    if block.cgi_threshold_from_sample:\n", "    if False:\n", False),
    ("расчётное значение порога не печатается", D,
     "        value = (f\"{block.cgi_threshold} (расчёт {block.cgi_threshold_raw:.3g}; в баллах — \"\n"
     "                 f\"вверх до целого)\")",
     "        value = str(block.cgi_threshold)", False),
    ("отметки вне диапазона — OverflowError", D,
     "        except (TypeError, OverflowError) as exc:", "        except TypeError as exc:", False),
    ("источник порога CGI-S не печатается", D,
     "f\"{block.cgi_threshold_source}{label}\")", "f\"{label}\")", False),
    # вывод
    ("нет указателя при ухудшении", D,
     "if d.client_change is Change.DETERIORATED\n             else",
     "if False\n             else", False),
    ("нет строки об отставании", D,
     "            and latest.session in block.client_only:", "            and False:", False),
    ("строка об отставании при незасчитанной оценке", D,
     "            and latest.session in block.client_only:", "            and True:", False),
    ("нет строки о сдвинутой базе", D, "    if pairs[0] != status_base:", "    if False:", False),
    ("нет пометки о сессии 1", D, "    if pairs[0] == 1:", "    if False:", False),
    ("нет строки «уровни не сравниваются»", D,
     "каждая шкала — сама с собой; уровни не сравниваются, поэтому устойчивое расхождение \"\n"
     "             \"уровней здесь не видно\"",
     "каждая шкала — сама с собой\"", False),
    ("нет оговорки о зависимых подтверждениях", D,
     "f\"не независимые подтверждения)\")", "f\")\")", False),
    ("нет строки «только со шкалой отчёта»", D,
     "    if not block.other_measures:\n", "    if False:\n", False),
    ("«надёжно» в направлении терапевта", D,
     'BETTER = "лучше"', 'BETTER = "надежно лучше"', False),
    ("окно светофора — только для окна 2", D,
     "and (d.from_session, d.to_session) != (base.session, latest.session) else \"\")",
     "and d.window is Window.ADJACENT else \"\")", False),
    ("пересчёт базы не помечается", D,
     "client_prorated=tuple(p.session for p in (a, b) if p.client.prorated),",
     "client_prorated=tuple(p.session for p in (b,) if p.client.prorated),", False),
    ("«из 1 сессий»", D,
     "    return \"сессии\" if n % 10 == 1 and n % 100 != 11 else \"сессий\"",
     "    return \"сессий\"", False),
    ("«сессия 1, 3» в единственном числе", D,
     "else f\"после просмотра самоотчёта: {_sessions(d.non_blind)}\")",
     "else f\"после просмотра самоотчёта: сессия {', '.join(map(str, d.non_blind))}\")", False),
    ("запрещённая основа в шапке других шкал", D,
     "(к объяснению «стороны измеряют разное»):\")",
     "(на самом деле клиент преуменьшает):\")", False),
    ("запрещённая основа в строке слепоты записи", D,
     "else f\"после просмотра самоотчёта: {_sessions(d.non_blind)}\")",
     "else f\"терапевт мог ошибаться: {_sessions(d.non_blind)}\")", False),
    # альтернативы
    ("экспозиция без подтверждения", D,
     "    return (p.avoidance_delta is not None and p.avoidance_delta < 0\n"
     "            and p.intensity_delta is not None and p.intensity_delta <= 0)",
     "    return True", False),
    ("экспозиция: интенсивность должна снизиться", D,
     "p.intensity_delta <= 0)", "p.intensity_delta < 0)", False),
    ("экспозиция при любом направлении клиента", D,
     "    if client is Direction.WORSE:\n        alts +=", "    if True:\n        alts +=", False),
    ("экспозиция «в эти недели» без дат", D,
     "{_week_span(first.week, last.week)}", "эти", False),
    ("регрессия к среднему в любой сессии", D,
     "if client is Direction.BETTER and from_session <= 2:", "if client is Direction.BETTER:", False),
    ("регрессия к среднему только из сессии 1", D,
     "from_session <= 2:", "from_session == 1:", False),
    ("«картина полнее» в любой сессии", D, "        if from_session == 1:\n", "        if True:\n",
     False),
    ("оптимизм и альянс всегда", D,
     "    if _FAVOURABLE[therapist] > _FAVOURABLE[client]:", "    if True:", False),
    ("буквальное прочтение не первым", D,
     "    alts = [_literal(kind, client, therapist),",
     "    alts = [\"Возможно, дело в другом.\", _literal(kind, client, therapist),", False),
    ("конструкт снова под GAD-7", D,
     "f\"Возможно, стороны измеряют разное: самоотчёт — шкала {report.measure.code}, \"",
     "f\"Возможно, стороны измеряют разное: самоотчёт — симптомы по шкале {report.measure.code}, \"",
     False),
    ("окно опроса снова «две недели»", D,
     "f\"{report.measure.code} (период опроса у шкал разный), а оценка терапевта отражает \"",
     "f\"{report.measure.code} за последние две недели, а оценка терапевта отражает \"", False),
    ("экспозиция без названия паттерна", D,
     "f\"Если в недели паттерна «{p.label}» ({_week_span(first.week, last.week)}) шла \"",
     "f\"Если в недели ({_week_span(first.week, last.week)}) шла \"", False),
    ("«сессия 1, 3» в отчёте", "src/mbc/report.py",
     "f\"{_sessions(sorted(set(prorated)))}\")",
     "f\"сессия {', '.join(map(str, sorted(set(prorated))))}\")", False),
    ("другая шкала — от своей базы, а не из окна", D,
     "    x1, x2 = totals.get(start), totals.get(end)",
     "    x1, x2 = (other.baseline.total if other.baseline else None), totals.get(end)", False),
    ("другая шкала — не то окно", D,
     "others=tuple(_other_scale(o, a.session, b.session) for o in others)",
     "others=tuple(_other_scale(o, a.session, a.session) for o in others)", False),
    ("другая шкала дважды", D, "        if other.measure.code in codes:\n", "        if False:\n", False),
    ("сигналы безопасности другой шкалы теряются", D, "        if lost:\n", "        if False:\n", False),
    # отчёт и шкала клинициста
    ("блок от другого отчёта принимается", "src/mbc/report.py",
     "    if discrepancies is not None and discrepancies.report != report:\n", "    if False:\n",
     False),
    ("шкала клинициста проходит везде", "src/mbc/measures.py",
     '    if measure.rater != "client":\n', "    if False:\n", False),
    ("для одного пункта — любой коэффициент", "src/mbc/reliable_change.py",
     "    if measure.n_items == 1 and kind is not ReliabilityKind.RETEST:\n", "    if False:\n", False),
    ("траектория: проверка шкалы после раннего выхода", "src/mbc/trajectory.py",
     "    require_client_measure(measure)\n    if len(scores) < 2:\n"
     "        return TrajectoryFlag(Signal.INSUFFICIENT, \"нужно минимум два замера\", None)\n",
     "    if len(scores) < 2:\n"
     "        return TrajectoryFlag(Signal.INSUFFICIENT, \"нужно минимум два замера\", None)\n"
     "    require_client_measure(measure)\n", False),
]


COPIED = ("src", "tests", "pyproject.toml")
"""Всё, что нужно тестам; данные, секреты и окружения не копируются."""


def _copy(work: Path) -> None:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, work / name,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info"))
        else:
            shutil.copy2(source, work / name)


def _pytest(work: Path) -> int:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
        cwd=work, capture_output=True, text=True).returncode


def run() -> int:
    survived, killed, equivalent = [], 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        _copy(work)
        baseline = _pytest(work)
        if baseline != 0:
            print(f"ОШИБКА   тесты без мутаций не зелёные (код pytest {baseline}) — проверка "
                  f"бессмысленна")
            return 2
        for name, path, old, new, is_equivalent in MUTANTS:
            _copy(work)
            target = work / path
            source = target.read_text(encoding="utf-8")
            if source.count(old) != 1:
                print(f"ОШИБКА   {name}: образец найден {source.count(old)} раз — обновите список")
                return 2
            target.write_text(source.replace(old, new), encoding="utf-8")
            code = _pytest(work)
            if code not in (0, 1):
                print(f"ОШИБКА   {name}: код pytest {code} — сбой окружения или сбора тестов, "
                      f"а не упавший тест")
                return 2
            dead = code == 1
            if is_equivalent:
                equivalent += 1
                print(f"{'убит    ' if dead else 'эквивал.'}  {name}")
            elif dead:
                killed += 1
                print(f"убит      {name}")
            else:
                survived.append(name)
                print(f"ВЫЖИЛ     {name}")
    total = len(MUTANTS) - equivalent
    print(f"\nУбито {killed} из {total} неэквивалентных мутантов; эквивалентных: {equivalent}")
    if survived:
        print("Выжили: " + "; ".join(survived))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(run())
