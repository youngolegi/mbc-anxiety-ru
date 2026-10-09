from datetime import datetime, timedelta

import pytest

from mbc.triggers import (Reaction, TriggerEntry, find_patterns, pattern_change,
                          somatic_confounders, weekly_progress)

START = datetime(2026, 10, 5, 19, 0)  # понедельник


def entry(i, day, intensity, before=None, with_whom=None, reaction=None, **kw):
    return TriggerEntry(f"e{i}", START + timedelta(days=day), intensity,
                        before=before, with_whom=with_whom, reaction=reaction, **kw)


ENTRIES = [
    entry(1, 0, 8, before="звонок матери", reaction=Reaction.AVOIDED),
    entry(2, 2, 9, before="звонок матери", reaction=Reaction.AVOIDED),
    entry(3, 3, 5, before="лента"),
    entry(4, 8, 7, before="звонок матери", reaction=Reaction.REASSURANCE),
    entry(5, 9, 4, before="лента"),
    entry(6, 15, 5, before="звонок матери", reaction=Reaction.STAYED),
    entry(7, 16, 4, before="звонок матери", reaction=Reaction.STAYED),
]


def test_intensity_validation():
    with pytest.raises(ValueError):
        entry(99, 0, 11)


def test_pattern_needs_min_episodes():
    labels = [p.label for p in find_patterns(ENTRIES, ("before",), min_episodes=3)]
    assert labels == ["before=звонок матери"]


def test_weekly_progress_tracks_intensity_and_avoidance():
    pattern = find_patterns(ENTRIES, ("before",), min_episodes=3)[0]
    stats = weekly_progress(ENTRIES, pattern)
    assert [s.episodes for s in stats] == [2, 1, 2]
    assert stats[0].avoidance_rate == 1.0
    assert stats[-1].avoidance_rate == 0.0
    change = pattern_change(stats)
    assert change["intensity_delta"] < 0
    assert change["avoidance_delta"] == -1.0


def test_support_seeking_is_not_avoidance():
    assert entry(50, 0, 5, reaction=Reaction.SUPPORT).avoidant is False
    assert entry(51, 0, 5, reaction=Reaction.REASSURANCE).avoidant is True


def test_somatic_confounders():
    e = entry(60, 0, 8, sleep_hours=4.5, caffeine=True)
    assert somatic_confounders(e) == ["сон 4.5 ч", "кофеин/энергетик"]


def test_unknown_dimension_rejected():
    with pytest.raises(ValueError):
        find_patterns(ENTRIES, ("mood",))
