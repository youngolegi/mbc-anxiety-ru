from mbc.measures import GAD7
from mbc.trajectory import Signal, rule_based_flag


def test_insufficient_with_one_point():
    assert rule_based_flag(GAD7, [(1, 15)]).signal is Signal.INSUFFICIENT


def test_red_on_reliable_deterioration():
    flag = rule_based_flag(GAD7, [(1, 10), (2, 12), (3, 15)])
    assert flag.signal is Signal.RED
    assert flag.session == 3


def test_amber_when_no_improvement_by_session_4():
    flag = rule_based_flag(GAD7, [(1, 15), (2, 14), (3, 14), (4, 13)])
    assert flag.signal is Signal.AMBER


def test_green_early_without_improvement():
    assert rule_based_flag(GAD7, [(1, 15), (2, 14)]).signal is Signal.GREEN


def test_green_with_reliable_improvement():
    assert rule_based_flag(GAD7, [(1, 15), (4, 10)]).signal is Signal.GREEN


def test_unordered_input_is_sorted():
    flag = rule_based_flag(GAD7, [(4, 13), (1, 15), (2, 14), (3, 14)])
    assert flag.signal is Signal.AMBER
