from mbc.measures import GAD7, PHQ9
from mbc.safety import check_safety


def test_no_alert_when_item9_zero():
    assert check_safety(PHQ9, [1, 1, 1, 1, 1, 1, 1, 1, 0]) == []


def test_alert_when_item9_positive():
    alerts = check_safety(PHQ9, [0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert len(alerts) == 1 and alerts[0].item == 9 and alerts[0].value == 1


def test_alert_when_item9_missing():
    alerts = check_safety(PHQ9, [0, 0, 0, 0, 0, 0, 0, 0, None])
    assert len(alerts) == 1 and alerts[0].value is None


def test_measures_without_safety_items():
    assert check_safety(GAD7, [3] * 7) == []
