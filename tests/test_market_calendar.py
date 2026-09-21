import pandas as pd

from market_calendar import (
    efa_block_groups,
    is_complete_efa_group,
    last_complete_half_hour_start_times,
    last_complete_hour_start_times,
    n2ex_gate_times,
    realised_data_cutoff_times,
)


def test_information_times_are_on_previous_day():
    delivery = pd.DatetimeIndex([pd.Timestamp("2025-06-15 12:00", tz="UTC")])
    assert n2ex_gate_times(delivery)[0] == pd.Timestamp("2025-06-14 09:50", tz="UTC")
    assert realised_data_cutoff_times(delivery)[0] == pd.Timestamp("2025-06-14 09:00", tz="UTC")
    assert last_complete_half_hour_start_times(delivery)[0] == pd.Timestamp(
        "2025-06-14 08:30",
        tz="UTC",
    )
    assert last_complete_hour_start_times(delivery)[0] == pd.Timestamp(
        "2025-06-14 08:00",
        tz="UTC",
    )


def test_winter_efa_groups_from_midnight_utc():
    start = pd.Timestamp("2025-01-15 00:00", tz="UTC")
    assert efa_block_groups(start, 48) == [
        (0, 6),
        (6, 14),
        (14, 22),
        (22, 30),
        (30, 38),
        (38, 46),
        (46, 48),
    ]


def test_summer_efa_groups_from_midnight_utc():
    start = pd.Timestamp("2025-06-15 00:00", tz="UTC")
    assert efa_block_groups(start, 48) == [
        (0, 4),
        (4, 12),
        (12, 20),
        (20, 28),
        (28, 36),
        (36, 44),
        (44, 48),
    ]


def test_complete_efa_group_detection():
    start = pd.Timestamp("2025-01-15 00:00", tz="UTC")
    index = pd.date_range(start, periods=48, freq="30min")
    groups = efa_block_groups(start, 48)
    assert not is_complete_efa_group(index, *groups[0])
    assert is_complete_efa_group(index, *groups[1])
    assert not is_complete_efa_group(index, *groups[-1])
