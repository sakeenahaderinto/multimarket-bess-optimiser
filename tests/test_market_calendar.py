import numpy as np
import pandas as pd

from market_calendar import (
    aggregate_to_complete_utc_days,
    aggregate_to_efa_blocks,
    apply_dc_price_domain,
    efa_block_bounds_utc,
    efa_block_groups,
    expand_efa_block_values,
    is_complete_efa_group,
    last_complete_half_hour_start_times,
    last_complete_hour_start_times,
    n2ex_gate_times,
    realised_data_cutoff_times,
)


def test_efa_expansion_uses_ten_period_autumn_clock_change_block():
    start = pd.Timestamp("2025-10-25 22:00", tz="UTC")
    values = pd.Series([7.5], index=pd.DatetimeIndex([start]))

    result = expand_efa_block_values(values)

    assert len(result) == 10
    assert result.index[0] == start
    assert result.index[-1] == pd.Timestamp("2025-10-26 02:30", tz="UTC")
    assert (result == 7.5).all()


def test_efa_expansion_uses_six_period_spring_clock_change_block():
    start = pd.Timestamp("2025-03-29 23:00", tz="UTC")
    values = pd.Series([4.5], index=pd.DatetimeIndex([start]))

    result = expand_efa_block_values(values)

    assert len(result) == 6
    assert result.index[0] == start
    assert result.index[-1] == pd.Timestamp("2025-03-30 01:30", tz="UTC")
    assert (result == 4.5).all()


def test_efa_expansion_does_not_fill_an_unobserved_block():
    first_start = pd.Timestamp("2025-01-01 03:00", tz="UTC")
    third_start = pd.Timestamp("2025-01-01 11:00", tz="UTC")
    values = pd.Series(
        [10.0, 30.0],
        index=pd.DatetimeIndex([first_start, third_start]),
    )

    result = expand_efa_block_values(values)

    missing_block = pd.date_range("2025-01-01 07:00", periods=8, freq="30min", tz="UTC")
    assert result.index.intersection(missing_block).empty
    assert len(result) == 16


def test_efa_expansion_rejects_conflicting_values_within_a_block():
    index = pd.date_range("2025-01-01 03:00", periods=2, freq="30min", tz="UTC")
    values = pd.Series([10.0, 11.0], index=index)

    with np.testing.assert_raises_regex(ValueError, "conflicting values"):
        expand_efa_block_values(values)


def test_complete_utc_day_aggregation_excludes_incomplete_day():
    complete = pd.date_range("2025-01-01", periods=48, freq="30min", tz="UTC")
    incomplete = pd.date_range("2025-01-02", periods=47, freq="30min", tz="UTC")
    index = complete.append(incomplete)
    values = pd.Series(np.ones(len(index)), index=index)

    result = aggregate_to_complete_utc_days(values)

    assert result.index.tolist() == [pd.Timestamp("2025-01-01", tz="UTC")]
    assert result.iloc[0] == 1.0


def test_dc_price_domain_is_regime_aware():
    index = pd.DatetimeIndex(
        ["2023-10-31 23:30Z", "2023-11-01 00:00Z", "2023-11-01 00:30Z"]
    )
    values = np.array([[-2.0, -3.0, 4.0], [-5.0, 6.0, -7.0]])

    result = apply_dc_price_domain(values, index)

    assert np.array_equal(result, [[0.0, -3.0, 4.0], [0.0, 6.0, -7.0]])


def test_dc_price_domain_requires_periods_on_final_axis():
    index = pd.date_range("2025-01-01", periods=2, freq="30min", tz="UTC")

    with np.testing.assert_raises_regex(ValueError, "final axis"):
        apply_dc_price_domain(np.zeros((2, 3)), index)


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


def test_complete_block_aggregation_retains_dst_short_and_long_auctions():
    keys = [
        (pd.Timestamp("2025-03-29").date(), 23),
        (pd.Timestamp("2025-10-25").date(), 23),
    ]
    indexes = [
        pd.date_range(*efa_block_bounds_utc(key), freq="30min", inclusive="left")
        for key in keys
    ]
    values = pd.Series(
        np.concatenate(
            [np.full(len(index), position + 1.0) for position, index in enumerate(indexes)]
        ),
        index=indexes[0].append(indexes[1]),
    )

    result = aggregate_to_efa_blocks(values, complete_only=True)

    assert [len(index) for index in indexes] == [6, 10]
    assert result.tolist() == [1.0, 2.0]
