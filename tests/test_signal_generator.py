import pandas as pd

from indicators import BreakoutPlatform
from signal_generator import (
    check_mode_a,
    check_mode_b,
    generate_entry_signal,
    get_intrabar_window,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _h4_row():
    return pd.Series(
        {
            "open_time": _ts("2026-01-10 12:00"),
            "close_time": _ts("2026-01-10 16:00"),
            "open": 101.0,
            "high": 108.0,
            "low": 99.8,
            "close": 107.0,
        }
    )


def _h4_history(rows: int = 20) -> pd.DataFrame:
    start = _ts("2026-01-07 04:00")
    values = []
    for index in range(rows):
        open_time = start + pd.Timedelta(hours=4 * index)
        values.append(
            {
                "open_time": open_time,
                "close_time": open_time + pd.Timedelta(hours=4),
                "open": 102.0,
                "high": 106.0 + index * 0.1,
                "low": 100.0,
                "close": 104.0 + index * 0.1,
            }
        )
    return pd.DataFrame(values)


def _m15_window_for_mode_a() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "open_time": _ts("2026-01-10 12:00"),
                "close_time": _ts("2026-01-10 12:15"),
                "open": 100.5,
                "high": 101.0,
                "low": 99.7,
                "close": 100.2,
                "vwap_rolling_24h": 100.4,
                "taker_buy_ratio": 0.9,
            },
            {
                "open_time": _ts("2026-01-10 12:15"),
                "close_time": _ts("2026-01-10 12:30"),
                "open": 100.0,
                "high": 102.0,
                "low": 99.6,
                "close": 101.7,
                "vwap_rolling_24h": 100.8,
                "taker_buy_ratio": 1.35,
            },
            {
                "open_time": _ts("2026-01-10 16:00"),
                "close_time": _ts("2026-01-10 16:15"),
                "open": 103.0,
                "high": 104.0,
                "low": 102.0,
                "close": 103.5,
                "vwap_rolling_24h": 100.8,
                "taker_buy_ratio": 2.0,
            },
        ]
    )


def test_get_intrabar_window_excludes_next_4h_candle():
    window = get_intrabar_window(_m15_window_for_mode_a(), _h4_row())

    assert len(window) == 2
    assert window["open_time"].max() < _h4_row()["close_time"]


def test_check_mode_a_requires_sweep_then_reclaim():
    signal = check_mode_a(
        symbol="TESTUSDT",
        h4_history=_h4_history(),
        h4_row=_h4_row(),
        df_15m=_m15_window_for_mode_a(),
        atr=2.0,
    )

    assert signal is not None
    assert signal.mode == "MODE_A_SWEEP_RECLAIM"
    assert signal.entry_price == 101.7
    assert signal.diagnostics["sweep_pct"] <= 0.005


def test_check_mode_a_rejects_deep_breakdown():
    df_15m = _m15_window_for_mode_a()
    df_15m.loc[0, "low"] = 98.0
    df_15m.loc[1, "low"] = 98.0

    signal = check_mode_a(
        symbol="TESTUSDT",
        h4_history=_h4_history(),
        h4_row=_h4_row(),
        df_15m=df_15m,
        atr=2.0,
    )

    assert signal is None


def _m15_for_mode_b() -> pd.DataFrame:
    rows = []
    start = _ts("2026-01-10 04:00")
    for index in range(48):
        open_time = start + pd.Timedelta(minutes=15 * index)
        if index < 36:
            taker_buy_quote = 100.0
            taker_sell_quote = 100.0
        else:
            taker_buy_quote = 150.0 + (index - 35) * 20.0
            taker_sell_quote = 80.0

        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + pd.Timedelta(minutes=15),
                "open": 100.0 + index * 0.1,
                "high": 101.0 + index * 0.1,
                "low": 99.0 + index * 0.1,
                "close": 100.5 + index * 0.1,
                "taker_buy_quote": taker_buy_quote,
                "taker_sell_quote": taker_sell_quote,
                "taker_buy_ratio": taker_buy_quote / taker_sell_quote,
            }
        )
    return pd.DataFrame(rows)


def _oi_aligned() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sumOpenInterestValue": [10_000_000.0, 10_200_000.0, 10_800_000.0],
        },
        index=[
            _ts("2026-01-10 04:00"),
            _ts("2026-01-10 08:00"),
            _ts("2026-01-10 12:00"),
        ],
    )


def test_check_mode_b_requires_breakout_oi_and_cvd_confirmation():
    platform = BreakoutPlatform(
        found=True,
        breakout_price=104.0,
        range_pct=0.08,
        extension_pct=2.0,
        lookback=50,
    )

    signal = check_mode_b(
        symbol="TESTUSDT",
        h4_history=_h4_history(rows=3),
        h4_row=_h4_row(),
        df_15m=_m15_for_mode_b(),
        oi_aligned=_oi_aligned(),
        platform=platform,
        atr=2.0,
    )

    assert signal is not None
    assert signal.mode == "MODE_B_OI_IMPULSE_BREAKOUT"
    assert signal.entry_price == 104.4
    assert signal.diagnostics["oi_change_2x4h"] > 0.05


def test_generate_entry_signal_prioritizes_mode_a_over_mode_b():
    platform = BreakoutPlatform(
        found=True,
        breakout_price=104.0,
        range_pct=0.08,
        extension_pct=2.0,
        lookback=50,
    )
    df_15m = pd.concat(
        [_m15_for_mode_b(), _m15_window_for_mode_a()],
        ignore_index=True,
    ).sort_values("open_time")

    signal = generate_entry_signal(
        symbol="TESTUSDT",
        h4_history=_h4_history(rows=20),
        h4_row=_h4_row(),
        df_15m=df_15m,
        oi_aligned=_oi_aligned(),
        platform=platform,
        atr=2.0,
    )

    assert signal is not None
    assert signal.mode == "MODE_A_SWEEP_RECLAIM"
