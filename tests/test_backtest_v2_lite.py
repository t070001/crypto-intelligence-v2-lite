import pandas as pd

from backtest_v2_lite import run_symbol_backtest, simulate_position
from signal_generator import EntrySignal


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _signal() -> EntrySignal:
    return EntrySignal(
        symbol="TESTUSDT",
        mode="MODE_A_SWEEP_RECLAIM",
        entry_time=_ts("2026-01-10 12:30"),
        entry_price=100.0,
        atr=10.0,
        reason="test",
        diagnostics={},
    )


def test_simulate_position_hard_stop_has_priority():
    df_15m = pd.DataFrame(
        [
            {
                "open_time": _ts("2026-01-10 12:30"),
                "close_time": _ts("2026-01-10 12:45"),
                "open": 100.0,
                "high": 110.0,
                "low": 81.0,
                "close": 90.0,
                "taker_buy_quote": 100.0,
                "taker_sell_quote": 100.0,
            }
        ]
    )

    trade = simulate_position(_signal(), df_15m)

    assert trade is not None
    assert trade.reason == "Hard Stop"
    assert trade.exit_price == 82.0
    assert trade.tp1_price is None


def test_simulate_position_tp1_then_atr_trailing_stop():
    df_15m = pd.DataFrame(
        [
            {
                "open_time": _ts("2026-01-10 12:30"),
                "close_time": _ts("2026-01-10 12:45"),
                "open": 100.0,
                "high": 116.0,
                "low": 99.0,
                "close": 114.0,
                "taker_buy_quote": 100.0,
                "taker_sell_quote": 200.0,
            },
            {
                "open_time": _ts("2026-01-10 12:45"),
                "close_time": _ts("2026-01-10 13:00"),
                "open": 114.0,
                "high": 122.0,
                "low": 110.0,
                "close": 120.0,
                "taker_buy_quote": 100.0,
                "taker_sell_quote": 100.0,
            },
        ]
    )

    trade = simulate_position(_signal(), df_15m)

    assert trade is not None
    assert trade.reason == "ATR Trailing Stop"
    assert trade.tp1_price == 114.0
    assert trade.exit_price == 110.0
    assert round(trade.return_pct, 2) == 12.4


def _h4_dataset() -> pd.DataFrame:
    rows = []
    start = _ts("2026-01-01 00:00")
    for index in range(21):
        open_time = start + pd.Timedelta(hours=4 * index)
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + pd.Timedelta(hours=4),
                "open": 102.0,
                "high": 106.0,
                "low": 100.0,
                "close": 104.0,
            }
        )

    rows[-1].update(
        {
            "open_time": _ts("2026-01-10 12:00"),
            "close_time": _ts("2026-01-10 16:00"),
            "open": 101.0,
            "high": 108.0,
            "low": 99.6,
            "close": 106.0,
        }
    )
    return pd.DataFrame(rows)


def _m15_dataset() -> pd.DataFrame:
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
                "taker_buy_quote": 100.0,
                "taker_sell_quote": 100.0,
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
                "taker_buy_quote": 135.0,
                "taker_sell_quote": 100.0,
            },
            {
                "open_time": _ts("2026-01-10 12:30"),
                "close_time": _ts("2026-01-10 12:45"),
                "open": 101.7,
                "high": 103.0,
                "low": 80.0,
                "close": 90.0,
                "vwap_rolling_24h": 100.8,
                "taker_buy_ratio": 1.0,
                "taker_buy_quote": 100.0,
                "taker_sell_quote": 100.0,
            },
        ]
    )


def test_run_symbol_backtest_generates_trade_from_mode_a():
    trades = run_symbol_backtest(
        symbol="TESTUSDT",
        df_4h=_h4_dataset(),
        df_15m=_m15_dataset(),
        oi_aligned=pd.DataFrame(),
        min_history_bars=20,
    )

    assert len(trades) == 1
    assert trades[0].mode == "MODE_A_SWEEP_RECLAIM"
    assert trades[0].reason == "Hard Stop"
