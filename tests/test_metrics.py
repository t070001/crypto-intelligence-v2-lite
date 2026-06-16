import pandas as pd

from backtest_v2_lite import TradeResult
from metrics import (
    calc_max_drawdown,
    calc_profit_factor,
    evaluate_trades,
    trades_to_frame,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _trade(symbol: str, mode: str, return_pct: float) -> TradeResult:
    return TradeResult(
        symbol=symbol,
        mode=mode,
        entry_time=_ts("2026-01-01 00:00"),
        entry_price=100.0,
        exit_time=_ts("2026-01-01 04:00"),
        exit_price=100.0 + return_pct,
        sl_hard=90.0,
        tp1_price=None,
        trailing_stop=None,
        return_pct=return_pct,
        max_drawdown_pct=min(return_pct, 0.0),
        max_favorable_pct=max(return_pct, 0.0),
        reason="test",
        diagnostics={},
    )


def test_trades_to_frame_preserves_trade_fields():
    trades = [_trade("BTCUSDT", "MODE_A_SWEEP_RECLAIM", 2.0)]

    df = trades_to_frame(trades)

    assert len(df) == 1
    assert df.loc[0, "symbol"] == "BTCUSDT"
    assert df.loc[0, "return_pct"] == 2.0


def test_calc_profit_factor_uses_gross_gains_and_losses():
    returns = pd.Series([4.0, -2.0, 1.0, -1.0])

    assert calc_profit_factor(returns) == 5.0 / 3.0


def test_calc_max_drawdown_from_equity_curve():
    returns = pd.Series([10.0, -10.0, -10.0])

    assert round(calc_max_drawdown(returns), 2) == -19.0


def test_evaluate_trades_returns_empty_report_for_no_trades():
    report = evaluate_trades([])

    assert report.trade_count == 0
    assert report.mode_breakdown == {}


def test_evaluate_trades_calculates_summary_and_mode_breakdown():
    trades = [
        _trade("BTCUSDT", "MODE_A_SWEEP_RECLAIM", 4.0),
        _trade("ETHUSDT", "MODE_A_SWEEP_RECLAIM", -2.0),
        _trade("SOLUSDT", "MODE_B_OI_IMPULSE_BREAKOUT", 1.0),
    ]

    report = evaluate_trades(trades)

    assert report.trade_count == 3
    assert round(report.win_rate, 2) == 66.67
    assert report.avg_return_pct == 1.0
    assert report.profit_factor == 2.5
    assert report.best_trade_pct == 4.0
    assert report.worst_trade_pct == -2.0
    assert set(report.mode_breakdown) == {
        "MODE_A_SWEEP_RECLAIM",
        "MODE_B_OI_IMPULSE_BREAKOUT",
    }
    assert report.mode_breakdown["MODE_A_SWEEP_RECLAIM"].trade_count == 2
