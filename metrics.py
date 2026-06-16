from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest_v2_lite import TradeResult


@dataclass(frozen=True)
class ModeBreakdown:
    mode: str
    trade_count: int
    win_rate: float
    avg_return_pct: float
    profit_factor: float


@dataclass(frozen=True)
class PerformanceReport:
    trade_count: int
    win_rate: float
    avg_return_pct: float
    expectancy_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    best_trade_pct: float
    worst_trade_pct: float
    mode_breakdown: dict[str, ModeBreakdown]


def trades_to_frame(trades: list[TradeResult]) -> pd.DataFrame:
    rows = [
        {
            "symbol": trade.symbol,
            "mode": trade.mode,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "return_pct": trade.return_pct,
            "max_drawdown_pct": trade.max_drawdown_pct,
            "max_favorable_pct": trade.max_favorable_pct,
            "reason": trade.reason,
        }
        for trade in trades
    ]

    return pd.DataFrame(rows)


def calc_profit_factor(returns_pct: pd.Series) -> float:
    gains = returns_pct[returns_pct > 0].sum()
    losses = returns_pct[returns_pct < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return float(gains / abs(losses))


def calc_max_drawdown(returns_pct: pd.Series) -> float:
    if returns_pct.empty:
        return 0.0

    equity = (1 + returns_pct / 100).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity / running_peak - 1) * 100
    return float(drawdown.min())


def calc_sharpe(returns_pct: pd.Series) -> float:
    if len(returns_pct) < 2:
        return 0.0

    std = returns_pct.std(ddof=1)

    if std == 0 or np.isnan(std):
        return 0.0

    return float(returns_pct.mean() / std)


def calc_sortino(returns_pct: pd.Series) -> float:
    if len(returns_pct) < 2:
        return 0.0

    downside = returns_pct[returns_pct < 0]

    if downside.empty:
        return float("inf") if returns_pct.mean() > 0 else 0.0

    downside_std = downside.std(ddof=1)

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float(returns_pct.mean() / downside_std)


def _mode_breakdown(df: pd.DataFrame) -> dict[str, ModeBreakdown]:
    breakdown: dict[str, ModeBreakdown] = {}

    for mode, group in df.groupby("mode"):
        returns = group["return_pct"]
        breakdown[str(mode)] = ModeBreakdown(
            mode=str(mode),
            trade_count=int(len(group)),
            win_rate=float((returns > 0).mean() * 100),
            avg_return_pct=float(returns.mean()),
            profit_factor=calc_profit_factor(returns),
        )

    return breakdown


def evaluate_trades(trades: list[TradeResult]) -> PerformanceReport:
    df = trades_to_frame(trades)

    if df.empty:
        return PerformanceReport(
            trade_count=0,
            win_rate=0.0,
            avg_return_pct=0.0,
            expectancy_pct=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            mode_breakdown={},
        )

    returns = df["return_pct"]

    return PerformanceReport(
        trade_count=int(len(df)),
        win_rate=float((returns > 0).mean() * 100),
        avg_return_pct=float(returns.mean()),
        expectancy_pct=float(returns.mean()),
        profit_factor=calc_profit_factor(returns),
        max_drawdown_pct=calc_max_drawdown(returns),
        sharpe=calc_sharpe(returns),
        sortino=calc_sortino(returns),
        best_trade_pct=float(returns.max()),
        worst_trade_pct=float(returns.min()),
        mode_breakdown=_mode_breakdown(df),
    )
