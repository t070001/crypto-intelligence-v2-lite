from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from indicators import calc_atr
from indicators import calc_breakout_platform
from signal_generator import EntrySignal
from signal_generator import generate_entry_signal


class PositionState(str, Enum):
    HOLDING_FULL = "HOLDING_FULL"
    HOLDING_40_AFTER_TP1 = "HOLDING_40_AFTER_TP1"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    mode: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    sl_hard: float
    tp1_price: float | None
    trailing_stop: float | None
    return_pct: float
    max_drawdown_pct: float
    max_favorable_pct: float
    reason: str
    diagnostics: dict[str, Any]


def _slice_after_entry(
    df_15m: pd.DataFrame,
    entry_time: pd.Timestamp,
) -> pd.DataFrame:
    return df_15m[df_15m["close_time"] > entry_time].copy()


def _weighted_return(
    entry_price: float,
    tp1_price: float | None,
    exit_price: float,
    tp1_weight: float = 0.6,
) -> float:
    if tp1_price is None:
        return ((exit_price - entry_price) / entry_price) * 100

    tp1_return = ((tp1_price - entry_price) / entry_price) * 100
    final_return = ((exit_price - entry_price) / entry_price) * 100
    return (tp1_return * tp1_weight) + (final_return * (1 - tp1_weight))


def _finalize_trade(
    signal: EntrySignal,
    exit_time: pd.Timestamp,
    exit_price: float,
    sl_hard: float,
    tp1_price: float | None,
    trailing_stop: float | None,
    reason: str,
    lows_seen: list[float],
    highs_seen: list[float],
) -> TradeResult:
    max_drawdown_pct = (
        ((min(lows_seen) - signal.entry_price) / signal.entry_price) * 100
        if lows_seen
        else 0.0
    )
    max_favorable_pct = (
        ((max(highs_seen) - signal.entry_price) / signal.entry_price) * 100
        if highs_seen
        else 0.0
    )

    return_pct = _weighted_return(
        entry_price=signal.entry_price,
        tp1_price=tp1_price,
        exit_price=exit_price,
    )

    return TradeResult(
        symbol=signal.symbol,
        mode=signal.mode,
        entry_time=signal.entry_time,
        entry_price=signal.entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        sl_hard=sl_hard,
        tp1_price=tp1_price,
        trailing_stop=trailing_stop,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_favorable_pct=max_favorable_pct,
        reason=reason,
        diagnostics=signal.diagnostics,
    )


def simulate_position(
    signal: EntrySignal,
    df_15m: pd.DataFrame,
    max_hold_bars: int = 96,
    hard_stop_atr: float = 1.8,
    tp1_check_atr: float = 1.5,
    tp1_force_atr: float = 2.0,
    trailing_atr: float = 1.0,
    taker_sell_ratio_threshold: float = 1.5,
) -> TradeResult | None:
    future = _slice_after_entry(df_15m, signal.entry_time).head(max_hold_bars)

    if future.empty:
        return None

    sl_hard = signal.entry_price - (hard_stop_atr * signal.atr)
    tp1_check_price = signal.entry_price + (tp1_check_atr * signal.atr)
    tp1_force_price = signal.entry_price + (tp1_force_atr * signal.atr)

    state = PositionState.HOLDING_FULL
    tp1_price: float | None = None
    trailing_stop: float | None = None
    highest_close_after_tp1: float | None = None

    lows_seen: list[float] = []
    highs_seen: list[float] = []

    for _, row in future.iterrows():
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        lows_seen.append(low)
        highs_seen.append(high)

        if low <= sl_hard:
            return _finalize_trade(
                signal=signal,
                exit_time=row["close_time"],
                exit_price=sl_hard,
                sl_hard=sl_hard,
                tp1_price=tp1_price,
                trailing_stop=trailing_stop,
                reason="Hard Stop",
                lows_seen=lows_seen,
                highs_seen=highs_seen,
            )

        if state == PositionState.HOLDING_FULL:
            taker_buy = float(row.get("taker_buy_quote", np.nan))
            taker_sell = float(row.get("taker_sell_quote", np.nan))
            taker_sell_ratio = (
                taker_sell / taker_buy
                if taker_buy > 0
                else np.inf
            )

            if high >= tp1_check_price and taker_sell_ratio > taker_sell_ratio_threshold:
                tp1_price = close
                state = PositionState.HOLDING_40_AFTER_TP1
            elif high >= tp1_force_price:
                tp1_price = tp1_force_price
                state = PositionState.HOLDING_40_AFTER_TP1

            if state == PositionState.HOLDING_40_AFTER_TP1:
                highest_close_after_tp1 = close
                trailing_stop = highest_close_after_tp1 - (trailing_atr * signal.atr)
                continue

        if state == PositionState.HOLDING_40_AFTER_TP1:
            highest_close_after_tp1 = max(highest_close_after_tp1 or close, close)
            trailing_stop = highest_close_after_tp1 - (trailing_atr * signal.atr)

            if low <= trailing_stop:
                return _finalize_trade(
                    signal=signal,
                    exit_time=row["close_time"],
                    exit_price=trailing_stop,
                    sl_hard=sl_hard,
                    tp1_price=tp1_price,
                    trailing_stop=trailing_stop,
                    reason="ATR Trailing Stop",
                    lows_seen=lows_seen,
                    highs_seen=highs_seen,
                )

    last = future.iloc[-1]
    return _finalize_trade(
        signal=signal,
        exit_time=last["close_time"],
        exit_price=float(last["close"]),
        sl_hard=sl_hard,
        tp1_price=tp1_price,
        trailing_stop=trailing_stop,
        reason="Max Hold Exit",
        lows_seen=lows_seen,
        highs_seen=highs_seen,
    )


def run_symbol_backtest(
    symbol: str,
    df_4h: pd.DataFrame,
    df_15m: pd.DataFrame,
    oi_aligned: pd.DataFrame,
    min_history_bars: int = 50,
) -> list[TradeResult]:
    df_4h = df_4h.copy()
    df_4h["atr14"] = calc_atr(df_4h)
    trades: list[TradeResult] = []
    next_allowed_entry_time: pd.Timestamp | None = None

    for index in range(min_history_bars, len(df_4h)):
        h4_row = df_4h.iloc[index]

        if (
            next_allowed_entry_time is not None
            and h4_row["open_time"] < next_allowed_entry_time
        ):
            continue

        h4_history = df_4h.iloc[:index]
        atr = float(h4_row["atr14"])

        if np.isnan(atr):
            continue

        platform = calc_breakout_platform(h4_history)
        signal = generate_entry_signal(
            symbol=symbol,
            h4_history=h4_history,
            h4_row=h4_row,
            df_15m=df_15m,
            oi_aligned=oi_aligned,
            platform=platform,
            atr=atr,
        )

        if signal is None:
            continue

        trade = simulate_position(
            signal=signal,
            df_15m=df_15m,
        )

        if trade is None:
            continue

        trades.append(trade)
        next_allowed_entry_time = trade.exit_time

    return trades
