from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from indicators import BreakoutPlatform
from indicators import calc_cvd_slope_zscore


@dataclass(frozen=True)
class EntrySignal:
    symbol: str
    mode: str
    entry_time: pd.Timestamp
    entry_price: float
    atr: float
    reason: str
    diagnostics: dict[str, Any]


def get_intrabar_window(
    df_15m: pd.DataFrame,
    h4_row: pd.Series,
) -> pd.DataFrame:
    """Return 15m candles inside the 4H candle without look-ahead."""

    return df_15m[
        (df_15m["open_time"] >= h4_row["open_time"])
        & (df_15m["open_time"] < h4_row["close_time"])
    ].copy()


def calc_prior_swing_low(
    h4_history: pd.DataFrame,
    lookback: int = 20,
) -> float | None:
    if len(h4_history) < lookback:
        return None

    return float(h4_history.tail(lookback)["low"].min())


def _lower_wick_rejection(row: pd.Series) -> float:
    candle_range = float(row["high"] - row["low"])

    if candle_range <= 0:
        return 0.0

    return float((row["close"] - row["low"]) / candle_range)


def check_mode_a(
    symbol: str,
    h4_history: pd.DataFrame,
    h4_row: pd.Series,
    df_15m: pd.DataFrame,
    atr: float,
    swing_lookback: int = 20,
    max_sweep_pct: float = 0.005,
    wick_threshold: float = 0.5,
    taker_buy_ratio_threshold: float = 1.1,
) -> EntrySignal | None:
    swing_low = calc_prior_swing_low(h4_history, swing_lookback)

    if swing_low is None or np.isnan(atr):
        return None

    window = get_intrabar_window(df_15m, h4_row)

    if window.empty:
        return None

    sweep_mask = (
        (window["low"] < swing_low)
        & (((swing_low - window["low"]) / swing_low) <= max_sweep_pct)
    )

    sweep_rows = window[sweep_mask]

    if sweep_rows.empty:
        return None

    first_sweep = sweep_rows.iloc[0]
    reclaim_candidates = window[window["open_time"] >= first_sweep["open_time"]]

    for _, row in reclaim_candidates.iterrows():
        wick_ratio = _lower_wick_rejection(row)
        taker_buy_ratio = float(row.get("taker_buy_ratio", np.nan))

        if (
            float(row["close"]) > float(row["vwap_rolling_24h"])
            and wick_ratio > wick_threshold
            and taker_buy_ratio > taker_buy_ratio_threshold
        ):
            return EntrySignal(
                symbol=symbol,
                mode="MODE_A_SWEEP_RECLAIM",
                entry_time=row["close_time"],
                entry_price=float(row["close"]),
                atr=float(atr),
                reason="Sweep under prior 20x4H low and reclaim rolling VWAP",
                diagnostics={
                    "swing_low": swing_low,
                    "sweep_time": first_sweep["open_time"],
                    "sweep_low": float(first_sweep["low"]),
                    "sweep_pct": float((swing_low - first_sweep["low"]) / swing_low),
                    "reclaim_time": row["open_time"],
                    "vwap_rolling_24h": float(row["vwap_rolling_24h"]),
                    "wick_ratio": wick_ratio,
                    "taker_buy_ratio": taker_buy_ratio,
                },
            )

    return None


def _oi_change_over_two_bars(
    oi_aligned: pd.DataFrame,
    h4_row: pd.Series,
) -> float | None:
    if oi_aligned.empty:
        return None

    eligible = oi_aligned[oi_aligned.index <= h4_row["open_time"]]

    if len(eligible) < 3:
        return None

    current_oi = float(eligible["sumOpenInterestValue"].iloc[-1])
    prior_oi = float(eligible["sumOpenInterestValue"].iloc[-3])

    if prior_oi <= 0:
        return None

    return (current_oi - prior_oi) / prior_oi


def check_mode_b(
    symbol: str,
    h4_history: pd.DataFrame,
    h4_row: pd.Series,
    df_15m: pd.DataFrame,
    oi_aligned: pd.DataFrame,
    platform: BreakoutPlatform,
    atr: float,
    oi_threshold: float = 0.025,
    cvd_z_threshold: float = 0.5,
    taker_buy_ratio_threshold: float = 1.15,
) -> EntrySignal | None:
    if (
        not platform.found
        or platform.breakout_price is None
        or np.isnan(atr)
        or float(h4_row["close"]) <= platform.breakout_price
    ):
        return None

    if h4_history.empty:
        return None

    price_up = float(h4_row["close"]) > float(h4_history["close"].iloc[-1])

    if not price_up:
        return None

    oi_change = _oi_change_over_two_bars(oi_aligned, h4_row)

    if oi_change is None or oi_change <= oi_threshold:
        return None

    window = df_15m[df_15m["open_time"] < h4_row["close_time"]].copy()

    if window.empty:
        return None

    cvd = calc_cvd_slope_zscore(window)
    recent = cvd.tail(8)

    if recent.empty:
        return None

    latest = recent.iloc[-1]
    recent_taker_buy_ratio = float(recent["taker_buy_ratio"].mean())
    cvd_slope = float(latest.get("cvd_slope", np.nan))
    cvd_zscore = float(latest.get("cvd_slope_zscore", np.nan))

    if (
        not np.isfinite(cvd_slope)
        or not np.isfinite(cvd_zscore)
        or cvd_slope <= 0
        or cvd_zscore <= cvd_z_threshold
        or recent_taker_buy_ratio <= taker_buy_ratio_threshold
    ):
        return None

    entry_price = float(platform.breakout_price + (0.2 * atr))

    return EntrySignal(
        symbol=symbol,
        mode="MODE_B_OI_IMPULSE_BREAKOUT",
        entry_time=h4_row["close_time"],
        entry_price=entry_price,
        atr=float(atr),
        reason="4H breakout with OI impulse and positive CVD_approx acceleration",
        diagnostics={
            "breakout_price": float(platform.breakout_price),
            "oi_change_2x4h": float(oi_change),
            "cvd_slope": cvd_slope,
            "cvd_slope_zscore": cvd_zscore,
            "recent_taker_buy_ratio": recent_taker_buy_ratio,
        },
    )


def generate_entry_signal(
    symbol: str,
    h4_history: pd.DataFrame,
    h4_row: pd.Series,
    df_15m: pd.DataFrame,
    oi_aligned: pd.DataFrame,
    platform: BreakoutPlatform,
    atr: float,
) -> EntrySignal | None:
    mode_a = check_mode_a(
        symbol=symbol,
        h4_history=h4_history,
        h4_row=h4_row,
        df_15m=df_15m,
        atr=atr,
    )

    if mode_a:
        return mode_a

    return check_mode_b(
        symbol=symbol,
        h4_history=h4_history,
        h4_row=h4_row,
        df_15m=df_15m,
        oi_aligned=oi_aligned,
        platform=platform,
        atr=atr,
    )
