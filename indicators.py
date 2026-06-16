from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreakoutPlatform:
    found: bool
    breakout_price: float | None
    range_pct: float | None
    extension_pct: float | None
    lookback: int


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period, min_periods=period).mean()


def calc_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    average_volume = df["volume"].shift(1).rolling(window, min_periods=window).mean()
    return df["volume"] / average_volume.replace(0, np.nan)


def calc_breakout_platform(
    df: pd.DataFrame,
    min_lookback: int = 30,
    max_lookback: int = 50,
    compression_threshold: float = 0.15,
    quantile: float = 0.80,
) -> BreakoutPlatform:
    lookback = min(max_lookback, len(df))

    if lookback < min_lookback:
        return BreakoutPlatform(False, None, None, None, lookback)

    window = df.tail(lookback)
    min_price = float(window["low"].min())
    max_price = float(window["high"].max())

    if min_price <= 0:
        return BreakoutPlatform(False, None, None, None, lookback)

    range_pct = (max_price - min_price) / min_price

    if range_pct >= compression_threshold:
        return BreakoutPlatform(False, None, range_pct, None, lookback)

    breakout_price = float(window["high"].quantile(quantile))
    current_close = float(window["close"].iloc[-1])
    extension_pct = ((current_close - breakout_price) / breakout_price) * 100

    return BreakoutPlatform(
        found=True,
        breakout_price=breakout_price,
        range_pct=range_pct,
        extension_pct=extension_pct,
        lookback=lookback,
    )


def calc_cvd_approx(df: pd.DataFrame) -> pd.Series:
    if "taker_buy_quote" in df.columns and "taker_sell_quote" in df.columns:
        taker_buy = df["taker_buy_quote"]
        taker_sell = df["taker_sell_quote"]
    else:
        taker_buy = df["taker_quote"]
        taker_sell = df["quote_volume"] - df["taker_quote"]

    delta = taker_buy - taker_sell
    return delta.cumsum()


def _linear_regression_slope(values: np.ndarray) -> float:
    if len(values) < 2 or np.isnan(values).any():
        return np.nan

    x = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x, values.astype(float), 1)
    return float(slope)


def calc_cvd_slope_zscore(
    df: pd.DataFrame,
    slope_window: int = 8,
    z_window: int = 32,
) -> pd.DataFrame:
    """Calculate CVD_approx, rolling slope, and slope z-score.

    v2.0 Lite uses Binance kline taker fields, so this is an approximation of
    trade-level CVD rather than raw tick-by-tick order flow.
    """

    if slope_window < 2:
        raise ValueError("slope_window must be at least 2")

    if z_window < 2:
        raise ValueError("z_window must be at least 2")

    result = df.copy()
    result["cvd_approx"] = calc_cvd_approx(result)
    result["cvd_slope"] = (
        result["cvd_approx"]
        .rolling(slope_window, min_periods=slope_window)
        .apply(_linear_regression_slope, raw=True)
    )

    slope_mean = (
        result["cvd_slope"]
        .shift(1)
        .rolling(z_window, min_periods=z_window)
        .mean()
    )
    slope_std = (
        result["cvd_slope"]
        .shift(1)
        .rolling(z_window, min_periods=z_window)
        .std()
    )

    result["cvd_slope_zscore"] = (
        (result["cvd_slope"] - slope_mean)
        / slope_std.replace(0, np.nan)
    )

    return result
