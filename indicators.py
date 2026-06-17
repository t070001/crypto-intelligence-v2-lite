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


def calc_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple Moving Average."""
    return df["close"].rolling(period, min_periods=period).mean()


def calc_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return df["close"].ewm(span=period, adjust=False).mean()


def calc_market_structure(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Determine market structure based on HH/HL pattern.

    Returns
    -------
    dict with keys:
      - structure: 'bullish' | 'neutral' | 'bearish'
      - hh_confirmed: bool
      - hl_confirmed: bool
      - msb: bool (Market Structure Break — price broke above prior high)
    """
    window = df.tail(lookback)
    if len(window) < lookback:
        return {"structure": "neutral", "hh_confirmed": False,
                "hl_confirmed": False, "msb": False}

    # Split window into two halves
    mid = lookback // 2
    first_half = window.iloc[:mid]
    second_half = window.iloc[mid:]

    first_high = float(first_half["high"].max())
    first_low = float(first_half["low"].min())
    second_high = float(second_half["high"].max())
    second_low = float(second_half["low"].min())

    hh = second_high > first_high  # Higher High
    hl = second_low > first_low    # Higher Low

    # MSB: recent close above first-half high
    recent_close = float(window["close"].iloc[-1])
    msb = recent_close > first_high

    if hh and hl:
        structure = "bullish"
    elif not hh and not hl:
        structure = "bearish"
    else:
        structure = "neutral"

    return {
        "structure": structure,
        "hh_confirmed": hh,
        "hl_confirmed": hl,
        "msb": msb,
    }


def calc_multi_structure(
    df: pd.DataFrame,
    windows: list[int] = None,
) -> dict[str, dict]:
    """Multi-timeframe structure analysis.

    For each window, computes HH/HL pattern by splitting into two halves.
    Returns dict keyed by window size, each with structure/hh/hl/msb.

    Example output:
    {5: {'structure': 'bullish', 'hh_confirmed': True, ...},
     10: {'structure': 'bullish', ...},
     20: {'structure': 'neutral', ...}}
    """
    if windows is None:
        windows = [5, 10, 20]

    result = {}
    for w in windows:
        result[str(w)] = calc_market_structure(df, lookback=w)
    return result


def calc_oi_breakout_strength(
    oi_series: pd.Series,
    lookback: int = 20,
) -> dict:
    """Compute OI breakout strength as a continuous value.

    strength = (current_oi - highest_oi_20) / highest_oi_20
      - Breakout 1%  → strength = 0.01
      - Breakout 20% → strength = 0.20
      - No breakout  → strength = 0.0

    Returns dict {strength: float, breakout_level: float, current_oi: float}
    """
    if len(oi_series) < lookback + 1:
        return {"strength": 0.0, "breakout_level": 0.0, "current_oi": 0.0}

    window = oi_series.tail(lookback + 1).iloc[:-1]
    prior_high = float(window.max())
    current_oi = float(oi_series.iloc[-1])

    if prior_high <= 0:
        return {"strength": 0.0, "breakout_level": prior_high, "current_oi": current_oi}

    strength = max((current_oi - prior_high) / prior_high, 0.0)
    return {"strength": strength, "breakout_level": prior_high, "current_oi": current_oi}


def calc_oi_slope(oi_series: pd.Series, window: int = 5) -> float:
    """Compute linear regression slope of OI values over window.

    Positive slope = OI expanding (capital flowing in).
    Negative slope = OI contracting (capital flowing out).
    """
    if len(oi_series) < window:
        return 0.0
    values = oi_series.tail(window).values.astype(float)
    if np.isnan(values).any():
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def calc_oi_breakout(oi_series: pd.Series, lookback: int = 20) -> dict:
    """Detect if OI has broken above its recent range.

    Returns
    -------
    dict: {breakout: bool, breakout_level: float, current_oi: float}
    """
    if len(oi_series) < lookback:
        return {"breakout": False, "breakout_level": 0.0, "current_oi": 0.0}

    window = oi_series.tail(lookback + 1).iloc[:-1]  # exclude current bar
    recent = oi_series.tail(2).iloc[0]  # second-to-last bar (confirmed)
    current = float(oi_series.iloc[-1])

    prior_high = float(window.max())
    breakout = recent > prior_high

    return {"breakout": breakout, "breakout_level": prior_high, "current_oi": current}


def calc_oi_relative_expansion(
    oi_series: pd.Series,
    window: int = 10,
    threshold: float = 1.5,
) -> dict:
    """Measure if current OI is expanding relative to its recent average.

    Returns
    -------
    dict: {expanding: bool, ratio: float, avg_oi: float, current_oi: float}
    """
    if len(oi_series) < window + 1:
        return {"expanding": False, "ratio": 0.0, "avg_oi": 0.0, "current_oi": 0.0}

    avg_oi = float(oi_series.tail(window).mean())
    current_oi = float(oi_series.iloc[-1])

    if avg_oi <= 0:
        return {"expanding": False, "ratio": 0.0, "avg_oi": avg_oi, "current_oi": current_oi}

    ratio = current_oi / avg_oi
    return {"expanding": ratio >= threshold, "ratio": ratio, "avg_oi": avg_oi, "current_oi": current_oi}


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
