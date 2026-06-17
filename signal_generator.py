"""
Signal Generator — v2.x (legacy) + v3 (scoring) engines
=========================================================
This file contains:

1. v2.x functions (frozen) — used by backtest_v2_lite.py and main.py (legacy mode)
2. v3 ScoreBreakdown engine — new scoring system for Crypto Intelligence v3

Both can coexist. v3 functions are prefixed with score_/calc_.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import (
    MODE_A_SWEEP_THRESHOLD, MODE_A_TAKER_RATIO, MODE_A_VWAP_RECLAIM_MODE,
    MODE_A_VWAP_RECOVERY_RATIO, MODE_A_WICK_RATIO,
    MODE_B_CVD_REQUIRED, MODE_B_EXCEPTION_CANDLE_BODY,
    MODE_B_EXCEPTION_TAKER_RATIO, MODE_B_EXCEPTION_VOLUME_RATIO,
    MODE_B_OI_CHANGE_THRESHOLD, MODE_B_OI_EXCEPTION_ENABLED,
    MODE_B_OI_EXCEPTION_THRESHOLD, MODE_B_TAKER_RATIO_REQUIRED,
)
from config import (
    V3_MODE_A_SWEEP_THRESHOLD, V3_MODE_A_VWAP_RECOVERY_RATIO,
    V3_MODE_A_WICK_RATIO, V3_MODE_A_TAKER_RATIO,
    V3_MODE_A_OI_REJECT_THRESHOLD,
    V3_OI_CHANGE_SHORT, V3_OI_CHANGE_MID,
    V3_OI_CHANGE_SHORT_WEIGHT, V3_OI_CHANGE_MID_WEIGHT,
    V3_OI_BREAKOUT_LOOKBACK,
    V3_OI_EXPANSION_WINDOW,
    V3_OI_SLOPE_SHORT, V3_OI_SLOPE_MID,
    V3_OI_SLOPE_SHORT_WEIGHT, V3_OI_SLOPE_MID_WEIGHT,
    V3_OI_SLOPE_POSITIVE_THRESHOLD,
    V3_STRUCTURE_WINDOW_5_WEIGHT, V3_STRUCTURE_WINDOW_10_WEIGHT, V3_STRUCTURE_WINDOW_20_WEIGHT,
    V3_WEIGHT_OI, V3_WEIGHT_STRUCTURE, V3_WEIGHT_VOLUME,
    V3_WEIGHT_FUNDING, V3_WEIGHT_LONGSHORT, V3_WEIGHT_LIQUIDITY,
    V3_GRADE_S_THRESHOLD, V3_GRADE_A_THRESHOLD,
    V3_GRADE_B_THRESHOLD, V3_GRADE_C_THRESHOLD,
    V3_HTF_MIN_SCORE,
    V3_LOTTERY_MAX_DECLINE_PCT, V3_LOTTERY_PENALTY,
    V3_SC_RALLY_PENALTY,
    V3_VOLUME_RATIO_MAX,
    V3_ENTRY_PRIORITY, V3_ENTRY_ZONE_SPREAD,
)
from indicators import (
    BreakoutPlatform, calc_atr,
    calc_breakout_platform, calc_cvd_slope_zscore,
    calc_market_structure, calc_multi_structure, calc_oi_slope,
    calc_oi_breakout, calc_oi_breakout_strength,
    calc_oi_relative_expansion, calc_sma,
)


# =====================================================================
# v2.x Legacy Dataclasses & Functions (FROZEN — do not modify)
# =====================================================================

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


def _vwap_reclaimed(
    row: pd.Series,
    swing_low: float,
    mode: str = MODE_A_VWAP_RECLAIM_MODE,
    recovery_ratio: float = MODE_A_VWAP_RECOVERY_RATIO,
) -> bool:
    vwap = float(row["vwap_rolling_24h"])
    if mode == "relative":
        if vwap <= swing_low:
            return False
        recovery = (float(row["high"]) - swing_low) / (vwap - swing_low)
        return recovery >= recovery_ratio
    if mode == "high":
        return float(row["high"]) >= vwap
    if mode == "close_995":
        return float(row["close"]) >= vwap * 0.995
    return float(row["close"]) > vwap


def _relative_recovery_ratio(row: pd.Series, swing_low: float) -> float | None:
    vwap = float(row["vwap_rolling_24h"])
    if vwap <= swing_low:
        return None
    return float((float(row["high"]) - swing_low) / (vwap - swing_low))


def _lower_wick_rejection(row: pd.Series) -> float:
    candle_range = float(row["high"] - row["low"])
    if candle_range <= 0:
        return 0.0
    return float((row["close"] - row["low"]) / candle_range)


def _recent_taker_buy_ratio(df_15m: pd.DataFrame, h4_row: pd.Series) -> float:
    eligible = df_15m[df_15m["open_time"] < h4_row["close_time"]].tail(8)
    if eligible.empty:
        return np.nan
    value = float(eligible["taker_buy_ratio"].mean())
    return value if np.isfinite(value) else np.nan


def _mode_b_oi_exception(
    h4_row: pd.Series,
    df_15m: pd.DataFrame,
    oi_change: float,
) -> bool:
    if not MODE_B_OI_EXCEPTION_ENABLED:
        return False
    if oi_change >= 0 or oi_change < MODE_B_OI_EXCEPTION_THRESHOLD:
        return False
    open_price = float(h4_row["open"])
    if open_price <= 0:
        return False
    body_pct = (float(h4_row["close"]) - open_price) / open_price
    if body_pct <= MODE_B_EXCEPTION_CANDLE_BODY:
        return False
    volume_ratio = float(h4_row.get("volume_ratio", np.nan))
    if not np.isfinite(volume_ratio) or volume_ratio <= MODE_B_EXCEPTION_VOLUME_RATIO:
        return False
    taker_buy_ratio = _recent_taker_buy_ratio(df_15m, h4_row)
    if not np.isfinite(taker_buy_ratio) or taker_buy_ratio <= MODE_B_EXCEPTION_TAKER_RATIO:
        return False
    return True


def check_mode_a(
    symbol: str,
    h4_history: pd.DataFrame,
    h4_row: pd.Series,
    df_15m: pd.DataFrame,
    atr: float,
    swing_lookback: int = 20,
    max_sweep_pct: float = MODE_A_SWEEP_THRESHOLD,
    wick_threshold: float = MODE_A_WICK_RATIO,
    taker_buy_ratio_threshold: float = MODE_A_TAKER_RATIO,
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
        recovery_ratio = _relative_recovery_ratio(row, swing_low)
        if (
            _vwap_reclaimed(row, swing_low)
            and wick_ratio > wick_threshold
            and taker_buy_ratio > taker_buy_ratio_threshold
        ):
            return EntrySignal(
                symbol=symbol,
                mode="MODE_A_SWEEP_RECLAIM",
                entry_time=row["close_time"],
                entry_price=float(row["close"]),
                atr=float(atr),
                reason=(
                    "Sweep under prior 20x4H low and relative VWAP recovery "
                    f">= {MODE_A_VWAP_RECOVERY_RATIO:.1%}"
                ),
                diagnostics={
                    "swing_low": swing_low,
                    "sweep_time": first_sweep["open_time"],
                    "sweep_low": float(first_sweep["low"]),
                    "sweep_pct": float((swing_low - first_sweep["low"]) / swing_low),
                    "reclaim_time": row["open_time"],
                    "vwap_rolling_24h": float(row["vwap_rolling_24h"]),
                    "recovery_ratio": recovery_ratio,
                    "wick_ratio": wick_ratio,
                    "taker_buy_ratio": taker_buy_ratio,
                    "signal_grade": "AB",
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
    oi_threshold: float = MODE_B_OI_CHANGE_THRESHOLD,
    cvd_required: bool = MODE_B_CVD_REQUIRED,
    taker_buy_ratio_threshold: float | None = (
        1.15 if MODE_B_TAKER_RATIO_REQUIRED else None
    ),
) -> EntrySignal | None:
    if not platform.found or platform.breakout_price is None or np.isnan(atr) or float(h4_row["close"]) <= platform.breakout_price:
        return None
    if h4_history.empty:
        return None
    price_up = float(h4_row["close"]) > float(h4_history["close"].iloc[-1])
    if not price_up:
        return None
    oi_change = _oi_change_over_two_bars(oi_aligned, h4_row)
    if oi_change is None:
        return None
    oi_exception = False
    if oi_change >= oi_threshold:
        signal_grade = "AB"
    elif _mode_b_oi_exception(h4_row, df_15m, oi_change):
        signal_grade = "C"
        oi_exception = True
    else:
        return None
    window = df_15m[df_15m["open_time"] < h4_row["close_time"]].copy()
    cvd_slope = np.nan
    cvd_zscore = np.nan
    recent_taker_buy_ratio = _recent_taker_buy_ratio(df_15m, h4_row)
    if cvd_required:
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
        if not np.isfinite(cvd_slope) or cvd_slope <= 0:
            return None
        if taker_buy_ratio_threshold is not None and recent_taker_buy_ratio <= taker_buy_ratio_threshold:
            return None
    elif taker_buy_ratio_threshold is not None and np.isfinite(recent_taker_buy_ratio) and recent_taker_buy_ratio <= taker_buy_ratio_threshold:
        return None
    entry_price = float(platform.breakout_price + (0.2 * atr))
    reason = (
        "4H breakout via OI technical exception (C-grade FOMO sample)"
        if oi_exception
        else "4H breakout with OI impulse"
    )
    return EntrySignal(
        symbol=symbol,
        mode="MODE_B_OI_IMPULSE_BREAKOUT",
        entry_time=h4_row["close_time"],
        entry_price=entry_price,
        atr=float(atr),
        reason=reason,
        diagnostics={
            "breakout_price": float(platform.breakout_price),
            "oi_change_2x4h": float(oi_change),
            "oi_exception": oi_exception,
            "signal_grade": signal_grade,
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
    mode_a = check_mode_a(symbol=symbol, h4_history=h4_history, h4_row=h4_row, df_15m=df_15m, atr=atr)
    if mode_a:
        return mode_a
    return check_mode_b(symbol=symbol, h4_history=h4_history, h4_row=h4_row, df_15m=df_15m, oi_aligned=oi_aligned, platform=platform, atr=atr)


# =====================================================================
# v3.1 ScoreBreakdown Engine — Capital Flow Validation Layer
# =====================================================================

@dataclass(frozen=True)
class EntryZone:
    low: float
    high: float
    priority_source: str

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True)
class OIDiagnostics:
    change_pct_short: float    # 3-bar OI change
    change_pct_mid: float      # 12-bar OI change
    change_score: float        # 0-10 composite
    slope_short: float         # 5-bar slope
    slope_mid: float           # 12-bar slope
    slope_score: float         # 0-10 composite
    breakout_strength: float   # continuous, 0.00+
    breakout_score: float      # 0-10
    expansion_ratio: float     # current / average
    expansion_score: float     # 0-10 continuous


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    grade: str
    oi_score: float            # 0-100, 35% weight
    structure_score: float     # 0-100, 25% weight
    volume_score: float        # 0-100, 20% weight
    funding_score: float       # 0-100, 5% weight
    longshort_score: float     # 0-100, 5% weight
    liquidity_score: float     # 0-100, 10% weight
    oi_diag: OIDiagnostics | None = None
    entry_zone: EntryZone | None = None
    is_lottery_coin: bool = False
    is_short_covering_rally: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _oi_change_pct_v3(oi_aligned, h4_row, lookback: int) -> float:
    """OI change % over `lookback` 4H bars."""
    if oi_aligned is None or oi_aligned.empty:
        return 0.0
    eligible = oi_aligned[oi_aligned.index <= h4_row["open_time"]]
    if len(eligible) < lookback + 1:
        return 0.0
    current = float(eligible["sumOpenInterestValue"].iloc[-1])
    prior = float(eligible["sumOpenInterestValue"].iloc[-(lookback + 1)])
    if prior <= 0:
        return 0.0
    return ((current - prior) / prior) * 100


def _expansion_continuous_score(ratio: float) -> float:
    """Continuous OI expansion score 0-10.
    1.00x → 0, 1.10x → 3, 1.20x → 6, 1.30x → 8, 1.50x → 10
    """
    if ratio <= 1.0:
        return 0.0
    if ratio >= 1.50:
        return 10.0
    # Piecewise linear
    if ratio <= 1.10:
        return (ratio - 1.0) / 0.10 * 3.0
    if ratio <= 1.20:
        return 3.0 + (ratio - 1.10) / 0.10 * 3.0
    if ratio <= 1.30:
        return 6.0 + (ratio - 1.20) / 0.10 * 2.0
    # 1.30 ~ 1.50
    return 8.0 + (ratio - 1.30) / 0.20 * 2.0


def _calc_oi_score_v31(oi_aligned, h4_row) -> tuple[float, OIDiagnostics]:
    """V3.1 OI scoring: short+mid change, short+mid slope, continuous breakout, continuous expansion.
    Returns (oi_score_0to1, diagnostics).
    """
    oi_series = oi_aligned["sumOpenInterestValue"] if oi_aligned is not None and not oi_aligned.empty else pd.Series(dtype=float)

    # --- Part 1: OI Change (short + mid) ---
    change_short = _oi_change_pct_v3(oi_aligned, h4_row, V3_OI_CHANGE_SHORT)
    change_mid = _oi_change_pct_v3(oi_aligned, h4_row, V3_OI_CHANGE_MID)

    change_short_decimal = change_short / 100.0
    change_mid_decimal = change_mid / 100.0

    # Score each: -5% → 0, 0% → 5, +5% → 10
    short_score = np.clip((change_short_decimal + 0.05) / 0.01, 0, 10)
    mid_score = np.clip((change_mid_decimal + 0.05) / 0.01, 0, 10)
    change_composite = short_score * V3_OI_CHANGE_SHORT_WEIGHT + mid_score * V3_OI_CHANGE_MID_WEIGHT

    # --- Part 4: OI Slope (short + mid) ---
    slope_short_val = calc_oi_slope(oi_series, window=V3_OI_SLOPE_SHORT)
    slope_mid_val = calc_oi_slope(oi_series, window=V3_OI_SLOPE_MID)

    # Normalize: slope / positive_threshold, capped 0-10
    def _slope_to_score(slope: float) -> float:
        if V3_OI_SLOPE_POSITIVE_THRESHOLD <= 0:
            return 5.0
        return np.clip(slope / V3_OI_SLOPE_POSITIVE_THRESHOLD * 5, 0, 10)

    slope_short_score = _slope_to_score(slope_short_val)
    slope_mid_score = _slope_to_score(slope_mid_val)
    slope_composite = (
        slope_short_score * V3_OI_SLOPE_SHORT_WEIGHT +
        slope_mid_score * V3_OI_SLOPE_MID_WEIGHT
    )

    # --- Part 2: OI Breakout Strength (continuous) ---
    brk = calc_oi_breakout_strength(oi_series, lookback=V3_OI_BREAKOUT_LOOKBACK)
    strength = brk.get("strength", 0.0)
    # 0% breakout → 0, 1% → 3, 5% → 7, 10%+ → 10
    breakout_score_val = np.clip(strength * 100, 0, 10)

    # --- Part 3: OI Expansion (continuous) ---
    exp = calc_oi_relative_expansion(oi_series, window=V3_OI_EXPANSION_WINDOW, threshold=0)
    expansion_ratio = exp.get("ratio", 1.0)
    expansion_score_val = _expansion_continuous_score(expansion_ratio)

    # --- Composite OI Score (0-10) ---
    oi_composite = (
        change_composite * 0.25 +
        slope_composite * 0.25 +
        breakout_score_val * 0.30 +
        expansion_score_val * 0.20
    )

    diag = OIDiagnostics(
        change_pct_short=change_short,
        change_pct_mid=change_mid,
        change_score=round(change_composite, 2),
        slope_short=slope_short_val,
        slope_mid=slope_mid_val,
        slope_score=round(slope_composite, 2),
        breakout_strength=strength,
        breakout_score=round(breakout_score_val, 2),
        expansion_ratio=expansion_ratio,
        expansion_score=round(expansion_score_val, 2),
    )

    return float(np.clip(oi_composite / 10, 0, 1)), diag


def _calc_liquidity_setup_score(
    h4_history: pd.DataFrame,
    h4_row: pd.Series,
    df_15m: pd.DataFrame,
) -> tuple[float, dict]:
    """Mode A liquidity sweep score (0-1)."""
    swing_low = calc_prior_swing_low(h4_history)
    if swing_low is None:
        return 0.0, {"reason": "no_swing_low"}
    window = df_15m[
        (df_15m["open_time"] >= h4_row["open_time"])
        & (df_15m["open_time"] < h4_row["close_time"])
    ].copy()
    if window.empty:
        return 0.0, {"reason": "no_intrabar_15m"}
    sweep_mask = (
        (window["low"] < swing_low)
        & (((swing_low - window["low"]) / swing_low) <= V3_MODE_A_SWEEP_THRESHOLD)
    )
    sweep_rows = window[sweep_mask]
    if sweep_rows.empty:
        wider = window["low"] < swing_low
        if not wider.any():
            return 0.0, {"reason": "no_sweep"}
        return 0.2, {"reason": "sweep_too_deep"}
    first_sweep = sweep_rows.iloc[0]
    reclaim_candidates = window[window["open_time"] >= first_sweep["open_time"]]
    score = 0.0
    for _, row in reclaim_candidates.iterrows():
        vwap = float(row.get("vwap_rolling_24h", np.nan))
        if np.isnan(vwap) or vwap <= swing_low:
            continue
        recovery_ratio = (float(row["high"]) - swing_low) / (vwap - swing_low)
        wick_ratio = (float(row["close"]) - float(row["low"])) / max(float(row["high"]) - float(row["low"]), 0.001)
        taker_buy = float(row.get("taker_buy_ratio", 0))
        if recovery_ratio >= V3_MODE_A_VWAP_RECOVERY_RATIO:
            score += 0.4
        if wick_ratio > V3_MODE_A_WICK_RATIO:
            score += 0.3
        if taker_buy > V3_MODE_A_TAKER_RATIO:
            score += 0.3
        if score >= 0.7:
            break
    return min(score, 1.0), {"reason": "ok"}


def _check_lottery_coin(h4_history: pd.DataFrame, h4_row: pd.Series) -> bool:
    if len(h4_history) < 50:
        return False
    high_50 = float(h4_history.tail(50)["high"].max())
    current_close = float(h4_row["close"])
    decline_pct = ((high_50 - current_close) / high_50) * 100
    if decline_pct < V3_LOTTERY_MAX_DECLINE_PCT:
        return False
    if len(h4_history) >= 99:
        ma99 = calc_sma(h4_history.tail(99), 99)
        if len(ma99) > 0 and float(ma99.iloc[-1]) < current_close:
            return False
    return True


def _check_short_covering_rally(oi_change_pct: float) -> bool:
    return oi_change_pct <= V3_MODE_A_OI_REJECT_THRESHOLD * 100


def _calc_htf_filter(h4_history: pd.DataFrame) -> bool:
    if len(h4_history) < 99:
        return False
    ma25 = calc_sma(h4_history.tail(100), 25)
    ma99 = calc_sma(h4_history.tail(100), 99)
    if len(ma25) < 1 or len(ma99) < 1:
        return False
    last_price = float(h4_history["close"].iloc[-1])
    return (last_price > float(ma99.iloc[-1])) or (float(ma25.iloc[-1]) > float(ma99.iloc[-1]))


def _structure_to_pct(struct_val: str) -> float:
    return 100.0 if struct_val == "bullish" else 50.0 if struct_val == "neutral" else 0.0


def _calc_structure_score_v31(h4_history: pd.DataFrame) -> float:
    """Multi-timeframe structure: 5/10/20 with weights 40/35/25."""
    multi = calc_multi_structure(h4_history, windows=[5, 10, 20])
    s5 = _structure_to_pct(multi.get("5", {}).get("structure", "neutral"))
    s10 = _structure_to_pct(multi.get("10", {}).get("structure", "neutral"))
    s20 = _structure_to_pct(multi.get("20", {}).get("structure", "neutral"))
    weighted = (
        s5 * V3_STRUCTURE_WINDOW_5_WEIGHT +
        s10 * V3_STRUCTURE_WINDOW_10_WEIGHT +
        s20 * V3_STRUCTURE_WINDOW_20_WEIGHT
    )
    return weighted / 100.0  # normalize to 0-1


def _compute_entry_zone(h4_row: pd.Series, h4_history: pd.DataFrame, platform) -> EntryZone | None:
    for source in V3_ENTRY_PRIORITY:
        if source == "breakout_platform" and platform.found and platform.breakout_price:
            base = platform.breakout_price
        elif source == "ma25":
            if len(h4_history) >= 25:
                ma25 = calc_sma(h4_history.tail(30), 25)
                if len(ma25) > 0:
                    base = float(ma25.iloc[-1])
                else:
                    continue
            else:
                continue
        elif source == "vwap":
            base = float(h4_row["close"])
        else:
            continue
        spread = base * V3_ENTRY_ZONE_SPREAD
        return EntryZone(low=base - spread, high=base + spread, priority_source=source)
    return None


def score_symbol_v3(
    symbol: str,
    h4_history: pd.DataFrame,
    h4_row: pd.Series,
    df_15m: pd.DataFrame,
    oi_aligned: pd.DataFrame,
    atr: float,
) -> ScoreBreakdown | None:
    """V3.1 scoring: OI > Structure > Volume > Liquidity > Funding/LS.
    Returns ScoreBreakdown or None (< 40).
    """
    if len(h4_history) < 50:
        return None

    platform = calc_breakout_platform(h4_history)

    # ---- Part 1-4: V3.1 OI Scoring (35% weight) ----
    oi_score_01, oi_diag = _calc_oi_score_v31(oi_aligned, h4_row)

    # ---- Part 5: V3.1 Structure Scoring (25% weight) ----
    struct_score_01 = _calc_structure_score_v31(h4_history)
    struct_raw = calc_market_structure(h4_history, lookback=20)

    # ---- Checks ----
    is_lottery = _check_lottery_coin(h4_history, h4_row)
    is_sc_rally = _check_short_covering_rally(oi_diag.change_pct_short if oi_diag else 0.0)
    liq_score, liq_diag = _calc_liquidity_setup_score(h4_history, h4_row, df_15m)

    # ---- Volume ----
    vol_ratio = float(h4_row.get("volume_ratio", 0))
    vol_score_01 = min(max(vol_ratio / V3_VOLUME_RATIO_MAX, 0), 1.0)

    # ---- Funding & L/S (neutral defaults) ----
    funding_score_01 = 0.5
    longshort_score_01 = 0.5

    # ---- Composite raw score (before penalties) ----
    raw_score = (
        oi_score_01 * V3_WEIGHT_OI * 100 +
        struct_score_01 * V3_WEIGHT_STRUCTURE * 100 +
        vol_score_01 * V3_WEIGHT_VOLUME * 100 +
        funding_score_01 * V3_WEIGHT_FUNDING * 100 +
        longshort_score_01 * V3_WEIGHT_LONGSHORT * 100 +
        liq_score * V3_WEIGHT_LIQUIDITY * 100
    )

    # ---- HTF Cap ----
    htf_pass = _calc_htf_filter(h4_history)
    if not htf_pass:
        raw_score = min(raw_score, V3_HTF_MIN_SCORE)

    # ---- Part 8: Penalties (not filters) ----
    if is_lottery:
        raw_score += V3_LOTTERY_PENALTY   # -20
    if is_sc_rally:
        raw_score += V3_SC_RALLY_PENALTY  # -15

    raw_score = max(raw_score, 0.0)

    # ---- Grade ----
    if raw_score >= V3_GRADE_S_THRESHOLD:
        grade = "S"
    elif raw_score >= V3_GRADE_A_THRESHOLD:
        grade = "A"
    elif raw_score >= V3_GRADE_B_THRESHOLD:
        grade = "B"
    elif raw_score >= V3_GRADE_C_THRESHOLD:
        grade = "C"
    else:
        return None

    # ---- Entry Zone ----
    entry_zone = _compute_entry_zone(h4_row, h4_history, platform)

    return ScoreBreakdown(
        total=round(raw_score, 2), grade=grade,
        oi_score=round(oi_score_01 * 100, 2),
        structure_score=round(struct_score_01 * 100, 2),
        volume_score=round(vol_score_01 * 100, 2),
        funding_score=round(funding_score_01 * 100, 2),
        longshort_score=round(longshort_score_01 * 100, 2),
        liquidity_score=round(liq_score * 100, 2),
        oi_diag=oi_diag, entry_zone=entry_zone,
        is_lottery_coin=is_lottery, is_short_covering_rally=is_sc_rally,
        diagnostics={
            "htf_pass": htf_pass,
            "structure_5_10_20": calc_multi_structure(h4_history, windows=[5, 10, 20]),
            "structure_20": struct_raw,
            "liq_diag": liq_diag,
            "platform_found": platform.found,
        },
    )
