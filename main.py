from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd

from config import MODE_A_SWEEP_THRESHOLD
from config import MODE_A_TAKER_RATIO
from config import MODE_A_WICK_RATIO
from config import MODE_B_OI_CHANGE_THRESHOLD
from config import MODE_B_OI_EXCEPTION_ENABLED
from config import MODE_B_OI_EXCEPTION_THRESHOLD
from data_fetcher import DataFetcher
from indicators import calc_atr
from indicators import calc_breakout_platform
from indicators import calc_cvd_slope_zscore
from indicators import calc_volume_ratio
from output_formatter import MarketSnapshot
from output_formatter import Recommendation
from output_formatter import RiskPlan
from output_formatter import V3Recommendation
from output_formatter import format_telegram_top10
from output_formatter import format_v3_top10_telegram
from signal_generator import _mode_b_oi_exception
from signal_generator import _vwap_reclaimed
from signal_generator import generate_entry_signal
from signal_generator import get_intrabar_window
from signal_generator import calc_prior_swing_low
from signal_generator import score_symbol_v3
from telegram_bot import send_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Crypto Intelligence scan (v2.5 legacy or v3 scoring)."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-volume", type=float, default=20_000_000)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", choices=["v2", "v3"], default="v2",
                        help="Scan mode: v2 (legacy) or v3 (scoring engine)")
    return parser.parse_args()


def get_candidate_symbols(
    fetcher: DataFetcher,
    limit: int,
    min_volume: float,
    symbol: str | None = None,
) -> list[dict]:
    if symbol:
        return [
            {
                "symbol": symbol.upper(),
                "quoteVolume": str(min_volume),
                "priceChangePercent": "0",
            }
        ]

    tickers = fetcher.scanner.get_24h_tickers()
    candidates = []

    for ticker in tickers:
        ticker_symbol = ticker["symbol"]

        if not ticker_symbol.endswith("USDT"):
            continue

        if ticker_symbol in fetcher.config.stable_symbols:
            continue

        quote_volume = float(ticker["quoteVolume"])

        if quote_volume < min_volume:
            continue

        candidates.append(ticker)

    return sorted(
        candidates,
        key=lambda item: float(item["quoteVolume"]),
        reverse=True,
    )[:limit]


def _oi_change_pct(oi_aligned, h4_row) -> float:
    if oi_aligned.empty:
        return 0.0

    eligible = oi_aligned[oi_aligned.index <= h4_row["open_time"]]

    if len(eligible) < 3:
        return 0.0

    current_oi = float(eligible["sumOpenInterestValue"].iloc[-1])
    prior_oi = float(eligible["sumOpenInterestValue"].iloc[-3])

    if prior_oi <= 0:
        return 0.0

    return ((current_oi - prior_oi) / prior_oi) * 100


def _latest_cvd_zscore(df_15m, h4_row) -> float:
    eligible = df_15m[df_15m["open_time"] < h4_row["close_time"]].copy()

    if len(eligible) < 40:
        return 0.0

    cvd = calc_cvd_slope_zscore(eligible)
    value = float(cvd["cvd_slope_zscore"].iloc[-1])

    return value if np.isfinite(value) else 0.0


def _last_closed_h4_index(h4, now: pd.Timestamp | None = None) -> int | None:
    now = now or pd.Timestamp.now(tz="UTC")
    # 10-minute buffer ensures the candle is fully closed (not forming)
    closed = h4[h4["close_time"] <= now - pd.Timedelta(minutes=10)]
    if closed.empty:
        return None
    return int(closed.index[-1])


def _closed_15m_context(df_15m, h4_row):
    return df_15m[df_15m["close_time"] <= h4_row["close_time"]].copy()


def _latest_oi_value_for_h4(oi_aligned, h4_row) -> float:
    if oi_aligned.empty:
        return 0.0

    eligible = oi_aligned[oi_aligned.index <= h4_row["open_time"]]

    if eligible.empty:
        return 0.0

    return float(eligible["sumOpenInterestValue"].iloc[-1])


def _recent_taker_buy_ratio(df_15m, h4_row) -> float:
    eligible = df_15m[df_15m["open_time"] < h4_row["close_time"]].tail(8)

    if eligible.empty:
        return 0.0

    value = float(eligible["taker_buy_ratio"].mean())
    return value if np.isfinite(value) else 0.0


def _diagnose_symbol(fetcher: DataFetcher, ticker: dict) -> dict[str, Counter]:
    """Wrapper that fetches data then delegates to _diagnose_symbol_data. Kept for backward compat."""
    symbol = ticker["symbol"]
    h4 = fetcher.fetch_h4(symbol)
    m15 = fetcher.fetch_intraday_15m(symbol)
    oi = fetcher.fetch_oi(symbol, period="4h")
    oi_aligned = fetcher.align_oi_to_klines(oi, h4)
    return _diagnose_with_data(ticker, h4, m15, oi_aligned)


def _diagnose_with_data(ticker: dict, h4: pd.DataFrame, m15: pd.DataFrame, oi_aligned: pd.DataFrame) -> dict[str, Counter]:
    """Diagnose using pre-fetched data (no additional API calls)."""
    counters = {
        "mode_a": Counter(),
        "mode_b": Counter(),
    }

    if len(h4) < 51:
        counters["mode_a"]["not_enough_h4"] += 1
        counters["mode_b"]["not_enough_h4"] += 1
        return counters

    h4["atr14"] = calc_atr(h4)
    h4["volume_ratio"] = calc_volume_ratio(h4)
    closed_index = _last_closed_h4_index(h4)

    if closed_index is None or closed_index < 50:
        counters["mode_a"]["no_closed_h4"] += 1
        counters["mode_b"]["no_closed_h4"] += 1
        return counters

    h4_row = h4.iloc[closed_index]
    h4_history = h4.iloc[:closed_index]
    m15 = _closed_15m_context(m15, h4_row)
    atr = float(h4_row["atr14"])

    if not np.isfinite(atr):
        counters["mode_a"]["atr_nan"] += 1
        counters["mode_b"]["atr_nan"] += 1
        return counters

    # Mode A diagnostics.
    swing_low = calc_prior_swing_low(h4_history)
    window = get_intrabar_window(m15, h4_row)

    if swing_low is None or window.empty:
        counters["mode_a"]["no_swing_or_15m"] += 1
    else:
        sweep_mask = (
            (window["low"] < swing_low)
            & (
                ((swing_low - window["low"]) / swing_low)
                <= MODE_A_SWEEP_THRESHOLD
            )
        )
        sweep_rows = window[sweep_mask]

        if sweep_rows.empty:
            counters["mode_a"]["fail_sweep"] += 1
        else:
            counters["mode_a"]["sweep_candidate"] += 1
            first_sweep = sweep_rows.iloc[0]
            reclaim_candidates = window[window["open_time"] >= first_sweep["open_time"]]
            reclaim_rows = reclaim_candidates[
                reclaim_candidates.apply(
                    lambda row: _vwap_reclaimed(row, swing_low),
                    axis=1,
                )
            ]

            if reclaim_rows.empty:
                counters["mode_a"]["fail_vwap_reclaim"] += 1
            else:
                counters["mode_a"]["vwap_reclaim_candidate"] += 1
                wick_ratio = (
                    (reclaim_rows["close"] - reclaim_rows["low"])
                    / (reclaim_rows["high"] - reclaim_rows["low"]).replace(0, np.nan)
                )
                wick_rows = reclaim_rows[wick_ratio > MODE_A_WICK_RATIO]

                if wick_rows.empty:
                    counters["mode_a"]["fail_wick"] += 1
                else:
                    counters["mode_a"]["wick_candidate"] += 1
                    taker_rows = wick_rows[
                        wick_rows["taker_buy_ratio"] > MODE_A_TAKER_RATIO
                    ]

                    if taker_rows.empty:
                        counters["mode_a"]["fail_taker"] += 1
                    else:
                        counters["mode_a"]["final_candidate"] += 1

    # Mode B diagnostics.
    platform = calc_breakout_platform(h4_history)

    if not platform.found or platform.breakout_price is None:
        counters["mode_b"]["fail_platform"] += 1
        return counters

    if float(h4_row["close"]) <= platform.breakout_price:
        counters["mode_b"]["fail_breakout_close"] += 1
        return counters

    counters["mode_b"]["breakout_candidate"] += 1

    oi_change = _oi_change_pct(oi_aligned, h4_row)
    oi_change_decimal = oi_change / 100

    if oi_change_decimal >= MODE_B_OI_CHANGE_THRESHOLD:
        counters["mode_b"]["oi_candidate"] += 1
        counters["mode_b"]["final_candidate"] += 1
        counters["mode_b"]["final_ab"] += 1
        return counters

    if (
        MODE_B_OI_EXCEPTION_ENABLED
        and oi_change_decimal < 0
        and oi_change_decimal >= MODE_B_OI_EXCEPTION_THRESHOLD
        and _mode_b_oi_exception(h4_row, m15, oi_change_decimal)
    ):
        counters["mode_b"]["oi_exception"] += 1
        counters["mode_b"]["final_candidate"] += 1
        counters["mode_b"]["final_c"] += 1
        return counters

    counters["mode_b"]["fail_oi"] += 1
    return counters


def _merge_diagnostics(target: dict[str, Counter], source: dict[str, Counter]) -> None:
    target["mode_a"].update(source["mode_a"])
    target["mode_b"].update(source["mode_b"])


def _format_diagnostics(counters: dict[str, Counter]) -> str:
    mode_a = counters["mode_a"]
    mode_b = counters["mode_b"]
    lines = [
        "Diagnostics",
        (
            "Mode A: "
            f"sweep={mode_a['sweep_candidate']} "
            f"vwap={mode_a['vwap_reclaim_candidate']} "
            f"wick={mode_a['wick_candidate']} "
            f"final={mode_a['final_candidate']}"
        ),
        (
            "Mode A rejects: "
            f"sweep={mode_a['fail_sweep']} "
            f"vwap={mode_a['fail_vwap_reclaim']} "
            f"wick={mode_a['fail_wick']} "
            f"taker={mode_a['fail_taker']}"
        ),
        (
            "Mode B: "
            f"breakout={mode_b['breakout_candidate']} "
            f"oi={mode_b['oi_candidate']} "
            f"exception={mode_b['oi_exception']} "
            f"final={mode_b['final_candidate']} "
            f"(A/B={mode_b['final_ab']} C={mode_b['final_c']})"
        ),
        (
            "Mode B rejects: "
            f"platform={mode_b['fail_platform']} "
            f"breakout_close={mode_b['fail_breakout_close']} "
            f"oi={mode_b['fail_oi']} "
            f"cvd={mode_b['fail_cvd']} "
            f"taker={mode_b['fail_taker']}"
        ),
    ]
    return "\n".join(lines)


def _build_risk_plan(entry: float, atr: float, h4_history) -> RiskPlan:
    risk = 1.8 * atr
    stop = entry - risk
    tp1 = entry + (1.5 * atr)
    structural_target = float(h4_history.tail(30)["high"].max())
    theoretical_target = min(entry + (3.5 * risk), structural_target)

    if theoretical_target <= entry:
        theoretical_target = entry + (3.5 * risk)

    theoretical_rr = (theoretical_target - entry) / risk if risk > 0 else 0.0

    return RiskPlan(
        entry=entry,
        stop=stop,
        tp1=tp1,
        tp2=theoretical_target,
        trailing_atr=1.0,
        theoretical_rr=theoretical_rr,
    )


def _score_recommendation(
    mode: str,
    oi_change_pct: float,
    volume_ratio: float,
    theoretical_rr: float,
    extension_pct: float | None,
) -> float:
    mode_score = 25 if mode == "MODE_A_SWEEP_RECLAIM" else 15
    oi_score = min(max(oi_change_pct, 0) / 10, 5) * 3
    volume_score = min(max(volume_ratio, 0), 5) * 4
    rr_score = min(max(theoretical_rr, 0), 3.5) * 8
    extension_penalty = 15 if extension_pct is not None and extension_pct > 10 else 0

    return round(mode_score + oi_score + volume_score + rr_score - extension_penalty, 2)


def _market_type(
    score: float,
    theoretical_rr: float,
    ticker: dict,
    latest_oi_value: float,
) -> str:
    price_change_pct = float(ticker.get("priceChangePercent", 0))

    if price_change_pct > 30 and latest_oi_value < 5_000_000:
        return "C"

    if score >= 70 and theoretical_rr >= 2.5:
        return "A"

    return "B"


def scan_symbol(fetcher: DataFetcher, ticker: dict) -> Recommendation | None:
    """Wrapper that fetches data then delegates to _scan_symbol_data. Kept for backward compat."""
    symbol = ticker["symbol"]
    h4 = fetcher.fetch_h4(symbol)
    m15 = fetcher.fetch_intraday_15m(symbol)
    oi = fetcher.fetch_oi(symbol, period="4h")
    oi_aligned = fetcher.align_oi_to_klines(oi, h4)
    return _scan_symbol_with_data(ticker, h4, m15, oi_aligned)


def _scan_symbol_with_data(ticker: dict, h4: pd.DataFrame, m15: pd.DataFrame, oi_aligned: pd.DataFrame) -> Recommendation | None:
    """Scan using pre-fetched data (no additional API calls)."""
    symbol = ticker["symbol"]

    if len(h4) < 51:
        return None

    h4["atr14"] = calc_atr(h4)
    h4["volume_ratio"] = calc_volume_ratio(h4)

    closed_index = _last_closed_h4_index(h4)

    if closed_index is None or closed_index < 50:
        return None

    h4_row = h4.iloc[closed_index]
    h4_history = h4.iloc[:closed_index]
    m15 = _closed_15m_context(m15, h4_row)

    if m15.empty:
        return None

    atr = float(h4_row["atr14"])

    if not np.isfinite(atr):
        return None

    platform = calc_breakout_platform(h4_history)
    signal = generate_entry_signal(
        symbol=symbol,
        h4_history=h4_history,
        h4_row=h4_row,
        df_15m=m15,
        oi_aligned=oi_aligned,
        platform=platform,
        atr=atr,
    )

    if signal is None:
        return None

    oi_change = _oi_change_pct(oi_aligned, h4_row)
    volume_ratio = float(h4_row["volume_ratio"])
    volume_ratio = volume_ratio if np.isfinite(volume_ratio) else 0.0
    taker_buy_ratio = _recent_taker_buy_ratio(m15, h4_row)
    cvd_zscore = _latest_cvd_zscore(m15, h4_row)
    risk = _build_risk_plan(signal.entry_price, atr, h4_history)
    latest_oi_value = _latest_oi_value_for_h4(oi_aligned, h4_row)
    score = _score_recommendation(
        mode=signal.mode,
        oi_change_pct=oi_change,
        volume_ratio=volume_ratio,
        theoretical_rr=risk.theoretical_rr,
        extension_pct=platform.extension_pct,
    )
    market_type = _market_type(score, risk.theoretical_rr, ticker, latest_oi_value)

    if signal.diagnostics.get("signal_grade") == "C":
        market_type = "C"
    elif signal.diagnostics.get("oi_exception"):
        market_type = "C"

    return Recommendation(
        rank=0,
        symbol=symbol,
        mode=signal.mode,
        score=score,
        market_type=market_type,
        risk=risk,
        market=MarketSnapshot(
            oi_change_pct=oi_change,
            volume_ratio=volume_ratio,
            taker_buy_ratio=taker_buy_ratio,
            cvd_zscore=cvd_zscore,
            extension_pct=platform.extension_pct,
        ),
        comment=signal.reason,
    )


def assign_ranks(recommendations: list[Recommendation]) -> list[Recommendation]:
    ranked = sorted(
        recommendations,
        key=lambda item: item.score,
        reverse=True,
    )
    return [
        replace(item, rank=index + 1)
        for index, item in enumerate(ranked)
    ]


def _struct_pct_from_multi(multi: dict, key: str) -> float:
    """Extract structure % (bullish=100, neutral=50, bearish=0) from multi-structure dict."""
    val = multi.get(key, {}).get("structure", "neutral")
    return 100.0 if val == "bullish" else 50.0 if val == "neutral" else 0.0


def scan_symbol_v3(
    ticker: dict,
    h4: pd.DataFrame,
    m15: pd.DataFrame,
    oi_aligned: pd.DataFrame,
) -> V3Recommendation | None:
    """v3 scan using pre-fetched data (no additional API calls)."""
    symbol = ticker["symbol"]
    h4["atr14"] = calc_atr(h4)
    h4["volume_ratio"] = calc_volume_ratio(h4)
    closed_index = _last_closed_h4_index(h4)

    if closed_index is None or closed_index < 50:
        return None

    h4_row = h4.iloc[closed_index]
    h4_history = h4.iloc[:closed_index]
    m15_cut = _closed_15m_context(m15, h4_row)

    if m15_cut.empty:
        return None

    atr = float(h4_row["atr14"])
    if not np.isfinite(atr):
        return None

    score_result = score_symbol_v3(
        symbol=symbol,
        h4_history=h4_history,
        h4_row=h4_row,
        df_15m=m15_cut,
        oi_aligned=oi_aligned,
        atr=atr,
    )

    if score_result is None:
        return None

    entry_zone = score_result.entry_zone
    oi_diag = score_result.oi_diag
    entry_price = entry_zone.midpoint if entry_zone else 0.0

    # Use _build_risk_plan logic for consistent TP2
    risk_plan = _build_risk_plan(entry_price, atr, h4_history)
    stop = risk_plan.stop
    tp1 = risk_plan.tp1
    tp2 = risk_plan.tp2

    # Structure 5/10/20 breakdown
    multi = score_result.diagnostics.get("structure_5_10_20", {})
    s5 = _struct_pct_from_multi(multi, "5")
    s10 = _struct_pct_from_multi(multi, "10")
    s20 = _struct_pct_from_multi(multi, "20")

    return V3Recommendation(
        rank=0,
        symbol=ticker["symbol"],
        grade=score_result.grade,
        score=score_result.total,
        oi_score=score_result.oi_score,
        structure_score=score_result.structure_score,
        volume_score=score_result.volume_score,
        funding_score=score_result.funding_score,
        longshort_score=score_result.longshort_score,
        liquidity_score=score_result.liquidity_score,
        struct_5=s5,
        struct_10=s10,
        struct_20=s20,
        oi_change_short=oi_diag.change_pct_short if oi_diag else 0.0,
        oi_change_mid=oi_diag.change_pct_mid if oi_diag else 0.0,
        oi_breakout_strength=oi_diag.breakout_strength if oi_diag else 0.0,
        oi_expansion_ratio=oi_diag.expansion_ratio if oi_diag else 0.0,
        oi_change_pct=oi_diag.change_pct_short if oi_diag else 0.0,
        oi_slope=oi_diag.slope_short if oi_diag else 0.0,
        oi_breakout=oi_diag.breakout_strength > 0 if oi_diag else False,
        oi_expanding=oi_diag.expansion_ratio > 1.0 if oi_diag else False,
        market_structure=score_result.diagnostics.get("structure_20", {}).get("structure", "neutral"),
        entry_zone_low=entry_zone.low if entry_zone else 0.0,
        entry_zone_high=entry_zone.high if entry_zone else 0.0,
        entry_zone_source=entry_zone.priority_source if entry_zone else "none",
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        atr=atr,
        htf_pass=score_result.diagnostics.get("htf_pass", False),
        is_lottery=score_result.is_lottery_coin,
        is_short_covering=score_result.is_short_covering_rally,
        diagnostics=score_result.diagnostics,
    )


def _prepare_data(
    fetcher: DataFetcher,
    ticker: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Fetch and prepare all data for a symbol. Returns (h4, m15, oi_aligned, ok)."""
    symbol = ticker["symbol"]
    try:
        h4 = fetcher.fetch_h4(symbol)
        m15 = fetcher.fetch_intraday_15m(symbol)
        oi = fetcher.fetch_oi(symbol, period="4h")
        oi_aligned = fetcher.align_oi_to_klines(oi, h4)
        if len(h4) < 51:
            return h4, m15, oi_aligned, False
        return h4, m15, oi_aligned, True
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False


def main() -> None:
    args = parse_args()
    fetcher = DataFetcher()
    tickers = get_candidate_symbols(
        fetcher=fetcher,
        limit=args.limit,
        min_volume=args.min_volume,
        symbol=args.symbol,
    )

    recommendations_v3: list[V3Recommendation] = []
    recommendations_v2: list[Recommendation] = []
    diagnostics: dict[str, Counter] = {"mode_a": Counter(), "mode_b": Counter()}

    for ticker in tickers:
        h4, m15, oi_aligned, ok = _prepare_data(fetcher, ticker)
        if not ok or m15.empty or oi_aligned.empty:
            continue

        # Single data fetch → both v3 scoring and v2 diagnostics (no extra API calls)
        try:
            rec_v3 = scan_symbol_v3(ticker, h4, m15, oi_aligned)
            if rec_v3 is not None:
                recommendations_v3.append(rec_v3)

            diag = _diagnose_with_data(ticker, h4, m15, oi_aligned)
            _merge_diagnostics(diagnostics, diag)

            rec_v2 = _scan_symbol_with_data(ticker, h4, m15, oi_aligned)
            if rec_v2 is not None:
                recommendations_v2.append(rec_v2)
        except Exception:
            continue

    # Sort and rank
    ranked_v3 = sorted(recommendations_v3, key=lambda r: r.score, reverse=True)
    ranked_v2 = assign_ranks(recommendations_v2)

    # Build Telegram message: v3 first, then v2
    msg_v3 = format_v3_top10_telegram(ranked_v3)
    msg_v2 = format_telegram_top10(ranked_v2)

    # Split with separator
    full_message = f"{msg_v3}\n\n━━━━━━━━━━━━━━━\n\n{msg_v2}"
    send_message(full_message)

    # No console output — Telegram only
    pass


if __name__ == "__main__":
    main()
