from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from data_fetcher import DataFetcher
from indicators import calc_atr
from indicators import calc_breakout_platform
from indicators import calc_cvd_slope_zscore
from indicators import calc_volume_ratio
from output_formatter import MarketSnapshot
from output_formatter import Recommendation
from output_formatter import RiskPlan
from output_formatter import format_telegram_top10
from signal_generator import generate_entry_signal
from telegram_bot import send_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Project_Whale_Footprint v2.0 Lite scan."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-volume", type=float, default=20_000_000)
    parser.add_argument("--symbol", type=str, default=None)
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
    closed = h4[h4["close_time"] <= now]

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
    symbol = ticker["symbol"]
    h4 = fetcher.fetch_h4(symbol)
    m15 = fetcher.fetch_intraday_15m(symbol)
    oi = fetcher.fetch_oi(symbol, period="4h")
    oi_aligned = fetcher.align_oi_to_klines(oi, h4)

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


def main() -> None:
    args = parse_args()
    fetcher = DataFetcher()
    tickers = get_candidate_symbols(
        fetcher=fetcher,
        limit=args.limit,
        min_volume=args.min_volume,
        symbol=args.symbol,
    )
    recommendations: list[Recommendation] = []

    for ticker in tickers:
        try:
            recommendation = scan_symbol(fetcher, ticker)
        except Exception:
            continue

        if recommendation is None:
            continue

        recommendations.append(recommendation)

    ranked = assign_ranks(recommendations)
    send_message(format_telegram_top10(ranked))


if __name__ == "__main__":
    main()
