from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class RiskPlan:
    entry: float
    stop: float
    tp1: float
    tp2: float
    trailing_atr: float
    theoretical_rr: float


@dataclass(frozen=True)
class MarketSnapshot:
    oi_change_pct: float
    volume_ratio: float
    taker_buy_ratio: float
    cvd_zscore: float
    extension_pct: float | None = None


@dataclass(frozen=True)
class Recommendation:
    rank: int
    symbol: str
    mode: str
    score: float
    market_type: str
    risk: RiskPlan
    market: MarketSnapshot
    comment: str


def rank_recommendations(
    recommendations: list[Recommendation],
    limit: int = 10,
) -> list[Recommendation]:
    official = [
        item
        for item in recommendations
        if item.market_type in {"A", "B"}
    ]
    lottery = [
        item
        for item in recommendations
        if item.market_type == "C"
    ]

    ranked = sorted(
        official,
        key=lambda item: item.score,
        reverse=True,
    )[:limit]

    return ranked + sorted(
        lottery,
        key=lambda item: item.score,
        reverse=True,
    )


def _format_price(value: float) -> str:
    return f"{value:,.6g}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value:.2f}%"


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def format_console_recommendation(item: Recommendation) -> str:
    return "\n".join(
        [
            f"#{item.rank} {item.symbol} | {item.mode} | Type {item.market_type}",
            f"Score: {item.score:.1f}",
            (
                f"Entry: {_format_price(item.risk.entry)} | "
                f"SL: {_format_price(item.risk.stop)} | "
                f"TP1: {_format_price(item.risk.tp1)} | "
                f"TP2: {_format_price(item.risk.tp2)}"
            ),
            (
                f"RR: {_format_float(item.risk.theoretical_rr)} | "
                f"Trail: {_format_float(item.risk.trailing_atr)} ATR"
            ),
            (
                f"OI: {_format_pct(item.market.oi_change_pct)} | "
                f"Vol Ratio: {_format_float(item.market.volume_ratio)}x | "
                f"Taker Buy: {_format_float(item.market.taker_buy_ratio)} | "
                f"CVD Z: {_format_float(item.market.cvd_zscore)}"
            ),
            f"Extension: {_format_pct(item.market.extension_pct)}",
            f"Comment: {item.comment}",
        ]
    )


def format_console_top10(recommendations: list[Recommendation]) -> str:
    ranked = rank_recommendations(recommendations)

    if not ranked:
        return "No v2 Lite recommendations."

    sections = ["Project_Whale_Footprint v2.0 Lite", "Top Recommendations"]
    sections.extend(format_console_recommendation(item) for item in ranked)
    return "\n\n".join(sections)


def _html(value: object) -> str:
    return escape(str(value), quote=False)


def format_telegram_recommendation(item: Recommendation) -> str:
    return "\n".join(
        [
            (
                f"<b>#{item.rank} {_html(item.symbol)}</b> | "
                f"{_html(item.mode)} | Type <b>{_html(item.market_type)}</b>"
            ),
            f"Score: <b>{item.score:.1f}</b>",
            (
                f"Entry: <code>{_html(_format_price(item.risk.entry))}</code>\n"
                f"SL: <code>{_html(_format_price(item.risk.stop))}</code>\n"
                f"TP1: <code>{_html(_format_price(item.risk.tp1))}</code>\n"
                f"TP2: <code>{_html(_format_price(item.risk.tp2))}</code>"
            ),
            (
                f"RR: <b>{_format_float(item.risk.theoretical_rr)}</b> | "
                f"Trail: <b>{_format_float(item.risk.trailing_atr)} ATR</b>"
            ),
            (
                f"OI: <b>{_format_pct(item.market.oi_change_pct)}</b> | "
                f"Vol: <b>{_format_float(item.market.volume_ratio)}x</b>"
            ),
            (
                f"Taker Buy: <b>{_format_float(item.market.taker_buy_ratio)}</b> | "
                f"CVD Z: <b>{_format_float(item.market.cvd_zscore)}</b>"
            ),
            f"Extension: <b>{_format_pct(item.market.extension_pct)}</b>",
            f"<b>Comment</b>\n{_html(item.comment)}",
        ]
    )


def format_telegram_top10(recommendations: list[Recommendation]) -> str:
    ranked = rank_recommendations(recommendations)

    if not ranked:
        return "<b>Project_Whale_Footprint v2.0 Lite</b>\nNo recommendations."

    messages = [
        (
            "<b>Project_Whale_Footprint v2.0 Lite</b>\n"
            f"Official A/B: <b>{sum(1 for item in ranked if item.market_type in {'A', 'B'})}</b>\n"
            f"Lottery C: <b>{sum(1 for item in ranked if item.market_type == 'C')}</b>"
        )
    ]
    messages.extend(format_telegram_recommendation(item) for item in ranked)
    return "\n\n".join(messages)
