from __future__ import annotations

from dataclasses import dataclass, field, replace
from html import escape
from typing import Any


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
class V3Recommendation:
    """v3 recommendation with scoring breakdown."""
    rank: int
    symbol: str
    grade: str  # S/A/B/C
    score: float
    oi_score: float
    structure_score: float
    volume_score: float
    funding_score: float
    longshort_score: float
    liquidity_score: float
    # Structure 5/10/20 breakdown
    struct_5: float
    struct_10: float
    struct_20: float
    # OI diagnostics (V3.1)
    oi_change_short: float
    oi_change_mid: float
    oi_breakout_strength: float
    oi_expansion_ratio: float
    oi_change_pct: float
    oi_slope: float
    oi_breakout: bool
    oi_expanding: bool
    market_structure: str
    entry_zone_low: float
    entry_zone_high: float
    entry_zone_source: str
    stop: float
    tp1: float
    tp2: float
    atr: float
    htf_pass: bool
    is_lottery: bool
    is_short_covering: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


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


# =====================================================================
# v3 Formatting
# =====================================================================

def _format_grade(grade: str) -> str:
    emojis = {"S": "🟣 S", "A": "🟢 A", "B": "🟡 B", "C": "⚪ C"}
    return emojis.get(grade, grade)


def _format_entry_zone(low: float, high: float, source: str) -> str:
    return f"{source}: {_format_price(low)}–{_format_price(high)}"


def _format_check(val: bool) -> str:
    return "✅" if val else "❌"


def _struct_line(item: V3Recommendation) -> str:
    """Structure breakdown: 5/10/20 scores."""
    return f"S5:{item.struct_5:.0f} S10:{item.struct_10:.0f} S20:{item.struct_20:.0f}"

def format_v3_console(item: V3Recommendation) -> str:
    return "\n".join([
        f"#{item.rank} {item.symbol} | Grade {_format_grade(item.grade)}",
        f"Score: {item.score:.1f}/100",
        f"OI: {item.oi_score:.1f} | Structure: {item.structure_score:.1f} | Vol: {item.volume_score:.1f}",
        f"  Struct breakdown: {_struct_line(item)}",
        f"Funding: {item.funding_score:.1f} | L/S: {item.longshort_score:.1f} | Liq: {item.liquidity_score:.1f}",
        f"ENTRY ZONE: {_format_entry_zone(item.entry_zone_low, item.entry_zone_high, item.entry_zone_source)}",
        f"SL: {_format_price(item.stop)} | TP1: {_format_price(item.tp1)} | TP2: {_format_price(item.tp2)}",
        f"OI Diagnostics: short={item.oi_change_short:.2f}% mid={item.oi_change_mid:.2f}% break_str={item.oi_breakout_strength:.4f} exp_ratio={item.oi_expansion_ratio:.4f}",
        f"Structure: {item.market_structure} | HTF: {_format_check(item.htf_pass)} | Lottery: {_format_check(item.is_lottery)} | SC Rally: {_format_check(item.is_short_covering)}",
        f"ATR: {_format_price(item.atr)}",
    ])


def format_v3_telegram(item: V3Recommendation) -> str:
    return "\n".join([
        f"<b>#{item.rank} {_html(item.symbol)}</b> | Grade {_html(_format_grade(item.grade))}",
        f"Score: <b>{item.score:.1f}/100</b>",
        f"OI: {item.oi_score:.1f} | Structure: {item.structure_score:.1f} | Vol: {item.volume_score:.1f}",
        f"Struct: {_struct_line(item)}",
        f"Funding: {item.funding_score:.1f} | L/S: {item.longshort_score:.1f} | Liq: {item.liquidity_score:.1f}",
        f"<b>Entry Zone</b>",
        f"<code>{_html(_format_entry_zone(item.entry_zone_low, item.entry_zone_high, item.entry_zone_source))}</code>",
        f"SL: <code>{_html(_format_price(item.stop))}</code> | TP1: <code>{_html(_format_price(item.tp1))}</code> | TP2: <code>{_html(_format_price(item.tp2))}</code>",
        f"OI short: <b>{item.oi_change_short:.2f}%</b> mid: <b>{item.oi_change_mid:.2f}%</b>",
        f"break_str: <b>{item.oi_breakout_strength:.4f}</b> | exp: <b>{item.oi_expansion_ratio:.4f}</b>",
        f"Structure: <b>{item.market_structure}</b> | HTF: {_format_check(item.htf_pass)}",
    ])


def format_v3_top10_console(recommendations: list[V3Recommendation]) -> str:
    ranked = sorted(recommendations, key=lambda r: r.score, reverse=True)[:10]
    if not ranked:
        return "No v3 recommendations."
    sections = ["Crypto Intelligence v3 — Scoring Results"]
    sections.extend(format_v3_console(r) for r in ranked)
    return "\n\n".join(sections)


def _assign_rank_v3(recommendations: list[V3Recommendation]) -> list[V3Recommendation]:
    ranked = sorted(recommendations, key=lambda r: r.score, reverse=True)
    return [replace(rec, rank=idx + 1) for idx, rec in enumerate(ranked)]


def format_v3_top10_telegram(recommendations: list[V3Recommendation]) -> str:
    ranked = _assign_rank_v3(recommendations)[:10]
    if not ranked:
        return "<b>Crypto Intelligence v3</b>\nNo recommendations."

    grades = {r.grade for r in ranked}
    messages = [
        "<b>Crypto Intelligence v3 — Score Breakdown</b>",
        f"Signals: {len(ranked)} | Grades: {', '.join(sorted(grades, reverse=True))}",
    ]
    messages.extend(
        format_v3_telegram(rec) for rec in ranked
    )
    return "\n\n".join(messages)
