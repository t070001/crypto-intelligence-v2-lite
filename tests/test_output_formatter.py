from output_formatter import (
    MarketSnapshot,
    Recommendation,
    RiskPlan,
    format_console_top10,
    format_telegram_top10,
    rank_recommendations,
)


def _recommendation(
    symbol: str,
    score: float,
    market_type: str = "A",
    rank: int = 1,
    comment: str = "Wait for pullback. No chase.",
) -> Recommendation:
    return Recommendation(
        rank=rank,
        symbol=symbol,
        mode="MODE_A_SWEEP_RECLAIM",
        score=score,
        market_type=market_type,
        risk=RiskPlan(
            entry=100.0,
            stop=92.0,
            tp1=112.0,
            tp2=128.0,
            trailing_atr=1.0,
            theoretical_rr=3.5,
        ),
        market=MarketSnapshot(
            oi_change_pct=12.5,
            volume_ratio=2.1,
            taker_buy_ratio=1.35,
            cvd_zscore=1.4,
            extension_pct=4.2,
        ),
        comment=comment,
    )


def test_rank_recommendations_keeps_c_type_outside_official_limit():
    recommendations = [
        _recommendation("AUSDT", 80.0, "A"),
        _recommendation("BUSDT", 90.0, "B"),
        _recommendation("CUSDT", 99.0, "C"),
    ]

    ranked = rank_recommendations(recommendations, limit=1)

    assert [item.symbol for item in ranked] == ["BUSDT", "CUSDT"]


def test_format_console_top10_includes_key_fields():
    output = format_console_top10([_recommendation("NEARUSDT", 88.0)])

    assert "Project_Whale_Footprint v2.0 Lite" in output
    assert "#1 NEARUSDT" in output
    assert "Entry:" in output
    assert "RR:" in output
    assert "Vol Ratio:" in output


def test_format_telegram_top10_escapes_html_comment():
    output = format_telegram_top10(
        [
            _recommendation(
                "NEARUSDT",
                88.0,
                comment="Sweep < VWAP & reclaim",
            )
        ]
    )

    assert "<b>Project_Whale_Footprint v2.0 Lite</b>" in output
    assert "<b>#1 NEARUSDT</b>" in output
    assert "Sweep &lt; VWAP &amp; reclaim" in output


def test_formatters_handle_empty_recommendations():
    assert format_console_top10([]) == "No v2 Lite recommendations."
    assert "No recommendations" in format_telegram_top10([])
