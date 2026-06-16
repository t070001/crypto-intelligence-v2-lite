import pandas as pd

from main import _build_risk_plan
from main import _closed_15m_context
from main import _last_closed_h4_index
from main import _latest_oi_value_for_h4
from main import _market_type
from main import _score_recommendation
from main import assign_ranks
from main import parse_args
from output_formatter import MarketSnapshot
from output_formatter import Recommendation
from output_formatter import RiskPlan


def test_build_risk_plan_uses_conservative_structural_target():
    h4_history = pd.DataFrame({"high": [110.0, 120.0, 130.0]})

    risk = _build_risk_plan(entry=100.0, atr=10.0, h4_history=h4_history)

    assert risk.stop == 82.0
    assert risk.tp1 == 115.0
    assert risk.tp2 == 130.0
    assert round(risk.theoretical_rr, 2) == 1.67


def test_parse_args_supports_scan_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--limit", "5", "--min-volume", "20000000"],
    )

    args = parse_args()

    assert args.limit == 5
    assert args.min_volume == 20_000_000


def test_score_recommendation_penalizes_extension():
    normal = _score_recommendation(
        mode="MODE_A_SWEEP_RECLAIM",
        oi_change_pct=20.0,
        volume_ratio=2.0,
        theoretical_rr=3.0,
        extension_pct=5.0,
    )
    extended = _score_recommendation(
        mode="MODE_A_SWEEP_RECLAIM",
        oi_change_pct=20.0,
        volume_ratio=2.0,
        theoretical_rr=3.0,
        extension_pct=11.0,
    )

    assert normal - extended == 15


def test_market_type_marks_lottery_coin_as_c():
    market_type = _market_type(
        score=90.0,
        theoretical_rr=3.0,
        ticker={"priceChangePercent": "35"},
        latest_oi_value=4_000_000,
    )

    assert market_type == "C"


def test_last_closed_h4_index_ignores_current_running_candle():
    h4 = pd.DataFrame(
        {
            "close_time": [
                pd.Timestamp("2026-01-01 04:00", tz="UTC"),
                pd.Timestamp("2026-01-01 08:00", tz="UTC"),
                pd.Timestamp("2026-01-01 12:00", tz="UTC"),
            ]
        }
    )

    index = _last_closed_h4_index(
        h4,
        now=pd.Timestamp("2026-01-01 09:00", tz="UTC"),
    )

    assert index == 1


def test_closed_15m_context_trims_after_h4_close():
    h4_row = pd.Series(
        {
            "close_time": pd.Timestamp("2026-01-01 08:00", tz="UTC"),
        }
    )
    df_15m = pd.DataFrame(
        {
            "close_time": [
                pd.Timestamp("2026-01-01 07:45", tz="UTC"),
                pd.Timestamp("2026-01-01 08:00", tz="UTC"),
                pd.Timestamp("2026-01-01 08:15", tz="UTC"),
            ]
        }
    )

    result = _closed_15m_context(df_15m, h4_row)

    assert len(result) == 2
    assert result["close_time"].max() <= h4_row["close_time"]


def test_latest_oi_value_for_h4_uses_aligned_h4_row_not_future_row():
    h4_row = pd.Series(
        {
            "open_time": pd.Timestamp("2026-01-01 08:00", tz="UTC"),
        }
    )
    oi_aligned = pd.DataFrame(
        {
            "sumOpenInterestValue": [100.0, 120.0, 999.0],
        },
        index=[
            pd.Timestamp("2026-01-01 04:00", tz="UTC"),
            pd.Timestamp("2026-01-01 08:00", tz="UTC"),
            pd.Timestamp("2026-01-01 12:00", tz="UTC"),
        ],
    )

    assert _latest_oi_value_for_h4(oi_aligned, h4_row) == 120.0


def test_assign_ranks_sorts_by_score():
    def recommendation(symbol: str, score: float) -> Recommendation:
        return Recommendation(
            rank=0,
            symbol=symbol,
            mode="MODE_A_SWEEP_RECLAIM",
            score=score,
            market_type="A",
            risk=RiskPlan(100.0, 90.0, 115.0, 130.0, 1.0, 3.0),
            market=MarketSnapshot(10.0, 2.0, 1.2, 1.1),
            comment="test",
        )

    ranked = assign_ranks(
        [
            recommendation("LOW", 10.0),
            recommendation("HIGH", 90.0),
        ]
    )

    assert ranked[0].symbol == "HIGH"
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
