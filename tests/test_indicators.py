import numpy as np
import pandas as pd

from indicators import (
    calc_atr,
    calc_breakout_platform,
    calc_cvd_approx,
    calc_cvd_slope_zscore,
    calc_volume_ratio,
)


def test_calc_atr_uses_true_range():
    df = pd.DataFrame(
        {
            "high": [12.0, 13.0, 15.0],
            "low": [10.0, 11.0, 12.0],
            "close": [11.0, 12.0, 14.0],
        }
    )

    result = calc_atr(df, period=2)

    assert np.isnan(result.iloc[0])
    assert result.iloc[-1] == 2.5


def test_calc_volume_ratio_uses_prior_average():
    df = pd.DataFrame({"volume": [10.0, 10.0, 10.0, 30.0]})

    result = calc_volume_ratio(df, window=3)

    assert result.iloc[-1] == 3.0


def test_calc_breakout_platform_detects_compression():
    df = pd.DataFrame(
        {
            "high": [104.0] * 49 + [106.0],
            "low": [100.0] * 50,
            "close": [103.0] * 49 + [105.0],
        }
    )

    result = calc_breakout_platform(df)

    assert result.found is True
    assert result.breakout_price is not None
    assert result.range_pct < 0.15
    assert result.extension_pct is not None


def test_calc_breakout_platform_rejects_wide_range():
    df = pd.DataFrame(
        {
            "high": [130.0] * 50,
            "low": [100.0] * 50,
            "close": [120.0] * 50,
        }
    )

    result = calc_breakout_platform(df)

    assert result.found is False
    assert result.range_pct >= 0.15


def test_calc_cvd_approx_uses_quote_taker_delta():
    df = pd.DataFrame(
        {
            "taker_buy_quote": [70.0, 40.0],
            "taker_sell_quote": [30.0, 60.0],
        }
    )

    result = calc_cvd_approx(df)

    assert result.tolist() == [40.0, 20.0]


def test_calc_cvd_slope_zscore_adds_expected_columns():
    values = []
    for index in range(60):
        taker_buy = 60.0 + index
        taker_sell = 40.0
        values.append(
            {
                "taker_buy_quote": taker_buy,
                "taker_sell_quote": taker_sell,
            }
        )
    df = pd.DataFrame(values)

    result = calc_cvd_slope_zscore(df, slope_window=8, z_window=16)

    assert {"cvd_approx", "cvd_slope", "cvd_slope_zscore"}.issubset(
        result.columns
    )
    assert result["cvd_slope"].iloc[-1] > 0
    assert not np.isnan(result["cvd_slope_zscore"].iloc[-1])
