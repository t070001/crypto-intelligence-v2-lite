import pandas as pd

from data_fetcher import add_taker_fields, calc_vwap_rolling_24h


def test_add_taker_fields_uses_quote_ratio():
    df = pd.DataFrame(
        {
            "volume": [10.0],
            "quote_volume": [100.0],
            "taker_base": [6.0],
            "taker_quote": [70.0],
        }
    )

    result = add_taker_fields(df)

    assert result.loc[0, "taker_buy_quote"] == 70.0
    assert result.loc[0, "taker_sell_quote"] == 30.0
    assert round(result.loc[0, "taker_buy_ratio"], 4) == 2.3333


def test_calc_vwap_rolling_24h_returns_series():
    df = pd.DataFrame(
        {
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.0, 11.0],
            "quote_volume": [100.0, 200.0],
        }
    )

    result = calc_vwap_rolling_24h(df, window=2)

    assert len(result) == 2
    assert result.iloc[-1] > 0
