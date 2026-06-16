from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from scanner import BinanceScanner


TAKER_FIELD_MAPPING = {
    "taker_buy_base": "taker_base",
    "taker_buy_quote": "taker_quote",
    "taker_sell_base": "volume - taker_base",
    "taker_sell_quote": "quote_volume - taker_quote",
}


def add_taker_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add quote-volume based taker buy/sell fields.

    Quote volume is the primary unit for ratios because base volume is not
    comparable across symbols.
    """

    df = df.copy()
    df["taker_buy_base"] = df["taker_base"]
    df["taker_buy_quote"] = df["taker_quote"]
    df["taker_sell_base"] = (df["volume"] - df["taker_base"]).clip(lower=0)
    df["taker_sell_quote"] = (
        df["quote_volume"] - df["taker_quote"]
    ).clip(lower=0)

    denominator = df["taker_sell_quote"].replace(0, np.nan)
    df["taker_buy_ratio"] = (df["taker_buy_quote"] / denominator).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


def calc_vwap_rolling_24h(df_15m: pd.DataFrame, window: int = 96) -> pd.Series:
    typical_price = (df_15m["high"] + df_15m["low"] + df_15m["close"]) / 3
    price_volume = typical_price * df_15m["quote_volume"]
    rolling_volume = df_15m["quote_volume"].rolling(window, min_periods=1).sum()
    rolling_price_volume = price_volume.rolling(window, min_periods=1).sum()
    return rolling_price_volume / rolling_volume.replace(0, np.nan)


def calc_vwap_utc_day(df_15m: pd.DataFrame) -> pd.Series:
    df = df_15m.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["price_volume"] = typical_price * df["quote_volume"]
    day = df["open_time"].dt.floor("D")
    cumulative_pv = df.groupby(day)["price_volume"].cumsum()
    cumulative_volume = df.groupby(day)["quote_volume"].cumsum()
    return cumulative_pv / cumulative_volume.replace(0, np.nan)


class DataFetcher:
    def __init__(self, scanner: BinanceScanner | None = None, config: Config | None = None):
        self.config = config or Config()
        self.scanner = scanner or BinanceScanner(self.config)

    def fetch_h4(self, symbol: str, limit: int | None = None) -> pd.DataFrame:
        df = self.scanner.get_klines(
            symbol=symbol,
            interval="4h",
            limit=limit or self.config.h4_limit,
        )
        return add_taker_fields(df)

    def fetch_intraday_15m(self, symbol: str, lookback: int | None = None) -> pd.DataFrame:
        df = self.scanner.get_klines(
            symbol=symbol,
            interval="15m",
            limit=lookback or self.config.m15_limit,
        )
        df = add_taker_fields(df)
        df["vwap_rolling_24h"] = calc_vwap_rolling_24h(df)
        df["vwap_utc_day"] = calc_vwap_utc_day(df)
        return df

    def fetch_oi(
        self,
        symbol: str,
        period: str = "4h",
        limit: int | None = None,
    ) -> pd.DataFrame:
        return self.scanner.get_open_interest_history(
            symbol=symbol,
            period=period,
            limit=limit or self.config.oi_limit,
        )

    def align_oi_to_klines(
        self,
        oi_df: pd.DataFrame,
        kline_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align OI rows to 4H candles without using future OI samples.

        Each kline receives the latest OI sample whose timestamp is less than or
        equal to that candle's close_time. The returned frame is indexed by the
        kline open_time to keep the main 4H signal loop explicit.
        """

        if oi_df.empty or kline_df.empty:
            return pd.DataFrame()

        left = kline_df[["open_time", "close_time"]].sort_values("close_time")
        right = oi_df.sort_values("timestamp")

        aligned = pd.merge_asof(
            left,
            right,
            left_on="close_time",
            right_on="timestamp",
            direction="backward",
        )

        return aligned.set_index("open_time")
