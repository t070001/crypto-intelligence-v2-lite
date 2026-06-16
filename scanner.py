from __future__ import annotations

from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]

KLINE_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_base",
    "taker_quote",
]


class BinanceScanner:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.session = requests.Session()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=20),
    )
    def _get(self, endpoint: str, params: dict[str, Any] | None = None):
        response = self.session.get(
            f"{self.config.base_url}{endpoint}",
            params=params,
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_exchange_info(self) -> dict[str, Any]:
        return self._get("/fapi/v1/exchangeInfo")

    def get_24h_tickers(self) -> list[dict[str, Any]]:
        return self._get("/fapi/v1/ticker/24hr")

    def get_usdt_perpetual_symbols(self) -> list[str]:
        data = self.get_exchange_info()
        symbols: list[str] = []

        for item in data["symbols"]:
            symbol = item["symbol"]
            if (
                item.get("quoteAsset") == "USDT"
                and item.get("contractType") == "PERPETUAL"
                and item.get("status") == "TRADING"
                and symbol not in self.config.stable_symbols
            ):
                symbols.append(symbol)

        return symbols

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        data = self._get(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
        )

        df = pd.DataFrame(data, columns=KLINE_COLUMNS)
        df[KLINE_NUMERIC_COLUMNS] = df[KLINE_NUMERIC_COLUMNS].astype(float)
        df["trades"] = df["trades"].astype(int)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df

    def get_open_interest_history(
        self,
        symbol: str,
        period: str = "4h",
        limit: int | None = None,
    ) -> pd.DataFrame:
        data = self._get(
            "/futures/data/openInterestHist",
            {
                "symbol": symbol,
                "period": period,
                "limit": limit or self.config.oi_limit,
            },
        )

        df = pd.DataFrame(data)
        if df.empty:
            return df

        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
