# Crypto Intelligence v2 Lite

Project codename: Project_Whale_Footprint_v2.0_Lite

Status: ALPHA - development

This repository contains the v2 Lite rewrite of the crypto intelligence system.
The design uses 4H candles for candidate generation and 15m candles for intrabar
entry/exit validation.

## Current Scope

T-001 Data Adapter:

- Binance Futures kline fetcher
- 15m intraday fetch helper
- 4H open interest fetch helper
- Quote-volume based taker buy/sell mapping
- Rolling 24h VWAP helper
- OI-to-4H-kline alignment helper without look-ahead

## Design Notes

- `taker_buy_quote` is used as the primary taker buy volume measure.
- `taker_sell_quote = quote_volume - taker_buy_quote`.
- `taker_buy_ratio = taker_buy_quote / taker_sell_quote`.
- `CVD_approx` in later modules will be derived from Binance kline taker fields,
  not raw trade-level data.
- Hard stop rules will be implemented in the v2 backtest engine and must never be
  softened by OI or narrative conditions.
