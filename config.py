from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    base_url: str = "https://fapi.binance.com"
    request_timeout: int = 20

    h4_limit: int = 200
    m15_limit: int = 96
    oi_limit: int = 200

    min_24h_volume_usdt: float = 10_000_000

    stable_symbols: set[str] = field(
        default_factory=lambda: {
            "USDCUSDT",
            "FDUSDUSDT",
            "TUSDUSDT",
            "USDPUSDT",
        }
    )
