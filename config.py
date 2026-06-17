from dataclasses import dataclass, field

# ============================================================
# v2.5 部位管理 (不變)
# ============================================================
RISK_PER_TRADE = 0.02

# ============================================================
# TELEGRAM 訊號推送
# ============================================================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# ============================================================
# v2.4 最終參數 (凍結)
# ============================================================
MODE_A_SWEEP_THRESHOLD = 0.02
MODE_A_RECOVERY_ANCHOR = "sweep_low"
MODE_A_VWAP_RECLAIM_MODE = "relative"
MODE_A_VWAP_RECOVERY_RATIO = 0.382
MODE_A_WICK_RATIO = 0.3
MODE_A_TAKER_RATIO = 1.05
MODE_B_OI_CHANGE_THRESHOLD = 0.005
MODE_B_OI_EXCEPTION_ENABLED = True
MODE_B_OI_EXCEPTION_THRESHOLD = -0.01
MODE_B_EXCEPTION_VOLUME_RATIO = 2.0
MODE_B_EXCEPTION_CANDLE_BODY = 0.015
MODE_B_EXCEPTION_TAKER_RATIO = 1.3
MODE_B_CVD_REQUIRED = False
MODE_B_TAKER_RATIO_REQUIRED = False

# ============================================================
# v3.1 Capital Flow Validation Layer 參數
# ============================================================

# --- Part 1: OI Change 重構 (short + mid) ---
V3_OI_CHANGE_SHORT = 3                # 短期 OI 變化 (3 x 4H = 12 小時)
V3_OI_CHANGE_MID = 12                 # 中期 OI 變化 (12 x 4H = 48 小時)
V3_OI_CHANGE_SHORT_WEIGHT = 0.4       # short 權重 40%
V3_OI_CHANGE_MID_WEIGHT = 0.6         # mid 權重 60%

# --- Part 2: OI Breakout Strength (連續評分取代二元) ---
V3_OI_BREAKOUT_LOOKBACK = 20          # OI 突破前高回溯窗口

# --- Part 3: OI Expansion Continuous Scoring ---
V3_OI_EXPANSION_WINDOW = 10           # OI 相對平均窗口
# 連續評分 f(x):
# ratio < 1.00 → 0
# 1.00~1.10 → 0~3
# 1.10~1.20 → 3~6
# 1.20~1.30 → 6~8
# 1.30~1.50 → 8~10
# ≥ 1.50    → 10

# --- Part 4: OI Slope 強化 (short + mid) ---
V3_OI_SLOPE_SHORT = 5                 # 短期斜率窗口 (5 x 4H)
V3_OI_SLOPE_MID = 12                  # 中期斜率窗口 (12 x 4H)
V3_OI_SLOPE_SHORT_WEIGHT = 0.5        # short 權重 50%
V3_OI_SLOPE_MID_WEIGHT = 0.5          # mid 權重 50%
V3_OI_SLOPE_POSITIVE_THRESHOLD = 0.001 # 斜率正值門檻（標準化用）

# --- Part 5: Structure Engine 重構 (5/10/20 三層) ---
V3_STRUCTURE_WINDOW_5 = 5             # 短期結構窗口
V3_STRUCTURE_WINDOW_10 = 10           # 中期結構窗口
V3_STRUCTURE_WINDOW_20 = 20           # 長期結構窗口
V3_STRUCTURE_WINDOW_5_WEIGHT = 0.40   # 短期結構權重 40%
V3_STRUCTURE_WINDOW_10_WEIGHT = 0.35  # 中期結構權重 35%
V3_STRUCTURE_WINDOW_20_WEIGHT = 0.25  # 長期結構權重 25%
# 各層級內：bullish=100, neutral=50, bearish=0 (%)

# --- Mode A (Liquidity Sweep + Capital Retention) ---
V3_MODE_A_SWEEP_THRESHOLD = 0.008
V3_MODE_A_VWAP_RECOVERY_RATIO = 0.5
V3_MODE_A_WICK_RATIO = 0.3
V3_MODE_A_TAKER_RATIO = 1.05
V3_MODE_A_MIN_OI_CHANGE = -0.03
V3_MODE_A_OI_REJECT_THRESHOLD = -0.05

# --- Mode B (Capital Impulse Breakout) ---
V3_MODE_B_OI_CHANGE_THRESHOLD = 0.005
V3_MODE_B_OI_EXCEPTION_ENABLED = True
V3_MODE_B_OI_EXCEPTION_THRESHOLD = -0.01
V3_MODE_B_EXCEPTION_VOLUME_RATIO = 2.0
V3_MODE_B_EXCEPTION_CANDLE_BODY = 0.015
V3_MODE_B_EXCEPTION_TAKER_RATIO = 1.3
V3_MODE_B_EXCEPTION_MAX_SCORE = 60.0

# --- Higher Timeframe Filter ---
V3_MA25_PERIOD = 25
V3_MA99_PERIOD = 99
V3_HTF_MIN_SCORE = 60.0

# --- V3.1 Scoring Weights ---
V3_WEIGHT_OI = 0.35                    # OI Engine 35% (↑ 5%)
V3_WEIGHT_STRUCTURE = 0.25             # Structure Engine 25% (→ 不變)
V3_WEIGHT_VOLUME = 0.20                # Volume Expansion 20% (→ 不變)
V3_WEIGHT_LIQUIDITY = 0.10             # Liquidity Context 10% (→ 不變)
V3_WEIGHT_FUNDING = 0.05               # Funding 5% (→ 不變)
V3_WEIGHT_LONGSHORT = 0.05             # Long/Short 5% (↓ 5%)

# --- Scoring Calibration ---
V3_VOLUME_RATIO_MAX = 3.0

# --- Grade Thresholds ---
V3_GRADE_S_THRESHOLD = 85.0
V3_GRADE_A_THRESHOLD = 70.0
V3_GRADE_B_THRESHOLD = 55.0
V3_GRADE_C_THRESHOLD = 40.0

# --- Entry Zone ---
V3_ENTRY_PRIORITY = [
    "breakout_platform",
    "ma25",
    "vwap",
    "fvg_midpoint",
]
V3_ENTRY_ZONE_SPREAD = 0.02

# --- Part 8: Lottery Coin Penalty (不再過濾, 改為 -20 懲罰) ---
V3_LOTTERY_MAX_DECLINE_PCT = 80.0
V3_LOTTERY_PENALTY = -20.0            # Lottery 扣 20 分
V3_LOTTERY_MIN_MA99_DISTANCE = 0.0

# --- Short Covering Rally Penalty ---
V3_SC_RALLY_PENALTY = -15.0           # Short Covering Rally 扣 15 分


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
            "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT",
        }
    )