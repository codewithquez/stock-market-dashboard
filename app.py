import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sharp Key Level Options Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Sharp Key Level Options Screener")
st.caption(
    "Built for tight intraday and 2-3 day swing setups with sharp key levels, tight stops, 3 targets, and swing EMA structure."
)


# ============================================================
# CONFIG
# ============================================================
DEFAULT_UNIVERSE = sorted(
    list(
        set(
            [
                "SPY", "QQQ", "IWM", "DIA", "SMH", "XLF",
                "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "AMD",
                "NFLX", "PLTR", "AVGO", "JPM", "BAC", "XOM", "INTC",
                "COIN", "MSTR", "SHOP", "CRM", "MU", "UBER", "SNOW", "PANW", "ORCL",
            ]
        )
    )
)
WATCHLIST_FILE = Path("sharp_key_level_watchlist.csv")
BULLISH_PATTERNS = {"Hammer", "Inverted Hammer", "Bullish Engulfing", "Tweezer Bottom", "Morning Star"}
BEARISH_PATTERNS = {"Hanging Man", "Shooting Star", "Bearish Engulfing", "Tweezer Top", "Evening Star"}


# ============================================================
# BASIC HELPERS
# ============================================================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df.copy())
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.dropna(subset=needed)


def fetch_data(symbol: str, interval: str, period: str, prepost: bool = False) -> pd.DataFrame:
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
        prepost=prepost,
    )
    return ensure_ohlcv(df)


def normalize_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    try:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            out.index = out.index.tz_convert("America/New_York")
    except Exception:
        pass
    return out


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def body_size(row: pd.Series) -> float:
    return abs(row["Close"] - row["Open"])


def candle_range(row: pd.Series) -> float:
    return max(row["High"] - row["Low"], 1e-9)


def upper_wick(row: pd.Series) -> float:
    return row["High"] - max(row["Open"], row["Close"])


def lower_wick(row: pd.Series) -> float:
    return min(row["Open"], row["Close"]) - row["Low"]


def is_bullish(row: pd.Series) -> bool:
    return row["Close"] > row["Open"]


def is_bearish(row: pd.Series) -> bool:
    return row["Close"] < row["Open"]


def trend_slope(series: pd.Series, length: int = 6) -> float:
    if len(series) < length:
        return 0.0
    y = series.tail(length).values
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def atr_percent(df: pd.DataFrame, length: int = 14) -> float:
    a = atr(df, length).iloc[-1]
    c = df["Close"].iloc[-1]
    if pd.isna(a) or c == 0:
        return 0.0
    return float((a / c) * 100)


def rv_20(df: pd.DataFrame) -> float:
    returns = df["Close"].pct_change().dropna()
    if len(returns) < 20:
        return 0.0
    return float(returns.tail(20).std() * math.sqrt(252))


def average_dollar_volume(df: pd.DataFrame, length: int = 20) -> float:
    recent = df.tail(length).copy()
    return float((recent["Close"] * recent["Volume"]).mean())


def volume_ratio(df: pd.DataFrame, length: int = 20) -> float:
    if len(df) < length + 1:
        return 1.0
    avg = df["Volume"].tail(length).mean()
    cur = df["Volume"].iloc[-1]
    if avg == 0:
        return 1.0
    return float(cur / avg)


def pct_move(df: pd.DataFrame, bars: int = 10) -> float:
    if len(df) < bars + 1:
        return 0.0
    start = df["Close"].iloc[-bars - 1]
    end = df["Close"].iloc[-1]
    if start == 0:
        return 0.0
    return float(((end - start) / start) * 100)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_pv = (typical * df["Volume"]).cumsum()
    cumulative_volume = df["Volume"].cumsum().replace(0, np.nan)
    return cumulative_pv / cumulative_volume


# ============================================================
# WATCHLIST STORAGE
# ============================================================
def load_saved_watchlist() -> pd.DataFrame:
    if WATCHLIST_FILE.exists():
        try:
            return pd.read_csv(WATCHLIST_FILE)
        except Exception:
            return pd.DataFrame(
                columns=["Symbol", "Setup Type", "Score", "Bias", "Setup Label", "Timestamp"]
            )
    return pd.DataFrame(columns=["Symbol", "Setup Type", "Score", "Bias", "Setup Label", "Timestamp"])


def save_watchlist_rows(rows: List[Dict]) -> None:
    if not rows:
        return
    existing = load_saved_watchlist()
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Symbol", "Setup Type", "Setup Label"], keep="last")
    combined.to_csv(WATCHLIST_FILE, index=False)


# ============================================================
# PATTERN DETECTION
# ============================================================
def detect_single_candle_patterns(df: pd.DataFrame) -> List[str]:
    if len(df) < 3:
        return []

    cur = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    patterns: List[str] = []

    cur_body = body_size(cur)
    cur_range = candle_range(cur)
    cur_upper = upper_wick(cur)
    cur_lower = lower_wick(cur)

    short_trend_down = trend_slope(df["Close"].iloc[:-1], 5) < 0
    short_trend_up = trend_slope(df["Close"].iloc[:-1], 5) > 0

    if cur_body <= cur_range * 0.1:
        patterns.append("Doji")

    if cur_lower >= cur_body * 2 and cur_upper <= max(cur_body * 0.5, cur_range * 0.15):
        if short_trend_down:
            patterns.append("Hammer")
        elif short_trend_up:
            patterns.append("Hanging Man")

    if cur_upper >= cur_body * 2 and cur_lower <= max(cur_body * 0.5, cur_range * 0.15):
        if short_trend_down:
            patterns.append("Inverted Hammer")
        elif short_trend_up:
            patterns.append("Shooting Star")

    if is_bearish(prev) and is_bullish(cur):
        if cur["Open"] <= prev["Close"] and cur["Close"] >= prev["Open"]:
            patterns.append("Bullish Engulfing")

    if is_bullish(prev) and is_bearish(cur):
        if cur["Open"] >= prev["Close"] and cur["Close"] <= prev["Open"]:
            patterns.append("Bearish Engulfing")

    low_close = abs(cur["Low"] - prev["Low"]) / max(cur["Low"], 1e-9) < 0.003
    high_close = abs(cur["High"] - prev["High"]) / max(cur["High"], 1e-9) < 0.003

    if low_close and short_trend_down:
        patterns.append("Tweezer Bottom")
    if high_close and short_trend_up:
        patterns.append("Tweezer Top")

    c1, c2, c3 = prev2, prev, cur
    c1_body = body_size(c1)
    c2_body = body_size(c2)

    if is_bearish(c1) and c2_body < c1_body * 0.6 and is_bullish(c3):
        midpoint = (c1["Open"] + c1["Close"]) / 2
        if c3["Close"] > midpoint:
            patterns.append("Morning Star")

    if is_bullish(c1) and c2_body < c1_body * 0.6 and is_bearish(c3):
        midpoint = (c1["Open"] + c1["Close"]) / 2
        if c3["Close"] < midpoint:
            patterns.append("Evening Star")

    return patterns


@dataclass
class FlagResult:
    name: Optional[str]
    direction: Optional[str]
    score: float
    details: str
    upper_line: Optional[Tuple[int, float, int, float]] = None
    lower_line: Optional[Tuple[int, float, int, float]] = None


def detect_flag_pattern(df: pd.DataFrame, lookback: int = 20) -> FlagResult:
    if len(df) < lookback + 5:
        return FlagResult(None, None, 0.0, "Not enough bars")

    recent = df.tail(lookback).copy().reset_index(drop=True)
    pole = recent.iloc[:8]
    cons = recent.iloc[8:]
    cons_len = len(cons)

    pole_move = ((pole["Close"].iloc[-1] - pole["Open"].iloc[0]) / pole["Open"].iloc[0]) * 100
    x = np.arange(cons_len)
    high_fit = np.polyfit(x, cons["High"].values, 1)
    low_fit = np.polyfit(x, cons["Low"].values, 1)
    cons_high_slope = high_fit[0]
    cons_low_slope = low_fit[0]
    cons_close_slope = np.polyfit(x, cons["Close"].values, 1)[0]

    cons_range_mean = float((cons["High"] - cons["Low"]).mean())
    pole_range_mean = float((pole["High"] - pole["Low"]).mean())

    narrowing = abs(cons_high_slope - cons_low_slope) < max(abs(cons_close_slope) * 1.5, 1e-6)
    tighter_than_pole = cons_range_mean < pole_range_mean
    bullish_pole = pole_move > 3.0
    bearish_pole = pole_move < -3.0

    start_idx = len(recent) - cons_len
    upper_line = (start_idx, high_fit[1], len(recent) - 1, high_fit[0] * (cons_len - 1) + high_fit[1])
    lower_line = (start_idx, low_fit[1], len(recent) - 1, low_fit[0] * (cons_len - 1) + low_fit[1])

    if bullish_pole and tighter_than_pole:
        if cons_high_slope < 0 and cons_low_slope < 0:
            score = min(100.0, abs(pole_move) * 8 + (pole_range_mean / max(cons_range_mean, 1e-9)) * 10)
            return FlagResult("Bullish Flag", "Bullish", float(score), "Upside pole with downward consolidation", upper_line, lower_line)
        if cons_high_slope < 0 and cons_low_slope > 0 and narrowing:
            score = min(100.0, abs(pole_move) * 8 + 20)
            return FlagResult("Bullish Pennant", "Bullish", float(score), "Upside pole with contracting pennant", upper_line, lower_line)

    if bearish_pole and tighter_than_pole:
        if cons_high_slope > 0 and cons_low_slope > 0:
            score = min(100.0, abs(pole_move) * 8 + (pole_range_mean / max(cons_range_mean, 1e-9)) * 10)
            return FlagResult("Bearish Flag", "Bearish", float(score), "Downside pole with upward consolidation", upper_line, lower_line)
        if cons_high_slope < 0 and cons_low_slope > 0 and narrowing:
            score = min(100.0, abs(pole_move) * 8 + 20)
            return FlagResult("Bearish Pennant", "Bearish", float(score), "Downside pole with contracting pennant", upper_line, lower_line)

    return FlagResult(None, None, 0.0, "No clean flag structure found", upper_line, lower_line)


# ============================================================
# KEY LEVELS
# ============================================================
@dataclass
class LevelInfo:
    price: float
    touches: int
    level_type: str


def cluster_levels(levels: List[float], tolerance_pct: float = 0.35) -> List[float]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters: List[List[float]] = [[levels[0]]]

    for level in levels[1:]:
        last_cluster = clusters[-1]
        center = float(np.mean(last_cluster))
        if abs((level - center) / center) * 100 <= tolerance_pct:
            last_cluster.append(level)
        else:
            clusters.append([level])

    return [round(float(np.mean(cluster)), 2) for cluster in clusters]


def extract_pivot_levels(df: pd.DataFrame, left: int = 3, right: int = 3, recent_bars: int = 120) -> Tuple[List[LevelInfo], List[LevelInfo]]:
    recent = df.tail(recent_bars).copy().reset_index(drop=True)
    highs: List[float] = []
    lows: List[float] = []

    for i in range(left, len(recent) - right):
        cur_high = recent.loc[i, "High"]
        cur_low = recent.loc[i, "Low"]

        if cur_high == recent.loc[i - left:i + right, "High"].max():
            highs.append(float(cur_high))
        if cur_low == recent.loc[i - left:i + right, "Low"].min():
            lows.append(float(cur_low))

    clustered_highs = cluster_levels(highs)
    clustered_lows = cluster_levels(lows)

    high_infos: List[LevelInfo] = []
    low_infos: List[LevelInfo] = []

    for level in clustered_highs:
        touches = sum(1 for h in highs if abs((h - level) / level) * 100 <= 0.35)
        if touches >= 2:
            high_infos.append(LevelInfo(level, touches, "Resistance"))

    for level in clustered_lows:
        touches = sum(1 for l in lows if abs((l - level) / level) * 100 <= 0.35)
        if touches >= 2:
            low_infos.append(LevelInfo(level, touches, "Support"))

    high_infos = sorted(high_infos, key=lambda x: x.price)
    low_infos = sorted(low_infos, key=lambda x: x.price)
    return low_infos, high_infos


def nearest_levels(price: float, supports: List[LevelInfo], resistances: List[LevelInfo]) -> Tuple[Optional[LevelInfo], Optional[LevelInfo], List[LevelInfo], List[LevelInfo]]:
    support_below = [s for s in supports if s.price <= price]
    resistance_above = [r for r in resistances if r.price >= price]

    nearest_support = max(support_below, key=lambda x: x.price) if support_below else None
    nearest_resistance = min(resistance_above, key=lambda x: x.price) if resistance_above else None

    next_resistances = sorted(resistance_above, key=lambda x: x.price)[:3]
    next_supports = sorted([s for s in supports if s.price < price], key=lambda x: x.price, reverse=True)[:3]
    return nearest_support, nearest_resistance, next_supports, next_resistances


def price_near_level(price: float, level: float, threshold_pct: float = 0.35) -> bool:
    if level == 0:
        return False
    return abs((price - level) / level) * 100 <= threshold_pct


def detect_breakout_breakdown(price: float, support: Optional[LevelInfo], resistance: Optional[LevelInfo], threshold_pct: float = 0.2) -> str:
    if resistance and price > resistance.price * (1 + threshold_pct / 100):
        return "Breakout"
    if support and price < support.price * (1 - threshold_pct / 100):
        return "Breakdown"
    return "Inside Range"


# ============================================================
# BIAS / TREND / EMAS / STRUCTURE
# ============================================================
def get_trend(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "Neutral"
    ema8 = ema(df["Close"], 8)
    ema21 = ema(df["Close"], 21)
    ema50 = ema(df["Close"], 50)
    price = df["Close"].iloc[-1]
    if price > ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]:
        return "Bullish"
    if price < ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]:
        return "Bearish"
    return "Neutral"


def get_ema_state(df: pd.DataFrame) -> Dict[str, float]:
    close = df["Close"]
    ema8 = ema(close, 8)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    last_price = close.iloc[-1]

    bullish_stack = last_price > ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]
    bearish_stack = last_price < ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]

    return {
        "EMA 8": round(float(ema8.iloc[-1]), 2),
        "EMA 21": round(float(ema21.iloc[-1]), 2),
        "EMA 50": round(float(ema50.iloc[-1]), 2),
        "Bullish EMA Stack": bullish_stack,
        "Bearish EMA Stack": bearish_stack,
    }


def pattern_bias(patterns: List[str]) -> str:
    bull = sum(1 for p in patterns if p in BULLISH_PATTERNS)
    bear = sum(1 for p in patterns if p in BEARISH_PATTERNS)
    if bull > bear:
        return "Bullish"
    if bear > bull:
        return "Bearish"
    return "Neutral"


def volume_spike(df: pd.DataFrame, length: int = 20, multiple: float = 1.5) -> bool:
    if len(df) < length + 1:
        return False
    avg_vol = df["Volume"].tail(length).mean()
    current_vol = df["Volume"].iloc[-1]
    return bool(current_vol > multiple * avg_vol)


def relative_strength_vs_benchmark(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, bars: int = 10) -> float:
    if len(stock_df) < bars + 1 or len(benchmark_df) < bars + 1:
        return 0.0
    stock_ret = (stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-bars - 1]) - 1
    bench_ret = (benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-bars - 1]) - 1
    return round((stock_ret - bench_ret) * 100, 2)


def get_vwap_bias(intraday_df: pd.DataFrame) -> str:
    try:
        vwap_series = calculate_vwap(intraday_df.copy())
        last_close = intraday_df["Close"].iloc[-1]
        last_vwap = vwap_series.iloc[-1]
        if last_close > last_vwap:
            return "Bullish"
        if last_close < last_vwap:
            return "Bearish"
        return "Neutral"
    except Exception:
        return "Neutral"


def classify_setup(
    bias: str,
    trend_trade: str,
    trend_higher: str,
    near_support: bool,
    near_resistance: bool,
    breakout_state: str,
    flag_name: str,
) -> str:
    if bias == "Bullish" and breakout_state == "Breakout":
        return "Bullish Breakout"
    if bias == "Bearish" and breakout_state == "Breakdown":
        return "Bearish Breakdown"
    if bias == "Bullish" and near_support:
        return "Bullish Support Bounce"
    if bias == "Bearish" and near_resistance:
        return "Bearish Resistance Rejection"
    if "Bullish" in flag_name and trend_trade == "Bullish" and trend_higher == "Bullish":
        return "Bullish Continuation Flag"
    if "Bearish" in flag_name and trend_trade == "Bearish" and trend_higher == "Bearish":
        return "Bearish Continuation Flag"
    return "Developing Setup"


# ============================================================
# OPTIONS SUGGESTION HELPERS
# ============================================================
def suggest_strike(price: float) -> float:
    if price < 50:
        return round(price)
    if price < 200:
        return round(price / 5) * 5
    return round(price / 10) * 10


def suggest_expiration(days_out: int = 10) -> str:
    target = datetime.today() + timedelta(days=days_out)
    return target.strftime("%Y-%m-%d")


def iv_proxy(df: pd.DataFrame) -> float:
    returns = df["Close"].pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(252) * 100)


# ============================================================
# OPTIONS LIQUIDITY
# ============================================================
def get_options_liquidity(symbol: str) -> Dict[str, float]:
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return {
                "Nearest Expiration": "N/A",
                "Options OI": 0.0,
                "Options Volume": 0.0,
                "Spread %": 999.0,
                "Calls Liquid": False,
                "Puts Liquid": False,
            }

        nearest = expirations[0]
        chain = ticker.option_chain(nearest)
        calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
        puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()

        def summarize(side_df: pd.DataFrame) -> Tuple[float, float, float, bool]:
            if side_df.empty:
                return 0.0, 0.0, 999.0, False
            filtered = side_df.copy()
            filtered = filtered[(filtered["bid"].fillna(0) > 0) & (filtered["ask"].fillna(0) > 0)]
            if filtered.empty:
                return 0.0, 0.0, 999.0, False

            filtered["spread_pct"] = ((filtered["ask"] - filtered["bid"]) / ((filtered["ask"] + filtered["bid"]) / 2)).replace([np.inf, -np.inf], np.nan) * 100
            filtered = filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=["spread_pct"])
            if filtered.empty:
                return 0.0, 0.0, 999.0, False

            oi = float(filtered["openInterest"].fillna(0).max())
            vol = float(filtered["volume"].fillna(0).max())
            spread = float(filtered["spread_pct"].median())
            liquid = bool(oi >= 1000 and vol >= 100 and spread <= 8)
            return oi, vol, spread, liquid

        call_oi, call_vol, call_spread, calls_liquid = summarize(calls)
        put_oi, put_vol, put_spread, puts_liquid = summarize(puts)

        return {
            "Nearest Expiration": nearest,
            "Options OI": round(max(call_oi, put_oi), 0),
            "Options Volume": round(max(call_vol, put_vol), 0),
            "Spread %": round(min(call_spread, put_spread), 2),
            "Calls Liquid": calls_liquid,
            "Puts Liquid": puts_liquid,
        }
    except Exception:
        return {
            "Nearest Expiration": "N/A",
            "Options OI": 0.0,
            "Options Volume": 0.0,
            "Spread %": 999.0,
            "Calls Liquid": False,
            "Puts Liquid": False,
        }


# ============================================================
# 3-TARGET TRADE PLANS
# ============================================================
def derive_intraday_trade_plan(
    df: pd.DataFrame,
    price: float,
    bias: str,
    nearest_support: Optional[LevelInfo],
    nearest_resistance: Optional[LevelInfo],
    next_supports: List[LevelInfo],
    next_resistances: List[LevelInfo],
    atr_value: float,
) -> Tuple[float, float, float, float, float]:
    recent = df.tail(6).copy()
    recent_low = float(recent["Low"].min()) if not recent.empty else price - atr_value
    recent_high = float(recent["High"].max()) if not recent.empty else price + atr_value
    buffer_amt = max(atr_value * 0.15, price * 0.0015)

    if bias == "Bullish":
        structure_stop = recent_low - buffer_amt
        if nearest_support:
            structure_stop = max(structure_stop, nearest_support.price - buffer_amt)
        stop = structure_stop
        risk = max(price - stop, atr_value * 0.25)

        targets = [price + risk * 1.0, price + risk * 1.5, price + risk * 2.0]
        res_prices = [lvl.price for lvl in next_resistances if lvl.price > price]
        for i in range(min(3, len(res_prices))):
            targets[i] = min(targets[i], res_prices[i])

        return round(price, 2), round(stop, 2), round(targets[0], 2), round(targets[1], 2), round(targets[2], 2)

    if bias == "Bearish":
        structure_stop = recent_high + buffer_amt
        if nearest_resistance:
            structure_stop = min(structure_stop, nearest_resistance.price + buffer_amt) if nearest_resistance.price > price else structure_stop
        stop = structure_stop
        risk = max(stop - price, atr_value * 0.25)

        targets = [price - risk * 1.0, price - risk * 1.5, price - risk * 2.0]
        sup_prices = [lvl.price for lvl in next_supports if lvl.price < price]
        for i in range(min(3, len(sup_prices))):
            targets[i] = max(targets[i], sup_prices[i])

        return round(price, 2), round(stop, 2), round(targets[0], 2), round(targets[1], 2), round(targets[2], 2)

    return round(price, 2), round(price - atr_value, 2), round(price + atr_value, 2), round(price + atr_value * 1.5, 2), round(price + atr_value * 2.0, 2)


def derive_swing_trade_plan(
    df: pd.DataFrame,
    price: float,
    bias: str,
    nearest_support: Optional[LevelInfo],
    nearest_resistance: Optional[LevelInfo],
    next_supports: List[LevelInfo],
    next_resistances: List[LevelInfo],
    atr_value: float,
) -> Tuple[float, float, float, float, float]:
    recent = df.tail(8).copy()
    recent_low = float(recent["Low"].min()) if not recent.empty else price - atr_value
    recent_high = float(recent["High"].max()) if not recent.empty else price + atr_value
    buffer_amt = max(atr_value * 0.22, price * 0.002)

    if bias == "Bullish":
        stop_base = recent_low
        if nearest_support:
            stop_base = max(stop_base, nearest_support.price)
        stop = stop_base - buffer_amt
        risk = max(price - stop, atr_value * 0.4)

        targets = [price + risk * 1.25, price + risk * 1.75, price + risk * 2.25]
        res_prices = [lvl.price for lvl in next_resistances if lvl.price > price]
        for i in range(min(3, len(res_prices))):
            targets[i] = min(targets[i], res_prices[i])

        return round(price, 2), round(stop, 2), round(targets[0], 2), round(targets[1], 2), round(targets[2], 2)

    if bias == "Bearish":
        stop_base = recent_high
        if nearest_resistance and nearest_resistance.price > price:
            stop_base = min(stop_base, nearest_resistance.price)
        stop = stop_base + buffer_amt
        risk = max(stop - price, atr_value * 0.4)

        targets = [price - risk * 1.25, price - risk * 1.75, price - risk * 2.25]
        sup_prices = [lvl.price for lvl in next_supports if lvl.price < price]
        for i in range(min(3, len(sup_prices))):
            targets[i] = max(targets[i], sup_prices[i])

        return round(price, 2), round(stop, 2), round(targets[0], 2), round(targets[1], 2), round(targets[2], 2)

    return round(price, 2), round(price - atr_value, 2), round(price + atr_value, 2), round(price + atr_value * 1.5, 2), round(price + atr_value * 2.0, 2)


# ============================================================
# SCORING
# ============================================================
def grade_setup(score: float) -> str:
    if score >= 85:
        return "A+"
    if score >= 75:
        return "A"
    if score >= 65:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def score_setup(
    bias: str,
    trade_patterns: List[str],
    higher_patterns: List[str],
    flag_result: FlagResult,
    trend_trade: str,
    trend_higher: str,
    near_support: bool,
    near_resistance: bool,
    breakout_state: str,
    has_volume_spike: bool,
    rel_strength: float,
    vwap_bias: str,
    daily_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    options_liquidity: Dict[str, float],
    ema_stack_ok: bool,
) -> float:
    score = 0.0

    if bias in {"Bullish", "Bearish"}:
        score += 10
    if len([p for p in trade_patterns if p != "Doji"]) > 0:
        score += 6
    if len([p for p in higher_patterns if p != "Doji"]) > 0:
        score += 4

    if bias == trend_trade and bias == trend_higher and bias != "Neutral":
        score += 20
    elif bias == trend_trade and bias != "Neutral":
        score += 10

    if ema_stack_ok:
        score += 10

    if bias == "Bullish" and near_support:
        score += 10
    if bias == "Bearish" and near_resistance:
        score += 10
    if bias == "Bullish" and breakout_state == "Breakout":
        score += 12
    if bias == "Bearish" and breakout_state == "Breakdown":
        score += 12

    if has_volume_spike:
        score += 10

    if flag_result.name:
        score += min(12, flag_result.score / 8)

    if bias == vwap_bias and bias != "Neutral":
        score += 6

    if bias == "Bullish" and rel_strength > 0:
        score += min(8, rel_strength)
    if bias == "Bearish" and rel_strength < 0:
        score += min(8, abs(rel_strength))

    score += min(8, atr_percent(daily_df) * 1.3)
    score += min(6, rv_20(daily_df) * 4)
    score += min(6, average_dollar_volume(daily_df) / 100_000_000)

    if options_liquidity.get("Options OI", 0) >= 1000:
        score += 4
    if options_liquidity.get("Options Volume", 0) >= 100:
        score += 4
    if options_liquidity.get("Spread %", 999) <= 8:
        score += 4

    return round(min(score, 100), 2)


def is_fresh_setup(patterns: List[str], flag_name: str, min_score: float, score: float) -> bool:
    return ((patterns and patterns != ["Doji"]) or (flag_name and flag_name != "None")) and score >= min_score


# ============================================================
# ANALYSIS ENGINE
# ============================================================
def analyze_symbol(symbol: str, benchmark_daily: pd.DataFrame, intraday_interval: str) -> Dict:
    result: Dict = {"Symbol": symbol, "Status": "OK"}

    try:
        daily_df = fetch_data(symbol, "1d", "1y")
        intraday_df = fetch_data(symbol, intraday_interval, "60d")
        weekly_df = fetch_data(symbol, "1wk", "2y")
    except Exception as e:
        result["Status"] = f"Data error: {e}"
        return result

    if len(daily_df) < 80 or len(intraday_df) < 40 or len(weekly_df) < 40:
        result["Status"] = "Not enough data"
        return result

    intraday_patterns = detect_single_candle_patterns(intraday_df)
    daily_patterns = detect_single_candle_patterns(daily_df)
    intraday_flag = detect_flag_pattern(intraday_df, lookback=20)
    daily_flag = detect_flag_pattern(daily_df, lookback=20)
    options_liquidity = get_options_liquidity(symbol)

    intraday_bias = pattern_bias(intraday_patterns)
    swing_bias = pattern_bias(daily_patterns)

    intraday_trend = get_trend(intraday_df)
    daily_trend = get_trend(daily_df)
    weekly_trend = get_trend(weekly_df)

    intraday_ema_state = get_ema_state(intraday_df)
    daily_ema_state = get_ema_state(daily_df)
    weekly_ema_state = get_ema_state(weekly_df)

    support_levels, resistance_levels = extract_pivot_levels(daily_df)
    last_price = float(daily_df["Close"].iloc[-1])
    nearest_support, nearest_resistance, next_supports, next_resistances = nearest_levels(last_price, support_levels, resistance_levels)

    near_support = nearest_support is not None and price_near_level(last_price, nearest_support.price)
    near_resistance = nearest_resistance is not None and price_near_level(last_price, nearest_resistance.price)
    breakout_state = detect_breakout_breakdown(last_price, nearest_support, nearest_resistance)

    intraday_volume_spike = volume_spike(intraday_df)
    swing_volume_spike = volume_spike(daily_df)
    vwap_bias = get_vwap_bias(intraday_df)
    rel_strength = relative_strength_vs_benchmark(daily_df, benchmark_daily, bars=10)

    intraday_setup_label = classify_setup(
        intraday_bias,
        intraday_trend,
        daily_trend,
        near_support,
        near_resistance,
        breakout_state,
        intraday_flag.name or "",
    )

    swing_setup_label = classify_setup(
        swing_bias,
        daily_trend,
        weekly_trend,
        near_support,
        near_resistance,
        breakout_state,
        daily_flag.name or "",
    )

    intraday_score = score_setup(
        bias=intraday_bias,
        trade_patterns=intraday_patterns,
        higher_patterns=daily_patterns,
        flag_result=intraday_flag,
        trend_trade=intraday_trend,
        trend_higher=daily_trend,
        near_support=near_support,
        near_resistance=near_resistance,
        breakout_state=breakout_state,
        has_volume_spike=intraday_volume_spike,
        rel_strength=rel_strength,
        vwap_bias=vwap_bias,
        daily_df=daily_df,
        trade_df=intraday_df,
        options_liquidity=options_liquidity,
        ema_stack_ok=(intraday_ema_state["Bullish EMA Stack"] if intraday_bias == "Bullish" else intraday_ema_state["Bearish EMA Stack"]),
    )

    swing_score = score_setup(
        bias=swing_bias,
        trade_patterns=daily_patterns,
        higher_patterns=daily_patterns,
        flag_result=daily_flag,
        trend_trade=daily_trend,
        trend_higher=weekly_trend,
        near_support=near_support,
        near_resistance=near_resistance,
        breakout_state=breakout_state,
        has_volume_spike=swing_volume_spike,
        rel_strength=rel_strength,
        vwap_bias=vwap_bias,
        daily_df=daily_df,
        trade_df=daily_df,
        options_liquidity=options_liquidity,
        ema_stack_ok=(daily_ema_state["Bullish EMA Stack"] if swing_bias == "Bullish" else daily_ema_state["Bearish EMA Stack"]),
    )

    current_atr = safe_float(atr(daily_df).iloc[-1])
    suggested_strike = suggest_strike(last_price)
    suggested_expiration = suggest_expiration(10)
    iv_est = iv_proxy(daily_df)

    intraday_entry, intraday_stop, intraday_t1, intraday_t2, intraday_t3 = derive_intraday_trade_plan(
        intraday_df,
        last_price,
        intraday_bias,
        nearest_support,
        nearest_resistance,
        next_supports,
        next_resistances,
        current_atr,
    )

    swing_entry, swing_stop, swing_t1, swing_t2, swing_t3 = derive_swing_trade_plan(
        daily_df,
        last_price,
        swing_bias,
        nearest_support,
        nearest_resistance,
        next_supports,
        next_resistances,
        current_atr,
    )

    result.update(
        {
            "Last Price": round(last_price, 2),
            "ATR %": round(atr_percent(daily_df), 2),
            "RV20": round(rv_20(daily_df), 2),
            "Dollar Volume": round(average_dollar_volume(daily_df), 0),
            "10-Day Move %": round(pct_move(daily_df, 10), 2),
            "Nearest Support": nearest_support.price if nearest_support else np.nan,
            "Nearest Resistance": nearest_resistance.price if nearest_resistance else np.nan,
            "Support Touches": nearest_support.touches if nearest_support else 0,
            "Resistance Touches": nearest_resistance.touches if nearest_resistance else 0,
            "Near Support": near_support,
            "Near Resistance": near_resistance,
            "Breakout State": breakout_state,
            "Intraday Trend": intraday_trend,
            "Daily Trend": daily_trend,
            "Weekly Trend": weekly_trend,
            "Intraday EMA 8": intraday_ema_state["EMA 8"],
            "Intraday EMA 21": intraday_ema_state["EMA 21"],
            "Intraday EMA 50": intraday_ema_state["EMA 50"],
            "Daily EMA 8": daily_ema_state["EMA 8"],
            "Daily EMA 21": daily_ema_state["EMA 21"],
            "Daily EMA 50": daily_ema_state["EMA 50"],
            "Weekly EMA 8": weekly_ema_state["EMA 8"],
            "Weekly EMA 21": weekly_ema_state["EMA 21"],
            "Weekly EMA 50": weekly_ema_state["EMA 50"],
            "VWAP Bias": vwap_bias,
            "Relative Strength vs SPY %": rel_strength,
            "Intraday Patterns": ", ".join(intraday_patterns) if intraday_patterns else "None",
            "Daily Patterns": ", ".join(daily_patterns) if daily_patterns else "None",
            "Intraday Bias": intraday_bias,
            "Swing Bias": swing_bias,
            "Intraday Flag": intraday_flag.name if intraday_flag.name else "None",
            "Swing Flag": daily_flag.name if daily_flag.name else "None",
            "Intraday Setup": intraday_setup_label,
            "Swing Setup": swing_setup_label,
            "Intraday Score": intraday_score,
            "Swing Score": swing_score,
            "Intraday Grade": grade_setup(intraday_score),
            "Swing Grade": grade_setup(swing_score),
            "Intraday Entry": intraday_entry,
            "Intraday Stop": intraday_stop,
            "Intraday Target 1": intraday_t1,
            "Intraday Target 2": intraday_t2,
            "Intraday Target 3": intraday_t3,
            "Swing Entry": swing_entry,
            "Swing Stop": swing_stop,
            "Swing Target 1": swing_t1,
            "Swing Target 2": swing_t2,
            "Swing Target 3": swing_t3,
            "Suggested Strike": suggested_strike,
            "Suggested Expiration": suggested_expiration,
            "IV Proxy %": round(iv_est, 2),
            "Options OI": options_liquidity["Options OI"],
            "Options Volume": options_liquidity["Options Volume"],
            "Spread %": options_liquidity["Spread %"],
            "Calls Liquid": options_liquidity["Calls Liquid"],
            "Puts Liquid": options_liquidity["Puts Liquid"],
            "Support Levels": support_levels,
            "Resistance Levels": resistance_levels,
            "Intraday Pattern List": intraday_patterns,
            "Daily Pattern List": daily_patterns,
            "Intraday Flag Object": intraday_flag,
            "Swing Flag Object": daily_flag,
            "Intraday DF": intraday_df,
            "Daily DF": daily_df,
            "Status": "OK",
        }
    )
    return result


# ============================================================
# CHARTING
# ============================================================
def add_flag_lines(fig: go.Figure, df: pd.DataFrame, flag_result: FlagResult) -> None:
    if not flag_result or not flag_result.upper_line or not flag_result.lower_line:
        return

    lookback = min(20, len(df))
    recent = df.tail(lookback)
    xvals = recent.index.tolist()

    ux1, uy1, ux2, uy2 = flag_result.upper_line
    lx1, ly1, lx2, ly2 = flag_result.lower_line

    if ux2 < len(xvals):
        fig.add_trace(go.Scatter(x=[xvals[ux1], xvals[ux2]], y=[uy1, uy2], mode="lines", name="Flag Upper", line=dict(dash="dash")))
    if lx2 < len(xvals):
        fig.add_trace(go.Scatter(x=[xvals[lx1], xvals[lx2]], y=[ly1, ly2], mode="lines", name="Flag Lower", line=dict(dash="dash")))


def add_pattern_labels(fig: go.Figure, df: pd.DataFrame, patterns: List[str]) -> None:
    if not patterns:
        return
    last_x = df.index[-1]
    last_high = float(df["High"].iloc[-1])
    label_text = " | ".join(patterns)
    fig.add_annotation(x=last_x, y=last_high, text=label_text, showarrow=True, arrowhead=2, yshift=20)


def add_key_levels(fig: go.Figure, supports: List[LevelInfo], resistances: List[LevelInfo]) -> None:
    for level in supports[-5:]:
        fig.add_hline(y=level.price, line_dash="dot", annotation_text=f"S {level.price:.2f} ({level.touches})")
    for level in resistances[:5]:
        fig.add_hline(y=level.price, line_dash="dot", annotation_text=f"R {level.price:.2f} ({level.touches})")


def add_trade_plan(fig: go.Figure, entry: float, stop: float, t1: float, t2: float, t3: float) -> None:
    fig.add_hline(y=entry, line_dash="solid", annotation_text=f"Entry {entry:.2f}")
    fig.add_hline(y=stop, line_dash="dash", annotation_text=f"Stop {stop:.2f}")
    fig.add_hline(y=t1, line_dash="dot", annotation_text=f"T1 {t1:.2f}")
    fig.add_hline(y=t2, line_dash="dot", annotation_text=f"T2 {t2:.2f}")
    fig.add_hline(y=t3, line_dash="dot", annotation_text=f"T3 {t3:.2f}")


def build_chart(
    df: pd.DataFrame,
    symbol: str,
    title: str,
    patterns: List[str],
    flag_result: FlagResult,
    supports: List[LevelInfo],
    resistances: List[LevelInfo],
    entry: float,
    stop: float,
    t1: float,
    t2: float,
    t3: float,
    add_swing_emas: bool = False,
) -> go.Figure:
    plot_df = df.tail(120).copy()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                name=symbol,
            )
        ]
    )

    if add_swing_emas:
        fig.add_trace(go.Scatter(x=plot_df.index, y=ema(plot_df["Close"], 8), mode="lines", name="EMA 8"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=ema(plot_df["Close"], 21), mode="lines", name="EMA 21"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=ema(plot_df["Close"], 50), mode="lines", name="EMA 50"))

    add_pattern_labels(fig, plot_df, patterns)
    add_flag_lines(fig, plot_df, flag_result)
    add_key_levels(fig, supports, resistances)
    add_trade_plan(fig, entry, stop, t1, t2, t3)
    fig.update_layout(
        title=f"{symbol} — {title}",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=760,
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Sharp Screener Settings")
mode = st.sidebar.radio("Focus", ["Both", "Intraday", "Swing"], index=0)
watchlist_size = st.sidebar.slider("Watchlist size", min_value=3, max_value=5, value=5)

preset_universe = st.sidebar.multiselect(
    "Universe",
    options=DEFAULT_UNIVERSE,
    default=["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD", "META", "AMZN", "MSFT"],
)

custom_symbols = st.sidebar.text_input(
    "Add custom tickers (comma-separated)",
    value="",
    help="Example: PLTR, COIN, SMH",
)

intraday_interval = st.sidebar.selectbox("Intraday timeframe", options=["15m", "30m", "60m"], index=0)
min_atr_pct = st.sidebar.slider("Minimum ATR %", min_value=1.0, max_value=8.0, value=2.0, step=0.25)
min_dollar_volume_m = st.sidebar.slider("Minimum avg dollar volume ($M)", min_value=25, max_value=500, value=100, step=25)
minimum_score = st.sidebar.slider("Minimum setup score", min_value=40, max_value=90, value=65, step=5)
run_scan = st.sidebar.button("Run Sharp Scanner", type="primary")


# ============================================================
# MAIN UI
# ============================================================
st.markdown(
    """
### What this version does
- marks **sharp support and resistance** using repeated pivot touches
- gives **tight intraday stops** and **tight swing stops**
- gives **3 target levels** for each trade plan
- adds **8 / 21 / 50 EMA structure** for swing trading
- marks key levels, entry, stop, and all targets directly on the chart
- focuses on liquid stocks and ETFs with tradable options
- suggests ATM strike, swing-friendly expiration, and IV proxy
"""
)

saved_watchlist_df = load_saved_watchlist()

if run_scan:
    all_symbols = set(preset_universe)
    if custom_symbols.strip():
        all_symbols.update([s.strip().upper() for s in custom_symbols.split(",") if s.strip()])
    symbols = sorted(list(all_symbols))

    if not symbols:
        st.warning("Add at least one ticker.")
        st.stop()

    benchmark_daily = fetch_data("SPY", "1d", "6mo")
    progress = st.progress(0, text="Scanning symbols...")
    rows: List[Dict] = []

    for i, symbol in enumerate(symbols, start=1):
        rows.append(analyze_symbol(symbol, benchmark_daily, intraday_interval))
        progress.progress(i / len(symbols), text=f"Scanning {symbol} ({i}/{len(symbols)})")

    df = pd.DataFrame(rows)
    df_ok = df[df["Status"] == "OK"].copy()

    if df_ok.empty:
        st.error("No valid symbols returned.")
        st.dataframe(df)
        st.stop()

    df_ok = df_ok[
        (df_ok["ATR %"] >= min_atr_pct)
        & (df_ok["Dollar Volume"] >= min_dollar_volume_m * 1_000_000)
        & ((df_ok["Intraday Score"] >= minimum_score) | (df_ok["Swing Score"] >= minimum_score))
    ].copy()

    if df_ok.empty:
        st.warning("No symbols passed your filters.")
        st.stop()

    intraday_candidates = df_ok.sort_values(
        by=["Intraday Score", "Relative Strength vs SPY %", "ATR %"],
        ascending=[False, False, False],
    )
    swing_candidates = df_ok.sort_values(
        by=["Swing Score", "Relative Strength vs SPY %", "ATR %"],
        ascending=[False, False, False],
    )

    st.subheader("Scanner Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe scanned", len(symbols))
    c2.metric("Passed filters", len(df_ok))
    c3.metric("Min ATR %", f"{min_atr_pct:.2f}%")
    c4.metric("Min score", minimum_score)

    new_alert_rows: List[Dict] = []

    if mode in ["Both", "Intraday"]:
        st.subheader(f"🔥 Intraday Watchlist ({watchlist_size} max)")
        intraday_watchlist = intraday_candidates.head(watchlist_size).copy()
        st.dataframe(
            intraday_watchlist[
                [
                    "Symbol", "Last Price", "Intraday Grade", "Intraday Score", "Intraday Setup",
                    "Intraday Bias", "Intraday Trend", "Daily Trend",
                    "Intraday EMA 8", "Intraday EMA 21", "Intraday EMA 50",
                    "Nearest Support", "Nearest Resistance",
                    "Suggested Strike", "Suggested Expiration", "IV Proxy %",
                    "Intraday Entry", "Intraday Stop", "Intraday Target 1", "Intraday Target 2", "Intraday Target 3",
                ]
            ],
            use_container_width=True,
        )

        for _, row in intraday_watchlist.iterrows():
            fresh = is_fresh_setup(row["Intraday Pattern List"], row["Intraday Flag"], minimum_score, row["Intraday Score"])
            with st.container(border=True):
                st.markdown(f"**{row['Symbol']}** — Grade: **{row['Intraday Grade']}** | Score: **{row['Intraday Score']}**")
                st.write(
                    f"Setup: {row['Intraday Setup']} | Bias: {row['Intraday Bias']} | Intraday trend: {row['Intraday Trend']} | Daily trend: {row['Daily Trend']}"
                )
                st.write(
                    f"EMA 8/21/50: {row['Intraday EMA 8']} / {row['Intraday EMA 21']} / {row['Intraday EMA 50']}"
                )
                st.write(
                    f"Nearest support: {row['Nearest Support']} | Nearest resistance: {row['Nearest Resistance']} | Breakout state: {row['Breakout State']}"
                )
                st.write(
                    f"Strike: {row['Suggested Strike']} | Exp: {row['Suggested Expiration']} | IV Proxy: {row['IV Proxy %']}% | Entry: {row['Intraday Entry']} | Stop: {row['Intraday Stop']} | T1: {row['Intraday Target 1']} | T2: {row['Intraday Target 2']} | T3: {row['Intraday Target 3']}"
                )
                if fresh:
                    new_alert_rows.append(
                        {
                            "Symbol": row["Symbol"],
                            "Setup Type": "Intraday",
                            "Score": row["Intraday Score"],
                            "Bias": row["Intraday Bias"],
                            "Setup Label": row["Intraday Setup"],
                            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

    if mode in ["Both", "Swing"]:
        st.subheader(f"📌 Swing Watchlist ({watchlist_size} max)")
        swing_watchlist = swing_candidates.head(watchlist_size).copy()
        st.dataframe(
            swing_watchlist[
                [
                    "Symbol", "Last Price", "Swing Grade", "Swing Score", "Swing Setup",
                    "Swing Bias", "Daily Trend", "Weekly Trend",
                    "Daily EMA 8", "Daily EMA 21", "Daily EMA 50",
                    "Nearest Support", "Nearest Resistance",
                    "Suggested Strike", "Suggested Expiration", "IV Proxy %",
                    "Swing Entry", "Swing Stop", "Swing Target 1", "Swing Target 2", "Swing Target 3",
                ]
            ],
            use_container_width=True,
        )

        for _, row in swing_watchlist.iterrows():
            fresh = is_fresh_setup(row["Daily Pattern List"], row["Swing Flag"], minimum_score, row["Swing Score"])
            with st.container(border=True):
                st.markdown(f"**{row['Symbol']}** — Grade: **{row['Swing Grade']}** | Score: **{row['Swing Score']}**")
                st.write(
                    f"Setup: {row['Swing Setup']} | Bias: {row['Swing Bias']} | Daily trend: {row['Daily Trend']} | Weekly trend: {row['Weekly Trend']}"
                )
                st.write(
                    f"EMA 8/21/50: {row['Daily EMA 8']} / {row['Daily EMA 21']} / {row['Daily EMA 50']}"
                )
                st.write(
                    f"Nearest support: {row['Nearest Support']} | Nearest resistance: {row['Nearest Resistance']} | Breakout state: {row['Breakout State']}"
                )
                st.write(
                    f"Strike: {row['Suggested Strike']} | Exp: {row['Suggested Expiration']} | IV Proxy: {row['IV Proxy %']}% | Entry: {row['Swing Entry']} | Stop: {row['Swing Stop']} | T1: {row['Swing Target 1']} | T2: {row['Swing Target 2']} | T3: {row['Swing Target 3']}"
                )
                if fresh:
                    new_alert_rows.append(
                        {
                            "Symbol": row["Symbol"],
                            "Setup Type": "Swing",
                            "Score": row["Swing Score"],
                            "Bias": row["Swing Bias"],
                            "Setup Label": row["Swing Setup"],
                            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

    if new_alert_rows:
        save_watchlist_rows(new_alert_rows)
        saved_watchlist_df = load_saved_watchlist()

    st.subheader("Saved Watchlist")
    if saved_watchlist_df.empty:
        st.info("No saved alerts yet.")
    else:
        st.dataframe(saved_watchlist_df.sort_values(by="Timestamp", ascending=False), use_container_width=True)

    st.subheader("Chart View")
    chart_symbol = st.selectbox("Choose a symbol to chart", options=df_ok["Symbol"].tolist())
    chart_mode = st.radio("Chart Mode", ["Intraday", "Swing"], horizontal=True)
    chart_row = df_ok[df_ok["Symbol"] == chart_symbol].iloc[0]

    if chart_mode == "Intraday":
        fig = build_chart(
            chart_row["Intraday DF"],
            chart_symbol,
            f"{intraday_interval} chart",
            chart_row["Intraday Pattern List"],
            chart_row["Intraday Flag Object"],
            chart_row["Support Levels"],
            chart_row["Resistance Levels"],
            chart_row["Intraday Entry"],
            chart_row["Intraday Stop"],
            chart_row["Intraday Target 1"],
            chart_row["Intraday Target 2"],
            chart_row["Intraday Target 3"],
            False,
        )
    else:
        fig = build_chart(
            chart_row["Daily DF"],
            chart_symbol,
            "Daily swing chart",
            chart_row["Daily Pattern List"],
            chart_row["Swing Flag Object"],
            chart_row["Support Levels"],
            chart_row["Resistance Levels"],
            chart_row["Swing Entry"],
            chart_row["Swing Stop"],
            chart_row["Swing Target 1"],
            chart_row["Swing Target 2"],
            chart_row["Swing Target 3"],
            True,
        )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Scanner Output")
    display_df = df_ok.drop(
        columns=[
            "Support Levels", "Resistance Levels", "Intraday Pattern List", "Daily Pattern List",
            "Intraday Flag Object", "Swing Flag Object", "Intraday DF", "Daily DF"
        ],
        errors="ignore",
    )
    st.dataframe(display_df.sort_values(by=["Swing Score", "Intraday Score"], ascending=False), use_container_width=True)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", data=csv, file_name="sharp_key_level_options_screener.csv", mime="text/csv")

else:
    st.info("Set your universe and click **Run Sharp Scanner**.")
    st.subheader("Saved Watchlist")
    if saved_watchlist_df.empty:
        st.info("No saved alerts yet.")
    else:
        st.dataframe(saved_watchlist_df.sort_values(by="Timestamp", ascending=False), use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
)
