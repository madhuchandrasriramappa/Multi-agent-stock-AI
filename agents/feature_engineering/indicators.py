"""
Pure technical indicator functions.

Every function takes a pandas Series (or DataFrame for VWAP) and returns
a Series of the same length.  All are stateless — no DB access, no side
effects.  This makes them fast to unit-test and easy to swap out.

NaN policy
----------
Early rows that don't have enough history for a window are returned as NaN.
min_periods is always set equal to the window so partial windows produce NaN
rather than misleading partial averages.  Downstream code should treat NaN
as "not enough data yet" and skip those rows.
"""
from __future__ import annotations

import pandas as pd


# ── Moving averages ────────────────────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average over `window` periods."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average with smoothing factor 2/(span+1).
    EMA does not use min_periods — it starts computing from row 0 using
    the first value as the seed, which is standard financial practice.
    """
    return series.ewm(span=span, adjust=False).mean()


# ── Oscillators ────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothed method).

    Returns values in [0, 100].
    RSI > 70  → overbought (potential sell signal)
    RSI < 30  → oversold  (potential buy signal)
    """
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()

    # Guard against division by zero (all gains, no losses → RSI = 100)
    rs = gain / loss.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Returns (macd_line, signal_line, histogram).
      macd_line  = EMA(fast) − EMA(slow)
      signal     = EMA(macd_line, signal)
      histogram  = macd_line − signal
    """
    ema_fast   = ema(series, fast)
    ema_slow   = ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Bollinger Bands ────────────────────────────────────────────────────────────

def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: upper, middle (SMA), lower.

    Bands widen during high volatility and contract during low volatility.
    Price touching the upper band → overbought territory.
    Price touching the lower band → oversold territory.
    """
    middle = series.rolling(window=window, min_periods=window).mean()
    std    = series.rolling(window=window, min_periods=window).std()
    upper  = middle + (num_std * std)
    lower  = middle - (num_std * std)
    return upper, middle, lower


# ── Volatility ─────────────────────────────────────────────────────────────────

def rolling_volatility(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Rolling standard deviation of period-over-period returns.
    Returned as a decimal (0.02 = 2% average price swing per candle).
    """
    returns = series.pct_change()
    return returns.rolling(window=window, min_periods=window).std()


# ── VWAP ───────────────────────────────────────────────────────────────────────

def vwap_daily(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price, resetting at the start of each UTC day.

    VWAP = cumulative(typical_price × volume) / cumulative(volume)
    where typical_price = (high + low + close) / 3

    Requires a DataFrame with columns: timestamp, high, low, close, volume.
    The timestamp column must be UTC-aware (or UTC-naive treated as UTC).
    Returns a Series aligned to df's index.
    """
    df = df.copy()
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    df["_date"] = timestamps.dt.date

    result = pd.Series(index=df.index, dtype=float)

    for _date, idx in df.groupby("_date", sort=True).groups.items():
        grp           = df.loc[idx]
        typical_price = (grp["high"] + grp["low"] + grp["close"]) / 3.0
        cumtp_vol     = (typical_price * grp["volume"]).cumsum()
        cumvol        = grp["volume"].cumsum()
        result.loc[idx] = (cumtp_vol / cumvol.replace(0, float("nan"))).values

    return result
