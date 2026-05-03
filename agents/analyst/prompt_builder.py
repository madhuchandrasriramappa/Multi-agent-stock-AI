"""
Pure prompt-building functions for the AI Analyst Agent.

No DB access, no side effects — stateless and easy to unit-test.
"""
from __future__ import annotations

from datetime import datetime


def build_system_prompt() -> str:
    return (
        "You are a quantitative financial analyst specialising in technical analysis "
        "for stocks and cryptocurrencies. You receive pre-computed OHLCV indicators "
        "and statistical anomaly alerts. Your analysis must be data-driven, concise "
        "(3–5 sentences), and professional. Always end your response with exactly one "
        "of: OUTLOOK: BULLISH, OUTLOOK: BEARISH, or OUTLOOK: NEUTRAL."
    )


def build_user_prompt(
    symbol: str,
    asset_type: str,
    latest: dict,
    alerts: list[dict],
    as_of: datetime | None = None,
) -> str:
    """
    Build the user-turn prompt from feature data and anomaly alerts.

    Parameters
    ----------
    symbol      : ticker / coin id
    asset_type  : 'stock' or 'crypto'
    latest      : dict from FeatureSet row (close, rsi_14, macd_histogram, …)
    alerts      : list of dicts from AnomalyAlert rows (up to 10 shown)
    as_of       : timestamp of the latest data point (for context)
    """
    lines: list[str] = []

    ts = (as_of or latest.get("timestamp", "unknown"))
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"## {symbol} ({asset_type.upper()}) — as of {ts}\n")

    lines.append("### Latest Technical Indicators")
    lines.append(f"- Close price    : {_fmt(latest.get('close'), 4)}")
    lines.append(f"- RSI-14         : {_fmt(latest.get('rsi_14'), 2)}  (oversold <30, overbought >70)")
    lines.append(f"- MACD histogram : {_fmt(latest.get('macd_histogram'), 6)}")
    lines.append(f"- Volatility-14  : {_fmt(latest.get('volatility_14'), 4)}  (rolling std of returns)")
    lines.append(f"- VWAP           : {_fmt(latest.get('vwap'), 4)}")
    lines.append(f"- SMA-20         : {_fmt(latest.get('sma_20'), 4)}")
    lines.append(f"- EMA-12         : {_fmt(latest.get('ema_12'), 4)}")
    lines.append("")

    shown = alerts[:10]
    if shown:
        lines.append(f"### Anomaly Alerts ({len(alerts)} recent, showing {len(shown)})")
        for a in shown:
            fv = f", value={_fmt(a.get('feature_value'), 4)}" if a.get("feature_value") else ""
            lines.append(
                f"- [{a['severity'].upper():6s}] {a['detector']:20s} "
                f"on {a['feature']:20s} score={_fmt(a.get('score'), 4)}{fv}"
            )
    else:
        lines.append("### Anomaly Alerts\n- None detected in the recent window.")
    lines.append("")

    lines.append(
        "Analyse the data above. In 3–5 sentences explain what the indicators and "
        "alerts suggest about current market conditions and the key risk factors. "
        "End with exactly one of: OUTLOOK: BULLISH, OUTLOOK: BEARISH, or OUTLOOK: NEUTRAL."
    )

    return "\n".join(lines)


def extract_outlook(report_text: str) -> str:
    """
    Parse the OUTLOOK line from a model response.
    Returns 'BULLISH', 'BEARISH', 'NEUTRAL', or 'UNKNOWN' if not found.
    """
    upper = report_text.upper()
    for label in ("BULLISH", "BEARISH", "NEUTRAL"):
        if f"OUTLOOK: {label}" in upper:
            return label
    return "UNKNOWN"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(value, decimals: int = 4, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return default
