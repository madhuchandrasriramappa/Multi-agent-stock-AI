"""
Pure anomaly detection functions.

Each function accepts a DataFrame that already has non-null feature values
and returns a DataFrame of alert rows with columns:
    symbol, asset_type, timestamp, interval,
    detector, feature, feature_value, score, severity

No DB access, no side effects — stateless and straightforward to unit-test.

Groups with fewer than 10 rows (univariate) or 20 rows (multivariate) are
skipped to avoid false positives from insufficient history.
"""
from __future__ import annotations

import math

import pandas as pd

try:
    from sklearn.ensemble import IsolationForest as _IsolationForest
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

_ALERT_COLS = [
    "symbol", "asset_type", "timestamp", "interval",
    "detector", "feature", "feature_value", "score", "severity",
]

_MIN_ROWS_UNIVARIATE   = 10
_MIN_ROWS_MULTIVARIATE = 20


# ── Severity helpers ───────────────────────────────────────────────────────────

def _zscore_severity(z: float) -> str:
    if z >= 5.0:
        return "high"
    if z >= 4.0:
        return "medium"
    return "low"


def _iqr_severity(distance: float) -> str:
    if distance >= 3.0:
        return "high"
    if distance >= 2.0:
        return "medium"
    return "low"


def _if_severity(score: float) -> str:
    """IF score_samples: lower = more anomalous (returns negative values)."""
    if score <= -0.3:
        return "high"
    if score <= -0.2:
        return "medium"
    return "low"


# ── Z-score detector ───────────────────────────────────────────────────────────

def zscore_anomalies(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Flag rows where |z-score| > threshold for any of the given features.
    Statistics are computed per (symbol, interval) group.
    Groups with < 10 rows or zero std are skipped.
    """
    alerts: list[dict] = []

    for (symbol, interval), grp in df.groupby(["symbol", "interval"], sort=False):
        for feat in features:
            if feat not in grp.columns:
                continue
            col = grp[feat].dropna()
            if len(col) < _MIN_ROWS_UNIVARIATE:
                continue
            std = col.std()
            if std < 1e-10:
                continue
            mean = col.mean()
            z = (grp[feat] - mean) / std
            for idx in z.index[z.abs() > threshold]:
                z_val = abs(float(z.loc[idx]))
                row = grp.loc[idx]
                alerts.append({
                    "symbol":        symbol,
                    "asset_type":    row["asset_type"],
                    "timestamp":     row["timestamp"],
                    "interval":      interval,
                    "detector":      "zscore",
                    "feature":       feat,
                    "feature_value": _safe_float(row[feat]),
                    "score":         round(z_val, 6),
                    "severity":      _zscore_severity(z_val),
                })

    return pd.DataFrame(alerts, columns=_ALERT_COLS) if alerts else _empty_alerts()


# ── IQR detector ──────────────────────────────────────────────────────────────

def iqr_anomalies(
    df: pd.DataFrame,
    features: list[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Flag rows where a feature lies outside [Q1 − mult*IQR, Q3 + mult*IQR].
    Groups with < 10 rows or IQR ≈ 0 are skipped.
    """
    alerts: list[dict] = []

    for (symbol, interval), grp in df.groupby(["symbol", "interval"], sort=False):
        for feat in features:
            if feat not in grp.columns:
                continue
            col = grp[feat].dropna()
            if len(col) < _MIN_ROWS_UNIVARIATE:
                continue
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1
            if iqr < 1e-10:
                continue
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            flagged = grp.index[(grp[feat] < lower) | (grp[feat] > upper)]
            for idx in flagged:
                row = grp.loc[idx]
                val = float(row[feat])
                distance = max(val - upper, lower - val) / iqr
                alerts.append({
                    "symbol":        symbol,
                    "asset_type":    row["asset_type"],
                    "timestamp":     row["timestamp"],
                    "interval":      interval,
                    "detector":      "iqr",
                    "feature":       feat,
                    "feature_value": _safe_float(row[feat]),
                    "score":         round(distance, 6),
                    "severity":      _iqr_severity(distance),
                })

    return pd.DataFrame(alerts, columns=_ALERT_COLS) if alerts else _empty_alerts()


# ── Isolation Forest detector ─────────────────────────────────────────────────

def isolation_forest_anomalies(
    df: pd.DataFrame,
    features: list[str],
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Multivariate anomaly detection using sklearn IsolationForest.
    Operates on the intersection of available feature columns.
    Groups with < 20 complete rows are skipped.
    Returns empty DataFrame if scikit-learn is not installed.
    """
    if not _SKLEARN_AVAILABLE:
        return _empty_alerts()

    alerts: list[dict] = []

    for (symbol, interval), grp in df.groupby(["symbol", "interval"], sort=False):
        avail = [f for f in features if f in grp.columns]
        if not avail:
            continue
        sub = grp[avail].dropna()
        if len(sub) < _MIN_ROWS_MULTIVARIATE:
            continue

        clf = _IsolationForest(contamination=contamination, random_state=42, n_jobs=1)
        clf.fit(sub.values)
        preds  = clf.predict(sub.values)        # 1 = normal, -1 = anomaly
        scores = clf.score_samples(sub.values)  # lower = more anomalous

        for i, idx in enumerate(sub.index):
            if preds[i] == -1:
                row = grp.loc[idx]
                alerts.append({
                    "symbol":        symbol,
                    "asset_type":    row["asset_type"],
                    "timestamp":     row["timestamp"],
                    "interval":      interval,
                    "detector":      "isolation_forest",
                    "feature":       "multivariate",
                    "feature_value": float("nan"),
                    "score":         round(float(scores[i]), 6),
                    "severity":      _if_severity(float(scores[i])),
                })

    return pd.DataFrame(alerts, columns=_ALERT_COLS) if alerts else _empty_alerts()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _empty_alerts() -> pd.DataFrame:
    return pd.DataFrame(columns=_ALERT_COLS)


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
