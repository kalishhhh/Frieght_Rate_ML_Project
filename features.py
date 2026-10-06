"""Cleaning + feature engineering shared by training, validation and prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]
TARGET = "posted_rate"


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame, reference_dates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fix data-quality problems. Uses only information available at prediction time.

    * weight: 300 missing, ~290 negative (sign flips, e.g. -47500) -> abs(); missing kept as NaN
      (tree models handle it natively) plus an explicit flag.
    * market_index: ~0.8% missing. It is a *market-level* signal (within-day std ~0.025 vs a
      0.17 overall std), so a missing value is filled with the median of the same calendar day
      across every row we have for that day (train+validation features, never the target).
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs()
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    pool = out[["date", "market_index"]]
    if reference_dates is not None:
        pool = pd.concat([pool, reference_dates[["date", "market_index"]]])
    day_median = pool.groupby("date")["market_index"].median()
    out["market_index"] = out["market_index"].fillna(out["date"].map(day_median))
    # a day with no market_index at all -> nearest known day
    if out["market_index"].isna().any():
        full = day_median.sort_index().interpolate(limit_direction="both")
        out["market_index"] = out["market_index"].fillna(out["date"].map(full))
    return out


def build_features(df: pd.DataFrame, drop_quote_signal: bool = False) -> pd.DataFrame:
    """Numeric feature matrix. Geography (coords) is used instead of city IDs so that
    cities never seen in training (8 of them appear in validation) are still handled."""
    x = pd.DataFrame(index=df.index)
    x["distance"] = df["distance"]
    x["log_distance"] = np.log(df["distance"])
    x["inv_distance"] = 1.0 / df["distance"]
    hav = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    x["hav_dist"] = hav
    x["circuity"] = df["distance"] / hav
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        x[c] = df[c]
    x["d_lat"] = df.delivery_lat - df.pickup_lat
    x["d_lon"] = df.delivery_lon - df.pickup_lon
    for e in EQUIPMENT:
        x[f"eq_{e.replace(' ', '_')}"] = (df["equipment"] == e).astype(int)
    x["equipment_code"] = df["equipment"].map({e: i for i, e in enumerate(EQUIPMENT)})
    x["weight"] = df["weight"]
    x["weight_missing"] = df["weight_missing"]
    x["market_index"] = df["market_index"]
    x["market_index_missing"] = df["market_index_missing"]
    if not drop_quote_signal:
        x["quote_signal"] = df["quote_signal"]
    # No calendar features (dow / day-of-month / month): CV showed they add only noise, and
    # month / day-of-year cannot extrapolate to December, which is absent from training.
    return x


START = pd.Timestamp("2025-01-01")


def add_trend_features(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Linear time trend (rates drift ~+6% over Jan-Oct that market_index does not explain) and
    interactions the linear stage needs (quote_signal acts differently on short vs long hauls)."""
    x = x.copy()
    x["t"] = (df["date"] - START).dt.days / 300.0
    ld = x["log_distance"]
    x["ld2"] = ld**2
    x["ld3"] = ld**3
    x["short_haul"] = np.maximum(0, np.log(300) - ld)  # hinge: extra premium on short hauls
    x["qs_x_ld"] = df["quote_signal"] * ld if "quote_signal" in df else 0
    x["qs_dev2"] = (df["quote_signal"] - 2.05) ** 2 if "quote_signal" in df else 0
    x["mi_x_ld"] = x["market_index"] * ld
    x["w_x_ld"] = np.log(x["weight"].fillna(31000)) * ld
    x["w_cap"] = (df["weight"] >= 47000).astype(int)
    for e in EQUIPMENT:
        k = e.replace(" ", "_")
        x[f"eq_{k}_x_ld"] = x[f"eq_{k}"] * ld
    return x
