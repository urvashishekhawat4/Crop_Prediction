"""
data_loader.py
Loads crop_dataset_clean.csv and msp_dataset_clean.csv,
returns preprocessed DataFrames and encoded label mappings.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, "data")

CROP_CSV  = os.path.join(DATA_DIR, "crop_dataset_clean.csv")
MSP_CSV   = os.path.join(DATA_DIR, "msp_dataset_clean.csv")


def load_crop_df() -> pd.DataFrame:
    df = pd.read_csv(CROP_CSV)
    df.fillna({
        "area":       df["area"].median(),
        "yield":      df["yield"].median(),
        "production": df["production"].median()
    }, inplace=True)
    return df


def load_msp_df() -> pd.DataFrame:
    df = pd.read_csv(MSP_CSV)
    return df


def build_feature_matrix(crop_df: pd.DataFrame):
    """
    Returns (X_scaled, y, le_crop, le_state, scaler, feature_names).
    Features: area | yield | year | crop_enc | state_enc
    Target  : production (tonnes)
    """
    df = crop_df.copy()

    le_crop  = LabelEncoder()
    le_state = LabelEncoder()
    df["crop_enc"]  = le_crop.fit_transform(df["crop"].astype(str))
    df["state_enc"] = le_state.fit_transform(df["state"].astype(str))

    features = ["area", "yield", "year", "crop_enc", "state_enc"]
    X = df[features].values.astype(float)
    y = df["production"].values.astype(float)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    return X_sc, y, le_crop, le_state, scaler, features


def get_summary_stats(crop_df: pd.DataFrame) -> dict:
    """Returns key statistics for the dashboard overview cards."""
    return {
        "total_rows":   int(len(crop_df)),
        "num_crops":    int(crop_df["crop"].nunique()),
        "num_states":   int(crop_df["state"].nunique()),
        "year_min":     int(crop_df["year"].min()),
        "year_max":     int(crop_df["year"].max()),
        "crops":        sorted(crop_df["crop"].unique().tolist()),
        "states":       sorted(crop_df["state"].unique().tolist()),
    }


def get_production_trend(crop_df: pd.DataFrame) -> dict:
    """Aggregated national production per crop per year."""
    agg = (crop_df
           .groupby(["year", "crop"])["production"]
           .sum()
           .reset_index())
    result = {}
    for crop, grp in agg.groupby("crop"):
        result[crop] = {
            "years":  grp["year"].tolist(),
            "values": (grp["production"] / 1e6).round(3).tolist()   # M Tonnes
        }
    return result


def get_area_trend(crop_df: pd.DataFrame) -> dict:
    agg = (crop_df
           .groupby(["year", "crop"])["area"]
           .sum()
           .reset_index())
    result = {}
    for crop, grp in agg.groupby("crop"):
        result[crop] = {
            "years":  grp["year"].tolist(),
            "values": (grp["area"] / 1e6).round(3).tolist()         # M Hectares
        }
    return result


def get_yield_trend(crop_df: pd.DataFrame) -> dict:
    agg = (crop_df
           .groupby(["year", "crop"])["yield"]
           .mean()
           .reset_index())
    result = {}
    for crop, grp in agg.groupby("crop"):
        result[crop] = {
            "years":  grp["year"].tolist(),
            "values": grp["yield"].round(3).tolist()                 # T/Ha
        }
    return result


def get_state_heatmap(crop_df: pd.DataFrame, crop: str, top_n: int = 12) -> dict:
    """Returns pivot table data for state × year heatmap."""
    sub = crop_df[crop_df["crop"] == crop.lower()]
    piv = sub.pivot_table(index="state", columns="year",
                          values="production", aggfunc="sum")
    top_states = piv.mean(axis=1).nlargest(top_n).index.tolist()
    piv = piv.loc[top_states].fillna(0)
    return {
        "states": piv.index.tolist(),
        "years":  [int(c) for c in piv.columns.tolist()],
        "matrix": (piv.values / 1e6).round(3).tolist()              # M Tonnes
    }


def get_msp_trend(msp_df: pd.DataFrame) -> dict:
    sub = (msp_df[~msp_df["anticipated"]]
           .drop_duplicates(["year", "crop"])
           [["year", "crop", "msp"]]
           .dropna()
           .sort_values("year"))
    result = {}
    for crop, grp in sub.groupby("crop"):
        result[crop] = {
            "years":  grp["year"].tolist(),
            "values": grp["msp"].tolist()
        }
    return result


def get_msp_district_data(msp_df: pd.DataFrame, crop: str) -> dict:
    """District-wise production across years for a single crop."""
    sub = (msp_df[(msp_df["crop"] == crop.lower()) & (~msp_df["anticipated"])]
           .dropna(subset=["production"]))
    piv = sub.pivot_table(index="revenue_district", columns="year",
                          values="production", aggfunc="sum").fillna(0)
    return {
        "districts": piv.index.tolist(),
        "years":     piv.columns.tolist(),
        "matrix":    piv.values.round(2).tolist()
    }


def get_top_states(crop_df: pd.DataFrame, top_n: int = 10) -> dict:
    """Top states by cumulative production for each crop."""
    result = {}
    for crop, grp in crop_df.groupby("crop"):
        top = (grp.groupby("state")["production"]
               .sum()
               .nlargest(top_n)
               .reset_index())
        result[crop] = {
            "states": top["state"].tolist(),
            "values": (top["production"] / 1e9).round(3).tolist()   # B Tonnes
        }
    return result


def get_yoy_growth(crop_df: pd.DataFrame) -> dict:
    """Year-on-year % change in total national production per crop."""
    agg = crop_df.groupby(["year", "crop"])["production"].sum().reset_index()
    result = {}
    for crop, grp in agg.groupby("crop"):
        grp  = grp.set_index("year")["production"].sort_index()
        pct  = grp.pct_change().dropna() * 100
        result[crop] = {
            "years":  pct.index.tolist(),
            "values": pct.round(2).tolist()
        }
    return result
