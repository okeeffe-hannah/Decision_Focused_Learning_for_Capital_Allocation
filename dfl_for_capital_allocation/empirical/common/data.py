"""Empirical setting data loading used by pto/dfl/bb models

Description: Contains all code common to empirical pipelines focusing on data loading,
`simulation/common/` instead contains Basel/Vasicek math and model/training code shared between settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def annual_chargeoff_to_quarterly(c_annual):
    """Invert FRED's annualisation of quarterly net charge-off rate."""
    c_annual = np.asarray(c_annual, dtype=float)
    return c_annual / 4.0


def load_quarterly_pd_panel(macro_csv, chargeoff_csv=None, dcfg=None, *, date_col: str = "DATE",
                             target_col: str = "charge_off_rate",
                             feature_cols: Optional[Sequence[str]] = None, horizon: int = 1,
                             charge_off_is_percent: bool = True,
                             include_current_cq: bool = True) -> pd.DataFrame:
    """Load, align, and construct the empirical quarterly PD panel.

    The forecasting row is `x_t -> PD_Q_{t+1}`; result columns are named for the
    realised next-period outcome, because that is the period used for decision
    evaluation. Requires the shared decision math (`clip_pd_q`, `quarterly_pd_to_annual`)
    via `dcfg`, so realised PD/annualised-PD columns use exactly the same formulas and
    calibration as the rest of the pipeline.

    Returns: DataFrame with `df.attrs["feature_cols"]` as feature list.
    """
    from common import decision as dec  # study/common - already on sys.path by the caller

    # 01_build_macro_panel normally writes the aligned charge-off columns into
    # the processed macro panel.  Keep the optional second input for compatibility
    # with older panels, but use the prepared single-file contract by default.
    macro = pd.read_csv(macro_csv)
    if chargeoff_csv is None:
        required_prepared = {"charge_off_rate", "c_q"}
        missing_prepared = required_prepared.difference(macro.columns)
        if missing_prepared:
            raise ValueError(f"Prepared macro panel is missing charge-off columns: {sorted(missing_prepared)}")
        charge = macro[[date_col, "charge_off_rate", "c_q"]].copy()
        charge["c_annual_raw"] = pd.to_numeric(charge["charge_off_rate"], errors="coerce")
        charge["c_annual_decimal"] = charge["c_annual_raw"] / 100.0
        macro = macro.drop(columns=["charge_off_rate", "c_q", "c_q_lag1"], errors="ignore")
    else:
        charge = pd.read_csv(chargeoff_csv)
    macro[date_col] = pd.to_datetime(macro[date_col]).dt.to_period("Q").dt.start_time
    charge[date_col] = pd.to_datetime(charge[date_col]).dt.to_period("Q").dt.start_time

    if chargeoff_csv is not None and target_col not in charge.columns:
        raise ValueError(f"{target_col!r} not found in charge-off columns: {list(charge.columns)}")

    if chargeoff_csv is not None:
        charge["c_annual_raw"] = pd.to_numeric(charge[target_col], errors="coerce")
        charge["c_annual_decimal"] = charge["c_annual_raw"] / 100.0 if charge_off_is_percent else charge["c_annual_raw"]
        charge["c_q"] = annual_chargeoff_to_quarterly(charge["c_annual_decimal"].to_numpy())
    charge["pd_q"] = dec.clip_pd_q(charge["c_q"].to_numpy() / dcfg.LGD, dcfg)
    charge["pd_a"] = dec.quarterly_pd_to_annual(charge["pd_q"].to_numpy(), dcfg)

    keep_charge = [date_col, "c_annual_raw", "c_annual_decimal", "c_q", "pd_q", "pd_a"]
    df = pd.merge(macro, charge[keep_charge], on=date_col, how="inner").sort_values(date_col).reset_index(drop=True)

    # Observed autoregressive signals. PD columns are never features.
    df["c_q_lag1"] = df["c_q"].shift(1)

    df["target_date"] = df[date_col].shift(-horizon)
    df["c_annual_raw_next"] = df["c_annual_raw"].shift(-horizon)
    df["c_q_realised"] = df["c_q"].shift(-horizon)
    df["pd_q_realised"] = df["pd_q"].shift(-horizon)
    df["pd_a_realised"] = df["pd_a"].shift(-horizon)
    df["target_pd_q_next"] = df["pd_q_realised"]

    if feature_cols is None:
        exclude = {date_col, "c_annual_raw", "c_annual_decimal", "pd_q", "pd_a", "target_date",
                   "c_annual_raw_next", "c_q_realised", "pd_q_realised", "pd_a_realised", "target_pd_q_next"}
        if not include_current_cq:
            exclude.add("c_q")
        feature_cols = [c for c in df.columns if c not in exclude]
    else:
        feature_cols = list(feature_cols)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    required = list(feature_cols) + ["c_q", "target_date", "c_annual_raw_next", "c_q_realised",
                                      "pd_q_realised", "pd_a_realised", "target_pd_q_next"]
    required = list(dict.fromkeys(required))
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} usable observations after alignment; check dates and missing values.")
    df.attrs["feature_cols"] = feature_cols
    return df


def write_panel_metadata(path, dcfg, *, feature_cols: List[str], n_observations: int,
                          date_range: Sequence[str], charge_off_is_percent: bool,
                          include_current_cq: bool,
                          pd_clip_low: float, pd_clip_high: float,
                          description: Optional[str] = None) -> dict:
    """Write decision/Basel constants and realised feature set to a metadata JSON, used by
    03_black_box and 04_dfl to ensure all models use same decision setup.
    
    Written in 02_pto.

    Returns: Dictionary containing metadata."""
    metadata = {
        "description": description or (
            "Empirical quarterly PD panel (FRED macro features + net charge-off rate). "
            "Realised quarterly PD is c_q_realised / LGD; annual PD for Basel IRB is the "
            "compounded quarterly rate. There is no latent 'true' PD process to condition "
            "on (unlike the simulated panel), so only a foresight oracle is defined here - "
            "there is no Bayes oracle for the empirical panel."
        ),
        "constants": {
            "LGD": dcfg.LGD, "M": dcfg.M, "K1": dcfg.K1, "ell": dcfg.ell, "Y_BAR": dcfg.Y_BAR,
            "lambda_shortfall": dcfg.lambda_shortfall, "RHO_SCALE": dcfg.RHO_SCALE,
            "APPLY_PD_FLOOR": dcfg.APPLY_PD_FLOOR, "PD_FLOOR": dcfg.PD_FLOOR,
            "allocation_optimisation": "continuous",
            "alpha_bounds_min": float(dcfg.ALPHA_BOUNDS[0]), "alpha_bounds_max": float(dcfg.ALPHA_BOUNDS[1]),
            "alpha_opt_xatol": float(dcfg.ALPHA_OPT_XATOL),
            "PD_CLIP_LOW": pd_clip_low, "PD_CLIP_HIGH": pd_clip_high,
            "CHARGE_OFF_IS_PERCENT": charge_off_is_percent,
            "INCLUDE_CURRENT_CQ": include_current_cq,
        },
        "feature_cols": list(feature_cols),
        "n_observations": int(n_observations),
        "date_range": list(date_range),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def read_panel_metadata(path) -> dict:
    """Read back metadata JSON written by `write_panel_metadata`"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
