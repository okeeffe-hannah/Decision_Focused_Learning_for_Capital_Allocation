"""
capital_allocation_utils.py

Shared utilities for the MSc capital allocation project.

This file contains the pieces that are common to PTO, black-box and later DFL
experiments:
  - data loading/alignment
  - Basel-style IRB capital requirement
  - realised capital-allocation utility
  - oracle allocation and regret

Keep this file model-agnostic. Model-specific training loops should live in
separate scripts such as pto_baselines.py, blackbox_baselines.py and later
 dfl_baselines.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm
except ImportError as exc:  # pragma: no cover
    raise ImportError("This project requires scipy. Install with: pip install scipy") from exc


@dataclass
class ModelConfig:
    """Generic neural-network training configuration."""

    hidden_dim: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 1500
    patience: int = 100
    val_frac: float = 0.2
    seed: int = 42


@dataclass
class CapitalConfig:
    """Capital-allocation and Gaussian-mixture constants."""

    # Basel / objective constants
    lgd: float = 0.45
    y_bar: float = 0.05
    lambda_penalty: float = 10.0
    maturity: float = 2.5
    leverage: float = 1.0

    # Charge-off Gaussian mixture parameters, expressed as decimals.
    # Defaults match the interim-report table: 0.61%, 0.42%, 1.43%, 0.67%.
    mu_normal: float = 0.0061
    sigma_normal: float = 0.0042
    mu_stress: float = 0.0143
    sigma_stress: float = 0.0067

    # Numerical safety
    pd_floor: float = 1e-6
    pd_cap: float = 0.99
    alpha_eps: float = 1e-8


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def _normalise_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def load_and_align_data(
    macro_csv: str | Path,
    chargeoff_csv: str | Path,
    date_col: str,
    target_col: str,
    feature_cols: Optional[Iterable[str]],
    horizon: int,
    target_is_percent: bool,
) -> pd.DataFrame:
    """
    Merge macro covariates and charge-offs by date, then create c_{t+horizon}.

    The returned DataFrame has an attribute df.attrs["feature_cols"] storing the
    final list of feature columns used by the models.
    """

    macro = _normalise_date_column(pd.read_csv(macro_csv), date_col)
    charge = _normalise_date_column(pd.read_csv(chargeoff_csv), date_col)

    if target_col not in charge.columns:
        raise ValueError(
            f"target_col='{target_col}' not found in charge-off CSV columns: {list(charge.columns)}"
        )

    if target_is_percent:
        charge[target_col] = charge[target_col] / 100.0

    df = pd.merge(macro, charge[[date_col, target_col]], on=date_col, how="inner")
    df = df.sort_values(date_col).reset_index(drop=True)

    # Forecasting setup: x_t -> c_{t+horizon}.
    df["target_date"] = df[date_col].shift(-horizon)
    df["target_chargeoff_next"] = df[target_col].shift(-horizon)
    df = df.dropna(subset=["target_chargeoff_next"]).reset_index(drop=True)

    if feature_cols is None:
        exclude = {date_col, target_col, "target_date", "target_chargeoff_next"}
        feature_cols = [c for c in df.columns if c not in exclude]
    else:
        feature_cols = list(feature_cols)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found in merged data: {missing}")

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols + ["target_chargeoff_next"]).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} usable observations after alignment/missing-value removal. "
            "Check date alignment, target column and feature columns."
        )

    df.attrs["feature_cols"] = feature_cols
    return df


def chronological_split(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single fixed-origin chronological train/test split."""

    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")
    split_idx = int(np.floor(len(df) * train_frac))
    if split_idx < 10 or len(df) - split_idx < 5:
        raise ValueError("Not enough observations after train/test split. Try a larger dataset or different train_frac.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# -----------------------------------------------------------------------------
# Basel / capital-allocation objective
# -----------------------------------------------------------------------------


def annualised_pd_from_chargeoff(chargeoff: np.ndarray | float, cfg: CapitalConfig) -> np.ndarray | float:
    """Convert quarterly charge-off rate to annualised PD proxy using c ≈ LGD * PD."""

    pd_q = np.asarray(chargeoff, dtype=float) / cfg.lgd
    pd_q = np.clip(pd_q, cfg.pd_floor, cfg.pd_cap)
    pd_a = 1.0 - np.power(1.0 - pd_q, 4)
    return np.clip(pd_a, cfg.pd_floor, cfg.pd_cap)


def asset_correlation(pd_a: np.ndarray | float) -> np.ndarray | float:
    """Basel corporate IRB asset-correlation function."""

    pd_a = np.asarray(pd_a, dtype=float)
    exp_term = (1.0 - np.exp(-50.0 * pd_a)) / (1.0 - np.exp(-50.0))
    return 0.12 * exp_term + 0.24 * (1.0 - exp_term)


def irb_capital_requirement(pd_a: np.ndarray | float, cfg: CapitalConfig) -> np.ndarray | float:
    """Basel-style IRB capital requirement per unit exposure."""

    pd_a = np.clip(np.asarray(pd_a, dtype=float), cfg.pd_floor, cfg.pd_cap)
    r = asset_correlation(pd_a)
    z = norm.ppf(pd_a) / np.sqrt(1.0 - r) + np.sqrt(r / (1.0 - r)) * norm.ppf(0.999)
    b = np.square(0.11852 - 0.05478 * np.log(pd_a))
    maturity_adj = (1.0 + (cfg.maturity - 2.5) * b) / (1.0 - 1.5 * b)
    k_irb = cfg.lgd * (norm.cdf(z) - pd_a) * maturity_adj
    return np.maximum(k_irb, 0.0)


def mixture_mean(p_hat: np.ndarray | float, cfg: CapitalConfig) -> np.ndarray | float:
    """Expected charge-off rate implied by a predicted stress probability."""

    return np.asarray(p_hat, dtype=float) * cfg.mu_stress + (1.0 - np.asarray(p_hat, dtype=float)) * cfg.mu_normal


def expected_shortfall_regime(alpha: float, mu_i: float, sigma_i: float, k_irb: float, cfg: CapitalConfig) -> float:
    """
    Closed-form expected shortfall for one Gaussian charge-off regime.

    Uses the leveraged extension if cfg.leverage != 1:
        shortfall = max{K_IRB * alpha * ell - [alpha*ell*(y-c) + (1-alpha)], 0}.
    """

    if alpha <= cfg.alpha_eps:
        return 0.0

    ell = cfg.leverage
    a = k_irb * alpha * ell - alpha * ell * cfg.y_bar - (1.0 - alpha)
    b = alpha * ell
    threshold = -a / b
    z = (threshold - mu_i) / sigma_i

    tail_prob = 1.0 - norm.cdf(z)
    tail_first_moment = mu_i * tail_prob + sigma_i * norm.pdf(z)
    return float(a * tail_prob + b * tail_first_moment)


def expected_shortfall(alpha: float, p_hat: float, k_irb: float, cfg: CapitalConfig) -> float:
    """Mixture-weighted expected shortfall for PTO/DFL predicted utility."""

    es_n = expected_shortfall_regime(alpha, cfg.mu_normal, cfg.sigma_normal, k_irb, cfg)
    es_s = expected_shortfall_regime(alpha, cfg.mu_stress, cfg.sigma_stress, k_irb, cfg)
    return float((1.0 - p_hat) * es_n + p_hat * es_s)


def predicted_utility(alpha: float, p_hat: float, cfg: CapitalConfig) -> float:
    """Utility used by PTO/DFL to choose alpha from a stress-probability forecast."""

    p_hat = float(np.clip(p_hat, 0.0, 1.0))
    mu_c = float(mixture_mean(p_hat, cfg))
    pd_a = float(annualised_pd_from_chargeoff(mu_c, cfg))
    k_irb = float(irb_capital_requirement(pd_a, cfg))

    ell = cfg.leverage
    expected_terminal_capital = 1.0 + alpha * ell * (cfg.y_bar - mu_c)
    penalty = cfg.lambda_penalty * expected_shortfall(alpha, p_hat, k_irb, cfg)
    return float(expected_terminal_capital - penalty)


def realised_utility(alpha: float, realised_c: float, cfg: CapitalConfig) -> float:
    """Ex-post utility for regret/oracle calculations using realised charge-off c."""

    pd_a = float(annualised_pd_from_chargeoff(realised_c, cfg))
    k_irb = float(irb_capital_requirement(pd_a, cfg))
    ell = cfg.leverage

    terminal_capital = 1.0 + alpha * ell * (cfg.y_bar - realised_c)
    available_buffer = alpha * ell * (cfg.y_bar - realised_c) + (1.0 - alpha)
    required_capital = k_irb * alpha * ell
    shortfall = max(required_capital - available_buffer, 0.0)
    return float(terminal_capital - cfg.lambda_penalty * shortfall)


def optimise_alpha_from_p(p_hat: float, cfg: CapitalConfig) -> float:
    """Solve alpha*(p_hat) = argmax U(alpha; p_hat), alpha in [0, 1]."""

    p_hat = float(np.clip(p_hat, 0.0, 1.0))
    res = minimize_scalar(
        lambda a: -predicted_utility(float(a), p_hat, cfg),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not res.success:
        grid = np.linspace(0.0, 1.0, 1001)
        vals = np.array([predicted_utility(float(a), p_hat, cfg) for a in grid])
        return float(grid[int(np.argmax(vals))])
    return float(np.clip(res.x, 0.0, 1.0))


def optimise_oracle_alpha(realised_c: float, cfg: CapitalConfig) -> float:
    """Oracle alpha using realised charge-off c."""

    res = minimize_scalar(
        lambda a: -realised_utility(float(a), realised_c, cfg),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not res.success:
        grid = np.linspace(0.0, 1.0, 1001)
        vals = np.array([realised_utility(float(a), realised_c, cfg) for a in grid])
        return float(grid[int(np.argmax(vals))])
    return float(np.clip(res.x, 0.0, 1.0))


def regret_from_alpha(alpha_model: float, realised_c: float, cfg: CapitalConfig) -> float:
    """Ex-post regret = U(alpha_oracle; c) - U(alpha_model; c)."""

    alpha_oracle = optimise_oracle_alpha(realised_c, cfg)
    return realised_utility(alpha_oracle, realised_c, cfg) - realised_utility(alpha_model, realised_c, cfg)


def build_capital_config_from_args(args) -> CapitalConfig:
    """Small helper for CLI scripts that expose the same capital arguments."""

    return CapitalConfig(
        lgd=args.lgd,
        y_bar=args.y_bar,
        lambda_penalty=args.lambda_penalty,
        leverage=args.leverage,
        mu_normal=args.mu_normal,
        sigma_normal=args.sigma_normal,
        mu_stress=args.mu_stress,
        sigma_stress=args.sigma_stress,
    )
