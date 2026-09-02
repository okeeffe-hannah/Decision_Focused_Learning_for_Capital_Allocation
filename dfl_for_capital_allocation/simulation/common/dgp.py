"""Data-generating process: simulated quarterly PD panel.

This can be called for each combination of seed, lambda_shortfall, T, and rho_scale 
to generate a new panel. The seed controls random draws and lambda and rho_scale affect 
the decision environment.

Calibration numbers are constants from calibration testing (in notebooks/calibration).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import decision as dec
from .decision import DecisionConfig, DEFAULT_DECISION_CONFIG


@dataclass(frozen=True)
class DGPConfig:
    """Configuration for latent-state DGP and associated decision environment."""
    T: int = 1000

    # Simulates additional leading quarters, then discards them to stabilise latent state
    burn_in: int = 200

    # Values calculated in calibration exercise (find in notebooks/calibration.ipynb)
    pd_min: float = 0.001
    pd_max: float = 0.014
    beta0: float = -1.764472
    beta1: float = 1.782915
    beta2: float = 0.038343

    # Latent-state AR(1) parameters
    phi: float = 0.95
    sigma: float = 0.35
    cycle_ar: float = 0.75
    cycle_sigma: float = 0.45

    decision: DecisionConfig = field(default_factory=lambda: DEFAULT_DECISION_CONFIG)

    def with_lambda(self, lambda_shortfall: float) -> "DGPConfig":
        """Returns DGP variant with only changed shortfall penalty."""
        return self.variant(lambda_shortfall=lambda_shortfall)

    def variant(self, lambda_shortfall: float = None, T: int = None, rho_scale: float = None,
                burn_in: int = None) -> "DGPConfig":
        """Return a modified copy of this simulation configuration. Supplied arguments 
        override the previous settings. All other DGP/decision params remain unchanged."""
        decision = self.decision
        if lambda_shortfall is not None:
            decision = decision.with_lambda(lambda_shortfall)
        if rho_scale is not None:
            decision = decision.with_rho_scale(rho_scale)
        return DGPConfig(
            T=self.T if T is None else int(T),
            burn_in=self.burn_in if burn_in is None else int(burn_in),
            pd_min=self.pd_min, pd_max=self.pd_max,
            beta0=self.beta0, beta1=self.beta1, beta2=self.beta2,
            phi=self.phi, sigma=self.sigma, cycle_ar=self.cycle_ar, cycle_sigma=self.cycle_sigma,
            decision=decision,
        )


DEFAULT_DGP_CONFIG = DGPConfig()

# Simulated macro-covariates 
MACRO_COLS = [
    "x_growth", "x_unemployment", "x_credit_spread", "x_yield_curve", "x_house_price_growth",
]

FEATURE_COLS = [*MACRO_COLS, "c", "c_lag1"]

DIAGNOSTIC_COLS = [
    "latent_state", "latent_cycle", "PD_quarterly_true", "PD_annual_true", "PD_true",
    "expected_c_quarterly", "expected_c_annualised", "default_rate", "c_annualised_sim",
    "R_true", "K_IRB_true", "alpha_oracle", "utility_oracle", "shortfall_oracle",
    "alpha_bayes", "expected_utility_bayes", "utility_bayes_realised",
]

PREFERRED_ORDER = [
    "date", "t", *MACRO_COLS, "c_lag1",
    "PD_quarterly_true", "PD_annual_true", "PD_true",
    "expected_c_quarterly", "expected_c_annualised",
    "default_rate", "c", "c_annualised_sim",
    "R_true", "K_IRB_true",
    "alpha_oracle", "utility_oracle", "shortfall_oracle",
    "terminal_capital_oracle", "required_capital_oracle", "available_buffer_oracle",
    "alpha_bayes", "expected_utility_bayes", "utility_bayes_realised", "shortfall_bayes_realised",
    "regret_bayes_vs_oracle",
    "alpha_full", "utility_full", "shortfall_full",
    "alpha_zero", "utility_zero",
    "latent_state", "latent_cycle",
    "simulation_seed",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_latent_state(T, seed, phi=0.95, sigma=0.35, cycle_ar=0.75, cycle_sigma=0.45):
    """Simulate persistent latent stress and a second less persistent component."""
    rng = np.random.default_rng(seed)
    state = np.zeros(T)
    cycle = np.zeros(T)
    for t in range(1, T):
        state[t] = phi * state[t - 1] + rng.normal(0, sigma)
        cycle[t] = cycle_ar * cycle[t - 1] + rng.normal(0, cycle_sigma)
    state = (state - state.mean()) / state.std(ddof=0)
    cycle = (cycle - cycle.mean()) / cycle.std(ddof=0)
    return pd.DataFrame({"latent_state": state, "latent_cycle": cycle})


def generate_macro_variables(latent_df, seed):
    """Create noisy observed macro-financial predictors from latent stress."""
    rng = np.random.default_rng(seed)
    out = latent_df.copy()
    s = out["latent_state"].to_numpy()
    q = out["latent_cycle"].to_numpy()
    T_local = len(out)

    out["t"] = np.arange(T_local)
    out["date"] = pd.period_range(start="1980Q1", periods=T_local, freq="Q").astype(str)
    out["x_growth"] = -0.80 * s + 0.15 * q + rng.normal(0, 0.45, T_local)
    out["x_unemployment"] = 0.90 * s + 0.10 * q + rng.normal(0, 0.40, T_local)
    out["x_credit_spread"] = 0.70 * s + 0.30 * s ** 2 + 0.15 * q + rng.normal(0, 0.45, T_local)
    out["x_yield_curve"] = -0.50 * s + 0.25 * q + rng.normal(0, 0.50, T_local)
    out["x_house_price_growth"] = -0.60 * s - 0.15 * q + rng.normal(0, 0.45, T_local)
    return out


def quarterly_pd_from_latent(latent_df, beta0, beta1, beta2, pd_min, pd_max,
                              cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Map latent stress into a bounded quarterly PD process."""
    s = latent_df["latent_state"].to_numpy()
    q = latent_df["latent_cycle"].to_numpy()
    nonlinear_component = 0.5 * s ** 2 + 0.25 * q
    pd_index = beta0 + beta1 * s + beta2 * nonlinear_component
    pd_quarterly = pd_min + sigmoid(pd_index) * (pd_max - pd_min)
    return np.clip(pd_quarterly, cfg.PD_Q_CLIP_LOW, cfg.PD_Q_CLIP_HIGH)


def sample_vasicek_default_rate(pd_quarterly_true, pd_annual_true, seed,
                                 cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Draw quarterly default and charge-off rates from the Vasicek model."""
    rng = np.random.default_rng(seed)
    pd_quarterly_true = dec.clip_pd_q(pd_quarterly_true, cfg)
    pd_annual_true = dec.clip_pd_a(pd_annual_true, cfg)
    rho = dec.vasicek_rho(pd_annual_true, cfg)
    Z = rng.normal(0, 1, size=pd_quarterly_true.shape)

    default_rate_q = norm.cdf(
        (norm.ppf(pd_quarterly_true) - np.sqrt(rho) * Z) / np.sqrt(1 - rho)
    )
    default_rate_q = np.clip(default_rate_q, 0, 1)
    charge_off_rate_q = np.clip(cfg.LGD * default_rate_q, 0, cfg.LGD)
    return default_rate_q, charge_off_rate_q


def build_simulated_panel(seed: int, cfg: DGPConfig = DEFAULT_DGP_CONFIG) -> pd.DataFrame:
    """Generates the simulated panel for one configured scenario. Simulates latent stress,
    observed macro-covariates, probabilities, Vasicek losses and allocation benchmarks."""
    dcfg = cfg.decision
    total_T = int(cfg.T) + int(cfg.burn_in)
    latent = simulate_latent_state(total_T, seed=seed, phi=cfg.phi, sigma=cfg.sigma,
                                    cycle_ar=cfg.cycle_ar, cycle_sigma=cfg.cycle_sigma)
    panel = generate_macro_variables(latent, seed=seed + 1)

    panel["PD_quarterly_true"] = quarterly_pd_from_latent(
        panel, cfg.beta0, cfg.beta1, cfg.beta2, cfg.pd_min, cfg.pd_max, dcfg)
    panel["PD_annual_true"] = 1 - (1 - panel["PD_quarterly_true"]) ** 4
    panel["PD_true"] = panel["PD_quarterly_true"]
    panel["expected_c_quarterly"] = dcfg.LGD * panel["PD_quarterly_true"]
    panel["expected_c_annualised"] = 4 * panel["expected_c_quarterly"]

    default_rate_q, c_q = sample_vasicek_default_rate(
        panel["PD_quarterly_true"].to_numpy(), panel["PD_annual_true"].to_numpy(), seed=seed + 2, cfg=dcfg)
    panel["default_rate"] = default_rate_q
    panel["c"] = c_q
    panel["c_annualised_sim"] = 4 * panel["c"]
    panel["c_lag1"] = panel["c"].shift(1)

    panel["R_true"] = dec.basel_r_from_annual_pd(panel["PD_annual_true"].to_numpy(), dcfg)
    panel["K_IRB_true"] = dec.k_irb_from_annual_pd(panel["PD_annual_true"].to_numpy(), dcfg)

    oracle = dec.compute_oracle_allocation(panel["c"].to_numpy(), panel["K_IRB_true"].to_numpy(), dcfg)
    panel = pd.concat([panel.reset_index(drop=True), oracle], axis=1)

    full = dec.decision_components_vec(1.0, panel["c"].to_numpy(), panel["K_IRB_true"].to_numpy(), dcfg)
    zero = dec.decision_components_vec(0.0, panel["c"].to_numpy(), panel["K_IRB_true"].to_numpy(), dcfg)
    panel["alpha_full"], panel["utility_full"], panel["shortfall_full"] = 1.0, full["utility"], full["shortfall"]
    panel["alpha_zero"], panel["utility_zero"] = 0.0, zero["utility"]

    bayes = dec.compute_bayes_allocation(
        panel["PD_quarterly_true"].to_numpy(), panel["PD_annual_true"].to_numpy(),
        panel["K_IRB_true"].to_numpy(), dcfg)
    panel = pd.concat([panel.reset_index(drop=True), bayes.reset_index(drop=True)], axis=1)

    bayes_realised = dec.decision_components_vec(
        panel["alpha_bayes"].to_numpy(), panel["c"].to_numpy(), panel["K_IRB_true"].to_numpy(), dcfg)
    panel["utility_bayes_realised"] = bayes_realised["utility"]
    panel["shortfall_bayes_realised"] = bayes_realised["shortfall"]
    panel["regret_bayes_vs_oracle"] = panel["utility_oracle"] - panel["utility_bayes_realised"]

    if cfg.burn_in:
        # Trim after losses are generated 
        panel = panel.iloc[int(cfg.burn_in):].reset_index(drop=True)
    panel = panel.dropna(subset=["c_lag1"]).reset_index(drop=True)
    panel["t"] = np.arange(len(panel))
    panel["simulation_seed"] = seed

    return panel[PREFERRED_ORDER]


def build_metadata(seed: int, cfg: DGPConfig, panel: pd.DataFrame, output_csv_name: str) -> dict:
    """Metadata JSON for each new configuration. Records reproducibility information."""
    dcfg = cfg.decision
    return {
        "description": (
            "Synthetic quarterly PD-centric panel."
        ),
        "constants": {
            "T_raw": cfg.T,
            "burn_in_quarters": int(cfg.burn_in),
            "T_after_lag_drop": int(len(panel)),
            "LGD": dcfg.LGD, "K1": dcfg.K1, "ell": dcfg.ell, "Y_BAR": dcfg.Y_BAR,
            "lambda_shortfall": dcfg.lambda_shortfall, "M": dcfg.M, "RHO_SCALE": dcfg.RHO_SCALE,
            "APPLY_PD_FLOOR": dcfg.APPLY_PD_FLOOR, "PD_FLOOR": dcfg.PD_FLOOR,
            "allocation_optimisation": "continuous",
            "alpha_bounds_min": float(dcfg.ALPHA_BOUNDS[0]), "alpha_bounds_max": float(dcfg.ALPHA_BOUNDS[1]),
            "alpha_opt_xatol": float(dcfg.ALPHA_OPT_XATOL),
            "PD_CLIP_LOW": dcfg.PD_Q_CLIP_LOW, "PD_CLIP_HIGH": dcfg.PD_Q_CLIP_HIGH,
            "pd_min_quarterly": cfg.pd_min, "pd_max_quarterly": cfg.pd_max,
            "beta0": float(cfg.beta0), "beta1": float(cfg.beta1), "beta2": cfg.beta2,
        },
        "timing_convention": (
            "Downstream models predict quarter t+1 from features dated t (macro x_t, c_t, "
            "c_lag1). x_t and c_t are observable at decision time by construction; no reporting lag."
        ),
        "feature_cols": FEATURE_COLS,
        "diagnostic_cols": DIAGNOSTIC_COLS,
        "random_seed": seed,
        "output_csv": output_csv_name,
        "simulation_seed_column": "simulation_seed",
    }
