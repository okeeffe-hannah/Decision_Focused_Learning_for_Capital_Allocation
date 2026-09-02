"""Shared Basel/Vasicek/allocation functions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import owens_t
from scipy.stats import norm


@dataclass(frozen=True)
class DecisionConfig:
    """Constants for the decision problem + Basel/Vasicek.
    """

    LGD: float = 0.45
    K1: float = 1.0
    ell: float = 10.0                
    Y_BAR: float = 0.0168
    lambda_shortfall: float = 10.0
    M: float = 2.5
    RHO_SCALE: float = 0.03

    APPLY_PD_FLOOR: bool = True
    PD_FLOOR: float = 3e-4

    PD_Q_CLIP_LOW: float = 1e-6
    PD_Q_CLIP_HIGH: float = 0.99
    PD_A_CLIP_LOW: float = 1e-6
    PD_A_CLIP_HIGH: float = 0.99

    ALPHA_BOUNDS: Tuple[float, float] = (0.0, 1.0)
    ALPHA_OPT_XATOL: float = 1e-10

    def with_lambda(self, lambda_shortfall: float) -> "DecisionConfig":
        """Copy of the config with a different shortfall penalty."""
        return replace(self, lambda_shortfall=float(lambda_shortfall))

    def with_rho_scale(self, rho_scale: float) -> "DecisionConfig":
        """Copy of the config with a different asset-correlation scale."""
        return replace(self, RHO_SCALE=float(rho_scale))


DEFAULT_DECISION_CONFIG = DecisionConfig()

def clip_pd_q(pd_q, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    return np.clip(np.asarray(pd_q, dtype=float), cfg.PD_Q_CLIP_LOW, cfg.PD_Q_CLIP_HIGH)


def clip_pd_a(pd_a, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    return np.clip(np.asarray(pd_a, dtype=float), cfg.PD_A_CLIP_LOW, cfg.PD_A_CLIP_HIGH)


def quarterly_pd_to_annual(pd_q, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    pd_q = clip_pd_q(pd_q, cfg)
    return clip_pd_a(1.0 - np.power(1.0 - pd_q, 4.0), cfg)


# ---------------------------------------------------------------------------
# Basel IRB Functions and Derivatives
# ---------------------------------------------------------------------------


def basel_b_and_bprime(pd_a):
    """Computes b and its derivative wrt PD as defined in the writeup (see thesis appendix)."""
    pd_a = np.asarray(pd_a, dtype=float)
    u = 0.11852 - 0.05478 * np.log(pd_a)
    b = u ** 2
    b_prime = -0.10956 * u / pd_a
    return b, b_prime


def maturity_adjustment_and_prime(m, b, b_prime):
    """Computes the maturity adjustment and its derivative wrt PD (see thesis appendix)."""
    n = 1.0 + (m - 2.5) * b
    d = 1.0 - 1.5 * b
    ma = n / d
    ma_prime = (b_prime * ((m - 2.5) * d + 1.5 * n)) / (d ** 2)
    return ma, ma_prime


def k_irb_prime_annual(pd_a, cfg):
    """Analytic K_IRB derivative (before applying quarterly chain rule factor). 
    See thesis for full derivation of component parts."""

    pd_a_raw = np.atleast_1d(np.asarray(pd_a, dtype=float))
    lgd, m = cfg.LGD, cfg.M # Losses given default and loan maturity (set to 2.5) 


    pd_a_clipped = clip_pd_a(pd_a_raw, cfg)

    # Apply IRB required floor
    if cfg.APPLY_PD_FLOOR:
        pd_a_eval = np.maximum(pd_a_clipped, cfg.PD_FLOOR)
    else:
        pd_a_eval = pd_a_clipped

    # Set derivative to zero if PD is at the floor or clipped
    at_floor = cfg.APPLY_PD_FLOOR & (pd_a_clipped <= cfg.PD_FLOOR)
    at_clip = (pd_a_raw <= cfg.PD_A_CLIP_LOW) | (pd_a_raw >= cfg.PD_A_CLIP_HIGH)
    flat_region = at_floor | at_clip

    r = basel_r_from_annual_pd(pd_a_eval, cfg)
    r_prime = basel_r_prime(pd_a_eval, cfg)
    b, b_prime = basel_b_and_bprime(pd_a_eval)
    ma, ma_prime = maturity_adjustment_and_prime(m, b, b_prime)

    a_probit = norm.ppf(pd_a_eval)
    phi_a = norm.pdf(a_probit)
    sqrt_1mr = np.sqrt(1.0 - r)
    z = a_probit / sqrt_1mr + np.sqrt(r / (1.0 - r)) * norm.ppf(0.999)

    # dZ/dPD|_R (direct effect, R held fixed)
    dz_dpd_given_r = 1.0 / (sqrt_1mr * np.maximum(phi_a, 1e-300)) # numerical stability floor

    # dZ/dR|_PD (indirect effect through R(PD))
    dz_dr_given_pd = (a_probit / (2.0 * (1.0 - r) ** 1.5)
                      + norm.ppf(0.999) / (2.0 * (1.0 - r) ** 2 * np.sqrt(r / (1.0 - r))))
    
    z_prime = dz_dpd_given_r + dz_dr_given_pd * r_prime

    phi_z = norm.pdf(z)
    cap_z = norm.cdf(z)

    # Overall K_irb derivative
    dk_dpd = lgd * ((phi_z * z_prime - 1.0) * ma + (cap_z - pd_a_eval) * ma_prime)

    # Return zero where floor/cap binds
    return np.where(flat_region, 0.0, dk_dpd)


def k_irb_prime_quarterly(pd_q, cfg):
    """Chained annualised of PD with K_IRB derivative."""
    pd_q = np.atleast_1d(np.asarray(pd_q, dtype=float))
    pd_q_c = clip_pd_q(pd_q, cfg)
    pd_a = quarterly_pd_to_annual(pd_q_c, cfg)
    # Apply chaing rule
    dpda_dpdq = 4.0 * (1.0 - pd_q_c) ** 3
    return k_irb_prime_annual(pd_a, cfg) * dpda_dpdq


def basel_r_from_annual_pd(pd_a, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Basel asset correlation from annual PD."""
    pd_a = clip_pd_a(pd_a, cfg)
    exp_term = (1.0 - np.exp(-50.0 * pd_a)) / (1.0 - np.exp(-50.0))
    return 0.12 * exp_term + 0.24 * (1.0 - exp_term)


def basel_r_prime(pd_a, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Derivative of Basel asset correlation wrt PD."""
    pd_a = clip_pd_a(pd_a, cfg)
    return -6.0 * np.exp(-50.0 * pd_a) / (1.0 - np.exp(-50.0))

def k_irb_from_annual_pd(pd_a, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG,
                          lgd=None, m=None, apply_floor=None):
    """Basel IRB capital requirement per unit exposure. Input must be annual PD
    Formulas taken directly from Basel handbook."""

    lgd = cfg.LGD if lgd is None else lgd
    m = cfg.M if m is None else m
    apply_floor = cfg.APPLY_PD_FLOOR if apply_floor is None else apply_floor
    pd_a = clip_pd_a(pd_a, cfg)
    if apply_floor:
        pd_a = np.maximum(pd_a, cfg.PD_FLOOR)
    
    # Asset correlation
    r = basel_r_from_annual_pd(pd_a, cfg)

    # Maturity adjustment
    b = np.square(0.11852 - 0.05478 * np.log(pd_a))
    maturity_adj = (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)

    z = norm.ppf(pd_a) / np.sqrt(1.0 - r) + np.sqrt(r / (1.0 - r)) * norm.ppf(0.999)
    k_irb = lgd * (norm.cdf(z) - pd_a) * maturity_adj

    return np.maximum(k_irb, 0.0)


def vasicek_rho(pd_a, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Basel R times RHO_SCALE for Vasicek distribution."""
    return np.clip(cfg.RHO_SCALE * basel_r_from_annual_pd(pd_a, cfg), 1e-4, 0.999)


# ---------------------------------------------------------------------------
# Vasicek expected shortfall
# ---------------------------------------------------------------------------

def bivariate_normal_cdf(h, k, rho):
    """Computes P(X <= h, Y <= k) for correlated standard normal variables.
    
    Uses Owen's T-function representation.

    Reference:
        Owen, D. B. (1956), "Tables for Computing Bivariate Normal
        Probabilities," Annals of Mathematical Statistics, 27(4),
        1075–1090.
    """
    # h and k are the two standard normal thresholds
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    # asset correlation
    rho = np.clip(np.asarray(rho, dtype=float), -0.999999, 0.999999)

    # Avoid division by zero
    h_safe = np.where(h == 0.0, 1e-12, h)
    k_safe = np.where(k == 0.0, 1e-12, k)

    denom = np.sqrt(np.maximum(1.0 - rho ** 2, 1e-300))

    a = (k / h_safe - rho) / denom
    b = (h / k_safe - rho) / denom

    #Owen's representation of the bivariate normal CDF
    val = 0.5 * (norm.cdf(h) + norm.cdf(k)) - owens_t(h_safe, a) - owens_t(k_safe, b)
    hk = h * k
    delta = np.where((hk > 0.0) | ((hk == 0.0) & (h + k >= 0.0)), 0.0, 0.5)
    return val - delta


def bivariate_normal_pdf(h, k, rho):
    """Standard bivariate normal density phi_2(h, k; rho).
    
    Used for the rho-channel contribution to the Vasicek derivative.
    By Plackett's identity: d/d(rho)[Phi_2(h, k; rho)] = phi_2(h, k; rho).
    
    Reference:
    Plackett, R. L. (1954), "A Reduction Formula for Normal
    Multivariate Integrals," Biometrika, 41(3–4), 351–360.
    """
    rho = np.clip(np.atleast_1d(np.asarray(rho, dtype=float)), -0.999999, 0.999999)
    h = np.atleast_1d(np.asarray(h, dtype=float))
    k = np.atleast_1d(np.asarray(k, dtype=float))
    one_minus_rho2 = np.maximum(1.0 - rho ** 2, 1e-300)
    z = (h ** 2 - 2.0 * rho * h * k + k ** 2) / (2.0 * one_minus_rho2)
    return np.exp(-z) / (2.0 * np.pi * np.sqrt(one_minus_rho2))


def vasicek_expected_shortfall(x, pd_q, rho, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Closed-form Vasicek expected excess loss rate.
    Used by the PTO allocation optimisation  (see thesis for full derivation).
    """
    x = np.asarray(x, dtype=float)
    pd_q = float(clip_pd_q(pd_q, cfg))
    rho = float(np.clip(rho, 1e-6, 1.0 - 1e-6))
    a = norm.ppf(pd_q)
    sqrt_rho = np.sqrt(rho)
    sqrt_1mrho = np.sqrt(1.0 - rho)

    # Loss rate
    x_clipped = np.clip(x, cfg.PD_Q_CLIP_LOW, cfg.PD_Q_CLIP_HIGH)

    # Systematic factor threshold
    y_x = (a - norm.ppf(x_clipped) * sqrt_1mrho) / sqrt_rho

    # Expected excess default rate above threshold x
    es = bivariate_normal_cdf(a, y_x, sqrt_rho) - x_clipped * norm.cdf(y_x)

    # Threshold below minimum possible so every loss exceeds
    es = np.where(x <= 0.0, pd_q - x, es)
    
    # Vasicek loss rate L cannot exceed 1 so replace with zero
    es = np.where(x >= 1.0, 0.0, es)

    return np.maximum(es, 0.0) 


# ---------------------------------------------------------------------------
# Allocation objective and realised-utility decision components
# ---------------------------------------------------------------------------

def allocation_utility(alpha, c_q, k, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Computes utility derived from the allocation with the capital-shortfall penalty.
 
    Used by optimise_alpha_oracle(), DFL, BB use differentiable pytorch verisons found
    in individual model files."""
    alpha = np.asarray(alpha, dtype=float)
    exposure = cfg.ell * alpha * cfg.K1
    terminal_capital = cfg.K1 + exposure * (cfg.Y_BAR - float(c_q))
    committed_capital = alpha * cfg.K1
    available_buffer = terminal_capital - committed_capital
    required_capital = float(k) * exposure
    shortfall = np.maximum(required_capital - available_buffer, 0.0)
    return terminal_capital - cfg.lambda_shortfall * shortfall


def decision_components(alpha, c_q, k, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Returns dictionary containing components of decision problem for a given alpha
     allocation.

    Called by the PTO, black-box, and DFL model when scoring realised
    and predicted allocations, and decision_problem/motivating_facts notebook. """
    exposure = cfg.ell * alpha * cfg.K1
    terminal_capital = cfg.K1 + exposure * (cfg.Y_BAR - c_q)
    committed_capital = alpha * cfg.K1
    available_buffer = terminal_capital - committed_capital
    required_capital = float(k) * exposure
    shortfall = max(required_capital - available_buffer, 0.0)
    utility = terminal_capital - cfg.lambda_shortfall * shortfall
    return dict(exposure=exposure, terminal_capital=terminal_capital, committed_capital=committed_capital,
                available_buffer=available_buffer, required_capital=required_capital, shortfall=shortfall,
                utility=utility, k_irb=float(k))


def decision_components_vec(alpha, c_q, k, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Vectorised version of decision_components for processing panel inputs.

    Accepts arrays of allocations, charge-off rates, and capital requirements.
    """
    alpha = np.asarray(alpha, dtype=float)
    c_q = np.asarray(c_q, dtype=float)
    k = np.asarray(k, dtype=float)
    exposure = cfg.ell * alpha * cfg.K1
    terminal_capital = cfg.K1 + exposure * (cfg.Y_BAR - c_q)
    committed_capital = alpha * cfg.K1
    available_buffer = terminal_capital - committed_capital
    required_capital = k * exposure
    shortfall = np.maximum(required_capital - available_buffer, 0.0)
    utility = terminal_capital - cfg.lambda_shortfall * shortfall
    return dict(utility=utility, shortfall=shortfall, terminal_capital=terminal_capital,
                required_capital=required_capital, available_buffer=available_buffer)


def optimise_alpha_oracle(c_q, k, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Finds the best allocation for given quarterly charge-off rate and 
    required capital. Used for oracle decisions ie observed charge-off rate.
    
    Uses bounded one-dimensional scalar minimisation of negative utility.
    """

    # Minimises negative utility to maximise utility
    def neg_utility(alpha):
        return -float(allocation_utility(alpha, c_q=c_q, k=k, cfg=cfg))

    opt = minimize_scalar(neg_utility, bounds=cfg.ALPHA_BOUNDS, method="bounded",
                           options={"xatol": cfg.ALPHA_OPT_XATOL})
    
    # Checks both boundaries against opt and returns candidate with highest utility.
    candidates = [(0.0, -neg_utility(0.0)), (1.0, -neg_utility(1.0)), (float(opt.x), -float(opt.fun))]
    return max(candidates, key=lambda item: item[1])


def compute_oracle_allocation(c_values, k_irb_values, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Applies optimise_alpha_oracle across entire dataset. Used for simulated panels to get
    history of oracle allocations.
    """

    keep_keys = ("utility", "shortfall", "terminal_capital", "required_capital", "available_buffer")
    records = []
    
    # Handles many observations, loops through each quarter
    for c_t, k_t in zip(c_values, k_irb_values):
        # Finds best allocation
        alpha_star, _ = optimise_alpha_oracle(c_t, k_t, cfg)
        # Computes associated decision components
        best = decision_components(alpha_star, c_t, k_t, cfg)
        # Stores results
        records.append({"alpha_oracle": alpha_star,
                        **{f"{key}_oracle": best[key] for key in keep_keys}})
    return pd.DataFrame(records)


def optimise_alpha_expected(pd_q_hat, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Finds PTO allocation that maximises utility.
   
    Uses Vasicek loss distribution implied by model's own predicted PD. Expected 
    utility is maximised using bounded 1d scalar minimisation (same as oracle).
    """
    pd_q_hat = float(clip_pd_q(pd_q_hat, cfg))
    pd_a_hat = float(quarterly_pd_to_annual(pd_q_hat, cfg))
    k_hat = float(k_irb_from_annual_pd(pd_a_hat, cfg))
    rho = float(vasicek_rho(pd_a_hat, cfg))

    # Expected utility closed form derivation in thesis
    def expected_utility(alpha):
        alpha = float(alpha)
        expected_terminal_capital = cfg.K1 + cfg.ell * alpha * (cfg.Y_BAR - cfg.LGD * pd_q_hat)
        if alpha <= 0.0:
            return expected_terminal_capital
        x = ((1.0 - alpha) + cfg.ell * alpha * (cfg.Y_BAR - k_hat)) / (cfg.ell * alpha * cfg.LGD)
        es = vasicek_expected_shortfall(np.array([x]), pd_q_hat, rho, cfg)[0]
        return expected_terminal_capital - cfg.lambda_shortfall * cfg.ell * alpha * cfg.LGD * es


    opt = minimize_scalar(lambda a: -expected_utility(a), bounds=cfg.ALPHA_BOUNDS,
                           method="bounded", options={"xatol": cfg.ALPHA_OPT_XATOL})
    candidates = [(0.0, expected_utility(0.0)), (1.0, expected_utility(1.0)),
                  (float(opt.x), -float(opt.fun))]
    alpha_star, expected_u_star = max(candidates, key=lambda item: item[1])
    return alpha_star, expected_u_star


def compute_bayes_allocation(pd_q_values, pd_a_values, k_irb_values,
                              cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Computes allocation that maximises expected utility under the true Vasicek loss
    distribution given the true PD. Ie gives the best allocation achievable by a
    perfectly specified model. 
    
    Used as benchmark comparison (as asset correlation increases). Gap between this 
    and oracle is irreducible uncertainty.
    """
    records = []
    # Loops across observations for whole simulated panel 
    for pd_q_t, pd_a_t, k_t in zip(pd_q_values, pd_a_values, k_irb_values):
        rho_t = float(vasicek_rho(pd_a_t, cfg))
        pd_q_t = float(clip_pd_q(pd_q_t, cfg))

        # Uses expected utility with known true PD and loss distribution
        def expected_utility(alpha, pd_q_t=pd_q_t, k_t=k_t, rho_t=rho_t):
            alpha = float(alpha)
            expected_terminal_capital = cfg.K1 + cfg.ell * alpha * (cfg.Y_BAR - cfg.LGD * pd_q_t)
            if alpha <= 0.0:
                return expected_terminal_capital
            x = ((1.0 - alpha) + cfg.ell * alpha * (cfg.Y_BAR - k_t)) / (cfg.ell * alpha * cfg.LGD)
            es = vasicek_expected_shortfall(np.array([x]), pd_q_t, rho_t, cfg)[0]
            return expected_terminal_capital - cfg.lambda_shortfall * cfg.ell * alpha * cfg.LGD * es

        opt = minimize_scalar(lambda a: -expected_utility(a), bounds=cfg.ALPHA_BOUNDS,
                               method="bounded", options={"xatol": cfg.ALPHA_OPT_XATOL})
        candidates = [(0.0, expected_utility(0.0)), (1.0, expected_utility(1.0)),
                      (float(opt.x), -float(opt.fun))]
        alpha_star, expected_u_star = max(candidates, key=lambda item: item[1])
        records.append({"alpha_bayes": alpha_star, "expected_utility_bayes": expected_u_star})
    return pd.DataFrame(records)


def resolve_first_test_idx(df, split_cfg):
    """Function to determine start of training/test windows for expanding window refits."""

    if "initial_train_frac" in split_cfg: # Used in simulation
        first_test_idx = int(np.floor(len(df) * float(split_cfg["initial_train_frac"])))
    elif "first_test_date" in split_cfg: # Used in empirical notebooks
        mask = (df["target_date"] >= pd.Timestamp(split_cfg["first_test_date"])).to_numpy()
        if not mask.any():
            raise ValueError("first_test_date is after the last available target date.")
        first_test_idx = int(np.flatnonzero(mask)[0])
    else:
        raise ValueError("Split config needs 'initial_train_frac' or 'first_test_date'.")
    
    if first_test_idx < 50 or len(df) - first_test_idx < 20: # Checks there is enough data to train
        raise ValueError(f"Split leaves too little data: first_test_idx={first_test_idx} of {len(df)}.")
    
    return first_test_idx


def newey_west_lrv(d, lags):
    """HAC estimate of Var(mean(d)) used in auditability notebook."""
    d = np.asarray(d, dtype=float)
    d = d - d.mean()
    n = len(d)
    v = float(np.mean(d * d))
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        v += 2.0 * w * float(np.mean(d[k:] * d[:-k]))
    return v / n
