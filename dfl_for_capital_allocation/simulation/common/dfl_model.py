"""Decision-focused learning full model pipeline.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


from . import decision as dec
from .decision import DecisionConfig, DEFAULT_DECISION_CONFIG
from .model_config import ModelConfig, DEFAULT_MODEL_CONFIG

DATE_COL = "date"
HORIZON = 1
ARCHITECTURES = ["linear", "mlp"]

DFL_ALPHA_LO = 1e-4
DFL_BISECT_ITERS = 30
DFL_GRAD_CLIP = 1e3
_X_EPS = 1e-9

# Determine if rho channel in Vasicek asset correlation included in IFT grad
DFL_INCLUDE_RHO_CHANNEL = True

# ---------------------------------------------------------------------------
# Load and align the simulated panel
# ---------------------------------------------------------------------------

def load_simulated_panel(panel_csv, feature_cols, horizon=HORIZON, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    df = pd.read_csv(panel_csv)
    df[DATE_COL] = pd.PeriodIndex(df[DATE_COL], freq="Q").to_timestamp()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing feature columns: {missing}")

    df["target_date"] = df[DATE_COL].shift(-horizon)
    df["c_q_realised"] = df["c"].shift(-horizon)
    df["pd_q_realised"] = dec.clip_pd_q(df["c_q_realised"] / cfg.LGD, cfg)
    df["pd_q_true_next"] = df["PD_quarterly_true"].shift(-horizon)
    df["pd_a_true_next"] = df["PD_annual_true"].shift(-horizon)
    df["k_irb_true_next"] = df["K_IRB_true"].shift(-horizon)
    df["alpha_oracle_next"] = df["alpha_oracle"].shift(-horizon)
    df["utility_oracle_next"] = df["utility_oracle"].shift(-horizon)
    df["alpha_bayes_next"] = df["alpha_bayes"].shift(-horizon)
    df["utility_bayes_realised_next"] = df["utility_bayes_realised"].shift(-horizon)
    df["target_pd_q_next"] = df["pd_q_realised"]

    required = list(feature_cols) + ["target_date", "c_q_realised", "pd_q_realised", "pd_q_true_next",
                                     "pd_a_true_next", "k_irb_true_next", "alpha_oracle_next",
                                     "utility_oracle_next", "alpha_bayes_next",
                                     "utility_bayes_realised_next", "target_pd_q_next"]
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} usable observations; check the panel.")
    return df


# ---------------------------------------------------------------------------
# DFL core: Forecast-based objective, bisection forward pass, IFT backward pass
# ---------------------------------------------------------------------------

def dfl_rho(pd_a, dcfg: DecisionConfig):
    return dec.vasicek_rho(pd_a, dcfg)


def dfl_threshold_x(alpha, k, dcfg: DecisionConfig):
    return ((1.0 - alpha) + dcfg.ell * alpha * (dcfg.Y_BAR - k)) / (dcfg.ell * alpha * dcfg.LGD)


def vasicek_es_exceed(x, pd_q, rho, dcfg: DecisionConfig):
    """DFL version of vasicek_expected_shortfall() in decision.py
    
    Computes the expected excess loss above a threshold along with 
    the probability the loss exceeds the threshold. Additionally 
    returns systematic factor threshold corresponding to loss rate x.
    """

    # Convert to array for batches of observations
    x = np.atleast_1d(np.asarray(x, dtype=float))
    pd_q = np.atleast_1d(np.asarray(pd_q, dtype=float))
    rho = np.atleast_1d(np.asarray(rho, dtype=float))

    # Inverse standard normal CDF 
    a = ndtri(np.clip(pd_q, dcfg.PD_Q_CLIP_LOW, 1.0 - dcfg.PD_Q_CLIP_LOW))

    # Check x lies in the valid loss rate range
    inside = (x > _X_EPS) & (x < 1.0 - _X_EPS)
    x_in = np.clip(x, _X_EPS, 1.0 - _X_EPS)

    # Calculate systematic factor threshold associated with loss level x
    y_x = (a - np.sqrt(1.0 - rho) * ndtri(x_in)) / np.sqrt(rho)
    # Probability of exceeding loss threshold x
    p_in = norm.cdf(y_x)

    # Calculate expected excess loss using closed-form expression (derivation in thesis)
    es_in = dec.bivariate_normal_cdf(a, y_x, np.sqrt(rho)) - x_in * p_in

    # Handle thresholds outside valid loss-rate range
    es = np.where(inside, es_in, np.where(x <= _X_EPS, pd_q - x, 0.0))
    p = np.where(inside, p_in, np.where(x <= _X_EPS, 1.0, 0.0))
    return es, p, y_x, inside


def vasicek_pdf(x, pd_q, rho, dcfg: DecisionConfig):
    """Computes Vasicek loss-rate density f_L(x; pd_q, rho).

    Density is used in the IFT gradient calculation. Set to zero 
    for thresholds outside (0,1)."""

    x = np.atleast_1d(np.asarray(x, dtype=float))
    pd_q = np.atleast_1d(np.asarray(pd_q, dtype=float))
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    inside = (x > _X_EPS) & (x < 1.0 - _X_EPS)
    x_in = np.clip(x, _X_EPS, 1.0 - _X_EPS)
    a = ndtri(np.clip(pd_q, dcfg.PD_Q_CLIP_LOW, 1.0 - dcfg.PD_Q_CLIP_LOW))
    xi = ndtri(x_in)
    # Closed-form Vasicek loss-rate density
    dens = (np.sqrt((1.0 - rho) / rho)
            * np.exp(-((np.sqrt(1.0 - rho) * xi - a) ** 2) / (2.0 * rho) + 0.5 * xi ** 2))
    return np.where(inside, dens, 0.0)


def dfl_smooth_utility(alpha, pd_q, k, rho, dcfg: DecisionConfig):
    """Computes DFL forecast-based utility for proposed alpha allocation.
    
    Uses predicted default probability and Vasicek loss distribution to 
    calculate expected terminal capital net expected shortfall penalty.
    """

    # Loss threshold
    x = dfl_threshold_x(alpha, k, dcfg)

    # Expected excess loss
    es, _, _, _ = vasicek_es_exceed(x, pd_q, rho, dcfg)

    # Expected terminal capital - expected shortfall penalty
    return (dcfg.K1 + dcfg.ell * alpha * (dcfg.Y_BAR - dcfg.LGD * pd_q)
            - dcfg.lambda_shortfall * dcfg.ell * alpha * dcfg.LGD * es)


def dfl_foc(alpha, pd_q, k, rho, dcfg: DecisionConfig):
    """Computes first-order condition of the DFL utility wrt alpha.

    Uses the probability that Vasicek losses exceed allocation 
    dependent loss threshold to include marginal shortfall effect.
    """
    x = dfl_threshold_x(alpha, k, dcfg)
    es, p, _, _ = vasicek_es_exceed(x, pd_q, rho, dcfg)
    return (dcfg.ell * (dcfg.Y_BAR - dcfg.LGD * pd_q)
            - dcfg.lambda_shortfall * dcfg.ell * dcfg.LGD * es - (dcfg.lambda_shortfall / alpha) * p)


def dfl_context(pd_q, dcfg: DecisionConfig):
    """Converts predicted quarterly default probability into
    quantities required by allocation solver and implicit-gradient 
    calculation.

    Calculates annual PD, implied IRB capital requirement, Vasicek
    correlation and the derivatives of capital and correlation wrt 
    quarterly PD. Correlation derivative supports optional rho channel
    controlled by DFL_INCLUDE_RHO_CHANNEL.
    """
    pd_q = dec.clip_pd_q(pd_q, dcfg)
    # Convert to annual probability
    pd_a = dec.quarterly_pd_to_annual(pd_q, dcfg)
    # Calculate implied capital requirement
    k = dec.k_irb_from_annual_pd(pd_a, dcfg)

    # Get Vasicek asset correlation
    rho_raw = dcfg.RHO_SCALE * dec.basel_r_from_annual_pd(pd_a, dcfg)
    rho = np.clip(rho_raw, 1e-4, 0.999)

    # When clip binds, rho is locally constant in pd_q, so derivative is zero 
    at_clip = (rho_raw <= 1e-4) | (rho_raw >= 0.999)
    r_prime_a = dec.basel_r_prime(pd_a, dcfg)

    # Apply the chain rule through the quarterly-to-annual PD conversion
    rho_prime_q = np.where(at_clip, 0.0, dcfg.RHO_SCALE * r_prime_a * 4.0 * (1.0 - pd_q) ** 3)

    # Get K_IRB gradient
    kp = dec.k_irb_prime_quarterly(pd_q, dcfg)
    return pd_a, k, rho, kp, rho_prime_q


def solve_alpha_star(pd_q, dcfg: DecisionConfig, include_rho_channel: Optional[bool] = None):
    """Perform the forward and backward passes of the implicit allocation layer.
    
    The forward pass finds the optimal allocation alpha_star by bisection on
    the first-order condition. The backward pass computes d_alpha_star / d(pd_q) 
    using the implicit-function theorem.

    (bisection on the FOC) + backward (IFT) pass of the
    implicit allocation layer. Returns (alpha_star, dalpha_dpd).

    If include_rho_channel is True, the backward pass also accounts for the
    dependence of Vasicek correlation rho on pd_q.

    Full derivation in thesis.
    """

    use_rho_channel = DFL_INCLUDE_RHO_CHANNEL if include_rho_channel is None else include_rho_channel

    pd_q = dec.clip_pd_q(np.atleast_1d(np.asarray(pd_q, dtype=float)), dcfg)
    n = pd_q.shape[0]

    # Compute capital requirement, Vasicek corr and their derivatives
    _, k, rho, kp, rho_prime_q = dfl_context(pd_q, dcfg)

    lo = np.full(n, DFL_ALPHA_LO)
    hi = np.ones(n)

    # Evaluate FOC at both ends of the boundary
    f_lo = dfl_foc(lo, pd_q, k, rho, dcfg)
    f_hi = dfl_foc(hi, pd_q, k, rho, dcfg)
    # Classify boundary points
    at_one = f_hi >= 0.0
    at_zero = (~at_one) & (f_lo <= 0.0)
    # Classify interior points
    interior = ~(at_one | at_zero)

    a_lo, a_hi = lo.copy(), hi.copy()

    # FORWARD PASS
    # For interior points, repeatedly evaluate dfl_foc at midpoint
    for _ in range(DFL_BISECT_ITERS):
        mid = 0.5 * (a_lo + a_hi)
        f_mid = dfl_foc(mid, pd_q, k, rho, dcfg)
        go_up = f_mid > 0.0
        a_lo = np.where(interior & go_up, mid, a_lo)
        a_hi = np.where(interior & ~go_up, mid, a_hi)
    # Compute final alpha
    alpha = np.where(at_one, 1.0, np.where(at_zero, 0.0, 0.5 * (a_lo + a_hi)))
    a_safe = np.clip(alpha, DFL_ALPHA_LO, 1.0)

    # BACKWARD PASS
    # Compute quantities needed for the IFT gradient
    x = dfl_threshold_x(a_safe, k, dcfg)
    es, p, y_x, inside = vasicek_es_exceed(x, pd_q, rho, dcfg)
    f_l = vasicek_pdf(x, pd_q, rho, dcfg)
    a_probit = ndtri(np.clip(pd_q, dcfg.PD_Q_CLIP_LOW, 1.0 - dcfg.PD_Q_CLIP_LOW))
    sqrt_rho = np.sqrt(rho)
    psi = np.where(inside, norm.cdf((y_x - sqrt_rho * a_probit) / np.sqrt(1.0 - rho)),
                   np.where(x <= _X_EPS, 1.0, 0.0))
    
    # Denominator of IFT gradient
    dF_dalpha = -dcfg.lambda_shortfall * f_l / (dcfg.ell * a_safe ** 3 * dcfg.LGD)
    
    phi_a = np.maximum(norm.pdf(a_probit), 1e-300)
    dist_term = np.where(inside, norm.pdf(y_x) / (sqrt_rho * phi_a), 0.0)

    # d(rho)/d(pd_q) contribution to dF/d(pd_q) (not included in main thesis results)
    if use_rho_channel:
        # dES/d_rho * d_rho/dPD (rho_prime_q includes rho_scale and annualisation factor)
        # rho dependent part uses bivariate normal CDF (defined in decision.py)
        rho_channel_es = np.where(
            inside,
            dec.bivariate_normal_pdf(a_probit, y_x, sqrt_rho) / (2.0 * sqrt_rho) * rho_prime_q,
            0.0,
        )
        # rho channel contribution to derivative of P(L > x)
        rho_channel_phi = np.where(
            inside,
            norm.pdf(y_x) * (a_probit * sqrt_rho - y_x) / (2.0 * rho * (1.0 - rho)) * rho_prime_q,
            0.0,
        )
    # If rho channel excluded, these terms are set to zero
    else:
        rho_channel_es = 0.0
        rho_channel_phi = 0.0

    # Numerator of IFT gradient 
    dF_dpd = (-dcfg.ell * dcfg.LGD
              - dcfg.lambda_shortfall * dcfg.ell * dcfg.LGD * (p * kp / dcfg.LGD + psi + rho_channel_es)
              - (dcfg.lambda_shortfall / a_safe) * (f_l * kp / dcfg.LGD + dist_term + rho_channel_phi))
    
     # Calculates implicit gradient only for interior solutions.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Sets gradient to zero if optimum is at boundary or denominator is close to zero
        grad = np.where(interior & (np.abs(dF_dalpha) > 1e-300), -dF_dpd / dF_dalpha, 0.0)
    grad = np.clip(np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0),
                   -DFL_GRAD_CLIP, DFL_GRAD_CLIP)
    return alpha, grad


def make_implicit_allocation(dcfg: DecisionConfig):
    """Creates custom PyTorch autograd operation for DFL allocation layer.

    Forward pass computes the optimal allocation, while the backward pass
    uses the implicit-function-theorem gradient.
    """

    # Nested class has access to decision config
    class ImplicitAllocation(torch.autograd.Function):
        @staticmethod
        # Receives predicted PD and computes alpha star
        def forward(ctx, pd_q_tensor):
            pd_np = pd_q_tensor.detach().cpu().numpy().astype(float).ravel()
            alpha, grad = solve_alpha_star(pd_np, dcfg)
            # Save d(alpha_star)/d(pd_q) for use in backward.
            ctx.save_for_backward(torch.as_tensor(grad, dtype=pd_q_tensor.dtype))
            return torch.as_tensor(alpha, dtype=pd_q_tensor.dtype).reshape(pd_q_tensor.shape)

        @staticmethod
        # Combine the upstream gradient with the allocation sensitivity
        def backward(ctx, grad_output):
            (dalpha_dpd,) = ctx.saved_tensors
            return grad_output * dalpha_dpd.reshape(grad_output.shape)

    return ImplicitAllocation


def torch_allocation_utility(alpha, c_q, k, dcfg: DecisionConfig):
    """Compute realised allocation utility using PyTorch tensors.
    Supports autograd through the returned utility.
    """
    exposure = dcfg.ell * alpha * dcfg.K1
    terminal_capital = dcfg.K1 + exposure * (dcfg.Y_BAR - c_q)
    committed_capital = alpha * dcfg.K1
    available_buffer = terminal_capital - committed_capital
    required_capital = k * exposure
    shortfall = torch.clamp(required_capital - available_buffer, min=0.0)
    return terminal_capital - dcfg.lambda_shortfall * shortfall


def allocation_utility_vec(alpha, c_q, k, dcfg: DecisionConfig):
    """Vectorised version of torch_allocation_utility().
    """
    alpha = np.asarray(alpha, dtype=float)
    exposure = dcfg.ell * alpha * dcfg.K1
    terminal_capital = dcfg.K1 + exposure * (dcfg.Y_BAR - np.asarray(c_q, dtype=float))
    available_buffer = terminal_capital - alpha * dcfg.K1
    shortfall = np.maximum(np.asarray(k, dtype=float) * exposure - available_buffer, 0.0)
    return terminal_capital - dcfg.lambda_shortfall * shortfall


# ---------------------------------------------------------------------------
# Model training loop
# ---------------------------------------------------------------------------

class PDForecaster(nn.Module):
    def __init__(self, input_dim, model_type, hidden_dim=16):
        super().__init__()
        if model_type == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif model_type == "mlp":
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        else:
            raise ValueError("model_type must be 'linear' or 'mlp'.")

    def forward(self, x):
        return self.net(x).squeeze(-1)


def init_output_bias_for_target(model, y_pre_val, dcfg: DecisionConfig):
    """Initialises model's output bias so predicted PD is close to average target PD
        prior to training instead of 0.5."""
    
    y_mean = float(np.clip(np.mean(y_pre_val), dcfg.PD_Q_CLIP_LOW, 1.0 - dcfg.PD_Q_CLIP_LOW))
    logit_mean = float(np.log(y_mean / (1.0 - y_mean)))
    with torch.no_grad():
        if isinstance(model.net, nn.Linear):
            model.net.bias.fill_(logit_mean)
        else:
            model.net[-1].bias.fill_(logit_mean)


def predict_pd_q(model, x, dcfg: DecisionConfig):
    """Get trained model predictions from input data."""
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).cpu().numpy()
    return dec.clip_pd_q(np.atleast_1d(pred), dcfg)


def predict_pd_q_ensemble(models, x, dcfg: DecisionConfig):
    preds = np.stack([predict_pd_q(m, x, dcfg) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def train_dfl_model(x_train, y_train_pd_q, c_next, k_next, u_oracle_next, model_type,
                     model_cfg: ModelConfig, dcfg: DecisionConfig, implicit_alloc_cls,
                     seed=None, return_history=False):
    """Train a PD forecaster using the DFL regret objective.

    The predicted PD is passed through the implicit allocation layer, and the
    model is trained by minimising mean realised regret.
    """
    seed = model_cfg.seed if seed is None else int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = x_train.shape[0]
    val_size = max(1, int(np.floor(model_cfg.val_frac * n)))
    train_size = n - val_size
    if train_size < 8:
        raise ValueError("Training set too small after validation split.")

    def to_t(arr):
        return torch.tensor(np.asarray(arr, dtype=float), dtype=torch.float32)

    x_tr, x_val = x_train[:train_size], x_train[train_size:]
    c_tr, c_val = c_next[:train_size], c_next[train_size:]
    k_tr, k_val = k_next[:train_size], k_next[train_size:]
    u_tr, u_val = u_oracle_next[:train_size], u_oracle_next[train_size:]
    y_val = y_train_pd_q[train_size:]

    train_ds = TensorDataset(to_t(x_tr), to_t(c_tr), to_t(k_tr), to_t(u_tr))
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=model_cfg.batch_size, shuffle=True, generator=loader_gen)

    model = PDForecaster(input_dim=x_train.shape[1], model_type=model_type, hidden_dim=model_cfg.hidden_dim)
    init_output_bias_for_target(model, y_train_pd_q[:train_size], dcfg)
    optimiser = torch.optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)

    # Define loss using implicit allocation
    def regret_loss(score, cb, kb, ub):
        # Convert the model score into a clipped quarterly PD forecast
        pd_q = torch.sigmoid(score).clamp(dcfg.PD_Q_CLIP_LOW, dcfg.PD_Q_CLIP_HIGH)
        # Compute optimal allocation and retain custom backward pass
        alpha = implicit_alloc_cls.apply(pd_q) # enters into computation graph

        return torch.mean(ub - torch_allocation_utility(alpha, cb, kb, dcfg))

    x_val_t, c_val_t, k_val_t, u_val_t = to_t(x_val), to_t(c_val), to_t(k_val), to_t(u_val)
    best_state, best_val, stale_epochs, history = None, float("inf"), 0, []

    for epoch in range(model_cfg.epochs):
        model.train()
        batch_losses = []
        for xb, cb, kb, ub in train_loader:
            optimiser.zero_grad()
            # Backpropagate through the implicit allocation layer
            loss = regret_loss(model(xb), cb, kb, ub)
            loss.backward()
            optimiser.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            score_val = model(x_val_t)
            pd_val = torch.sigmoid(score_val).clamp(dcfg.PD_Q_CLIP_LOW, dcfg.PD_Q_CLIP_HIGH).cpu().numpy()
        
        # Compute validation allocations using the same bisection solver as training
        alpha_val, _ = solve_alpha_star(pd_val, dcfg)
        val_regret = float(np.mean(u_val - allocation_utility_vec(alpha_val, c_val, k_val, dcfg)))
        val_rmse_pd_q = float(np.sqrt(mean_squared_error(y_val, pd_val)))
        val_mae_pd_q = float(mean_absolute_error(y_val, pd_val))
        history.append({"epoch": epoch + 1, "model": f"dfl_{model_type}", "seed": seed,
                        "train_loss": float(np.mean(batch_losses)), "val_loss": val_regret,
                        "val_rmse_pd_q": val_rmse_pd_q, "val_mae_pd_q": val_mae_pd_q,
                        "val_mean_pd_q_hat": float(np.mean(pd_val)),
                        "val_mean_alpha": float(np.mean(alpha_val))})

        # Check early stopping conditions
        if val_regret < best_val - 1e-12:
            best_val, stale_epochs = val_regret, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= model_cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, pd.DataFrame(history)) if return_history else model


def train_dfl_ensemble(x_train, y_train_pd_q, c_next, k_next, u_oracle_next, model_cfg: ModelConfig,
                        dcfg: DecisionConfig, implicit_alloc_cls, seeds: Sequence[int],
                        return_history=False):
    """Loop over ensemble seeds to train independently initialised MLP per seed."""
    models, first_hist = [], None
    for i, s in enumerate(seeds):
        want_hist = return_history and i == 0
        out = train_dfl_model(x_train, y_train_pd_q, c_next, k_next, u_oracle_next, "mlp",
                              model_cfg, dcfg, implicit_alloc_cls, seed=s, return_history=want_hist)
        if want_hist:
            model, first_hist = out
        else:
            model = out
        models.append(model)
    return models, first_hist


def build_result_row_dfl(base_row, pd_q_hat, model_name, split_name, dcfg: DecisionConfig,
                          pd_q_hat_std=float("nan")):
    """Builds evaluation row for each PTO forecast."""
    pd_q_hat = float(dec.clip_pd_q(pd_q_hat, dcfg))
    c_q_hat = float(dcfg.LGD * pd_q_hat)
    pd_a_hat = float(dec.quarterly_pd_to_annual(pd_q_hat, dcfg))
    k_irb_hat = float(dec.k_irb_from_annual_pd(pd_a_hat, dcfg))
    c_q_realised = float(base_row["c_q_realised"])
    k_true = float(base_row["k_irb_true_next"])

    _alpha, _ = solve_alpha_star(np.array([pd_q_hat]), dcfg)
    alpha_hat = float(_alpha[0])
    alpha_oracle, utility_oracle = dec.optimise_alpha_oracle(c_q=c_q_realised, k=k_true, cfg=dcfg)
    realised = dec.decision_components(alpha_hat, c_q=c_q_realised, k=k_true, cfg=dcfg)
    predicted = dec.decision_components(alpha_hat, c_q=c_q_hat, k=k_irb_hat, cfg=dcfg)
    regret = utility_oracle - realised["utility"]
    utility_bayes_realised = float(base_row["utility_bayes_realised_next"])

    return {"split": split_name, "model": model_name, "date": pd.to_datetime(base_row["target_date"]),
            "feature_date": pd.to_datetime(base_row[DATE_COL]),
            "c_q_realised": c_q_realised, "pd_q_realised": float(base_row["pd_q_realised"]),
            "pd_q_true": float(base_row["pd_q_true_next"]), "pd_a_true": float(base_row["pd_a_true_next"]),
            "k_irb_true": k_true,
            "pd_q_hat": pd_q_hat, "pd_q_hat_std": float(pd_q_hat_std),
            "pd_a_hat": pd_a_hat, "c_q_hat": c_q_hat, "k_irb_hat": k_irb_hat,
            "alpha_oracle": alpha_oracle, "alpha_bayes": float(base_row["alpha_bayes_next"]),
            "alpha_hat": alpha_hat,
            "utility_oracle": utility_oracle, "utility_bayes_realised": utility_bayes_realised,
            "utility": realised["utility"],
            "regret": regret,
            "regret_vs_bayes": utility_bayes_realised - realised["utility"],
            "regret_irreducible": utility_oracle - utility_bayes_realised,
            "alpha_oracle_panel": float(base_row["alpha_oracle_next"]),
            "utility_oracle_panel": float(base_row["utility_oracle_next"]),
            "shortfall_realised_at_hat": realised["shortfall"],
            "shortfall_predicted_at_hat": predicted["shortfall"]}



def run_expanding_window_dfl(df: pd.DataFrame, feature_cols: List[str], split_name: str, split_cfg: dict,
                              architecture: str, dcfg: DecisionConfig, model_cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
                              refit_every: int = 8, mlp_seeds: Sequence[int] = (42, 43, 44, 45, 46),
                              return_history: bool = True, stability_reference_X: Optional[np.ndarray] = None,
                              progress_every: int = 25):
    
    """Train + forecast one DFL architecture. Matches PTO/BB setup

    Returns (forecast_long, training_history, stability_linear_df, stability_mlp_df).
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown DFL architecture '{architecture}'. Expected one of {ARCHITECTURES}.")
    model_name = f"dfl_{architecture}"
    implicit_alloc_cls = make_implicit_allocation(dcfg)

    first_test_idx = dec.resolve_first_test_idx(df, split_cfg)
    total = len(df) - first_test_idx
    n_refits = int(np.ceil(total / refit_every))

    # Progress updates
    print(f"[dfl/{architecture}][{split_name}] first test target: "
          f"{pd.Timestamp(df['target_date'].iloc[first_test_idx]).date()}, {total} windows, "
          f"{n_refits} refits (every {refit_every}), {first_test_idx} initial training obs", flush=True)

    records, histories = [], []
    stability_linear_rows, stability_mlp_rows = [], []
    scaler, model_or_ensemble = None, None
    t0 = time.time()

    # Loop over test steps
    for step, test_idx in enumerate(range(first_test_idx, len(df))):
        test_row = df.iloc[test_idx]
        # Retrain only every refit_every steps
        if step % refit_every == 0:
            train_df = df.iloc[:test_idx]
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
            y_train = train_df["target_pd_q_next"].to_numpy(dtype=float)
            c_next_tr = train_df["c_q_realised"].to_numpy(dtype=float)
            k_next_tr = train_df["k_irb_true_next"].to_numpy(dtype=float)
            u_oracle_tr = train_df["utility_oracle_next"].to_numpy(dtype=float)

            if architecture == "linear":
                model_or_ensemble, hist = train_dfl_model(
                    x_train, y_train, c_next_tr, k_next_tr, u_oracle_tr, "linear",
                    model_cfg, dcfg, implicit_alloc_cls, return_history=True)
                if stability_reference_X is not None:
                    x_ref = scaler.transform(stability_reference_X)
                    ref_pd = predict_pd_q(model_or_ensemble, x_ref, dcfg)
                    ref_alpha, _ = solve_alpha_star(ref_pd, dcfg)
                    row = {"refit_step": step, "test_date": pd.to_datetime(test_row["target_date"]),
                           "train_end_date": pd.to_datetime(train_df[DATE_COL].iloc[-1]),
                           "n_train": len(train_df)}
                    for i, (m, a) in enumerate(zip(ref_pd, ref_alpha)):
                        row[f"ref{i}_pd_q_hat_mean"] = float(m)
                        row[f"ref{i}_alpha_hat_mean"] = float(a)
                    stability_linear_rows.append(row)
            else:
                model_or_ensemble, hist = train_dfl_ensemble(
                    x_train, y_train, c_next_tr, k_next_tr, u_oracle_tr, model_cfg, dcfg,
                    implicit_alloc_cls, seeds=mlp_seeds, return_history=True)
                if stability_reference_X is not None:
                    x_ref = scaler.transform(stability_reference_X)
                    member_pd = np.stack([predict_pd_q(m, x_ref, dcfg)
                                          for m in model_or_ensemble], axis=0)
                    ref_mean = member_pd.mean(axis=0)
                    ref_std = member_pd.std(axis=0)
                    ref_alpha, _ = solve_alpha_star(ref_mean, dcfg)
                    row = {"refit_step": step, "test_date": pd.to_datetime(test_row["target_date"]),
                           "train_end_date": pd.to_datetime(train_df[DATE_COL].iloc[-1]),
                           "n_train": len(train_df)}
                    for i, (m, s, a) in enumerate(zip(ref_mean, ref_std, ref_alpha)):
                        row[f"ref{i}_pd_q_hat_mean"] = float(m)
                        row[f"ref{i}_pd_q_hat_std"] = float(s)
                        row[f"ref{i}_alpha_hat_mean"] = float(a)
                        for seed, pred in zip(mlp_seeds, member_pd[:, i]):
                            row[f"ref{i}_pd_q_hat_seed{int(seed)}"] = float(pred)
                            seed_alpha, _ = solve_alpha_star(np.array([pred]), dcfg)
                            row[f"ref{i}_alpha_hat_seed{int(seed)}"] = float(seed_alpha[0])
                    stability_mlp_rows.append(row)

            if return_history and hist is not None:
                hist = hist.copy()
                hist["split"] = split_name
                hist["test_date"] = pd.to_datetime(test_row["target_date"])
                hist["train_end_date"] = pd.to_datetime(train_df[DATE_COL].iloc[-1])
                histories.append(hist)

        x_test = scaler.transform(test_row[feature_cols].to_numpy(dtype=float).reshape(1, -1))
        if architecture == "linear":
            pd_hat = float(predict_pd_q(model_or_ensemble, x_test, dcfg)[0])
            records.append(build_result_row_dfl(test_row, pd_hat, model_name, split_name, dcfg))
        else:
            dfl_mean, dfl_std = predict_pd_q_ensemble(model_or_ensemble, x_test, dcfg)
            records.append(build_result_row_dfl(test_row, float(dfl_mean[0]), model_name, split_name, dcfg,
                                                pd_q_hat_std=float(dfl_std[0])))

        done = step + 1
        if done % progress_every == 0 or test_idx == len(df) - 1:
            print(f"[dfl/{architecture}][{split_name}] completed {done}/{total} windows "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    pred_long = pd.DataFrame(records).sort_values(["date", "model"]).reset_index(drop=True)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    stability_linear = pd.DataFrame(stability_linear_rows)
    stability_mlp = pd.DataFrame(stability_mlp_rows)
    return pred_long, history, stability_linear, stability_mlp


def summarise_dfl_run(pred_long: pd.DataFrame, model_name: str) -> dict:
    g = pred_long[pred_long["model"] == model_name]
    err_real = g["pd_q_hat"].to_numpy() - g["pd_q_realised"].to_numpy()
    err_true = g["pd_q_hat"].to_numpy() - g["pd_q_true"].to_numpy()
    return {
        "n": int(len(g)),
        "rmse_vs_realised": float(np.sqrt(np.mean(err_real ** 2))),
        "rmse_vs_true": float(np.sqrt(np.mean(err_true ** 2))),
        "mae_vs_true": float(np.mean(np.abs(err_true))),
        "bias_vs_true": float(np.mean(err_true)),
        "mean_regret": float(g["regret"].mean()),
        "median_regret": float(g["regret"].median()),
        "mean_regret_vs_bayes": float(g["regret_vs_bayes"].mean()),
        "mean_regret_irreducible": float(g["regret_irreducible"].mean()),
        "mean_utility": float(g["utility"].mean()),
        "shortfall_bind_share": float((g["shortfall_realised_at_hat"] > 1e-12).mean()),
    }

