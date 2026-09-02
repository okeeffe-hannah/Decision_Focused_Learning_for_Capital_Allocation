"""Predict-then-optimise (PTO) train and evaluation core functions"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
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
ARCHITECTURES = ["linear", "mlp", "persistence", "persistence_ma4"]
NO_TRAIN_ARCHITECTURES = {"persistence", "persistence_ma4"}


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
# Model, training loss, training loop
# ---------------------------------------------------------------------------

def inverse_normal_np(p, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    # Clips realised x to (0,1) so inverse norm is finite 
    p = np.clip(np.asarray(p, dtype=float), cfg.PD_Q_CLIP_LOW, 1.0 - cfg.PD_Q_CLIP_LOW)
    # Standard inverse normal
    return norm.ppf(p)


def torch_vasicek_charge_off_log_density(z_realised_loss, pd_q, rho):

    # Pytorch inverse standard-normal CDF (pd_q is pytorch tensor)
    a = torch.special.ndtri(pd_q)

    # Return Vasicek log density of realised default rate
    return (0.5 * torch.log((1.0 - rho) / rho)
            - (torch.sqrt(1.0 - rho) * z_realised_loss - a) ** 2 / (2.0 * rho)
            + 0.5 * z_realised_loss ** 2)


def torch_vasicek_charge_off_nll(score, z_realised_loss, cfg: DecisionConfig, rho_fixed=None):
    """Negative log-likelihood of the realised quarterly charge-off rate under its
    closed-form Vasicek density.
    
    z_realised_loss: inverse_normal(PD^{realised}_q) which is the transformed observed target
    score: raw, unbounded output from PTO forecasting model (needs to be converted to get predicted PD)
    """

    # Transform score to predicted quarterly PD from logit-scale prediction
    pd_q = torch.sigmoid(score).clamp(cfg.PD_Q_CLIP_LOW, cfg.PD_Q_CLIP_HIGH)

    # Check if fixed rho value is given
    if rho_fixed is not None:
        rho = torch.as_tensor(float(rho_fixed))
    
    # Otherwise use rho_scale multipled by Basel correlation
    else:
        pd_a = (1.0 - torch.pow(1.0 - pd_q, 4.0)).clamp(cfg.PD_A_CLIP_LOW, cfg.PD_A_CLIP_HIGH)
        exp_term = (1.0 - torch.exp(-50.0 * pd_a)) / (1.0 - float(np.exp(-50.0)))
        rho = 0.12 * exp_term + 0.24 * (1.0 - exp_term)
        # Clips rho to ensure rho is between 0 and 1 (keeps f_L > 0)
        rho = torch.clamp(cfg.RHO_SCALE * rho, 1e-4, 0.999)

    # Compute Vasicek log density
    log_f_L = torch_vasicek_charge_off_log_density(z_realised_loss, pd_q, rho)

    # Convert to charge off log density
    log_f_C = log_f_L - float(np.log(cfg.LGD))

    # Return mean of negative log likelihood
    return torch.mean(-log_f_C)


class PDForecaster(nn.Module):
    """Outputs logit-scale raw score, predicted quarterly PD is sigmoid(score)."""

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


def init_output_bias_for_target(model, y_pre_val, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Initialises model's output bias so predicted PD is close to average target PD
        prior to training instead of 0.5."""
    
    # Calculates average target value
    y_mean = float(np.clip(np.mean(y_pre_val), cfg.PD_Q_CLIP_LOW, 1.0 - cfg.PD_Q_CLIP_LOW))

    # Apply inverse sigmoid
    logit_mean = float(np.log(y_mean / (1.0 - y_mean)))

    # Only changes output bias 
    with torch.no_grad():
        # Only layer for linear
        if isinstance(model.net, nn.Linear):
            model.net.bias.fill_(logit_mean)
        # MLP last layer
        else:
            model.net[-1].bias.fill_(logit_mean)


def train_pd_model(x_train, y_train_pd_q, model_type, model_cfg: ModelConfig, dcfg: DecisionConfig,
                    seed=None, return_history=False):
    """Main training loop for PTO model."""

    # Use supplied seed or configured default seed 
    seed = model_cfg.seed if seed is None else int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = x_train.shape[0]
    val_size = max(1, int(np.floor(model_cfg.val_frac * n)))
    train_size = n - val_size
    if train_size < 8:
        raise ValueError("Training set too small after validation split.")

    x_tr, x_val = x_train[:train_size], x_train[train_size:]
    # y_tr is realised quarterly default rate (already divided c/LGD)
    y_tr, y_val = y_train_pd_q[:train_size], y_train_pd_q[train_size:] 
    # apply inverse norm for NLL training 
    t_tr, t_val = inverse_normal_np(y_tr, dcfg), inverse_normal_np(y_val, dcfg)

    train_ds = TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(t_tr, dtype=torch.float32))
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=model_cfg.batch_size, shuffle=True, generator=loader_gen)

    # Initialise model, output bias and optimiser
    model = PDForecaster(input_dim=x_train.shape[1], model_type=model_type, hidden_dim=model_cfg.hidden_dim)
    init_output_bias_for_target(model, y_tr, dcfg)
    optimiser = torch.optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)


    # Setup up Vasicek NLL loss
    def compute_loss(score, target):
        return torch_vasicek_charge_off_nll(score, target, dcfg)

    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    t_val_t = torch.tensor(t_val, dtype=torch.float32)

    best_state, best_val, stale_epochs, history = None, float("inf"), 0, []

    # For each epoch
    for epoch in range(model_cfg.epochs):
        model.train()
        batch_losses = []

        # For each mini batch
        for xb, tb in train_loader:
            optimiser.zero_grad()
            loss = compute_loss(model(xb), tb) # Forward pass and NLL
            loss.backward() # Calculate grads
            optimiser.step() # Update params
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            score_val = model(x_val_t) # Produce validation scores
            val_loss = compute_loss(score_val, t_val_t).item() # Score validation preds using NLL 
            pd_q_val_np = dec.clip_pd_q(torch.sigmoid(score_val).cpu().numpy(), dcfg) # store predicted PDs

        history.append({"epoch": epoch + 1, "model": model_type, "seed": seed, 
                        "train_loss": float(np.mean(batch_losses)), "val_loss": float(val_loss),
                        "val_rmse_pd_q": float(np.sqrt(mean_squared_error(y_val, pd_q_val_np))),
                        "val_mae_pd_q": float(mean_absolute_error(y_val, pd_q_val_np)),
                        "val_mean_pd_q_hat": float(np.mean(pd_q_val_np))})

        # If validation NLL improves, save model params otherwise inc counter
        if val_loss < best_val - 1e-12:
            best_val, stale_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        # Early stopping condition
        if stale_epochs >= model_cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, pd.DataFrame(history)) if return_history else model


def predict_pd_q(model, x, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    """Prediction helper for PTO model. Used for stability ref points and 
    out-of-sample forecasts."""
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).cpu().numpy()
    return dec.clip_pd_q(np.atleast_1d(pred), cfg)


def train_mlp_ensemble(x_train, y_train_pd_q, model_cfg: ModelConfig, dcfg: DecisionConfig,
                        seeds: Sequence[int], return_history=False):
    """Loop over ensemble seeds to train independently initialised MLP per seed."""
    models, first_hist = [], None
    for i, s in enumerate(seeds):
        want_hist = return_history and i == 0
        out = train_pd_model(x_train, y_train_pd_q, "mlp", model_cfg, dcfg, seed=s, return_history=want_hist)
        if want_hist:
            model, first_hist = out
        else:
            model = out
        models.append(model)
    return models, first_hist


def predict_pd_q_ensemble(models, x, cfg: DecisionConfig = DEFAULT_DECISION_CONFIG):
    preds = np.stack([predict_pd_q(m, x, cfg) for m in models], axis=0)
    # Return ensemble mean prediction
    return preds.mean(axis=0), preds.std(axis=0)


def add_alpha_to_stability(stability_df: pd.DataFrame, dcfg: DecisionConfig) -> pd.DataFrame:
    """Adds optimal alphas to the stability table for each reference point.
    Allows stability to be examined in prediction space and decision space."""
    if stability_df is None or stability_df.empty:
        return stability_df
    pattern = re.compile(r"^ref(\d+)_pd_q_hat_mean$")
    ref_cols = [c for c in stability_df.columns if pattern.match(c)]
    out = stability_df.copy()
    new_cols = {
        f"ref{pattern.match(col).group(1)}_alpha_hat_mean": out[col].apply(
            lambda p: float(dec.optimise_alpha_expected(float(p), dcfg)[0]))
        for col in ref_cols
    }
    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def build_result_row(base_row, pd_q_hat, model_name, split_name, dcfg: DecisionConfig,
                      pd_q_hat_std=float("nan")):
    """Builds evaluation row for each PTO forecast."""

    # Forecast-based values
    pd_q_hat = float(dec.clip_pd_q(pd_q_hat, dcfg))
    c_q_hat = float(dcfg.LGD * pd_q_hat)
    pd_a_hat = float(dec.quarterly_pd_to_annual(pd_q_hat, dcfg))
    k_irb_hat = float(dec.k_irb_from_annual_pd(pd_a_hat, dcfg))
    c_q_realised = float(base_row["c_q_realised"])
    k_true = float(base_row["k_irb_true_next"])

    # Decision values and compute regret
    alpha_hat, _ = dec.optimise_alpha_expected(pd_q_hat, dcfg)
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


# ---------------------------------------------------------------------------
# Expanding-window evaluation, one architecture at a time
# ---------------------------------------------------------------------------

def training_relevant_config(dcfg: DecisionConfig, refit_every: int, mlp_seeds: Sequence[int],
                              model_cfg: ModelConfig) -> dict:
    """Returns the settings that affect PTO training. 
    
    Used to determine if an existing cached PTO forecast is valid. Changes that only affect the
    decision (eg lambda) do not require the forecasting model to be retrained."""
    return {
        "LGD": dcfg.LGD, "RHO_SCALE": dcfg.RHO_SCALE,
        "PD_Q_CLIP_LOW": dcfg.PD_Q_CLIP_LOW, "PD_Q_CLIP_HIGH": dcfg.PD_Q_CLIP_HIGH,
        "PD_A_CLIP_LOW": dcfg.PD_A_CLIP_LOW, "PD_A_CLIP_HIGH": dcfg.PD_A_CLIP_HIGH,
        "refit_every": int(refit_every), "mlp_seeds": list(mlp_seeds),
        # Increment if saved stability-audit format changes invalidating old caches
        "stability_seed_artifacts_version": 1,
        "model_cfg": {"hidden_dim": model_cfg.hidden_dim, "lr": model_cfg.lr,
                      "weight_decay": model_cfg.weight_decay, "batch_size": model_cfg.batch_size,
                      "epochs": model_cfg.epochs, "patience": model_cfg.patience,
                      "val_frac": model_cfg.val_frac, "seed": model_cfg.seed},
    }


def run_expanding_window_pto_forecast(df: pd.DataFrame, feature_cols: List[str], split_name: str, split_cfg: dict,
                                       architecture: str, dcfg: DecisionConfig,
                                       model_cfg: ModelConfig = DEFAULT_MODEL_CONFIG, refit_every: int = 8,
                                       mlp_seeds: Sequence[int] = (42, 43, 44, 45, 46), return_history: bool = True,
                                       stability_reference_X: Optional[np.ndarray] = None, progress_every: int = 25):
    """Train + forecast one PTO architecture. 

    Returns (forecast_long, training_history, stability_linear_df, stability_mlp_df).
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown PTO architecture '{architecture}'. Expected one of {ARCHITECTURES}.")

    # Get first test index based on train_frac (simulation) or starting test date (empirical)
    first_test_idx = dec.resolve_first_test_idx(df, split_cfg)
    total = len(df) - first_test_idx

    # Prints number of refits to let user know progress updates
    n_refits = int(np.ceil(total / refit_every)) if architecture not in NO_TRAIN_ARCHITECTURES else 0
    print(f"[pto/{architecture}][{split_name}] first test target: "
          f"{pd.Timestamp(df['target_date'].iloc[first_test_idx]).date()}, {total} windows, "
          f"{n_refits} refits (every {refit_every}), {first_test_idx} initial training obs", flush=True)

    records, histories = [], []
    stability_linear_rows, stability_mlp_rows = [], []
    scaler, model_or_ensemble = None, None
    t0 = time.time()

    for step, test_idx in enumerate(range(first_test_idx, len(df))):
        test_row = df.iloc[test_idx]

        # Two persistence baselines do not need to be trained.
        if architecture in NO_TRAIN_ARCHITECTURES:
            if architecture == "persistence":
                pd_q_hat = float(dec.clip_pd_q(test_row["c"] / dcfg.LGD, dcfg))
            else:  # persistence_ma4 - moving 4 quarter average model
                c_ma4 = float(df["c"].iloc[max(0, test_idx - 3):test_idx + 1].mean())
                pd_q_hat = float(dec.clip_pd_q(c_ma4 / dcfg.LGD, dcfg))
            records.append({"split": split_name, "model": architecture,
                            "date": pd.to_datetime(test_row["target_date"]),
                            "feature_date": pd.to_datetime(test_row[DATE_COL]),
                            "pd_q_hat": pd_q_hat, "pd_q_hat_std": float("nan")})
        else:
            # Only retrain every refit number of steps
            if step % refit_every == 0:
                train_df = df.iloc[:test_idx]
                scaler = StandardScaler()
                x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
                y_train = train_df["target_pd_q_next"].to_numpy(dtype=float)

                if architecture == "linear":
                    # Train linear model
                    model_or_ensemble, hist = train_pd_model(
                        x_train, y_train, "linear", model_cfg, dcfg, return_history=True)
                    # Record predictions at each refit if stability ref points provided 
                    if stability_reference_X is not None:
                        x_ref = scaler.transform(stability_reference_X)
                        # Get reference predictions
                        ref_pd = predict_pd_q(model_or_ensemble, x_ref, dcfg)
                        row = {"refit_step": step, "test_date": pd.to_datetime(test_row["target_date"]),
                               "train_end_date": pd.to_datetime(train_df[DATE_COL].iloc[-1]),
                               "n_train": len(train_df)}
                        for i, m in enumerate(ref_pd):
                            row[f"ref{i}_pd_q_hat_mean"] = float(m)
                        stability_linear_rows.append(row)
                else:  # mlp
                    model_or_ensemble, hist = train_mlp_ensemble(
                        x_train, y_train, model_cfg, dcfg, seeds=mlp_seeds, return_history=True)
                    if stability_reference_X is not None:
                        x_ref = scaler.transform(stability_reference_X)

                        # Saves both ensemble mean and individual training seed preds for each stability ref
                        member_pd = np.stack([predict_pd_q(m, x_ref, dcfg)
                                              for m in model_or_ensemble], axis=0)
                        ref_mean = member_pd.mean(axis=0)
                        ref_std = member_pd.std(axis=0)
                        row = {"refit_step": step, "test_date": pd.to_datetime(test_row["target_date"]),
                               "train_end_date": pd.to_datetime(train_df[DATE_COL].iloc[-1]),
                               "n_train": len(train_df)}
                        for i, (m, s) in enumerate(zip(ref_mean, ref_std)):
                            row[f"ref{i}_pd_q_hat_mean"] = float(m)
                            row[f"ref{i}_pd_q_hat_std"] = float(s)
                            for seed, pred in zip(mlp_seeds, member_pd[:, i]):
                                row[f"ref{i}_pd_q_hat_seed{int(seed)}"] = float(pred)
                                row[f"ref{i}_alpha_hat_seed{int(seed)}"] = float(
                                    dec.optimise_alpha_expected(float(pred), dcfg)[0])
                        # Compute ensemble mean implied alpha 
                        for i, m in enumerate(ref_mean):
                            row[f"ref{i}_alpha_hat_mean"] = float(
                                dec.optimise_alpha_expected(float(m), dcfg)[0])
                        stability_mlp_rows.append(row)

                if return_history and hist is not None:
                    hist = hist.copy()
                    hist["split"] = split_name
                    hist["test_date"] = pd.to_datetime(test_row["target_date"])
                    hist["train_end_date"] = pd.to_datetime(train_df[DATE_COL].iloc[-1])
                    histories.append(hist)

            # Get the test x_vals
            x_test = scaler.transform(test_row[feature_cols].to_numpy(dtype=float).reshape(1, -1))
            if architecture == "linear":
                # Test predictions
                pd_hat = float(predict_pd_q(model_or_ensemble, x_test, dcfg)[0])
                records.append({"split": split_name, "model": "linear",
                                "date": pd.to_datetime(test_row["target_date"]),
                                "feature_date": pd.to_datetime(test_row[DATE_COL]),
                                "pd_q_hat": pd_hat, "pd_q_hat_std": float("nan")})
            else:
                # Returns ensemble mean and stdev
                mlp_mean, mlp_std = predict_pd_q_ensemble(model_or_ensemble, x_test, dcfg)
                records.append({"split": split_name, "model": "mlp",
                                "date": pd.to_datetime(test_row["target_date"]),
                                "feature_date": pd.to_datetime(test_row[DATE_COL]),
                                "pd_q_hat": float(mlp_mean[0]), "pd_q_hat_std": float(mlp_std[0])})
        # Progress update
        done = step + 1
        if done % progress_every == 0 or test_idx == len(df) - 1:
            print(f"[pto/{architecture}][{split_name}] completed {done}/{total} windows "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    # Contains all out-of-sample forecast dates
    forecast_long = pd.DataFrame(records).sort_values(["date", "model"]).reset_index(drop=True)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    stability_linear = pd.DataFrame(stability_linear_rows)
    stability_mlp = pd.DataFrame(stability_mlp_rows)
    return forecast_long, history, stability_linear, stability_mlp


def score_pto_forecast(forecast_long: pd.DataFrame, scored_df: pd.DataFrame, dcfg: DecisionConfig,
                        split_name: str) -> pd.DataFrame:
    """ Adds decision variables and realised evaluation metrics to PTO forecasts."""
    base_by_date = scored_df.set_index("target_date", drop=False)
    records = []
    for _, r in forecast_long.iterrows():
        base_row = base_by_date.loc[r["date"]]
        records.append(build_result_row(base_row, r["pd_q_hat"], r["model"], split_name, dcfg,
                                        pd_q_hat_std=r["pd_q_hat_std"]))
    return pd.DataFrame(records).sort_values(["date", "model"]).reset_index(drop=True)


def run_expanding_window_pto(df: pd.DataFrame, feature_cols: List[str], split_name: str, split_cfg: dict,
                              architecture: str, dcfg: DecisionConfig, model_cfg: ModelConfig = DEFAULT_MODEL_CONFIG,
                              refit_every: int = 8, mlp_seeds: Sequence[int] = (42, 43, 44, 45, 46),
                              return_history: bool = True, stability_reference_X: Optional[np.ndarray] = None,
                              progress_every: int = 25):
    """Wrapper that calls run_expanding_window_pto_forecast and score_pto_forecast
    for main PTO functionality (keeps those two functions separate for when only
    one is needed).
    """
    forecast_long, history, stability_linear, stability_mlp = run_expanding_window_pto_forecast(
        df, feature_cols, split_name, split_cfg, architecture, dcfg, model_cfg, refit_every,
        mlp_seeds, return_history, stability_reference_X, progress_every)
    pred_long = score_pto_forecast(forecast_long, df, dcfg, split_name)
    return pred_long, history, stability_linear, stability_mlp


# ---------------------------------------------------------------------------
# Single-architecture metrics summary (panel-level metrics row)
# ---------------------------------------------------------------------------

def summarise_pto_run(pred_long: pd.DataFrame, model_name: str) -> dict:
    """Creates dictionary of panel-level metrics for a single architecture's 
    predictions. Appended to master metrics table."""
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
