"""Black-box end-to-end model pipeline"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Load and align the simulated panel
# ---------------------------------------------------------------------------

def load_simulated_panel(panel_csv, feature_cols, horizon=HORIZON):
    df = pd.read_csv(panel_csv)
    df[DATE_COL] = pd.PeriodIndex(df[DATE_COL], freq="Q").to_timestamp()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing feature columns: {missing}")

    df["target_date"] = df[DATE_COL].shift(-horizon)
    df["c_q_realised"] = df["c"].shift(-horizon)
    df["k_irb_true_next"] = df["K_IRB_true"].shift(-horizon)
    df["alpha_oracle_next"] = df["alpha_oracle"].shift(-horizon)
    df["utility_oracle_next"] = df["utility_oracle"].shift(-horizon)
    df["alpha_bayes_next"] = df["alpha_bayes"].shift(-horizon)
    df["utility_bayes_realised_next"] = df["utility_bayes_realised"].shift(-horizon)

    required = list(feature_cols) + ["target_date", "c_q_realised", "k_irb_true_next",
                                     "alpha_oracle_next", "utility_oracle_next",
                                     "alpha_bayes_next", "utility_bayes_realised_next"]
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} usable observations; check the panel.")
    return df

# ---------------------------------------------------------------------------
# Model, training loss, training loop
# ---------------------------------------------------------------------------

def torch_allocation_utility(alpha, c_q, k, dcfg: DecisionConfig):
    """Differentiable counterpart of decision.allocation_utility."""
    exposure = dcfg.ell * alpha * dcfg.K1
    terminal_capital = dcfg.K1 + exposure * (dcfg.Y_BAR - c_q)
    committed_capital = alpha * dcfg.K1
    available_buffer = terminal_capital - committed_capital
    required_capital = k * exposure
    # Applies clamp at shortfall kink (autograd uses subgradient)
    shortfall = torch.clamp(required_capital - available_buffer, min=0.0)
    return terminal_capital - dcfg.lambda_shortfall * shortfall


class PolicyNet(nn.Module):
    """Outputs raw score, allocation decision is sigmoid(score)."""

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


def init_output_bias_for_alpha(model, alpha_pre_val):
    """Initialise output bias for model training stability.
    Only uses training-window oracle allocations to avoid data leakage."""
    a_mean = float(np.clip(np.mean(alpha_pre_val), 0.01, 0.99))
    logit_mean = float(np.log(a_mean / (1.0 - a_mean)))
    with torch.no_grad():
        if isinstance(model.net, nn.Linear):
            model.net.bias.fill_(logit_mean)
        else:
            # Final layer for MLP
            model.net[-1].bias.fill_(logit_mean)


def train_policy_model(x_train, c_next, k_next, u_oracle_next, alpha_oracle_next, model_type,
                        model_cfg: ModelConfig, dcfg: DecisionConfig, seed=None, return_history=False):
    """Train alpha_theta(x) by minimising mean realised regret over the window."""

    seed = model_cfg.seed if seed is None else int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = x_train.shape[0]
    val_size = max(1, int(np.floor(model_cfg.val_frac * n)))
    train_size = n - val_size
    if train_size < 8:
        raise ValueError("Training set too small after validation split.")

    def to_t(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float32)

    x_tr, x_val = x_train[:train_size], x_train[train_size:]
    c_tr, c_val = c_next[:train_size], c_next[train_size:]

    # Requires IRB capital req and utility for training
    k_tr, k_val = k_next[:train_size], k_next[train_size:]
    u_tr, u_val = u_oracle_next[:train_size], u_oracle_next[train_size:]

    train_ds = TensorDataset(to_t(x_tr), to_t(c_tr), to_t(k_tr), to_t(u_tr))
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=model_cfg.batch_size, shuffle=True, generator=loader_gen)

    # Initialise model, output bias and optimiser
    model = PolicyNet(input_dim=x_train.shape[1], model_type=model_type, hidden_dim=model_cfg.hidden_dim)
    init_output_bias_for_alpha(model, alpha_oracle_next[:train_size])
    optimiser = torch.optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)

    def regret_loss(score, cb, kb, ub):
        alpha = torch.sigmoid(score)
        # Return mean regret as loss function
        return torch.mean(ub - torch_allocation_utility(alpha, cb, kb, dcfg))

    x_val_t, c_val_t, k_val_t, u_val_t = to_t(x_val), to_t(c_val), to_t(k_val), to_t(u_val)
    best_state, best_val, stale_epochs, history = None, float("inf"), 0, []

    for epoch in range(model_cfg.epochs):
        model.train()
        batch_losses = []
        for xb, cb, kb, ub in train_loader:
            optimiser.zero_grad()
            loss = regret_loss(model(xb), cb, kb, ub)
            loss.backward() # Compute parameter gradients using autograd 
            optimiser.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            score_val = model(x_val_t)
            val_regret = regret_loss(score_val, c_val_t, k_val_t, u_val_t).item()
            # Convert raw scores to allocations
            alpha_val = torch.sigmoid(score_val).cpu().numpy()
        history.append({"epoch": epoch + 1, "model": f"blackbox_{model_type}", "seed": seed,
                        "train_loss": float(np.mean(batch_losses)), "val_loss": float(val_regret),
                        "val_mean_alpha": float(np.mean(alpha_val))})

        # Early stopping conditions
        if val_regret < best_val - 1e-12:
            best_val, stale_epochs = val_regret, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= model_cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, pd.DataFrame(history)) if return_history else model


def predict_alpha(model, x):
    model.eval()
    with torch.no_grad():
        alpha = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).cpu().numpy()
    return np.clip(np.atleast_1d(alpha), 0.0, 1.0)


def train_policy_ensemble(x_train, c_next, k_next, u_oracle_next, alpha_oracle_next, model_cfg: ModelConfig,
                           dcfg: DecisionConfig, seeds: Sequence[int], return_history=False):
    models, first_hist = [], None
    # Loop over ensemble seeds to train independent models for each 
    for i, s in enumerate(seeds):
        want_hist = return_history and i == 0
        out = train_policy_model(x_train, c_next, k_next, u_oracle_next, alpha_oracle_next, "mlp",
                                  model_cfg, dcfg, seed=s, return_history=want_hist)
        if want_hist:
            model, first_hist = out
        else:
            model = out
        models.append(model)
    return models, first_hist


def predict_alpha_ensemble(models, x):
    # Get each ensemble prediction and return mean
    preds = np.stack([predict_alpha(m, x) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def build_result_row_alpha(base_row, alpha_hat, model_name, split_name, dcfg: DecisionConfig,
                            alpha_hat_std=float("nan")):
    """Builds evaluation row for each black-box alpha forecast."""

    alpha_hat = float(alpha_hat)
    c_q_realised = float(base_row["c_q_realised"])
    k_true = float(base_row["k_irb_true_next"])
    alpha_oracle, utility_oracle = dec.optimise_alpha_oracle(c_q=c_q_realised, k=k_true, cfg=dcfg)
    realised = dec.decision_components(alpha_hat, c_q=c_q_realised, k=k_true, cfg=dcfg)
    regret = utility_oracle - realised["utility"]
    utility_bayes_realised = float(base_row["utility_bayes_realised_next"])

    return {"split": split_name, "model": model_name, "date": pd.to_datetime(base_row["target_date"]),
            "feature_date": pd.to_datetime(base_row[DATE_COL]),
            "c_q_realised": c_q_realised, "k_irb_true": k_true,
            "alpha_oracle": alpha_oracle, "alpha_bayes": float(base_row["alpha_bayes_next"]),
            "alpha_hat": alpha_hat, "alpha_hat_std": float(alpha_hat_std),
            "utility_oracle": utility_oracle, "utility_bayes_realised": utility_bayes_realised,
            "utility": realised["utility"],
            "regret": regret,
            "regret_vs_bayes": utility_bayes_realised - realised["utility"],
            "regret_irreducible": utility_oracle - utility_bayes_realised,
            "alpha_oracle_panel": float(base_row["alpha_oracle_next"]),
            "utility_oracle_panel": float(base_row["utility_oracle_next"]),
            "shortfall_realised_at_hat": realised["shortfall"]}


def run_expanding_window_blackbox(df: pd.DataFrame, feature_cols: List[str], split_name: str, split_cfg: dict,
                                   architecture: str, dcfg: DecisionConfig,
                                   model_cfg: ModelConfig = DEFAULT_MODEL_CONFIG, refit_every: int = 8,
                                   mlp_seeds: Sequence[int] = (42, 43, 44, 45, 46), return_history: bool = True,
                                   stability_reference_X: Optional[np.ndarray] = None,
                                   progress_every: int = 25):
    """Expanding-window train and evaluation for one black-box architecture 
    
    Reference outputs for black-box are only the model's alpha_hat vals (no pd_q_hat)."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown black-box architecture '{architecture}'. Expected one of {ARCHITECTURES}.")
    model_name = f"blackbox_{architecture}"

    first_test_idx = dec.resolve_first_test_idx(df, split_cfg)
    total = len(df) - first_test_idx
    n_refits = int(np.ceil(total / refit_every))

    # Progress updates
    print(f"[blackbox/{architecture}][{split_name}] first test target: "
          f"{pd.Timestamp(df['target_date'].iloc[first_test_idx]).date()}, {total} windows, "
          f"{n_refits} refits (every {refit_every}), {first_test_idx} initial training obs", flush=True)

    records, histories = [], []
    stability_linear_rows, stability_mlp_rows = [], []
    scaler, model_or_ensemble = None, None
    t0 = time.time()

    # Expanding window refits
    for step, test_idx in enumerate(range(first_test_idx, len(df))):
        test_row = df.iloc[test_idx]
        # Only re-estimate the model every refit steps
        if step % refit_every == 0:
            train_df = df.iloc[:test_idx]
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
            c_next_tr = train_df["c_q_realised"].to_numpy(dtype=float)
            k_next_tr = train_df["k_irb_true_next"].to_numpy(dtype=float)
            u_oracle_tr = train_df["utility_oracle_next"].to_numpy(dtype=float)
            alpha_oracle_tr = train_df["alpha_oracle_next"].to_numpy(dtype=float)

            if architecture == "linear":
                model_or_ensemble, hist = train_policy_model(
                    x_train, c_next_tr, k_next_tr, u_oracle_tr, alpha_oracle_tr, "linear",
                    model_cfg, dcfg, return_history=True)
                # Record predictions at each refit if stability ref points provided 
                if stability_reference_X is not None:
                    x_ref = scaler.transform(stability_reference_X)
                    ref_alpha = predict_alpha(model_or_ensemble, x_ref)
                    row = {"refit_step": step, "test_date": pd.to_datetime(test_row["target_date"]),
                           "train_end_date": pd.to_datetime(train_df[DATE_COL].iloc[-1]),
                           "n_train": len(train_df)}
                    for i, a in enumerate(ref_alpha):
                        row[f"ref{i}_alpha_hat_mean"] = float(a)
                    stability_linear_rows.append(row)
            else: # mlp model
                model_or_ensemble, hist = train_policy_ensemble(
                    x_train, c_next_tr, k_next_tr, u_oracle_tr, alpha_oracle_tr, model_cfg, dcfg,
                    seeds=mlp_seeds, return_history=True)
                if stability_reference_X is not None:
                    x_ref = scaler.transform(stability_reference_X)
                    member_alpha = np.stack([predict_alpha(m, x_ref)
                                             for m in model_or_ensemble], axis=0)
                    ref_mean = member_alpha.mean(axis=0)
                    ref_std = member_alpha.std(axis=0)
                    row = {"refit_step": step, "test_date": pd.to_datetime(test_row["target_date"]),
                           "train_end_date": pd.to_datetime(train_df[DATE_COL].iloc[-1]),
                           "n_train": len(train_df)}
                    for i, (m, s) in enumerate(zip(ref_mean, ref_std)):
                        row[f"ref{i}_alpha_hat_mean"] = float(m)
                        row[f"ref{i}_alpha_hat_std"] = float(s)
                        for seed, pred in zip(mlp_seeds, member_alpha[:, i]):
                            row[f"ref{i}_alpha_hat_seed{int(seed)}"] = float(pred)
                    stability_mlp_rows.append(row)

            if return_history and hist is not None:
                hist = hist.copy()
                hist["split"] = split_name
                hist["test_date"] = pd.to_datetime(test_row["target_date"])
                hist["train_end_date"] = pd.to_datetime(train_df[DATE_COL].iloc[-1])
                histories.append(hist)

        # Out-of-sample test for each window before refitting 
        x_test = scaler.transform(test_row[feature_cols].to_numpy(dtype=float).reshape(1, -1))
        if architecture == "linear":
            alpha_hat = float(predict_alpha(model_or_ensemble, x_test)[0])
            records.append(build_result_row_alpha(test_row, alpha_hat, model_name, split_name, dcfg))
        else:
            bb_mean, bb_std = predict_alpha_ensemble(model_or_ensemble, x_test)
            records.append(build_result_row_alpha(test_row, float(bb_mean[0]), model_name, split_name, dcfg,
                                                   alpha_hat_std=float(bb_std[0])))

        done = step + 1
        if done % progress_every == 0 or test_idx == len(df) - 1:
            print(f"[blackbox/{architecture}][{split_name}] completed {done}/{total} windows "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    pred_long = pd.DataFrame(records).sort_values(["date", "model"]).reset_index(drop=True)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    stability_linear = pd.DataFrame(stability_linear_rows)
    stability_mlp = pd.DataFrame(stability_mlp_rows)
    return pred_long, history, stability_linear, stability_mlp


def summarise_blackbox_run(pred_long: pd.DataFrame, model_name: str) -> dict:
    """Creates dictionary of panel-level metrics for a single architecture's 
    predictions. Appended to master metrics table."""
    g = pred_long[pred_long["model"] == model_name]
    return {
        "n": int(len(g)),
        "mean_regret": float(g["regret"].mean()),
        "median_regret": float(g["regret"].median()),
        "mean_regret_vs_bayes": float(g["regret_vs_bayes"].mean()),
        "mean_regret_irreducible": float(g["regret_irreducible"].mean()),
        "mean_utility": float(g["utility"].mean()),
        "shortfall_bind_share": float((g["shortfall_realised_at_hat"] > 1e-12).mean()),
    }
