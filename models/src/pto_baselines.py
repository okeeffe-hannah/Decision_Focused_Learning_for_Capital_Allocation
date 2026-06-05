"""
pto_baselines.py

Basic Predict-Then-Optimise (PTO) pipeline for the MSc capital allocation project.

What this script does
---------------------
1. Loads macro-financial covariates X_t and realised charge-off rates c_t from CSVs.
2. Aligns them by date and constructs a one-step-ahead target c_{t+h}.
3. Trains two simple differentiable forecasting models:
   - Linear sigmoid model: x_t -> p_hat_{t+h}
   - Small MLP sigmoid model: x_t -> p_hat_{t+h}
4. Converts p_hat to an expected charge-off rate:
      mu_c(p_hat) = p_hat * mu_stress + (1 - p_hat) * mu_normal
5. Feeds p_hat into the Basel-style capital allocation objective and solves for alpha*.
6. Evaluates forecast errors and decision regret against an oracle allocation.

Example
-------
python pto_baselines.py \
    --macro_csv data/macro_covariates.csv \
    --chargeoff_csv data/chargeoffs.csv \
    --date_col DATE \
    --target_col chargeoff_rate \
    --output_csv outputs/pto_predictions.csv

Notes
-----
- Charge-off rates should be decimals, e.g. 0.0061 for 0.61%. If your CSV stores
  percentages, e.g. 0.61 or 1.43, use --target_is_percent.
- Default mixture parameters are taken from the interim report table:
  normal mean 0.61%, normal sd 0.42%, stress mean 1.43%, stress sd 0.67%.
- This is intentionally a clean baseline script, not the final DFL layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from capital_allocation_utils import (
    CapitalConfig,
    ModelConfig,
    annualised_pd_from_chargeoff,
    build_capital_config_from_args,
    chronological_split,
    irb_capital_requirement,
    load_and_align_data,
    mixture_mean,
    optimise_alpha_from_p,
    optimise_oracle_alpha,
    regret_from_alpha,
)


# -----------------------------------------------------------------------------
# PTO forecasting models
# -----------------------------------------------------------------------------


class PTOForecaster(nn.Module):
    """x_t -> p_hat_{t+h}; p_hat in [0,1]."""

    def __init__(self, input_dim: int, model_type: str, hidden_dim: int = 16):
        super().__init__()
        if model_type == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif model_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError("model_type must be 'linear' or 'mlp'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def p_to_mu_torch(p_hat: torch.Tensor, cfg: CapitalConfig) -> torch.Tensor:
    return p_hat * cfg.mu_stress + (1.0 - p_hat) * cfg.mu_normal


def train_pto_model(
    x_train: np.ndarray,
    y_train_chargeoff: np.ndarray,
    model_type: str,
    model_cfg: ModelConfig,
    cap_cfg: CapitalConfig,
) -> PTOForecaster:
    """Train x -> p_hat by minimising MSE between mu_c(p_hat) and realised charge-off."""
    torch.manual_seed(model_cfg.seed)
    np.random.seed(model_cfg.seed)

    n = x_train.shape[0]
    val_size = max(1, int(np.floor(model_cfg.val_frac * n)))
    train_size = n - val_size
    if train_size < 5:
        raise ValueError("Training set too small after validation split.")

    x_tr, x_val = x_train[:train_size], x_train[train_size:]
    y_tr, y_val = y_train_chargeoff[:train_size], y_train_chargeoff[train_size:]

    train_ds = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=model_cfg.batch_size, shuffle=False)

    model = PTOForecaster(input_dim=x_train.shape[1], model_type=model_type, hidden_dim=model_cfg.hidden_dim)
    optimiser = torch.optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)
    loss_fn = nn.MSELoss()

    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    stale_epochs = 0

    for _epoch in range(model_cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad()
            p_hat = model(xb)
            mu_hat = p_to_mu_torch(p_hat, cap_cfg)
            loss = loss_fn(mu_hat, yb)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            p_val = model(x_val_t)
            val_loss = loss_fn(p_to_mu_torch(p_val, cap_cfg), y_val_t).item()

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= model_cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_p(model: PTOForecaster, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    return np.clip(p, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate_predictions(y_true: np.ndarray, p_hat: np.ndarray, cap_cfg: CapitalConfig) -> dict[str, float]:
    mu_hat = np.asarray(mixture_mean(p_hat, cap_cfg), dtype=float)
    alpha_hat = np.array([optimise_alpha_from_p(float(p), cap_cfg) for p in p_hat])
    alpha_oracle = np.array([optimise_oracle_alpha(float(c), cap_cfg) for c in y_true])
    regrets = np.array([regret_from_alpha(float(a), float(c), cap_cfg) for a, c in zip(alpha_hat, y_true)])

    return {
        "rmse_chargeoff": float(np.sqrt(mean_squared_error(y_true, mu_hat))),
        "mae_chargeoff": float(mean_absolute_error(y_true, mu_hat)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "mean_alpha": float(np.mean(alpha_hat)),
        "mean_oracle_alpha": float(np.mean(alpha_oracle)),
    }


def build_prediction_frame(
    dates: pd.Series,
    y_true: np.ndarray,
    p_hat: np.ndarray,
    model_name: str,
    cap_cfg: CapitalConfig,
) -> pd.DataFrame:
    mu_hat = np.asarray(mixture_mean(p_hat, cap_cfg), dtype=float)
    pd_hat = annualised_pd_from_chargeoff(mu_hat, cap_cfg)
    k_hat = irb_capital_requirement(pd_hat, cap_cfg)
    alpha_hat = np.array([optimise_alpha_from_p(float(p), cap_cfg) for p in p_hat])
    alpha_oracle = np.array([optimise_oracle_alpha(float(c), cap_cfg) for c in y_true])
    regrets = np.array([regret_from_alpha(float(a), float(c), cap_cfg) for a, c in zip(alpha_hat, y_true)])

    return pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "model": model_name,
            "realised_chargeoff": y_true,
            "p_hat_stress": p_hat,
            "mu_hat_chargeoff": mu_hat,
            "pd_hat_annualised": pd_hat,
            "k_irb_hat": k_hat,
            "alpha_hat": alpha_hat,
            "alpha_oracle": alpha_oracle,
            "regret": regrets,
        }
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic PTO linear and MLP baselines for capital allocation.")

    parser.add_argument("--macro_csv", required=True, help="Path to macro covariates CSV.")
    parser.add_argument("--chargeoff_csv", required=True, help="Path to charge-off CSV.")
    parser.add_argument("--date_col", default="DATE", help="Date column shared by both CSVs.")
    parser.add_argument("--target_col", required=True, help="Charge-off rate column in chargeoff_csv.")
    parser.add_argument("--feature_cols", nargs="*", default=None, help="Optional explicit feature columns from the merged data.")
    parser.add_argument("--target_is_percent", action="store_true", help="Use if target is stored as percent values, e.g. 0.61 for 0.61%.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in rows/quarters. Default 1 means x_t -> c_{t+1}.")
    parser.add_argument("--train_frac", type=float, default=0.75, help="Chronological train fraction.")
    parser.add_argument("--output_csv", default="pto_predictions.csv", help="Where to save predictions and decisions.")

    # Capital parameters
    parser.add_argument("--lgd", type=float, default=0.45)
    parser.add_argument("--y_bar", type=float, default=0.028)
    parser.add_argument("--lambda_penalty", type=float, default=10.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--mu_normal", type=float, default=0.0061)
    parser.add_argument("--sigma_normal", type=float, default=0.0042)
    parser.add_argument("--mu_stress", type=float, default=0.0143)
    parser.add_argument("--sigma_stress", type=float, default=0.0067)

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap_cfg = build_capital_config_from_args(args)
    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    df = load_and_align_data(
        macro_csv=args.macro_csv,
        chargeoff_csv=args.chargeoff_csv,
        date_col=args.date_col,
        target_col=args.target_col,
        feature_cols=args.feature_cols,
        horizon=args.horizon,
        target_is_percent=args.target_is_percent,
    )
    feature_cols = df.attrs["feature_cols"]
    train_df, test_df = chronological_split(df, args.train_frac)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=float))
    y_train = train_df["target_chargeoff_next"].to_numpy(dtype=float)
    y_test = test_df["target_chargeoff_next"].to_numpy(dtype=float)

    all_outputs = []
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    print(f"Train observations: {len(train_df)} | Test observations: {len(test_df)}")

    for model_type in ["linear", "mlp"]:
        model = train_pto_model(x_train, y_train, model_type, model_cfg, cap_cfg)
        p_test = predict_p(model, x_test)
        metrics = evaluate_predictions(y_test, p_test, cap_cfg)

        print(f"\n{model_type.upper()} PTO")
        for key, val in metrics.items():
            print(f"  {key}: {val:.8f}")

        pred_frame = build_prediction_frame(
            dates=test_df[args.date_col],
            y_true=y_test,
            p_hat=p_test,
            model_name=model_type,
            cap_cfg=cap_cfg,
        )
        all_outputs.append(pred_frame)

    output = pd.concat(all_outputs, ignore_index=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"\nSaved PTO predictions/allocations to: {output_path}")


if __name__ == "__main__":
    main()
