"""
blackbox_baselines.py

Basic black-box / decision-first baseline for the MSc capital allocation project.

What this script does
---------------------
1. Loads macro-financial covariates X_t and realised charge-off rates c_t from CSVs.
2. Aligns them by date and constructs a one-step-ahead target c_{t+h}.
3. Trains two direct allocation models:
   - Linear sigmoid model: x_t -> alpha_hat_{t+h}
   - Small MLP sigmoid model: x_t -> alpha_hat_{t+h}
4. Trains using the realised downstream decision objective, not an intermediate
   charge-off / PD / stress-probability forecast.
5. Evaluates ex-post regret against the oracle allocation.

This is the benchmark corresponding to the report's black-box setup:
    x_t -> alpha_hat^{BB}_{t+1}
with no auditable intermediate risk estimate.

Example
-------
python blackbox_baselines.py \
    --macro_csv data/macro_covariates.csv \
    --chargeoff_csv data/chargeoffs.csv \
    --date_col DATE \
    --target_col chargeoff_rate \
    --eval_mode expanding \
    --output_csv outputs/blackbox_predictions.csv

Notes
-----
- Charge-off rates should be decimals, e.g. 0.0061 for 0.61%. If your CSV stores
  percentages, e.g. 0.61 or 1.43, use --target_is_percent.
- The black-box model does not output p_hat, mu_hat, PD_hat or K_IRB_hat. It only
  outputs alpha_hat, so only decision metrics are available.
- For debugging, use --eval_mode holdout. For thesis-style evaluation, use
  --eval_mode expanding.
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

from capital_allocation_utils import (
    CapitalConfig,
    ModelConfig,
    annualised_pd_from_chargeoff,
    build_capital_config_from_args,
    chronological_split,
    irb_capital_requirement,
    load_and_align_data,
    optimise_oracle_alpha,
    realised_utility,
    regret_from_alpha,
)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class BlackBoxAllocator(nn.Module):
    """Direct decision model: x_t -> alpha_hat_{t+h}, where alpha_hat in [0, 1]."""

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


# -----------------------------------------------------------------------------
# Differentiable realised utility for black-box training
# -----------------------------------------------------------------------------


def realised_utility_torch(
    alpha: torch.Tensor,
    realised_c: torch.Tensor,
    k_irb: torch.Tensor,
    cfg: CapitalConfig,
) -> torch.Tensor:
    """
    Ex-post utility as a differentiable function of alpha.

    k_irb is precomputed from realised_c using the Basel formula and treated as a
    target-dependent constant. Gradients flow only through alpha and therefore
    through the black-box model parameters.
    """

    ell = cfg.leverage
    terminal_capital = 1.0 + alpha * ell * (cfg.y_bar - realised_c)
    available_buffer = alpha * ell * (cfg.y_bar - realised_c) + (1.0 - alpha)
    required_capital = k_irb * alpha * ell
    shortfall = torch.relu(required_capital - available_buffer)
    return terminal_capital - cfg.lambda_penalty * shortfall


def precompute_k_irb_from_chargeoff(y_chargeoff: np.ndarray, cfg: CapitalConfig) -> np.ndarray:
    pd_a = annualised_pd_from_chargeoff(y_chargeoff, cfg)
    return np.asarray(irb_capital_requirement(pd_a, cfg), dtype=float)


# -----------------------------------------------------------------------------
# Training / prediction
# -----------------------------------------------------------------------------


def train_blackbox_model(
    x_train: np.ndarray,
    y_train_chargeoff: np.ndarray,
    model_type: str,
    model_cfg: ModelConfig,
    cap_cfg: CapitalConfig,
    return_history: False,
    print_history: bool=False,
    print_every = 100,
) -> BlackBoxAllocator:
    """
    Train x -> alpha_hat directly by maximising realised utility.

    The loss is -U(alpha_hat; c). This is equivalent for optimisation to regret
    U(alpha_oracle; c) - U(alpha_hat; c), because the oracle term is constant with
    respect to the model parameters.
    """

    torch.manual_seed(model_cfg.seed)
    np.random.seed(model_cfg.seed)

    n = x_train.shape[0]
    val_size = max(1, int(np.floor(model_cfg.val_frac * n)))
    train_size = n - val_size
    if train_size < 5:
        raise ValueError("Training set too small after validation split.")

    x_tr, x_val = x_train[:train_size], x_train[train_size:]
    y_tr, y_val = y_train_chargeoff[:train_size], y_train_chargeoff[train_size:]
    k_tr = precompute_k_irb_from_chargeoff(y_tr, cap_cfg)
    k_val = precompute_k_irb_from_chargeoff(y_val, cap_cfg)

    train_ds = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(k_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=model_cfg.batch_size, shuffle=False)

    model = BlackBoxAllocator(input_dim=x_train.shape[1], model_type=model_type, hidden_dim=model_cfg.hidden_dim)
    optimiser = torch.optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)

    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    k_val_t = torch.tensor(k_val, dtype=torch.float32)



    alpha_oracle_val = np.array([
        optimise_oracle_alpha(float(c), cap_cfg) for c in y_val
    ])

    oracle_utility_val = np.array([
        realised_utility(float(a), float(c), cap_cfg)
        for a, c in zip(alpha_oracle_val, y_val)
])
    best_state = None
    best_val = float("inf")
    stale_epochs = 0

    history = []

    alpha_oracle_val = np.array([
        optimise_oracle_alpha(float(c), cap_cfg) for c in y_val
    ])

    oracle_utility_val = np.array([
        realised_utility(float(a), float(c), cap_cfg)
        for a, c in zip(alpha_oracle_val, y_val)
    ])

    for _epoch in range(model_cfg.epochs):
        model.train()
        for xb, yb, kb in train_loader:
            optimiser.zero_grad()
            alpha_hat = model(xb)
            utility = realised_utility_torch(alpha_hat, yb, kb, cap_cfg)
            loss = -utility.mean()
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            alpha_val = model(x_val_t)
            val_loss = -realised_utility_torch(alpha_val, y_val_t, k_val_t, cap_cfg).mean().item()
            alpha_val_np = alpha_val.detach().cpu().numpy()

        utility_val_np = np.array([
            realised_utility(float(a), float(c), cap_cfg)
            for a, c in zip(alpha_val_np, y_val)
        ])

        val_regret = oracle_utility_val - utility_val_np

        epoch_record = {
            "epoch": _epoch + 1,
            "model": model_type,
            "train_loss": float(loss.item()),
            "val_loss": float(val_loss),
            "val_mean_regret": float(np.mean(val_regret)),
            "val_median_regret": float(np.median(val_regret)),
            "val_max_regret": float(np.max(val_regret)),
            "val_mean_alpha": float(np.mean(alpha_val_np)),
            "val_mean_oracle_alpha": float(np.mean(alpha_oracle_val)),
        }

        history.append(epoch_record)

        if print_history and ((_epoch + 1) % print_every == 0 or _epoch == 0):
            print(
                f"    {model_type:6s} epoch {_epoch + 1:4d} | "
                f"val loss {val_loss:.6f} | "
                f"mean regret {np.mean(val_regret):.6f} | "
                f"median regret {np.median(val_regret):.6f} | "
                f"max regret {np.max(val_regret):.6f} | "
                f"mean alpha {np.mean(alpha_val_np):.4f}"
            )

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


   
    if return_history:
        return model, pd.DataFrame(history)

    return model

def predict_alpha(model: BlackBoxAllocator, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        alpha = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    return np.clip(alpha, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def evaluate_alpha_predictions(y_true: np.ndarray, alpha_hat: np.ndarray, cap_cfg: CapitalConfig) -> dict[str, float]:
    alpha_oracle = np.array([optimise_oracle_alpha(float(c), cap_cfg) for c in y_true])
    regrets = np.array([regret_from_alpha(float(a), float(c), cap_cfg) for a, c in zip(alpha_hat, y_true)])
    utility_model = np.array([realised_utility(float(a), float(c), cap_cfg) for a, c in zip(alpha_hat, y_true)])
    utility_oracle = np.array([realised_utility(float(a), float(c), cap_cfg) for a, c in zip(alpha_oracle, y_true)])

    return {
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "mean_alpha": float(np.mean(alpha_hat)),
        "mean_oracle_alpha": float(np.mean(alpha_oracle)),
        "mae_alpha_vs_oracle": float(np.mean(np.abs(alpha_hat - alpha_oracle))),
        "mean_realised_utility": float(np.mean(utility_model)),
        "mean_oracle_utility": float(np.mean(utility_oracle)),
    }


def build_blackbox_prediction_frame(
    feature_dates: pd.Series,
    target_dates: pd.Series,    
    y_true: np.ndarray,
    alpha_hat: np.ndarray,
    model_name: str,
    eval_mode: str,
    cap_cfg: CapitalConfig,
) -> pd.DataFrame:
    alpha_oracle = np.array([optimise_oracle_alpha(float(c), cap_cfg) for c in y_true])
    regrets = np.array([regret_from_alpha(float(a), float(c), cap_cfg) for a, c in zip(alpha_hat, y_true)])
    utility_model = np.array([realised_utility(float(a), float(c), cap_cfg) for a, c in zip(alpha_hat, y_true)])
    utility_oracle = np.array([realised_utility(float(a), float(c), cap_cfg) for a, c in zip(alpha_oracle, y_true)])

    return pd.DataFrame(
        {
            "feature_date": feature_dates.to_numpy(),
            "target_date": target_dates.to_numpy(),            
            "model": model_name,
            "eval_mode": eval_mode,
            "realised_chargeoff": y_true,
            "alpha_hat": alpha_hat,
            "alpha_oracle": alpha_oracle,
            "realised_utility": utility_model,
            "oracle_utility": utility_oracle,
            "regret": regrets,
        }
    )


# -----------------------------------------------------------------------------
# Evaluation modes
# -----------------------------------------------------------------------------


def run_holdout(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    train_frac: float,
    model_cfg: ModelConfig,
    cap_cfg: CapitalConfig,
) -> pd.DataFrame:
    """Train once on the first train_frac observations and test on the remainder."""

    train_df, test_df = chronological_split(df, train_frac)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=float))
    y_train = train_df["target_chargeoff_next"].to_numpy(dtype=float)
    y_test = test_df["target_chargeoff_next"].to_numpy(dtype=float)

    outputs = []
    print(f"Train observations: {len(train_df)} | Test observations: {len(test_df)}")

    for model_type in ["linear", "mlp"]:
        model = train_blackbox_model(x_train, y_train, model_type, model_cfg, cap_cfg)
        alpha_test = predict_alpha(model, x_test)
        metrics = evaluate_alpha_predictions(y_test, alpha_test, cap_cfg)

        print(f"\n{model_type.upper()} BLACK-BOX HOLDOUT")
        for key, val in metrics.items():
            print(f"  {key}: {val:.8f}")

        outputs.append(
            build_blackbox_prediction_frame(
                feature_dates=test_df[date_col],
                target_dates=test_df["target_date"],
                y_true=y_test,
                alpha_hat=alpha_test,
                model_name=model_type,
                eval_mode="holdout",
                cap_cfg=cap_cfg,
            )
        )

    return pd.concat(outputs, ignore_index=True)


def run_expanding_window(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    initial_train_frac: float,
    model_cfg: ModelConfig,
    cap_cfg: CapitalConfig,
    return_history: False,
) -> pd.DataFrame:
    """
    Expanding-window evaluation.

    For each t in the evaluation window:
      1. train on rows [0, ..., t-1]
      2. predict alpha for row t
      3. expand the training set by one row and repeat

    This is slower than holdout because both models are retrained at every step.
    """

    if not (0.0 < initial_train_frac < 1.0):
        raise ValueError("initial_train_frac must be in (0, 1).")

    start_idx = int(np.floor(len(df) * initial_train_frac))
    if start_idx < 15 or len(df) - start_idx < 5:
        raise ValueError("Not enough observations for expanding-window evaluation.")

    rows = []
    history_rows = []
    total_steps = len(df) - start_idx
    print(f"Initial train observations: {start_idx} | Expanding-window test observations: {total_steps}")

    for step, test_idx in enumerate(range(start_idx, len(df)), start=1):
        train_df = df.iloc[:test_idx].copy()
        test_df = df.iloc[[test_idx]].copy()

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float))
        x_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=float))
        y_train = train_df["target_chargeoff_next"].to_numpy(dtype=float)
        y_test = test_df["target_chargeoff_next"].to_numpy(dtype=float)

        if step == 1 or step == total_steps or step % 10 == 0:
            print(f"  expanding step {step}/{total_steps}: train through index {test_idx - 1}, test index {test_idx}")



        for model_type in ["linear", "mlp"]:
            # Change seed slightly by step so MLP reinitialisations are reproducible but not identical across windows.
            step_cfg = ModelConfig(
                hidden_dim=model_cfg.hidden_dim,
                lr=model_cfg.lr,
                weight_decay=model_cfg.weight_decay,
                batch_size=model_cfg.batch_size,
                epochs=model_cfg.epochs,
                patience=model_cfg.patience,
                val_frac=model_cfg.val_frac,
                seed=model_cfg.seed + step,
            )
            if return_history:
                model, history = train_blackbox_model(
                    x_train,
                    y_train,
                    model_type,
                    step_cfg,
                    cap_cfg,
                    return_history=True,
                    print_history=(step == 1 or step % 10 == 0 or step == total_steps),
                    print_every=100,
                )

                history = history.copy()
                history["expanding_step"] = step
                history["feature_date"] = test_df[date_col].iloc[0]
                history["target_date"] = test_df["target_date"].iloc[0]
                history_rows.append(history)

            else:
                model = train_blackbox_model(
                    x_train,
                    y_train,
                    model_type,
                    step_cfg,
                    cap_cfg,
                    return_history=False,
                    print_history=(step == 1 or step % 10 == 0 or step == total_steps),
                    print_every=100,
                )
            alpha_hat = predict_alpha(model, x_test)
            
            rows.append(
                build_blackbox_prediction_frame(
                    feature_dates=test_df[date_col],
                    target_dates=test_df["target_date"],
                    y_true=y_test,
                    alpha_hat=alpha_hat,
                    model_name=model_type,
                    eval_mode="expanding",
                    cap_cfg=cap_cfg,
                )
            )

    output = pd.concat(rows, ignore_index=True)
    
    for model_type in ["linear", "mlp"]:
        sub = output[output["model"] == model_type]
        metrics = evaluate_alpha_predictions(
            sub["realised_chargeoff"].to_numpy(dtype=float),
            sub["alpha_hat"].to_numpy(dtype=float),
            cap_cfg,
        )
        print(f"\n{model_type.upper()} BLACK-BOX EXPANDING")
        for key, val in metrics.items():
            print(f"  {key}: {val:.8f}")

    if return_history:
        history_output = pd.concat(history_rows, ignore_index=True)
        return output, history_output

    return output

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Black-box direct allocation baselines for capital allocation.")

    parser.add_argument("--macro_csv", required=True, help="Path to macro covariates CSV.")
    parser.add_argument("--chargeoff_csv", required=True, help="Path to charge-off CSV.")
    parser.add_argument("--date_col", default="DATE", help="Date column shared by both CSVs.")
    parser.add_argument("--target_col", required=True, help="Charge-off rate column in chargeoff_csv.")
    parser.add_argument("--feature_cols", nargs="*", default=None, help="Optional explicit feature columns from the merged data.")
    parser.add_argument("--target_is_percent", action="store_true", help="Use if target is stored as percent values, e.g. 0.61 for 0.61%.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in rows/quarters. Default 1 means x_t -> c_{t+1}.")
    parser.add_argument("--eval_mode", choices=["holdout", "expanding"], default="expanding")
    parser.add_argument("--train_frac", type=float, default=0.75, help="Train fraction for holdout mode.")
    parser.add_argument("--initial_train_frac", type=float, default=0.60, help="Initial training fraction for expanding mode.")
    parser.add_argument("--output_csv", default="blackbox_predictions.csv", help="Where to save predictions and decisions.")

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

    print(f"Using {len(feature_cols)} features: {feature_cols}")
    print(f"Total usable observations: {len(df)}")
    print(f"Evaluation mode: {args.eval_mode}")

    if args.eval_mode == "holdout":
        output = run_holdout(df, args.date_col, feature_cols, args.train_frac, model_cfg, cap_cfg)
    else:
        output = run_expanding_window(df, args.date_col, feature_cols, args.initial_train_frac, model_cfg, cap_cfg)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"\nSaved black-box predictions/allocations to: {output_path}")


if __name__ == "__main__":
    main()
