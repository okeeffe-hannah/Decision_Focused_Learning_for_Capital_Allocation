"""Experiment-grid configuration. Defines the baseline vals.

DGP_seeds: 42, 43, 44, 45, 46
Lambda: 10
T: 500
rho_scale: 0.03

These can be changed adding arguments to run_01 or run_02.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from common.dgp import DEFAULT_DGP_CONFIG, DGPConfig

# DGP seeds for generating panel
SEEDS: List[int] = [42, 43, 44, 45, 46]

LAMBDAS: List[float] = [10.0]

# Raw simulated length before dropping the first row for c_lag1.
SAMPLE_LENGTHS: List[int] = [500]

# Additional leading quarters simulated to burn in the persistent latent state;
# these are discarded before each saved panel is written. 
BURN_IN_PERIODS = 200

RHO_SCALES: List[float] = [0.03]

BASE_DGP_CONFIG: DGPConfig = DEFAULT_DGP_CONFIG.variant(burn_in=BURN_IN_PERIODS)

MODEL_ARCHITECTURES = {
    "pto": ["linear", "mlp", "persistence", "persistence_ma4"],
    "blackbox": ["linear", "mlp"],
    "dfl": ["linear", "mlp"],
}

SPLIT_NAME = "main"
EVAL_SPLIT_CFG = {"initial_train_frac": 0.7}

# Refit cadence for the expanding-window evaluation. 
# REFIT_EVERY=8 is baseline to balance computation time with parameter staleness 
# REFIT_EVERY=1 produces strict every-window protocol in empirical setting.
REFIT_EVERY = 8

# MLP seed ensembles 
MLP_ENSEMBLE_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)


# Whether to compute and save the extra auditability artifacts
SAVE_AUDIT_ARTIFACTS = False # Off by default can be overidden


def all_combos():
    """Every (seed, lambda, T, rho_scale, model_family, architecture) combo in the current grid."""
    combos = []
    for seed in SEEDS:
        for lam in LAMBDAS:
            for T in SAMPLE_LENGTHS:
                for rho_scale in RHO_SCALES:
                    for model_family, archs in MODEL_ARCHITECTURES.items():
                        for arch in archs:
                            combos.append({
                                "seed": seed, "lambda_shortfall": lam, "T": T, "rho_scale": rho_scale,
                                "model_family": model_family, "architecture": arch,
                            })
    return combos


def all_panels():
    """Every (seed, lambda, T, rho_scale) panel the grid needs."""
    return [{"seed": seed, "lambda_shortfall": lam, "T": T, "rho_scale": rho_scale}
            for seed in SEEDS for lam in LAMBDAS for T in SAMPLE_LENGTHS for rho_scale in RHO_SCALES]
