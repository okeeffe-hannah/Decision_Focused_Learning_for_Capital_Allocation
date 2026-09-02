"""Standardised paths, output schema, and progress printing for the simulation pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def lambda_tag(lambda_shortfall: float) -> str:
    """Filesystem-safe tag for a lambda value, e.g. 10.0 -> 'lam10', 7.5 -> 'lam7p5'."""
    s = f"{float(lambda_shortfall):g}".replace(".", "p").replace("-", "m")
    return f"lam{s}"


def seed_tag(seed: int) -> str:
    return f"seed{int(seed)}"


def t_tag(T: int) -> str:
    """Filesystem-safe tag for the sample length T, e.g. 1000 -> 'T1000'."""
    return f"T{int(T)}"


def rho_tag(rho_scale: float) -> str:
    """Filesystem-safe tag for RHO_SCALE, e.g. 0.05 -> 'rho0p05', 0.1 -> 'rho0p1'."""
    s = f"{float(rho_scale):g}".replace(".", "p").replace("-", "m")
    return f"rho{s}"


@dataclass(frozen=True)
class StudyPaths:
    """All directories the pipeline reads from / writes to."""

    study_root: Path
    project_root: Optional[Path] = None

    @property
    def panels_dir(self) -> Path:
        return self.study_root / "data_panels"

    @property
    def results_dir(self) -> Path:
        return self.study_root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.study_root / "figures"

    @property
    def master_metrics_csv(self) -> Path:
        return self.results_dir / "master_metrics.csv"

    def ensure_dirs(self):
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # DGP panel paths
    def panel_csv(self, seed: int, lambda_shortfall: float, T: int, rho_scale: float) -> Path:
        return self.panels_dir / (f"panel_{seed_tag(seed)}_{lambda_tag(lambda_shortfall)}_"
                                   f"{t_tag(T)}_{rho_tag(rho_scale)}.csv")

    def panel_metadata_json(self, seed: int, lambda_shortfall: float, T: int, rho_scale: float) -> Path:
        return self.panels_dir / (f"panel_{seed_tag(seed)}_{lambda_tag(lambda_shortfall)}_"
                                   f"{t_tag(T)}_{rho_tag(rho_scale)}_metadata.json")

    # per-run (model family x architecture x seed x lambda x T x rho_scale) paths --
    def run_dir(self, model_family: str, architecture: str, seed: int, lambda_shortfall: float, T: int,
                rho_scale: float, split_name: str = "main") -> Path:
        d = (self.results_dir / model_family / architecture /
             f"{seed_tag(seed)}_{lambda_tag(lambda_shortfall)}_{t_tag(T)}_{rho_tag(rho_scale)}_{split_name}")
        return d

    def timeseries_csv(self, model_family: str, architecture: str, seed: int, lambda_shortfall: float, T: int,
                        rho_scale: float, split_name: str = "main") -> Path:
        return self.run_dir(model_family, architecture, seed, lambda_shortfall, T, rho_scale, split_name) / "timeseries.csv"

    def metrics_row_json(self, model_family: str, architecture: str, seed: int, lambda_shortfall: float, T: int,
                          rho_scale: float, split_name: str = "main") -> Path:
        return self.run_dir(model_family, architecture, seed, lambda_shortfall, T, rho_scale, split_name) / "metrics_row.json"

    def training_history_csv(self, model_family: str, architecture: str, seed: int, lambda_shortfall: float,
                              T: int, rho_scale: float, split_name: str = "main") -> Path:
        return self.run_dir(model_family, architecture, seed, lambda_shortfall, T, rho_scale, split_name) / "training_history.csv"

    def stability_linear_csv(self, model_family: str, seed: int, lambda_shortfall: float, T: int,
                              rho_scale: float, split_name: str = "main") -> Path:
        return self.run_dir(model_family, "linear", seed, lambda_shortfall, T, rho_scale, split_name) / "stability_linear_reference_outputs.csv"

    def stability_mlp_csv(self, model_family: str, seed: int, lambda_shortfall: float, T: int,
                           rho_scale: float, split_name: str = "main") -> Path:
        return self.run_dir(model_family, "mlp", seed, lambda_shortfall, T, rho_scale, split_name) / "stability_mlp_reference_outputs.csv"

    # PTO forecast cache
    def pto_forecast_cache_dir(self, architecture: str, seed: int, T: int, rho_scale: float,
                                split_name: str = "main") -> Path:
        return (self.results_dir / "pto" / architecture /
                f"{seed_tag(seed)}_{t_tag(T)}_{rho_tag(rho_scale)}_{split_name}_forecast_cache")

    def pto_forecast_csv(self, architecture: str, seed: int, T: int, rho_scale: float,
                          split_name: str = "main") -> Path:
        return self.pto_forecast_cache_dir(architecture, seed, T, rho_scale, split_name) / "forecast.csv"

    def pto_forecast_meta_json(self, architecture: str, seed: int, T: int, rho_scale: float,
                                split_name: str = "main") -> Path:
        return self.pto_forecast_cache_dir(architecture, seed, T, rho_scale, split_name) / "forecast_meta.json"

    def pto_forecast_training_history_csv(self, architecture: str, seed: int, T: int, rho_scale: float,
                                           split_name: str = "main") -> Path:
        return self.pto_forecast_cache_dir(architecture, seed, T, rho_scale, split_name) / "training_history.csv"

    def pto_forecast_stability_linear_csv(self, seed: int, T: int, rho_scale: float, split_name: str = "main") -> Path:
        return self.pto_forecast_cache_dir("linear", seed, T, rho_scale, split_name) / "stability_linear_reference_outputs.csv"

    def pto_forecast_stability_mlp_csv(self, seed: int, T: int, rho_scale: float, split_name: str = "main") -> Path:
        return self.pto_forecast_cache_dir("mlp", seed, T, rho_scale, split_name) / "stability_mlp_reference_outputs.csv"

    # Audit stability outputs stored with the corresponding PTO run rather than in the reusable forecast cache.
    def pto_stability_linear_csv(self, seed: int, lambda_shortfall: float, T: int,
                                 rho_scale: float, split_name: str = "main") -> Path:
        return self.run_dir("pto", "linear", seed, lambda_shortfall, T, rho_scale, split_name) / "stability_linear_reference_outputs.csv"

    def pto_stability_mlp_csv(self, seed: int, lambda_shortfall: float, T: int,
                              rho_scale: float, split_name: str = "main") -> Path:
        return self.run_dir("pto", "mlp", seed, lambda_shortfall, T, rho_scale, split_name) / "stability_mlp_reference_outputs.csv"


# Master metrics table: one row per (model_family, architecture, seed, lambda, T, rho_scale, split)
METRICS_KEY_COLS = ["model_family", "architecture", "seed", "lambda_shortfall", "T", "rho_scale", "split"]


def upsert_metrics_row(master_csv_path: Path, row: dict):
    """Insert a metrics row into the master CSV. If a row with the same model 
    family, architecture, seed, shortfall penalty, sample length, correlation scale, 
    and split already exists, replace it."""
    master_csv_path = Path(master_csv_path)
    master_csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if master_csv_path.exists():
        existing = pd.read_csv(master_csv_path)
        key = tuple(row[c] for c in METRICS_KEY_COLS)
        if not existing.empty:
            existing_key = existing[METRICS_KEY_COLS].apply(tuple, axis=1)
            existing = existing[existing_key != key]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(master_csv_path, index=False)


def read_metrics_table(master_csv_path: Path) -> pd.DataFrame:
    master_csv_path = Path(master_csv_path)
    if not master_csv_path.exists():
        return pd.DataFrame(columns=METRICS_KEY_COLS)
    return pd.read_csv(master_csv_path)


def combo_already_done(master_csv_path: Path, model_family: str, architecture: str, seed: int,
                        lambda_shortfall: float, T: int, rho_scale: float, split_name: str = "main") -> bool:
    table = read_metrics_table(master_csv_path)  
    if table.empty:
        return False
    mask = (
        (table["model_family"] == model_family)
        & (table["architecture"] == architecture)
        & (table["seed"] == seed)
        & (table["lambda_shortfall"].astype(float) == float(lambda_shortfall))
        & (table["T"] == int(T))
        & (table["rho_scale"].astype(float) == float(rho_scale))
        & (table["split"] == split_name)
    )
    return bool(mask.any())


# Progress printing
class ProgressTracker:
    """Prints '[k/n] label ... (elapsed Xs, ETA Ys)' after each unit of work.
    """

    def __init__(self, total_units: int, label: str = "combo"):
        self.total_units = total_units
        self.label = label
        self.done_units = 0
        self.t0 = time.time()

    def start_unit(self, description: str):
        elapsed = time.time() - self.t0
        print(f"\n=== [{self.done_units + 1}/{self.total_units}] {description} "
              f"(elapsed {elapsed:.0f}s) ===", flush=True)

    def finish_unit(self, description: str, skipped: bool = False):
        self.done_units += 1
        elapsed = time.time() - self.t0
        avg = elapsed / max(1, self.done_units)
        remaining = max(0, self.total_units - self.done_units)
        eta = avg * remaining
        status = "skipped (already done)" if skipped else "done"
        print(f"--- [{self.done_units}/{self.total_units}] {description}: {status} "
              f"(elapsed {elapsed:.0f}s, avg {avg:.0f}s/{self.label}, ETA {eta:.0f}s "
              f"for remaining {remaining}) ---", flush=True)


def print_grid_summary(combos: Iterable[dict], grid_name: str = "experiment grid"):
    combos = list(combos)
    print(f"{grid_name}: {len(combos)} combo(s) queued.")
    for i, c in enumerate(combos, 1):
        print(f"  {i:>3}. {c}")
