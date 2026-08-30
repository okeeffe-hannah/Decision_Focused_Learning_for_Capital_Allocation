#!/usr/bin/env python3
"""Step 1 of the pipeline: generate and save every DGP panel once.

    python run_01_generate_panels.py --list                            # see the grid, generate nothing
    python run_01_generate_panels.py --seeds 42 --lambdas 10 --rho-scales 0.03 # specify values to run
    python run_01_generate_panels.py                                   # generates the grid in configs.py

Each (seed, lambda_shortfall, T, rho_scale) gets own panel CSV + metadata JSON under `study/data_panels/`. 
- seed controls the random shocks drawn for latent state/cylce/macro-covariates
- T is simulated sample length before dropping the first row for
`c_lag1`; the final panel/model dataset has T-1 rows. 
 - rho_scale is the asset-correlation scale used to draw the panel's
realised default/charge-off outcomes (and downstream for PTO/DFL) 
- lambda controls the shortfall penalty which is used to compute the alpha oracle

Existing panels are skipped unless --overwrite is used
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import configs
from common.dgp import build_simulated_panel, build_metadata
from common.io_utils import StudyPaths, ProgressTracker, print_grid_summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Override configs.SEEDS, e.g. --seeds 42")
    p.add_argument("--lambdas", type=float, nargs="+", default=None,
                    help="Override configs.LAMBDAS, e.g. --lambdas 10")
    p.add_argument("--Ts", "--sample-lengths", dest="Ts", type=int, nargs="+", default=None,
                    help="Override configs.SAMPLE_LENGTHS (raw sample length T, panel ends up "
                         "with T-1 rows), e.g. --Ts 1000 151")
    p.add_argument("--rho-scales", dest="rho_scales", type=float, nargs="+", default=None,
                    help="Override configs.RHO_SCALES (asset-correlation scale used for both "
                         "realised loss draws and downstream decision math), e.g. --rho-scales 0.05 0.1")
    p.add_argument("--study-root", type=str, default=str(Path(__file__).resolve().parent),
                    help="Where data_panels/ lives (defaults to this file's directory).")
    p.add_argument("--project-root", type=str, default=None,
                    help="Optional: path to the existing project containing "
                         "simulation/data/fred_loan_return_aligned.csv, used only for "
                         "informational calibration diagnostics in the saved metadata. "
                         "Panel columns are identical with or without this.")
    p.add_argument("--overwrite", action="store_true", help="Rebuild panels that already exist.")
    p.add_argument("--list", "--dry-run", dest="list_only", action="store_true",
                    help="Print the panel grid and exit without generating anything.")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds or configs.SEEDS
    lambdas = args.lambdas or configs.LAMBDAS
    Ts = args.Ts or configs.SAMPLE_LENGTHS
    rho_scales = args.rho_scales or configs.RHO_SCALES
    combos = [{"seed": s, "lambda_shortfall": lam, "T": T, "rho_scale": rho}
              for s in seeds for lam in lambdas for T in Ts for rho in rho_scales]

    print_grid_summary(combos, grid_name="Panel grid (step 1: generate DGP panels)")
    if args.list_only:
        return

    paths = StudyPaths(study_root=Path(args.study_root),
                        project_root=Path(args.project_root) if args.project_root else None)
    paths.ensure_dirs()

    tracker = ProgressTracker(total_units=len(combos), label="panel")
    for combo in combos:
        seed, lam, T, rho = combo["seed"], combo["lambda_shortfall"], combo["T"], combo["rho_scale"]
        desc = f"seed={seed} lambda={lam} T={T} rho_scale={rho}"
        csv_path = paths.panel_csv(seed, lam, T, rho)
        meta_path = paths.panel_metadata_json(seed, lam, T, rho)

        if csv_path.exists() and meta_path.exists() and not args.overwrite:
            tracker.start_unit(desc)
            print(f"    already exists -> {csv_path.name} (use --overwrite to rebuild)")
            tracker.finish_unit(desc, skipped=True)
            continue

        tracker.start_unit(desc)
        t0 = time.time()
        cfg = configs.BASE_DGP_CONFIG.variant(lambda_shortfall=lam, T=T, rho_scale=rho)
        panel = build_simulated_panel(seed, cfg)
        panel.to_csv(csv_path, index=False)

        metadata = build_metadata(seed, cfg, panel, csv_path.name)
       
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"    {len(panel)} rows -> {csv_path.name}  ({time.time() - t0:.1f}s)")
        tracker.finish_unit(desc)

    print(f"\nAll requested panels are in {paths.panels_dir}")
    print("Next: run_02_experiment.py")


if __name__ == "__main__":
    main()
