# Simulation pipeline

This directory contains the simulation data-generating process, model
implementations, experiment runners, evaluation notebooks, and generated
outputs.

# Quickstart: simulation

From the `dfl_for_capital_allocation/simulation` directory:
```bash
python run_01_generate_panels.py --seeds 42 --lambdas 10 --Ts 500 --rho-scales 0.03
python run_02_experiment.py --seeds 42 --lambdas 10 --Ts 500 --rho-scales 0.03

```
## Layout

- `common/` — shared decision, DGP, model, and I/O code
- `configs.py` — simulation grid and experiment settings
- `run_01_generate_panels.py` — generate simulated panels
- `run_02_experiment.py` — run model experiments
- `notebooks/` — calibration, diagnostics, evaluation, and audit analyses
- `data_panels/` — generated panels and metadata
- `results/` — model outputs and metrics
- `figures/` — generated figures

