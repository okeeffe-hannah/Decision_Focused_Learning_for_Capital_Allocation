# Empirical pipeline

This directory contains the empirical part of the study. It prepares a quarterly macro-financial panel from historical data, runs the PTO, black-box, and DFL models under the shared decision environment, and produces evaluation and audit results and figures.

## Workflow

1. `01_build_macro_panel.ipynb` — constructs and saves the processed quarterly macro-financial panel.
2. `02_pto_empirical.ipynb` — runs the predict-then-optimise baseline.
3. `03_black_box_empirical.ipynb` — runs the black-box decision model.
4. `04_dfl_empirical.ipynb` — runs the decision-focused learning model.
5. `05_empirical_thesis_figures.ipynb` — combines results and create thesis figures and tables.
6. `06_empirical_regulatory_audit.ipynb` — run sthe calibration, discrimination, and stability audit analyses.

The notebooks use the shared decision and model code in `../simulation/common/`. 

## Layout

- `common/` — empirical data loading and empirical model configuration
- `data/` — raw and processed empirical data and panel metadata
- `notebooks/` — numbered data-construction, modelling, evaluation, and audit workflows
- `results/` — model predictions and audit reference outputs
- `figures/` — generated plots, tables, and thesis figures

## Main outputs

The modelling notebooks save prediction and stability files under `results/`. The later notebooks read these files to produce comparative figures and regulatory-audit diagnostics.
