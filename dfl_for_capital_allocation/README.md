# Decision Focused Learning for Capital Allocation

Repository containing code for MSc Computing (AI & ML) Individual Project at Imperial College London.

## Project summary 
This project tests decision-focused learning as an approach that addresses the limitations of standard predict-then-optimise and black-box approaches. It uses a macro-financial setting with a Basel-style regulatory capital constraint to build an end-to-end model mapping data to an optimal capital allocation decision, in a way that is economically motivated and aware of regulatory constraints.

The approach is evaluated in both simulation and empirical settings, assessing performance using forecasting and decision-quality metrics, as well as an audit framework aligned with regulatory requirements. It aims to evaluate if aligning learning with downstream decisions improves decision outcomes relative to alternative approaches, and what this alignment changes in terms of model behaviour and auditability.

## Methods

The project compares three approaches:
- Predict-then-optimise (PTO)
- Black-box (BB)
- Decision-focused learning (DFL)

All three approaches use the same allocation objective and are evaluated using expanding-window experiments. The simulation experiments involve generating panels of data with varied DGP seeds, sample length, shortfall penalties and asset-correlation scale to run sensitivity analysis across the results. 

## Repository structure

- `decision_problem/` — motivation and construction of the allocation problem and supporting figures
- `empirical/` — historical-data experiments and empirical results
- `simulation/` — data-generating process, model implementations, simulations, evaluation, and audit experiments

The empirical and simulation subfolders contain READMEs with more detailed file structure details.

## Reproducibility

All experiments reuse the same seeds. The baseline training and ensemble seeds are (42, 43, 44, 45, 46), and are defined in `simulation/configs.py`.

## Results

The project finds that DFL can reduce costly decision errors while retaining an interpretable intermediate estimate, but the meaning of this estimate is changed and should be treated as such. Therefore, DFL is a promising approach in settings where both decision-quality and auditability are required and offers a bridge between the two alternative approaches. 