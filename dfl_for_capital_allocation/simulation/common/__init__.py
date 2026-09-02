"""Shared, callable core of the simulated-panel study. Some shared functionality also used in empirical study.

Modules:
    decision.py        - Basel/Vasicek/allocation math, DecisionConfig 
    dgp.py             - panel generator, DGPConfig 
    pto_model.py       - predict-then-optimise train and eval, per architecture
    blackbox_model.py  - end-to-end black-box policy train and eval, per architecture
    dfl_model.py       - decision-focused learning train and eval, per architecture
    io_utils.py        - standardised outputs, progress printing
    model_config.py    - shared neural-network training configuration
"""
