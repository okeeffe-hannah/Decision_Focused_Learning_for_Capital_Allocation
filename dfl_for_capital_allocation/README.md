# Decision Focused Learning for Capital Allocation

Repository containing code for MSc Computing (AI & ML) Individual Project at Imperial College London.

## Description

This project tests decision-focused learning as an approach that addresses the limitations of standard predict-then-optimise and black-box decision-first pipelines. It uses a macro-financial setting with a Basel-style regulatory capital constraint to build an end-to-end model mapping data to an optimal capital allocation decision, in a way that is economically motivated and aware of regulatory constraints.

The approach is evaluated through in both simulation and empirical settings, assessing performance using forecasting and decision quality metrics, as well as an audit framework aligned with regulatory requirements. 

The results show that decision-focused learning offers a promising compromise between end-to-end optimisation, decision quality and supervisory auditability. 

## Setup

The project is split into three parts: decision_problem, empirical and simulation:
* decision_problem: uses realised historical charge-off data and the allocation objective to motivate the decision environment in the empirical setting.
* empirical: provides common data functions alongside a numbered set of notebooks that run the full modelling pipeline from constructing the macro panel, running each of the three models using expanding window evaluation under the same decision environment to produce thesis results and figures and running the audit framework. 
* simulation: contains majority of common code for the three models include basel requirements, allocation objective, IFT derivation used in both simulation and empirical setup, alongisde functions to produce the simulated panel. Contains two functions to first generate the synthetic panel `run_01_generate_panels.py` and then run the model experiments `run_02_experiment.py` supplying the rquired decision confugurations allowing sensitivty analysis to be checked across DGP seeds, lambdas, rhos and sample lengths. Separate folder contains notebooks performing calibration, gradient audit checks as well as pooling and evaluating results saved.

