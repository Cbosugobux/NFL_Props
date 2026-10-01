# Changelog

## 6.1.1

- Enforces simulation-level receiving accounting against the selected quarterback.
- Listed-player receptions can never exceed simulated QB completions.
- Listed-player receiving yards can never exceed simulated QB passing yards.
- Uses capacity-preserving reconciliation rather than inventing catches or yardage.
- Adds hard coupling QA for completion and passing-yard overruns.
- Retains V6.1 QB-conditioned catch/YPR coupling, validation, hurdle architecture, calibration gate, and 80,000-simulation framework.


## 6.1.0

- Added QB-conditioned shared passing environment for receiving simulations.
- Selected QB completion-rate simulations partially shift receiver catch probability.
- Selected QB yards-per-completion simulations partially shift receiver YPR.
- Added bounded coupling ratios to prevent overreaction.
- Added `PHOENIX_NFL_QB_RECEIVER_COUPLING_AUDIT_V6_1_...csv`.
- Added coupling metadata to JSON output.
- Added coupling diagnostics to generative QA.
- Versioned output directory and weekly model-artifact directory for V6.1.
- Retained all V6 validation, hurdle calibration, residual, warehouse, and model-freeze behavior.

## 6.0.0

- Five whole-week walk-forward folds.
- Causal OOF ensemble weighting.
- Optimized non-negative RF/XGB/Ridge weights.
- Short-term role features and position-defense context.
- Validated hurdle calibration gate.
- Adaptive residual dispersion.
- Weekly frozen model artifacts.
- Probability double-tempering disabled by default.
