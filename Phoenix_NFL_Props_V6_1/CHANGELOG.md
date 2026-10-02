# Changelog

## 6.2.0

- Fixes a team-usage composition defect exposed by the quantitative diagnostics: independently modeled target/carry share centers could sum above 100% before simulation.
- Reconciles overfull player-share centers with Euclidean simplex projection before injury/role scenarios and logistic-normal noise.
- The projection is the minimum-change valid composition: larger modeled roles are preserved while mutually incompatible fringe shares absorb more of the correction instead of proportionally diluting every player.
- Adds raw/projected center sums and L1 reconciliation size to composition QA.
- Retains V6.1.2 Phoenix-only reliability tempering, QB/receiver accounting constraints, and sportsbook firewall.


## 6.1.2

- Enables symmetric Phoenix-only reliability tempering for extreme final prop probabilities.
- Tempering uses player history reliability and football-context uncertainty only; sportsbook probabilities remain excluded.
- Adds directional-concentration diagnostics by stat family without forcing over/under balance.
- Persists the concentration audit in every production artifact.
- Makes current-board publishing concurrency-safe.


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
