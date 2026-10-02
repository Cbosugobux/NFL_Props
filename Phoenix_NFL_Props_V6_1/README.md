# Phoenix NFL Generative Player Props V6.1.2

Precision-first, model-first NFL player-prop projection engine.

V6.1 preserves the V6 hurdle/compositional architecture and adds **QB-conditioned receiver coupling** so the QB selected in each Monte Carlo simulation influences the receiving environment in that same simulation.

## Core philosophy

Sportsbook prices are not predictive inputs. Phoenix builds the football distribution first:

```text
team environment
→ pass/rush volume
→ player availability
→ P(any opportunity)
→ conditional target/carry composition
→ QB starter scenario
→ QB-conditioned passing environment
→ catch/completion/efficiency
→ 80,000 Monte Carlo simulations
→ Phoenix fair probabilities and fair prices
```

Book prices may be compared only downstream after Phoenix has frozen its projection.

## V6.1 change

For every simulation, Phoenix selects a starting QB from the current starter-probability scenario. The selected QB's simulated completion rate and yards-per-completion create a bounded shared latent passing environment.

That environment partially adjusts each eligible RB/WR/TE's:

- catch probability
- yards per reception

Receiver-specific skill remains the primary driver. The QB effect is deliberately partial and bounded so an uncertain starter does not mechanically apply a crude flat penalty to every pass catcher.

Default coupling parameters:

```text
catch coupling strength = 0.60
yards/reception coupling strength = 0.50
QB environment ratio bounds = 0.82 to 1.18
```

## Accuracy safeguards retained from V6

- Five expanding whole-week walk-forward folds.
- Validation weeks never train on themselves.
- Causal OOF ensemble weighting.
- Non-negative RF/XGB/Ridge ensemble weights constrained to sum to one.
- Three-game short-term form plus eight-game form.
- Position-specific opponent-defense context.
- Historical opponent features shifted before the game to prevent leakage.
- Separate target/carry opportunity hurdles preserving zero-opportunity games.
- Hurdle calibration promoted only when it improves Brier score on a later chronological holdout.
- Adaptive residual dispersion from OOF errors.
- Symmetric reliability tempering of extreme probabilities using Phoenix-only history/context uncertainty; sportsbook probabilities are never inputs.
- Weekly fitted-model fingerprinting for reproducibility.
- Persistent Parquet historical warehouse with SQLite state/manifest.

## Requirements

Python 3.11+ recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

The checked-in build defaults to 2026 Week 5. Change `TARGET_SEASON` and `TARGET_WEEK` near the top of `nfl_props_v6_1.py` for a different slate.

```powershell
python nfl_props_v6_1.py
```

or:

```powershell
.\run_nfl_props.ps1
```

## Persistent warehouse

By default local data are stored under:

```text
./phoenix_data/NFL_Props_Warehouse/
    warehouse_state.sqlite3
    parquet/
    model_artifacts_v6_1/
```

Historical analytical tables live in Parquet. SQLite is the control plane for cache/state information rather than the large analytical store.

## Generated output directory

```text
./phoenix_generative_props_v6_1/
```

Important V6.1 outputs include:

```text
PHOENIX_NFL_GENERATIVE_PROPS_V6_1_<season>_W<week>.csv
PHOENIX_NFL_GENERATIVE_FAIR_LADDER_V6_1_<season>_W<week>.csv
PHOENIX_NFL_GENERATIVE_MODEL_METRICS_V6_1_<season>_W<week>.csv
PHOENIX_NFL_ROLE_MODEL_METRICS_V6_1_<season>_W<week>.csv
PHOENIX_NFL_ENSEMBLE_AUDIT_V6_1_<season>_W<week>.csv
PHOENIX_NFL_ROLE_CALIBRATION_AUDIT_V6_1_<season>_W<week>.csv
PHOENIX_NFL_QB_RECEIVER_COUPLING_AUDIT_V6_1_<season>_W<week>.csv
PHOENIX_NFL_DIRECTIONAL_CONCENTRATION_V6_1_<season>_W<week>.csv
PHOENIX_NFL_GENERATIVE_PROPS_V6_1_<season>_W<week>.json
PHOENIX_NFL_GENERATIVE_PROPS_V6_1_<season>_W<week>.html
```

### QB/receiver coupling audit

The coupling audit reports, by team:

- correlation between simulated QB passing yards and listed-receiver receiving yards
- listed receptions / QB completions diagnostic
- mean and standard deviation of catch multipliers
- mean and standard deviation of YPR multipliers

The correlation is diagnostic, not itself a selection rule. Fringe/zero-inflated receiving groups can naturally have weaker correlation.

## Injury / role overrides

The source contains operator override dictionaries near the configuration section:

```python
PLAYER_AVAILABILITY_OVERRIDES = {}
PLAYER_ROLE_OVERRIDES = {}
QB_STARTER_OVERRIDES = {}
QB_START_PROB_OVERRIDES = {}
PLAYER_PRACTICE_OVERRIDES = {}
```

Use these only for football information newer or more authoritative than the automated feed.

## Repository hygiene

Runtime warehouses, Parquet partitions, SQLite files, fitted `.joblib` models, generated cards, and `.env` are ignored by Git. Do not commit large local data or credentials.

## Validation before promotion

Before using a weekly card operationally, review:

1. model and role validation metrics;
2. ensemble weights;
3. accepted/rejected hurdle calibrators;
4. QB starter probabilities;
5. QB/receiver coupling audit;
6. distribution flags, especially `VERY_WIDE` and `HIGH_SKEW`;
7. personnel/injury context;
8. the fair ladder at the exact line being considered.

No model guarantees profitable betting results. V6.1.2 is designed to improve measurement, calibration, structural coherence, and auditability.
