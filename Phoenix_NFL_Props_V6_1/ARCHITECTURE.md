# Architecture

## Prediction path

```text
NFL historical data
    ↓
Persistent Parquet warehouse
    ↓
Pregame / leakage-safe feature engineering
    ↓
RF + XGB + Ridge component models
    ↓
Walk-forward OOF validation and ensemble weights
    ↓
Team pass/rush volume
    ↓
Availability + target/carry hurdles
    ↓
Conditional compositional opportunity allocation
    ↓
QB starter draw
    ↓
QB completion-rate + passing-efficiency latent environment
    ↓
Receiver catch-rate + YPR coupling
    ↓
Monte Carlo stat distributions
    ↓
Phoenix probabilities / fair odds / ladders
```

## QB → receiver coupling

V6.1 does not apply a fixed backup-QB penalty. In each Monte Carlo draw:

1. A starting QB is selected from Phoenix's starter probability scenario.
2. That QB receives simulated completion-rate and yards-per-completion draws.
3. Those draws are normalized to the team's starter-weighted QB baseline.
4. Ratios are clipped to `[0.82, 1.18]`.
5. Receiver catch rates are shifted by `ratio^0.60`.
6. Receiver YPR is shifted by `ratio^0.50`.
7. Receiver-specific component uncertainty and micro noise remain in place.

This introduces shared passing-game covariance without allowing QB context to overwhelm receiver skill.

## Data persistence

- **Parquet:** large analytical historical tables.
- **SQLite:** manifest/state, timestamps, schema/cache control.
- **Joblib:** frozen weekly fitted model artifacts.

## Market firewall

Sportsbook lines, implied probabilities, consensus, and market movement do not enter model training or probability generation.
