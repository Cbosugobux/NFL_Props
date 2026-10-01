# Operator Guide

## Weekly workflow

1. Set `TARGET_SEASON` and `TARGET_WEEK`.
2. Let the persistent warehouse update current NFL data.
3. Review automated personnel context.
4. Apply manual overrides only when official/public information is newer than the feed.
5. Run `python nfl_props_v6_1.py`.
6. Review validation and calibration audits before the fair card.
7. Review `PHOENIX_NFL_QB_RECEIVER_COUPLING_AUDIT_V6_1_...csv`, especially teams with uncertain QBs.
8. Use the fair ladder for the exact proposition threshold being evaluated.

## When to force retraining

Set:

```python
FORCE_RETRAIN_WEEKLY_MODEL = True
```

only when the underlying training data/schema intentionally changed and a new fitted weekly artifact is desired.

Personnel/injury changes do not by themselves require retraining the historical component models; they are applied in the simulation layer.

## Troubleshooting

### It looks hung during training

RF/XGB walk-forward fitting can be CPU-heavy. Check Task Manager. Active Python CPU usage generally means training is continuing.

### Historical data appears to download again

Check `phoenix_data/NFL_Props_Warehouse/warehouse_state.sqlite3` and the Parquet partitions. Historical partitions are intended to be immutable after successful creation unless refresh flags are enabled.

### Dashboard/report schema error

The script has an explicit schema guard before HTML generation. Treat a missing-column message as a code/schema mismatch rather than a modeling failure.
