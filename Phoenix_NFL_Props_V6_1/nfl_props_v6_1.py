"""Phoenix NFL Generative Player Props V6.1.1 — completion-conserving QB-coupled production build.

V6.1 extends the V6 hurdle/generative architecture with QB-conditioned passing-environment coupling,
while retaining time-safe ensemble learning, short-term role features, position-specific opponent context,
validated role-probability calibration, adaptive residual dispersion, and reproducible weekly model artifacts.
Sportsbook lines remain excluded from prediction generation.
"""

# Local VS Code runtime helpers
from pathlib import Path as _PhoenixPath
try:
    from dotenv import load_dotenv as _phoenix_load_dotenv
    _phoenix_load_dotenv()
except Exception:
    pass

try:
    from IPython.display import display  # nicer output when run in VS Code Interactive
except Exception:
    def display(obj):
        try:
            print(obj.to_string())
        except Exception:
            print(obj)

SCRIPT_DIR = _PhoenixPath(__file__).resolve().parent if "__file__" in globals() else _PhoenixPath.cwd()

# %% [markdown]
# # PHOENIX NFL GENERATIVE PLAYER PROPS — V6.1
# ## Hurdle usage + calibrated tails + persistent warehouse
#
# V6 separates **whether a player gets an opportunity** from **how much share he gets if used**.
#
# **Availability → opportunity hurdle → compositional share → efficiency → Monte Carlo → fair distribution**
#
# Fixes:
# - RF + XGBoost + logistic hurdle models for `P(any target)` and `P(any carry)`.
# - Conditional target/carry share is applied only after the hurdle clears.
# - Zero outcomes can never be clipped upward by tail guards.
# - Backup-QB rushing is conditioned on that QB actually starting.
# - Half-point base fair lines avoid hidden push mass.
# - Raw/tempered/push probabilities remain auditable.
# - Persistent Parquet + SQLite warehouse is retained.

# %% [markdown]
# ## 0. Colab dependency guard
# Only install `nflreadpy` if Colab does not already have it. Do **not** reinstall NumPy/SciPy/scikit-learn/XGBoost.

# %%
import importlib.util, subprocess, sys

def ensure_package(import_name, pip_name, extra_args=None):
    if importlib.util.find_spec(import_name) is not None:
        return
    cmd=[sys.executable, "-m", "pip", "install", "-q"]
    if extra_args:
        cmd += list(extra_args)
    cmd.append(pip_name)
    subprocess.check_call(cmd)

# nflreadpy should not force replacement of Colab's native numerical stack.
ensure_package("nflreadpy", "nflreadpy", ["--no-deps"])

# Repeated Phoenix/Colab dependency failure: install explicitly when absent.
ensure_package("pydantic_settings", "pydantic-settings", ["--upgrade-strategy", "only-if-needed"])
ensure_package("pyarrow", "pyarrow", ["--upgrade-strategy", "only-if-needed"])

print("Dependency guard complete: nflreadpy + pydantic-settings + pyarrow available.")

# %% [markdown]
# ## 1. Imports

# %%
import os, sys, math, json, warnings, shutil, re, sqlite3, hashlib, time, joblib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import nflreadpy as nfl

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss
from scipy.optimize import minimize
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 260)

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)

# %% [markdown]
# ## 2. Configuration

# %%
MODEL_VERSION = "6.1.1"
TARGET_SEASON = 2026
TARGET_WEEK = 5

LOOKBACK_SEASONS = 4
RECENT_GAMES = 8
SHORT_RECENT_GAMES = 3
MIN_PLAYER_GAMES = 3
MIN_TEAM_GAMES = 4
N_WALK_FOLDS = 5
N_SIMS = 80_000
RANDOM_SEED = 42
RECENCY_HALF_LIFE_WEEKS = 18.0

# Optional fast game filter.
# Example for one game: TARGET_GAME_TEAMS = {"PIT","CLE"}
TARGET_GAME_TEAMS = None

# Runtime portability: Colab or VS Code/local Jupyter.
try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False
BASE_DIR = SCRIPT_DIR
CACHE_DIR = BASE_DIR / "phoenix_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Persistent warehouse ----------
USE_PERSISTENT_WAREHOUSE = True

# Colab: persist between runtime disconnects in Google Drive.
# Local/VS Code: store under ./phoenix_data by default.
USE_GOOGLE_DRIVE_WAREHOUSE = False
GOOGLE_DRIVE_WAREHOUSE_DIR = "/content/drive/MyDrive/PhoenixQuantData/NFL_Props_Warehouse"
LOCAL_WAREHOUSE_DIR = str(BASE_DIR / "phoenix_data" / "NFL_Props_Warehouse")

# Refresh controls
FORCE_WAREHOUSE_REFRESH = False
CURRENT_SEASON_REFRESH_HOURS = 6.0
CONTEXT_REFRESH_HOURS = 1.5
ROSTER_REFRESH_HOURS = 6.0
SCHEDULE_REFRESH_HOURS = 6.0

# Historical partitions are immutable by default after first successful save.
REFRESH_HISTORICAL_SEASONS = False


# Personnel / injury scenario settings
REFRESH_SLEEPER = True
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_CACHE_FILE = CACHE_DIR / "sleeper_nfl_players.json"  # repointed to persistent warehouse after storage init
USE_INJURY_SCENARIOS = True
REQUIRE_CURRENT_SEASON_OFFENSIVE_SNAPS = True
ALLOW_DEPTH_CHART_REPLACEMENT_EXCEPTION = True
MAX_CONTEXT_UNCERTAINTY_MULT = 1.85
MIN_OFFER_CONDITION_SIMS = 1500

# Football-context overrides only.
PLAYER_AVAILABILITY_OVERRIDES = {}
PLAYER_ROLE_OVERRIDES = {}
QB_STARTER_OVERRIDES = {}
QB_START_PROB_OVERRIDES = {}
PRIMARY_QB_OUTPUT_ONLY = False

# ---------- V6 opportunity hurdle ----------
HURDLE_MIN_PROB = 0.002
HURDLE_MAX_PROB = 0.998
HURDLE_ISOTONIC_MIN_OOF = 250

# Optional late official-report overrides.
# Example: PLAYER_PRACTICE_OVERRIDES = {"Rico Dowdle":"DNP"}
PLAYER_PRACTICE_OVERRIDES = {}

# ---------- V6 residual / distribution calibration ----------
ROBUST_RESIDUAL_LO_Q = 0.02
ROBUST_RESIDUAL_HI_Q = 0.98
ROBUST_RESIDUAL_MAX_SIGMA = 3.0
COMPONENT_RESIDUAL_SCALE = 1.00

# Small possession-level noise only. V5.2 double-counted efficiency variance by
# drawing game-level efficiency and then adding another full event distribution.
PASS_MICRO_SD_PER_COMPLETION = 1.25
REC_MICRO_SD_PER_RECEPTION = 1.50
RUSH_MICRO_SD_PER_CARRY = 1.25

# ---------- V6.1 QB -> receiver shared passing environment ----------
# The selected starter's simulated completion efficiency and yards-per-completion
# now create a shared latent passing environment for RB/WR/TE receiving outcomes.
# Coupling is intentionally partial: receiver-specific skill remains the dominant
# driver while QB quality shifts catch and YPR distributions coherently.
USE_QB_RECEIVER_COUPLING = True
QB_RECEIVER_CATCH_COUPLING = 0.60
QB_RECEIVER_YPR_COUPLING = 0.50
QB_ENV_RATIO_MIN = 0.82
QB_ENV_RATIO_MAX = 1.18

# Empirical final-stat tail guard.
SOFT_TAIL_LO_Q = 0.005
SOFT_TAIL_HI_Q = 0.995
HARD_TAIL_LO_Q = 0.0005
HARD_TAIL_HI_Q = 0.9995
TAIL_SHRINK_SLOPE = 0.20

# Probability guard: do not pretend 40k Monte Carlo draws are 40k independent
# observations. Temper extreme probabilities when player history/context is weak.
USE_UNCERTAINTY_TEMPERING = False
MAX_PROBABILITY_TEMPERATURE = 1.40

# ---------- V6.1 precision / reproducibility controls ----------
# Weekly football models are frozen by a fingerprint of the historical training data.
USE_WEEKLY_MODEL_ARTIFACTS = True
FORCE_RETRAIN_WEEKLY_MODEL = False
V61_MODEL_SCHEMA_VERSION = "NFL_PROPS_V6_1_2026_10_01"

# Hurdle calibration is accepted only if it improves a chronologically later holdout.
ROLE_CALIBRATION_HOLDOUT_FRAC = 0.30
ROLE_CALIBRATION_MIN_BRIER_GAIN = 0.0005

# Ensemble weights are learned from genuinely out-of-fold base predictions with
# non-negative weights summing to one.
ENSEMBLE_RMSE_PENALTY = 0.10

# Adaptive residual scale is learned from recent vs earlier OOF errors.
MIN_RESIDUAL_SCALE = 0.80
MAX_RESIDUAL_SCALE = 1.30


LADDER_STEPS_EACH_SIDE = 6
OUTPUT_DIR = BASE_DIR / "phoenix_generative_props_v6_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Runtime: {'COLAB' if IN_COLAB else 'LOCAL/JUPYTER'}")
print(f"Target: {TARGET_SEASON} Week {TARGET_WEEK}")
print(f"Simulations: {N_SIMS:,}")
print("Game filter:", TARGET_GAME_TEAMS if TARGET_GAME_TEAMS else "FULL WEEK")

# %% [markdown]
# ## 3. Helpers and transforms

# %%
def first_existing(df, names, required=False):
    for c in names:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns found: {names}")
    return None

def clean_name(s):
    return (
        s.astype(str).str.lower().str.strip()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

def normalize_position(p):
    if pd.isna(p): return "UNK"
    p = str(p).upper().strip()
    return "RB" if p in {"HB","FB"} else p

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    out = np.zeros(len(a), dtype=float) if hasattr(a, "__len__") else 0.0
    return np.divide(a, b, out=out, where=np.abs(b) > 1e-9)

def logit(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-4, 1-1e-4)
    return np.log(x/(1-x))

def inv_logit(z):
    z = np.clip(np.asarray(z, dtype=float), -12, 12)
    return 1/(1+np.exp(-z))

def transform_target(x, kind):
    x = np.asarray(x, dtype=float)
    if kind == "logit": return logit(x)
    if kind == "log1p": return np.log1p(np.maximum(x, 0.0))
    return x

def inverse_target(z, kind):
    z = np.asarray(z, dtype=float)
    if kind == "logit": return inv_logit(z)
    if kind == "log1p": return np.maximum(np.expm1(np.clip(z, -10, 10)), 0.0)
    return z

def recency_weight(weeks_ago, half_life=RECENCY_HALF_LIFE_WEEKS):
    return np.exp(-np.log(2)*np.maximum(np.asarray(weeks_ago, dtype=float), 0)/half_life)

def fair_american(p):
    p = float(p)
    if not np.isfinite(p) or p <= 0 or p >= 1: return np.nan
    if p >= 0.5: return -100.0*p/(1.0-p)
    return 100.0*(1.0-p)/p

def rmse(y,p):
    return float(np.sqrt(mean_squared_error(y,p)))

def to_py(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    try:
        if pd.isna(v): return None
    except Exception:
        pass
    return v

def finite_fill(df, cols, fill=0.0):
    for c in cols:
        if c not in df.columns: df[c] = fill
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(fill)
    return df

def market_style_line(x):
    # Standard prop threshold nearest the median, on a half point to avoid pushes.
    return math.floor(float(x)) + 0.5

def weighted_mean_predictions(preds, weights):
    keys = [k for k in preds if k in weights]
    return sum(weights[k]*np.asarray(preds[k], dtype=float) for k in keys)

rng = np.random.default_rng(RANDOM_SEED)

# %% [markdown]
# ## 4. Persistent nflverse warehouse
#
# Large analytical tables are partitioned by season in Parquet. A small SQLite manifest tracks what is cached and when it was last refreshed.
#
# **Behavior**
# - Old seasons: fetch once, then local forever unless `REFRESH_HISTORICAL_SEASONS=True`.
# - Current season stats/schedule/snaps: refresh on TTL or when the target week advances.
# - Injuries/depth charts: shorter TTL.
# - `FORCE_WAREHOUSE_REFRESH=True`: force all current-season/current-context sources to refresh.

# %%
# ---------- Resolve persistent storage ----------
if USE_PERSISTENT_WAREHOUSE:
    if IN_COLAB and USE_GOOGLE_DRIVE_WAREHOUSE:
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            WAREHOUSE_DIR = Path(GOOGLE_DRIVE_WAREHOUSE_DIR)
        except Exception as e:
            print("Google Drive mount failed; falling back to /content cache:", e)
            WAREHOUSE_DIR = BASE_DIR / "phoenix_data" / "NFL_Props_Warehouse"
    else:
        WAREHOUSE_DIR = Path(LOCAL_WAREHOUSE_DIR)
else:
    WAREHOUSE_DIR = CACHE_DIR / "NFL_Props_Warehouse"

PARQUET_DIR = WAREHOUSE_DIR / "parquet"
STATE_DB = WAREHOUSE_DIR / "warehouse_state.sqlite3"
SLEEPER_CACHE_FILE = WAREHOUSE_DIR / "sleeper_nfl_players.json"

PARQUET_DIR.mkdir(parents=True, exist_ok=True)
WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

print("Warehouse:", WAREHOUSE_DIR)
print("Manifest:", STATE_DB)

# ---------- SQLite manifest ----------
def _db():
    con = sqlite3.connect(STATE_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS dataset_state (
            dataset TEXT NOT NULL,
            season INTEGER NOT NULL,
            refreshed_at_utc TEXT NOT NULL,
            row_count INTEGER,
            max_week REAL,
            schema_hash TEXT,
            source_status TEXT,
            parquet_path TEXT,
            PRIMARY KEY (dataset, season)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS run_log (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at_utc TEXT NOT NULL,
            target_season INTEGER,
            target_week INTEGER,
            dataset TEXT,
            season INTEGER,
            action TEXT,
            rows INTEGER,
            note TEXT
        )
    """)
    con.commit()
    return con

def _state(dataset, season):
    with _db() as con:
        row = con.execute(
            "SELECT refreshed_at_utc,row_count,max_week,schema_hash,source_status,parquet_path "
            "FROM dataset_state WHERE dataset=? AND season=?",
            (dataset, int(season))
        ).fetchone()
    if row is None:
        return None
    return {
        "refreshed_at_utc": row[0],
        "row_count": row[1],
        "max_week": row[2],
        "schema_hash": row[3],
        "source_status": row[4],
        "parquet_path": row[5],
    }

def _log(dataset, season, action, rows=None, note=""):
    with _db() as con:
        con.execute(
            "INSERT INTO run_log(run_at_utc,target_season,target_week,dataset,season,action,rows,note) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                int(TARGET_SEASON), int(TARGET_WEEK),
                dataset, int(season), action,
                None if rows is None else int(rows), str(note)[:500]
            )
        )
        con.commit()

def _schema_hash(df):
    payload = "|".join(f"{c}:{df[c].dtype}" for c in df.columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _max_week(df):
    for c in ["week", "game_week"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().any():
                return float(x.max())
    return None

def _partition_path(dataset, season):
    d = PARQUET_DIR / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"season={int(season)}.parquet"

def _age_hours(iso_ts):
    if not iso_ts:
        return float("inf")
    try:
        dt = datetime.fromisoformat(str(iso_ts).replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc)-dt).total_seconds()/3600.0
    except Exception:
        return float("inf")

def _save_partition(dataset, season, df, source_status="OK"):
    path = _partition_path(dataset, season)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    refreshed = datetime.now(timezone.utc).isoformat()
    with _db() as con:
        con.execute("""
            INSERT INTO dataset_state
            (dataset,season,refreshed_at_utc,row_count,max_week,schema_hash,source_status,parquet_path)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(dataset,season) DO UPDATE SET
                refreshed_at_utc=excluded.refreshed_at_utc,
                row_count=excluded.row_count,
                max_week=excluded.max_week,
                schema_hash=excluded.schema_hash,
                source_status=excluded.source_status,
                parquet_path=excluded.parquet_path
        """, (
            dataset, int(season), refreshed, int(len(df)), _max_week(df),
            _schema_hash(df), source_status, str(path)
        ))
        con.commit()
    _log(dataset, season, "REFRESHED", len(df), str(path))
    return df

def _read_partition(dataset, season):
    path = _partition_path(dataset, season)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        _log(dataset, season, "CACHE_HIT", len(df), str(path))
        return df
    except Exception as e:
        _log(dataset, season, "CACHE_READ_FAILED", None, repr(e))
        print(f"{dataset} {season}: cached parquet unreadable, refreshing:", e)
        return None

def _should_refresh(dataset, season, ttl_hours, require_target_week=False, historical=False):
    path = _partition_path(dataset, season)
    st = _state(dataset, season)
    if not path.exists() or st is None:
        return True, "MISSING"
    if FORCE_WAREHOUSE_REFRESH and not historical:
        return True, "FORCED"
    if historical:
        return bool(REFRESH_HISTORICAL_SEASONS), "HISTORICAL_POLICY"
    if require_target_week and st.get("max_week") is not None and float(st["max_week"]) < float(TARGET_WEEK):
        return True, f"TARGET_WEEK_ADVANCED_{st['max_week']}_TO_{TARGET_WEEK}"
    if _age_hours(st.get("refreshed_at_utc")) >= float(ttl_hours):
        return True, "TTL_EXPIRED"
    return False, "FRESH_CACHE"

def cached_dataset(dataset, season, loader, ttl_hours, require_target_week=False, historical=False, allow_empty=False):
    refresh, reason = _should_refresh(
        dataset, season, ttl_hours,
        require_target_week=require_target_week,
        historical=historical
    )
    if not refresh:
        df = _read_partition(dataset, season)
        if df is not None:
            print(f"[CACHE] {dataset} {season}: {len(df):,} rows ({reason})")
            return df

    print(f"[FETCH] {dataset} {season}: {reason}")
    try:
        obj = loader()
        df = obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)
        if len(df)==0 and not allow_empty:
            cached = _read_partition(dataset, season)
            if cached is not None:
                print(f"[STALE FALLBACK] {dataset} {season}: source returned empty; using cache")
                _log(dataset, season, "STALE_FALLBACK_EMPTY", len(cached))
                return cached
        _save_partition(dataset, season, df, "OK")
        return df
    except Exception as e:
        cached = _read_partition(dataset, season)
        if cached is not None:
            print(f"[STALE FALLBACK] {dataset} {season}: {e}")
            _log(dataset, season, "STALE_FALLBACK_ERROR", len(cached), repr(e))
            return cached
        _log(dataset, season, "FETCH_FAILED_NO_CACHE", None, repr(e))
        if allow_empty:
            print(f"[EMPTY FALLBACK] {dataset} {season}: {e}")
            return pd.DataFrame()
        raise

# ---------- Load analytical history one season at a time ----------
seasons = list(range(TARGET_SEASON-LOOKBACK_SEASONS+1, TARGET_SEASON+1))

player_parts=[]
team_parts=[]
schedule_parts=[]
snap_parts=[]

for season in seasons:
    hist = int(season) < int(TARGET_SEASON)

    player_parts.append(cached_dataset(
        "player_stats_weekly", season,
        lambda s=season: nfl.load_player_stats([s], summary_level="week"),
        CURRENT_SEASON_REFRESH_HOURS,
        require_target_week=not hist,
        historical=hist
    ))

    team_parts.append(cached_dataset(
        "team_stats_weekly", season,
        lambda s=season: nfl.load_team_stats([s], summary_level="week"),
        CURRENT_SEASON_REFRESH_HOURS,
        require_target_week=not hist,
        historical=hist
    ))

    schedule_parts.append(cached_dataset(
        "schedules", season,
        lambda s=season: nfl.load_schedules([s]),
        SCHEDULE_REFRESH_HOURS,
        require_target_week=not hist,
        historical=hist
    ))

    snap_parts.append(cached_dataset(
        "snap_counts", season,
        lambda s=season: nfl.load_snap_counts([s]),
        CURRENT_SEASON_REFRESH_HOURS,
        require_target_week=not hist,
        historical=hist,
        allow_empty=True
    ))

player_stats = pd.concat(player_parts, ignore_index=True, sort=False)
team_stats = pd.concat(team_parts, ignore_index=True, sort=False)
schedule = pd.concat(schedule_parts, ignore_index=True, sort=False)
snap_counts = pd.concat(snap_parts, ignore_index=True, sort=False) if any(len(x) for x in snap_parts) else pd.DataFrame()

# ---------- Current-season context ----------
rosters_weekly = cached_dataset(
    "rosters_weekly", TARGET_SEASON,
    lambda: nfl.load_rosters_weekly(TARGET_SEASON),
    ROSTER_REFRESH_HOURS,
    require_target_week=True,
    historical=False,
    allow_empty=True
)

rosters = cached_dataset(
    "rosters", TARGET_SEASON,
    lambda: nfl.load_rosters(TARGET_SEASON),
    ROSTER_REFRESH_HOURS,
    historical=False,
    allow_empty=True
)

depth_charts = cached_dataset(
    "depth_charts", TARGET_SEASON,
    lambda: nfl.load_depth_charts(TARGET_SEASON),
    CONTEXT_REFRESH_HOURS,
    require_target_week=True,
    historical=False,
    allow_empty=True
)

injuries = cached_dataset(
    "injuries", TARGET_SEASON,
    lambda: nfl.load_injuries(TARGET_SEASON),
    CONTEXT_REFRESH_HOURS,
    require_target_week=True,
    historical=False,
    allow_empty=True
)

print("\nLoaded frames:")
print("player_stats", player_stats.shape)
print("team_stats", team_stats.shape)
print("schedule", schedule.shape)
print("snap_counts", snap_counts.shape)
print("rosters_weekly", rosters_weekly.shape)
print("rosters", rosters.shape)
print("depth_charts", depth_charts.shape)
print("injuries", injuries.shape)

# Warehouse status dashboard
with _db() as con:
    warehouse_status = pd.read_sql_query(
        "SELECT dataset,season,refreshed_at_utc,row_count,max_week,source_status,parquet_path "
        "FROM dataset_state ORDER BY dataset,season",
        con
    )
display(warehouse_status)

# %% [markdown]
# ### 4A. Warehouse controls
# Use this optional cell to inspect cache state or deliberately invalidate only a specific current-season dataset. Normally you should not need to touch it.

# %%
def warehouse_summary():
    with _db() as con:
        return pd.read_sql_query(
            "SELECT dataset,season,refreshed_at_utc,row_count,max_week,source_status,parquet_path "
            "FROM dataset_state ORDER BY dataset,season", con
        )

def invalidate_cache(dataset, season=TARGET_SEASON):
    """Mark one partition stale without deleting its fallback parquet."""
    with _db() as con:
        con.execute(
            "UPDATE dataset_state SET refreshed_at_utc=? WHERE dataset=? AND season=?",
            ("1970-01-01T00:00:00+00:00", dataset, int(season))
        )
        con.commit()
    print(f"Marked stale: {dataset} {season}")

display(warehouse_summary())

# %% [markdown]
# ## 5. Sleeper current availability

# %%
def load_sleeper_players(force_refresh=False):
    SLEEPER_REFRESH_HOURS = 12.0
    if SLEEPER_CACHE_FILE.exists() and not force_refresh:
        age_h=(time.time()-SLEEPER_CACHE_FILE.stat().st_mtime)/3600.0
        if age_h < SLEEPER_REFRESH_HOURS:
            with open(SLEEPER_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    if SLEEPER_CACHE_FILE.exists() and not force_refresh:
        with open(SLEEPER_CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        print("Downloading Sleeper player state...")
        resp = requests.get(SLEEPER_PLAYERS_URL, timeout=60)
        resp.raise_for_status()
        raw = resp.json()
        with open(SLEEPER_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f)

    rows=[]
    for sid,p in raw.items():
        if not isinstance(p,dict): continue
        name=f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip()
        rows.append({
            "sleeper_id":str(sid), "player_name":name, "team":p.get("team"),
            "position":normalize_position(p.get("position")), "status":p.get("status"),
            "injury_status":p.get("injury_status"), "practice_participation":p.get("practice_participation"),
            "depth_chart_order":p.get("depth_chart_order"), "depth_chart_position":p.get("depth_chart_position"),
            "active":p.get("active"), "news_updated":p.get("news_updated")
        })
    d=pd.DataFrame(rows)
    d["name_key"]=clean_name(d["player_name"])
    return d

sleeper_players=load_sleeper_players(REFRESH_SLEEPER)
print("Sleeper rows:",len(sleeper_players))

# %% [markdown]
# ## 5A. Standardized personnel context — injuries, depth chart, preparation, replacements
# This is the shared context contract. It produces one row per current offensive player with availability probability, depth rank, practice/preparation score, roster tenure, and replacement weight. Questionable players are not automatically downgraded to zero; their status becomes a scenario probability.

# %%
def norm_text(v): return "" if pd.isna(v) else str(v).strip().lower()

def _pick_col(df, names):
    return first_existing(df, names, required=False) if len(df) else None

def current_roster_frame():
    # Prefer target-week weekly roster; otherwise latest available week <= target.
    if len(rosters_weekly):
        rw=rosters_weekly.copy()
        w=pd.to_numeric(rw.get("week"),errors="coerce")
        exact=rw[w.eq(TARGET_WEEK)].copy()
        if len(exact): use=exact; used=TARGET_WEEK
        else:
            prior=w[w.le(TARGET_WEEK)].dropna()
            used=int(prior.max()) if len(prior) else None
            use=rw[w.eq(used)].copy() if used is not None else pd.DataFrame()
        if len(use):
            n=_pick_col(use,["full_name","player_name","football_name"])
            t=_pick_col(use,["team","team_abbr","recent_team"])
            p=_pick_col(use,["position","position_group"])
            gid=_pick_col(use,["gsis_id","player_id"])
            st=_pick_col(use,["status"])
            out=pd.DataFrame({
                "player_name":use[n].astype(str), "team":use[t].astype(str),
                "position":use[p].map(normalize_position) if p else "UNK",
                "gsis_id":use[gid].astype(str) if gid else "",
                "roster_status":use[st].astype(str) if st else "",
            })
            out["roster_week_used"]=used
            out["name_key"]=clean_name(out["player_name"])
            return out.drop_duplicates(["name_key","team"])
    # Fallback season roster
    if len(rosters):
        n=_pick_col(rosters,["full_name","player_name","football_name"]); t=_pick_col(rosters,["team"]); p=_pick_col(rosters,["position","position_group"]); gid=_pick_col(rosters,["gsis_id"]); st=_pick_col(rosters,["status"])
        out=pd.DataFrame({"player_name":rosters[n].astype(str),"team":rosters[t].astype(str),"position":rosters[p].map(normalize_position) if p else "UNK","gsis_id":rosters[gid].astype(str) if gid else "","roster_status":rosters[st].astype(str) if st else ""})
        out["roster_week_used"]=np.nan; out["name_key"]=clean_name(out["player_name"])
        return out.drop_duplicates(["name_key","team"])
    return pd.DataFrame(columns=["player_name","team","position","gsis_id","roster_status","roster_week_used","name_key"])

def latest_depth_frame():
    if not len(depth_charts): return pd.DataFrame(columns=["name_key","team","depth_rank","depth_position"])
    dc=depth_charts.copy()
    if "dt" in dc.columns:
        dc["_dt"]=pd.to_datetime(dc["dt"],errors="coerce",utc=True)
        latest=dc["_dt"].max(); dc=dc[dc["_dt"].eq(latest)].copy()
        t=_pick_col(dc,["team","club_code"]); n=_pick_col(dc,["player_name","full_name","football_name"]); p=_pick_col(dc,["pos_abb","position","pos_name"]); rank=_pick_col(dc,["pos_rank","depth_team"])
    else:
        w=pd.to_numeric(dc.get("week"),errors="coerce") if "week" in dc.columns else pd.Series(np.nan,index=dc.index)
        prior=w[w.le(TARGET_WEEK)].dropna()
        if len(prior): dc=dc[w.eq(prior.max())].copy()
        t=_pick_col(dc,["club_code","team"]); n=_pick_col(dc,["full_name","football_name","player_name"]); p=_pick_col(dc,["position","depth_position"]); rank=_pick_col(dc,["depth_team","pos_rank"])
    if not all([t,n]): return pd.DataFrame(columns=["name_key","team","depth_rank","depth_position"])
    out=pd.DataFrame({"player_name":dc[n].astype(str),"team":dc[t].astype(str),"depth_position":dc[p].astype(str) if p else "","depth_rank":pd.to_numeric(dc[rank],errors="coerce") if rank else np.nan})
    out["name_key"]=clean_name(out["player_name"])
    out=out.sort_values("depth_rank").drop_duplicates(["name_key","team"],keep="first")
    return out[["name_key","team","depth_rank","depth_position"]]

def _practice_value(v):
    x=norm_text(v)
    if x in {"did not participate","dnp","did not practice"}: return 0.25
    if "limited" in x: return 0.65
    if "full" in x: return 1.00
    return np.nan

def current_injury_frame():
    cols=["name_key","team","official_report_status","official_practice_status",
          "primary_injury","practice_trend_score","practice_observations"]
    if not len(injuries): return pd.DataFrame(columns=cols)
    inj=injuries.copy()
    if "season" in inj.columns:
        inj=inj[pd.to_numeric(inj["season"],errors="coerce").eq(TARGET_SEASON)]
    if "week" in inj.columns:
        w=pd.to_numeric(inj["week"],errors="coerce")
        exact=inj[w.eq(TARGET_WEEK)].copy()
        if len(exact): inj=exact
        else:
            prior=w[w.le(TARGET_WEEK)].dropna()
            inj=inj[w.eq(prior.max())].copy() if len(prior) else inj.iloc[0:0]
    if not len(inj): return pd.DataFrame(columns=cols)

    name_col=_pick_col(inj,["full_name","player_name","football_name"])
    team_col=_pick_col(inj,["team","team_abbr","club_code"])
    report_col=_pick_col(inj,["report_status","game_status"])
    practice_col=_pick_col(inj,["practice_status","practice_participation"])
    injury_col=_pick_col(inj,["report_primary_injury","primary_injury","injury"])
    date_col=_pick_col(inj,["date_modified","report_date","date","practice_date"])

    if not name_col or not team_col:
        return pd.DataFrame(columns=cols)

    inj["_practice_value"]=inj[practice_col].map(_practice_value) if practice_col else np.nan
    if date_col:
        inj["_sort_dt"]=pd.to_datetime(inj[date_col],errors="coerce",utc=True)
    else:
        inj["_sort_dt"]=pd.RangeIndex(len(inj))

    rows=[]
    for (team,name),g in inj.sort_values("_sort_dt").groupby([team_col,name_col],dropna=False):
        g=g.copy()
        pvals=pd.to_numeric(g["_practice_value"],errors="coerce").dropna().tail(3).to_numpy(float)
        if len(pvals):
            weights=np.arange(1,len(pvals)+1,dtype=float); weights/=weights.sum()
            trend=float(np.dot(pvals,weights))
            # Small direction adjustment: improving late-week practice matters.
            if len(pvals)>=2:
                trend=float(np.clip(trend + 0.08*np.sign(pvals[-1]-pvals[0]),0,1))
        else:
            trend=np.nan
        last=g.iloc[-1]
        rows.append({
            "player_name":str(name),"team":str(team),
            "official_report_status":str(last[report_col]) if report_col else "",
            "official_practice_status":str(last[practice_col]) if practice_col else "",
            "primary_injury":str(last[injury_col]) if injury_col else "",
            "practice_trend_score":trend,
            "practice_observations":int(len(pvals))
        })
    out=pd.DataFrame(rows)
    out["name_key"]=clean_name(out["player_name"])
    return out[cols]

def status_availability(row):
    roster=norm_text(row.get("roster_status"))
    off=norm_text(row.get("official_report_status"))
    sl=norm_text(row.get("injury_status"))
    prac=norm_text(row.get("official_practice_status")) or norm_text(row.get("practice_participation"))
    trend=pd.to_numeric(pd.Series([row.get("practice_trend_score",np.nan)]),errors="coerce").iloc[0]

    hard_tokens={"out","ir","injured reserve","pup","reserve","res","suspended","sus","ret","cut"}
    if off=="out" or sl in hard_tokens or roster in {"res","pup","sus","ret","cut","dev","trc","trd","trl","trt"}:
        p=0.0; label="OUT"
    elif off=="doubtful" or sl=="doubtful":
        # Doubtful stays low, but a genuine late-week full practice can raise it modestly.
        base=0.12
        p=float(np.clip(base + (0.18*((trend if np.isfinite(trend) else 0.45)-0.45)),0.05,0.30))
        label="DOUBTFUL"
    elif off=="questionable" or sl=="questionable":
        # Do not assign every questionable player the same 68%.
        # Use practice trajectory when available; otherwise conservative fallback.
        t=float(trend) if np.isfinite(trend) else (_practice_value(prac) if np.isfinite(_practice_value(prac)) else 0.60)
        p=float(np.clip(0.42 + 0.48*t,0.48,0.88))
        label="QUESTIONABLE"
    else:
        p=0.995; label="ACTIVE_OR_CLEAR"

    if prac in {"did not participate","dnp","did not practice"}: practice_score=0.30
    elif "limited" in prac: practice_score=0.65
    elif "full" in prac: practice_score=1.0
    elif np.isfinite(trend): practice_score=float(trend)
    else: practice_score=0.82

    return pd.Series({
        "availability_probability":p,
        "availability_status":label,
        "practice_score":float(np.clip(practice_score,0,1))
    })

CURRENT_ROSTER=current_roster_frame()
DEPTH_NOW=latest_depth_frame()
INJURY_NOW=current_injury_frame()

sp=sleeper_players.copy()
sp=sp[[c for c in ["name_key","team","status","injury_status","practice_participation","depth_chart_order","depth_chart_position"] if c in sp.columns]].drop_duplicates(["name_key","team"])
PLAYER_CONTEXT=CURRENT_ROSTER.merge(DEPTH_NOW,on=["name_key","team"],how="left").merge(INJURY_NOW,on=["name_key","team"],how="left").merge(sp,on=["name_key","team"],how="left")
# Sleeper depth order is a fallback if nflverse depth rank is absent.
PLAYER_CONTEXT["depth_rank"]=pd.to_numeric(PLAYER_CONTEXT["depth_rank"],errors="coerce")
if "depth_chart_order" in PLAYER_CONTEXT.columns:
    PLAYER_CONTEXT["depth_rank"]=PLAYER_CONTEXT["depth_rank"].fillna(pd.to_numeric(PLAYER_CONTEXT["depth_chart_order"],errors="coerce"))
PLAYER_CONTEXT["depth_rank"]=PLAYER_CONTEXT["depth_rank"].fillna(9.0).clip(1,20)
PLAYER_CONTEXT=pd.concat([PLAYER_CONTEXT,PLAYER_CONTEXT.apply(status_availability,axis=1)],axis=1)

# Optional late official practice-status override.
if PLAYER_PRACTICE_OVERRIDES:
    for pname,pstatus in PLAYER_PRACTICE_OVERRIDES.items():
        m=PLAYER_CONTEXT["player_name"].eq(pname)
        if not m.any(): continue
        pv=_practice_value(pstatus)
        if np.isfinite(pv):
            PLAYER_CONTEXT.loc[m,"official_practice_status"]=str(pstatus)
            PLAYER_CONTEXT.loc[m,"practice_score"]=float(pv)


# Roster tenure through the target week is a preparation proxy.
if len(rosters_weekly):
    rw=rosters_weekly.copy(); rn=_pick_col(rw,["full_name","player_name","football_name"]); rt=_pick_col(rw,["team","team_abbr","recent_team"])
    if rn and rt and "week" in rw.columns:
        rw=rw[(pd.to_numeric(rw["week"],errors="coerce")<TARGET_WEEK)].copy(); rw["name_key"]=clean_name(rw[rn]); rw["team_key"]=rw[rt].astype(str)
        tenure=rw.groupby(["name_key","team_key"],as_index=False)["week"].nunique().rename(columns={"team_key":"team","week":"roster_weeks_prior"})
        PLAYER_CONTEXT=PLAYER_CONTEXT.merge(tenure,on=["name_key","team"],how="left")
PLAYER_CONTEXT["roster_weeks_prior"]=pd.to_numeric(PLAYER_CONTEXT.get("roster_weeks_prior",0),errors="coerce").fillna(0)
PLAYER_CONTEXT["system_tenure_score"]=(PLAYER_CONTEXT["roster_weeks_prior"]/6.0).clip(0,1)
PLAYER_CONTEXT["preparation_score"]=(0.65*PLAYER_CONTEXT["practice_score"]+0.35*PLAYER_CONTEXT["system_tenure_score"]).clip(0,1)
PLAYER_CONTEXT["replacement_weight"]=(np.exp(-0.70*(PLAYER_CONTEXT["depth_rank"]-1))*(0.55+0.45*PLAYER_CONTEXT["preparation_score"])).clip(0.02,1.0)

# Manual availability overrides take precedence.
PLAYER_CONTEXT["availability_probability"]=PLAYER_CONTEXT.apply(lambda r: float(PLAYER_AVAILABILITY_OVERRIDES.get(r["player_name"],r["availability_probability"])),axis=1).clip(0,1)
PLAYER_CONTEXT["availability_flag"]=np.select([
    PLAYER_CONTEXT["availability_probability"].le(0.01),
    PLAYER_CONTEXT["availability_probability"].lt(0.50),
    PLAYER_CONTEXT["availability_probability"].lt(0.90),
],["HARD_EXCLUDE","HIGH_RISK","REVIEW"],default="CLEAR")

print("Personnel context rows:",len(PLAYER_CONTEXT))
display(PLAYER_CONTEXT[PLAYER_CONTEXT["position"].isin(["QB","RB","WR","TE"])][["team","player_name","position","depth_rank","availability_probability","practice_score","preparation_score","replacement_weight","availability_flag"]].sort_values(["team","position","depth_rank"]).head(150))

# %% [markdown]
# ## 6. Standardize schedule, team games, and player games

# %%
# Schedule → one row per team-game
sch=schedule.copy()
if "game_type" in sch.columns:
    sch=sch[sch["game_type"].astype(str).str.upper().isin(["REG","REGULAR"])]

team_game_rows=[]
for g in sch.itertuples(index=False):
    d=g._asdict()
    season=int(d.get("season")) if pd.notna(d.get("season")) else None
    week=int(d.get("week")) if pd.notna(d.get("week")) else None
    game_id=d.get("game_id")
    home=d.get("home_team"); away=d.get("away_team")
    if season is None or week is None or not home or not away: continue
    team_game_rows += [
        {"season":season,"week":week,"game_id":game_id,"team":str(home),"opponent":str(away),"home":1},
        {"season":season,"week":week,"game_id":game_id,"team":str(away),"opponent":str(home),"home":0},
    ]
sched_team=pd.DataFrame(team_game_rows).drop_duplicates(["season","week","team"])

# Team weekly stats
TS=team_stats.copy()
if "season_type" in TS.columns: TS=TS[TS["season_type"].eq("REG")].copy()
for c in ["attempts","completions","passing_yards","carries","rushing_yards","targets"]:
    if c not in TS.columns: TS[c]=0.0
    TS[c]=pd.to_numeric(TS[c],errors="coerce").fillna(0.0)
TS=TS.merge(sched_team,on=["season","week","team"],how="left",suffixes=("","_sch"))
if "opponent_team" in TS.columns:
    TS["opponent"]=TS["opponent"].fillna(TS["opponent_team"].astype(str))
TS["target_rate"]=np.clip(safe_div(TS["targets"],TS["attempts"]),0.50,1.00)
TS["pass_ypa"]=safe_div(TS["passing_yards"],TS["attempts"])
TS["rush_ypc"]=safe_div(TS["rushing_yards"],TS["carries"])

# Player weekly stats
PS=player_stats.copy()
if "season_type" in PS.columns: PS=PS[PS["season_type"].eq("REG")].copy()
name_col=first_existing(PS,["player_display_name","player_name"],True)
id_col=first_existing(PS,["player_id","gsis_id"])
pos_col=first_existing(PS,["position","position_group"],True)
opp_col=first_existing(PS,["opponent_team","opponent"])
PS["player_name"]=PS[name_col]
PS["player_id_key"]=PS[id_col].astype(str) if id_col else clean_name(PS["player_name"])
PS["name_key"]=clean_name(PS["player_name"])
PS["position"]=PS[pos_col].map(normalize_position)
if opp_col: PS["opponent"]=PS[opp_col].astype(str)
else: PS=PS.merge(sched_team[["season","week","team","opponent"]],on=["season","week","team"],how="left")

for c in ["attempts","completions","passing_yards","carries","rushing_yards","targets","receptions","receiving_yards","receiving_air_yards"]:
    if c not in PS.columns: PS[c]=0.0
    PS[c]=pd.to_numeric(PS[c],errors="coerce").fillna(0.0)
PS["scrimmage_yards"]=PS["rushing_yards"]+PS["receiving_yards"]

# Restrict history to games before the requested forecast week.
def before_target(df):
    s=pd.to_numeric(df["season"],errors="coerce"); w=pd.to_numeric(df["week"],errors="coerce")
    return df[(s<TARGET_SEASON)|((s==TARGET_SEASON)&(w<TARGET_WEEK))].copy()
TS=before_target(TS); PS=before_target(PS)

# If team target counts are absent, reconstruct from player targets.
if TS["targets"].sum() == 0:
    fallback=PS.groupby(["season","week","team"],as_index=False)["targets"].sum().rename(columns={"targets":"targets_fb"})
    TS=TS.merge(fallback,on=["season","week","team"],how="left")
    TS["targets"]=TS["targets_fb"].fillna(0.0)
    TS["target_rate"]=np.clip(safe_div(TS["targets"],TS["attempts"]),0.50,1.00)

print("Historical team-games:",len(TS))
print("Historical player-games:",len(PS))

# %% [markdown]
# ## 7. Leakage-safe team and opponent features

# %%
TEAM_RAW=["attempts","carries","targets","target_rate","pass_ypa","rush_ypc"]
TS=TS.sort_values(["team","season","week"]).reset_index(drop=True)

# Current-season rolling form: reset every season and shift one game to prevent leakage.
for c in TEAM_RAW:
    TS[f"team_{c}_roll"]=(
        TS.groupby(["team","season"])[c]
          .transform(lambda s:s.shift(1).rolling(RECENT_GAMES,min_periods=1).mean())
    )
    TS[f"team_{c}_roll3"]=(
        TS.groupby(["team","season"])[c]
          .transform(lambda s:s.shift(1).rolling(SHORT_RECENT_GAMES,min_periods=1).mean())
    )
TS["team_games_prior"]=TS.groupby(["team","season"]).cumcount()

# Explicit previous-season prior.
team_prior=(TS.groupby(["team","season"],as_index=False)[TEAM_RAW].mean())
team_prior["season"]+=1
team_prior=team_prior.rename(columns={c:f"team_{c}_prior" for c in TEAM_RAW})
TS=TS.merge(team_prior,on=["team","season"],how="left")

# What each defense allowed in the opponent's offensive game.
DEF=TS[["season","week","opponent","attempts","carries","targets","target_rate","pass_ypa","rush_ypc"]].copy()
DEF=DEF.rename(columns={"opponent":"defense", **{c:f"def_{c}_allowed" for c in TEAM_RAW}})
DEF=DEF.sort_values(["defense","season","week"]).reset_index(drop=True)
DEF_RAW=[f"def_{c}_allowed" for c in TEAM_RAW]
for c in DEF_RAW:
    DEF[f"{c}_roll"]=(
        DEF.groupby(["defense","season"])[c]
           .transform(lambda s:s.shift(1).rolling(RECENT_GAMES,min_periods=1).mean())
    )
    DEF[f"{c}_roll3"]=(
        DEF.groupby(["defense","season"])[c]
           .transform(lambda s:s.shift(1).rolling(SHORT_RECENT_GAMES,min_periods=1).mean())
    )
def_prior=DEF.groupby(["defense","season"],as_index=False)[DEF_RAW].mean()
def_prior["season"]+=1
def_prior=def_prior.rename(columns={c:f"{c}_prior" for c in DEF_RAW})
DEF=DEF.merge(def_prior,on=["defense","season"],how="left")

DEF_FEATURES=(
    [f"{c}_roll" for c in DEF_RAW]
    + [f"{c}_roll3" for c in DEF_RAW]
    + [f"{c}_prior" for c in DEF_RAW]
)
TS=TS.merge(DEF[["season","week","defense"]+DEF_FEATURES],left_on=["season","week","opponent"],right_on=["season","week","defense"],how="left").drop(columns=["defense"],errors="ignore")

TEAM_FEATURES=(
    [f"team_{c}_roll" for c in TEAM_RAW]
    + [f"team_{c}_roll3" for c in TEAM_RAW]
    + [f"team_{c}_prior" for c in TEAM_RAW]
    + DEF_FEATURES
    + ["home","team_games_prior"]
)
finite_fill(TS,TEAM_FEATURES,0.0)

TEAM_TARGETS={
    "team_pass_attempts":{"column":"attempts","transform":"identity"},
    "team_rush_attempts":{"column":"carries","transform":"identity"},
    "team_target_rate":{"column":"target_rate","transform":"logit"},
}

print("Team features:",len(TEAM_FEATURES))
display(TS[["season","week","team","opponent"]+TEAM_FEATURES[:8]].tail(20))

# %% [markdown]
# ## 8. Player opportunity shares, efficiencies, priors, and rolling form

# %%
# Join snap share where available.
if len(snap_counts):
    SC=snap_counts.copy()
    sc_name=first_existing(SC,["player","player_name","player_display_name","full_name"])
    sc_team=first_existing(SC,["team"])
    sc_pct=first_existing(SC,["offense_pct","offensive_pct"])
    if sc_name and sc_team and sc_pct:
        SC["name_key"]=clean_name(SC[sc_name])
        SC["snap_pct"]=pd.to_numeric(SC[sc_pct],errors="coerce")
        SC.loc[SC["snap_pct"]>1.5,"snap_pct"]/=100.0
        snap_small=SC.groupby(["season","week",sc_team,"name_key"],as_index=False)["snap_pct"].max().rename(columns={sc_team:"team"})
        PS=PS.merge(snap_small,on=["season","week","team","name_key"],how="left")
    else: PS["snap_pct"]=np.nan
else: PS["snap_pct"]=np.nan

team_den=TS[["season","week","team","attempts","carries","targets"]].rename(columns={
    "attempts":"team_attempts","carries":"team_carries","targets":"team_targets"
})
PS=PS.merge(team_den,on=["season","week","team"],how="left")

PS["pass_attempt_share"]=np.clip(safe_div(PS["attempts"],PS["team_attempts"]),0,1)
PS["target_share"]=np.clip(safe_div(PS["targets"],PS["team_targets"]),0,1)
PS["carry_share"]=np.clip(safe_div(PS["carries"],PS["team_carries"]),0,1)
PS["target_active"]=(pd.to_numeric(PS["targets"],errors="coerce").fillna(0)>0).astype(int)
PS["carry_active"]=(pd.to_numeric(PS["carries"],errors="coerce").fillna(0)>0).astype(int)
PS["completion_rate"]=np.clip(safe_div(PS["completions"],PS["attempts"]),0.01,0.99)
PS["pass_ypc"]=safe_div(PS["passing_yards"],PS["completions"])
PS["catch_rate"]=np.clip(safe_div(PS["receptions"],PS["targets"]),0.01,0.99)
PS["rec_ypr"]=safe_div(PS["receiving_yards"],PS["receptions"])
PS["rush_ypc"]=safe_div(PS["rushing_yards"],PS["carries"])

PLAYER_RAW=[
    "attempts","completions","passing_yards","targets","receptions","receiving_yards","carries","rushing_yards",
    "pass_attempt_share","target_share","carry_share","target_active","carry_active","completion_rate","pass_ypc","catch_rate","rec_ypr","rush_ypc","snap_pct"
]
PS=PS.sort_values(["player_id_key","season","week"]).reset_index(drop=True)
for c in PLAYER_RAW:
    PS[f"{c}_roll"]=(
        PS.groupby(["player_id_key","season"])[c]
          .transform(lambda s:s.shift(1).rolling(RECENT_GAMES,min_periods=1).mean())
    )
    PS[f"{c}_roll3"]=(
        PS.groupby(["player_id_key","season"])[c]
          .transform(lambda s:s.shift(1).rolling(SHORT_RECENT_GAMES,min_periods=1).mean())
    )
PS["games_prior_season"]=PS.groupby(["player_id_key","season"]).cumcount()
PS["career_games_prior"]=PS.groupby("player_id_key").cumcount()

player_prior=PS.groupby(["player_id_key","season"],as_index=False)[PLAYER_RAW].mean()
player_prior["season"]+=1
player_prior=player_prior.rename(columns={c:f"{c}_prior" for c in PLAYER_RAW})
PS=PS.merge(player_prior,on=["player_id_key","season"],how="left")

# Team + defense contextual features already computed pregame.
team_context=["season","week","team"]+TEAM_FEATURES
PS=PS.merge(TS[team_context],on=["season","week","team"],how="left")

for pos in ["QB","RB","WR","TE"]:
    PS[f"pos_{pos}"]=(PS["position"].eq(pos)).astype(int)

# V6 position-specific opponent context.  This answers questions such as:
# - how many targets/yards has this defense been allowing to WRs?
# - how efficiently have QBs produced against it?
# - how much rushing work has it allowed to RBs/QBs?
# Every historical row is shifted within defense/position/season before use.
POSDEF_COUNT_COLS=["attempts","completions","passing_yards","targets","receptions","receiving_yards","carries","rushing_yards"]
PD=(PS.groupby(["season","week","opponent","position"],as_index=False)[POSDEF_COUNT_COLS].sum()
      .rename(columns={"opponent":"defense"}))
PD["completion_rate"]=np.clip(safe_div(PD["completions"],PD["attempts"]),0,1)
PD["pass_ypa"]=safe_div(PD["passing_yards"],PD["attempts"])
PD["catch_rate"]=np.clip(safe_div(PD["receptions"],PD["targets"]),0,1)
PD["rec_ypr"]=safe_div(PD["receiving_yards"],PD["receptions"])
PD["rush_ypc"]=safe_div(PD["rushing_yards"],PD["carries"])
POSDEF_RAW=POSDEF_COUNT_COLS+["completion_rate","pass_ypa","catch_rate","rec_ypr","rush_ypc"]
PD=PD.sort_values(["defense","position","season","week"]).reset_index(drop=True)
for c in POSDEF_RAW:
    PD[f"posdef_{c}_roll"]=(PD.groupby(["defense","position","season"])[c]
        .transform(lambda x:x.shift(1).rolling(RECENT_GAMES,min_periods=1).mean()))
    PD[f"posdef_{c}_roll3"]=(PD.groupby(["defense","position","season"])[c]
        .transform(lambda x:x.shift(1).rolling(SHORT_RECENT_GAMES,min_periods=1).mean()))
pd_prior=PD.groupby(["defense","position","season"],as_index=False)[POSDEF_RAW].mean()
pd_prior["season"]+=1
pd_prior=pd_prior.rename(columns={c:f"posdef_{c}_prior" for c in POSDEF_RAW})
PD=PD.merge(pd_prior,on=["defense","position","season"],how="left")
POSDEF_FEATURES=(
    [f"posdef_{c}_roll" for c in POSDEF_RAW]
    + [f"posdef_{c}_roll3" for c in POSDEF_RAW]
    + [f"posdef_{c}_prior" for c in POSDEF_RAW]
)
PS=PS.merge(PD[["season","week","defense","position"]+POSDEF_FEATURES],
            left_on=["season","week","opponent","position"],
            right_on=["season","week","defense","position"],how="left").drop(columns=["defense"],errors="ignore")

PLAYER_FEATURES=(
    [f"{c}_roll" for c in PLAYER_RAW]
    + [f"{c}_roll3" for c in PLAYER_RAW]
    + [f"{c}_prior" for c in PLAYER_RAW]
    + TEAM_FEATURES
    + POSDEF_FEATURES
    + ["games_prior_season","career_games_prior","pos_QB","pos_RB","pos_WR","pos_TE"]
)
finite_fill(PS,PLAYER_FEATURES,0.0)

COMPONENTS={
    "qb_attempt_share":{"column":"pass_attempt_share","transform":"logit","positions":{"QB"},"min_opps_col":"attempts","min_opps":1},
    "qb_completion_rate":{"column":"completion_rate","transform":"logit","positions":{"QB"},"min_opps_col":"attempts","min_opps":8},
    "qb_pass_ypc":{"column":"pass_ypc","transform":"log1p","positions":{"QB"},"min_opps_col":"completions","min_opps":5},
    "target_share":{"column":"target_share","transform":"logit","positions":{"RB","WR","TE"},"min_opps_col":"targets","min_opps":1},
    "catch_rate":{"column":"catch_rate","transform":"logit","positions":{"RB","WR","TE"},"min_opps_col":"targets","min_opps":2},
    "rec_ypr":{"column":"rec_ypr","transform":"log1p","positions":{"RB","WR","TE"},"min_opps_col":"receptions","min_opps":1},
    "carry_share":{"column":"carry_share","transform":"logit","positions":{"QB","RB","WR","TE"},"min_opps_col":"carries","min_opps":1},
    "rush_ypc":{"column":"rush_ypc","transform":"identity","positions":{"QB","RB","WR","TE"},"min_opps_col":"carries","min_opps":2},
}

print("Player features:",len(PLAYER_FEATURES))
display(PS[["season","week","player_name","team","position","target_share","carry_share","completion_rate","catch_rate"]].tail(30))

# %% [markdown]
# ## 9. Current-week player pool and forecast feature snapshots

# %%
week_schedule=schedule[(pd.to_numeric(schedule["season"],errors="coerce")==TARGET_SEASON)&(pd.to_numeric(schedule["week"],errors="coerce")==TARGET_WEEK)].copy()
if len(week_schedule)==0: raise RuntimeError("No target-week schedule found.")

if TARGET_GAME_TEAMS:
    tf=set(TARGET_GAME_TEAMS)
    week_schedule=week_schedule[
        week_schedule["home_team"].astype(str).isin(tf) |
        week_schedule["away_team"].astype(str).isin(tf)
    ].copy()
    if len(week_schedule)==0:
        raise RuntimeError(f"No target-week game matched TARGET_GAME_TEAMS={TARGET_GAME_TEAMS}.")

pairs=[]
for _,g in week_schedule.iterrows():
    pairs += [
        {"game_id":g["game_id"],"team":g["away_team"],"opponent":g["home_team"],"home":0},
        {"game_id":g["game_id"],"team":g["home_team"],"opponent":g["away_team"],"home":1},
    ]
pairs=pd.DataFrame(pairs)

# Seed the player pool from the CURRENT roster, not historical box scores. This is the
# critical replacement fix: a veteran backup with zero current-season snaps still exists.
current=PLAYER_CONTEXT[PLAYER_CONTEXT["position"].isin(["QB","RB","WR","TE"])].copy()
current=current[current["team"].isin(pairs["team"])].copy()
current=current.merge(pairs,on="team",how="inner")

# Historical identity / latest team history. Prefer GSIS id; otherwise name-key mapping.
hist_identity=(PS.sort_values(["season","week"]).groupby("player_id_key",as_index=False).tail(1)[["player_id_key","player_name","name_key","position","team"]])
name_to_pid=(hist_identity.sort_values("player_id_key").drop_duplicates("name_key")[["name_key","player_id_key"]])
current=current.merge(name_to_pid,on="name_key",how="left")
current["player_id_key"]=current["player_id_key"].where(current["player_id_key"].notna(),current["gsis_id"].replace({"":"nan"}))
current["player_id_key"]=current["player_id_key"].where(current["player_id_key"].notna() & current["player_id_key"].ne("nan"),"ROSTER_"+current["name_key"])
current["player_name"]=current["player_name"].astype(str)

# Current-season offensive snap totals. Players with zero snaps are retained if they are
# plausible depth-chart replacements (especially backup QBs).
current["current_season_off_snaps"]=0.0
if len(snap_counts) and TARGET_WEEK>1:
    SC=snap_counts.copy(); sn=first_existing(SC,["player","player_name","player_display_name","full_name"],True); st=first_existing(SC,["team"],True); ss=first_existing(SC,["offense_snaps","offensive_snaps"]); spct=first_existing(SC,["offense_pct","offensive_pct"])
    SC=SC[(pd.to_numeric(SC["season"],errors="coerce")==TARGET_SEASON)&(pd.to_numeric(SC["week"],errors="coerce")<TARGET_WEEK)].copy(); SC["name_key"]=clean_name(SC[sn]); SC["team_key"]=SC[st].astype(str)
    if ss: SC["off_value"]=pd.to_numeric(SC[ss],errors="coerce").fillna(0)
    elif spct: SC["off_value"]=(pd.to_numeric(SC[spct],errors="coerce").fillna(0)>0).astype(float)
    else: SC["off_value"]=0.0
    sg=SC.groupby(["name_key","team_key"],as_index=False)["off_value"].sum().rename(columns={"team_key":"team","off_value":"snap_total"})
    current=current.merge(sg,on=["name_key","team"],how="left"); current["current_season_off_snaps"]=pd.to_numeric(current["snap_total"],errors="coerce").fillna(0); current=current.drop(columns=["snap_total"])

current["replacement_candidate"]=(current["depth_rank"].le(3)&current["availability_probability"].gt(0.01)).astype(int)
if REQUIRE_CURRENT_SEASON_OFFENSIVE_SNAPS and TARGET_WEEK>1:
    keep=current["current_season_off_snaps"].gt(0)
    if ALLOW_DEPTH_CHART_REPLACEMENT_EXCEPTION: keep=keep|current["replacement_candidate"].eq(1)
    current=current[keep].copy()
current=current[current["availability_probability"].gt(0.001)].copy()

# -------- forecast snapshots: INCLUDE every completed game through Week TARGET_WEEK-1 --------
snap_rows=[]
last_rows=[]
for pid,g in PS.groupby("player_id_key"):
    g=g.sort_values(["season","week"]); cur=g[g["season"].eq(TARGET_SEASON)]; prior=g[g["season"].eq(TARGET_SEASON-1)]
    row={"player_id_key":pid}
    for c in PLAYER_RAW:
        row[f"{c}_roll"]=float(cur[c].tail(RECENT_GAMES).mean()) if len(cur) else np.nan
        row[f"{c}_roll3"]=float(cur[c].tail(SHORT_RECENT_GAMES).mean()) if len(cur) else np.nan
        row[f"{c}_prior"]=float(prior[c].mean()) if len(prior) else np.nan
    row["games_prior_season"]=len(cur); row["career_games_prior"]=len(g); snap_rows.append(row)
    last=g.iloc[-1]; last_rows.append({"player_id_key":pid,"last_season":int(last["season"]),"last_week":int(last["week"]),"career_box_games":len(g),"qb_pseudo_starts":int(((g["position"].eq("QB"))&(pd.to_numeric(g["attempts"],errors="coerce").fillna(0)>=10)).sum())})
player_snapshot=pd.DataFrame(snap_rows); last_activity=pd.DataFrame(last_rows)

team_snap=[]
for team,g in TS.groupby("team"):
    cur=g[g["season"].eq(TARGET_SEASON)]; prior=g[g["season"].eq(TARGET_SEASON-1)]; r={"team":team}
    for c in TEAM_RAW:
        r[f"team_{c}_roll"]=float(cur[c].tail(RECENT_GAMES).mean()) if len(cur) else 0.0
        r[f"team_{c}_roll3"]=float(cur[c].tail(SHORT_RECENT_GAMES).mean()) if len(cur) else 0.0
        r[f"team_{c}_prior"]=float(prior[c].mean()) if len(prior) else 0.0
    r["team_games_prior"]=len(cur); team_snap.append(r)
team_snapshot=pd.DataFrame(team_snap)

def_snap=[]
for defense,g in DEF.groupby("defense"):
    cur=g[g["season"].eq(TARGET_SEASON)]; prior=g[g["season"].eq(TARGET_SEASON-1)]; r={"defense":defense}
    for c in DEF_RAW:
        r[f"{c}_roll"]=float(cur[c].tail(RECENT_GAMES).mean()) if len(cur) else 0.0
        r[f"{c}_roll3"]=float(cur[c].tail(SHORT_RECENT_GAMES).mean()) if len(cur) else 0.0
        r[f"{c}_prior"]=float(prior[c].mean()) if len(prior) else 0.0
    def_snap.append(r)
def_snapshot=pd.DataFrame(def_snap)

# Position-specific opponent snapshot for the forecast week.
posdef_snap=[]
for (defense,pos),g in PD.groupby(["defense","position"]):
    cur=g[g["season"].eq(TARGET_SEASON)].sort_values("week")
    prior=g[g["season"].eq(TARGET_SEASON-1)]
    r={"defense":defense,"position":pos}
    for c in POSDEF_RAW:
        r[f"posdef_{c}_roll"]=float(cur[c].tail(RECENT_GAMES).mean()) if len(cur) else np.nan
        r[f"posdef_{c}_roll3"]=float(cur[c].tail(SHORT_RECENT_GAMES).mean()) if len(cur) else np.nan
        r[f"posdef_{c}_prior"]=float(prior[c].mean()) if len(prior) else np.nan
    posdef_snap.append(r)
posdef_snapshot=pd.DataFrame(posdef_snap)

current=current.merge(player_snapshot,on="player_id_key",how="left").merge(last_activity,on="player_id_key",how="left")
current=current.merge(team_snapshot,on="team",how="left").merge(def_snapshot,left_on="opponent",right_on="defense",how="left").drop(columns=["defense"],errors="ignore")
if len(posdef_snapshot):
    current=current.merge(posdef_snapshot,left_on=["opponent","position"],right_on=["defense","position"],how="left").drop(columns=["defense"],errors="ignore")

# Position-population fallback for true backups/rookies with little or no box-score history.
for c in PLAYER_RAW:
    roll=f"{c}_roll"; prior=f"{c}_prior"
    pos_med=PS.groupby("position")[c].median().to_dict()
    roll3=f"{c}_roll3"
    current[roll]=pd.to_numeric(current[roll],errors="coerce")
    current[roll3]=pd.to_numeric(current[roll3],errors="coerce")
    current[prior]=pd.to_numeric(current[prior],errors="coerce")
    current[roll]=current.apply(lambda r: pos_med.get(r["position"],0.0) if pd.isna(r[roll]) else r[roll],axis=1)
    current[roll3]=current.apply(lambda r: pos_med.get(r["position"],0.0) if pd.isna(r[roll3]) else r[roll3],axis=1)
    current[prior]=current.apply(lambda r: pos_med.get(r["position"],0.0) if pd.isna(r[prior]) else r[prior],axis=1)
current["games_prior_season"]=pd.to_numeric(current["games_prior_season"],errors="coerce").fillna(0)
current["career_games_prior"]=pd.to_numeric(current["career_games_prior"],errors="coerce").fillna(0)

# Inactivity and uncertainty. A stale veteran is shrunk/widened, not treated as zero-quality.
target_ord=TARGET_SEASON*25+TARGET_WEEK
last_ord=pd.to_numeric(current["last_season"],errors="coerce").fillna(TARGET_SEASON-4)*25+pd.to_numeric(current["last_week"],errors="coerce").fillna(1)
current["inactivity_weeks"]=(target_ord-last_ord).clip(lower=0)
current["history_reliability"]=(np.sqrt(current["career_games_prior"].clip(lower=0)/16.0).clip(0,1)*np.exp(-current["inactivity_weeks"]/110.0)).clip(0.05,1.0)
current["uncertainty_multiplier"]=(1.0 + 0.22/np.sqrt(current["career_games_prior"].clip(lower=0)+1) + 0.13*np.log1p(current["inactivity_weeks"]/8.0) + 0.30*(1-current["preparation_score"]) + 0.18*(1-current["availability_probability"])).clip(1.0,MAX_CONTEXT_UNCERTAINTY_MULT)

current["manual_role_factor"]=current["player_name"].map(PLAYER_ROLE_OVERRIDES).fillna(1.0).astype(float).clip(0,2)
current=current[current["manual_role_factor"].gt(0)].copy()
for pos in ["QB","RB","WR","TE"]: current[f"pos_{pos}"]=(current["position"].eq(pos)).astype(int)
finite_fill(current,PLAYER_FEATURES,0.0)

current_team=pairs.merge(team_snapshot,on="team",how="left").merge(def_snapshot,left_on="opponent",right_on="defense",how="left").drop(columns=["defense"],errors="ignore")
finite_fill(current_team,TEAM_FEATURES,0.0)

print("Current players:",len(current),"| teams:",len(current_team))
display(current[["game_id","team","opponent","player_name","position","depth_rank","availability_probability","preparation_score","current_season_off_snaps","replacement_candidate","career_games_prior","inactivity_weeks","history_reliability","uncertainty_multiplier"]].sort_values(["game_id","team","position","depth_rank"]).head(160))

# %% [markdown]
# ## 10. RF / XGB / Ridge walk-forward ensemble

# %%
def make_models():
    return {
        "RF":RandomForestRegressor(
            n_estimators=520,max_depth=11,min_samples_leaf=5,max_features=0.78,
            random_state=RANDOM_SEED,n_jobs=-1
        ),
        "XGB":XGBRegressor(
            n_estimators=520,max_depth=5,learning_rate=0.025,subsample=0.88,
            colsample_bytree=0.88,min_child_weight=6,reg_lambda=3.0,reg_alpha=0.05,
            objective="reg:squarederror",random_state=RANDOM_SEED,n_jobs=-1
        ),
        "REG":Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=10.0))])
    }

def chronological_folds(d, n_folds=N_WALK_FOLDS):
    """Expanding whole-week folds. No validation week can appear in its own training set."""
    order=(pd.to_numeric(d["season"],errors="coerce").fillna(0).astype(int)*100 +
           pd.to_numeric(d["week"],errors="coerce").fillna(0).astype(int))
    uniq=np.array(sorted(order.unique()))
    if len(uniq)<8:
        cut_week=max(1,int(len(uniq)*0.75))
        tr=np.flatnonzero(order.isin(uniq[:cut_week]).to_numpy())
        va=np.flatnonzero(order.isin(uniq[cut_week:]).to_numpy())
        return [(tr,va)] if len(tr) and len(va) else []
    first_valid=max(4,int(len(uniq)*0.48))
    valid_weeks=uniq[first_valid:]
    blocks=[x for x in np.array_split(valid_weeks,min(n_folds,len(valid_weeks))) if len(x)]
    folds=[]
    for block in blocks:
        cutoff=int(block[0])
        tr=np.flatnonzero((order<cutoff).to_numpy())
        va=np.flatnonzero(order.isin(block).to_numpy())
        if len(tr) and len(va): folds.append((tr,va))
    return folds

def _inverse_error_weights(score_history, model_names):
    vals={}
    for name in model_names:
        hist=score_history.get(name,[])
        vals[name]=1.0/max(float(np.mean(hist)),1e-6) if hist else 1.0
    z=sum(vals.values()) or 1.0
    return {k:v/z for k,v in vals.items()}

def _optimize_regression_weights(oof_df, transform_kind, model_names):
    cols=[f"pred_tx_{m}" for m in model_names]
    q=oof_df.dropna(subset=cols+["y_nat"]).copy()
    if len(q)<50:
        return {m:1/len(model_names) for m in model_names}
    P=q[cols].to_numpy(float); y=q["y_nat"].to_numpy(float)
    def obj(w):
        pred=inverse_target(P@w,transform_kind)
        mae=float(np.mean(np.abs(y-pred)))
        r=float(np.sqrt(np.mean((y-pred)**2)))
        return mae + ENSEMBLE_RMSE_PENALTY*r
    x0=np.full(len(model_names),1/len(model_names))
    try:
        res=minimize(obj,x0,method="SLSQP",bounds=[(0,1)]*len(model_names),
                     constraints={"type":"eq","fun":lambda w:np.sum(w)-1},
                     options={"maxiter":300,"ftol":1e-10})
        w=res.x if res.success and np.isfinite(res.x).all() else x0
    except Exception:
        w=x0
    w=np.clip(w,0,1); w=w/(w.sum() or 1.0)
    return {m:float(v) for m,v in zip(model_names,w)}

def _adaptive_residual_scale(oof_df):
    """Compare recent OOF absolute errors with earlier OOF errors; future-only scale."""
    if len(oof_df)<120 or "season_week" not in oof_df: return 1.0
    q=oof_df.sort_values("season_week").copy()
    split=max(60,int(len(q)*0.70))
    early=np.abs(q.iloc[:split]["resid_tx"].to_numpy(float))
    recent=np.abs(q.iloc[split:]["resid_tx"].to_numpy(float))
    e=np.median(early[np.isfinite(early)]) if np.isfinite(early).any() else np.nan
    r=np.median(recent[np.isfinite(recent)]) if np.isfinite(recent).any() else np.nan
    if not np.isfinite(e) or not np.isfinite(r) or e<1e-8: return 1.0
    return float(np.clip(r/e,MIN_RESIDUAL_SCALE,MAX_RESIDUAL_SCALE))

def fit_component(d, features, target_col, transform_kind, label):
    d=d.copy().sort_values(["season","week"]).reset_index(drop=True)
    d["_y_nat"]=pd.to_numeric(d[target_col],errors="coerce")
    d=d[np.isfinite(d["_y_nat"])].copy().reset_index(drop=True)
    d["_y_tx"]=transform_target(d["_y_nat"].to_numpy(),transform_kind)
    d["season_week"]=d["season"].astype(int)*100+d["week"].astype(int)
    folds=chronological_folds(d)
    oof=[]; metric_rows=[]; score_history={k:[] for k in make_models()}
    for fold_no,(tr_idx,va_idx) in enumerate(folds,1):
        tr=d.iloc[tr_idx]; va=d.iloc[va_idx]
        if len(tr)<100 or len(va)<20: continue
        age=(tr["season"].astype(int)*25+tr["week"].astype(int)).max()-(tr["season"].astype(int)*25+tr["week"].astype(int))
        sw=recency_weight(age); preds_tx={}; preds_nat={}
        for name,m in make_models().items():
            try:
                if name in {"RF","XGB"}: m.fit(tr[features],tr["_y_tx"],sample_weight=sw)
                else: m.fit(tr[features],tr["_y_tx"])
                ptx=np.asarray(m.predict(va[features]),float); pnat=inverse_target(ptx,transform_kind)
                preds_tx[name]=ptx; preds_nat[name]=pnat
                mae=mean_absolute_error(va["_y_nat"],pnat)
                metric_rows.append({"component":label,"fold":fold_no,"model":name,"MAE":mae,"RMSE":rmse(va["_y_nat"],pnat),"n_train":len(tr),"n_valid":len(va)})
            except Exception as e:
                print(label,name,"fold",fold_no,"failed:",e)
        if not preds_tx: continue
        # Causal OOF ensemble: current validation outcomes never determine their own weights.
        fw=_inverse_error_weights(score_history,list(preds_tx))
        ens_tx=sum(fw[k]*preds_tx[k] for k in preds_tx)
        for j in range(len(va)):
            rec={"y_nat":float(va.iloc[j]["_y_nat"]),"y_tx":float(va.iloc[j]["_y_tx"]),
                 "pred_tx":float(ens_tx[j]),"resid_tx":float(va.iloc[j]["_y_tx"]-ens_tx[j]),
                 "season_week":int(va.iloc[j]["season_week"]),"fold":fold_no}
            for k in preds_tx: rec[f"pred_tx_{k}"]=float(preds_tx[k][j])
            oof.append(rec)
        for k in preds_nat:
            score_history[k].append(max(mean_absolute_error(va["_y_nat"],preds_nat[k]),1e-6))
    metrics=pd.DataFrame(metric_rows); oof_df=pd.DataFrame(oof)
    if len(metrics)==0 or len(oof_df)==0: raise RuntimeError(f"No valid walk-forward fits for {label}")
    model_names=[m for m in make_models() if f"pred_tx_{m}" in oof_df.columns]
    weights=_optimize_regression_weights(oof_df,transform_kind,model_names)
    age=(d["season"].astype(int)*25+d["week"].astype(int)).max()-(d["season"].astype(int)*25+d["week"].astype(int))
    sw=recency_weight(age); fitted={}
    for name,m in make_models().items():
        if name not in weights: continue
        if name in {"RF","XGB"}: m.fit(d[features],d["_y_tx"],sample_weight=sw)
        else: m.fit(d[features],d["_y_tx"])
        fitted[name]=m
    residual=oof_df["resid_tx"].to_numpy(float); residual=residual[np.isfinite(residual)]
    return {"models":fitted,"weights":weights,"metrics":metrics,"oof":oof_df,
            "residual_tx":residual,"residual_scale":_adaptive_residual_scale(oof_df),
            "transform":transform_kind,"features":features,"train_rows":len(d),"target_col":target_col}


def make_role_models():
    return {
        "RF":RandomForestClassifier(n_estimators=520,max_depth=11,min_samples_leaf=5,max_features=0.78,random_state=RANDOM_SEED,n_jobs=-1),
        "XGB":XGBClassifier(n_estimators=520,max_depth=5,learning_rate=0.025,subsample=0.88,colsample_bytree=0.88,min_child_weight=6,reg_lambda=3.0,reg_alpha=0.05,objective="binary:logistic",eval_metric="logloss",random_state=RANDOM_SEED,n_jobs=-1),
        "LOG":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=0.30,max_iter=1200))])
    }

def _optimize_brier_weights(oof_df, model_names):
    cols=[f"p_{m}" for m in model_names]
    q=oof_df.dropna(subset=cols+["y"]).copy()
    if len(q)<100: return {m:1/len(model_names) for m in model_names}
    P=q[cols].to_numpy(float); y=q["y"].to_numpy(float)
    def obj(w): return float(np.mean((y-P@w)**2))
    x0=np.full(len(model_names),1/len(model_names))
    try:
        res=minimize(obj,x0,method="SLSQP",bounds=[(0,1)]*len(model_names),
                     constraints={"type":"eq","fun":lambda w:np.sum(w)-1},options={"maxiter":300,"ftol":1e-12})
        w=res.x if res.success and np.isfinite(res.x).all() else x0
    except Exception: w=x0
    w=np.clip(w,0,1); w=w/(w.sum() or 1.0)
    return {m:float(v) for m,v in zip(model_names,w)}

def _validated_isotonic_calibrator(oof_df, model_names):
    """
    Choose isotonic only if it improves a chronologically later holdout.
    The holdout's outcomes do NOT participate in the ensemble weights used to test
    calibration, so this remains a genuine later-block validation.
    """
    q=oof_df.sort_values("season_week").copy()
    if len(q)<HURDLE_ISOTONIC_MIN_OOF or q["y"].nunique()<2:
        return None,{"accepted":False,"reason":"INSUFFICIENT_OOF"}
    cut=max(100,int(len(q)*(1-ROLE_CALIBRATION_HOLDOUT_FRAC)))
    tr=q.iloc[:cut].copy(); va=q.iloc[cut:].copy()
    if len(va)<50 or tr["y"].nunique()<2 or va["y"].nunique()<2:
        return None,{"accepted":False,"reason":"INSUFFICIENT_HOLDOUT"}
    try:
        test_w=_optimize_brier_weights(tr,model_names)
        tr_raw=sum(test_w[m]*tr[f"p_{m}"] for m in model_names)
        va_raw=sum(test_w[m]*va[f"p_{m}"] for m in model_names)
        raw=float(brier_score_loss(va["y"],va_raw))
        test_iso=IsotonicRegression(out_of_bounds="clip").fit(tr_raw,tr["y"])
        pc=np.clip(test_iso.predict(va_raw),1e-5,1-1e-5)
        cal=float(brier_score_loss(va["y"],pc)); gain=raw-cal
        if gain>=ROLE_CALIBRATION_MIN_BRIER_GAIN:
            final_w=_optimize_brier_weights(q,model_names)
            q_raw=sum(final_w[m]*q[f"p_{m}"] for m in model_names)
            final_iso=IsotonicRegression(out_of_bounds="clip").fit(q_raw,q["y"])
            return final_iso,{"accepted":True,"reason":"HOLDOUT_BRIER_IMPROVED","raw_brier":raw,"cal_brier":cal,"gain":gain,"n":len(q)}
        return None,{"accepted":False,"reason":"NO_HOLDOUT_GAIN","raw_brier":raw,"cal_brier":cal,"gain":gain,"n":len(q)}
    except Exception as e:
        return None,{"accepted":False,"reason":f"CALIBRATION_FAILED:{type(e).__name__}"}

def fit_role_classifier(d, features, target_col, label):
    d=d.copy().sort_values(["season","week"]).reset_index(drop=True)
    d["_y"]=pd.to_numeric(d[target_col],errors="coerce").fillna(0).astype(int).clip(0,1)
    d["season_week"]=d["season"].astype(int)*100+d["week"].astype(int)
    folds=chronological_folds(d)
    oof=[]; metric_rows=[]; score_history={k:[] for k in make_role_models()}
    for fold_no,(tr_idx,va_idx) in enumerate(folds,1):
        tr=d.iloc[tr_idx]; va=d.iloc[va_idx]
        if len(tr)<150 or len(va)<30 or tr["_y"].nunique()<2 or va["_y"].nunique()<2: continue
        age=(tr["season"].astype(int)*25+tr["week"].astype(int)).max()-(tr["season"].astype(int)*25+tr["week"].astype(int))
        sw=recency_weight(age); preds={}
        for name,m in make_role_models().items():
            try:
                if name in {"RF","XGB"}: m.fit(tr[features],tr["_y"],sample_weight=sw)
                else: m.fit(tr[features],tr["_y"])
                p=np.clip(m.predict_proba(va[features])[:,1],1e-5,1-1e-5); preds[name]=p
                br=brier_score_loss(va["_y"],p)
                metric_rows.append({"component":label,"fold":fold_no,"model":name,"Brier":br,"LogLoss":log_loss(va["_y"],p,labels=[0,1]),"n_train":len(tr),"n_valid":len(va)})
            except Exception as e: print(label,name,"fold",fold_no,"failed:",e)
        if not preds: continue
        fw=_inverse_error_weights(score_history,list(preds))
        ens=sum(fw[k]*preds[k] for k in preds)
        for j,y in enumerate(va["_y"].to_numpy(int)):
            rec={"y":int(y),"p_raw":float(ens[j]),"season_week":int(va.iloc[j]["season_week"]),"fold":fold_no}
            for k in preds: rec[f"p_{k}"]=float(preds[k][j])
            oof.append(rec)
        for k,p in preds.items(): score_history[k].append(max(brier_score_loss(va["_y"],p),1e-6))
    metrics=pd.DataFrame(metric_rows); oof_df=pd.DataFrame(oof)
    if len(metrics)==0 or len(oof_df)==0: raise RuntimeError(f"No valid walk-forward role fits for {label}")
    model_names=[m for m in make_role_models() if f"p_{m}" in oof_df.columns]
    weights=_optimize_brier_weights(oof_df,model_names)
    # Production OOF probability under final historical weights is retained for audit,
    # but calibration-family acceptance uses its own pre-holdout weights.
    oof_df["p_raw_final_weight"]=sum(weights[m]*oof_df[f"p_{m}"] for m in model_names)
    calibrator,calibration_info=_validated_isotonic_calibrator(oof_df,model_names)
    age=(d["season"].astype(int)*25+d["week"].astype(int)).max()-(d["season"].astype(int)*25+d["week"].astype(int))
    sw=recency_weight(age); fitted={}
    for name,m in make_role_models().items():
        if name in {"RF","XGB"}: m.fit(d[features],d["_y"],sample_weight=sw)
        else: m.fit(d[features],d["_y"])
        fitted[name]=m
    return {"models":fitted,"weights":weights,"metrics":metrics,"oof":oof_df,"features":features,
            "train_rows":len(d),"target_col":target_col,"calibrator":calibrator,
            "calibration_info":calibration_info,"population_rate":float(d["_y"].mean())}

ROLE_META={
    "target_active_prob":{"column":"target_active","positions":{"RB","WR","TE"}},
    "carry_active_prob":{"column":"carry_active","positions":{"QB","RB","WR","TE"}},
}
ROLE_MODELS={}; TEAM_MODELS={}; PLAYER_MODELS={}
role_metrics=[]; all_metrics=[]

MODEL_ARTIFACT_DIR=WAREHOUSE_DIR / "model_artifacts_v6_1"
MODEL_ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)

def _frame_fingerprint(df, cols):
    use=[c for c in cols if c in df.columns]
    if not use: return "EMPTY"
    q=df[use].copy()
    hv=pd.util.hash_pandas_object(q,index=True).values.tobytes()
    return hashlib.sha256(hv).hexdigest()

def _training_fingerprint():
    payload={
        "schema":V61_MODEL_SCHEMA_VERSION,"season":TARGET_SEASON,"week":TARGET_WEEK,
        "team_features":TEAM_FEATURES,"player_features":PLAYER_FEATURES,
        "team_hash":_frame_fingerprint(TS,["season","week","team"]+TEAM_FEATURES+[x["column"] for x in TEAM_TARGETS.values()]),
        "player_hash":_frame_fingerprint(PS,["season","week","player_id_key","position"]+PLAYER_FEATURES+[x["column"] for x in COMPONENTS.values()]+[x["column"] for x in ROLE_META.values()]),
        "folds":N_WALK_FOLDS,"seed":RANDOM_SEED,"recency_half_life":RECENCY_HALF_LIFE_WEEKS
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()[:24]

MODEL_FINGERPRINT=_training_fingerprint()
MODEL_ARTIFACT_PATH=MODEL_ARTIFACT_DIR / f"nfl_props_v6_1_{TARGET_SEASON}_w{TARGET_WEEK}_{MODEL_FINGERPRINT}.joblib"
loaded_artifact=False
if USE_WEEKLY_MODEL_ARTIFACTS and MODEL_ARTIFACT_PATH.exists() and not FORCE_RETRAIN_WEEKLY_MODEL:
    try:
        art=joblib.load(MODEL_ARTIFACT_PATH)
        TEAM_MODELS=art["TEAM_MODELS"]; PLAYER_MODELS=art["PLAYER_MODELS"]; ROLE_MODELS=art["ROLE_MODELS"]
        metrics_df=art["metrics_df"]; role_metrics_df=art["role_metrics_df"]
        loaded_artifact=True
        print(f"[MODEL CACHE] loaded frozen V6.1 artifact: {MODEL_ARTIFACT_PATH.name}")
    except Exception as e:
        print("Frozen model artifact unreadable; retraining:",e)

if not loaded_artifact:
    # Team models
    for name,meta in TEAM_TARGETS.items():
        d=TS[TS["team_games_prior"]>=MIN_TEAM_GAMES].copy()
        result=fit_component(d,TEAM_FEATURES,meta["column"],meta["transform"],name)
        TEAM_MODELS[name]=result; all_metrics.append(result["metrics"])
        print(name,"rows",result["train_rows"],"weights",{k:round(v,3) for k,v in result["weights"].items()},"resid_scale",round(result["residual_scale"],3))

    # Player component models
    for name,meta in COMPONENTS.items():
        d=PS[(PS["position"].isin(meta["positions"]))&(PS["career_games_prior"]>=MIN_PLAYER_GAMES)].copy()
        d=d[pd.to_numeric(d[meta["min_opps_col"]],errors="coerce").fillna(0)>=meta["min_opps"]].copy()
        result=fit_component(d,PLAYER_FEATURES,meta["column"],meta["transform"],name)
        PLAYER_MODELS[name]=result; all_metrics.append(result["metrics"])
        print(name,"rows",result["train_rows"],"weights",{k:round(v,3) for k,v in result["weights"].items()},"resid_scale",round(result["residual_scale"],3))

    for name,meta in ROLE_META.items():
        d=PS[(PS["position"].isin(meta["positions"]))&(PS["career_games_prior"]>=MIN_PLAYER_GAMES)].copy()
        result=fit_role_classifier(d,PLAYER_FEATURES,meta["column"],name)
        ROLE_MODELS[name]=result; role_metrics.append(result["metrics"])
        print(name,"rows",result["train_rows"],"weights",{k:round(v,3) for k,v in result["weights"].items()},
              "population_rate",round(result["population_rate"],3),"calibration",result["calibration_info"])

    metrics_df=pd.concat(all_metrics,ignore_index=True)
    role_metrics_df=pd.concat(role_metrics,ignore_index=True) if role_metrics else pd.DataFrame()
    if USE_WEEKLY_MODEL_ARTIFACTS:
        joblib.dump({"schema":V61_MODEL_SCHEMA_VERSION,"fingerprint":MODEL_FINGERPRINT,
                     "TEAM_MODELS":TEAM_MODELS,"PLAYER_MODELS":PLAYER_MODELS,"ROLE_MODELS":ROLE_MODELS,
                     "metrics_df":metrics_df,"role_metrics_df":role_metrics_df},MODEL_ARTIFACT_PATH,compress=3)
        print(f"[MODEL CACHE] froze V6.1 artifact: {MODEL_ARTIFACT_PATH.name}")

display(metrics_df.groupby(["component","model"])[["MAE","RMSE"]].mean().reset_index().sort_values(["component","MAE"]))
if len(role_metrics_df): display(role_metrics_df.groupby(["component","model"])[["Brier","LogLoss"]].mean().reset_index().sort_values(["component","Brier"]))

# V6.1 audit tables for optimized ensemble weights and accepted/rejected hurdle calibration.
ensemble_audit_rows=[]
for family,models in [("TEAM",TEAM_MODELS),("PLAYER",PLAYER_MODELS)]:
    for comp,res in models.items():
        ensemble_audit_rows.append({"family":family,"component":comp,**{f"weight_{k}":v for k,v in res["weights"].items()},
                                    "residual_scale":res.get("residual_scale",1.0),"train_rows":res.get("train_rows",0)})
ensemble_audit_df=pd.DataFrame(ensemble_audit_rows)
role_calibration_audit_df=pd.DataFrame([
    {"component":comp,"train_rows":res.get("train_rows",0),**res.get("calibration_info",{}),
     **{f"weight_{k}":v for k,v in res["weights"].items()}}
    for comp,res in ROLE_MODELS.items()
])

# %% [markdown]
# ## 11. Freeze current component projections before Monte Carlo

# %%
def predict_component(result, d):
    preds={}
    for name,m in result["models"].items(): preds[name]=m.predict(d[result["features"]])
    ens_tx=weighted_mean_predictions(preds,result["weights"])
    ens_nat=inverse_target(ens_tx,result["transform"])
    indiv_nat={k:inverse_target(v,result["transform"]) for k,v in preds.items()}
    spread=np.ptp(np.vstack(list(indiv_nat.values())),axis=0) if indiv_nat else np.zeros(len(d))
    return ens_tx,ens_nat,indiv_nat,spread

def predict_role_probability(result,d):
    preds={name:np.clip(m.predict_proba(d[result["features"]])[:,1],1e-5,1-1e-5) for name,m in result["models"].items()}
    ens=weighted_mean_predictions(preds,result["weights"])
    if result.get("calibrator") is not None:
        ens=np.asarray(result["calibrator"].predict(np.clip(ens,1e-5,1-1e-5)),dtype=float)
    return np.clip(ens,HURDLE_MIN_PROB,HURDLE_MAX_PROB),preds

team_proj_rows=[]
for comp,result in TEAM_MODELS.items():
    tx,nat,indiv,spread=predict_component(result,current_team)
    for i,r in current_team.reset_index(drop=True).iterrows():
        row={"game_id":r["game_id"],"team":r["team"],"opponent":r["opponent"],"home":r["home"],"component":comp,"center_tx":float(tx[i]),"center":float(nat[i]),"model_spread":float(spread[i])}
        for m in ["RF","XGB","REG"]: row[m.lower()]=float(indiv[m][i]) if m in indiv else np.nan
        team_proj_rows.append(row)
team_component_proj=pd.DataFrame(team_proj_rows)

player_proj_rows=[]
for comp,meta in COMPONENTS.items():
    result=PLAYER_MODELS[comp]; elig=current["position"].isin(meta["positions"]); d=current[elig].copy().reset_index(drop=True)
    if len(d)==0: continue
    tx,nat,indiv,spread=predict_component(result,d)
    for i,r in d.iterrows():
        # Shrink stale/sparse player-specific projections toward the current positional population
        # later, after all rows exist. Keep raw centers here for transparency.
        row={"game_id":r["game_id"],"team":r["team"],"opponent":r["opponent"],"player_id_key":r["player_id_key"],"player_name":r["player_name"],"position":r["position"],"component":comp,"center_tx_raw":float(tx[i]),"center_raw":float(nat[i]),"model_spread":float(spread[i]),"history_reliability":float(r["history_reliability"]),"uncertainty_multiplier":float(r["uncertainty_multiplier"])}
        for m in ["RF","XGB","REG"]: row[m.lower()]=float(indiv[m][i]) if m in indiv else np.nan
        player_proj_rows.append(row)
player_component_proj=pd.DataFrame(player_proj_rows)

# Hierarchical shrinkage in transformed space. High-sample current starters barely move;
# stale/sparse backups move toward the league positional/component prior and get wider tails.
player_component_proj["population_center_tx"]=player_component_proj.groupby(["position","component"])["center_tx_raw"].transform("median")
player_component_proj["center_tx"]=(player_component_proj["history_reliability"]*player_component_proj["center_tx_raw"]+(1-player_component_proj["history_reliability"])*player_component_proj["population_center_tx"])
for comp,result in PLAYER_MODELS.items():
    mask=player_component_proj["component"].eq(comp)
    player_component_proj.loc[mask,"center"]=inverse_target(player_component_proj.loc[mask,"center_tx"].to_numpy(),result["transform"])

# Freeze opportunity-hurdle probabilities.
role_proj_rows=[]
for comp,meta in ROLE_META.items():
    result=ROLE_MODELS[comp]
    d=current[current["position"].isin(meta["positions"])].copy().reset_index(drop=True)
    if len(d)==0: continue
    p,indiv=predict_role_probability(result,d)
    pos_rate=PS.groupby("position")[meta["column"]].mean().to_dict()
    for i,r in d.iterrows():
        rel=float(r["history_reliability"]); prior=float(pos_rate.get(r["position"],result["population_rate"]))
        pp=float(np.clip(rel*p[i]+(1-rel)*prior,HURDLE_MIN_PROB,HURDLE_MAX_PROB))
        role_proj_rows.append({"game_id":r["game_id"],"team":r["team"],"opponent":r["opponent"],"player_id_key":r["player_id_key"],"player_name":r["player_name"],"position":r["position"],"component":comp,"probability":pp,"raw_probability":float(p[i]),"history_reliability":rel,"current_season_off_snaps":float(r["current_season_off_snaps"]),"depth_rank":float(r["depth_rank"]),"rf":float(indiv["RF"][i]) if "RF" in indiv else np.nan,"xgb":float(indiv["XGB"][i]) if "XGB" in indiv else np.nan,"log":float(indiv["LOG"][i]) if "LOG" in indiv else np.nan})
player_role_proj=pd.DataFrame(role_proj_rows)
display(player_role_proj.head(80))

# Build QB starter probabilities.
#
# V6 rule:
# - Explicit scenario / announced-starter overrides always win.
# - If the best available depth-chart QB is CLEAR, he is treated as the starter
#   (99.5%) rather than normalizing a generic depth prior across every rostered QB.
# - Material backup probability exists only when there is actual availability
#   uncertainty around the top available QB.
# - If QB1 is OUT and therefore absent from the eligible pool, the next healthy
#   depth-chart QB becomes the clear leader automatically.
QB_START_PROBS={}
QB_SCENARIO_REASON={}
qb_ctx=current[current["position"].eq("QB")].copy()

for team,g in qb_ctx.groupby("team"):
    g=g.sort_values(["depth_rank","player_name"]).copy().reset_index(drop=True)

    if team in QB_START_PROB_OVERRIDES:
        raw={str(k):float(v) for k,v in QB_START_PROB_OVERRIDES[team].items()}
        probs=np.array([raw.get(n,0.0) for n in g["player_name"]],float)
        reason="EXPLICIT_PROB_OVERRIDE"

    elif team in QB_STARTER_OVERRIDES and QB_STARTER_OVERRIDES[team] in set(g["player_name"]):
        probs=np.array([1.0 if n==QB_STARTER_OVERRIDES[team] else 0.0 for n in g["player_name"]],float)
        reason="ANNOUNCED_STARTER_OVERRIDE"

    else:
        probs=np.zeros(len(g),dtype=float)
        if len(g)==0:
            continue

        # The first row is the best currently eligible depth-chart candidate.
        leader=g.iloc[0]
        leader_avail=float(np.clip(leader["availability_probability"],0,1))
        leader_flag=str(leader.get("availability_flag","CLEAR"))

        if len(g)==1:
            probs[0]=1.0
            reason="ONLY_ELIGIBLE_QB"

        elif leader_avail >= 0.90 and leader_flag=="CLEAR":
            # Healthy/clear starter: allow only tiny pregame contingency mass.
            leader_prob=0.995
            probs[0]=leader_prob
            backups=g.iloc[1:]
            depth=np.exp(-0.80*(backups["depth_rank"].to_numpy(float)-float(backups["depth_rank"].min())))
            readiness=backups["availability_probability"].to_numpy(float)*(0.45+0.55*backups["preparation_score"].to_numpy(float))
            w=np.maximum(depth*readiness,0.0)
            if w.sum()<=0: w=np.ones(len(backups))
            probs[1:]=(1.0-leader_prob)*(w/w.sum())
            reason="CLEAR_DEPTH_LEADER"

        else:
            # Questionable/doubtful top option: availability drives the scenario,
            # while the remaining probability is allocated among actual backups.
            leader_prob=float(np.clip(leader_avail,0.01,0.95))
            probs[0]=leader_prob
            backups=g.iloc[1:]
            depth=np.exp(-0.80*(backups["depth_rank"].to_numpy(float)-float(backups["depth_rank"].min())))
            readiness=backups["availability_probability"].to_numpy(float)*(0.45+0.55*backups["preparation_score"].to_numpy(float))
            w=np.maximum(depth*readiness,0.0)
            if w.sum()<=0: w=np.ones(len(backups))
            probs[1:]=(1.0-leader_prob)*(w/w.sum())
            reason="INJURY_AVAILABILITY_SCENARIO"

    if probs.sum()<=0:
        probs=np.ones(len(g))/max(len(g),1)
        reason="FALLBACK_EQUAL"
    else:
        probs=probs/probs.sum()

    QB_START_PROBS[team]={n:float(p) for n,p in zip(g["player_name"],probs)}
    QB_SCENARIO_REASON[team]=reason

primary_qb={team:max(d,key=d.get) for team,d in QB_START_PROBS.items() if d}
current["starter_probability"]=current.apply(
    lambda r: QB_START_PROBS.get(r["team"],{}).get(r["player_name"],0.0) if r["position"]=="QB" else np.nan,
    axis=1
)
current["qb_scenario_reason"]=current.apply(
    lambda r: QB_SCENARIO_REASON.get(r["team"],"") if r["position"]=="QB" else "",
    axis=1
)

print("QB starter scenarios:")
for t,p in QB_START_PROBS.items():
    print(t, QB_SCENARIO_REASON.get(t), {k:round(v,4) for k,v in p.items()})
display(team_component_proj.head(20)); display(player_component_proj.head(50))

# %% [markdown]
# ## 12. Tail calibration from historical player outcomes

# %%
FINAL_STATS={
    "pass_attempts":({"QB"},"attempts"),
    "pass_completions":({"QB"},"completions"),
    "passing_yards":({"QB"},"passing_yards"),
    "receptions":({"RB","WR","TE"},"receptions"),
    "receiving_yards":({"RB","WR","TE"},"receiving_yards"),
    "rush_attempts":({"QB","RB","WR","TE"},"carries"),
    "rushing_yards":({"QB","RB","WR","TE"},"rushing_yards"),
    "rec_rush_yards":({"RB","WR","TE"},"scrimmage_yards"),
}

def robust_residual_pool(result):
    r=np.asarray(result.get("residual_tx",[]),dtype=float)
    r=r[np.isfinite(r)]
    if len(r)==0:
        return np.array([0.0]), {"n":0,"median":0.0,"robust_sigma":0.0,"lo":0.0,"hi":0.0}
    med=float(np.median(r))
    r=r-med
    lo=float(np.quantile(r,ROBUST_RESIDUAL_LO_Q))
    hi=float(np.quantile(r,ROBUST_RESIDUAL_HI_Q))
    rw=np.clip(r,lo,hi)
    mad=float(np.median(np.abs(rw-np.median(rw))))
    rsig=float(1.4826*mad) if mad>1e-12 else float(np.std(rw))
    if rsig>1e-12:
        rw=np.clip(rw,-ROBUST_RESIDUAL_MAX_SIGMA*rsig,ROBUST_RESIDUAL_MAX_SIGMA*rsig)
    learned_scale=float(result.get("residual_scale",1.0))
    rw=rw*COMPONENT_RESIDUAL_SCALE*learned_scale
    return rw, {"n":len(r),"median":med,"robust_sigma":rsig,"lo":lo,"hi":hi,"learned_scale":learned_scale}

RESIDUAL_POOLS={}
RESIDUAL_DIAGNOSTICS=[]
for name,result in {**TEAM_MODELS,**PLAYER_MODELS}.items():
    pool,diag=robust_residual_pool(result)
    RESIDUAL_POOLS[name]=pool
    RESIDUAL_DIAGNOSTICS.append({"component":name,**diag,"pool_sd":float(np.std(pool))})

residual_diag_df=pd.DataFrame(RESIDUAL_DIAGNOSTICS)
display(residual_diag_df.sort_values("component"))

def _opportunity_mask(stat,pos):
    d=PS[PS["position"].eq(pos)].copy()
    if stat in {"passing_yards","pass_attempts","pass_completions"}:
        return pd.to_numeric(d["attempts"],errors="coerce").fillna(0)>=5
    if stat in {"receiving_yards","receptions"}:
        return pd.to_numeric(d["targets"],errors="coerce").fillna(0)>=1
    if stat in {"rushing_yards","rush_attempts"}:
        return pd.to_numeric(d["carries"],errors="coerce").fillna(0)>=1
    if stat=="rec_rush_yards":
        return (pd.to_numeric(d["targets"],errors="coerce").fillna(0)+pd.to_numeric(d["carries"],errors="coerce").fillna(0))>=1
    return pd.Series(True,index=d.index)

TAIL_GUARDS={}
for stat,(positions,col) in FINAL_STATS.items():
    for pos in positions:
        d=PS[PS["position"].eq(pos)].copy()
        if not len(d) or col not in d.columns: continue
        mask=_opportunity_mask(stat,pos)
        vals=pd.to_numeric(d.loc[mask,col],errors="coerce").dropna()
        if len(vals)>=80:
            soft_lo=float(vals.quantile(SOFT_TAIL_LO_Q))
            soft_hi=float(vals.quantile(SOFT_TAIL_HI_Q))
            hard_lo=float(vals.quantile(HARD_TAIL_LO_Q))
            hard_hi=float(vals.quantile(HARD_TAIL_HI_Q))
            zero_is_valid = stat in {"pass_attempts","pass_completions","passing_yards","receptions","receiving_yards","rush_attempts","rec_rush_yards"}
            if zero_is_valid:
                soft_lo=0.0; hard_lo=0.0
            elif stat=="rushing_yards":
                soft_lo=min(0.0,soft_lo); hard_lo=min(0.0,hard_lo)
            hard_hi=max(hard_hi,soft_hi)
            TAIL_GUARDS[(stat,pos)]={"soft_lo":soft_lo,"soft_hi":soft_hi,"hard_lo":hard_lo,"hard_hi":hard_hi,"n":int(len(vals)),"zero_is_valid":zero_is_valid}

print("Tail guards built:",len(TAIL_GUARDS))

# %% [markdown]
# ## 13. Generative Monte Carlo

# %%
team_lookup={(r.team,r.component):r for r in team_component_proj.itertuples(index=False)}
player_lookup={(str(r.player_id_key),r.component):r for r in player_component_proj.itertuples(index=False)}
current_by_id=current.drop_duplicates("player_id_key").set_index("player_id_key",drop=False)
role_lookup={(str(r.player_id_key),r.component):r for r in player_role_proj.itertuples(index=False)}

ACTIVE_BANK={}; STARTER_BANK={}; OFFER_CONDITION_BANK={}
SIM_BANK={}; DRIVER_ROWS=[]; TEAM_SIM={}; TEAM_PASS_ENV={}
COMPOSITION_QA=[]; QB_RECEIVER_COUPLING_QA=[]

def draw_component(center_tx, result, component_name, n=N_SIMS, clip=None, uncertainty_multiplier=1.0):
    pool=np.asarray(RESIDUAL_POOLS.get(component_name,result.get("residual_tx",[])),dtype=float)
    pool=pool[np.isfinite(pool)]
    if len(pool)>=25:
        e=rng.choice(pool,size=n,replace=True)
    else:
        sd=float(np.std(pool)) if len(pool)>1 else 0.08
        e=rng.normal(0,sd,size=n)
    e=e*float(uncertainty_multiplier)
    x=inverse_target(float(center_tx)+e,result["transform"])
    if clip is not None:
        x=np.clip(x,clip[0],clip[1])
    return np.asarray(x,dtype=float)

def draw_share_noise(component_name, n, uncertainty_multiplier=1.0):
    pool=np.asarray(RESIDUAL_POOLS.get(component_name,[0.0]),dtype=float)
    pool=pool[np.isfinite(pool)]
    if len(pool)>=25:
        return rng.choice(pool,size=n,replace=True)*float(uncertainty_multiplier)
    return rng.normal(0,float(np.std(pool)) if len(pool)>1 else 0.08,size=n)*float(uncertainty_multiplier)

def allocate_counts(total, composition):
    """
    Sequential binomial allocation with hard numerical guards.

    V6:
    - replaces NaN/inf/negative composition weights before normalization;
    - repairs zero-sum rows by assigning the entire row to the final "other" bucket;
    - clips/repairs sequential conditional probabilities before rng.binomial();
    - guarantees no negative remaining counts.
    """
    total=np.asarray(total,dtype=float)
    total=np.nan_to_num(total,nan=0.0,posinf=0.0,neginf=0.0)
    total=np.maximum(np.rint(total),0).astype(int)

    W=np.asarray(composition,dtype=float)
    if W.ndim != 2:
        raise ValueError(f"composition must be 2D, got shape={W.shape}")

    # Sanitize every weight before normalization.
    W=np.nan_to_num(W,nan=0.0,posinf=0.0,neginf=0.0)
    W=np.maximum(W,0.0)

    row_sum=W.sum(axis=1,keepdims=True)
    bad=(~np.isfinite(row_sum[:,0])) | (row_sum[:,0] <= 1e-12)
    if np.any(bad):
        W[bad,:]=0.0
        W[bad,-1]=1.0
        row_sum=W.sum(axis=1,keepdims=True)

    W=W/np.maximum(row_sum,1e-12)

    # Final belt-and-suspenders normalization.
    W=np.nan_to_num(W,nan=0.0,posinf=0.0,neginf=0.0)
    row_sum=W.sum(axis=1,keepdims=True)
    bad=(row_sum[:,0] <= 1e-12)
    if np.any(bad):
        W[bad,:]=0.0
        W[bad,-1]=1.0
        row_sum=W.sum(axis=1,keepdims=True)
    W=W/row_sum

    n,k=W.shape
    out=np.zeros((n,k),dtype=int)
    rem=total.copy()
    remw=np.ones(n,dtype=float)

    for j in range(k-1):
        denom=np.maximum(remw,1e-12)
        p=W[:,j]/denom
        p=np.nan_to_num(p,nan=0.0,posinf=1.0,neginf=0.0)
        p=np.clip(p,0.0,1.0)

        # If no counts remain, force p=0 to avoid meaningless draws.
        p=np.where(rem>0,p,0.0)

        draw=rng.binomial(rem,p)
        out[:,j]=draw
        rem=np.maximum(rem-draw,0)
        remw=np.maximum(remw-W[:,j],0.0)

    out[:,-1]=rem
    return out

def reconcile_counts_to_capacity(count_matrix, capacity, rng):
    """Condition integer player counts on a team-level capacity without inventing events.

    Each row is treated as a population of modeled successes. If the raw player
    total exceeds the team capacity, a multivariate-hypergeometric equivalent
    selects which successes remain. If capacity is at least the raw total, the
    row is returned unchanged.
    """
    raw=np.asarray(count_matrix,dtype=int)
    if raw.ndim!=2:
        raise ValueError(f"count_matrix must be 2D, got {raw.shape}")
    raw=np.maximum(raw,0)
    cap=np.asarray(capacity,dtype=float)
    cap=np.nan_to_num(cap,nan=0.0,posinf=0.0,neginf=0.0)
    cap=np.maximum(np.rint(cap),0).astype(int)
    row_total=raw.sum(axis=1)
    cap=np.minimum(cap,row_total)

    n,k=raw.shape
    out=np.zeros_like(raw,dtype=int)
    rem_total=row_total.copy()
    rem_cap=cap.copy()

    for j in range(k-1):
        ngood=raw[:,j]
        nbad=np.maximum(rem_total-ngood,0)
        nsample=np.minimum(rem_cap,ngood+nbad)
        draw=rng.hypergeometric(ngood,nbad,nsample)
        out[:,j]=draw
        rem_cap=np.maximum(rem_cap-draw,0)
        rem_total=np.maximum(rem_total-ngood,0)

    if k:
        out[:,-1]=np.minimum(raw[:,-1],rem_cap)

    if np.any(out>raw):
        raise RuntimeError("Completion reconciliation invented player events.")
    if np.any(out.sum(axis=1)>cap):
        raise RuntimeError("Completion reconciliation exceeded team capacity.")
    return out


def logistic_normal_composition(base_centers, active, replacement_propensity, component_name,
                                uncertainty_multipliers, other_center):
    """
    V6 compositional usage:
      1) remove inactive players,
      2) reassign their baseline vacancy to active replacements,
      3) apply robust transformed-space noise,
      4) normalize player + other bucket together.
    Shares are born summing to 1; there is no post-hoc repair.
    """
    base=np.asarray(base_centers,dtype=float)
    base=np.nan_to_num(base,nan=1e-6,posinf=1.0,neginf=1e-6)
    base=np.maximum(base,1e-6)

    A=np.asarray(active,bool)

    rp=np.asarray(replacement_propensity,dtype=float)
    rp=np.nan_to_num(rp,nan=1e-6,posinf=1.0,neginf=1e-6)
    rp=np.maximum(rp,1e-6)

    um=np.asarray(uncertainty_multipliers,dtype=float)
    um=np.nan_to_num(um,nan=1.0,posinf=MAX_CONTEXT_UNCERTAINTY_MULT,neginf=1.0)
    um=np.clip(um,1.0,MAX_CONTEXT_UNCERTAINTY_MULT)

    n,k=A.shape

    B=np.tile(base,(n,1))
    active_base=B*A
    vacancy=(B*(~A)).sum(axis=1)
    rp_mat=rp[None,:]*A
    rp_denom=rp_mat.sum(axis=1,keepdims=True)
    rp_alloc=np.divide(rp_mat,rp_denom,out=np.zeros_like(rp_mat),where=rp_denom>1e-12)
    adjusted=active_base + vacancy[:,None]*rp_alloc

    # Logistic-normal perturbation. For small shares, logit residuals behave close to
    # log-share residuals; robust clipping prevents one noisy model from owning a slate.
    logw=np.log(np.maximum(adjusted,1e-10))
    for j in range(k):
        eps=draw_share_noise(component_name,n,um[j])
        logw[:,j]+=eps

    # Inactive players remain exactly zero regardless of noise.
    maxlog=np.max(np.where(A,logw,-1e30),axis=1,keepdims=True)
    pw=np.where(A,np.exp(logw-maxlog),0.0)

    # Preserve a real "other" bucket. Scale it into the same stabilized exponential frame.
    other=max(float(other_center),1e-5)
    other_w=np.full((n,1),other)*np.exp(-maxlog)

    full=np.column_stack([pw,other_w])
    full=np.nan_to_num(full,nan=0.0,posinf=0.0,neginf=0.0)
    full=np.maximum(full,0.0)

    row_sum=full.sum(axis=1,keepdims=True)
    bad=(~np.isfinite(row_sum[:,0])) | (row_sum[:,0] <= 1e-12)
    if np.any(bad):
        full[bad,:]=0.0
        full[bad,-1]=1.0
        row_sum=full.sum(axis=1,keepdims=True)

    full=full/np.maximum(row_sum,1e-12)

    # QA should never see non-finite composition rows.
    if not np.isfinite(full).all():
        raise RuntimeError("Non-finite values remained in logistic_normal_composition after sanitization.")
    return full

def soft_tail_guard(arr,stat,pos):
    a=np.asarray(arr,dtype=float).copy()
    g=TAIL_GUARDS.get((stat,pos))
    if g is None:
        return a
    slo,shi,hlo,hhi=g["soft_lo"],g["soft_hi"],g["hard_lo"],g["hard_hi"]

    hi=a>shi
    if np.any(hi):
        a[hi]=shi+(a[hi]-shi)*TAIL_SHRINK_SLOPE
    if slo < 0:
        lo=a<slo
        if np.any(lo): a[lo]=slo+(a[lo]-slo)*TAIL_SHRINK_SLOPE
    return np.clip(a,hlo,hhi)

for team,tg in current_team.groupby("team"):
    tr=tg.iloc[0]
    pass_volume_row=team_lookup[(team,"team_pass_attempts")]
    rush_volume_row=team_lookup[(team,"team_rush_attempts")]
    target_rate_row=team_lookup[(team,"team_target_rate")]

    pass_att=np.rint(draw_component(pass_volume_row.center_tx,TEAM_MODELS["team_pass_attempts"],"team_pass_attempts",clip=(12,65))).astype(int)
    rush_att=np.rint(draw_component(rush_volume_row.center_tx,TEAM_MODELS["team_rush_attempts"],"team_rush_attempts",clip=(10,55))).astype(int)
    target_rate=draw_component(target_rate_row.center_tx,TEAM_MODELS["team_target_rate"],"team_target_rate",clip=(0.60,0.995))
    team_targets=rng.binomial(pass_att,target_rate)
    TEAM_SIM[team]={"pass_attempts":pass_att,"rush_attempts":rush_att,"targets":team_targets,"target_rate":target_rate}

    team_players=current[current["team"].eq(team)].copy().reset_index(drop=True)

    for r in team_players.itertuples():
        pid=str(r.player_id_key)
        p=float(np.clip(r.availability_probability,0,1))
        ACTIVE_BANK[pid]=(rng.random(N_SIMS)<p) if USE_INJURY_SCENARIOS else np.full(N_SIMS,p>0.5,dtype=bool)

    # ---------------- QB starter scenario ----------------
    # V6.1 also creates a shared QB passing environment that will later shift
    # receiver catch rate and YPR in the same Monte Carlo draw.
    qbs=team_players[team_players["position"].eq("QB")].copy().reset_index(drop=True)
    if len(qbs):
        pmap=QB_START_PROBS.get(team,{})
        probs=np.array([pmap.get(n,0.0) for n in qbs["player_name"]],float)
        probs=probs/probs.sum() if probs.sum()>0 else np.ones(len(qbs))/len(qbs)
        starter_idx=rng.choice(np.arange(len(qbs)),size=N_SIMS,p=probs)

        qb_comp_centers=[]
        qb_ypc_centers=[]
        for r in qbs.itertuples():
            pid=str(r.player_id_key)
            cr=player_lookup.get((pid,"qb_completion_rate"))
            py=player_lookup.get((pid,"qb_pass_ypc"))
            qb_comp_centers.append(float(cr.center) if cr is not None and np.isfinite(cr.center) else np.nan)
            qb_ypc_centers.append(float(py.center) if py is not None and np.isfinite(py.center) else np.nan)

        comp_vals=np.asarray(qb_comp_centers,float)
        ypc_vals=np.asarray(qb_ypc_centers,float)
        comp_ok=np.isfinite(comp_vals)
        ypc_ok=np.isfinite(ypc_vals)
        baseline_comp=float(np.sum(probs[comp_ok]*comp_vals[comp_ok])/np.sum(probs[comp_ok])) if np.any(comp_ok) and np.sum(probs[comp_ok])>0 else 0.64
        baseline_ypc=float(np.sum(probs[ypc_ok]*ypc_vals[ypc_ok])/np.sum(probs[ypc_ok])) if np.any(ypc_ok) and np.sum(probs[ypc_ok])>0 else 11.0
        baseline_comp=float(np.clip(baseline_comp,0.40,0.82))
        baseline_ypc=float(np.clip(baseline_ypc,6.0,18.0))

        selected_comp_rate=np.full(N_SIMS,baseline_comp,dtype=float)
        selected_pass_ypc=np.full(N_SIMS,baseline_ypc,dtype=float)
        selected_completions=np.zeros(N_SIMS,dtype=int)
        selected_pass_yards=np.zeros(N_SIMS,dtype=float)

        for j,r in enumerate(qbs.itertuples()):
            pid=str(r.player_id_key)
            starts=(starter_idx==j)
            STARTER_BANK[pid]=starts
            OFFER_CONDITION_BANK[pid]=starts

            attempts=np.where(starts,rng.binomial(pass_att,0.985),0)
            cr=player_lookup.get((pid,"qb_completion_rate"))
            py=player_lookup.get((pid,"qb_pass_ypc"))
            if cr is None or py is None:
                continue

            um=float(r.uncertainty_multiplier)
            comp_rate=draw_component(cr.center_tx,PLAYER_MODELS["qb_completion_rate"],"qb_completion_rate",
                                     clip=(0.35,0.85),uncertainty_multiplier=um)
            # A completion cannot exceed the team's simulated targeted passes.
            completions=np.minimum(rng.binomial(attempts,comp_rate),team_targets)
            ypc=draw_component(py.center_tx,PLAYER_MODELS["qb_pass_ypc"],"qb_pass_ypc",
                               clip=(5.0,22.0),uncertainty_multiplier=um)

            # Game-level efficiency already carries component uncertainty.
            # Add only a small possession-level term; do not stack another full event distribution.
            micro=rng.normal(0,np.sqrt(np.maximum(completions,1))*PASS_MICRO_SD_PER_COMPLETION,size=N_SIMS)
            pass_yards=np.where(completions>0,np.maximum(0,completions*ypc+micro),0.0)

            # Shared latent environment used by all eligible receivers on this team.
            selected_comp_rate[starts]=comp_rate[starts]
            selected_pass_ypc[starts]=ypc[starts]
            selected_completions[starts]=completions[starts]
            selected_pass_yards[starts]=pass_yards[starts]

            SIM_BANK[(pid,"pass_attempts")]=soft_tail_guard(attempts,"pass_attempts","QB")
            SIM_BANK[(pid,"pass_completions")]=soft_tail_guard(completions,"pass_completions","QB")
            SIM_BANK[(pid,"passing_yards")]=soft_tail_guard(pass_yards,"passing_yards","QB")

            DRIVER_ROWS.append({
                "player_id_key":pid,"player_name":r.player_name,"team":team,"driver":"QB",
                "starter_probability":float(probs[j]),
                "availability_probability":float(r.availability_probability),
                "preparation_score":float(r.preparation_score),
                "history_reliability":float(r.history_reliability),
                "uncertainty_multiplier":um,
                "volume_center":float(pass_volume_row.center),"rate_center":float(cr.center),"eff_center":float(py.center)
            })

        if USE_QB_RECEIVER_COUPLING:
            comp_ratio=np.clip(selected_comp_rate/max(baseline_comp,1e-6),QB_ENV_RATIO_MIN,QB_ENV_RATIO_MAX)
            ypc_ratio=np.clip(selected_pass_ypc/max(baseline_ypc,1e-6),QB_ENV_RATIO_MIN,QB_ENV_RATIO_MAX)
            catch_mult=np.exp(QB_RECEIVER_CATCH_COUPLING*np.log(np.maximum(comp_ratio,1e-6)))
            ypr_mult=np.exp(QB_RECEIVER_YPR_COUPLING*np.log(np.maximum(ypc_ratio,1e-6)))
        else:
            catch_mult=np.ones(N_SIMS,dtype=float)
            ypr_mult=np.ones(N_SIMS,dtype=float)

        TEAM_PASS_ENV[team]={
            "baseline_completion_rate":baseline_comp,
            "baseline_pass_ypc":baseline_ypc,
            "selected_completion_rate":selected_comp_rate,
            "selected_pass_ypc":selected_pass_ypc,
            "selected_completions":selected_completions,
            "selected_pass_yards":selected_pass_yards,
            "receiver_catch_multiplier":catch_mult,
            "receiver_ypr_multiplier":ypr_mult,
        }
    else:
        TEAM_PASS_ENV[team]={
            "baseline_completion_rate":0.64,
            "baseline_pass_ypc":11.0,
            "selected_completion_rate":np.full(N_SIMS,0.64),
            "selected_pass_ypc":np.full(N_SIMS,11.0),
            "selected_completions":np.zeros(N_SIMS,dtype=int),
            "selected_pass_yards":np.zeros(N_SIMS,dtype=float),
            "receiver_catch_multiplier":np.ones(N_SIMS),
            "receiver_ypr_multiplier":np.ones(N_SIMS),
        }

    # ---------------- Receiving composition ----------------
    recs=team_players[team_players["position"].isin(["RB","WR","TE"])].copy().reset_index(drop=True)
    if len(recs):
        base=[]; active=[]; rp=[]; ums=[]
        for r in recs.itertuples():
            pid=str(r.player_id_key)
            pc=player_lookup.get((pid,"target_share"))
            center=float(pc.center if pc is not None else 0.01)
            base.append(max(center*float(r.manual_role_factor),1e-5))
            target_role_row=role_lookup.get((pid,"target_active_prob")); role_p=float(target_role_row.probability) if target_role_row is not None else 0.05
            role_event=(rng.random(N_SIMS)<role_p)
            active.append(ACTIVE_BANK[pid] & role_event)
            rp.append(float(r.replacement_weight)); ums.append(float(r.uncertainty_multiplier))

        center_sum=float(np.sum(base))
        other_center=max(0.05,1.0-min(center_sum,0.95))
        comp=logistic_normal_composition(
            base,np.column_stack(active),np.array(rp),"target_share",np.array(ums),other_center
        )
        if not np.isfinite(comp).all():
            raise RuntimeError(f"{team} target composition contains non-finite values.")
        COMPOSITION_QA.append({
            "team":team,"type":"targets",
            "max_sum_error":float(np.max(np.abs(comp.sum(axis=1)-1.0))),
            "min_weight":float(np.min(comp)),
            "max_weight":float(np.max(comp))
        })
        alloc=allocate_counts(team_targets,comp)

        # First generate player-level catch intent from targets and catch skill.
        # V6.1.1 then conditions those catches on the selected QB's simulated
        # completion count so player props and QB props obey the same game state.
        rec_temp=[]
        for j,r in enumerate(recs.itertuples()):
            pid=str(r.player_id_key)
            targets=alloc[:,j]
            ca=player_lookup.get((pid,"catch_rate"))
            ry=player_lookup.get((pid,"rec_ypr"))
            OFFER_CONDITION_BANK[pid]=ACTIVE_BANK[pid]
            if ca is None or ry is None:
                continue

            um=float(r.uncertainty_multiplier)
            catch=draw_component(ca.center_tx,PLAYER_MODELS["catch_rate"],"catch_rate",
                                 clip=(0.25,0.98),uncertainty_multiplier=um)
            ypr=draw_component(ry.center_tx,PLAYER_MODELS["rec_ypr"],"rec_ypr",
                               clip=(3.0,30.0),uncertainty_multiplier=um)

            # Shared QB environment moves player skill without replacing it.
            pass_env=TEAM_PASS_ENV.get(team,{})
            catch_mult=np.asarray(pass_env.get("receiver_catch_multiplier",np.ones(N_SIMS)),dtype=float)
            ypr_mult=np.asarray(pass_env.get("receiver_ypr_multiplier",np.ones(N_SIMS)),dtype=float)
            catch=np.clip(catch*catch_mult,0.05,0.995)
            ypr=np.clip(ypr*ypr_mult,2.0,35.0)

            raw_receptions=rng.binomial(targets,catch)
            rec_temp.append({
                "pid":pid,"r":r,"targets":targets,
                "raw_receptions":raw_receptions,"ypr":ypr,
                "um":um,"ca":ca,"ry":ry
            })

        if rec_temp:
            raw_rec_matrix=np.column_stack([x["raw_receptions"] for x in rec_temp]).astype(int)
            pass_env=TEAM_PASS_ENV.get(team,{})
            qb_comp=np.asarray(pass_env.get("selected_completions",raw_rec_matrix.sum(axis=1)),dtype=float)

            # If there is no modeled QB environment, do not zero an otherwise valid
            # receiver slate; otherwise the QB completion total is a hard capacity.
            if len(qbs):
                rec_matrix=reconcile_counts_to_capacity(raw_rec_matrix,qb_comp,rng)
            else:
                rec_matrix=raw_rec_matrix

            raw_yard_bank={}
            for j,item in enumerate(rec_temp):
                receptions=rec_matrix[:,j]
                ypr=item["ypr"]
                micro=rng.normal(
                    0,np.sqrt(np.maximum(receptions,1))*REC_MICRO_SD_PER_RECEPTION,
                    size=N_SIMS
                )
                raw_yard_bank[item["pid"]]=np.where(
                    receptions>0,
                    np.maximum(0,receptions*ypr+micro),
                    0.0
                )

            # Passing yards are the team-level yardage budget. Listed players may
            # consume at most that budget; any unused yards belong to the existing
            # unlisted/"other" receiving bucket.
            total_listed_raw=np.zeros(N_SIMS,dtype=float)
            for arr in raw_yard_bank.values():
                total_listed_raw += arr

            qb_py=np.asarray(
                pass_env.get("selected_pass_yards",total_listed_raw),
                dtype=float
            )
            yard_scale=np.ones(N_SIMS,dtype=float)
            if len(qbs):
                over=total_listed_raw>np.maximum(qb_py,0.0)
                yard_scale[over]=np.divide(
                    np.maximum(qb_py[over],0.0),
                    np.maximum(total_listed_raw[over],1e-12)
                )
                yard_scale=np.clip(yard_scale,0.0,1.0)

            for j,item in enumerate(rec_temp):
                pid=item["pid"]; r=item["r"]; receptions=rec_matrix[:,j]
                rec_yards=raw_yard_bank[pid]*yard_scale

                SIM_BANK[(pid,"receptions")]=soft_tail_guard(receptions,"receptions",r.position)
                SIM_BANK[(pid,"receiving_yards")]=soft_tail_guard(rec_yards,"receiving_yards",r.position)

                ca=item["ca"]; ry=item["ry"]; um=item["um"]
                DRIVER_ROWS.append({
                    "player_id_key":pid,"player_name":r.player_name,"team":team,"driver":"REC",
                    "availability_probability":float(r.availability_probability),
                    "depth_rank":float(r.depth_rank),"replacement_weight":float(r.replacement_weight),
                    "preparation_score":float(r.preparation_score),"history_reliability":float(r.history_reliability),
                    "uncertainty_multiplier":um,
                    "volume_center":float(target_rate_row.center*pass_volume_row.center),
                    "share_center":float(player_lookup[(pid,"target_share")].center),
                    "role_probability":float(role_lookup[(pid,"target_active_prob")].probability) if (pid,"target_active_prob") in role_lookup else np.nan,
                    "rate_center":float(ca.center),"eff_center":float(ry.center),
                    "qb_catch_coupling":float(QB_RECEIVER_CATCH_COUPLING if USE_QB_RECEIVER_COUPLING else 0.0),
                    "qb_ypr_coupling":float(QB_RECEIVER_YPR_COUPLING if USE_QB_RECEIVER_COUPLING else 0.0),
                    "qb_baseline_completion_rate":float(TEAM_PASS_ENV.get(team,{}).get("baseline_completion_rate",np.nan)),
                    "qb_baseline_pass_ypc":float(TEAM_PASS_ENV.get(team,{}).get("baseline_pass_ypc",np.nan))
                })

        # V6.1.1 coupling/accounting audit. This is diagnostic, not a hard requirement for every
        # team because low-volume / zero-inflated receiving groups can have weak correlation.
        listed_rec_yards=np.zeros(N_SIMS,dtype=float)
        listed_receptions=np.zeros(N_SIMS,dtype=float)
        for r in recs.itertuples():
            pid=str(r.player_id_key)
            if (pid,"receiving_yards") in SIM_BANK:
                listed_rec_yards += np.asarray(SIM_BANK[(pid,"receiving_yards")],dtype=float)
            if (pid,"receptions") in SIM_BANK:
                listed_receptions += np.asarray(SIM_BANK[(pid,"receptions")],dtype=float)

        pass_env=TEAM_PASS_ENV.get(team,{})
        qb_py=np.asarray(pass_env.get("selected_pass_yards",np.zeros(N_SIMS)),dtype=float)
        qb_comp=np.asarray(pass_env.get("selected_completions",np.zeros(N_SIMS)),dtype=float)
        valid=(np.isfinite(qb_py) & np.isfinite(listed_rec_yards))
        corr=np.nan
        if valid.sum()>10 and np.std(qb_py[valid])>1e-9 and np.std(listed_rec_yards[valid])>1e-9:
            corr=float(np.corrcoef(qb_py[valid],listed_rec_yards[valid])[0,1])
        completion_ratio=float(np.mean(listed_receptions/np.maximum(qb_comp,1.0)))
        yard_ratio=float(np.mean(listed_rec_yards/np.maximum(qb_py,1.0)))
        max_completion_overrun=float(np.max(listed_receptions-qb_comp))
        max_yard_overrun=float(np.max(listed_rec_yards-qb_py))
        QB_RECEIVER_COUPLING_QA.append({
            "team":team,
            "qb_receiving_yards_corr":corr,
            "mean_listed_receptions_to_qb_completions":completion_ratio,
            "mean_listed_receiving_yards_to_qb_passing_yards":yard_ratio,
            "max_listed_receptions_over_qb_completions":max_completion_overrun,
            "max_listed_receiving_yards_over_qb_passing_yards":max_yard_overrun,
            "mean_catch_multiplier":float(np.mean(pass_env.get("receiver_catch_multiplier",np.ones(N_SIMS)))),
            "sd_catch_multiplier":float(np.std(pass_env.get("receiver_catch_multiplier",np.ones(N_SIMS)))),
            "mean_ypr_multiplier":float(np.mean(pass_env.get("receiver_ypr_multiplier",np.ones(N_SIMS)))),
            "sd_ypr_multiplier":float(np.std(pass_env.get("receiver_ypr_multiplier",np.ones(N_SIMS)))),
            "coupling_enabled":bool(USE_QB_RECEIVER_COUPLING),
        })

    # ---------------- Rushing composition ----------------
    rushers=team_players[team_players["position"].isin(["QB","RB","WR","TE"])].copy().reset_index(drop=True)
    if len(rushers):
        base=[]; active=[]; rp=[]; ums=[]
        for r in rushers.itertuples():
            pid=str(r.player_id_key)
            pc=player_lookup.get((pid,"carry_share"))
            center=float(pc.center if pc is not None else 0.003)
            base.append(max(center*float(r.manual_role_factor),1e-5))
            carry_role_row=role_lookup.get((pid,"carry_active_prob")); role_p=float(carry_role_row.probability) if carry_role_row is not None else (0.90 if r.position=="RB" else 0.05)
            role_event=(rng.random(N_SIMS)<role_p); base_active=ACTIVE_BANK[pid]
            if r.position=="QB": base_active=base_active & STARTER_BANK.get(pid,np.zeros(N_SIMS,dtype=bool))
            active.append(base_active & role_event)
            rp.append(float(r.replacement_weight)*(1.25 if r.position=="RB" else 0.65)); ums.append(float(r.uncertainty_multiplier))

        center_sum=float(np.sum(base))
        other_center=max(0.03,1.0-min(center_sum,0.97))
        comp=logistic_normal_composition(
            base,np.column_stack(active),np.array(rp),"carry_share",np.array(ums),other_center
        )
        if not np.isfinite(comp).all():
            raise RuntimeError(f"{team} carry composition contains non-finite values.")
        COMPOSITION_QA.append({
            "team":team,"type":"carries",
            "max_sum_error":float(np.max(np.abs(comp.sum(axis=1)-1.0))),
            "min_weight":float(np.min(comp)),
            "max_weight":float(np.max(comp))
        })
        alloc=allocate_counts(rush_att,comp)

        for j,r in enumerate(rushers.itertuples()):
            pid=str(r.player_id_key)
            carries=alloc[:,j]
            ey=player_lookup.get((pid,"rush_ypc"))
            if r.position=="QB": OFFER_CONDITION_BANK[pid]=STARTER_BANK.get(pid,ACTIVE_BANK[pid])
            else: OFFER_CONDITION_BANK.setdefault(pid,ACTIVE_BANK[pid])
            if ey is None: continue

            um=float(r.uncertainty_multiplier)
            ypc=draw_component(ey.center_tx,PLAYER_MODELS["rush_ypc"],"rush_ypc",
                               clip=(-1.5,12.0),uncertainty_multiplier=um)

            micro=rng.normal(0,np.sqrt(np.maximum(carries,1))*RUSH_MICRO_SD_PER_CARRY,size=N_SIMS)
            rush_yards=np.where(carries>0,carries*ypc+micro,0.0)

            SIM_BANK[(pid,"rush_attempts")]=soft_tail_guard(carries,"rush_attempts",r.position)
            SIM_BANK[(pid,"rushing_yards")]=soft_tail_guard(rush_yards,"rushing_yards",r.position)

            DRIVER_ROWS.append({
                "player_id_key":pid,"player_name":r.player_name,"team":team,"driver":"RUSH",
                "availability_probability":float(r.availability_probability),
                "depth_rank":float(r.depth_rank),"replacement_weight":float(r.replacement_weight),
                "preparation_score":float(r.preparation_score),"history_reliability":float(r.history_reliability),
                "uncertainty_multiplier":um,
                "volume_center":float(rush_volume_row.center),
                "share_center":float(player_lookup[(pid,"carry_share")].center),
                "role_probability":float(role_lookup[(pid,"carry_active_prob")].probability) if (pid,"carry_active_prob") in role_lookup else np.nan,
                "eff_center":float(ey.center)
            })

# Derived receiving+rushing yards preserve scenario correlation.
for pid in current["player_id_key"].astype(str).unique():
    rec=SIM_BANK.get((pid,"receiving_yards"))
    rush=SIM_BANK.get((pid,"rushing_yards"))
    if rec is not None or rush is not None:
        if rec is None: rec=np.zeros(N_SIMS)
        if rush is None: rush=np.zeros(N_SIMS)
        pos=str(current_by_id.loc[pid,"position"]) if pid in current_by_id.index else "UNK"
        SIM_BANK[(pid,"rec_rush_yards")]=soft_tail_guard(rec+rush,"rec_rush_yards",pos)

print("Simulation arrays:",len(SIM_BANK))
display(pd.DataFrame(COMPOSITION_QA))
qb_receiver_coupling_audit_df=pd.DataFrame(QB_RECEIVER_COUPLING_QA)
if len(qb_receiver_coupling_audit_df):
    print("QB -> receiver coupling diagnostics:")
    display(qb_receiver_coupling_audit_df)

# %% [markdown]
# ## 14. Projection report + Phoenix fair lines

# %%
rows=[]
for (pid,stat),arr in SIM_BANK.items():
    if pid not in current_by_id.index: continue
    r=current_by_id.loc[pid]; r=r.iloc[0] if isinstance(r,pd.DataFrame) else r
    cond=np.asarray(OFFER_CONDITION_BANK.get(pid,np.ones(N_SIMS,dtype=bool)),bool)
    a=np.asarray(arr,dtype=float); eligible=a[cond & np.isfinite(a)]
    if len(eligible)<MIN_OFFER_CONDITION_SIMS: continue
    if PRIMARY_QB_OUTPUT_ONLY and stat in {"pass_attempts","pass_completions","passing_yards"} and r["position"]=="QB" and primary_qb.get(r["team"]) != r["player_name"]: continue
    rows.append({"game_id":r["game_id"],"team":r["team"],"opponent":r["opponent"],"player_id_key":pid,"player_name":r["player_name"],"position":r["position"],"stat":stat,"mean":float(np.mean(eligible)),"median":float(np.median(eligible)),"sd":float(np.std(eligible)),"p10":float(np.quantile(eligible,.10)),"p25":float(np.quantile(eligible,.25)),"p75":float(np.quantile(eligible,.75)),"p90":float(np.quantile(eligible,.90)),"availability_probability":float(r.get("availability_probability",1.0)),"starter_probability":float(r.get("starter_probability",np.nan)) if r["position"]=="QB" else np.nan,"preparation_score":float(r.get("preparation_score",np.nan)),"history_reliability":float(r.get("history_reliability",1.0)),"uncertainty_multiplier":float(r.get("uncertainty_multiplier",1.0)),"depth_rank":float(r.get("depth_rank",np.nan)),"availability_flag":r.get("availability_flag","REVIEW"),"offer_condition_sims":int(cond.sum())})

report=pd.DataFrame(rows)
def probability_temperature(history_reliability, uncertainty_multiplier):
    if not USE_UNCERTAINTY_TEMPERING:
        return 1.0
    rel=float(np.clip(history_reliability,0,1)); um=max(float(uncertainty_multiplier),1.0)
    return float(np.clip(1.0 + 0.35*(1.0-rel) + 0.25*(um-1.0),1.0,MAX_PROBABILITY_TEMPERATURE))
def temper_probability(p,temp):
    p=float(np.clip(p,1e-6,1-1e-6))
    if not USE_UNCERTAINTY_TEMPERING or temp<=1.000001: return p
    lp=np.log(p/(1-p)); return float(1/(1+np.exp(-lp/temp)))

report["fair_line"]=report["median"].map(lambda x:(math.floor(float(x))+0.5) if np.isfinite(x) else np.nan)
report["probability_temperature"]=report.apply(lambda r: probability_temperature(r["history_reliability"],r["uncertainty_multiplier"]),axis=1)
for c in ["p_over_unconditional","p_under_unconditional","p_push","p_over_raw","p_under_raw","p_over","p_under","fair_over_odds","fair_under_odds"]: report[c]=np.nan

ladder=[]
def ladder_increment(stat): return 5.0 if stat in {"passing_yards","receiving_yards","rushing_yards","rec_rush_yards"} else 1.0

for idx,row in report.iterrows():
    cond=np.asarray(OFFER_CONDITION_BANK.get(str(row.player_id_key),np.ones(N_SIMS,dtype=bool)),bool)
    arr=np.asarray(SIM_BANK[(str(row.player_id_key),row.stat)],dtype=float)[cond]; arr=arr[np.isfinite(arr)]
    base=float(row.fair_line); temp=float(row.probability_temperature)
    over=float(np.mean(arr>base)); under=float(np.mean(arr<base)); push=max(0.0,1.0-over-under); mass=over+under
    report.at[idx,"p_over_unconditional"]=over; report.at[idx,"p_under_unconditional"]=under; report.at[idx,"p_push"]=push
    if mass>0:
        por=over/mass; pur=under/mass; po=temper_probability(por,temp); pu=1.0-po
        report.at[idx,"p_over_raw"]=por; report.at[idx,"p_under_raw"]=pur; report.at[idx,"p_over"]=po; report.at[idx,"p_under"]=pu
        report.at[idx,"fair_over_odds"]=fair_american(po); report.at[idx,"fair_under_odds"]=fair_american(pu)
    inc=ladder_increment(row.stat)
    for step in range(-LADDER_STEPS_EACH_SIDE,LADDER_STEPS_EACH_SIDE+1):
        line=base+step*inc
        if line<0: continue
        over=float(np.mean(arr>line)); under=float(np.mean(arr<line)); push=max(0.0,1.0-over-under); mass=over+under
        if mass<=0: continue
        por=over/mass; pur=under/mass; po=temper_probability(por,temp); pu=1.0-po
        ladder.append({"game_id":row.game_id,"team":row.team,"opponent":row.opponent,"player_name":row.player_name,"position":row.position,"stat":row.stat,"line":float(line),"p_over_unconditional":over,"p_under_unconditional":under,"p_push":push,"p_over_raw":por,"p_under_raw":pur,"probability_temperature":temp,"p_over":po,"fair_over_odds":fair_american(po),"p_under":pu,"fair_under_odds":fair_american(pu),"is_base_fair_line":abs(line-base)<1e-9})
fair_ladder=pd.DataFrame(ladder)
display(report.sort_values(["game_id","team","position","depth_rank","player_name","stat"]).head(180))

# %% [markdown]
# ## 15. Generative accounting QA

# %%
qa=[]
for team,sim in TEAM_SIM.items():
    qa += [
        {"team":team,"check":"pass attempts plausible","pass":bool((sim["pass_attempts"]>=0).all() and (sim["pass_attempts"]<=70).all())},
        {"team":team,"check":"rush attempts plausible","pass":bool((sim["rush_attempts"]>=0).all() and (sim["rush_attempts"]<=60).all())},
        {"team":team,"check":"targets <= pass attempts","pass":bool((sim["targets"]<=sim["pass_attempts"]).all())},
    ]

for x in COMPOSITION_QA:
    qa.append({
        "team":x["team"],
        "check":f'{x["type"]} composition sums to 1',
        "pass":bool(x["max_sum_error"]<1e-8)
    })

for x in QB_RECEIVER_COUPLING_QA:
    vals=[x.get("mean_catch_multiplier"),x.get("sd_catch_multiplier"),
          x.get("mean_ypr_multiplier"),x.get("sd_ypr_multiplier")]
    qa.append({
        "team":x.get("team",""),
        "check":"QB receiver coupling diagnostics finite",
        "pass":bool(all(np.isfinite(v) for v in vals))
    })
    qa.append({
        "team":x.get("team",""),
        "check":"listed receptions <= QB completions",
        "pass":bool(float(x.get("max_listed_receptions_over_qb_completions",0.0))<=1e-9)
    })
    qa.append({
        "team":x.get("team",""),
        "check":"listed receiving yards <= QB passing yards",
        "pass":bool(float(x.get("max_listed_receiving_yards_over_qb_passing_yards",0.0))<=1e-7)
    })

for (pid,stat),arr in SIM_BANK.items():
    if stat=="pass_completions" and (pid,"pass_attempts") in SIM_BANK:
        qa.append({
            "team":str(current_by_id.loc[pid,"team"]),
            "check":f"{pid} completions <= attempts",
            "pass":bool((arr<=SIM_BANK[(pid,"pass_attempts")]+1e-9).all())
        })

for team,pmap in QB_START_PROBS.items():
    qa.append({"team":team,"check":"QB starter probabilities sum to 1","pass":bool(abs(sum(pmap.values())-1.0)<1e-6)})
    if QB_SCENARIO_REASON.get(team)=="CLEAR_DEPTH_LEADER":
        top=max(pmap.values()) if pmap else 0.0
        qa.append({"team":team,"check":"clear QB leader >=99% starter probability","pass":bool(top>=0.99)})
        qa.append({"team":team,"check":"clear QB backup mass <=1%","pass":bool((1.0-top)<=0.01+1e-9)})

for pid,mask in OFFER_CONDITION_BANK.items():
    qa.append({
        "team":str(current_by_id.loc[pid,"team"]) if pid in current_by_id.index else "",
        "check":f"{pid} offer-condition mask valid",
        "pass":bool(len(np.asarray(mask))==N_SIMS)
    })

if len(player_role_proj):
    qa.append({"team":"ALL","check":"role hurdle probabilities within [0,1]","pass":bool(player_role_proj["probability"].between(0,1).all())})
for (pid,stat),arr in SIM_BANK.items():
    if stat=="rush_attempts" and pid in current_by_id.index:
        rr=current_by_id.loc[pid]; rr=rr.iloc[0] if isinstance(rr,pd.DataFrame) else rr
        if rr["position"] in {"WR","TE"}:
            role_row=role_lookup.get((pid,"carry_active_prob"))
            if role_row is not None and float(role_row.probability)<0.995:
                qa.append({"team":str(rr["team"]),"check":f"{pid} WR/TE rush distribution retains zero mass","pass":bool(np.any(np.asarray(arr)<=0))})
qa_df=pd.DataFrame(qa)
print("QA failures:",int((~qa_df["pass"]).sum()) if len(qa_df) else 0)
display(qa_df[~qa_df["pass"]] if len(qa_df) else qa_df)
if len(qa_df) and not qa_df["pass"].all():
    raise RuntimeError("Generative accounting/scenario QA failed.")

report["cv"]=report["sd"]/np.maximum(np.abs(report["mean"]),1.0)
report["mean_median_gap_pct"]=np.abs(report["mean"]-report["median"])/np.maximum(np.abs(report["median"]),1.0)
report["distribution_flag"]=np.select(
    [
        report["cv"]<0.06,
        report["cv"]>1.15,
        report["mean_median_gap_pct"]>0.65
    ],
    ["TOO_NARROW","VERY_WIDE","HIGH_SKEW"],
    default="OK"
)

print("Distribution diagnostics:")
display(report.groupby(["stat","distribution_flag"]).size().reset_index(name="rows"))
print("Probability tempering:")
display(report.groupby("stat")[["probability_temperature","cv","mean_median_gap_pct"]].median().reset_index())

# %% [markdown]
# ## 16. Human-readable HTML dashboard

# %%
# V6.1 dashboard schema guard: fail with a clear message before rendering
# if the report schema ever changes again.
DASHBOARD_REQUIRED_COLUMNS = {
    "player_name","position","depth_rank","availability_probability",
    "starter_probability","preparation_score","uncertainty_multiplier",
    "stat","mean","median","fair_line","p_over","p_under",
    "fair_over_odds","fair_under_odds","p10","p90",
    "availability_flag","distribution_flag"
}
_missing_dashboard_cols = sorted(DASHBOARD_REQUIRED_COLUMNS - set(report.columns))
if _missing_dashboard_cols:
    raise RuntimeError(
        "Dashboard/report schema mismatch. Missing columns: "
        + ", ".join(_missing_dashboard_cols)
    )

def f1(v): return "—" if pd.isna(v) else f"{float(v):.1f}"
def pct(v): return "—" if pd.isna(v) else f"{100*float(v):.1f}%"
def odds(v): return "—" if pd.isna(v) else f"{float(v):+.0f}"
html=["""<!doctype html><html><head><meta charset='utf-8'><title>Phoenix NFL Generative Props V6.1</title><style>
body{font-family:Arial,Helvetica,sans-serif;background:#0d1015;color:#f2f5f9;margin:0}.wrap{max-width:1750px;margin:auto;padding:26px}h1{margin:0}.sub{color:#9aa7b8;margin:8px 0 24px}.game{font-size:22px;font-weight:700;background:#20293a;padding:14px 18px;border-radius:12px;margin-top:28px}.card{background:#171b23;border:1px solid #2d3544;border-radius:12px;padding:15px;margin:12px 0 24px;overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px}th{background:#202633;color:#cbd4e2;padding:8px;text-align:right}td{padding:8px;border-bottom:1px solid #2d3544;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.fair{background:#162943;font-weight:700}.review{background:#493716}.wide{background:#482120}.note{color:#9aa7b8;font-size:12px;margin-top:8px}</style></head><body><div class='wrap'>"""]
html.append(f"<h1>Phoenix NFL Generative Player Props — V6.1</h1><div class='sub'>QB-coupled injury/backup scenarios · model-first · {TARGET_SEASON} Week {TARGET_WEEK} · {N_SIMS:,} simulations</div>")
html.append("<div class='note'>Prop fair prices are conditioned on the player being active; QB passing props are conditioned on that QB starting. Receiver catch/YPR distributions share a partial latent environment from the selected QB in each simulation. Availability/starter probabilities are shown separately. Sportsbook prices are not inputs.</div>")
stat_order=["passing_yards","pass_attempts","pass_completions","receiving_yards","receptions","rushing_yards","rush_attempts","rec_rush_yards"]
for g in week_schedule[["game_id","away_team","home_team"]].itertuples(index=False):
    html.append(f"<div class='game'>{g.away_team} @ {g.home_team}</div>"); gd=report[report["game_id"].eq(g.game_id)]
    for team in [g.away_team,g.home_team]:
        td=gd[gd["team"].eq(team)]
        if len(td)==0: continue
        html.append(f"<div class='card'><h2>{team}</h2><table><tr><th>Player</th><th>Pos</th><th>Depth</th><th>Avail</th><th>Start</th><th>Prep</th><th>Unc×</th><th>Stat</th><th>Mean</th><th>Median</th><th>Fair line</th><th>P Over</th><th>Fair Over</th><th>P Under</th><th>Fair Under</th><th>P10</th><th>P90</th><th>Flag</th></tr>")
        td=td.assign(_ord=td["stat"].map({s:i for i,s in enumerate(stat_order)}).fillna(99)).sort_values(["depth_rank","player_name","_ord"])
        for r in td.itertuples():
            cls="review" if r.availability_flag!="CLEAR" else ("wide" if r.distribution_flag=="VERY_WIDE" else "")
            html.append(f"<tr class='{cls}'><td>{r.player_name}</td><td>{r.position}</td><td>{f1(r.depth_rank)}</td><td>{pct(r.availability_probability)}</td><td>{pct(r.starter_probability)}</td><td>{pct(r.preparation_score)}</td><td>{f1(r.uncertainty_multiplier)}</td><td>{r.stat}</td><td>{f1(r.mean)}</td><td>{f1(r.median)}</td><td class='fair'>{f1(r.fair_line)}</td><td>{pct(r.p_over)}</td><td>{odds(r.fair_over_odds)}</td><td>{pct(r.p_under)}</td><td>{odds(r.fair_under_odds)}</td><td>{f1(r.p10)}</td><td>{f1(r.p90)}</td><td>{r.availability_flag}/{r.distribution_flag}</td></tr>")
        html.append("</table></div>")
html.append("</div></body></html>"); html_file=OUTPUT_DIR/f"PHOENIX_NFL_GENERATIVE_PROPS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.html"; html_file.write_text("\n".join(html),encoding="utf-8"); print(html_file)

# %% [markdown]
# ## 17. Export CSV / JSON / HTML / ZIP

# %%
report_csv=OUTPUT_DIR/f"PHOENIX_NFL_GENERATIVE_PROPS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
ladder_csv=OUTPUT_DIR/f"PHOENIX_NFL_GENERATIVE_FAIR_LADDER_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
metrics_csv=OUTPUT_DIR/f"PHOENIX_NFL_GENERATIVE_MODEL_METRICS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
role_metrics_csv=OUTPUT_DIR/f"PHOENIX_NFL_ROLE_MODEL_METRICS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
role_components_csv=OUTPUT_DIR/f"PHOENIX_NFL_ROLE_COMPONENTS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
team_components_csv=OUTPUT_DIR/f"PHOENIX_NFL_TEAM_COMPONENTS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
player_components_csv=OUTPUT_DIR/f"PHOENIX_NFL_PLAYER_COMPONENTS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
drivers_csv=OUTPUT_DIR/f"PHOENIX_NFL_GENERATIVE_DRIVERS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
context_csv=OUTPUT_DIR/f"PHOENIX_NFL_PERSONNEL_CONTEXT_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
qb_scenarios_json=OUTPUT_DIR/f"PHOENIX_NFL_QB_START_SCENARIOS_{TARGET_SEASON}_W{TARGET_WEEK}.json"
residual_diag_csv=OUTPUT_DIR/f"PHOENIX_NFL_RESIDUAL_CALIBRATION_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
json_file=OUTPUT_DIR/f"PHOENIX_NFL_GENERATIVE_PROPS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.json"

report.to_csv(report_csv,index=False); fair_ladder.to_csv(ladder_csv,index=False); metrics_df.to_csv(metrics_csv,index=False); role_metrics_df.to_csv(role_metrics_csv,index=False); player_role_proj.to_csv(role_components_csv,index=False); residual_diag_df.to_csv(residual_diag_csv,index=False); team_component_proj.to_csv(team_components_csv,index=False); player_component_proj.to_csv(player_components_csv,index=False); pd.DataFrame(DRIVER_ROWS).to_csv(drivers_csv,index=False)
ctx_cols=[c for c in ["game_id","team","opponent","player_id_key","player_name","position","depth_rank","availability_probability","availability_status","availability_flag","practice_score","preparation_score","replacement_weight","current_season_off_snaps","replacement_candidate","career_games_prior","qb_pseudo_starts","inactivity_weeks","history_reliability","uncertainty_multiplier","starter_probability","qb_scenario_reason"] if c in current.columns]
current[ctx_cols].drop_duplicates().to_csv(context_csv,index=False); qb_scenarios_json.write_text(json.dumps(QB_START_PROBS,indent=2),encoding="utf-8")
ensemble_audit_csv=OUTPUT_DIR/f"PHOENIX_NFL_ENSEMBLE_AUDIT_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
role_calibration_csv=OUTPUT_DIR/f"PHOENIX_NFL_ROLE_CALIBRATION_AUDIT_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
qb_receiver_coupling_csv=OUTPUT_DIR/f"PHOENIX_NFL_QB_RECEIVER_COUPLING_AUDIT_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}.csv"
ensemble_audit_df.to_csv(ensemble_audit_csv,index=False)
role_calibration_audit_df.to_csv(role_calibration_csv,index=False)
qb_receiver_coupling_audit_df.to_csv(qb_receiver_coupling_csv,index=False)

payload={
    "engine":"PHOENIX_NFL_GENERATIVE_PROPS_V6_1",
    "model_version":MODEL_VERSION,
    "season":int(TARGET_SEASON),
    "week":int(TARGET_WEEK),
    "generated_at":datetime.now(timezone.utc).isoformat(),
    "n_sims":int(N_SIMS),
    "philosophy":"model-first; hurdle plus compositional usage; short-term role change; position-specific opponent context; causal walk-forward ensembles; robust adaptive residuals; QB-conditioned receiving environment; sportsbook lines excluded from predictive features",
    "pricing_condition":"skill props conditional on active; QB passing props conditional on QB starting",
    "probability_policy":"raw Monte Carlo probabilities drive Phoenix fair odds by default; uncertainty is expressed in the simulated distribution rather than double-tempering probabilities",
    "qb_receiver_coupling":{
        "enabled":bool(USE_QB_RECEIVER_COUPLING),
        "catch_strength":float(QB_RECEIVER_CATCH_COUPLING),
        "ypr_strength":float(QB_RECEIVER_YPR_COUPLING),
        "ratio_min":float(QB_ENV_RATIO_MIN),
        "ratio_max":float(QB_ENV_RATIO_MAX)
    },
    "model_fingerprint":MODEL_FINGERPRINT,
    "model_artifact":str(MODEL_ARTIFACT_PATH),
    "model_artifact_reused":bool(loaded_artifact),
    "qb_start_probabilities":QB_START_PROBS,
    "qb_scenario_reasons":QB_SCENARIO_REASON,
    "qb_receiver_coupling_audit":[{k:to_py(v) for k,v in x.items()} for x in qb_receiver_coupling_audit_df.to_dict(orient="records")],
    "final_stats":list(FINAL_STATS.keys()),
    "projections":[{k:to_py(v) for k,v in x.items()} for x in report.to_dict(orient="records")],
    "fair_ladder":[{k:to_py(v) for k,v in x.items()} for x in fair_ladder.to_dict(orient="records")],
    "team_components":[{k:to_py(v) for k,v in x.items()} for x in team_component_proj.to_dict(orient="records")],
    "player_components":[{k:to_py(v) for k,v in x.items()} for x in player_component_proj.to_dict(orient="records")],
    "role_components":[{k:to_py(v) for k,v in x.items()} for x in player_role_proj.to_dict(orient="records")]
}
json_file.write_text(json.dumps(payload,indent=2),encoding="utf-8"); zip_base=str(BASE_DIR/f"PHOENIX_NFL_GENERATIVE_PROPS_V6_1_{TARGET_SEASON}_W{TARGET_WEEK}"); zip_path=shutil.make_archive(zip_base,"zip",root_dir=str(OUTPUT_DIR))
print("Created:")
for p in [html_file,report_csv,ladder_csv,metrics_csv,role_metrics_csv,role_components_csv,residual_diag_csv,team_components_csv,player_components_csv,drivers_csv,context_csv,qb_scenarios_json,ensemble_audit_csv,role_calibration_csv,qb_receiver_coupling_csv,json_file,zip_path]: print(" ",p)
try:
    from google.colab import files
    files.download(zip_path)
except Exception: pass

# %% [markdown]
# ## 18. V6.1 promotion checklist
#
# 1. Healthy QB1 scenarios should be ~99.5%; backup-QB rushing must be starter-conditioned.
# 2. Review `target_active_prob` / `carry_active_prob` and Brier/log-loss.
# 3. WR/TE rushing distributions must retain zero mass unless role probability is essentially certain.
# 4. Base fair lines are half-points; integer custom lines explicitly report push mass.
# 5. Review raw vs tempered probability gaps and `VERY_WIDE` / `HIGH_SKEW` flags.
# 6. Refresh injury/depth context late on game day; use overrides for confirmed news newer than the feed.
# 7. Historical Parquet partitions should be cache hits.
# 8. Review QB->receiver coupling correlations and multiplier dispersion; uncertain-QB teams should transmit uncertainty into receiver distributions.\n# 9. Sportsbook price remains downstream only.
