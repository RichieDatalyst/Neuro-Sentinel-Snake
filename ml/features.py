"""
ml/features.py — Shared feature engineering.

Every ML module imports from here. This is the single source of truth
for which columns are features, which are labels, and how to load data.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import config as C


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Core 11-dim state vector — input to all classifiers
STATE_FEATURES = [
    "danger_up", "danger_down", "danger_left", "danger_right",
    "food_up", "food_down", "food_left", "food_right",
    "dir_x", "dir_y",
    "dist_to_food",
]

# Richer feature set for episode-level models (Opt 4, 5, 6)
EPISODE_FEATURES = [
    "total_steps", "final_score", "foods_eaten",
    "avg_dist_to_food", "direction_changes",
    "dead_end_entries", "path_optimality", "steps_per_food",
]

# Label columns
LABEL_ACTION      = "action"           # Opt 1+2 — what action was taken
LABEL_DIED_NEXT10 = "died_next_10"     # Opt 5  — will snake die soon
LABEL_DIED        = "died"             # Opt 6  — did episode end in death


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_game_log() -> pd.DataFrame:
    """Load step-level log. Raises FileNotFoundError if logger hasn't run."""
    if not os.path.exists(C.GAME_LOG_PATH):
        raise FileNotFoundError(
            f"Game log not found at {C.GAME_LOG_PATH}.\n"
            f"Run:  python -m ml.logger"
        )
    df = pd.read_csv(C.GAME_LOG_PATH)
    _validate_game_log(df)
    return df


def load_episode_stats() -> pd.DataFrame:
    """Load episode-level summary stats."""
    if not os.path.exists(C.EPISODE_STATS_PATH):
        raise FileNotFoundError(
            f"Episode stats not found at {C.EPISODE_STATS_PATH}.\n"
            f"Run:  python -m ml.logger"
        )
    return pd.read_csv(C.EPISODE_STATS_PATH)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_game_log(df: pd.DataFrame):
    """Catch common data quality issues early."""
    issues = []

    # Food should never be on a wall — we can't check the maze map here
    # but we can check that food coordinates are reasonable
    if (df["food_x"] < 0).any() or (df["food_y"] < 0).any():
        issues.append("Negative food coordinates detected.")

    # State feature columns must all be 0 or 1 (except dist_to_food, dir_x, dir_y)
    binary_cols = [c for c in STATE_FEATURES if c not in ("dist_to_food","dir_x","dir_y")]
    for col in binary_cols:
        if not df[col].isin([0, 1]).all():
            issues.append(f"Non-binary values in column '{col}'.")

    # No NaNs in feature columns
    nulls = df[STATE_FEATURES].isnull().sum()
    if nulls.any():
        issues.append(f"NaN values found: {nulls[nulls > 0].to_dict()}")

    if issues:
        print("[DATA VALIDATION WARNINGS]")
        for iss in issues:
            print(f"  ⚠ {iss}")
    else:
        print("[Data validation passed]")


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def get_imitation_data(agent: str = "AStar", exclude_mazes: list = None):
    """
    Return X (state features) and y (actions) for imitation learning.
    Optionally exclude mazes for held-out evaluation.
    """
    df = load_game_log()
    df = df[df["agent"] == agent].copy()
    if exclude_mazes:
        df = df[~df["maze"].isin(exclude_mazes)]

    X = df[STATE_FEATURES].values
    y = df[LABEL_ACTION].values
    return X, y, df


def get_classifier_data(exclude_mazes: list = None):
    """
    Return X, y for the multi-agent action classifier (Opt 2).
    Includes data from all agents — agent identity is NOT a feature,
    so the model learns behaviour, not identity.
    """
    df = load_game_log()
    if exclude_mazes:
        df = df[~df["maze"].isin(exclude_mazes)]

    X = df[STATE_FEATURES].values
    y = df[LABEL_ACTION].values
    return X, y, df


def get_failure_data():
    """
    Return X, y for failure prediction (Opt 5).
    Label: 1 if snake died within next 10 steps, else 0.
    """
    df = load_game_log()
    X  = df[STATE_FEATURES].values
    y  = df[LABEL_DIED_NEXT10].values
    return X, y, df


def get_episode_data():
    """
    Return episode-level feature matrix for clustering (Opt 4)
    and anomaly detection (Opt 6).
    """
    df = load_episode_stats()
    X  = df[EPISODE_FEATURES].values
    return X, df


def get_maze_features() -> pd.DataFrame:
    """
    Return maze structural features for difficulty regression (Opt 3).
    Computed from episode_stats by aggregating per maze.
    """
    ep = load_episode_stats()
    grouped = ep.groupby("maze").agg(
        avg_score        = ("final_score",      "mean"),
        avg_steps        = ("total_steps",       "mean"),
        death_rate       = ("died",              "mean"),
        avg_foods_eaten  = ("foods_eaten",       "mean"),
        avg_direction_ch = ("direction_changes", "mean"),
        avg_dead_ends    = ("dead_end_entries",  "mean"),
        avg_optimality   = ("path_optimality",   "mean"),
        avg_steps_per_food = ("steps_per_food",  "mean"),
    ).reset_index()

    # Difficulty score — normalised so easy mazes separate even with 0 deaths.
    # Weights: steps_per_food=40%, dead_ends=30%, optimality=20%, deaths=10%
    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 1e-9 else pd.Series(
            [0.5] * len(s), index=s.index)

    grouped["difficulty_score"] = (
        _norm(grouped["avg_steps_per_food"]) * 0.40 +
        _norm(grouped["avg_dead_ends"])      * 0.30 +
        (1 - _norm(grouped["avg_optimality"])) * 0.20 +
        _norm(grouped["death_rate"])         * 0.10
    ).round(4)

    grouped.to_csv(C.MAZE_FEATURES_PATH, index=False)
    return grouped


# ---------------------------------------------------------------------------
# Action encoding helper
# ---------------------------------------------------------------------------

ACTION_LABELS = {0: "Up", 3: "Right", 6: "Down", 9: "Left"}
ACTION_CODES  = sorted(ACTION_LABELS.keys())   # [0, 3, 6, 9]


def encode_actions(y: np.ndarray) -> np.ndarray:
    """Map [0,3,6,9] to [0,1,2,3] for sklearn classifiers."""
    mapping = {0: 0, 3: 1, 6: 2, 9: 3}
    return np.array([mapping[a] for a in y])


def decode_actions(y_encoded: np.ndarray) -> np.ndarray:
    """Map [0,1,2,3] back to [0,3,6,9]."""
    mapping = {0: 0, 1: 3, 2: 6, 3: 9}
    return np.array([mapping[a] for a in y_encoded])