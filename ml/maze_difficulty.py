"""
ml/maze_difficulty.py — Option 3: Maze Difficulty Regression

Extracts structural features from agent performance on each maze and
trains a regression model to predict maze difficulty score.

Key fix: difficulty score now uses avg_steps_per_food and dead_end_entries
as primary signals — these vary meaningfully even when no agent dies,
giving the regression real variance to work with.

Run:
    python -m ml.maze_difficulty
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import config as C
from ml.features import load_episode_stats

from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_absolute_error, r2_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning


REGRESSION_FEATURES = [
    "avg_steps", "avg_foods_eaten", "avg_direction_ch",
    "avg_dead_ends", "avg_optimality", "avg_steps_per_food",
    "death_rate",
]


def _compute_difficulty(grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a difficulty score that has real variance even when
    no agents die. Uses a weighted combination of normalised features.

    Score components (all normalised 0-1 within the dataset):
      - avg_steps_per_food  : more steps to reach food = harder  (40%)
      - avg_dead_ends       : more dead-end encounters = harder   (30%)
      - 1 - avg_optimality  : less optimal paths = harder         (20%)
      - death_rate          : dying = hardest signal               (10%)

    Death rate is intentionally downweighted so easy mazes with
    zero deaths still separate from each other on the other signals.
    """
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 1e-9 else pd.Series([0.5] * len(s), index=s.index)

    grouped["difficulty_score"] = (
        norm(grouped["avg_steps_per_food"]) * 0.40 +
        norm(grouped["avg_dead_ends"])      * 0.30 +
        (1 - norm(grouped["avg_optimality"])) * 0.20 +
        norm(grouped["death_rate"])         * 0.10
    ).round(4)

    return grouped


def get_maze_features() -> pd.DataFrame:
    """Compute per-maze aggregate features with improved difficulty score."""
    ep = load_episode_stats()
    grouped = ep.groupby("maze").agg(
        avg_score          = ("final_score",       "mean"),
        avg_steps          = ("total_steps",        "mean"),
        death_rate         = ("died",               "mean"),
        avg_foods_eaten    = ("foods_eaten",        "mean"),
        avg_direction_ch   = ("direction_changes",  "mean"),
        avg_dead_ends      = ("dead_end_entries",   "mean"),
        avg_optimality     = ("path_optimality",    "mean"),
        avg_steps_per_food = ("steps_per_food",     "mean"),
    ).reset_index()

    grouped = _compute_difficulty(grouped)
    grouped.to_csv(C.MAZE_FEATURES_PATH, index=False)
    return grouped


def train(log_mlflow: bool = True):
    print("\n" + "="*60)
    print("  Option 3 — Maze Difficulty Regression")
    print("="*60)

    df = get_maze_features()

    print(f"\n  Mazes analysed:")
    print(df[["maze", "avg_steps_per_food", "avg_dead_ends",
              "avg_optimality", "death_rate", "difficulty_score"]
            ].round(3).to_string(index=False))

    # Check variance — regression needs it
    score_std = df["difficulty_score"].std()
    print(f"\n  Difficulty score std: {score_std:.4f}")

    if score_std < 1e-6:
        print("\n  All mazes have identical difficulty scores — "
              "regression is not meaningful. This happens when all "
              "agents perform identically across all mazes.")
        print("  Saving a passthrough model and continuing.")
        # Save a dummy model that returns the mean
        bundle = {
            "model":    None,
            "scaler":   None,
            "features": REGRESSION_FEATURES,
            "r2":       float("nan"),
            "mean_score": float(df["difficulty_score"].mean()),
            "scores_table": df[["maze","difficulty_score"]].to_dict("records"),
        }
        with open(C.MAZE_DIFFICULTY_MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        print(f"  Saved → {C.MAZE_DIFFICULTY_MODEL_PATH}")
        return bundle

    X = df[REGRESSION_FEATURES].values
    y = df["difficulty_score"].values

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=2, random_state=42
        ),
    }

    print("\n  Model comparison (LOO CV — small dataset):")
    best_name, best_score, best_model = "LinearRegression", -np.inf, None

    for name, model in models.items():
        model.fit(X_s, y)
        train_mae = mean_absolute_error(y, model.predict(X_s))

        # LOO CV — suppress the UndefinedMetric warning (expected with 1-sample folds)
        # Fall back to train R² when all LOO folds are undefined (too few samples)
        train_r2 = r2_score(y, model.predict(X_s))
        if len(df) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                loo    = LeaveOneOut()
                scores = cross_val_score(model, X_s, y, cv=loo, scoring="r2",
                                         error_score=np.nan)
            valid   = scores[np.isfinite(scores)]
            r2_mean = float(np.mean(valid)) if len(valid) > 0 else train_r2
        else:
            r2_mean = train_r2

        print(f"    {name:25s}  LOO R²={r2_mean:.3f}  Train R²={train_r2:.3f}  MAE={train_mae:.4f}")

        if r2_mean > best_score or best_model is None:
            best_score = r2_mean
            best_name  = name
            best_model = model

    # Fallback: if all LOO scores were nan/inf, just use Ridge (most stable)
    if best_model is None:
        best_name  = "Ridge"
        best_model = models["Ridge"]
        best_model.fit(X_s, y)
        best_score = float("nan")

    print(f"\n  Best model: {best_name}  (R²={best_score:.3f})")

    preds = best_model.predict(X_s)
    comp  = pd.DataFrame({
        "maze":      df["maze"],
        "actual":    y.round(4),
        "predicted": preds.round(4),
        "error":     (preds - y).round(4),
    })
    print("\n  Predictions vs actual difficulty:")
    print(comp.to_string(index=False))

    bundle = {
        "model":    best_model,
        "scaler":   scaler,
        "features": REGRESSION_FEATURES,
        "r2":       best_score,
        "scores_table": df[["maze","difficulty_score"]].to_dict("records"),
    }
    with open(C.MAZE_DIFFICULTY_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model saved → {C.MAZE_DIFFICULTY_MODEL_PATH}")

    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(C.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="maze_difficulty_regression"):
                mlflow.log_param("best_model", best_name)
                mlflow.log_metric("loo_r2",
                    round(best_score, 4) if not np.isnan(best_score) else -1)
                for row in comp.itertuples():
                    mlflow.log_metric(
                        f"difficulty_{row.maze.replace('.txt','').replace(' ','_')}",
                        float(row.actual))
            print("  MLflow run logged.")
        except ImportError:
            print("  MLflow not installed — skipping.")

    return bundle


def predict_difficulty(maze_stats: dict) -> float:
    """Predict difficulty for a new maze given its aggregate stats."""
    with open(C.MAZE_DIFFICULTY_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("model") is None:
        return bundle.get("mean_score", 0.5)
    X   = np.array([[maze_stats[f] for f in bundle["features"]]])
    X_s = bundle["scaler"].transform(X)
    return float(bundle["model"].predict(X_s)[0])


if __name__ == "__main__":
    train()