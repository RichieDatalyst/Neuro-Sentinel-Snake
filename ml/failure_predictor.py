# failure_predictor.py —> Train a binary classifier to predict high-danger states. Run:
# python -m ml.failure_predictor

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import config as C
from ml.features import load_game_log, STATE_FEATURES

from sklearn.ensemble     import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    precision_recall_curve, roc_curve, average_precision_score,
)
from sklearn.preprocessing import StandardScaler


def _build_danger_label(df: pd.DataFrame) -> np.ndarray:
    """
    Build a binary label: 1 = high danger state, 0 = safe.

    High danger is defined as ANY of:
      - 2 or more danger flags are 1 simultaneously (boxed in on multiple sides)
      - All four food direction flags are 0 (snake is aligned with food on both axes
        but still surrounded disoriented state)
      - danger_straight = 1: the direction the snake is currently moving leads
        into a wall within 1 step

    This gives ~15-30% positive rate, enough for a meaningful binary classifier.
    """
    danger_cols = ["danger_up", "danger_down", "danger_left", "danger_right"]
    danger_sum  = df[danger_cols].sum(axis=1)

    # Condition 1: surrounded on 2+ sides
    boxed_in = (danger_sum >= 2).astype(int)

    # Condition 2: moving toward immediate wall
    # dir_x=1 → moving right → danger_right matters
    # dir_x=-1 → moving left  → danger_left matters
    # dir_y=1 → moving down   → danger_down matters
    # dir_y=-1 → moving up    → danger_up matters
    moving_right = (df["dir_x"] ==  1) & (df["danger_right"] == 1)
    moving_left  = (df["dir_x"] == -1) & (df["danger_left"]  == 1)
    moving_down  = (df["dir_y"] ==  1) & (df["danger_down"]  == 1)
    moving_up    = (df["dir_y"] == -1) & (df["danger_up"]    == 1)
    heading_into_wall = (moving_right | moving_left | moving_down | moving_up).astype(int)

    
    label = np.maximum(boxed_in.values, heading_into_wall.values)
    return label


def train(log_mlflow: bool = True):
    print("\n" + "="*60)
    print(f"  Option 5 — High-Danger State Prediction")
    print("="*60)

    
    df = load_game_log()
    y_full = _build_danger_label(df)
    X_full = df[STATE_FEATURES].values

    pos   = y_full.sum()
    neg   = len(y_full) - pos
    ratio = pos / len(y_full) * 100

    print(f"\n  Full dataset     : {len(y_full):,} steps")
    print(f"  High-danger (1)  : {int(pos):,}  ({ratio:.1f}%)")
    print(f"  Safe (0)         : {int(neg):,}  ({100-ratio:.1f}%)")
    print(f"\n  Label definition: 2+ danger flags OR heading directly into wall")

    
    if len(np.unique(y_full)) < 2:
        print("\n  WARNING: Only one class found in labels.")
        print("  This means all states are either all-safe or all-dangerous.")
        print("  Saving a dummy model and continuing.")
        bundle = {
            "model": None, "scaler": None,
            "roc_fpr": [0, 1], "roc_tpr": [0, 1],
            "pr_prec": [1, 0], "pr_rec": [0, 1],
            "threshold": C.FAILURE_ALERT_THRESHOLD,
            "features": STATE_FEATURES,
            "label_definition": "high_danger_proxy",
        }
        with open(C.FAILURE_MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        return bundle

    
    sample_size = min(getattr(C, 'FAILURE_SAMPLE_SIZE', 80_000), len(y_full))

    
    pos_idx = np.where(y_full == 1)[0]
    neg_idx = np.where(y_full == 0)[0]
    rng     = np.random.default_rng(C.FAILURE_RANDOM_STATE)

    n_pos   = min(len(pos_idx), sample_size // 2)
    n_neg   = min(len(neg_idx), sample_size - n_pos)
    sampled = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ])
    rng.shuffle(sampled)

    X = X_full[sampled]
    y = y_full[sampled]

    actual_pos = y.sum()
    print(f"\n  Working sample   : {len(y):,} rows")
    print(f"  Positive in sample: {int(actual_pos):,}  ({actual_pos/len(y)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=C.FAILURE_TEST_SIZE,
        random_state=C.FAILURE_RANDOM_STATE,
        stratify=y,
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=500,
            random_state=C.FAILURE_RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=C.FAILURE_RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=C.FAILURE_RANDOM_STATE,
        ),
    }

    print("\n  Model comparison:")
    results, best_name, best_auc, best_bundle = [], None, -1, None

    for name, model in models.items():
        needs_scaling = name == "LogisticRegression"
        Xtr = X_train_s if needs_scaling else X_train
        Xte = X_test_s  if needs_scaling else X_test

        model.fit(Xtr, y_train)
        y_prob = model.predict_proba(Xte)[:, 1]
        y_pred = (y_prob >= C.FAILURE_ALERT_THRESHOLD).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        ap  = average_precision_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_p, pr_r, _ = precision_recall_curve(y_test, y_prob)

        print(f"\n    {name}")
        print(f"      ROC-AUC : {auc:.4f}  |  Avg Precision: {ap:.4f}  |  Acc: {acc:.4f}")

        results.append({
            "model": name, "roc_auc": round(auc, 4),
            "avg_precision": round(ap, 4), "accuracy": round(acc, 4),
        })

        if auc > best_auc:
            best_auc  = auc
            best_name = name
            best_bundle = {
                "model":     model,
                "scaler":    scaler if needs_scaling else None,
                "roc_fpr":   fpr.tolist(),
                "roc_tpr":   tpr.tolist(),
                "pr_prec":   pr_p.tolist(),
                "pr_rec":    pr_r.tolist(),
                "threshold": C.FAILURE_ALERT_THRESHOLD,
                "features":  STATE_FEATURES,
                "label_definition": "high_danger_proxy",
            }

    print(f"\n  Best model: {best_name}  (ROC-AUC={best_auc:.4f})")

    best_m = best_bundle["model"]
    Xte_b  = X_test_s if best_bundle["scaler"] else X_test
    y_prob = best_m.predict_proba(Xte_b)[:, 1]
    y_pred = (y_prob >= C.FAILURE_ALERT_THRESHOLD).astype(int)
    print("\n  Classification report (best model, test set):")
    print(classification_report(y_test, y_pred,
          target_names=["safe", "high_danger"], zero_division=0))

    if hasattr(best_m, "feature_importances_"):
        imps = best_m.feature_importances_
        print("  Feature importances:")
        for feat, imp in sorted(zip(STATE_FEATURES, imps),
                                key=lambda x: x[1], reverse=True):
            print(f"    {feat:20s} {'█'*int(imp*50)} {imp:.4f}")

    with open(C.FAILURE_MODEL_PATH, "wb") as f:
        pickle.dump(best_bundle, f)
    print(f"\n  Model saved → {C.FAILURE_MODEL_PATH}")

    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(C.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="failure_predictor"):
                mlflow.log_params({
                    "best_model":      best_name,
                    "alert_threshold": C.FAILURE_ALERT_THRESHOLD,
                    "label":           "high_danger_proxy",
                    "positive_rate":   round(ratio, 2),
                })
                mlflow.log_metrics({
                    "roc_auc":       round(best_auc, 4),
                    "avg_precision": round(
                        [r["avg_precision"] for r in results
                         if r["model"] == best_name][0], 4),
                })
            print("  MLflow run logged.")
        except ImportError:
            print("  MLflow not installed — skipping.")

    return best_bundle


def predict_failure_prob(state_vector: list) -> float:
    """Return probability [0,1] that the current state is high-danger."""
    with open(C.FAILURE_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("model") is None:
        return 0.0
    X = np.array([state_vector])
    if bundle["scaler"]:
        X = bundle["scaler"].transform(X)
    return float(bundle["model"].predict_proba(X)[0, 1])


if __name__ == "__main__":
    train()