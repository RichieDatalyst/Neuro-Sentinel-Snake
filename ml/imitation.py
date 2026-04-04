"""
ml/imitation.py — Option 1: Imitation Learning (Behavioural Cloning)

Trains an MLP classifier to imitate A* by learning from its demonstrations.
Evaluates on held-out mazes to expose covariate shift / distribution shift.

Run:
    python -m ml.imitation
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import config as C
from ml.features import (
    get_imitation_data, encode_actions, decode_actions,
    ACTION_LABELS, STATE_FEATURES,
)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(log_mlflow: bool = True):
    print("\n" + "="*60)
    print("  Option 1 — Imitation Learning (Behavioural Cloning)")
    print("="*60)

    # Load data — exclude held-out mazes for honest eval
    X_train_full, y_train_full, df_train = get_imitation_data(
        agent=C.IMITATION_TRAIN_AGENT,
        exclude_mazes=C.IMITATION_TEST_MAZES,
    )
    X_test_full, y_test_full, df_test = get_imitation_data(
        agent=C.IMITATION_TRAIN_AGENT,
    )
    # Keep only held-out mazes for test set
    mask   = df_test["maze"].isin(C.IMITATION_TEST_MAZES)
    X_test = X_test_full[mask]
    y_test = y_test_full[mask]

    # Sample training data — MLP converges well at 100k; 1.2M just wastes time
    train_sample = min(100_000, len(X_train_full))
    rng = np.random.default_rng(C.IMITATION_RANDOM_STATE)
    idx = rng.choice(len(X_train_full), size=train_sample, replace=False)
    X_train = X_train_full[idx]
    y_train = y_train_full[idx]

    # Encode [0,3,6,9] → [0,1,2,3]
    y_train_enc = encode_actions(y_train)
    y_test_enc  = encode_actions(y_test)

    print(f"\n  Training samples : {len(X_train):,}  (sampled from {len(X_train_full):,})")
    print(f"  Test samples     : {len(X_test):,}  (held-out mazes: {C.IMITATION_TEST_MAZES})")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=C.IMITATION_HIDDEN_SIZES,
        max_iter=C.IMITATION_MAX_ITER,
        random_state=C.IMITATION_RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )
    print("\n  Training MLP...")
    mlp.fit(X_train_s, y_train_enc)

    # Evaluate
    train_acc = accuracy_score(y_train_enc, mlp.predict(X_train_s))
    test_acc  = accuracy_score(y_test_enc,  mlp.predict(X_test_s))
    gap       = train_acc - test_acc

    print(f"\n  Train accuracy   : {train_acc:.4f}")
    print(f"  Test accuracy    : {test_acc:.4f}  ← on unseen mazes")
    print(f"  Covariate shift  : {gap:.4f}  (train−test gap — higher = more overfitting)")

    print("\n  Classification report (test set):")
    label_names = [ACTION_LABELS[a] for a in sorted(ACTION_LABELS)]
    print(classification_report(
        y_test_enc, mlp.predict(X_test_s), target_names=label_names, zero_division=0
    ))

    # Feature importance via permutation (simple version)
    feature_importance = _permutation_importance(mlp, X_test_s, y_test_enc)
    print("  Feature importance (permutation):")
    for feat, imp in sorted(zip(STATE_FEATURES, feature_importance),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"    {feat:20s} {bar} {imp:.4f}")

    # Save model + scaler together
    bundle = {"model": mlp, "scaler": scaler}
    with open(C.IMITATION_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model saved → {C.IMITATION_MODEL_PATH}")

    # MLflow logging
    if log_mlflow:
        try:
            import mlflow
            import mlflow.sklearn
            mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(C.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="imitation_learning"):
                mlflow.log_params({
                    "agent":        C.IMITATION_TRAIN_AGENT,
                    "hidden_sizes": str(C.IMITATION_HIDDEN_SIZES),
                    "max_iter":     C.IMITATION_MAX_ITER,
                    "test_mazes":   str(C.IMITATION_TEST_MAZES),
                })
                mlflow.log_metrics({
                    "train_accuracy":   round(train_acc, 4),
                    "test_accuracy":    round(test_acc,  4),
                    "covariate_shift":  round(gap,        4),
                })
                mlflow.sklearn.log_model(mlp, "imitation_mlp")
            print("  MLflow run logged.")
        except ImportError:
            print("  MLflow not installed — skipping experiment logging. Run: pip install mlflow")

    return {
        "train_accuracy":  train_acc,
        "test_accuracy":   test_acc,
        "covariate_shift": gap,
        "model":           mlp,
        "scaler":          scaler,
    }


# ---------------------------------------------------------------------------
# Inference helper (used by dashboard)
# ---------------------------------------------------------------------------

def predict_action(state_vector: list) -> int:
    """Given an 11-dim state vector, return predicted action code."""
    with open(C.IMITATION_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    mlp, scaler = bundle["model"], bundle["scaler"]
    X = scaler.transform([state_vector])
    enc = mlp.predict(X)[0]
    return decode_actions(np.array([enc]))[0]


# ---------------------------------------------------------------------------
# Permutation importance (no extra dependency)
# ---------------------------------------------------------------------------

def _permutation_importance(model, X, y, n_repeats: int = 5) -> np.ndarray:
    base_score = accuracy_score(y, model.predict(X))
    importances = np.zeros(X.shape[1])
    rng = np.random.default_rng(42)
    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm       = X.copy()
            X_perm[:, col] = rng.permutation(X_perm[:, col])
            scores.append(accuracy_score(y, model.predict(X_perm)))
        importances[col] = base_score - np.mean(scores)
    # Clip negatives to 0 and normalise
    importances = np.clip(importances, 0, None)
    total = importances.sum()
    return importances / total if total > 0 else importances


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()