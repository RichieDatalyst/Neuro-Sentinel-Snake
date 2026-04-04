"""
ml/xai.py — Explainability (XAI) using SHAP

Runs SHAP analysis on all trained models:
  - Imitation MLP      (Option 1)
  - Random Forest      (Option 2)
  - Failure Predictor  (Option 5)

Saves SHAP values to data/ for the Streamlit dashboard to render.

Run:
    python -m ml.xai
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import numpy as np
import pandas as pd
import config as C
from ml.features import (
    get_classifier_data, get_failure_data, encode_actions, STATE_FEATURES
)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(X, y, n=500, seed=42):
    """Randomly sample n rows for SHAP (faster, still representative)."""
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


def _save_shap(name: str, shap_values, X_sample, feature_names: list):
    """Save mean |SHAP| per feature to JSON for the dashboard."""
    if isinstance(shap_values, list):
        # Multi-class: average absolute across classes
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
    
    # Ensure 1D array to prevent multi-element conversion errors
    mean_abs = np.asarray(mean_abs).flatten()

    result = {
        feat: round(float(val), 6)
        for feat, val in zip(feature_names, mean_abs)
    }
    # Sort descending
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    out_path = os.path.join(C.DATA_DIR, f"shap_{name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}")
    return result


def _print_shap(name: str, shap_dict: dict):
    print(f"\n  {name} — SHAP feature importance:")
    for feat, val in shap_dict.items():
        bar = "█" * int(val * 300)
        print(f"    {feat:20s} {bar} {val:.4f}")


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def explain_random_forest():
    import shap
    """SHAP TreeExplainer on the Random Forest classifier."""
    print("\n  ── Random Forest (Opt 2) ──")
    if not os.path.exists(C.CLASSIFIER_MODEL_PATH):
        print("  Model not found. Run: python -m ml.classifier")
        return None

    with open(C.CLASSIFIER_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    rf = bundle["model"]

    X, y, _ = get_classifier_data()
    y_enc   = encode_actions(y)
    X_s, _  = _sample(X, y_enc)

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_s)

    result = _save_shap("random_forest", shap_values, X_s, STATE_FEATURES)
    _print_shap("Random Forest", result)
    return result


def explain_failure_predictor():
    import shap
    """SHAP TreeExplainer or KernelExplainer on failure predictor."""
    print("\n  ── Failure Predictor (Opt 5) ──")
    if not os.path.exists(C.FAILURE_MODEL_PATH):
        print("  Model not found. Run: python -m ml.failure_predictor")
        return None

    with open(C.FAILURE_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model  = bundle["model"]
    scaler = bundle["scaler"]

    X, y, _ = get_failure_data()
    X_s, y_s = _sample(X, y, n=300)
    if scaler:
        X_s = scaler.transform(X_s)

    if hasattr(model, "estimators_"):
        # Tree-based — fast
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_s)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
    else:
        # Fallback: KernelExplainer (slower)
        background  = shap.sample(X_s, 50)
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_s, nsamples=50)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    result = _save_shap("failure_predictor", shap_values, X_s, STATE_FEATURES)
    _print_shap("Failure Predictor", result)
    return result


def explain_imitation():
    import shap
    """Permutation-based importance for the MLP (SHAP KernelExplainer is slow for MLPs)."""
    print("\n  ── Imitation MLP (Opt 1) — Permutation SHAP ──")
    if not os.path.exists(C.IMITATION_MODEL_PATH):
        print("  Model not found. Run: python -m ml.imitation")
        return None

    with open(C.IMITATION_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    mlp    = bundle["model"]
    scaler = bundle["scaler"]

    from ml.features import get_imitation_data
    X, y, _ = get_imitation_data(agent=C.IMITATION_TRAIN_AGENT)
    y_enc   = encode_actions(y)
    X_s     = scaler.transform(X)
    X_samp, y_samp = _sample(X_s, y_enc, n=200)

    background  = shap.sample(X_samp, 50)
    explainer   = shap.KernelExplainer(
        lambda x: mlp.predict_proba(x), background
    )
    shap_values = explainer.shap_values(X_samp[:50], nsamples=50)

    result = _save_shap("imitation_mlp", shap_values, X_samp[:50], STATE_FEATURES)
    _print_shap("Imitation MLP", result)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all():
    print("\n" + "="*60)
    print("  XAI — SHAP Explainability Analysis")
    print("="*60)

    results = {}
    results["random_forest"]      = explain_random_forest()
    results["failure_predictor"]  = explain_failure_predictor()
    results["imitation_mlp"]      = explain_imitation()

    print("\n  XAI analysis complete.")
    print(f"  SHAP JSON files written to: {C.DATA_DIR}")
    return results


if __name__ == "__main__":
    run_all()