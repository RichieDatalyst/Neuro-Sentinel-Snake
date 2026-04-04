"""
ml/anomaly_detector.py — Option 6: Anomaly Detection + Drift Monitoring

Uses Isolation Forest to profile each agent's normal behaviour.
Flags anomalous episodes. Computes rolling score window for drift detection.

Run:
    python -m ml.anomaly_detector
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
import numpy as np
import pandas as pd
import config as C
from ml.features import get_episode_data, EPISODE_FEATURES

from sklearn.ensemble      import IsolationForest
from sklearn.preprocessing import StandardScaler


def train(log_mlflow: bool = True):
    print("\n" + "="*60)
    print("  Option 6 — Anomaly Detection + Drift Monitoring")
    print("="*60)

    X, df = get_episode_data()

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # ── Per-agent Isolation Forest ────────────────────────────────────────
    agents   = df["agent"].unique()
    models   = {}
    all_scores = []

    print(f"\n  Contamination rate: {C.ANOMALY_CONTAMINATION} "
          f"(~{C.ANOMALY_CONTAMINATION*100:.0f}% expected anomalies)")

    for agent in agents:
        mask    = df["agent"] == agent
        X_agent = X_s[mask]

        iso = IsolationForest(
            contamination=C.ANOMALY_CONTAMINATION,
            random_state=C.ANOMALY_RANDOM_STATE,
            n_estimators=100,
        )
        iso.fit(X_agent)

        # Anomaly scores: negative = more anomalous
        scores = iso.decision_function(X_agent)
        labels = iso.predict(X_agent)   # -1 = anomaly, 1 = normal

        df.loc[mask, "anomaly_score"] = scores
        df.loc[mask, "is_anomaly"]    = (labels == -1).astype(int)

        n_anomalies = (labels == -1).sum()
        print(f"\n  {agent}:")
        print(f"    Episodes       : {mask.sum()}")
        print(f"    Anomalies found: {n_anomalies} "
              f"({n_anomalies/mask.sum()*100:.1f}%)")

        # Show most anomalous episodes
        agent_df = df[mask].copy()
        agent_df["anomaly_score"] = scores
        worst = agent_df.nsmallest(3, "anomaly_score")[
            ["episode_id", "maze", "final_score", "died",
             "total_steps", "anomaly_score"]
        ]
        print(f"    Top 3 anomalous episodes:\n{worst.to_string(index=False)}")

        models[agent] = iso

    # ── Drift detection ───────────────────────────────────────────────────
    print("\n\n  Drift detection (rolling window analysis):")
    drift_results = []

    for agent in agents:
        for maze in df["maze"].unique():
            subset = df[(df["agent"] == agent) & (df["maze"] == maze)].copy()
            if len(subset) < C.ANOMALY_DRIFT_WINDOW * 2:
                continue

            scores = subset["final_score"].values
            window = C.ANOMALY_DRIFT_WINDOW

            # Compute rolling means
            early_mean = scores[:window].mean()
            late_mean  = scores[-window:].mean()

            drop = (early_mean - late_mean) / max(abs(early_mean), 1e-6)
            alert = drop >= C.ANOMALY_DRIFT_DROP

            drift_results.append({
                "agent":       agent,
                "maze":        maze,
                "early_mean":  round(early_mean, 2),
                "late_mean":   round(late_mean,  2),
                "drop_pct":    round(drop * 100,  1),
                "drift_alert": alert,
            })

            if alert:
                print(f"  ⚠ DRIFT ALERT: {agent} on {maze} — "
                      f"score dropped {drop*100:.1f}% "
                      f"(early={early_mean:.1f} → late={late_mean:.1f})")

    drift_df = pd.DataFrame(drift_results)
    if not drift_df.empty and not drift_df["drift_alert"].any():
        print("  No drift detected across all agent/maze combinations.")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(C.DATA_DIR, "episode_anomaly.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Anomaly data saved → {out_path}")

    drift_path = os.path.join(C.DATA_DIR, "drift_report.csv")
    drift_df.to_csv(drift_path, index=False)
    print(f"  Drift report saved → {drift_path}")

    bundle = {
        "models":  models,
        "scaler":  scaler,
        "agents":  list(agents),
        "features": EPISODE_FEATURES,
    }
    with open(C.ANOMALY_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Models saved      → {C.ANOMALY_MODEL_PATH}")

    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(C.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="anomaly_detection"):
                mlflow.log_params({
                    "contamination": C.ANOMALY_CONTAMINATION,
                    "drift_window":  C.ANOMALY_DRIFT_WINDOW,
                    "drift_drop":    C.ANOMALY_DRIFT_DROP,
                })
                total_anomalies = int(df["is_anomaly"].sum())
                drift_alerts    = int(drift_df["drift_alert"].sum()) if not drift_df.empty else 0
                mlflow.log_metrics({
                    "total_anomalies": total_anomalies,
                    "drift_alerts":    drift_alerts,
                })
            print("  MLflow run logged.")
        except ImportError:
            print("  MLflow not installed — skipping. Run: pip install mlflow")

    return bundle, df, drift_df


def score_episode(episode_features: list, agent: str) -> float:
    """Return anomaly score for a single episode. Lower = more anomalous."""
    with open(C.ANOMALY_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    if agent not in bundle["models"]:
        raise ValueError(f"No anomaly model for agent '{agent}'")
    X = bundle["scaler"].transform([episode_features])
    return float(bundle["models"][agent].decision_function(X)[0])


if __name__ == "__main__":
    train()