"""
ml/clustering.py — Option 4: Unsupervised Behaviour Clustering

Clusters agents by their behavioral telemetry using K-Means.
Visualises with t-SNE. Produces the most visually striking output.

Run:
    python -m ml.clustering
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import config as C
from ml.features import get_episode_data, EPISODE_FEATURES

from sklearn.cluster         import KMeans
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import silhouette_score
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE


def train(log_mlflow: bool = True):
    print("\n" + "="*60)
    print("  Option 4 — Behaviour Clustering (K-Means + t-SNE)")
    print("="*60)

    X, df = get_episode_data()

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # ── Elbow + Silhouette to find optimal k ──────────────────────────────
    print("\n  Finding optimal k (elbow + silhouette):")
    inertias    = []
    sil_scores  = []
    k_range     = list(C.CLUSTERING_K_RANGE)

    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=C.CLUSTERING_RANDOM_STATE,
                     n_init=10)
        lbl = km.fit_predict(X_s)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_s, lbl) if k > 1 else 0
        sil_scores.append(sil)
        print(f"    k={k}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    best_k = k_range[int(np.argmax(sil_scores))]
    print(f"\n  Best k = {best_k}  (highest silhouette score)")

    # ── Final clustering ──────────────────────────────────────────────────
    km_final = KMeans(n_clusters=best_k, random_state=C.CLUSTERING_RANDOM_STATE,
                      n_init=10)
    df["cluster"] = km_final.fit_predict(X_s)

    print("\n  Cluster composition (agent × cluster):")
    pivot = df.groupby(["agent", "cluster"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    print("\n  Cluster centroids (feature means):")
    centroids_df = pd.DataFrame(
        scaler.inverse_transform(km_final.cluster_centers_),
        columns=EPISODE_FEATURES,
    ).round(3)
    centroids_df.index.name = "cluster"
    print(centroids_df.to_string())

    # ── t-SNE for 2D visualisation ─────────────────────────────────────
    print("\n  Running t-SNE (this takes ~10–30s)...")
    perp = min(C.TSNE_PERPLEXITY, len(X_s) - 1)
    tsne = TSNE(n_components=2, perplexity=perp,
                random_state=C.TSNE_RANDOM_STATE)

    # Reduce to 10 dims first with PCA for speed when dataset is large
    n_pca = min(10, X_s.shape[1], X_s.shape[0] - 1)
    X_pca = PCA(n_components=n_pca,
                random_state=C.TSNE_RANDOM_STATE).fit_transform(X_s)
    X_2d  = tsne.fit_transform(X_pca)

    df["tsne_x"] = X_2d[:, 0]
    df["tsne_y"] = X_2d[:, 1]

    # Save enriched episode stats with cluster + tsne coords
    out_path = os.path.join(C.DATA_DIR, "episode_clustered.csv")
    df.to_csv(out_path, index=False)
    print(f"  Clustered data saved → {out_path}")

    # Save clustering artifacts
    artifacts = {
        "best_k":       best_k,
        "inertias":     inertias,
        "sil_scores":   sil_scores,
        "k_range":      k_range,
        "feature_names": EPISODE_FEATURES,
    }
    art_path = os.path.join(C.DATA_DIR, "clustering_artifacts.json")
    with open(art_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"  Artifacts saved   → {art_path}")

    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(C.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="behaviour_clustering"):
                mlflow.log_params({"best_k": best_k, "perplexity": perp})
                mlflow.log_metric("silhouette_score", round(max(sil_scores), 4))
            print("  MLflow run logged.")
        except ImportError:
            print("  MLflow not installed — skipping. Run: pip install mlflow")

    return df, km_final, artifacts


if __name__ == "__main__":
    train()