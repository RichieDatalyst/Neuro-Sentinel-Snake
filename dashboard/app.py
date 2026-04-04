"""
dashboard/app.py — Neuro-Sentinel Snake: ML Analytics Dashboard

Launch:
    python main.py --mode dashboard
    OR directly:
    streamlit run dashboard/app.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import config as C

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=C.DASHBOARD_TITLE,
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_episode_stats():
    if not os.path.exists(C.EPISODE_STATS_PATH):
        return None
    return pd.read_csv(C.EPISODE_STATS_PATH)


@st.cache_data(ttl=30)
def load_game_log():
    if not os.path.exists(C.GAME_LOG_PATH):
        return None
    return pd.read_csv(C.GAME_LOG_PATH)


@st.cache_data(ttl=30)
def load_clustered():
    p = os.path.join(C.DATA_DIR, "episode_clustered.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(ttl=30)
def load_anomaly():
    p = os.path.join(C.DATA_DIR, "episode_anomaly.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(ttl=30)
def load_drift():
    p = os.path.join(C.DATA_DIR, "drift_report.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(ttl=30)
def load_maze_features():
    return pd.read_csv(C.MAZE_FEATURES_PATH) if os.path.exists(C.MAZE_FEATURES_PATH) else None


def load_shap(name: str):
    p = os.path.join(C.DATA_DIR, f"shap_{name}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_failure_model_meta():
    """Load ROC/PR curve data from saved failure predictor bundle."""
    import pickle
    if not os.path.exists(C.FAILURE_MODEL_PATH):
        return None
    with open(C.FAILURE_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def _not_ready(name: str):
    st.info(
        f"**{name}** data not found. "
        f"Run the pipeline first:\n\n"
        f"```\npython main.py --mode simulate\n"
        f"python main.py --mode train\n"
        f"python main.py --mode analyze\n```"
    )


AGENT_COLORS = {
    "AStar":          "#4ec9b0",
    "GreedyBestFirst": "#ce9178",
    "BreadthFirst":   "#569cd6",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f40d.svg",
                 width=48)
st.sidebar.title("Neuro-Sentinel Snake")
st.sidebar.markdown("**ML Analytics Dashboard**")
st.sidebar.divider()

pages = [
    "Overview",
    "Benchmark Results",
    "Behaviour Clustering",
    "Failure Prediction",
    "Anomaly & Drift",
    "Explainability (XAI)",
    "Maze Difficulty",
    "Raw Data Explorer",
]
page = st.sidebar.radio("Navigate", pages)

st.sidebar.divider()
st.sidebar.markdown(
    "**Pipeline commands**\n"
    "```\n"
    "python main.py --mode simulate\n"
    "python main.py --mode train\n"
    "python main.py --mode analyze\n"
    "```"
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("🐍 Neuro-Sentinel Snake")
    st.subheader("ML-Powered Pathfinding Analysis Framework")

    st.markdown("""
    This project benchmarks classical search algorithms (A\\*, BFS, GBFS)
    across multiple maze configurations and applies a full ML analysis stack:

    | Module | Technique | Covers |
    |---|---|---|
    | Imitation Learning | MLP Behavioural Cloning | Supervised learning, covariate shift |
    | Classifier Comparison | RF / GBM / LR + GridSearchCV | Model selection, feature importance |
    | Maze Difficulty | Regression on structural features | Regression, feature engineering |
    | Behaviour Clustering | K-Means + t-SNE | Unsupervised learning, dimensionality reduction |
    | Failure Prediction | Binary classifier + ROC | Imbalanced classification, predictive maintenance |
    | Anomaly Detection | Isolation Forest | Anomaly detection, drift monitoring |
    | Explainability | SHAP values | XAI, model interpretability |
    """)

    st.divider()
    ep = load_episode_stats()
    gl = load_game_log()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Episodes",  f"{len(ep):,}" if ep is not None else "—")
    with col2:
        st.metric("Total Steps",     f"{len(gl):,}" if gl is not None else "—")
    with col3:
        st.metric("Agents",          str(len(C.AGENTS)))
    with col4:
        st.metric("Mazes",           str(len(C.MAZES)))

    if ep is not None:
        st.divider()
        st.subheader("Score distribution by agent")
        fig = px.box(
            ep, x="agent", y="final_score", color="agent",
            color_discrete_map=AGENT_COLORS,
            labels={"final_score": "Final score", "agent": "Agent"},
        )
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Benchmark Results
# ---------------------------------------------------------------------------

elif page == "Benchmark Results":
    st.title("📊 Benchmark Results")

    ep = load_episode_stats()
    if ep is None:
        _not_ready("Episode stats")
        st.stop()

    # Summary table
    summary = ep.groupby("agent").agg(
        avg_score       =("final_score",      "mean"),
        median_score    =("final_score",      "median"),
        avg_steps       =("total_steps",       "mean"),
        death_rate      =("died",              "mean"),
        avg_foods_eaten =("foods_eaten",       "mean"),
        avg_optimality  =("path_optimality",   "mean"),
        avg_steps_per_food=("steps_per_food",  "mean"),
    ).round(3).reset_index()

    st.subheader("Agent performance summary (all mazes)")
    st.dataframe(summary, use_container_width=True)

    st.divider()

    # Per-maze breakdown
    st.subheader("Average score per agent per maze")
    pivot = ep.groupby(["maze","agent"])["final_score"].mean().reset_index()
    fig = px.bar(
        pivot, x="maze", y="final_score", color="agent",
        barmode="group",
        color_discrete_map=AGENT_COLORS,
        labels={"final_score": "Avg score", "maze": "Maze"},
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Death rate heatmap
    st.subheader("Death rate heatmap (agent × maze)")
    heat = ep.groupby(["agent","maze"])["died"].mean().reset_index()
    heat_pivot = heat.pivot(index="agent", columns="maze", values="died")
    fig2 = px.imshow(
        heat_pivot, text_auto=".2f",
        color_continuous_scale="Reds",
        labels={"color": "Death rate"},
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    # Path optimality
    st.subheader("Path optimality (1.0 = perfect)")
    opt = ep.groupby(["agent","maze"])["path_optimality"].mean().reset_index()
    fig3 = px.bar(
        opt, x="maze", y="path_optimality", color="agent",
        barmode="group", color_discrete_map=AGENT_COLORS,
        labels={"path_optimality": "Path optimality", "maze": "Maze"},
    )
    fig3.add_hline(y=1.0, line_dash="dash", line_color="gray",
                   annotation_text="Perfect")
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Confusion Matrix ───────────────────────────────────────────────
    st.divider()
    st.subheader("Confusion matrix → ML classifier vs actual actions")
    st.markdown(
        "Shows where the Random Forest classifier makes mistakes. "
        "Rows = actual action, Columns = predicted action."
    )

    import pickle, os
    if os.path.exists(C.CLASSIFIER_MODEL_PATH):
        try:
            from ml.features import get_classifier_data, encode_actions
            from sklearn.metrics import confusion_matrix

            with open(C.CLASSIFIER_MODEL_PATH, "rb") as _f:
                _bundle = pickle.load(_f)
            _rf = _bundle["model"]

            _X, _y, _ = get_classifier_data()
            _y_enc = encode_actions(_y)

            # Use a 10k sample for speed — confusion matrix doesn't need full data
            import numpy as np
            _rng = np.random.default_rng(42)
            _idx = _rng.choice(len(_X), size=min(10_000, len(_X)), replace=False)
            _Xs, _ys = _X[_idx], _y_enc[_idx]

            _y_pred = _rf.predict(_Xs)
            _cm = confusion_matrix(_ys, _y_pred)
            _labels = ["Up", "Right", "Down", "Left"]

            import pandas as pd
            _cm_df = pd.DataFrame(_cm, index=_labels, columns=_labels)
            _cm_df.index.name = "Actual"

            fig_cm = px.imshow(
                _cm_df, text_auto=True,
                color_continuous_scale="Blues",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                title="Confusion matrix (10k sample)",
            )
            fig_cm.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown(
                "**Reading this:** Diagonal = correct predictions (darker = better). "
                "Off-diagonal = where the model confuses actions. "
                "A strong diagonal means the model learned distinct patterns for each direction."
            )
        except Exception as _e:
            st.info(f"Confusion matrix unavailable: {_e}")
    else:
        st.info("Train the classifier first: `python main.py --mode train`")

    # ── Imitation Learning accuracy ────────────────────────────────────
    st.divider()
    st.subheader("Imitation Learning → Covariate Shift Analysis")
    if os.path.exists(C.IMITATION_MODEL_PATH):
        try:
            with open(C.IMITATION_MODEL_PATH, "rb") as _f:
                _im = pickle.load(_f)
            _mlp = _im["model"]
            _sc  = _im["scaler"]

            from ml.features import get_imitation_data, encode_actions
            from sklearn.metrics import confusion_matrix as _cm2_fn, accuracy_score

            _Xtr, _ytr, _ = get_imitation_data(
                agent=C.IMITATION_TRAIN_AGENT,
                exclude_mazes=C.IMITATION_TEST_MAZES,
            )
            _Xte_f, _yte_f, _df_te = get_imitation_data(agent=C.IMITATION_TRAIN_AGENT)
            _mask = _df_te["maze"].isin(C.IMITATION_TEST_MAZES)
            _Xte = _Xte_f[_mask]; _yte = _yte_f[_mask]

            _ytr_enc = encode_actions(_ytr); _yte_enc = encode_actions(_yte)

            import numpy as np
            _idx2 = np.random.default_rng(42).choice(len(_Xtr), size=min(5000,len(_Xtr)), replace=False)
            _tr_acc = accuracy_score(_ytr_enc[_idx2], _mlp.predict(_sc.transform(_Xtr[_idx2])))
            _idx3 = np.random.default_rng(42).choice(len(_Xte), size=min(5000,len(_Xte)), replace=False)
            _te_acc = accuracy_score(_yte_enc[_idx3], _mlp.predict(_sc.transform(_Xte[_idx3])))

            c1, c2, c3 = st.columns(3)
            c1.metric("Train accuracy",  f"{_tr_acc:.1%}", help="On seen mazes")
            c2.metric("Test accuracy",   f"{_te_acc:.1%}", help="On unseen mazes (held-out)")
            c3.metric("Covariate shift", f"{(_tr_acc-_te_acc):.1%}",
                      delta=f"-{(_tr_acc-_te_acc):.1%}", delta_color="inverse",
                      help="Gap = how much performance drops on new mazes")

            st.markdown(
                f"The imitation MLP achieves **{_tr_acc:.1%}** on mazes it trained on "
                f"but only **{_te_acc:.1%}** on unseen mazes. "
                f"This **{(_tr_acc-_te_acc):.1%} covariate shift** demonstrates a core ML "
                f"concept: a model that memorises expert demonstrations without generalising "
                f"fails when the environment changes — even when the expert (A*) would still succeed."
            )
        except Exception as _e:
            st.info(f"Imitation analysis unavailable: {_e}")
    else:
        st.info("Train the imitation model first: `python main.py --mode train`")

# ---------------------------------------------------------------------------
# Page: Behaviour Clustering
# ---------------------------------------------------------------------------

elif page == "Behaviour Clustering":
    st.title("🔵 Behaviour Clustering")
    st.markdown(
        "K-Means clusters agents by behavioral telemetry. "
        "t-SNE reduces to 2D for visualization."
    )

    cl = load_clustered()
    if cl is None:
        _not_ready("Clustering data")
        st.stop()

    # t-SNE scatter
    st.subheader("t-SNE: agent behavioral profiles")
    fig = px.scatter(
        cl, x="tsne_x", y="tsne_y",
        color="agent", symbol="cluster",
        color_discrete_map=AGENT_COLORS,
        hover_data=["maze", "final_score", "total_steps", "died"],
        labels={"tsne_x": "t-SNE dim 1", "tsne_y": "t-SNE dim 2"},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Cluster composition
    st.subheader("Cluster composition")
    comp = cl.groupby(["cluster","agent"]).size().reset_index(name="count")
    fig2 = px.bar(
        comp, x="cluster", y="count", color="agent",
        color_discrete_map=AGENT_COLORS,
        labels={"cluster": "Cluster", "count": "Episodes"},
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    # Cluster mean stats
    st.subheader("Cluster feature means")
    from ml.features import EPISODE_FEATURES
    means = cl.groupby("cluster")[EPISODE_FEATURES].mean().round(3)
    st.dataframe(means, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Failure Prediction
# ---------------------------------------------------------------------------

elif page == "Failure Prediction":
    st.title("⚠️ Failure Prediction")
    st.markdown(
        f"Binary classifier predicts whether the snake will die within "
        f"the next **{C.FAILURE_LOOKAHEAD_STEPS} steps**. "
        "Maps to real-world *predictive maintenance*."
    )

    bundle = load_failure_model_meta()
    if bundle is None:
        _not_ready("Failure predictor model")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fpr = bundle.get("roc_fpr", [])
        tpr = bundle.get("roc_tpr", [])
        if fpr:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name="ROC", line=dict(color="#4ec9b0", width=2)
            ))
            fig.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(dash="dash", color="gray")
            )
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Precision-Recall Curve")
        pr_p = bundle.get("pr_prec", [])
        pr_r = bundle.get("pr_rec", [])
        if pr_p:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=pr_r, y=pr_p, mode="lines",
                name="PR", line=dict(color="#ce9178", width=2)
            ))
            fig2.update_layout(
                xaxis_title="Recall",
                yaxis_title="Precision",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Live danger simulator
    st.divider()
    st.subheader("🔴 Live danger simulator")
    st.markdown("Adjust the state features and see the predicted failure probability in real time.")

    c1, c2, c3, c4 = st.columns(4)
    danger_up    = c1.selectbox("Danger up",    [0, 1], key="du")
    danger_down  = c2.selectbox("Danger down",  [0, 1], key="dd")
    danger_left  = c3.selectbox("Danger left",  [0, 1], key="dl")
    danger_right = c4.selectbox("Danger right", [0, 1], key="dr")

    c5, c6, c7, c8 = st.columns(4)
    food_up    = c5.selectbox("Food up",    [0, 1], key="fu")
    food_down  = c6.selectbox("Food down",  [0, 1], key="fd")
    food_left  = c7.selectbox("Food left",  [0, 1], key="fl")
    food_right = c8.selectbox("Food right", [0, 1], key="fr")

    c9, c10, c11 = st.columns(3)
    dir_x     = c9.selectbox("Dir X",  [-1, 0, 1], key="dx")
    dir_y     = c10.selectbox("Dir Y", [-1, 0, 1], key="dy")
    dist_food = c11.slider("Dist to food (norm)", 0.0, 1.0, 0.3, key="dist")

    state_vec = [
        danger_up, danger_down, danger_left, danger_right,
        food_up, food_down, food_left, food_right,
        dir_x, dir_y, dist_food,
    ]

    try:
        from ml.failure_predictor import predict_failure_prob
        prob = predict_failure_prob(state_vec)
        color = "red" if prob >= C.FAILURE_ALERT_THRESHOLD else "green"
        st.markdown(
            f"<h2 style='color:{color}'>Failure probability: {prob:.1%}</h2>",
            unsafe_allow_html=True,
        )
        st.progress(float(prob))
        if prob >= C.FAILURE_ALERT_THRESHOLD:
            st.error(f"⚠ DANGER ALERT — probability exceeds threshold "
                     f"({C.FAILURE_ALERT_THRESHOLD:.0%})")
    except Exception as e:
        st.warning(f"Cannot run live predictor: {e}")

# ---------------------------------------------------------------------------
# Page: Anomaly & Drift
# ---------------------------------------------------------------------------

elif page == "Anomaly & Drift":
    st.title("🔍 Anomaly Detection & Drift Monitoring")

    an = load_anomaly()
    dr = load_drift()

    if an is None:
        _not_ready("Anomaly detection data")
        st.stop()

    # Anomaly score timeline
    st.subheader("Anomaly scores by agent (lower = more anomalous)")
    fig = px.box(
        an, x="agent", y="anomaly_score", color="agent",
        color_discrete_map=AGENT_COLORS,
        labels={"anomaly_score": "Isolation Forest score"},
    )
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Anomalous episodes table
    st.subheader("Most anomalous episodes")
    worst = an.nsmallest(20, "anomaly_score")[
        ["episode_id", "agent", "maze", "final_score",
         "total_steps", "died", "anomaly_score"]
    ].round(4)
    st.dataframe(worst, use_container_width=True)

    # Drift report
    st.divider()
    st.subheader("📉 Drift report")
    if dr is None or dr.empty:
        st.success("No drift data available. Run training pipeline first.")
    else:
        alerts = dr[dr["drift_alert"] == True]
        if alerts.empty:
            st.success("✅ No performance drift detected across any agent/maze combination.")
        else:
            st.error(f"⚠ {len(alerts)} drift alert(s) detected!")
            st.dataframe(alerts, use_container_width=True)

        st.subheader("Score trend: early vs late episodes")
        fig2 = go.Figure()
        for _, row in dr.iterrows():
            label = f"{row['agent']} / {row['maze']}"
            color = "red" if row["drift_alert"] else "#4ec9b0"
            fig2.add_trace(go.Scatter(
                x=["Early", "Late"],
                y=[row["early_mean"], row["late_mean"]],
                mode="lines+markers",
                name=label,
                line=dict(color=color),
            ))
        fig2.update_layout(
            xaxis_title="Episode window",
            yaxis_title="Mean score",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Explainability (XAI)
# ---------------------------------------------------------------------------

elif page == "Explainability (XAI)":
    st.title("🧠 Explainability → SHAP Analysis")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) measures how much each "
        "state feature contributes to each model's decisions."
    )

    model_map = {
        "Random Forest (Opt 2)":     "random_forest",
        "Failure Predictor (Opt 5)": "failure_predictor",
        "Imitation MLP (Opt 1)":     "imitation_mlp",
    }

    for label, key in model_map.items():
        shap_data = load_shap(key)
        if shap_data is None:
            st.info(f"SHAP data for **{label}** not found. Run `python main.py --mode analyze`.")
            continue

        st.subheader(f"🔹 {label}")
        features = list(shap_data.keys())
        values   = list(shap_data.values())

        fig = go.Figure(go.Bar(
            x=values, y=features,
            orientation="h",
            marker_color=[
                "#4ec9b0" if "danger" in f else
                "#ce9178" if "food"   in f else
                "#569cd6"
                for f in features
            ],
        ))
        fig.update_layout(
            xaxis_title="Mean |SHAP value|",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

# ---------------------------------------------------------------------------
# Page: Maze Difficulty
# ---------------------------------------------------------------------------

elif page == "Maze Difficulty":
    st.title("🗺️ Maze Difficulty Analysis")
    st.markdown(
        "Regression model predicts maze difficulty from structural features "
        "derived from agent performance."
    )

    mf = load_maze_features()
    if mf is None:
        _not_ready("Maze features")
        st.stop()

    st.subheader("Difficulty scores")
    fig = px.bar(
        mf.sort_values("difficulty_score", ascending=False),
        x="maze", y="difficulty_score",
        color="difficulty_score",
        color_continuous_scale="Reds",
        labels={"difficulty_score": "Difficulty score (0–1)"},
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature breakdown per maze")
    cols_to_show = [
        "maze", "death_rate", "avg_steps", "avg_foods_eaten",
        "avg_dead_ends", "avg_optimality", "difficulty_score",
    ]
    st.dataframe(mf[cols_to_show].round(3), use_container_width=True)

    st.subheader("Death rate vs difficulty score")
    fig2 = px.scatter(
        mf, x="difficulty_score", y="death_rate",
        text="maze", size="avg_steps",
        color="difficulty_score", color_continuous_scale="Reds",
        labels={
            "difficulty_score": "Difficulty score",
            "death_rate":       "Death rate",
        },
    )
    fig2.update_traces(textposition="top center")
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Raw Data Explorer
# ---------------------------------------------------------------------------

elif page == "Raw Data Explorer":
    st.title("🗄️ Raw Data Explorer")

    tab1, tab2 = st.tabs(["Episode stats", "Game log (step level)"])

    with tab1:
        ep = load_episode_stats()
        if ep is None:
            _not_ready("Episode stats")
        else:
            agents_sel = st.multiselect(
                "Filter agents", ep["agent"].unique().tolist(),
                default=ep["agent"].unique().tolist()
            )
            mazes_sel  = st.multiselect(
                "Filter mazes", ep["maze"].unique().tolist(),
                default=ep["maze"].unique().tolist()
            )
            filtered = ep[
                ep["agent"].isin(agents_sel) & ep["maze"].isin(mazes_sel)
            ]
            st.write(f"{len(filtered):,} rows")
            st.dataframe(filtered, use_container_width=True)
            st.download_button(
                "Download CSV", filtered.to_csv(index=False),
                file_name="episode_stats_filtered.csv", mime="text/csv"
            )

    with tab2:
        gl = load_game_log()
        if gl is None:
            _not_ready("Game log")
        else:
            ep_id = st.number_input("Jump to episode ID", min_value=1,
                                    max_value=int(gl["episode_id"].max()),
                                    value=1)
            ep_data = gl[gl["episode_id"] == ep_id]
            st.write(f"Episode {ep_id}: {len(ep_data)} steps")
            st.dataframe(ep_data, use_container_width=True)