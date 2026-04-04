"""
tests/test_pipeline.py — Smoke tests for the full pipeline.

Runs a 2-episode mini-simulation, checks data quality, and verifies
all ML modules can instantiate and train without crashing.

Run:
    python -m pytest tests/ -v
"""

import os
import sys
import pytest
import tempfile
import shutil

# Point at project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory):
    """Isolated data directory for the test session."""
    return str(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def mini_run(tmp_data_dir):
    """
    Run a 2-episode simulation and return the CSVs.
    All subsequent tests depend on this data.
    """
    import config as C

    # Redirect output paths to tmp dir
    C.DATA_DIR             = tmp_data_dir
    C.GAME_LOG_PATH        = os.path.join(tmp_data_dir, "game_log.csv")
    C.EPISODE_STATS_PATH   = os.path.join(tmp_data_dir, "episode_stats.csv")
    C.MAZE_FEATURES_PATH   = os.path.join(tmp_data_dir, "maze_features.csv")
    C.MODELS_DIR           = os.path.join(tmp_data_dir, "models")
    C.EXPERIMENTS_DIR      = os.path.join(tmp_data_dir, "experiments")
    C.MLFLOW_TRACKING_URI  = os.path.join(tmp_data_dir, "experiments", "mlruns")

    os.makedirs(C.MODELS_DIR,      exist_ok=True)
    os.makedirs(C.EXPERIMENTS_DIR, exist_ok=True)

    # Update model paths
    C.IMITATION_MODEL_PATH     = os.path.join(C.MODELS_DIR, "imitation_mlp.pkl")
    C.CLASSIFIER_MODEL_PATH    = os.path.join(C.MODELS_DIR, "rf_classifier.pkl")
    C.FAILURE_MODEL_PATH       = os.path.join(C.MODELS_DIR, "failure_predictor.pkl")
    C.ANOMALY_MODEL_PATH       = os.path.join(C.MODELS_DIR, "anomaly_detector.pkl")
    C.MAZE_DIFFICULTY_MODEL_PATH = os.path.join(C.MODELS_DIR, "maze_difficulty.pkl")

    from ml.logger import run_simulation
    step_rows, ep_rows = run_simulation(episodes_per_combo=2, verbose=False)
    return step_rows, ep_rows


# ---------------------------------------------------------------------------
# Logger tests
# ---------------------------------------------------------------------------

class TestLogger:
    def test_step_rows_not_empty(self, mini_run):
        step_rows, _ = mini_run
        assert len(step_rows) > 0, "No step data generated"

    def test_episode_rows_not_empty(self, mini_run):
        _, ep_rows = mini_run
        assert len(ep_rows) > 0, "No episode data generated"

    def test_game_log_csv_exists(self, mini_run, tmp_data_dir):
        import config as C
        assert os.path.exists(C.GAME_LOG_PATH), "game_log.csv not written"

    def test_episode_stats_csv_exists(self, mini_run, tmp_data_dir):
        import config as C
        assert os.path.exists(C.EPISODE_STATS_PATH), "episode_stats.csv not written"

    def test_required_columns_present(self, mini_run):
        import pandas as pd
        import config as C
        from ml.features import STATE_FEATURES
        df = pd.read_csv(C.GAME_LOG_PATH)
        for col in STATE_FEATURES:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nan_in_state_features(self, mini_run):
        import pandas as pd
        import config as C
        from ml.features import STATE_FEATURES
        df = pd.read_csv(C.GAME_LOG_PATH)
        nulls = df[STATE_FEATURES].isnull().sum().sum()
        assert nulls == 0, f"Found {nulls} NaN values in state features"

    def test_binary_danger_flags(self, mini_run):
        import pandas as pd
        import config as C
        df = pd.read_csv(C.GAME_LOG_PATH)
        for col in ["danger_up","danger_down","danger_left","danger_right"]:
            assert df[col].isin([0, 1]).all(), f"{col} has non-binary values"

    def test_died_next_10_is_binary(self, mini_run):
        import pandas as pd
        import config as C
        df = pd.read_csv(C.GAME_LOG_PATH)
        assert df["died_next_10"].isin([0, 1]).all()

    def test_all_agents_present(self, mini_run):
        import pandas as pd
        import config as C
        df = pd.read_csv(C.EPISODE_STATS_PATH)
        for agent in C.AGENTS:
            assert agent in df["agent"].values, f"Agent '{agent}' missing from data"


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestState:
    def test_get_state_length(self):
        import config as C
        from core.State import SnakeState
        maze_path = os.path.join(C.MAZES_DIR, C.MAZES[0])
        if not os.path.exists(maze_path):
            pytest.skip("Maze file not found")
        state = SnakeState("red", 10, 10, 0, 1, maze_path)
        vec = state.get_state()
        assert len(vec) == 11, f"Expected 11 features, got {len(vec)}"

    def test_get_state_values_in_range(self):
        import config as C
        from core.State import SnakeState
        maze_path = os.path.join(C.MAZES_DIR, C.MAZES[0])
        if not os.path.exists(maze_path):
            pytest.skip("Maze file not found")
        state = SnakeState("red", 10, 10, 0, 1, maze_path)
        vec = state.get_state()
        for i, v in enumerate(vec[:10]):   # first 10 are 0/1 or -1/0/1
            assert v in (-1, 0, 1), f"Feature {i} out of range: {v}"
        assert 0.0 <= vec[10] <= 1.0, "dist_to_food not normalised"

    def test_get_reward_death(self):
        import config as C
        from core.State import SnakeState
        maze_path = os.path.join(C.MAZES_DIR, C.MAZES[0])
        if not os.path.exists(maze_path):
            pytest.skip("Maze file not found")
        state = SnakeState("red", 10, 10, 0, 1, maze_path)
        state.snake.isAlive = False
        reward = state.get_reward(prev_dist=5.0)
        assert reward == -100.0


# ---------------------------------------------------------------------------
# ML module smoke tests (train on tiny data, should not crash)
# ---------------------------------------------------------------------------

class TestMLModules:
    def test_features_load(self, mini_run):
        from ml.features import load_game_log, load_episode_stats
        gl = load_game_log()
        ep = load_episode_stats()
        assert gl is not None
        assert ep is not None

    def test_imitation_train(self, mini_run):
        from ml.imitation import train
        result = train(log_mlflow=False)
        assert "train_accuracy" in result
        assert 0.0 <= result["train_accuracy"] <= 1.0

    def test_classifier_train(self, mini_run):
        from ml.classifier import train
        results_df, _ = train(log_mlflow=False)
        assert len(results_df) == 3   # RF, SVM, GBM

    def test_clustering_train(self, mini_run):
        from ml.clustering import train
        df, km, artifacts = train(log_mlflow=False)
        assert "cluster" in df.columns
        assert artifacts["best_k"] >= 2

    def test_failure_predictor_train(self, mini_run):
        from ml.failure_predictor import train
        bundle = train(log_mlflow=False)
        assert "model" in bundle
        assert "roc_fpr" in bundle

    def test_anomaly_detector_train(self, mini_run):
        from ml.anomaly_detector import train
        bundle, df, drift_df = train(log_mlflow=False)
        assert "models" in bundle
        assert "is_anomaly" in df.columns

    def test_maze_difficulty_train(self, mini_run):
        from ml.maze_difficulty import train
        result = train(log_mlflow=False)
        # May return None if < 3 mazes ran — that's acceptable in smoke test
        if result is not None:
            assert "model" in result