# config.py —> Centralized configuration for the Neuro-Sentinel Snake project.

import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MAZES_DIR       = os.path.join(BASE_DIR, "mazes")
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

# Output CSVs (written by logger.py, read by every ML module)
GAME_LOG_PATH      = os.path.join(DATA_DIR, "game_log.csv")
GAME_LOG_ZIP_PATH  = os.path.join(DATA_DIR, "game_log.csv.zip")
EPISODE_STATS_PATH = os.path.join(DATA_DIR, "episode_stats.csv")
MAZE_FEATURES_PATH = os.path.join(DATA_DIR, "maze_features.csv")

# Ensure directories exist at import time so no module needs to mkdir itself
for _dir in [DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

AGENTS = ["AStar", "GreedyBestFirst", "BreadthFirst"]

MAZES = [
    "Maze1_easy.txt",
    "Maze2_medium.txt",
    "Maze3_medium.txt",
    "Maze4_hard.txt",
    "Maze5_dense.txt",
]

EPISODES_PER_AGENT_MAZE = 200   # games each agent plays on each maze
MAX_STEPS_PER_EPISODE   = 2000  # safety cap avoids infinite loops
SNAKE_START_X           = 10
SNAKE_START_Y           = 10
SNAKE_DIR_X             = 0
SNAKE_DIR_Y             = 1
SNAKE_COLOR             = "red"

# Columns written to game_log.csv for every step
GAME_LOG_COLUMNS = [
    "episode_id",       # unique int across entire run
    "agent",            # AStar | GreedyBestFirst | BreadthFirst
    "maze",             # filename
    "step",             # step index within episode
    "head_x", "head_y",
    "food_x", "food_y",
    "dir_x", "dir_y",
    "danger_up", "danger_down", "danger_left", "danger_right",
    "food_up", "food_down", "food_left", "food_right",
    "dist_to_food",     # Manhattan distance
    "score",
    "action",           # 0=Up 3=Right 6=Down 9=Left
    "died_next_10",     # 1 if snake died within 10 steps — label for Opt 5
]

# Columns written to episode_stats.csv for every episode (aggregated features)
EPISODE_STATS_COLUMNS = [
    "episode_id",
    "agent",
    "maze",
    "total_steps",
    "final_score",
    "foods_eaten",
    "died",             # 1 = hit wall/out of bounds, 0 = ran out of plan
    "avg_dist_to_food",
    "direction_changes", # how often direction changed behaviour feature
    "dead_end_entries",  # steps where danger_straight=1 and no food ahead
    "path_optimality",   # final_score / max_possible_score proxy
    "steps_per_food",    # total_steps / max(foods_eaten,1)
]

IMITATION_TEST_MAZES   = ["Maze4_hard.txt", "Maze5_dense.txt"]  # held-out mazes
IMITATION_TRAIN_AGENT  = "AStar"        # we imitate A*
IMITATION_HIDDEN_SIZES = (128, 64)      # MLP hidden layer sizes
IMITATION_MAX_ITER     = 500
IMITATION_RANDOM_STATE = 42
IMITATION_MODEL_PATH   = os.path.join(MODELS_DIR, "imitation_mlp.pkl")

CLASSIFIER_MODELS = ["RandomForest", "GradientBoosting", "LogisticRegression"]
CLASSIFIER_CV_FOLDS     = 3             # 3-fold is sufficient; 5-fold triples the time
CLASSIFIER_RANDOM_STATE = 42
CLASSIFIER_MODEL_PATH   = os.path.join(MODELS_DIR, "rf_classifier.pkl")

# Use a subset of the data for faster training and tuning 50k rows is enough to get good performance
CLASSIFIER_SAMPLE_SIZE  = 50_000
FAILURE_SAMPLE_SIZE     = 80_000       # failure predictor also sampled

# Lean param grids one or two values per hyperparameter is enough
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth":    [10, 20],
}

GBM_PARAM_GRID = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth":     [3, 5],
}

LR_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
}


MAZE_DIFFICULTY_MODEL_PATH = os.path.join(MODELS_DIR, "maze_difficulty.pkl")

CLUSTERING_K_RANGE      = range(2, 8)   # k values to try for elbow method
CLUSTERING_RANDOM_STATE = 42
TSNE_PERPLEXITY         = 15
TSNE_RANDOM_STATE       = 42


FAILURE_LOOKAHEAD_STEPS   = 10          # "will die in next N steps"
FAILURE_TEST_SIZE         = 0.2
FAILURE_RANDOM_STATE      = 42
FAILURE_ALERT_THRESHOLD   = 0.70        # probability above which we show alert
FAILURE_MODEL_PATH        = os.path.join(MODELS_DIR, "failure_predictor.pkl")


ANOMALY_CONTAMINATION  = 0.05          # expected fraction of anomalous episodes
ANOMALY_RANDOM_STATE   = 42
ANOMALY_MODEL_PATH     = os.path.join(MODELS_DIR, "anomaly_detector.pkl")
ANOMALY_DRIFT_WINDOW   = 10            # rolling window for drift detection
ANOMALY_DRIFT_DROP     = 0.30          # 30% score drop triggers alert

# On Windows, os.path.join gives "D:\\path" which MLflow reads as scheme "D".
# Prefixing with file:/// fixes this on all platforms.
_mlflow_local_path  = os.path.join(EXPERIMENTS_DIR, "mlflow_runs")
MLFLOW_TRACKING_URI = "file:///" + _mlflow_local_path.replace("\\", "/")
MLFLOW_EXPERIMENT_NAME = "neuro-sentinel-snake"


DASHBOARD_TITLE         = "Neuro-Sentinel Snake — ML Analytics Dashboard"
DASHBOARD_REFRESH_SECS  = 5


SNAKE_SPEED             = 30
UNIT_SIZE               = 10