"""
main.py — Neuro-Sentinel Snake: entry point

Usage:
    python main.py --mode simulate          # collect game data (headless, fast)
    python main.py --mode train             # run all ML modules in order
    python main.py --mode analyze           # XAI + reports (after train)
    python main.py --mode dashboard         # launch Streamlit dashboard
    python main.py --mode play              # Tkinter gameplay (A* default)
    python main.py --mode play --agent BreadthFirst
    python main.py --mode simulate --episodes 50   # quick smoke test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

def mode_simulate(episodes: int):
    from ml.logger import run_simulation
    run_simulation(episodes_per_combo=episodes, verbose=True)


def mode_train():
    print("\n" + "="*60)
    print("  Running full ML training pipeline")
    print("="*60)

    # 1. Feature engineering check
    from ml.features import load_game_log
    try:
        df = load_game_log()
        print(f"\n  Game log loaded: {len(df):,} rows")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    # 2. Option 1: Imitation Learning
    print("\n[1/5] Imitation Learning...")
    from ml.imitation import train as train_imitation
    train_imitation(log_mlflow=True)

    # 3. Option 2: Classical Classifiers
    print("\n[2/5] Classical ML Classifiers...")
    from ml.classifier import train as train_classifier
    train_classifier(log_mlflow=True)

    # 4. Option 3: Maze Difficulty
    print("\n[3/5] Maze Difficulty Regression...")
    from ml.maze_difficulty import train as train_maze
    train_maze(log_mlflow=True)

    # 5. Option 4: Clustering
    print("\n[4/5] Behaviour Clustering...")
    from ml.clustering import train as train_clustering
    train_clustering(log_mlflow=True)

    # 6. Option 5: Failure Prediction
    print("\n[5/5] Failure Prediction...")
    from ml.failure_predictor import train as train_failure
    train_failure(log_mlflow=True)

    # 7. Option 6: Anomaly Detection (runs on episode data produced above)
    print("\n[+] Anomaly Detection...")
    from ml.anomaly_detector import train as train_anomaly
    train_anomaly(log_mlflow=True)

    print("\n" + "="*60)
    print("  Training pipeline complete.")
    print(f"  Models saved to: {C.MODELS_DIR}")
    print(f"  Data saved to  : {C.DATA_DIR}")
    print("="*60)


def mode_analyze():
    print("\n[XAI] Running SHAP explainability analysis...")
    from ml.xai import run_all
    run_all()


def mode_dashboard():
    import subprocess
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py"
    )
    print(f"\n  Launching Streamlit dashboard: {dashboard_path}")
    print("  Open your browser at http://localhost:8501\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path],
        check=True,
    )


def mode_play(agent_name: str):
    import tkinter as tk
    from core.State     import SnakeState
    from core.agentsnake import AgentSnake
    from core.view      import SnakeViewer
    import threading
    import time

    maze_path = os.path.join(C.MAZES_DIR, C.MAZES[3])  # Maze2_medium by default
    state  = SnakeState(
        C.SNAKE_COLOR,
        C.SNAKE_START_X, C.SNAKE_START_Y,
        C.SNAKE_DIR_X,   C.SNAKE_DIR_Y,
        maze_path,
    )
    agent  = AgentSnake(agent_name)
    viewer = SnakeViewer(state, SPEED=C.SNAKE_SPEED)

    ACTION_TO_DIR = {0:(0,-1), 3:(1,0), 6:(0,1), 9:(-1,0)}

    def game_loop():
        plan_good = True
        while state.snake.isAlive and plan_good:
            score_before = state.snake.score
            plan = agent.SearchSolution(state)
            if not plan:
                plan_good = False
                break
            for action in plan:
                dx, dy = ACTION_TO_DIR[action]
                state.snake.HeadDirection.X = dx
                state.snake.HeadDirection.Y = dy
                state.snake.moveSnake(state)
                if not state.snake.isAlive:
                    break
                time.sleep(1 / C.SNAKE_SPEED)
                viewer.UpdateView()
            if state.snake.score > score_before:
                state.generateFood()
            elif state.snake.isAlive:
                plan_good = False

        msg = "Game Over"
        msg += " — bad plan" if state.snake.isAlive else " — hit wall"
        viewer.ShowGameOverMessage(msg)

    t = threading.Thread(target=game_loop, daemon=True)
    t.start()
    viewer.top.mainloop()


def parse_args():
    p = argparse.ArgumentParser(
        description="Neuro-Sentinel Snake — ML Pathfinding Analysis Framework"
    )
    p.add_argument(
        "--mode", required=True,
        choices=["simulate", "train", "analyze", "dashboard", "play"],
        help="Execution mode",
    )
    p.add_argument(
        "--agent", default="AStar",
        choices=["AStar", "BreadthFirst", "GreedyBestFirst"],
        help="Agent for play mode",
    )
    p.add_argument(
        "--episodes", type=int, default=None,
        help="Override episodes per combo in simulate mode",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "simulate":
        mode_simulate(args.episodes)
    elif args.mode == "train":
        mode_train()
    elif args.mode == "analyze":
        mode_analyze()
    elif args.mode == "dashboard":
        mode_dashboard()
    elif args.mode == "play":
        mode_play(args.agent)


if __name__ == "__main__":
    main()