"""
ml/logger.py — Unified data collection engine.

Runs every agent on every maze for N episodes (headless — no GUI).
Writes two CSVs that every ML module reads from:
  - data/game_log.csv      : one row per step
  - data/episode_stats.csv : one row per episode

Run directly:
    python -m ml.logger
"""

import os
import sys
import csv
import time

# Make project root importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from core.State      import SnakeState
from core.agentsnake import AgentSnake


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

ACTION_TO_DIR = {
    0: (0, -1),   # Up
    3: (1,  0),   # Right
    6: (0,  1),   # Down
    9: (-1, 0),   # Left
}


def _set_direction(state: SnakeState, action: int):
    dx, dy = ACTION_TO_DIR[action]
    state.snake.HeadDirection.X = dx
    state.snake.HeadDirection.Y = dy


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(agent_name: str, maze_filename: str,
                episode_id: int, step_rows: list, episode_rows: list):
    """
    Play one full game headlessly.
    Appends step-level rows to step_rows and one episode row to episode_rows.
    Returns nothing — caller owns the lists.
    """
    maze_path = os.path.join(C.MAZES_DIR, maze_filename)

    # Clear error before the cryptic unpack crash
    if not os.path.exists(maze_path):
        raise FileNotFoundError(
            f"\nMaze file not found: {maze_path}"
            f"\nEnsure all maze .txt files are inside: {C.MAZES_DIR}"
            f"\nExpected: {C.MAZES}"
        )

    state = SnakeState(
        C.SNAKE_COLOR,
        C.SNAKE_START_X, C.SNAKE_START_Y,
        C.SNAKE_DIR_X,   C.SNAKE_DIR_Y,
        maze_path,
    )
    agent = AgentSnake(agent_name)

    step_count      = 0
    direction_changes = 0
    dead_end_entries  = 0
    foods_eaten       = 0
    dist_sum          = 0.0
    prev_action       = None
    plan_good         = True

    # We collect per-step data first, then back-fill died_next_10
    episode_steps = []

    while state.snake.isAlive and plan_good and step_count < C.MAX_STEPS_PER_EPISODE:
        score_before = state.snake.score
        plan         = agent.SearchSolution(state)

        if not plan:
            plan_good = False
            break

        for action in plan:
            if step_count >= C.MAX_STEPS_PER_EPISODE:
                break

            # Capture state BEFORE the move
            sd = state.get_state_dict()
            prev_dist = (abs(state.FoodPosition.X - state.snake.HeadPosition.X) +
                         abs(state.FoodPosition.Y - state.snake.HeadPosition.Y))

            # Direction change tracking
            if prev_action is not None and action != prev_action:
                direction_changes += 1
            prev_action = action

            # Dead-end detection (danger straight ahead, no food in that direction)
            danger_flags = state.get_danger_flags()
            dx, dy = ACTION_TO_DIR[action]
            going_into_danger = False
            if dx ==  0 and dy == -1: going_into_danger = bool(danger_flags["danger_up"])
            elif dx == 0 and dy ==  1: going_into_danger = bool(danger_flags["danger_down"])
            elif dx == -1:             going_into_danger = bool(danger_flags["danger_left"])
            elif dx ==  1:             going_into_danger = bool(danger_flags["danger_right"])
            if going_into_danger:
                dead_end_entries += 1

            # Execute move
            _set_direction(state, action)
            state.snake.moveSnake(state)
            step_count += 1
            dist_sum   += prev_dist

            row = {
                "episode_id":    episode_id,
                "agent":         agent_name,
                "maze":          maze_filename,
                "step":          step_count,
                **sd,
                "score":         state.snake.score,
                "action":        action,
                "died_next_10":  0,  # placeholder — back-filled below
            }
            episode_steps.append(row)

            if not state.snake.isAlive:
                break

        score_after = state.snake.score
        if score_after > score_before:
            foods_eaten += 1
            state.generateFood()

        if score_after == score_before and state.snake.isAlive:
            plan_good = False

    # Back-fill died_next_10: mark steps within 10 of death as 1
    died = not state.snake.isAlive
    if died:
        death_step = step_count
        for row in episode_steps:
            if death_step - row["step"] <= C.FAILURE_LOOKAHEAD_STEPS:
                row["died_next_10"] = 1

    step_rows.extend(episode_steps)

    # Episode summary
    max_possible = foods_eaten * 10 if foods_eaten > 0 else 1
    episode_rows.append({
        "episode_id":       episode_id,
        "agent":            agent_name,
        "maze":             maze_filename,
        "total_steps":      step_count,
        "final_score":      state.snake.score,
        "foods_eaten":      foods_eaten,
        "died":             1 if died else 0,
        "avg_dist_to_food": round(dist_sum / max(step_count, 1), 4),
        "direction_changes": direction_changes,
        "dead_end_entries": dead_end_entries,
        "path_optimality":  round(state.snake.score / max_possible, 4),
        "steps_per_food":   round(step_count / max(foods_eaten, 1), 2),
    })


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(episodes_per_combo: int = None, verbose: bool = True):
    """
    Run all agents × all mazes × N episodes.
    Writes game_log.csv and episode_stats.csv.
    """
    n = episodes_per_combo or C.EPISODES_PER_AGENT_MAZE

    step_rows    = []
    episode_rows = []
    episode_id   = 0
    total_combos = len(C.AGENTS) * len(C.MAZES) * n

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Neuro-Sentinel Snake — Data Collection")
        print(f"  Agents : {C.AGENTS}")
        print(f"  Mazes  : {C.MAZES}")
        print(f"  Episodes per combo : {n}")
        print(f"  Total episodes     : {total_combos}")
        print(f"{'='*60}\n")

    t_start = time.time()

    for agent_name in C.AGENTS:
        for maze_file in C.MAZES:
            maze_path = os.path.join(C.MAZES_DIR, maze_file)
            if not os.path.exists(maze_path):
                print(f"  [SKIP] Maze not found: {maze_path}")
                continue

            for ep in range(n):
                episode_id += 1
                try:
                    run_episode(agent_name, maze_file,
                                episode_id, step_rows, episode_rows)
                except Exception as exc:
                    print(f"  [ERROR] ep {episode_id} "
                          f"({agent_name}/{maze_file}): {exc}")

            if verbose:
                done = episode_id
                pct  = done / total_combos * 100
                elapsed = time.time() - t_start
                print(f"  [{pct:5.1f}%] {agent_name:20s} {maze_file:25s} "
                      f"— {done}/{total_combos} eps | {elapsed:.1f}s")

    # Write game_log.csv
    with open(C.GAME_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=C.GAME_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(step_rows)

    # Write episode_stats.csv
    with open(C.EPISODE_STATS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=C.EPISODE_STATS_COLUMNS)
        writer.writeheader()
        writer.writerows(episode_rows)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Steps logged   : {len(step_rows):,}")
        print(f"  Episodes logged: {len(episode_rows):,}")
        print(f"  → {C.GAME_LOG_PATH}")
        print(f"  → {C.EPISODE_STATS_PATH}\n")

    return step_rows, episode_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_simulation()