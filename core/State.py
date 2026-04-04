"""
core/State.py — Game state, snake, and maze.

Original skeleton by Mirza Mubasher Baig, NUCES:FAST Lahore.
Extended with get_state() and get_reward() for ML data collection.
"""

import random


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class Const:
    UNIT_SIZE = 10


class Vector:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def show(self):
        print("[", self.X, ", ", self.Y, "]")

    def Update(self, X, Y):
        self.X = X
        self.Y = Y

    def Add(self, Vec):
        self.X = self.X + Vec.X
        self.Y = self.Y + Vec.Y


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

class Maze:
    def __init__(self, PuzzleFileName):
        self.MAP = []
        self.LoadMaze(PuzzleFileName)

    def LoadMaze(self, filename):
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            # Skip blank lines at the top (handles CRLF/LF differences on Windows)
            header = ''
            for raw in f:
                stripped = raw.strip()
                if stripped:
                    header = stripped
                    break

            parts = header.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Maze file '{filename}' has a malformed header line: {repr(header)}\n"
                    f"Expected format: '<HEIGHT> <WIDTH>' on the first non-blank line."
                )
            self.HEIGHT, self.WIDTH = int(parts[0]), int(parts[1])

            self.MAP = []
            for raw in f:
                stripped = raw.strip()
                if not stripped:
                    continue   # skip blank lines within the map
                row = [int(d) for d in stripped.split()]
                self.MAP.append(row)

        # Sanity check
        if len(self.MAP) != self.HEIGHT:
            raise ValueError(
                f"Maze '{filename}': expected {self.HEIGHT} rows, got {len(self.MAP)}. "
                f"File may be truncated or corrupted."
            )


# ---------------------------------------------------------------------------
# Snake
# ---------------------------------------------------------------------------

class Snake:
    def __init__(self, Color, HeadPositionX=10, HeadPositionY=10,
                 HeadDirectionX=0, HeadDirectionY=1):
        self.Size           = 1
        self.Body           = []
        self.Color          = Color
        self.HeadPosition   = Vector(HeadPositionX, HeadPositionY)
        self.HeadDirection  = Vector(HeadDirectionX, HeadDirectionY)
        self.score          = 0
        self.isAlive        = True
        self._prev_score    = 0   # used by get_reward()

    def moveSnake(self, State):
        if not self.isAlive:
            return

        self.HeadPosition.Add(self.HeadDirection)
        r = self.HeadPosition.Y
        c = self.HeadPosition.X

        if (r >= State.maze.HEIGHT or r < 0) or (c >= State.maze.WIDTH or c < 0):
            self.isAlive = False
        elif State.maze.MAP[r][c] == -1:
            self.isAlive = False
        elif c == State.FoodPosition.X and r == State.FoodPosition.Y:
            self.score += 10


# ---------------------------------------------------------------------------
# SnakeState — the main environment object
# ---------------------------------------------------------------------------

class SnakeState:
    def __init__(self, Color, HeadPositionX, HeadPositionY,
                 HeadDirectionX, HeadDirectionY, mazeFileName):
        self.snake = Snake(Color, HeadPositionX, HeadPositionY,
                           HeadDirectionX, HeadDirectionY)
        self.maze  = Maze(mazeFileName)
        self.generateFood()
        self._step_count = 0

    # ------------------------------------------------------------------
    # Core helpers (unchanged from original)
    # ------------------------------------------------------------------

    def getAdjacentNodes(self, node):
        """Return walkable neighbours of a grid cell."""
        x, y       = node
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbours = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.maze.WIDTH - 1 and
                    0 <= ny < self.maze.HEIGHT - 1 and
                    self.maze.MAP[ny][nx] != -1):
                neighbours.append((nx, ny))
        return neighbours

    def generateFood(self):
        placed = False
        while not placed:
            x = random.randrange(3, self.maze.WIDTH - 3)
            y = random.randrange(3, self.maze.HEIGHT - 3)
            if self.maze.MAP[y][x] != -1:
                placed = True
        self.FoodPosition = Vector(x, y)

    # ------------------------------------------------------------------
    # NEW: get_state()
    # Returns an 11-element feature vector describing the current moment.
    # Used by every ML module as the canonical state representation.
    #
    # Features:
    #   [0]  danger_up    — 1 if moving up leads to wall/boundary
    #   [1]  danger_down
    #   [2]  danger_left
    #   [3]  danger_right
    #   [4]  food_up      — 1 if food is above head
    #   [5]  food_down
    #   [6]  food_left
    #   [7]  food_right
    #   [8]  dir_x        — current head direction X (-1, 0, 1)
    #   [9]  dir_y        — current head direction Y (-1, 0, 1)
    #   [10] dist_to_food — normalised Manhattan distance (0–1)
    # ------------------------------------------------------------------

    def get_state(self):
        hx = self.snake.HeadPosition.X
        hy = self.snake.HeadPosition.Y
        fx = self.FoodPosition.X
        fy = self.FoodPosition.Y
        W  = self.maze.WIDTH
        H  = self.maze.HEIGHT

        def is_blocked(x, y):
            if x < 0 or x >= W or y < 0 or y >= H:
                return 1
            if self.maze.MAP[y][x] == -1:
                return 1
            return 0

        danger_up    = is_blocked(hx,     hy - 1)
        danger_down  = is_blocked(hx,     hy + 1)
        danger_left  = is_blocked(hx - 1, hy)
        danger_right = is_blocked(hx + 1, hy)

        food_up    = 1 if fy < hy else 0
        food_down  = 1 if fy > hy else 0
        food_left  = 1 if fx < hx else 0
        food_right = 1 if fx > hx else 0

        max_dist   = W + H
        dist_norm  = (abs(fx - hx) + abs(fy - hy)) / max_dist if max_dist > 0 else 0

        return [
            danger_up, danger_down, danger_left, danger_right,
            food_up, food_down, food_left, food_right,
            self.snake.HeadDirection.X,
            self.snake.HeadDirection.Y,
            dist_norm,
        ]

    # ------------------------------------------------------------------
    # NEW: get_state_dict()
    # Same information as get_state() but as a named dict for CSV logging.
    # ------------------------------------------------------------------

    def get_state_dict(self):
        hx = self.snake.HeadPosition.X
        hy = self.snake.HeadPosition.Y
        fx = self.FoodPosition.X
        fy = self.FoodPosition.Y
        W  = self.maze.WIDTH
        H  = self.maze.HEIGHT

        def is_blocked(x, y):
            if x < 0 or x >= W or y < 0 or y >= H:
                return 1
            if self.maze.MAP[y][x] == -1:
                return 1
            return 0

        max_dist  = W + H
        dist_raw  = abs(fx - hx) + abs(fy - hy)
        dist_norm = dist_raw / max_dist if max_dist > 0 else 0

        return {
            "head_x":       hx,
            "head_y":       hy,
            "food_x":       fx,
            "food_y":       fy,
            "dir_x":        self.snake.HeadDirection.X,
            "dir_y":        self.snake.HeadDirection.Y,
            "danger_up":    is_blocked(hx,     hy - 1),
            "danger_down":  is_blocked(hx,     hy + 1),
            "danger_left":  is_blocked(hx - 1, hy),
            "danger_right": is_blocked(hx + 1, hy),
            "food_up":      1 if fy < hy else 0,
            "food_down":    1 if fy > hy else 0,
            "food_left":    1 if fx < hx else 0,
            "food_right":   1 if fx > hx else 0,
            "dist_to_food": dist_norm,
        }

    # ------------------------------------------------------------------
    # NEW: get_reward()
    # Reward signal used by logger for labelling steps.
    #   +10  ate food
    #   -100 died
    #   -1   each step (time penalty encourages efficiency)
    #   +1   moved closer to food vs previous step
    # ------------------------------------------------------------------

    def get_reward(self, prev_dist: float) -> float:
        if not self.snake.isAlive:
            return -100.0

        score_delta = self.snake.score - self.snake._prev_score
        if score_delta > 0:
            self.snake._prev_score = self.snake.score
            return 10.0

        hx       = self.snake.HeadPosition.X
        hy       = self.snake.HeadPosition.Y
        cur_dist = abs(self.FoodPosition.X - hx) + abs(self.FoodPosition.Y - hy)
        closer   = 1.0 if cur_dist < prev_dist else 0.0

        return -1.0 + closer

    # ------------------------------------------------------------------
    # NEW: get_danger_flags()
    # Convenience wrapper used by logger for the four danger columns.
    # ------------------------------------------------------------------

    def get_danger_flags(self):
        hx = self.snake.HeadPosition.X
        hy = self.snake.HeadPosition.Y
        W  = self.maze.WIDTH
        H  = self.maze.HEIGHT

        def is_blocked(x, y):
            if x < 0 or x >= W or y < 0 or y >= H:
                return 1
            return 1 if self.maze.MAP[y][x] == -1 else 0

        return {
            "danger_up":    is_blocked(hx,     hy - 1),
            "danger_down":  is_blocked(hx,     hy + 1),
            "danger_left":  is_blocked(hx - 1, hy),
            "danger_right": is_blocked(hx + 1, hy),
        }