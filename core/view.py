"""
core/view.py — Tkinter game viewer.

Supports single-agent view and side-by-side comparison mode.
Original skeleton by Mirza Mubasher Baig, extended for benchmarking.
"""

import tkinter as tk
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from core import State as st


# ---------------------------------------------------------------------------
# Single-agent viewer (original, cleaned up)
# ---------------------------------------------------------------------------

class SnakeViewer:
    """View for one snake agent."""

    def __init__(self, state, SPEED=60, UnitSize=10):
        self.SPEED    = SPEED
        self.state    = state
        self.UnitSize = UnitSize
        self.top      = tk.Tk()

        self.CANVAS_WIDTH  = state.maze.WIDTH  * UnitSize
        self.CANVAS_HEIGHT = state.maze.HEIGHT * UnitSize

        self._build(self.CANVAS_WIDTH, self.CANVAS_HEIGHT, "SNAKE AI AGENT")

    def _build(self, width, height, title):
        self.top.minsize(width=width, height=height)
        self.top.title(title)
        self.canvas = tk.Canvas(
            self.top, width=width + 1, height=height + 1, bg="white"
        )
        self.canvas.pack(padx=10, pady=10)
        self.score_text = self.canvas.create_text(
            width * 0.05, height * 0.01,
            text="Score : 0",
            anchor=tk.NW,
            font=("Times", 12, "bold"),
            fill="green",
        )
        self._draw_maze(self.state.maze)
        self.UpdateView()

    def _draw_maze(self, maze):
        U = self.UnitSize
        for j in range(min(59, maze.HEIGHT)):
            for i in range(min(59, maze.WIDTH)):
                if maze.MAP[j][i] == -1:
                    self.canvas.create_rectangle(
                        i*U, j*U, (i+1)*U, (j+1)*U, fill="red", tags="wall"
                    )
        W = 6
        for i in range(1, maze.HEIGHT * U - W, W - 1):
            self.canvas.create_rectangle(0, i, W+2, i+W, fill="red", tags="wall")
            self.canvas.create_rectangle(
                maze.WIDTH*U - W+2, i, maze.WIDTH*U+2, i+W, fill="red", tags="wall"
            )
        for i in range(1, maze.WIDTH * U, W - 1):
            self.canvas.create_rectangle(i, 1, i+W, W, fill="red", tags="wall")
            self.canvas.create_rectangle(
                i, maze.HEIGHT*U-W, i+W, maze.HEIGHT*U, fill="red", tags="wall"
            )

    def _draw_snake(self, snake):
        x0 = snake.HeadPosition.X * self.UnitSize
        y0 = snake.HeadPosition.Y * self.UnitSize
        W  = x0 + self.UnitSize
        H  = y0 + self.UnitSize
        self.canvas.delete("snake")
        if snake.HeadDirection.X in (1, -1):
            W += 3
        else:
            H += 3
        self.canvas.create_oval(x0, y0, W, H, fill=snake.Color, tags="snake")

    def _draw_food(self, food):
        x0 = food.X * self.UnitSize
        y0 = food.Y * self.UnitSize
        self.canvas.delete("food")
        self.canvas.create_oval(
            x0, y0, x0 + self.UnitSize, y0 + self.UnitSize,
            fill="green", tags="food"
        )

    def UpdateView(self):
        self._draw_snake(self.state.snake)
        self._draw_food(self.state.FoodPosition)
        self.canvas.itemconfig(
            self.score_text, text=f"Score : {self.state.snake.score}"
        )
        self.top.update_idletasks()
        self.top.update()

    def ShowGameOverMessage(self, message: str):
        self.top.title(message)


# ---------------------------------------------------------------------------
# Side-by-side comparison viewer (two agents, one window)
# ---------------------------------------------------------------------------

class DualSnakeViewer:
    """
    Renders two agents side-by-side for benchmark comparison.
    Left canvas = agent_a, Right canvas = agent_b.
    """

    def __init__(self, state_a, state_b,
                 label_a: str = "Agent A", label_b: str = "Agent B",
                 SPEED: int = 30, UnitSize: int = 10):

        self.SPEED    = SPEED
        self.UnitSize = UnitSize
        self.state_a  = state_a
        self.state_b  = state_b

        W = state_a.maze.WIDTH  * UnitSize
        H = state_a.maze.HEIGHT * UnitSize

        self.top = tk.Tk()
        self.top.title(f"Benchmark: {label_a}  vs  {label_b}")
        self.top.configure(bg="#1a1a1a")

        # ── Left panel ────────────────────────────────────────────────────
        frame_a = tk.Frame(self.top, bg="#1a1a1a")
        frame_a.pack(side=tk.LEFT, padx=8, pady=8)

        tk.Label(
            frame_a, text=label_a,
            font=("Courier", 11, "bold"), fg="#4ec9b0", bg="#1a1a1a"
        ).pack()

        self.canvas_a = tk.Canvas(frame_a, width=W+1, height=H+1, bg="white")
        self.canvas_a.pack()

        self.score_a = self.canvas_a.create_text(
            W*0.05, H*0.01, text="Score: 0",
            anchor=tk.NW, font=("Times", 11, "bold"), fill="green"
        )

        # ── Divider ───────────────────────────────────────────────────────
        tk.Frame(self.top, width=2, bg="#444").pack(
            side=tk.LEFT, fill=tk.Y, padx=2
        )

        # ── Right panel ───────────────────────────────────────────────────
        frame_b = tk.Frame(self.top, bg="#1a1a1a")
        frame_b.pack(side=tk.LEFT, padx=8, pady=8)

        tk.Label(
            frame_b, text=label_b,
            font=("Courier", 11, "bold"), fg="#ce9178", bg="#1a1a1a"
        ).pack()

        self.canvas_b = tk.Canvas(frame_b, width=W+1, height=H+1, bg="white")
        self.canvas_b.pack()

        self.score_b = self.canvas_b.create_text(
            W*0.05, H*0.01, text="Score: 0",
            anchor=tk.NW, font=("Times", 11, "bold"), fill="green"
        )

        # Draw mazes
        self._draw_maze(self.canvas_a, state_a.maze)
        self._draw_maze(self.canvas_b, state_b.maze)

        self.UpdateView()

    def _draw_maze(self, canvas, maze):
        U = self.UnitSize
        for j in range(min(59, maze.HEIGHT)):
            for i in range(min(59, maze.WIDTH)):
                if maze.MAP[j][i] == -1:
                    canvas.create_rectangle(
                        i*U, j*U, (i+1)*U, (j+1)*U, fill="red", tags="wall"
                    )
        W = 6
        for i in range(1, maze.HEIGHT*U - W, W-1):
            canvas.create_rectangle(0, i, W+2, i+W, fill="red", tags="wall")
            canvas.create_rectangle(
                maze.WIDTH*U-W+2, i, maze.WIDTH*U+2, i+W, fill="red", tags="wall"
            )
        for i in range(1, maze.WIDTH*U, W-1):
            canvas.create_rectangle(i, 1, i+W, W, fill="red", tags="wall")
            canvas.create_rectangle(
                i, maze.HEIGHT*U-W, i+W, maze.HEIGHT*U, fill="red", tags="wall"
            )

    def _draw_snake(self, canvas, snake):
        x0 = snake.HeadPosition.X * self.UnitSize
        y0 = snake.HeadPosition.Y * self.UnitSize
        W  = x0 + self.UnitSize
        H  = y0 + self.UnitSize
        canvas.delete("snake")
        if snake.HeadDirection.X in (1, -1):
            W += 3
        else:
            H += 3
        canvas.create_oval(x0, y0, W, H, fill=snake.Color, tags="snake")

    def _draw_food(self, canvas, food):
        x0 = food.X * self.UnitSize
        y0 = food.Y * self.UnitSize
        canvas.delete("food")
        canvas.create_oval(
            x0, y0, x0+self.UnitSize, y0+self.UnitSize,
            fill="green", tags="food"
        )

    def UpdateView(self):
        self._draw_snake(self.canvas_a, self.state_a.snake)
        self._draw_food(self.canvas_a,  self.state_a.FoodPosition)
        self.canvas_a.itemconfig(
            self.score_a, text=f"Score: {self.state_a.snake.score}"
        )

        self._draw_snake(self.canvas_b, self.state_b.snake)
        self._draw_food(self.canvas_b,  self.state_b.FoodPosition)
        self.canvas_b.itemconfig(
            self.score_b, text=f"Score: {self.state_b.snake.score}"
        )

        self.top.update_idletasks()
        self.top.update()

    def ShowGameOverMessage(self, msg_a: str = "", msg_b: str = ""):
        self.top.title(
            f"{'DONE' if msg_a else 'Playing'}: {msg_a or '...'} | "
            f"{'DONE' if msg_b else 'Playing'}: {msg_b or '...'}"
        )