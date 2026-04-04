"""core/astar.py — A* search agent (unchanged algorithm, clean package)."""

import heapq


class AStar:
    def __init__(self, state):
        self.state = state

    def search(self):
        start = (self.state.snake.HeadPosition.X, self.state.snake.HeadPosition.Y)
        goal  = (self.state.FoodPosition.X,        self.state.FoodPosition.Y)

        frontier    = [(0, start)]
        came_from   = {}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in self.state.getAdjacentNodes(current):
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self._heuristic(goal, nxt)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return []

        return self._path_to_actions(self._reconstruct(came_from, start, goal))

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct(self, came_from, start, goal):
        path, cur = [], goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]
        path.append(start)
        path.reverse()
        return path

    def _path_to_actions(self, path):
        actions = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            if   dx ==  1: actions.append(3)   # Right
            elif dx == -1: actions.append(9)   # Left
            elif dy ==  1: actions.append(6)   # Down
            elif dy == -1: actions.append(0)   # Up
        return actions