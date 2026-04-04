"""core/greedybestfirst.py — Greedy Best-First search agent."""

import heapq


class GreedyBestFirst:
    def __init__(self, state):
        self.state = state

    def search(self):
        start = (self.state.snake.HeadPosition.X, self.state.snake.HeadPosition.Y)
        goal  = (self.state.FoodPosition.X,        self.state.FoodPosition.Y)

        frontier  = [(self._heuristic(start, goal), start)]
        came_from = {start: None}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in self.state.getAdjacentNodes(current):
                if nxt not in came_from:
                    came_from[nxt] = current
                    heapq.heappush(frontier, (self._heuristic(nxt, goal), nxt))

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
            if   dx ==  1: actions.append(3)
            elif dx == -1: actions.append(9)
            elif dy ==  1: actions.append(6)
            elif dy == -1: actions.append(0)
        return actions