"""BFS search agent."""

from collections import deque


class BreadthFirst:
    def __init__(self, state):
        self.state = state

    def search(self):
        start = (self.state.snake.HeadPosition.X, self.state.snake.HeadPosition.Y)
        goal  = (self.state.FoodPosition.X,        self.state.FoodPosition.Y)

        frontier  = deque([start])
        came_from = {start: None}

        while frontier:
            current = frontier.popleft()
            if current == goal:
                break
            for nxt in self.state.getAdjacentNodes(current):
                if nxt not in came_from:
                    came_from[nxt] = current
                    frontier.append(nxt)

        if goal not in came_from:
            return []

        return self._path_to_actions(self._reconstruct(came_from, start, goal))

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