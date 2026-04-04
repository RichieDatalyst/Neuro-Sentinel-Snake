# Snake AI Agent: Unified Search Interface
# Provides a single AgentSnake class that can use different search algorithms.
from core.astar          import AStar
from core.breadthfirst   import BreadthFirst
from core.greedybestfirst import GreedyBestFirst


ALGORITHM_REGISTRY = {
    "AStar":          AStar,
    "BreadthFirst":   BreadthFirst,
    "GreedyBestFirst": GreedyBestFirst,
}


class AgentSnake:
    """Unified interface for all search-based snake agents."""

    def __init__(self, algorithm: str = "AStar"):
        if algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
        self.algorithm_name = algorithm
        self._cls           = ALGORITHM_REGISTRY[algorithm]

    def SearchSolution(self, state) -> list:
        """Return a list of action codes for the current state."""
        return self._cls(state).search()

    @staticmethod
    def available_agents() -> list:
        return list(ALGORITHM_REGISTRY.keys())