from Node import Node
from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
        Algorithm class is an abstract class. You should implement the subclasses which are AStar and UniformCostSearch.
        You can use the variables and methods of this class while implementing the subclasses.
    """
    number_of_nodes: int    # Number of total nodes
    start_node: Node        # Initial node. Your path will start from start_node.
    target_node: Node       # End node. Your path's last node must be target_node.
    iteration: int          # Number of iteration.

    def __init__(self, number_of_nodes: int, start_node: Node, target_node: Node):
        """
            Constructor of Algorithm class. Do not change anything.
        :param number_of_nodes: Number of nodes, int.
        :param start_node: Initial node.
        :param target_node: Target node.
        """
        self.number_of_nodes = number_of_nodes
        self.start_node = start_node
        self.target_node = target_node
        self.iteration = 0

    @abstractmethod
    def solve(self) -> list:
        """
            This is an abstract method, you should implement the solve method in the subclasses.
        :return: A path which is a list of nodes.
        """
        pass

    def calculate_cost(self, path: list) -> int:
        """
            This method calculates and returns the total cost of a given path. Do not change it.
        :param path: List of nodes.
        :return: Total cost of the given path.
        """
        cost = 0

        for i in range(1, len(path)):
            cost += path[i - 1].get_distance(path[i])

        return cost

    def validity(self, path: list) -> bool:
        """
            Check whether the given path is valid or not. Do not change it.
        :param path: List of nodes.
        :return: Validity of the given path.
        """
        if len(path) != self.number_of_nodes:  # Path length must be number_of_nodes
            return False

        if path[0] != self.start_node:  # Path must be start from start_node
            return False

        if path[-1] != self.target_node:  # The end of path must be target_node
            return False

        if len(list(set(path))) != len(path): # All nodes must be unique
            return False

        for i in range(len(path) - 1):  # Check all connections.
            if path[i + 1] not in path[i].connections:
                return False

        return True
