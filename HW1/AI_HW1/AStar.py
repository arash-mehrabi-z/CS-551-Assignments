from Node import Node
import math
from PriorityQueue import PriorityQueue
from Algorithm import Algorithm

# Do not import anything else. Use only provided imports.


class AStar(Algorithm):
    """
        You should implement the solve method. You can also create new methods inside the class.
    """
    def __init__(self, number_of_nodes: int, start_node: Node, target_node: Node):
        super().__init__(number_of_nodes, start_node, target_node)

    def solve(self) -> list:
        """
            Implement A* algorithm here to solve the problem. You must return the complete path, not the cost.
            self.iteration must be equal number of iteration. Do not forget to update it!
        :return: The path which is a list of nodes.
        """
        # TODO: You should implement inside of this method!
        queue = PriorityQueue()
        visited = []

        heuristic = self.start_node.get_estimated_distance(self.target_node)
        total_estimated_cost = 0 + heuristic
        item = ([self.start_node], 0)
        queue.enqueue(item, total_estimated_cost)

        while len(queue) > 0:
            item = queue.dequeue()
            path, path_cost = item

            if path in visited:
                continue
            else:
                visited.append(path)

            self.iteration += 1
            last_node = path[-1]

            if last_node == self.target_node:
                if not self.validity(path):  # If we reached the target node without visiting all other nodes, it is a dead end.
                    continue
                else:
                    return path
            else:
                for neighbor in last_node.connections:
                    if not neighbor in path:
                        distance_to_neighbor = last_node.get_distance(neighbor)
                        new_path_cost = path_cost + distance_to_neighbor
                        new_path = path + [neighbor]

                        heuristic = neighbor.get_estimated_distance(self.target_node)
                        queue.enqueue((new_path, new_path_cost), new_path_cost + heuristic)

        return []  # If we return empty list, it means there is no way to reach T starting from S with meeting each node only once.
