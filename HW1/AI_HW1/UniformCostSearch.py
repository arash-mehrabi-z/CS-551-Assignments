from Node import Node
import math
from PriorityQueue import PriorityQueue
from Algorithm import Algorithm

# Do not import anything else. Use only provided imports.


class UniformCostSearch(Algorithm):
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

        item = ([self.start_node], 0)
        queue.enqueue(item, 0)

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
                if not self.validity(path): #If we reached the target node without visiting all other nodes, it is a dead end.
                    continue
                else:
                    return path
            else:
                for neighbor in last_node.connections:
                    if neighbor not in path:
                        distance = last_node.get_distance(neighbor)
                        total_distance_to_neighbor = path_cost+distance
                        new_path = path + [neighbor]
                        queue.enqueue((new_path, total_distance_to_neighbor), total_distance_to_neighbor)

        return [] #If we return empty list, it means there is no way to reach T starting from S with meeting each node only once.
