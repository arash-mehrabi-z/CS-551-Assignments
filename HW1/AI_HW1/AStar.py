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
                if not self.validity(
                        path):  # If we reached the target node without visiting all other nodes, it is a dead end.
                    continue
                else:
                    return path
            else:
                for neighbor in last_node.connections:
                    if not neighbor in path:
                        distance_to_neighbor = last_node.get_distance(neighbor)
                        new_path_cost = path_cost + distance_to_neighbor
                        new_path = path + [neighbor]

                        # ### Heuristic 1 ###
                        # heuristic = neighbor.get_estimated_distance(self.target_node)

                        # ### Heuristic 2 ###
                        # heuristic = 0
                        # if neighbor != self.target_node: #If the neighbor is the goal node, we don't need to use the Dijkstra alg.
                        #     dist = self.dijkstra(neighbor)
                        #     heuristic = dist[self.target_node]

                        ### Heuristic 3 ###
                        heuristic = self.lazy_prims(neighbor, path)

                        queue.enqueue((new_path, new_path_cost), new_path_cost + heuristic)

        return []  # If we return empty list, it means there is no way to reach T starting from S with meeting each node only once.

    def dijkstra(self, start_node):
        """
        returns a dictionary that contains the shortest distance to every node from start_node.
        @param start_node:Node
        @return:dictionary
        """
        visited = []
        dist = dict()
        dist[start_node] = 0
        pq = PriorityQueue()

        pq.enqueue(start_node, 0)

        while len(pq) > 0:
            current_node = pq.dequeue()
            visited.append(current_node)
            for neighbor in current_node.connections:
                if neighbor in visited:  # We already find the shortest path from start_node to this neighbor.
                    continue
                total_distance_to_neighbor = dist[current_node] + current_node.get_distance(neighbor)
                if (neighbor not in dist) or (total_distance_to_neighbor < dist[neighbor]):
                    dist[neighbor] = total_distance_to_neighbor
                    pq.enqueue(neighbor, total_distance_to_neighbor)

        return dist

    # ### Third Heuristic ###
    def lazy_prims(self, start_node, vis):
        visited = vis[:]  # Make a copy to ensure we don't change the original data
        mst_expected_num_edges = self.number_of_nodes - len(visited) - 1
        mst_cost, edge_count = 0, 0
        pq = PriorityQueue()
        for neighbor in start_node.connections:
            if neighbor not in visited:
                edge_cost = start_node.get_distance(neighbor)
                pq.enqueue((neighbor, edge_cost), edge_cost)

        while len(pq) > 0 and edge_count < mst_expected_num_edges:
            current_node, edge_cost = pq.dequeue()

            if current_node in visited:
                continue

            mst_cost += edge_cost
            edge_count += 1
            visited.append(current_node)

            for neighbor in current_node.connections:
                if neighbor not in visited:
                    edge_cost = current_node.get_distance(neighbor)
                    pq.enqueue((neighbor, edge_cost), edge_cost)

        # no_mst_exists = None
        # if edge_count != mst_expected_num_edges:
        #     return no_mst_exists

        return mst_cost
