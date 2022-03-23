import math


class Node:
    """
        This class contains node info (name, coordinates, connected nodes, distances).
        You can use this class but cannot change!
    """
    name: str           # Name of Node
    connections: list   # Connected nodes list.
    x: int              # X coordinate of the node.
    y: int              # Y coordinate of the node.
    __distances: dict   # Do not use this!

    def __init__(self, node_info: dict):
        """
            Do not use this method!
        """
        self.name = node_info["name"]
        self.x = int(node_info["x"])
        self.y = int(node_info["y"])
        self.__distances = {connection["node"]: int(connection["distance"]) for connection in node_info["connections"]}
        self.connections = []

    def compute_connections(self, nodes: dict):
        """
            Do not call this method!
        """

        self.connections = [nodes[node_name] for node_name in self.__distances.keys()]

    def get_distance(self, target) -> int:
        """
            Use this method to get real distance between two connected nodes.
        :param target: Target Node
        :return: Real distance between current and target nodes.
        If there is no connection from current to target node, it returns -1
        """
        return self.__distances.get(str(target), -1)

    def get_estimated_distance(self, target) -> float:
        """
            Use this method to get estimated distance between two connected nodes.
        :param target: Target Node
        :return: Real distance between current and target nodes.
        """
        return math.sqrt(math.pow(self.x - target.x, 2.) + math.pow(self.y - target.y, 2.))

    def __eq__(self, other) -> bool:
        """
            Use this method to check if two nodes are same or not.
        :param other: A node or name of the node
        :return: Nodes are same or not.
        """
        return self.name == str(other)

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return self.name.__hash__()
