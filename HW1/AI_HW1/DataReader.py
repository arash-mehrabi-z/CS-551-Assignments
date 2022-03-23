import json
from Node import Node


def read_data(file_name: str = "data.json"):
    """
        Do not call or change this method!
    """
    with open(file_name, "r") as f:
        data = json.load(f)

    nodes = {data["nodes"][i]["name"]: Node(data["nodes"][i]) for i in range(data["numberOfNodes"])}

    for node in nodes.values():
        node.compute_connections(nodes)

    return nodes
