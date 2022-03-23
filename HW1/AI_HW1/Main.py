from Node import Node
from DataReader import read_data
from AStar import AStar
from UniformCostSearch import UniformCostSearch
import time


DATA_PATH = "data1.json"  # The data file path. You can edit this.


def run(alg):
    """
        Do not change it.
    """
    start_time = time.time_ns()
    path = alg.solve()
    end_time = time.time_ns()

    print("Path:\t", path)
    print("Cost:\t", alg.calculate_cost(path))
    print("Validity:\t", alg.validity(path))
    print("Iter:\t", alg.iteration)
    print("Elapsed Time (sec):\t%.3f" % ((end_time - start_time) / 1000000000))


if __name__ == "__main__":
    """
        Run this provided main. Do not change!
    """
    nodes = read_data(DATA_PATH)

    start_node = nodes["S"]
    target_node = nodes["T"]

    print("Number of nodes:\t", len(nodes))

    print("A*:")

    run(AStar(len(nodes), start_node, target_node))

    print("-" * 50)

    print("Uniform Cost Search:")

    run(UniformCostSearch(len(nodes), start_node, target_node))
