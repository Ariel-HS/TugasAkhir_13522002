import numpy as np

class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.degree_sum = 0.0
        self.nset = []  # List of neighbor_id

        self.color = 0

class NNgraph:
    def __init__(self, size):
        self.size = size
        self.nodes = [Node(i) for i in range(size)]
        self.total_degree = 0.0
        self.adj_matrix = np.zeros((size, size), dtype=float)
        self.dist_matrix = np.full((size, size), np.inf)