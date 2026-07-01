import numpy as np
import copy

class Clustering:
    # Clustering is already initialized with random partition
    
    def __init__(self, N, K):
        # self.count 
        self.N = N # node count
        self.K = K # cluster count
        self.part = np.random.randint(0, K, size=N) # size of N, part[i] = cluster of node-i
        self.clusize = np.zeros(K, dtype=int) # cluster size
        self.ntr_sums = np.zeros(K, dtype=float) # internal degree
        self.ext_sums = np.zeros(K, dtype=float) # external degree
        self.total_sums = np.zeros(K, dtype=float) # total degree
        self.costs = np.zeros(K, dtype=float)
        
        # node_to_clu_w = K x N matrix
        self.node_to_clu_w = np.zeros((K, N), dtype=float)
        
        self.cost = -float('inf')
        self.conductance = 0.0
        self.balance_factor = 0.0
        self.min_part_size = 0
        self.max_part_size = 0
        self.min_max_part_ratio = 0.0
        
    def copyClustering(self, new_clu: Clustering):
        self.N = new_clu.N
        self.K = new_clu.K
        
        self.cost = new_clu.cost
        self.conductance = new_clu.conductance
        self.balance_factor = new_clu.balance_factor
        self.min_part_size = new_clu.min_part_size
        self.max_part_size = new_clu.max_part_size
        self.min_max_part_ratio = new_clu.min_max_part_ratio
        
        # deep copy
        self.part = copy.deepcopy(new_clu.part)
        self.clusize = copy.deepcopy(new_clu.clusize)
        self.ntr_sums = copy.deepcopy(new_clu.ntr_sums)
        self.ext_sums = copy.deepcopy(new_clu.ext_sums)
        self.total_sums = copy.deepcopy(new_clu.total_sums)
        self.costs = copy.deepcopy(new_clu.costs)
        
        self.node_to_clu_w = copy.deepcopy(new_clu.node_to_clu_w)
        
        return self

    def clone(self):
        new_clu = Clustering(self.N, self.K)
        
        return new_clu.copyClustering(self)
    
    def change_K(self, K):
        self.K = K # cluster count
        self.part = np.random.randint(0, K, size=self.N) # size of N, part[i] = cluster of node-i
        self.clusize = np.zeros(K, dtype=int) # cluster size
        self.ntr_sums = np.zeros(K, dtype=float) # internal degree
        self.ext_sums = np.zeros(K, dtype=float) # external degree
        self.total_sums = np.zeros(K, dtype=float) # total degree
        self.costs = np.zeros(K, dtype=float)
        
        # node_to_clu_w = K x N matrix
        self.node_to_clu_w = np.zeros((K, self.N), dtype=float)