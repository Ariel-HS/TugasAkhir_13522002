import numpy as np
import time
import random

from .clustering import Clustering
from .coloring import Coloring
from .graph import NNgraph

def grow_cluster(clu, graph: NNgraph, seed_id, new_part, steps):
    """
    Expands a cluster from a seed node by greedily adding nodes that 
    maximize the internal weight of the cluster.
    """
    N = clu.N
    
    # 1. Reset the connection weights for this specific partition
    # node_to_clu_w[new_part] stores the sum of weights (degrees) from each node to 'new_part'
    clu.node_to_clu_w[new_part].fill(0.0)

    # 2. Dissolve the existing cluster
    # Any node currently in 'new_part' is reassigned randomly to other clusters
    current_members_mask = (clu.part == new_part)
    if np.any(current_members_mask):
        clu.part[current_members_mask] = np.random.randint(0, clu.K, size=np.sum(current_members_mask))

    # 3. Initialize with the seed node
    seed_node = graph.nodes[seed_id]
    clu.part[seed_id] = new_part
    
    # Use -inf to mark that the node is already 'taken' so it isn't picked again
    clu.node_to_clu_w[new_part][seed_id] = -np.inf

    # Update weights for neighbors of the seed
    for neighbor_id in seed_node.nset:
        clu.node_to_clu_w[new_part][neighbor_id] += 1

    # 4. Greedy expansion loop
    for _ in range(steps):
        # Find the node with the maximum connection to our growing cluster
        # find_max_val equivalent in NumPy:
        max_node = np.argmax(clu.node_to_clu_w[new_part])
        max_val = clu.node_to_clu_w[new_part][max_node]

        # Stop if there are no more nodes connected to this cluster
        if max_val <= 0.0:
            break

        # Absorb the max_node into the cluster
        node_to_add = graph.nodes[max_node]
        clu.part[max_node] = new_part
        clu.node_to_clu_w[new_part][max_node] = -np.inf # Mark as taken

        # Update connections for neighbors of the newly added node
        for neighbor_id in node_to_add.nset:
            # Only update nodes not already in the cluster
            if clu.part[neighbor_id] != new_part:
                clu.node_to_clu_w[new_part][neighbor_id] += 1

def density_init_partition(graph: NNgraph, clu):
    """
    Initializes clusters by identifying high-density seed nodes and 
    growing clusters from them.
    """
    start_time = time.time()
    
    N = graph.size
    densities = np.zeros(N)

    # 1. Calculate densities for each node
    for i in range(N):
        node_a = graph.nodes[i]
        
        # Weight to neighbor multiplied by neighbor's total weight sum
        d = 0.0
        for neighbor_id in node_a.nset:
            node_b = graph.nodes[neighbor_id]
            d += node_b.weight_sum
        densities[i] = d

    # 2. Sort nodes by density in descending order
    # item_ids will contain the node indices sorted from highest to lowest density
    item_ids = np.argsort(-densities)

    # 3. Initialize all partitions to -1 (unassigned)
    clu.part.fill(-1)

    # 4. Grow K clusters from the highest density unassigned nodes
    # grow_factor determines how many nodes to pull into the seed cluster
    grow_factor = 0.8
    grow_size = int((clu.N * grow_factor) / clu.K)
    
    for i_clu in range(clu.K):
        for nid in item_ids:
            if clu.part[nid] == -1:
                # Expand the cluster
                grow_cluster(clu, graph, nid, i_clu, grow_size)
                break

    # 5. Handle "orphan" nodes
    # Any node not captured by the grow_cluster phase is assigned randomly
    unhandled_mask = (clu.part == -1)
    num_unhandled = np.sum(unhandled_mask)
    
    if num_unhandled > 0:
        clu.part[unhandled_mask] = [random.randint(0, clu.K - 1) for _ in range(num_unhandled)]
        
    end_time = time.time()
    init_time = end_time - start_time
    
    return init_time

def coloring_init_partition(graph, clu, coloring):
    """
    Initializes clusters using colored nodes as seeds and
    growing clusters from them
    """
    start_time = time.time()
    
    N = graph.size

    # NOTE: assume first color is best color (already sorted during coloring)
    seeds = coloring.part_color[0]
    densities = np.full(N, -1.0) # non seeds are ranked in bottom
    
    K = len(seeds)
    clu.change_K(K)
    
    for nid in seeds:
        node_a = graph.nodes[nid]
        
        # Simple density by node degree
        densities[nid] = len(node_a.nset) 

    # Sort nodes 
    item_ids = np.argsort(-densities)

    # Initialize all partitions to -1 (unassigned)
    clu.part.fill(-1)

    # Grow K clusters
    # grow_factor determines how many nodes to pull into the seed cluster
    grow_factor = 0.8
    grow_size = int((clu.N * grow_factor) / clu.K)
    
    clu_counter = 0
    for nid in item_ids:
        if clu_counter > clu.K:
            break
        
        if densities[nid] >= 0: # Only consider nodes with valid density (i.e., seeds)
            
            # Expand the cluster using the greedy weight maximization
            grow_cluster(clu, graph, nid, clu_counter, grow_size)
            clu_counter += 1

    # Handle orphan nodes
    # Any node not captured by the grow_cluster phase is assigned randomly
    unhandled_mask = (clu.part == -1)
    num_unhandled = np.sum(unhandled_mask)
    
    if num_unhandled > 0:
        clu.part[unhandled_mask] = [random.randint(0, clu.K - 1) for _ in range(num_unhandled)]
        
    end_time = time.time()
    init_time = end_time - start_time
    
    return init_time

def get_all_coloring_init_partition(graph: NNgraph, clu: Clustering, coloring: Coloring):
    """
    Initializes clusters using colored nodes as seeds and
    growing clusters from them based on density.
    Try all colors, return as list of clusterings
    """
    start_time = time.time()
    
    N = graph.size
    
    list_of_clusterings = []

    # 1. Sort Nodes
    for i in range(len(coloring.part_color)):
        partition = coloring.part_color[i]

        # Skip partitions with only one community
        if len(partition) < 2:
            continue

        clu_copy = clu.clone()
        seeds = partition
        densities = np.full(N, -1.0) # non seeds are ranked in bottom
        
        K = len(seeds)
        clu_copy.change_K(K)
        
        for nid in seeds:
            node_a = graph.nodes[nid]
            
            densities[nid] = len(node_a.nset)  # Simple density by node degree

        # item_ids will contain the node indices sorted from highest to lowest density
        item_ids = np.argsort(-densities)

        # 2. Initialize all partitions to -1 (unassigned)
        clu_copy.part.fill(-1)

        # 3. Grow K clusters 
        # grow_factor determines how many nodes to pull into the seed cluster
        grow_factor = 0.8
        grow_size = int((clu_copy.N * grow_factor) / clu_copy.K)
        
        clu_counter = 0
        for nid in item_ids:
            if clu_counter > clu_copy.K:
                break
            
            if densities[nid] >= 0:
                # Expand the cluster 
                grow_cluster(clu_copy, graph, nid, clu_counter, grow_size)
                clu_counter += 1

        # 4. Handle "orphan" nodes
        # Any node not captured by the grow_cluster phase is assigned randomly
        unhandled_mask = (clu_copy.part == -1)
        num_unhandled = np.sum(unhandled_mask)
        
        if num_unhandled > 0:
            clu_copy.part[unhandled_mask] = [random.randint(0, clu_copy.K - 1) for _ in range(num_unhandled)]
            
        list_of_clusterings.append(clu_copy)
        
    end_time = time.time()
    init_time = end_time - start_time
        
    return list_of_clusterings, init_time