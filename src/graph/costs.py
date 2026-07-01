import numpy as np
from .graph import NNgraph

def calculate_full_cost(graph: NNgraph, clu): 
    """
    Recalculates cluster metrics and returns total cost.
    input:
        graph: NNgraph
        clu: Clustering
    """
    clu.ntr_sums.fill(0.0)
    clu.ext_sums.fill(0.0)
    clu.clusize.fill(0)
    clu.total_sums.fill(0.0)

    for i in range(graph.size):
        node = graph.nodes[i]
        p_a = clu.part[i]
        clu.clusize[p_a] += 1
        clu.total_sums[p_a] += node.degree_sum

        for neighbor_id in node.nset:
            p_b = clu.part[neighbor_id]
            if p_a == p_b:
                clu.ntr_sums[p_a] += 1
            else:
                clu.ext_sums[p_a] += 1
                # NOTE Symmetric graph assumes dist is added to both

    total_conductance = 0.0
    
    for i in range(clu.K):
        if clu.total_sums[i] > 0:
            ext_sum = clu.total_sums[i] - clu.ntr_sums[i]
            total_conductance += ext_sum / clu.total_sums[i]

    clu.conductance = -(total_conductance / clu.K)
    clu.min_part_size = np.min(clu.clusize)
    clu.max_part_size = np.max(clu.clusize)
    
    return clu.conductance

def choose_best_by_delta(nid, graph: NNgraph, clu):
    """
    Calculates the cost delta for moving a node to every other cluster.
    Returns the index of the best partition and the associated deltas.
    """
    node_a = graph.nodes[nid]
    old_part = clu.part[nid]
    
    # Calculate d_ntr_sums: how much degree node_a shares with each cluster
    # Using a local array for speed
    d_ntr_sums = np.zeros(clu.K)
    for neighbor_id in node_a.nset:
        p_b = clu.part[neighbor_id]
        d_ntr_sums[p_b] += 1

    # 1. Calculate the 'Removal' delta (cost change for the cluster it leaves)
    # Conductance: -(total_ext / total_vol)
    old_val = -(clu.total_sums[old_part] - clu.ntr_sums[old_part]) / clu.total_sums[old_part]
    
    new_vol = clu.total_sums[old_part] - node_a.degree_sum
    new_ntr = clu.ntr_sums[old_part] - d_ntr_sums[old_part]
    new_val = -((new_vol - new_ntr) / new_vol) if new_vol > 0 else 0
    
    d_removal = new_val - old_val

    # 2. Vectorized Addition Delta (cost change for every cluster it could join)
    # We ignore the old_part by setting its cost to a very low value
    all_indices = np.arange(clu.K)
    
    # Conductance delta for all clusters
    new_vols = clu.total_sums + node_a.degree_sum
    new_ntrs = clu.ntr_sums + d_ntr_sums
    
    # safe division 
    old_vals = -np.divide(clu.total_sums - clu.ntr_sums, 
                        clu.total_sums, 
                        out=np.zeros_like(clu.total_sums), 
                        where=clu.total_sums != 0)
    new_vals = -np.divide(new_vols - new_ntrs, 
                        new_vols, 
                        out=np.zeros_like(new_vols), 
                        where=new_vols != 0)
    
    d_adds = new_vals - old_vals

    # Total delta per candidate cluster
    total_deltas = (d_adds + d_removal) / clu.K
    total_deltas[old_part] = -np.inf # Don't select the cluster we are already in

    best_part = np.argmax(total_deltas)
    best_cost_delta = total_deltas[best_part]

    if best_cost_delta <= 0:
        return old_part, 0.0, 0.0, 0.0, 0.0

    # Return: best cluster, cost delta, ntr_old_delta, ntr_new_delta, total_sum_delta
    return (best_part, 
            best_cost_delta, 
            -d_ntr_sums[old_part], 
            d_ntr_sums[best_part], 
            node_a.degree_sum)