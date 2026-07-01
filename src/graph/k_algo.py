import numpy as np
import time
import random
from .costs import calculate_full_cost, choose_best_by_delta
from .graph import NNgraph

def k_algo(graph, clu):
    """
    Iteratively refines clusters by moving nodes to partitions that 
    yield the highest cost delta (improvement).
    """
    start_time = time.time()
    
    # Calculate initial cost and metrics
    conductance = 0.0
    cost_prev = calculate_full_cost(graph, clu)
    cur_cost = cost_prev
    
    # We use a copy for the internal partition state if needed, 
    # but usually, we update 'clu' in place.
    iter_count = 0
    
    max_iter = 50
    for iter_count in range(max_iter):
        # Shuffle nodes to avoid order-bias in optimization
        node_indices = list(range(clu.N))
        random.shuffle(node_indices)
        
        changes_made = 0
        
        for nid in node_indices:
            old_part = clu.part[nid]
            
            # Use our optimized delta function to find the best candidate partition
            best_part, d_cost, d_sum_old, d_sum_new, d_total_sum = \
                choose_best_by_delta(nid, graph, clu)
            
            # If a better partition is found (d_cost > 0)
            if best_part != old_part:
                # Update local metrics incrementally (faster than full recalculation)
                clu.ntr_sums[old_part] += d_sum_old
                clu.ntr_sums[best_part] += d_sum_new
                
                clu.total_sums[old_part] -= d_total_sum
                clu.total_sums[best_part] += d_total_sum
                
                clu.clusize[old_part] -= 1
                clu.clusize[best_part] += 1
                
                # Update the actual partition map
                clu.part[nid] = best_part
                
                # Update the running cost estimate
                cur_cost += d_cost
                changes_made += 1

        # Recalculate full cost to correct any floating point drift from increments
        actual_cost = calculate_full_cost(graph, clu)
        
        # Convergence check: if cost didn't improve or no changes were made, stop
        if actual_cost <= cost_prev or changes_made == 0:
            break
            
        cost_prev = actual_cost

    end_time = time.time()
    k_time = end_time - start_time
    
    return iter_count, k_time, cost_prev

def get_best_k_algo(graph, clusterings):
    start_time = time.time()
    
    best_cond = None
    best_clu = None
    total_iter_count = 0
    for clu in clusterings:
        clu_copy = clu.clone()
        iter_count, k_time, best_cost = k_algo(graph, clu_copy)
        
        total_iter_count += iter_count
        cond = clu_copy.conductance
        
        if best_cond is None or cond > best_cond: # NOTE conductance is negative so bigger = better
            best_cond = cond
            best_clu = clu_copy.clone()
    
    end_time = time.time()
    clust_time = end_time - start_time
    
    return total_iter_count, best_cond, best_clu, clust_time


