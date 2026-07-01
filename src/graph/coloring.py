import numpy as np
import copy
import time
import networkx as nx
from collections import defaultdict
from .graph import NNgraph

class Coloring:
    def __init__(self, N, C):
        self.N = N # node count
        self.C = C # color count
        self.part_node = np.random.randint(0, C, size=N) # size of N, part[i] = color of node-i
        self.colsize = np.zeros(C, dtype=int) # color-group size
        self.part_color: list[list[int]] = [[] for _ in range(C)] # same as part_list but in dict, part_dict[color] = list of nodes with color of 'color'
        
    def copyColoring(self, new_col: Coloring):
        self.N = new_col.N
        self.C = new_col.C
        
        # deep copy
        self.part_node = copy.deepcopy(new_col.part_node)
        self.colsize = copy.deepcopy(new_col.colsize)
        self.part_color = copy.deepcopy(new_col.part_color)
        
        return self

    def clone(self):
        new_col = Coloring(self.N, self.C)
        
        return new_col.copyColoring(self)
    
def greedy_coloring(G):
    """Return mapping node -> color (integer) using networkx greedy_color
    """
    start_time = time.time()
    greedy_coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    
    N = G.number_of_nodes()
    
    part_node = np.zeros(N, dtype=int)
    part_color = defaultdict(list)
    
    for node, color in greedy_coloring.items():
        part_node[node] = color
        part_color[color].append(node)
    
    C = len(part_color)
    coloring = Coloring(N, C)
    
    # convert part_color from dict to list of list
    final_part_color = [[] for _ in range(C)]
    clusize = np.zeros(C, dtype=int)
    
    for color in range(C):
        nodes_in_color = part_color[color]
        final_part_color[color] = nodes_in_color
        clusize[color] = len(nodes_in_color)
        
    coloring.part_node = part_node
    coloring.part_color = final_part_color
    coloring.colsize = clusize
    
    end_time = time.time()
    coloring_time = end_time - start_time
    
    return coloring, coloring_time

def greedy_balanced_coloring(G):
    """
    Balanced greedy coloring:
    - Maintain proper coloring.
    - Prefer colors with smallest group sizes.
    
    G: networkx graph
    """
    start_time = time.time()
    
    N = G.number_of_nodes()
    coloring = {}
    part_color = defaultdict(list)
    
    # Process nodes in decreasing degree (largest-first heuristic)
    nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)

    for node in nodes:
        # Colors used by neighbors
        neighbor_colors = {coloring[n] for n in G.neighbors(node) if n in coloring}

        # List available colors (those not used by neighbors)
        valid_colors = [
            c for c in part_color.keys()
            if c not in neighbor_colors
        ]

        if valid_colors:
            # Pick the smallest-sized color group to keep balance
            chosen_color = min(valid_colors, key=lambda c: len(part_color[c]))
        else:
            # No valid color exists => create a new color
            chosen_color = len(part_color)

        # Assign color
        coloring[node] = chosen_color
        part_color[chosen_color].append(node)
    
    
    C = len(part_color)    
    node_colors = np.zeros(N, dtype=int)
    
    for node, color in coloring.items():
        node_colors[node] = color
        
    # convert part_color from dict to list of list
    final_part_color = [[] for _ in range(C)]
    clusize = np.zeros(C, dtype=int)
    
    for color in range(C):
        nodes_in_color = part_color[color]
        final_part_color[color] = nodes_in_color
        clusize[color] = len(nodes_in_color)
    
    final_coloring = Coloring(N, C)
    final_coloring.colsize = clusize
    final_coloring.part_node = node_colors
    final_coloring.part_color = final_part_color
    
    end_time = time.time()
    coloring_time = end_time - start_time

    return final_coloring, coloring_time

def biased_coloring(graph: NNgraph):
    start_time = time.time()
    N = graph.size
    
    # Calculate score of each node
    # NOTE: Local score is by sum of similarity with its neighbor
    # sim uses common neighbors (CN) 
    
    Adj_matrix = graph.adj_matrix
    Adj_squared = np.dot(Adj_matrix, Adj_matrix)

    coloring = {}
    part_color = defaultdict(set)
    part_color[0]
    
    nodes = graph.nodes

    # Stores score for order of conflict check later
    full_score = np.zeros(N, dtype=float)

    # Check every egonet. Nodes with best score in each egonet get color 0.
    for node in nodes:
        nbr_ids = [nbr for nbr in node.nset]
        egonet_ids = [node.id] + nbr_ids

        local_scores = np.sum(Adj_squared[egonet_ids][:, egonet_ids], axis=1)
        id_to_score = {global_id: score for global_id, score in zip(egonet_ids, local_scores)}

        full_score[node.id] = id_to_score[node.id]

        max_score = id_to_score[node.id]
        best_nbr = node.id
        for nbr_id in node.nset:

            if id_to_score[nbr_id] > max_score:
                max_score = id_to_score[nbr_id]
                best_nbr = nbr_id
        
        coloring[best_nbr] = 0

    nodes = sorted(graph.nodes, key=lambda x: full_score[x.id], reverse=True)
    # Check for conflicts and assign colors to remaining nodes
    for node in nodes:
        # Colors used by neighbors
        neighbor_colors = {coloring[n] for n in node.nset if n in coloring}

        # List available colors (those not used by neighbors)
        valid_colors = [
            c for c in part_color.keys()
            if c not in neighbor_colors and c != 0
        ]

        # Preemptively choose possible color
        if valid_colors:
            chosen_color = min(valid_colors, key=lambda c: len(part_color[c]))
        else:
            # No valid color exists => create a new color
            chosen_color = len(part_color)
        
        if node.id in coloring:
            # Check and resolve conflicts
            for nbr_id in node.nset:
                if (nbr_id in coloring and node.id in coloring 
                and coloring[node.id] == coloring[nbr_id] == 0):
                    if full_score[node.id] <= full_score[nbr_id]:
                        coloring[node.id] = chosen_color
                    else:
                        coloring.pop(nbr_id)
        else:
            coloring[node.id] = chosen_color

        final_color = coloring[node.id]
        part_color[final_color].add(node.id)
    
    
    C = len(part_color)    
    node_colors = np.zeros(N, dtype=int)
    
    for node, color in coloring.items():
        node_colors[node] = color
        
    # convert part_color from dict to list of list
    final_part_color = [[] for _ in range(C)]
    clusize = np.zeros(C, dtype=int)
    
    for color in range(C):
        nodes_in_color = part_color[color]
        final_part_color[color] = nodes_in_color
        clusize[color] = len(nodes_in_color)
    
    final_coloring = Coloring(N, C)
    final_coloring.colsize = clusize
    final_coloring.part_node = node_colors
    final_coloring.part_color = final_part_color
    
    end_time = time.time()
    coloring_time = end_time - start_time

    return final_coloring, coloring_time

def biased_distance_coloring(graph: NNgraph, distance = 2):
    start_time = time.time()
    N = graph.size
    
    # Calculate score of each node
    # NOTE: Local score is by sum of similarity with its neighbor
    # sim uses common neighbors (CN) 
    Adj_matrix = graph.adj_matrix
    Adj_squared = np.dot(Adj_matrix, Adj_matrix)

    coloring = {}
    part_color = defaultdict(set)
    
    # nodes = sorted(graph.nodes, key=lambda x: score[x.id], reverse=True)
    nodes = graph.nodes

    # Stores score for order of conflict check later
    full_score = np.zeros(N, dtype=float)

    # Check every egonet. Nodes with best score in each egonet get color 0.
    for node in nodes:
        nbr_ids = [nbr for nbr in node.nset]
        egonet_ids = [node.id] + nbr_ids

        local_scores = np.sum(Adj_squared[egonet_ids][:, egonet_ids], axis=1)
        id_to_score = {global_id: score for global_id, score in zip(egonet_ids, local_scores)}

        full_score[node.id] = id_to_score[node.id]

        max_score = id_to_score[node.id]
        best_nbr = node.id
        for nbr_id in node.nset:
            if id_to_score[nbr_id] > max_score:
                max_score = id_to_score[nbr_id]
                best_nbr = nbr_id
        
        coloring[best_nbr] = 0
        part_color[0].add(best_nbr)

    nodes = sorted(graph.nodes, key=lambda x: full_score[x.id], reverse=True)
    dist_matrix = graph.dist_matrix

    # Check for conflicts and assign colors to remaining nodes
    for node in nodes:
        node_id = node.id

        colored_ids = np.array(list(coloring.keys()))
        distances = dist_matrix[node_id, colored_ids]
        mask = (distances > 0) & (distances < distance)
        used_colors = {coloring[nid] for nid in colored_ids[mask]}
        
        valid_colors = [
            c for c in part_color.keys()
            if c not in used_colors and c != 0
        ]

        # Preemptively choose possible color
        if valid_colors:
            chosen_color = min(valid_colors, key=lambda c: len(part_color[c]))
        else:
            # No valid color exists => create a new color
            chosen_color = len(part_color)
        
        if node.id in coloring and coloring[node.id] == 0:
            # Check and resolve conflicts

            seeds_ids = np.array(list(part_color[0]))

            distances_seeds = dist_matrix[node_id, seeds_ids]
            conflict_mask = (distances_seeds > 0) & (distances_seeds < distance)
            conflicting_seeds = seeds_ids[conflict_mask]

            for nbr_id in conflicting_seeds:
                if full_score[node.id] <= full_score[nbr_id]:
                    coloring[node.id] = chosen_color
                    part_color[chosen_color].add(node.id)
                    part_color[0].remove(node.id)
                    break # Node changed color, no need to check other seeds
                else:
                    if nbr_id in coloring: # Safety check, shouldn't be possible
                        coloring.pop(nbr_id)
                    part_color[0].remove(nbr_id)
        elif node.id not in coloring:
            coloring[node.id] = chosen_color
            part_color[chosen_color].add(node.id)
    
    C = len(part_color)    
    node_colors = np.zeros(N, dtype=int)
    
    for node, color in coloring.items():
        node_colors[node] = color
        
    # convert part_color from dict to list of list
    final_part_color = [[] for _ in range(C)]
    clusize = np.zeros(C, dtype=int)
    
    for color in range(C):
        nodes_in_color = part_color[color]
        final_part_color[color] = nodes_in_color
        clusize[color] = len(nodes_in_color)
    
    final_coloring = Coloring(N, C)
    final_coloring.colsize = clusize
    final_coloring.part_node = node_colors
    final_coloring.part_color = final_part_color
    
    end_time = time.time()
    coloring_time = end_time - start_time

    return final_coloring, coloring_time
