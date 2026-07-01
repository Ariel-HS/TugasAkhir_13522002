import networkx as nx
import numpy as np
from .graph import NNgraph

def make_distance_matrix(nx_graph: nx.Graph, size):
    path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph))
    
    # Initialize with infinity
    dist_matrix = np.full((size, size), float('inf'))
    
    for source, targets in path_lengths.items():
        if source < size:
            for target, distance in targets.items():
                if target < size:
                    dist_matrix[source, target] = distance
                    
    return dist_matrix

def partition_to_set_helper(clustering):
    cluster_indexes = np.unique(clustering.part)
    cluster_set = [
        set(np.where(clustering.part == label)[0]) 
        for label in cluster_indexes
    ]
    
    return cluster_set

# IMPORTS

def from_networkx(nx_graph):
    """
    Converts a NetworkX graph into the custom NNgraph 
    Automatically set weight to 1 (so no need to scale weight if the same)
    """
    N = nx_graph.number_of_nodes()
    custom_graph = NNgraph(N)
    
    # Create Adjacency Matrix
    if hasattr(custom_graph, 'adj_matrix'):
        custom_graph.adj_matrix = nx.to_numpy_array(nx_graph, dtype=float, weight=None)
        # Convert adj_matrix to list-of-lists
        custom_graph.adj_matrix = custom_graph.adj_matrix.tolist()

    degrees = dict(nx_graph.degree(weight=None))
    
    custom_nodes = custom_graph.nodes
    
    for i, neighbors_dict in nx_graph.adj.items():
        if i >= N: 
            continue # Safety guard in case node IDs are non-sequential or out of bounds
            
        node_i = custom_nodes[i]
        
        node_i.nset = np.array([neighbor for neighbor in neighbors_dict])
        
        node_i.degree_sum = float(degrees.get(i, 0))

    # Total graph degree calculation
    # NOTE: Assume undirected unweighted graph, sum of degrees = 2 * total_edges.
    custom_graph.total_degree = float(sum(degrees.values()))
    
    # Make Distance Matrix
    custom_graph.dist_matrix = make_distance_matrix(nx_graph, N)
    
    return custom_graph


def adj_list_to_graphs(file, size):
    nn_graph = NNgraph(size)
    
    try:
        nx_graph = nx.read_edgelist(
            file, 
            nodetype=int, 
            data=False, # Assumes unweighted
            create_using=nx.Graph
        )
        nx_graph.add_nodes_from(range(size))
        
        # NOTE: Simple filter for edges above size
        invalid_nodes = [n for n in nx_graph.nodes if n >= size]
        if invalid_nodes:
            nx_graph.remove_nodes_from(invalid_nodes)
            
        nx_graph.add_nodes_from(range(size))

        # Create Adjacency Matrix using NumPy/NetworkX
        if hasattr(nn_graph, 'adj_matrix'):
            nn_graph.adj_matrix = nx.to_numpy_array(nx_graph, dtype=int)
            
        nn_nodes = nn_graph.nodes
        weight = 1
        
        # Update parsed edges
        for node1, node2 in nx_graph.edges():
            n1 = nn_nodes[node1]
            n1.nset.append((node2, weight))
            n1.degree_sum += weight
            
            n2 = nn_nodes[node2]
            n2.nset.append((node1, weight))
            n2.degree_sum += weight
            
            nn_graph.total_degree += weight
            
            nx_graph[node1][node2]['weight'] = weight

        nn_graph.dist_matrix = make_distance_matrix(nx_graph, size)
    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while parsing: {e}")
        return None, None
        
    return nn_graph, nx_graph

def import_ground_truth(file, G: nx.Graph):
    community_mapping = {}
    
    try:
        with open(file, 'r') as f:
            for node_id, line in enumerate(f):
                community_id = int(line.strip())
                community_mapping[node_id] = community_id

        nx.set_node_attributes(G, community_mapping, name="community")
        
    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while parsing: {e}")
        return None, None
        
    return G