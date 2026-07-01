import networkx as nx
import time
import numpy as np
from sklearn.cluster import KMeans
from .graph import NNgraph
from .clustering import Clustering

def seeded_page_rank(G: nx.Graph, graph: NNgraph, clu: Clustering, seed):
    start_time = time.time()
    # Uses conductance as optimum function

    def get_conductance(G, cluster):
        # Number of edges crossing out of the cluster
        cut_size = nx.cut_size(G, cluster)
        # Total degrees of nodes inside the cluster (Volume)
        volume = sum(dict(G.degree(cluster)).values())
        # Total degrees of nodes outside the cluster
        vol_total = sum(dict(G.degree()).values())
        vol_complement = vol_total - volume
        
        return cut_size / min(volume, vol_complement)

    clu.part.fill(-1)
    best_score = np.zeros(G.number_of_nodes())
    seed_clu_map = {}

    for i, node_idx in enumerate(seed):
        seed_clu_map[node_idx] = i
        # 1. Get PPR scores 
        ppr = nx.pagerank(G, personalization={node_idx: 1})

        # 2. Sort nodes by score descending
        nodes_sorted = sorted(ppr.items(), key=lambda x: x[1], reverse=True)

        # 3. Sweep: check conductance at every step
        best_conductance = float('inf')
        best_cluster = []

        current_set = set()
        for node, _ in nodes_sorted[:-1]: # We don't check the very last node (full graph)
            current_set.add(node)
            phi = get_conductance(G, current_set)
            
            if phi < best_conductance:
                best_conductance = phi
                best_cluster = list(current_set)

        # add best_cluster in clustering
        for node in best_cluster:
            if clu.part[node] != -1:
                # Pick cluster with highest PPR
                if best_score[node] < ppr[node]:
                    clu.part[node] = i
                    best_score[node] = ppr[node]
                # else ignore
            else:
                clu.part[node] = i
                best_score[node] = ppr[node]

    for i, clu_id in enumerate(clu.part): 
        if clu_id == -1:
            # get seed with min distance
            min_seed = min(seed, key=lambda s: graph.dist_matrix[i][s])
            clu.part[i] = seed_clu_map[min_seed]

    # return clu.part
    end_time = time.time()
    final_time = end_time - start_time

    return final_time
    
def seeded_spectral(graph: NNgraph, seed):
    def make_degree_matrix(nngraph: NNgraph):
        degrees = np.array([len(nngraph.nodes[i].nset) for i in range(nngraph.size)])
        return np.diag(degrees)
    
    def make_normalized_laplacian(A, D):
        A = np.array(A)
        # D^(-1/2)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        # L_sym = I - D^(-1/2) * A * D^(-1/2)
        return np.eye(A.shape[0]) - d_inv_sqrt @ A @ d_inv_sqrt

    def get_spectral_embedding(laplacian, k):
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        spectral_embedding = eigenvectors[:, :k]
        
        return spectral_embedding
    
    A = graph.adj_matrix
    D = make_degree_matrix(graph)
    L = make_normalized_laplacian(A, D)

    K = len(seed)
    spectral_embedding = get_spectral_embedding(L, K)

    # Normalization step
    rows_norm = np.linalg.norm(spectral_embedding, axis=1, keepdims=True)
    embedded_norm = spectral_embedding / rows_norm

    embedded_seed = [spectral_embedding[i] for i in seed]
    kmeans = KMeans(n_clusters=K, init=embedded_seed, random_state=42, n_init=1)
    
    return kmeans.fit(embedded_norm)