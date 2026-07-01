"""
Run locally with:
    streamlit run app.py
"""

from scipy.io import mmread
import io
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

import graph as gl


# ──────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Graph Clustering Explorer", layout="wide")
st.title("Graph Clustering Explorer")


# ──────────────────────────────────────────────────────────────────────────
# 2. PREDEFINED GRAPHS
# ──────────────────────────────────────────────────────────────────────────
def _clean(G: nx.Graph) -> nx.Graph:
    for _, _, data in G.edges(data=True):
        data.pop("weight", None)
    return nx.convert_node_labels_to_integers(G, first_label=0, label_attribute="name")

PREDEFINED_GRAPHS = {
    "Karate Club":    lambda: _clean(nx.karate_club_graph()),
    "Les Miserables": lambda: _clean(nx.les_miserables_graph()),
}

def load_uploaded_graph(uploaded_file) -> nx.Graph:
    name = uploaded_file.name
    
    if name.endswith(".mtx"):
        adj_matrix = mmread(io.BytesIO(uploaded_file.read()))
        G = nx.from_scipy_sparse_array(adj_matrix, edge_attribute=None)
    else:
        text = uploaded_file.read().decode("utf-8")
        G = nx.parse_edgelist(text.splitlines(), nodetype=int)

    G = _clean(G)

    return G


# ──────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR — CONTROLS
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Graph")
    graph_source = st.radio("Source", ["Predefined", "Upload"])

    if graph_source == "Predefined":
        graph_choice = st.selectbox("Choose a graph", list(PREDEFINED_GRAPHS.keys()))
        G = PREDEFINED_GRAPHS[graph_choice]()
    else:
        uploaded = st.file_uploader("Upload graph (edge list .txt/.csv/.mtx)", type=["txt", "csv", "edgelist", "mtx"])
        G = load_uploaded_graph(uploaded) if uploaded is not None else None

    
    st.header("2. Clustering Method")
    clustering_method = st.selectbox("Method", ["K-Algo", "PageRank", "Spectral"])

    st.header("3. Initialization Method")
    options = ["Biased", "Biased Distance"]
    # Only K-Algo supports Greedy/Balanced coloring-based initialization
    if clustering_method == "K-Algo":
        options = ["Greedy", "Balanced"] + options
    init_method = st.selectbox(
        "Method",
        options,
    )
    # Only Biased Distance needs an extra parameter
    distance_param = None
    if init_method == "Biased Distance":
        distance_param = st.slider("Distance", min_value=2, max_value=10, value=2)

    run_button = st.button("Run Clustering", type="primary", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# 4. PIPELINE — INIT METHOD DISPATCH
# ──────────────────────────────────────────────────────────────────────────
def build_initial_clustering(graph_obj, G, method: str, distance=2):
    """Dispatch to the right initialization routine based on the dropdown.
    """
    clu = gl.Clustering(graph_obj.size, 1)  # Start with K=1, will be updated by init methods
    final_clu = clu  # Placeholder for the final clustering to return
    coloring_time = 0.0
    init_time = 0.0

    if method == "Greedy":
        coloring, coloring_time = gl.greedy_coloring(G)
        list_of_clusterings, init_time = gl.get_all_coloring_init_partition(graph_obj, clu, coloring)
        
        final_clu = list_of_clusterings
    elif method == "Balanced":
        coloring, coloring_time = gl.greedy_balanced_coloring(G)
        list_of_clusterings, init_time = gl.get_all_coloring_init_partition(graph_obj, clu, coloring)
        
        final_clu = list_of_clusterings
    elif method == "Biased":
        coloring, coloring_time = gl.biased_coloring(graph_obj)
        init_time = gl.coloring_init_partition(graph_obj, clu, coloring)

        final_clu = clu
    elif method == "Biased Distance":
        coloring, coloring_time = gl.biased_distance_coloring(graph_obj, distance=distance)
        init_time = gl.coloring_init_partition(graph_obj, clu, coloring)

        final_clu = clu

    return final_clu, coloring_time, init_time


def run_clustering(G: nx.Graph, init_method: str, clustering_method: str,
                    distance=2):
    """Full pipeline: build NNgraph -> initialize -> cluster -> return results.

    Returns a dict with at least:
        partition: list/array of length N, partition[i] = cluster id of node i
        metrics: dict of metric name -> value (cost, conductance, etc.)
        elapsed: seconds taken
    """
    start = time.time()

    graph_obj = gl.from_networkx(G)
    
    print("Tes:", clustering_method, init_method, distance)
    if clustering_method == "K-Algo":
        clu, coloring_time, init_time = build_initial_clustering(graph_obj, G, init_method, distance)
        if init_method in ["Greedy", "Balanced"]:
            _, best_cond, best_clu, clu_time = gl.get_best_k_algo(graph_obj, clu)
            partition = best_clu.part
            clu_set = gl.partition_to_set_helper(best_clu)
            metrics = {
                "K": best_clu.K,
                "conductance": -best_cond,
                "modularity": nx.algorithms.community.modularity(G, clu_set),
                "coloring_time": coloring_time,
                "init_time": init_time,
                "clustering_time": clu_time,
            }
        else:
            _, clu_time, _ = gl.k_algo(graph_obj, clu)
            partition = clu.part
            clu_set = gl.partition_to_set_helper(clu)
            metrics = {
                "K": clu.K,
                "conductance": -clu.conductance,
                "modularity": nx.algorithms.community.modularity(G, clu_set),
                "coloring_time": coloring_time,
                "init_time": init_time,
                "clustering_time": clu_time,
            }
    elif clustering_method == "PageRank":
        if init_method not in ["Biased", "Biased Distance"]:
            raise ValueError(f"PageRank clustering only supports Biased or Biased Distance initialization, got {init_method}")
        
        clu = gl.Clustering(graph_obj.size, 1)  # Start with K=1, will be updated 
        if init_method == "Biased":
            coloring, coloring_time = gl.biased_coloring(graph_obj)
        elif init_method == "Biased Distance":
            coloring, coloring_time = gl.biased_distance_coloring(graph_obj, distance=distance)

        seed = coloring.part_color[0]
        clu.change_K(len(seed))

        clu_time = gl.seeded_page_rank(G, graph_obj, clu, seed)
        
        conductance = gl.calculate_full_cost(graph_obj, clu)
        clu_set = gl.partition_to_set_helper(clu)
        modularity = nx.algorithms.community.modularity(G, clu_set)

        partition = clu.part

        metrics = {
            "K": clu.K,
            "conductance": -conductance,
            "modularity": modularity,
            "coloring_time": coloring_time,
            "init_time": 0.0,  # No separate init time for PageRank
            "clustering_time": clu_time,
        }

    elif clustering_method == "Spectral":
        clu = gl.Clustering(graph_obj.size, 1)  # Start with K=1, will be updated 
        if init_method == "Biased":
            coloring, coloring_time = gl.biased_coloring(graph_obj)
        elif init_method == "Biased Distance":
            coloring, coloring_time = gl.biased_distance_coloring(graph_obj, distance=distance)

        seed = coloring.part_color[0]
    
        start_time = time.time()

        labels = gl.seeded_spectral(graph_obj, seed=seed).labels_

        end_time = time.time()
        clu_time = end_time - start_time

        new_K = len(seed)
        clu.change_K(new_K)
        clu.part = labels

        conductance = gl.calculate_full_cost(graph_obj, clu)

        clu_set = gl.partition_to_set_helper(clu)
        modularity = nx.algorithms.community.modularity(G, clu_set)
        
        partition = clu.part
        metrics = {
            "K": clu.K,
            "conductance": -conductance,
            "modularity": modularity,
            "coloring_time": coloring_time,
            "init_time": 0.0,  # No separate init time for Spectral
            "clustering_time": clu_time,
        }
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")

    elapsed = time.time() - start
    return {"partition": partition, "metrics": metrics, "elapsed": elapsed}


# ──────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION HELPERS
# ──────────────────────────────────────────────────────────────────────────
def plot_partition(G: nx.Graph, partition) -> plt.Figure:
    """Render the graph colored by cluster assignment.

    Reuses the spring_layout + node_color pattern from draw_graph_clustering
    in your notebook, but returns a Figure instead of calling plt.show(),
    so Streamlit can render it.
    """
    with st.spinner("Visualizing graph..."):
        fig, ax = plt.subplots(figsize=(7, 6))
        pos = nx.spring_layout(G, seed=42)

        node_size = 100
        width = 0.5

        if G.number_of_nodes() > 1000:
            node_size = 10
            width = 0.1
        elif G.number_of_nodes() > 4000:
            node_size = 5
            width = 0.05

        nx.draw(
            G, pos=pos, ax=ax, with_labels=False,
            node_color=partition, node_size=node_size, width=width, cmap=plt.cm.rainbow,
        )
        return fig


def plot_metrics(metrics: dict):
    """Display scalar metrics + cluster size distribution.
    """
    st.metric("K", f"{metrics.get('K', 0)}")
    st.metric("Conductance", f"{metrics.get('conductance', 0):.4f}")
    st.metric("Modularity", f"{metrics.get('modularity', 0):.4f}")
    st.metric("Coloring Time", f"{metrics.get('coloring_time', 0):.4f}")
    st.metric("Init Time", f"{metrics.get('init_time', 0):.4f}")
    st.metric("Clustering Time", f"{metrics.get('clustering_time', 0):.4f}")


# ──────────────────────────────────────────────────────────────────────────
# 6. MAIN PANEL
# ──────────────────────────────────────────────────────────────────────────
if G is None:
    st.info("Upload a graph file or choose a predefined graph from the sidebar to get started.")
elif run_button:
    with st.spinner("Running clustering..."):
        result = run_clustering(G, init_method, clustering_method, distance_param)

    st.success(f"Done in {result['elapsed']:.2f}s")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Graph Visualization")
        fig = plot_partition(G, result["partition"])
        st.pyplot(fig)

    with right:
        st.subheader("Metrics")
        plot_metrics(result["metrics"])
else:
    st.subheader("Graph Preview")
    fig = plot_partition(G, [0] * G.number_of_nodes())
    st.pyplot(fig)
    st.caption(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges. Configure options in the sidebar, then click **Run Clustering**.")
