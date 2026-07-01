# Graph Clustering using Graph Coloring-Based Initialization – Interactive Web Application

Interactive Streamlit app and Python library for experimenting with graph clustering algorithms using graph coloring as initialization.

**Key features**
- Explore clustering using K-Algo, PageRank and Spectral methods
- Multiple graph coloring-based initialization strategies: Greedy, Balanced, Biased, Biased Distance
- Visualize cluster assignments and scalar metrics (conductance, modularity)
- Supports built-in example graphs and uploaded edge lists / Matrix Market files

## Quick start

Prerequisites: Python 3.8+ and a virtual environment.

1. Create and activate a virtual environment

```bash
python -m venv env
env\Scripts\activate
```

2. Install dependencies

```bash
pip install -r src/requirements.txt
```

3. Run the Streamlit app

```bash
cd src
streamlit run app.py
```

Open http://localhost:8501 in your browser. Use the sidebar to choose a predefined graph or upload an edge list (.txt/.csv/.mtx), select an initialization and clustering method, then click **Run Clustering**.

## Deployment

Deployed demo URL: https://graph-clustering-with-coloring-init.streamlit.app

## What this project contains
- `src/app.py` — Streamlit UI and pipeline glue to run clustering experiments.
- `src/graph/` — Core graph data structures and algorithms:
	- `graph.py`, `clustering.py` — simple `Graph` and `Clustering` types
	- `coloring.py` — multiple coloring/initialization routines
	- `k_algo.py` — iterative local-refinement clustering algorithm
	- `algo.py` — seeded PageRank and seeded spectral routines
	- `costs.py`, `init_methods.py`, `utility.py` — helpers and cost computations

## How to use (examples)
- Run with the included examples (Karate Club, Les Miserables) from the sidebar.
- Upload a `.mtx` Matrix Market adjacency matrix or an edge-list file to analyze your graph.

## Dependencies
See `src/requirements.txt` — main dependencies include `streamlit`, `networkx`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

## Authors
- Ariel Herfrison (arielherfrison@gmail.com)

## References
<a id="1">[1]</a> 
Moradi, F., Olovsson, T., & Tsigas, P. (2014). A local seed selection algorithm for overlapping community detection. 2014 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2014), 1–8. 
<br />
<a id="2">[2]</a> 
Sieranoja, S., & Fränti, P. (2022). Adapting k-means for graph clustering. Knowledge and Information Systems, 64(1), 115–142. https://doi.org/10.1007/s10115-021-01623-y