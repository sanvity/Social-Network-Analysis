import pandas as pd
import networkx as nx
from typing import Dict, Tuple


def compute_all_centralities(G: nx.DiGraph, k_betweenness: int = 50) -> pd.DataFrame:
    """
    Computes major centrality metrics for a given NetworkX directed graph.
    
    Metrics:
    - In-Degree Centrality
    - Out-Degree Centrality
    - Betweenness Centrality (uses k-node sampling approximation if graph > 500 nodes)
    - Closeness Centrality
    - PageRank
    
    Returns a Pandas DataFrame indexed by Node ID with columns for each metric.
    """
    print(f"[Centrality] Computing centrality metrics for {G.number_of_nodes()} nodes...")
    
    # 1. Degree Centrality
    in_degree_cent = nx.in_degree_centrality(G)
    out_degree_cent = nx.out_degree_centrality(G)
    
    # 2. PageRank (Handles directed graphs smoothly)
    try:
        pagerank_cent = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        print("[Centrality] PageRank convergence warning. Falling back to default initial values.")
        pagerank_cent = {node: 1.0 / G.number_of_nodes() for node in G.nodes()}

    # 3. Betweenness Centrality (use sampling if node count > 500 for high performance)
    if G.number_of_nodes() > 500 and k_betweenness is not None:
        print(f"[Centrality] Graph large ({G.number_of_nodes()} nodes). Sampling {k_betweenness} nodes for Betweenness...")
        betweenness_cent = nx.betweenness_centrality(G, k=min(k_betweenness, G.number_of_nodes()), seed=42)
    else:
        betweenness_cent = nx.betweenness_centrality(G)

    # 4. Closeness Centrality
    if G.number_of_nodes() > 1000:
        top_sample_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:300]
        closeness_cent = {node: 0.0 for node in G.nodes()}
        for node in top_sample_nodes:
            closeness_cent[node] = nx.closeness_centrality(G, u=node)
    else:
        closeness_cent = nx.closeness_centrality(G)

    df = pd.DataFrame({
        "in_degree_centrality": in_degree_cent,
        "out_degree_centrality": out_degree_cent,
        "betweenness_centrality": betweenness_cent,
        "closeness_centrality": closeness_cent,
        "pagerank": pagerank_cent,
    })
    
    # Composite Influence Score (Normalized rank average)
    df["composite_rank"] = (
        df["in_degree_centrality"].rank(ascending=False) +
        df["betweenness_centrality"].rank(ascending=False) +
        df["pagerank"].rank(ascending=False)
    ) / 3.0

    return df.sort_values(by="pagerank", ascending=False)


def get_top_influencers(centrality_df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
    """Returns top N nodes for each centrality metric."""
    return {
        "top_pagerank": centrality_df.nlargest(top_n, "pagerank")[["pagerank", "in_degree_centrality"]],
        "top_betweenness": centrality_df.nlargest(top_n, "betweenness_centrality")[["betweenness_centrality", "pagerank"]],
        "top_in_degree": centrality_df.nlargest(top_n, "in_degree_centrality")[["in_degree_centrality", "out_degree_centrality"]],
    }
