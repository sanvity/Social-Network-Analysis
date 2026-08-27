import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
from typing import Dict, Any, Tuple, List


def detect_communities(G: nx.DiGraph) -> Tuple[Dict[Any, int], Dict[str, Any]]:
    """
    Performs community detection using Louvain Modularity Optimization.
    Converts directed graph to undirected for community structure extraction.
    
    Returns:
    - node_to_community: Dictionary mapping Node ID -> Community ID
    - stats: Modularity Q score, community counts, and sizes
    """
    print(f"[CommunityDetection] Running community detection on {G.number_of_nodes()} nodes...")
    
    # Convert to undirected graph for community detection algorithms
    G_undirected = G.to_undirected()

    try:
        # Louvain Modularity Maximization
        communities_list = nx_comm.louvain_communities(G_undirected, seed=42)
        algorithm_used = "Louvain Modularity"
    except Exception as e:
        print(f"[CommunityDetection] Louvain failed ({e}), falling back to Label Propagation...")
        communities_generator = nx_comm.label_propagation_communities(G_undirected)
        communities_list = [list(c) for c in communities_generator]
        algorithm_used = "Label Propagation"

    # Compute Modularity Score (Q)
    try:
        modularity_score = nx_comm.modularity(G_undirected, communities_list)
    except Exception:
        modularity_score = 0.0

    # Build node -> community mapping
    node_to_community = {}
    community_sizes = []
    
    for comm_id, nodes in enumerate(communities_list):
        community_sizes.append(len(nodes))
        for node in nodes:
            node_to_community[node] = comm_id

    stats = {
        "algorithm": algorithm_used,
        "num_communities": len(communities_list),
        "modularity_q": round(modularity_score, 4),
        "largest_community_size": max(community_sizes) if community_sizes else 0,
        "smallest_community_size": min(community_sizes) if community_sizes else 0,
        "avg_community_size": round(sum(community_sizes) / len(community_sizes), 2) if community_sizes else 0,
    }
    
    print(f"[CommunityDetection] Detected {stats['num_communities']} communities using {algorithm_used} (Modularity Q = {stats['modularity_q']})")
    
    return node_to_community, stats
