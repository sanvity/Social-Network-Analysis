import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Any
from pyvis.network import Network


def plot_static_network(
    G: nx.DiGraph,
    centrality_df: pd.DataFrame,
    community_map: Dict[Any, int],
    title: str = "Social Graph Dynamics",
    output_path: str = "network_plot.png",
    max_nodes: int = 150
):
    """
    Generates a static high-resolution network visualizer using Matplotlib.
    - Node size scaled by PageRank
    - Node color mapped to Community ID
    """
    print(f"[Visualizer] Plotting static network visualization ({output_path})...")
    
    # Subgraph for display clarity if graph is large
    if G.number_of_nodes() > max_nodes:
        top_nodes = centrality_df.nlargest(max_nodes, "pagerank").index
        sub_G = G.subgraph(top_nodes).copy()
    else:
        sub_G = G.copy()

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(sub_G, seed=42, k=0.15, iterations=50)

    # Community Colors
    communities = [community_map.get(node, 0) for node in sub_G.nodes()]
    
    # Node Sizes scaled by PageRank
    pr_vals = [centrality_df.loc[node, "pagerank"] if node in centrality_df.index else 0.001 for node in sub_G.nodes()]
    node_sizes = [max(100, val * 8000) for val in pr_vals]

    nx.draw_networkx_nodes(
        sub_G, pos,
        node_size=node_sizes,
        node_color=communities,
        cmap=plt.cm.tab20,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.5
    )

    nx.draw_networkx_edges(
        sub_G, pos,
        alpha=0.25,
        edge_color="gray",
        arrows=True,
        arrowsize=8
    )

    # Label top 10 influencers
    top_10 = centrality_df.nlargest(10, "pagerank").index
    labels = {node: str(node) for node in sub_G.nodes() if node in top_10}
    nx.draw_networkx_labels(sub_G, pos, labels=labels, font_size=9, font_weight="bold", font_color="darkblue")

    plt.title(f"{title}\n(Node Size: PageRank | Node Color: Community Partition)", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved static plot to {output_path}")


def create_interactive_pyvis_network(
    G: nx.DiGraph,
    centrality_df: pd.DataFrame,
    community_map: Dict[Any, int],
    output_html_path: str = "interactive_network.html",
    max_nodes: int = 200
):
    """
    Creates an interactive HTML dynamic network visualization powered by PyVis.
    Allows zooming, panning, physics simulation, and node hovering.
    """
    print(f"[Visualizer] Generating dynamic interactive PyVis graph ({output_html_path})...")
    
    if G.number_of_nodes() > max_nodes:
        top_nodes = centrality_df.nlargest(max_nodes, "pagerank").index
        sub_G = G.subgraph(top_nodes).copy()
    else:
        sub_G = G.copy()

    net = Network(height="750px", width="100%", bgcolor="#1a1a1a", font_color="white", directed=True)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    palette = [
        "#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#1D3557",
        "#A8DADC", "#E76F51", "#9B5DE5", "#F15BB5", "#00BBF9"
    ]

    for node in sub_G.nodes():
        comm_id = community_map.get(node, 0)
        color = palette[comm_id % len(palette)]
        
        pr = centrality_df.loc[node, "pagerank"] if node in centrality_df.index else 0.001
        in_deg = centrality_df.loc[node, "in_degree_centrality"] if node in centrality_df.index else 0.0
        betweenness = centrality_df.loc[node, "betweenness_centrality"] if node in centrality_df.index else 0.0
        
        title_hover = (
            f"Node: {node}\n"
            f"Community ID: {comm_id}\n"
            f"PageRank: {pr:.4f}\n"
            f"In-Degree Cent: {in_deg:.4f}\n"
            f"Betweenness Cent: {betweenness:.4f}"
        )
        
        size = max(8, int(pr * 400))
        net.add_node(str(node), label=str(node), title=title_hover, color=color, size=size)

    for u, v, data in sub_G.edges(data=True):
        weight = data.get("weight", 1.0)
        net.add_edge(str(u), str(v), value=float(weight), alpha=0.3)

    net.save_graph(output_html_path)
    print(f"[Visualizer] Interactive network export complete: {output_html_path}")
