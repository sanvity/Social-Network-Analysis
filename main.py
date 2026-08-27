import os
import sys
import pandas as pd
import networkx as nx

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import fetch_bitcoin_alpha, fetch_reddit_hyperlinks
from src.graph_loader import load_bitcoin_alpha_graph, load_reddit_graph
from src.centrality_analysis import compute_all_centralities, get_top_influencers
from src.community_detection import detect_communities
from src.link_prediction import train_eval_link_prediction
from src.visualizer import plot_static_network, create_interactive_pyvis_network


def run_pipeline():
    print("=" * 70)
    print("  SOCIAL NETWORK ANALYSIS PIPELINE (MatSoc - NetworkX & ML)")
    print("=" * 70)

    # -------------------------------------------------------------
    # STEP 1: Dataset Fetching & Preparation
    # -------------------------------------------------------------
    print("\n--- STEP 1: Data Ingestion ---")
    btc_path = fetch_bitcoin_alpha()
    reddit_path = fetch_reddit_hyperlinks()

    # Load NetworkX Graph Objects
    G_btc, btc_stats = load_bitcoin_alpha_graph(btc_path, positive_only=True)
    G_reddit, reddit_stats = load_reddit_graph(reddit_path)

    print(f"\n[Graph Loaded] Bitcoin Alpha: {btc_stats['num_nodes']} nodes, {btc_stats['num_edges']} edges, Density: {btc_stats['density']:.5f}")
    print(f"[Graph Loaded] Reddit Network: {reddit_stats['num_nodes']} nodes, {reddit_stats['num_edges']} edges, Density: {reddit_stats['density']:.5f}")

    # -------------------------------------------------------------
    # STEP 2: Centrality Analysis & Key Influencer Identification
    # -------------------------------------------------------------
    print("\n--- STEP 2: Centrality Analysis & Key Influencer Identification ---")
    print("\n> Bitcoin Alpha Network Centralities:")
    centrality_btc = compute_all_centralities(G_btc)
    top_btc = get_top_influencers(centrality_btc, top_n=5)
    print("\nTop PageRank Users (Bitcoin Alpha):")
    print(top_btc["top_pagerank"])
    print("\nTop Betweenness Brokers (Bitcoin Alpha):")
    print(top_btc["top_betweenness"])

    print("\n> Reddit Subreddit Network Centralities:")
    centrality_reddit = compute_all_centralities(G_reddit)
    top_reddit = get_top_influencers(centrality_reddit, top_n=5)
    print("\nTop PageRank Subreddits (Reddit):")
    print(top_reddit["top_pagerank"])

    # -------------------------------------------------------------
    # STEP 3: Community Detection & Group Dynamics
    # -------------------------------------------------------------
    print("\n--- STEP 3: Community Detection & Clustering ---")
    comm_map_btc, comm_stats_btc = detect_communities(G_btc)
    print(f"Bitcoin Alpha Communities: {comm_stats_btc['num_communities']} clusters | Modularity Q = {comm_stats_btc['modularity_q']}")

    comm_map_reddit, comm_stats_reddit = detect_communities(G_reddit)
    print(f"Reddit Subreddit Communities: {comm_stats_reddit['num_communities']} clusters | Modularity Q = {comm_stats_reddit['modularity_q']}")

    # -------------------------------------------------------------
    # STEP 4: Link Prediction Engine (Graph Theory & Supervised ML)
    # -------------------------------------------------------------
    print("\n--- STEP 4: Link Prediction Algorithms ---")
    print("\n> Evaluating Supervised Link Predictor on Bitcoin Alpha Trust Graph:")
    lp_metrics_btc, forecast_btc = train_eval_link_prediction(G_btc)
    print("\nTop 5 Forecasted New Trust Connections (Bitcoin Alpha):")
    print(forecast_btc.head(5).to_string(index=False))

    print("\n> Evaluating Supervised Link Predictor on Reddit Subreddit Hyperlink Network:")
    lp_metrics_reddit, forecast_reddit = train_eval_link_prediction(G_reddit)
    print("\nTop 5 Forecasted Subreddit Cross-Links (Reddit):")
    print(forecast_reddit.head(5).to_string(index=False))

    # -------------------------------------------------------------
    # STEP 5: Visualizing Network Dynamics
    # -------------------------------------------------------------
    print("\n--- STEP 5: Exporting Static Plots & Interactive HTML Visualizations ---")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Bitcoin Alpha Visualizations
    plot_static_network(
        G_btc, centrality_btc, comm_map_btc,
        title="Bitcoin Alpha Trust Network Dynamics",
        output_path=os.path.join(output_dir, "bitcoin_network.png")
    )
    create_interactive_pyvis_network(
        G_btc, centrality_btc, comm_map_btc,
        output_html_path=os.path.join(output_dir, "bitcoin_network.html")
    )

    # Reddit Visualizations
    plot_static_network(
        G_reddit, centrality_reddit, comm_map_reddit,
        title="Reddit Subreddit Hyperlink Network Dynamics",
        output_path=os.path.join(output_dir, "reddit_network.png")
    )
    create_interactive_pyvis_network(
        G_reddit, centrality_reddit, comm_map_reddit,
        output_html_path=os.path.join(output_dir, "reddit_network.html")
    )

    print("\n" + "=" * 70)
    print("  ANALYSIS PIPELINE EXECUTED SUCCESSFULLY!")
    print("  Generated Artifacts:")
    print("  - Static Plots: bitcoin_network.png, reddit_network.png")
    print("  - Interactive HTML Maps: bitcoin_network.html, reddit_network.html")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
