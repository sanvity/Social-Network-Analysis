import pandas as pd
import networkx as nx
from typing import Tuple, Dict, Any


def load_bitcoin_alpha_graph(csv_path: str, positive_only: bool = False) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Loads SNAP Bitcoin Alpha edge list dataset into a NetworkX DiGraph.
    
    Format: SOURCE, TARGET, RATING, TIME
    - RATING ranges from -10 (total distrust) to +10 (absolute trust)
    """
    df = pd.read_csv(csv_path, names=["SOURCE", "TARGET", "RATING", "TIME"], header=None)

    df["SOURCE"] = df["SOURCE"].astype(int)
    df["TARGET"] = df["TARGET"].astype(int)
    df["RATING"] = df["RATING"].astype(float)

    if positive_only:
        df = df[df["RATING"] > 0].copy()

    G = nx.DiGraph()
    for _, row in df.iterrows():
        u, v, rating, timestamp = int(row["SOURCE"]), int(row["TARGET"]), float(row["RATING"]), int(row["TIME"])
        G.add_edge(u, v, rating=rating, weight=abs(rating), timestamp=timestamp)

    stats = {
        "name": "Bitcoin Alpha Trust Network" + (" (Positive Trust Subgraph)" if positive_only else ""),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": True,
        "density": nx.density(G),
        "is_strongly_connected": nx.is_strongly_connected(G),
        "is_weakly_connected": nx.is_weakly_connected(G),
        "num_positive_edges": len(df[df["RATING"] > 0]),
        "num_negative_edges": len(df[df["RATING"] < 0]),
    }
    
    return G, stats


def load_reddit_graph(tsv_path: str, top_subreddits_only: int = 2500) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Loads SNAP Reddit Hyperlink dataset into a NetworkX DiGraph.
    Aggregates multiple post links between subreddits into edge weights and average sentiments.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    
    # Normalize column names
    df.columns = [col.upper() for col in df.columns]

    agg_df = df.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]).agg(
        weight=("POST_ID", "count"),
        sentiment=("LINK_SENTIMENT", "mean")
    ).reset_index()

    if top_subreddits_only and len(agg_df) > top_subreddits_only:
        # Filter top interacting subreddit pairs
        agg_df = agg_df.nlargest(top_subreddits_only, "weight")

    G = nx.DiGraph()
    for _, row in agg_df.iterrows():
        src, tgt = row["SOURCE_SUBREDDIT"], row["TARGET_SUBREDDIT"]
        G.add_edge(src, tgt, weight=float(row["weight"]), sentiment=float(row["sentiment"]))

    stats = {
        "name": "Reddit Subreddit Hyperlink Network",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": True,
        "density": nx.density(G),
        "total_posts": len(df),
        "avg_sentiment": df["LINK_SENTIMENT"].mean() if "LINK_SENTIMENT" in df.columns else 0.0,
    }
    
    return G, stats
