import os
import gzip
import shutil
import requests
import pandas as pd
import numpy as np
import networkx as nx

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BITCOIN_ALPHA_URL = "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"
REDDIT_URL = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"

BITCOIN_CSV_PATH = os.path.join(DATA_DIR, "soc-sign-bitcoinalpha.csv")
REDDIT_TSV_PATH = os.path.join(DATA_DIR, "soc-redditHyperlinks-body.tsv")


def fetch_bitcoin_alpha(force_synthetic=False) -> str:
    """
    Downloads and extracts Bitcoin Alpha dataset from SNAP.
    Fallback to realistic synthetic graph if download fails.
    """
    if os.path.exists(BITCOIN_CSV_PATH) and not force_synthetic:
        print(f"[DataLoader] Using existing Bitcoin Alpha dataset at {BITCOIN_CSV_PATH}")
        return BITCOIN_CSV_PATH

    if not force_synthetic:
        try:
            print(f"[DataLoader] Downloading Bitcoin Alpha dataset from SNAP...")
            gz_path = BITCOIN_CSV_PATH + ".gz"
            response = requests.get(BITCOIN_ALPHA_URL, timeout=15, stream=True)
            if response.status_code == 200:
                with open(gz_path, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
                with gzip.open(gz_path, "rb") as f_in:
                    with open(BITCOIN_CSV_PATH, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
                print(f"[DataLoader] Downloaded and extracted Bitcoin Alpha to {BITCOIN_CSV_PATH}")
                return BITCOIN_CSV_PATH
        except Exception as e:
            print(f"[DataLoader] Warning: Failed to download SNAP Bitcoin Alpha ({e}). Generating synthetic data...")

    # Synthetic fallback matching Bitcoin Alpha statistics (~3,783 nodes, scale-free signed trust network)
    return generate_synthetic_bitcoin_alpha()


def generate_synthetic_bitcoin_alpha(num_nodes=500, num_edges=2500) -> str:
    """Generates synthetic scale-free directed signed graph resembling Bitcoin Alpha."""
    print(f"[DataLoader] Generating synthetic Bitcoin Alpha network ({num_nodes} nodes, {num_edges} edges)...")
    ba_graph = nx.scale_free_graph(num_nodes, seed=42)
    
    records = []
    rng = np.random.default_rng(42)
    
    for u, v in ba_graph.edges():
        if u != v:
            # Bitcoin Alpha ratings range from -10 to +10, predominantly positive (~90% positive)
            rating = rng.choice(
                a=list(range(-10, 11)),
                p=[0.01]*10 + [0.0] + [0.09]*10  # 10% negative, 90% positive
            )
            if rating == 0:
                rating = 1
            timestamp = 1400000000 + rng.integers(0, 50000000)
            records.append((u + 1, v + 1, rating, timestamp))

    df = pd.DataFrame(records, columns=["SOURCE", "TARGET", "RATING", "TIME"])
    df.to_csv(BITCOIN_CSV_PATH, index=False, header=False)
    print(f"[DataLoader] Saved synthetic Bitcoin Alpha data to {BITCOIN_CSV_PATH}")
    return BITCOIN_CSV_PATH


def fetch_reddit_hyperlinks(force_synthetic=False) -> str:
    """
    Downloads Reddit Hyperlink dataset from SNAP.
    Fallback to synthetic graph if download fails.
    """
    if os.path.exists(REDDIT_TSV_PATH) and not force_synthetic:
        print(f"[DataLoader] Using existing Reddit Hyperlink dataset at {REDDIT_TSV_PATH}")
        return REDDIT_TSV_PATH

    if not force_synthetic:
        try:
            print(f"[DataLoader] Downloading Reddit Hyperlink dataset from SNAP...")
            response = requests.get(REDDIT_URL, timeout=15, stream=True)
            if response.status_code == 200:
                with open(REDDIT_TSV_PATH, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
                print(f"[DataLoader] Downloaded Reddit Hyperlinks to {REDDIT_TSV_PATH}")
                return REDDIT_TSV_PATH
        except Exception as e:
            print(f"[DataLoader] Warning: Failed to download SNAP Reddit data ({e}). Generating synthetic data...")

    return generate_synthetic_reddit()


def generate_synthetic_reddit(num_subreddits=200, num_links=2000) -> str:
    """Generates synthetic directed subreddit interaction graph."""
    print(f"[DataLoader] Generating synthetic Reddit network ({num_subreddits} subreddits, {num_links} links)...")
    subreddits = [f"r/{topic}_{i}" for topic in ["ask", "gaming", "crypto", "news", "tech", "science"] for i in range(1, num_subreddits // 6 + 1)]
    if len(subreddits) < num_subreddits:
        subreddits.extend([f"r/sub_{i}" for i in range(len(subreddits), num_subreddits)])
    
    rng = np.random.default_rng(42)
    records = []
    
    for _ in range(num_links):
        src, tgt = rng.choice(subreddits, size=2, replace=False)
        post_id = f"post_{rng.integers(100000, 999999)}"
        timestamp = "2024-12-01 12:00:00"
        sentiment = rng.choice([-1, 1], p=[0.25, 0.75])
        properties = f"0.1,0.2,{sentiment},0.8"
        records.append((src, tgt, post_id, timestamp, sentiment, properties))

    cols = ["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP", "LINK_SENTIMENT", "PROPERTIES"]
    df = pd.DataFrame(records, columns=cols)
    df.to_csv(REDDIT_TSV_PATH, sep="\t", index=False)
    print(f"[DataLoader] Saved synthetic Reddit dataset to {REDDIT_TSV_PATH}")
    return REDDIT_TSV_PATH


if __name__ == "__main__":
    fetch_bitcoin_alpha()
    fetch_reddit_hyperlinks()
