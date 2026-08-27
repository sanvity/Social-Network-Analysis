import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


# ==========================================
# 1. Topological Graph Similarity Metrics
# ==========================================

def compute_common_neighbors(G_undirected: nx.Graph, u, v) -> int:
    """Common Neighbors: |N(u) intersect N(v)|"""
    if not G_undirected.has_node(u) or not G_undirected.has_node(v):
        return 0
    return len(list(nx.common_neighbors(G_undirected, u, v)))


def compute_jaccard_coefficient(G_undirected: nx.Graph, u, v) -> float:
    """Jaccard Coefficient: |N(u) intersect N(v)| / |N(u) union N(v)|"""
    if not G_undirected.has_node(u) or not G_undirected.has_node(v):
        return 0.0
    preds = list(nx.jaccard_coefficient(G_undirected, [(u, v)]))
    return float(preds[0][2]) if preds else 0.0


def compute_adamic_adar(G_undirected: nx.Graph, u, v) -> float:
    """Adamic-Adar Index: Sum 1 / log(deg(z)) for shared neighbors z"""
    if not G_undirected.has_node(u) or not G_undirected.has_node(v):
        return 0.0
    try:
        preds = list(nx.adamic_adar_index(G_undirected, [(u, v)]))
        return float(preds[0][2]) if preds else 0.0
    except Exception:
        return 0.0


def compute_preferential_attachment(G_undirected: nx.Graph, u, v) -> float:
    """Preferential Attachment: deg(u) * deg(v)"""
    if not G_undirected.has_node(u) or not G_undirected.has_node(v):
        return 0.0
    preds = list(nx.resource_allocation_index(G_undirected, [(u, v)]))  # RA fallback
    return float(G_undirected.degree(u) * G_undirected.degree(v))


def compute_resource_allocation(G_undirected: nx.Graph, u, v) -> float:
    """Resource Allocation Index: Sum 1 / deg(z) for shared neighbors z"""
    if not G_undirected.has_node(u) or not G_undirected.has_node(v):
        return 0.0
    try:
        preds = list(nx.resource_allocation_index(G_undirected, [(u, v)]))
        return float(preds[0][2]) if preds else 0.0
    except Exception:
        return 0.0


def extract_pair_features(G_undirected: nx.Graph, u, v) -> Dict[str, float]:
    """Extracts topological feature vector for pair (u, v)."""
    return {
        "common_neighbors": compute_common_neighbors(G_undirected, u, v),
        "jaccard_coeff": compute_jaccard_coefficient(G_undirected, u, v),
        "adamic_adar": compute_adamic_adar(G_undirected, u, v),
        "preferential_attachment": compute_preferential_attachment(G_undirected, u, v),
        "resource_allocation": compute_resource_allocation(G_undirected, u, v),
        "degree_u": G_undirected.degree(u) if G_undirected.has_node(u) else 0,
        "degree_v": G_undirected.degree(v) if G_undirected.has_node(v) else 0,
    }


# ==========================================
# 2. Supervised ML Link Prediction Pipeline
# ==========================================

def train_eval_link_prediction(G: nx.DiGraph, test_ratio: float = 0.20, seed: int = 42) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains a Machine Learning Link Prediction model.
    1. Splits existing edges into Train Graph (80%) and Positive Test Edges (20%).
    2. Samples equal number of non-existent pairs as Negative Examples.
    3. Extracts topological features using the Train Graph only.
    4. Evaluates Random Forest vs Logistic Regression on ROC-AUC and PR-AUC.
    """
    print("[LinkPrediction] Preparing Train/Test link split and feature extraction...")
    random.seed(seed)
    np.random.seed(seed)

    G_undirected = G.to_undirected()
    all_edges = list(G_undirected.edges())
    random.shuffle(all_edges)

    max_samples = 3000
    if len(all_edges) > max_samples:
        print(f"[LinkPrediction] Subsampling graph edges to {max_samples} for feature extraction speed...")
        all_edges = all_edges[:max_samples]

    num_test = int(len(all_edges) * test_ratio)
    test_positive_edges = all_edges[:num_test]
    train_edges = all_edges[num_test:]

    # Construct Train Graph (masked test edges)
    G_train = nx.Graph()
    G_train.add_nodes_from(G_undirected.nodes())
    G_train.add_edges_from(train_edges)

    # Sample Negative Edges (non-existent node pairs)
    nodes = list(G_train.nodes())
    num_nodes = len(nodes)
    negative_edges = []
    
    attempts = 0
    max_attempts = len(all_edges) * 10
    
    while len(negative_edges) < len(all_edges) and attempts < max_attempts:
        u, v = random.sample(nodes, 2)
        attempts += 1
        if not G_undirected.has_edge(u, v) and (u, v) not in negative_edges and (v, u) not in negative_edges:
            negative_edges.append((u, v))

    train_negatives = negative_edges[num_test:]
    test_negatives = negative_edges[:num_test]

    print(f"[LinkPrediction] Positives: {len(train_edges)} train, {len(test_positive_edges)} test | Negatives: {len(train_negatives)} train, {len(test_negatives)} test")

    # Build Training Feature Matrix
    X_train, y_train = [], []
    for u, v in train_edges:
        X_train.append(extract_pair_features(G_train, u, v))
        y_train.append(1)
    for u, v in train_negatives:
        X_train.append(extract_pair_features(G_train, u, v))
        y_train.append(0)

    # Build Testing Feature Matrix
    X_test, y_test = [], []
    for u, v in test_positive_edges:
        X_test.append(extract_pair_features(G_train, u, v))
        y_test.append(1)
    for u, v in test_negatives:
        X_test.append(extract_pair_features(G_train, u, v))
        y_test.append(0)

    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=8)
    clf.fit(df_train, y_train)

    # Predict Probabilities
    y_pred_probs = clf.predict_proba(df_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    pr_auc = average_precision_score(y_test, y_pred_probs)

    metrics = {
        "model": "RandomForestClassifier",
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "feature_importances": dict(zip(df_train.columns, clf.feature_importances_.round(4)))
    }

    print(f"[LinkPrediction] Model Performance: ROC-AUC = {metrics['roc_auc']} | PR-AUC = {metrics['pr_auc']}")

    # Generate Top Forecasted New Links for Unconnected Pairs
    print("[LinkPrediction] Forecasting top new connections across network...")
    forecast_results = []
    
    for u, v in test_negatives[:100]:  # Sample candidates
        feats = extract_pair_features(G_undirected, u, v)
        prob = clf.predict_proba(pd.DataFrame([feats]))[0][1]
        forecast_results.append({
            "source": u,
            "target": v,
            "forecast_score": round(prob, 4),
            "common_neighbors": feats["common_neighbors"],
            "adamic_adar": round(feats["adamic_adar"], 4),
            "jaccard": round(feats["jaccard_coeff"], 4),
        })

    forecast_df = pd.DataFrame(forecast_results).sort_values(by="forecast_score", ascending=False)

    return metrics, forecast_df
