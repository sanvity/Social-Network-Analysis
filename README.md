# Social Network Analysis (MatSoc)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

An end-to-end Social Network Analysis pipeline built with **NetworkX**, **Python**, **scikit-learn**, and **PyVis**. Designed to uncover key influencers, detect community partitions, and predict new connections on benchmark social graph datasets including **Bitcoin Alpha** (trust network) and **Reddit** (subreddit hyperlink network).

---

## 📌 Features & Highlights

1. **Centrality Measures & Key Influencer Analysis**:
   - **Degree Centrality** ($C_D$): Identifies direct connection volume and activity.
   - **Betweenness Centrality** ($C_B$): Locates structural bottlenecks and information bridges across clusters.
   - **Closeness Centrality** ($C_C$): Quantifies reachability and information broadcast speed.
   - **PageRank / Eigenvector Centrality**: Evaluates global authority based on connection quality.

2. **Community Detection & Group Dynamics**:
   - **Louvain Modularity Optimization**: Maximizes the modularity score ($Q$) to partition networks into dense sub-communities.
   - Computes modularity score $Q \in [-0.5, 1.0]$ and size distributions across clusters.

3. **Link Prediction Engine**:
   - **Graph Similarity Metrics**: Common Neighbors, Jaccard Coefficient, Adamic-Adar Index, Preferential Attachment, Resource Allocation Index.
   - **Supervised Machine Learning Pipeline**: Extracts topological edge features, samples negative pairs, and trains a `RandomForestClassifier` evaluated via **ROC-AUC** and **PR-AUC**.

4. **Network Visualization**:
   - **Static High-Res Plots**: Matplotlib Fruchterman-Reingold layout with node radius scaled by PageRank and color-coded by community partition.
   - **Dynamic Interactive Maps**: PyVis HTML exports allowing zooming, node hovering, physics simulation, and property inspection.

---

## 📐 Mathematical Overview

### 1. Centrality Equations
- **Betweenness Centrality**:
  $$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
- **Closeness Centrality**:
  $$C_C(v) = \frac{N - 1}{\sum_{u \neq v} d(v, u)}$$

### 2. Link Prediction Metrics
- **Jaccard Coefficient**: $\text{Jaccard}(u, v) = \frac{|\Gamma(u) \cap \Gamma(v)|}{|\Gamma(u) \cup \Gamma(v)|}$
- **Adamic-Adar Index**: $\text{AA}(u, v) = \sum_{z \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log(\deg(z))}$
- **Resource Allocation Index**: $\text{RA}(u, v) = \sum_{z \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\deg(z)}$

---

## 📂 Project Structure

```text
SocialNetworkAnalysis/
├── data/
│   └── data_loader.py         # Automates SNAP dataset downloads & synthetic fallbacks
├── src/
│   ├── graph_loader.py        # Graph construction for directed/signed networks
│   ├── centrality_analysis.py # Centrality metrics calculator & influencer ranker
│   ├── community_detection.py # Louvain modularity optimization & group discovery
│   ├── link_prediction.py     # Graph heuristics & ML link classifier
│   └── visualizer.py          # Static PNG plots & interactive PyVis HTML exports
├── main.py                    # End-to-end execution script
├── requirements.txt           # Dependency requirements
└── README.md                  # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Analysis Pipeline
```bash
python main.py
```

### 3. Explore Outputs
- **Static Plots**: `bitcoin_network.png`, `reddit_network.png`
- **Interactive Graphs**: Open `bitcoin_network.html` or `reddit_network.html` in any web browser.
