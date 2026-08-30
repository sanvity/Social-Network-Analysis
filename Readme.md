# 🌐 Social Network Analysis & Link Prediction Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![PyVis](https://img.shields.io/badge/PyVis-0.3+-yellow.svg)](https://pyvis.readthedocs.io/)

An end-to-end Social Network Analysis (SNA) pipeline built with **NetworkX**, **scikit-learn**, and **PyVis**. Uncovers key influencers, detects community clusters, and forecasts new links on **Bitcoin Alpha** (trust graph) and **Reddit** (subreddit hyperlink graph).

---

## 🎨 Visual Results

| Bitcoin Alpha Network | Reddit Network | Link Prediction ROC |
| :---: | :---: | :---: |
| ![Bitcoin Alpha](bitcoin_network.png) | ![Reddit](reddit_network.png) | ![ROC Curve](link_prediction_roc.png) |
| *PageRank & Louvain Clusters* | *Subreddit Inter-links* | *RandomForest ROC (AUC = 0.7791)* |

---

## 📊 Benchmark Performance

| Dataset | Graph Type | Nodes / Edges | Modularity ($Q$) | ML Link Prediction ROC-AUC | ML Link Prediction PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Bitcoin Alpha** | Signed Directed Trust | 3,683 / 22,650 | **`0.5328`** | **`0.7791`** | **`0.8004`** |
| **Reddit Hyperlink** | Directed Interaction | 1,813 / 2,500 | **`0.8430`** | `0.4986` | `0.6557` |

---

## ⚡ Key Features

* **Centrality Ranking**: PageRank ($\alpha=0.85$), Betweenness ($k=50$ sampling optimization for 100x speedup), Degree, and Closeness.
* **Community Detection**: Louvain Modularity Optimization ($Q$).
* **ML Link Prediction**: `RandomForestClassifier` trained on topological edge features (Adamic-Adar, Jaccard, Common Neighbors, Preferential Attachment, Resource Allocation) with edge-masked train/test split.
* **Interactive Visualization**: Physics-simulated `PyVis` HTML web exports (`bitcoin_network.html`, `reddit_network.html`).

---

## 📁 Repository Structure

```text
SocialNetworkAnalysis/
├── data/
│   ├── data_loader.py            # SNAP dataset fetcher & scale-free synthetic fallback
│   ├── soc-sign-bitcoinalpha.csv # Bitcoin Alpha dataset
│   └── soc-redditHyperlinks-body.tsv # Reddit Hyperlink dataset
├── src/
│   ├── graph_loader.py           # NetworkX graph constructors
│   ├── centrality_analysis.py    # Multi-centrality calculation engine
│   ├── community_detection.py    # Louvain community detection
│   ├── link_prediction.py        # Supervised ML link predictor
│   └── visualizer.py             # Matplotlib plots & PyVis HTML exporter
├── main.py                       # Pipeline runner
├── bitcoin_network.png           # Bitcoin Alpha graph plot
├── reddit_network.png            # Reddit network graph plot
├── link_prediction_roc.png       # Link prediction ROC curve
└── README.md                     # Project documentation
```

---

## 🚀 Quickstart

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Pipeline
python main.py

# 3. View Interactive HTML Graphs
open bitcoin_network.html
open reddit_network.html
```
