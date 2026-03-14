# Renewing Innovation Systems: Knowledge Pathways Structuring Technological Development

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-1.0+-green.svg)](https://www.dgl.ai/)

> **Paper under review** — *Journal of Innovation and Knowledge* (Elsevier)

This repository contains the code, model artefacts, embeddings, and results for the paper:

> Lagos, E. E. (2025). *Renewing Innovation Systems: Knowledge Pathways Structuring Technological Development*. Under review at Journal of Innovation and Knowledge.

---

## Overview

This project operationalizes **knowledge pathways** — recurrent relational routes connecting heterogeneous actors and artifacts — as the meso-level mechanism through which innovation systems organize learning-based technological development.

We model the European innovation system as a **heterogeneous graph** (96,921 firms, EU27+UK, medium-high and high-tech sectors) and apply pathway-sensitive representation learning (GeniZ) to identify which relational channels exhibit system-structuring capacity.

### Key results

| Metric | Baseline (10 mp) | Final model (7 mp) | Δ |
|--------|------------------|--------------------|---|
| Spearman ρ | 0.664 | **0.678** | +0.014 |
| NDCG@100 | 0.438 | **0.481** | +0.044 |
| Combined | 0.551 | **0.580** | +0.029 |
| RBP (p=0.95) | 0.078 | **0.080** | +0.002 |

**Core–periphery:** 325 firms (0.3%) form the structural core, with a mean z-score of 5.247 — approximately 64× the semi-periphery mean.

---

## Repository structure
```
renewing-innovation-system/
│
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
│
├── 01_GeniZ_PathwaySelection_and_SystemContour.ipynb
│                                    ← Main model: GeniZ architecture, training,
│                                       pathway selection, PFI, core-periphery
├── 02_GeniZ_TrimmingExperiments_MetapathComparison.ipynb
│                                    ← Trimming experiments (Exp-A, B, C)
│                                       comparing 10mp → 9mp → 8mp → 7mp
│
├── data/                            ← Large files hosted on Zenodo (see below)
│   ├── graph_FullNodes.dgl          ← Heterogeneous DGL graph (6 node types)
│   ├── diccionarios/
│   │   ├── entity2idx.pkl           ← Global node ID → integer index
│   │   └── entity2idx_rev.pkl       ← Reverse mapping
│   └── embeddings/
│       ├── metapath_2_emb.pkl       ← Firm–FoS–FoS–Firm
│       ├── metapath_4_emb.pkl       ← Firm–Patent–Country–Patent–Firm ★
│       ├── metapath_5_emb.pkl       ← Firm–Product–FoS–Product–Firm
│       ├── metapath_7_emb.pkl       ← Firm–Country–Firm
│       ├── metapath_8_emb.pkl       ← Firm–Univ–Country–Univ–Firm
│       ├── metapath_10_emb.pkl      ← FoS–Product–FoS
│       └── metapath_14_emb.pkl      ← FoS–FoS–FoS ★
│           (★ = backbone pathways; MP4 attention=0.405, MP14 attention=0.237)
│
└── results/
    ├── core_periphery_expc.csv      ← GMM zone assignments (96,817 firms)
    ├── summary_final_expc.json      ← Full metrics, attention weights, PFI
    └── figures/
        ├── Fig1_HeterogeneousGraph.png
        ├── Fig2_Metapath2vec.png
        ├── Fig3_GeniZ.png
        ├── comparison_chart.png
        └── core_periphery_expc.png
```

---

## Data and large artefacts (Zenodo)

Due to file size constraints, the following artefacts are hosted on Zenodo:

📦 **[Download from Zenodo — DOI: 10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)**

| File | Size (approx.) | Description |
|------|---------------|-------------|
| `graph_FullNodes.dgl` | ~500 MB | Full heterogeneous DGL graph |
| `embeddings/*.pkl` | ~200 MB total | Metapath2vec embeddings (7 final metapaths) |
| `diccionarios/*.pkl` | ~50 MB | Node ID dictionaries |
| `pagerank_zscores.pkl` | ~30 MB | PageRank z-score signal (training target) |
| `GeniZ_Final_checkpoint.pt` | ~25 MB | Trained GeniZ Final model weights |

To reproduce the full pipeline, download the Zenodo archive and place files in the `data/` directory following the structure above.

---

## Embeddings: methodology and attribution

Metapath-based embeddings were generated using the **metapath2vec** algorithm:

> Dong, Y., Chawla, N. V., & Swami, A. (2017). metapath2vec: Scalable representation learning for heterogeneous networks. *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 135–144. ACM. https://doi.org/10.1145/3097983.3098036

**Walk parameters:**
- Walk length: 100 steps per schema
- Number of walks per node: 5
- Context window size: 5
- Embedding dimension: 128
- Negative samples: 5
- Optimization: Adam, 3 epochs

One embedding file is generated per metapath schema. Each `.pkl` file contains a dictionary `{node_id: np.array(128,)}` covering all nodes reachable under that schema.

---

## GeniZ model

GeniZ is an adaptation of **GENI** (Graph Embedding for Node Importance) for heterogeneous graphs with metapath-aware message passing:

> Park, N., Kan, A., Dong, X. L., Zhao, T., & Faloutsos, C. (2019). Estimating node importance in knowledge graphs using graph neural networks. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 596–606. ACM. https://doi.org/10.1145/3292500.3330855

The adaptation repurposes node-importance estimation to evaluate **pathway-level structural contribution** in heterogeneous innovation graphs. Key modifications:
- Input: metapath-specific embeddings (one per pathway schema) instead of single-channel node features
- Attention: masked softmax over active metapaths per node (handles missing coverage)
- Target: PageRank z-scores (contextual structural importance, representation-dependent)
- Evaluation: Spearman ρ, NDCG@100, RBP(p=0.95), RBP(p=0.80)

---

## Data source

The heterogeneous graph is built from:

> Ashouri, S., Suominen, A., Hajikhani, A., Pukelis, L., Schubert, T., Türkeli, S., Van Beers, C., & Cunningham, S. (2022). Indicators on firm level innovation activities from web scraped data. *Data in Brief*, 42, 108246. https://doi.org/10.34894/BS9XVR

- **Coverage:** 96,921 firms, EU27 + United Kingdom
- **Sectors:** Medium-high and high-technology manufacturing (Eurostat NACE Rev. 2)
- **Sources linked:** Orbis (firm IDs), PATSTAT (patents), Microsoft Academic Graph 2019 (fields of study)
- **Structure:** Single cross-section (no panel dimension)

---

## Reproducibility

### Requirements
```bash
pip install -r requirements.txt
```

### Quickstart (with Zenodo data)
```python
# 1. Load the graph
import dgl
g = dgl.load_graphs('data/graph_FullNodes.dgl')[0][0]

# 2. Load embeddings
import pickle
with open('data/embeddings/metapath_4_emb.pkl', 'rb') as f:
    mp4_emb = pickle.load(f)   # {node_id: np.array(128,)}

# 3. Load dictionaries
with open('data/diccionarios/entity2idx.pkl', 'rb') as f:
    entity2idx = pickle.load(f)

# 4. Run notebook 01 for full pipeline
```

### Notebooks

| Notebook | Purpose | Runtime (GPU) |
|----------|---------|---------------|
| `01_GeniZ_PathwaySelection_and_SystemContour.ipynb` | Full pipeline: data loading → GeniZ Lite → branch selection → GeniZ Final → PFI → GMM | ~2–3 hours |
| `02_GeniZ_TrimmingExperiments_MetapathComparison.ipynb` | Trimming experiments comparing 10mp vs 9mp vs 8mp vs 7mp | ~4–6 hours |

Both notebooks are designed to run on **Google Colab Pro** with GPU acceleration. Set `BASE_DIR` to your Google Drive path.

---

## Selected pathway results

| Pathway | Schema | Branch | PFI NDCG drop | Attention weight |
|---------|--------|--------|---------------|-----------------|
| MP4  | Firm–Patent–Country–Patent–Firm | B | 94.12% | 0.405 |
| MP14 | FoS–FoS–FoS | A | 14.65% | 0.237 |
| MP7  | Firm–Country–Firm | B | 6.76% | 0.105 |
| MP2  | Firm–FoS–FoS–Firm | A | 2.81% | 0.033 |
| MP8  | Firm–Univ–Country–Univ–Firm | B | 0.48% | 0.051 |
| MP10 | FoS–Product–FoS | A | −1.46% | 0.039 |
| MP5  | Firm–Product–FoS–Product–Firm | A | −22.43% | 0.129 |

---

## Citation

If you use this code, data, or results, please cite:
```bibtex
@article{lagos2025renewing,
  title   = {Renewing Innovation Systems: Knowledge Pathways Structuring Technological Development},
  author  = {Lagos, Edgardo E.},
  journal = {Journal of Innovation and Knowledge},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

- **Code:** MIT License
- **Data and results:** Creative Commons Attribution 4.0 (CC BY 4.0)

---

## Contact

Edgardo E. Lagos — [GitHub: @EdgLag](https://github.com/EdgLag)
```
