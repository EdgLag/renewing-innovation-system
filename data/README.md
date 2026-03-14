# Data

Large files (graph, embeddings, dictionaries) are hosted on Zenodo.

Download: https://doi.org/10.5281/zenodo.XXXXXXX

## Files expected here

- `graph_FullNodes.dgl` — Heterogeneous DGL graph (6 node types, 1.5M nodes)
- `pagerank_zscores.pkl` — PageRank z-score signal (training target)
- `diccionarios/entity2idx.pkl` — Global node ID → integer index
- `diccionarios/entity2idx_rev.pkl` — Reverse mapping
- `embeddings/metapath_2_emb.pkl` — Firm–FoS–FoS–Firm
- `embeddings/metapath_4_emb.pkl` — Firm–Patent–Country–Patent–Firm ★
- `embeddings/metapath_5_emb.pkl` — Firm–Product–FoS–Product–Firm
- `embeddings/metapath_7_emb.pkl` — Firm–Country–Firm
- `embeddings/metapath_8_emb.pkl` — Firm–Univ–Country–Univ–Firm
- `embeddings/metapath_10_emb.pkl` — FoS–Product–FoS
- `embeddings/metapath_14_emb.pkl` — FoS–FoS–FoS ★
- `GeniZ_Final_checkpoint.pt` — Trained model weights

★ Backbone pathways (MP4 attention=0.405, MP14 attention=0.237)
