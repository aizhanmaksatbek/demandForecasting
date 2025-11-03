# Explainable Demand Forecasting for Retailers using GNN and TFT
- Graph Neural Networks (GNN)
- Temporal Fusion Transformers (TFT)

## Virtual Environment
``` bash
python -m venv venv_demand #Create
source venv_demand/bin/activate #Activate
```

# Install the Requirements
```bash
pip install -r requirements.txt
```

# Data source
Favorita stores located in Ecuador - https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

# End-to-End Temporal Fusion Transformer (TFT) — Favorita Store Sales

This repository provides a complete pipeline for training and evaluating a Temporal Fusion Transformer on the Kaggle "Store Sales - Time Series Forecasting" (Favorita) dataset.

What’s included
- Data preparation: merges train, stores, transactions, oil, holidays; adds calendar/holiday features.
- Model: single-file TFT implementation with variable selection, GRNs, LSTM backbone, attention, and quantile head.
- Training: time-based splits, sliding-window dataset, quantile loss optimization.
- Evaluation: Pinball loss, WAPE, sMAPE on validation and test splits.

Folder structure
- data/
  - raw/           ← place Kaggle CSVs here
  - processed/     ← generated panel dataset (panel.csv)
- src/
  - data/
    - preprocess_favorita.py
    - dataset.py
  - models/
    - tft.py
  - train.py

Quickstart
1) Install
   ```
   pip install -r requirements.txt
   ```

2) Download Kaggle Favorita data into data/raw:
   - train.csv, stores.csv, items.csv, oil.csv, transactions.csv, holidays_events.csv

3) Preprocess
   ```
   python src/data/preprocess_favorita.py
   ```
   Output: data/processed/panel.csv

4) Train and evaluate
   ```
   python src/train.py --enc-len 56 --dec-len 28 --epochs 20 --batch-size 256 --lr 1e-3
   ```
   - Uses last 28 days for validation and last 28 days for test (rolling, non-overlapping).

Notes
- Known-future features: onpromotion, holidays, calendar fields (dow/month/weekofyear).
- Observed-past features: sales, transactions, oil price.
- Static features: store_nbr, family, state, cluster (one-hot).

You can adjust variables in src/train.py (enc_vars, dec_vars, static columns) as needed.



# Graph Neural Network for Multi‑horizon Demand Forecasting — Favorita

This repo adds a spatio‑temporal Graph Neural Network (GNN) baseline for the Kaggle Favorita dataset, predicting multiple future days for all (store, family) nodes jointly.

Key ideas
- Nodes = (store_nbr, family) pairs (about 54 × 33 ≈ 1782 nodes).
- Adjacency = Kronecker sum of:
  - Store–store similarity graph (same cluster/state + transactions correlation).
  - Family–family similarity graph (national daily sales correlation).
- Model = Temporal dilated causal convolutions (TCN) + first‑order Graph Convolution (GCN) blocks, repeated with increasing dilations, decoding from the last hidden state to multi‑horizon quantiles.

Pipeline

1. Preprocess data
   Reuse your existing preprocessing that writes `data/processed/panel.csv`.
   If needed:
   ```
   python src/data/preprocess_favorita.py
   ```

2. Build graphs and adjacency
   ```
   python scripts/build_graphs_favorita_gnn.py
   ```
   Outputs:
   - `data/processed/store_graph_edges.csv`
   - `data/processed/family_graph_edges.csv`
   - `data/processed/node_index.csv`  (rows: node_id, store_nbr, family)
   - `data/processed/adjacency.npy`   (N×N normalized adjacency)

3. Train the ST‑GNN
   ```
   python src/gnn/train_stgnn.py --enc-len 56 --horizon 28 --epochs 20 --batch-size 32 --lr 1e-3
   ```

4. Evaluate
   The training script reports Pinball loss, WAPE, sMAPE on validation and test sets.

Notes
- This baseline uses only past covariates to forecast (no explicit known‑future injection). You can extend the decoder to condition on planned promotions/holidays.
- Quantiles: defaults to [0.1, 0.5, 0.9]; median (q≈0.5) is used for point metrics.
- Adjacency can be tuned via thresholds in the graph builder script.

References
- STGCN, Graph WaveNet (for the block design inspiration)
- Lim et al., TFT (for quantile and multi‑horizon evaluation alignment)