# FraudGuard — Real-Time Fraud Detection API

> Graph Neural Network + Vector Similarity Search for sub-100ms fraud detection on financial transactions.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen.svg)](https://www.mongodb.com/atlas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem

Financial fraud is rising in volume and sophistication. Traditional rule-based and tabular ML systems (Logistic Regression, Random Forest, XGBoost) struggle with:

- **Severe class imbalance** (~0.13% of transactions are fraudulent)
- **Coordinated multi-entity fraud** that tabular features can't capture
- **Adapting to new fraud patterns** without retraining

FraudGuard tackles all three using a hybrid Graph Neural Network + Vector Search architecture.

## 🏗️ Architecture

> _Architecture diagram coming in Phase 9_

**Pipeline:** Transaction → Graph Construction → GNN Embedding → Vector Search → Decision Engine → Classification

**Three signals fused for final decision:**
- **50%** — Graph Attention Network fraud probability
- **35%** — k-nearest-neighbour vector similarity ratio
- **15%** — Rule-based heuristic signals

## 🚀 Demo

> _Live demo and screenshots coming in Phase 9_

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **ML Framework** | PyTorch + PyTorch Geometric |
| **Model** | Heterogeneous Graph Attention Network (GAT) |
| **API** | FastAPI (async) |
| **Vector DB** | MongoDB Atlas Vector Search |
| **Dashboard** | Streamlit |
| **Deployment** | Docker + free-tier cloud |
| **CI/CD** | GitHub Actions |

## 📊 Dataset

- **PaySim** — synthetic mobile money transactions (Kaggle)
- **Size used:** 200,000 transactions (subsampled for hardware constraints)
- **Fraud rate:** ~0.13%

## 📈 Results

> _Benchmark numbers coming in Phase 6_

## ⚙️ Setup

### Prerequisites
- Python 3.12+
- Git
- A GitHub account (for cloning)

### Local development setup

Clone the repo:

​```bash
git clone git@github.com:Swayam020/fraudguard.git
cd fraudguard
​```

Create and activate a virtual environment:

​```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# OR
venv\Scripts\activate      # Windows
​```

Install dependencies:

​```bash
pip install --upgrade pip
pip install -r requirements.txt
​```

Install pre-commit hooks (one-time, per clone):

​```bash
pre-commit install
​```

Verify the setup:

​```bash
pre-commit run --all-files
​```

All hooks should report `Passed` or `Skipped`. If anything `Failed`, see the error message and re-run.

### Project structure

​```
fraudguard/
├── src/fraudguard/    # Application source code
├── tests/             # Pytest test suite
├── notebooks/         # Jupyter notebooks for EDA
├── data/              # PaySim raw + processed (gitignored)
├── dashboard/         # Streamlit monitoring UI
├── docs/              # Architecture, ADRs, glossary
└── scripts/           # Standalone runnable scripts
​```
