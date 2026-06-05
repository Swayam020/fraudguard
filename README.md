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

> _Setup instructions coming as we build_
