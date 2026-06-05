# FraudGuard — Architecture

> Living document. Updated as the system evolves.

## High-Level Pipeline

Transaction -> Graph Construction -> GNN Embedding -> Vector Search -> Decision Engine -> Classification

## Modules

| Module | Path | Responsibility | Phase Added |
|---|---|---|---|
| Data | src/fraudguard/data/ | Loading, preprocessing, feature engineering of PaySim transactions | Phase 2 |
| Models | src/fraudguard/models/ | Baseline ML models (LR, RF, XGBoost) and FraudGAT | Phase 3, 6 |
| Graph | src/fraudguard/graph/ | Heterogeneous transaction graph construction | Phase 6 |
| Vector | src/fraudguard/vector/ | MongoDB Atlas vector search integration | Phase 7 |
| Decision | src/fraudguard/decision/ | Fusion of GNN + vector + rule signals | Phase 8 |
| API | src/fraudguard/api/ | FastAPI inference endpoints, auth, rate limiting | Phase 8 |
| Utils | src/fraudguard/utils/ | Shared helpers (logging, config, validation) | Phase 1+ |
| Dashboard | dashboard/ | Streamlit monitoring UI | Phase 9 |

## Data Flow

To be diagrammed in Phase 9 with a proper architecture diagram.

## Decision Engine Formula

Final Score = 0.50 * GNN Score + 0.35 * Vector Ratio + 0.15 * Rule Score

See ADR (TBD in Phase 8) for weight justification.

## Deployment

Docker containers for FastAPI + Streamlit, deployed to free-tier cloud. See ADR (TBD in Phase 10).
