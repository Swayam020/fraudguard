# ADR-001: Initial Technology Stack

- **Status:** Accepted
- **Date:** 2026-06-05
- **Deciders:** Swayam Maheshwari

## Context

FraudGuard is a real-time fraud detection system requiring three capabilities working together: graph-based machine learning, vector similarity search, and a low-latency API. The original research paper used a specific stack (Python + PyTorch + PyTorch Geometric + MongoDB Atlas + Streamlit). For the production-grade rebuild, the stack must additionally support a real REST API, containerized deployment, and CI/CD — i.e., it must look like an engineering project, not a research notebook.

Stack choices made on day one constrain every later decision: model framework affects training and deployment, database choice affects vector search latency, API framework affects async patterns, and deployment platform affects all of the above.

## Decision

The initial stack is:

**Language: Python 3.12**
The dominant language in the ML ecosystem. Every library FraudGuard needs (PyTorch, pandas, FastAPI, MongoDB drivers) is Python-native. No reasonable alternative for an ML-heavy project.

**Deep Learning Framework: PyTorch + PyTorch Geometric (PyG)**
PyTorch is the de-facto research framework and has won the industry vs. TensorFlow over the last few years. PyG is the most mature Python library for Graph Neural Networks, with first-class support for heterogeneous graphs and the GATConv layer the project needs. Alternatives like DGL exist but are smaller communities.

**API Framework: FastAPI**
Chosen over Flask and Django REST Framework. FastAPI is async-native (critical for low-latency inference under concurrent load), has automatic OpenAPI / Swagger doc generation, and uses Pydantic for request/response validation — which catches malformed input cleanly. Flask is older and synchronous by default; Django REST is overkill for a single-purpose inference service.

**Vector Database: MongoDB Atlas (with Atlas Vector Search)**
The original paper's choice. Chosen over Pinecone, Qdrant, and Weaviate because: (a) it's a dual-purpose store — document storage AND vector search in one DB, which avoids running a separate vector service; (b) Atlas has a free tier sufficient for this project's scale (~200K embeddings of 128 dimensions); (c) cosine-similarity ANN search is built in. Tradeoff: Atlas vector search is younger than Pinecone and benchmarks slightly behind on raw QPS at extreme scale — fine for this project, would re-evaluate at production scale.

**Dashboard: Streamlit**
Chosen over a React frontend for time-budget reasons. Streamlit lets a single Python developer produce a credible monitoring UI in hours, not weeks. The dashboard is for demo/screenshot value, not as a production frontend. A React rewrite would be a future-work item.

**Containerization: Docker + docker-compose**
Industry standard. Non-negotiable for any project claiming to be "deployment-ready" on a resume. Compose chosen over plain Docker for orchestrating the FastAPI service and Streamlit dashboard together locally.

**Deployment Target: Free-tier cloud (specific platform TBD)**
Deferred to Phase 10. Candidates are Hugging Face Spaces, Render, Fly.io. Decision will depend on which free tier still works at the time of deployment.

**CI/CD: GitHub Actions**
Chosen over CircleCI / Travis because it's free for public repos, lives in the same place as the code, and is the dominant choice in 2026. No real alternative for a portfolio project.

**Testing: pytest**
Chosen over unittest. Pytest is the modern standard, requires less boilerplate, has better fixtures and parametrization.

**Linting / Formatting: ruff + black**
Chosen over pylint + autopep8. Ruff is dramatically faster (written in Rust), black is the de-facto Python formatter. Both will run as pre-commit hooks.

## Alternatives Considered

- **TensorFlow + TF-GNN** instead of PyTorch + PyG: rejected because PyTorch dominates research and PyG has better heterogeneous graph support.
- **Flask** instead of FastAPI: rejected because of synchronous defaults and lack of built-in validation.
- **Pinecone** instead of MongoDB Atlas Vector Search: rejected because it would require running a separate service alongside a document DB. Atlas consolidates both.
- **React + FastAPI** dashboard: rejected because of the time budget. Streamlit gets us 80% of the visual value at 10% of the effort.
- **Kubernetes** for deployment: rejected as massive overkill for a single-instance demo. Docker + free-tier cloud is sufficient.

## Consequences

**Positive:**
- Stack is small, free, and locally runnable on the developer's GTX 1650 + 8GB RAM.
- Every tool chosen is on a major resume keyword list (FastAPI, PyTorch, Docker, MongoDB) — explicit choice given this is a portfolio project.
- Reproducible: every dependency is pinned in requirements.txt, environment is containerized.

**Negative / risks:**
- MongoDB Atlas free tier has connection and storage limits. May need to upgrade or migrate at higher scale.
- Streamlit is not a production frontend; recruiters comparing to React-based dashboards will notice.
- Free-tier deployment platforms can deprecate / change pricing. We'll commit to a specific one only when we reach Phase 10.

## Notes

Individual choices in this ADR may be revisited later via their own ADRs (e.g., if MongoDB Atlas is replaced, ADR-XXX will supersede the relevant part of this ADR). This ADR will not be edited beyond status changes.
