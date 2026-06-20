"""Tests for HeteroData graph construction (Milestone 4.2)."""

from pathlib import Path

import pandas as pd
import pytest
import torch

GRAPH_DIR = Path("data/processed/graph")
GRAPH_PT = GRAPH_DIR / "hetero_data.pt"

pytestmark = pytest.mark.skipif(not GRAPH_PT.exists(), reason="hetero_data.pt not built")


@pytest.fixture(scope="module")
def data():
    return torch.load(GRAPH_PT, weights_only=False)


@pytest.fixture(scope="module")
def acc():
    return pd.read_parquet(GRAPH_DIR / "account_id_map.parquet")


@pytest.fixture(scope="module")
def txn():
    return pd.read_parquet(GRAPH_DIR / "txn_id_map.parquet")


class TestNodes:
    def test_node_counts(self, data, acc, txn):
        assert data["user"].num_nodes == (acc.node_type == "user").sum()
        assert data["merchant"].num_nodes == (acc.node_type == "merchant").sum()
        assert data["transaction"].x.size(0) == len(txn)

    def test_feature_shape(self, data):
        assert data["transaction"].x.shape == (200000, 9)
        assert data["transaction"].x.dtype == torch.float

    def test_labels(self, data):
        y = data["transaction"].y
        assert y.shape == (200000,)
        assert y.dtype == torch.long
        assert set(y.unique().tolist()) <= {0, 1}
        assert int(y.sum()) == 147


class TestMasks:
    def test_counts(self, data):
        tm = data["transaction"]
        assert int(tm.train_mask.sum()) == 140000
        assert int(tm.val_mask.sum()) == 30000
        assert int(tm.test_mask.sum()) == 30000

    def test_disjoint_and_complete(self, data):
        tm = data["transaction"]
        stacked = torch.stack([tm.train_mask, tm.val_mask, tm.test_mask])
        assert (stacked.sum(0) == 1).all()


class TestEdges:
    def test_edge_counts(self, data):
        assert data["user", "sends", "transaction"].edge_index.size(1) == 200000
        r = data["transaction", "receives", "merchant"].edge_index.size(1)
        p = data["transaction", "paid_to", "user"].edge_index.size(1)
        assert r + p == 200000

    def test_in_bounds(self, data):
        nn = {
            "user": data["user"].num_nodes,
            "merchant": data["merchant"].num_nodes,
            "transaction": data["transaction"].x.size(0),
        }
        for (s, _, d), store in data.edge_items():
            ei = store.edge_index
            assert ei.min() >= 0
            assert ei[0].max() < nn[s]
            assert ei[1].max() < nn[d]

    def test_reverses_are_flips(self, data):
        pairs = [
            (("user", "sends", "transaction"), ("transaction", "rev_sends", "user")),
            (
                ("transaction", "receives", "merchant"),
                ("merchant", "rev_receives", "transaction"),
            ),
            (("transaction", "paid_to", "user"), ("user", "rev_paid_to", "transaction")),
        ]
        for fwd, rev in pairs:
            assert torch.equal(data[fwd].edge_index, data[rev].edge_index.flip(0))


class TestValidate:
    def test_pyg_validate(self, data):
        assert data.validate()
