"""Tests for the HeteroGAT model (Milestone 4.3)."""

from pathlib import Path

import pytest
import torch

from fraudguard.models.hetero_gat import EDGE_TYPES, HeteroGAT

GRAPH_PT = Path("data/processed/graph/hetero_data.pt")

pytestmark = pytest.mark.skipif(not GRAPH_PT.exists(), reason="hetero_data.pt not built")


@pytest.fixture(scope="module")
def data():
    return torch.load(GRAPH_PT, weights_only=False)


@pytest.fixture(scope="module")
def model():
    return HeteroGAT()


class TestForward:
    def test_output_shape(self, data, model):
        model.eval()
        with torch.no_grad():
            out = model(data)
        assert out.shape == (data["transaction"].x.size(0), 2)

    def test_finite(self, data, model):
        model.eval()
        with torch.no_grad():
            out = model(data)
        assert torch.isfinite(out).all()

    def test_dropout_train_vs_eval(self, data, model):
        torch.manual_seed(0)
        model.train()
        a = model(data)
        model.eval()
        with torch.no_grad():
            b1 = model(data)
            b2 = model(data)
        assert torch.allclose(b1, b2)
        assert not torch.allclose(a, b1)


class TestConfig:
    def test_edge_types_match_graph(self, data):
        assert set(EDGE_TYPES) == set(data.edge_index_dict.keys())

    def test_hidden_divisible_by_heads(self):
        with pytest.raises(AssertionError):
            HeteroGAT(hidden=64, heads=5)

    def test_backward_runs(self, data, model):
        model.train()
        out = model(data)
        loss = out[data["transaction"].train_mask].sum()
        loss.backward()
        # Core params that feed transaction logits must receive gradients.
        # Final-layer convs producing user/merchant outputs are unused
        # (only transaction logits are read) so they legitimately get none.
        core = {
            "txn_lin.weight",
            "txn_lin.bias",
            "user_emb",
            "merchant_emb",
            "head.weight",
            "head.bias",
        }
        grads = {n: p.grad for n, p in model.named_parameters()}
        for n in core:
            assert grads[n] is not None, n
            assert torch.isfinite(grads[n]).all(), n
        assert any(g is not None for n, g in grads.items() if "convs" in n)
