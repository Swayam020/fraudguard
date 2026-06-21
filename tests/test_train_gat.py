"""Tests for HeteroGAT training (Milestone 4.4)."""

from pathlib import Path

import pytest
import torch

from fraudguard.models.hetero_gat import HeteroGAT
from fraudguard.models import train_gat

GRAPH_PT = Path("data/processed/graph/hetero_data.pt")

pytestmark = pytest.mark.skipif(not GRAPH_PT.exists(), reason="hetero_data.pt not built")


class TestFeatureStats:
    def test_default_is_identity(self):
        m = HeteroGAT()
        assert torch.allclose(m.feat_mean, torch.zeros(9))
        assert torch.allclose(m.feat_std, torch.ones(9))

    def test_set_clamps_zero_std(self):
        m = HeteroGAT()
        m.set_feature_stats(torch.zeros(9), torch.zeros(9))
        assert (m.feat_std >= 1e-6).all()

    def test_stats_buffers_persist_in_state_dict(self):
        m = HeteroGAT()
        m.set_feature_stats(torch.arange(9).float(), torch.ones(9) * 2)
        assert "feat_mean" in m.state_dict()
        assert "feat_std" in m.state_dict()


class TestClassWeights:
    def test_inverse_frequency(self):
        y = torch.tensor([0, 0, 0, 1])
        mask = torch.ones(4, dtype=torch.bool)
        w = train_gat.class_weights(y, mask)
        assert w[1] > w[0]
        assert torch.isclose(w[0], torch.tensor(4 / 6.0))
        assert torch.isclose(w[1], torch.tensor(4 / 2.0))

    def test_beta_shrinks_fraud_weight(self):
        y = torch.tensor([0, 0, 0, 1])
        mask = torch.ones(4, dtype=torch.bool)
        full = train_gat.class_weights(y, mask)
        soft = train_gat.class_weights(y, mask, beta=0.5)
        assert soft[1] < full[1]
        assert torch.isclose(soft[1], full[1] ** 0.5)


class TestTrainSmoke:
    def test_short_run_writes_checkpoint(self, tmp_path):
        ckpt = tmp_path / "smoke.pt"
        pr = train_gat.train(epochs=3, patience=99, ckpt=ckpt, verbose=False)
        assert ckpt.exists()
        blob = torch.load(ckpt, weights_only=False)
        assert "state_dict" in blob
        assert "val_pr_auc" in blob
        assert 0.0 <= pr <= 1.0
        assert not ckpt.samefile(train_gat.CKPT)
