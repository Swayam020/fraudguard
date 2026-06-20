"""Heterogeneous GAT for fraud detection on the transaction graph (Milestone 4.3)."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv

EDGE_TYPES = [
    ("user", "sends", "transaction"),
    ("transaction", "rev_sends", "user"),
    ("transaction", "receives", "merchant"),
    ("merchant", "rev_receives", "transaction"),
    ("transaction", "paid_to", "user"),
    ("user", "rev_paid_to", "transaction"),
]


class HeteroGAT(nn.Module):
    def __init__(
        self,
        in_txn=9,
        hidden=64,
        heads=4,
        num_layers=2,
        dropout=0.3,
        num_classes=2,
    ):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.dropout = dropout

        self.txn_lin = nn.Linear(in_txn, hidden)
        self.user_emb = nn.Parameter(torch.empty(hidden))
        self.merchant_emb = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.user_emb, std=0.1)
        nn.init.normal_(self.merchant_emb, std=0.1)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        et: GATConv(
                            hidden,
                            hidden // heads,
                            heads=heads,
                            dropout=dropout,
                            add_self_loops=False,
                        )
                        for et in EDGE_TYPES
                    },
                    aggr="sum",
                )
            )
        self.head = nn.Linear(hidden, num_classes)

    def encode(self, data):
        return {
            "transaction": self.txn_lin(data["transaction"].x),
            "user": self.user_emb.unsqueeze(0).expand(data["user"].num_nodes, -1),
            "merchant": self.merchant_emb.unsqueeze(0).expand(data["merchant"].num_nodes, -1),
        }

    def forward(self, data):
        x_dict = self.encode(data)
        edge_index_dict = data.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()
            }
        return self.head(x_dict["transaction"])


if __name__ == "__main__":
    from pathlib import Path

    data = torch.load(Path("data/processed/graph/hetero_data.pt"), weights_only=False)
    model = HeteroGAT()
    model.eval()
    with torch.no_grad():
        out = model(data)
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params)
    print("logits:", tuple(out.shape))
    print("y:", tuple(data["transaction"].y.shape))
