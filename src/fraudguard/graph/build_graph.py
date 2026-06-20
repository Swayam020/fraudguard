"""Build heterogeneous PyG graph from indexed accounts/transactions (Milestone 4.2)."""

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

FEATURE_COLS = [
    "amount_log",
    "amount_z",
    "step_hour",
    "step_day",
    "is_transfer",
    "is_cashout",
    "orig_balance_delta",
    "dest_balance_delta",
    "balance_mismatch",
]

PROC_DIR = Path("data/processed")
GRAPH_DIR = PROC_DIR / "graph"
SPLITS = ["train", "val", "test"]


def build_graph() -> HeteroData:
    acc = pd.read_parquet(GRAPH_DIR / "account_id_map.parquet")
    txn = pd.read_parquet(GRAPH_DIR / "txn_id_map.parquet")

    users = acc[acc.node_type == "user"].reset_index(drop=True)
    merchants = acc[acc.node_type == "merchant"].reset_index(drop=True)
    user_local = {n: i for i, n in enumerate(users.account_name)}
    merch_local = {n: i for i, n in enumerate(merchants.account_name)}
    n_user, n_merch = len(user_local), len(merch_local)
    n_txn = len(txn)

    frames = {}
    for split in SPLITS:
        df = pd.read_parquet(PROC_DIR / f"{split}.parquet").reset_index(drop=True)
        ids = txn[txn.split == split].sort_values("row_in_split")
        assert len(ids) == len(df), f"{split} row count mismatch"
        df = df.assign(txn_id=ids.txn_id.values)
        frames[split] = df

    full = pd.concat([frames[s] for s in SPLITS]).sort_values("txn_id").reset_index(drop=True)
    assert len(full) == n_txn
    assert (full.txn_id.values == range(n_txn)).all(), "txn_id not contiguous 0..n-1"

    orig_local = full.nameOrig.map(user_local)
    assert orig_local.notna().all(), "nameOrig with no user node"
    assert full.nameOrig.str.startswith("M").sum() == 0, "unexpected M-prefix sender"

    is_m = full.nameDest.str.startswith("M")
    md, cd = full[is_m], full[~is_m]
    m_local = md.nameDest.map(merch_local)
    u_local = cd.nameDest.map(user_local)
    assert m_local.notna().all(), "M-dest with no merchant node"
    assert u_local.notna().all(), "C-dest with no user node"

    data = HeteroData()
    data["transaction"].x = torch.tensor(full[FEATURE_COLS].values, dtype=torch.float)
    data["transaction"].y = torch.tensor(full.isFraud.values, dtype=torch.long)
    data["user"].num_nodes = n_user
    data["merchant"].num_nodes = n_merch

    for split in SPLITS:
        mask = torch.zeros(n_txn, dtype=torch.bool)
        mask[frames[split].txn_id.values] = True
        data["transaction"][f"{split}_mask"] = mask

    t = torch.tensor(full.txn_id.values, dtype=torch.long)
    sends = torch.stack([torch.tensor(orig_local.values, dtype=torch.long), t])
    data["user", "sends", "transaction"].edge_index = sends
    data["transaction", "rev_sends", "user"].edge_index = sends.flip(0)

    recv = torch.stack(
        [
            torch.tensor(md.txn_id.values, dtype=torch.long),
            torch.tensor(m_local.values, dtype=torch.long),
        ]
    )
    data["transaction", "receives", "merchant"].edge_index = recv
    data["merchant", "rev_receives", "transaction"].edge_index = recv.flip(0)

    paid = torch.stack(
        [
            torch.tensor(cd.txn_id.values, dtype=torch.long),
            torch.tensor(u_local.values, dtype=torch.long),
        ]
    )
    data["transaction", "paid_to", "user"].edge_index = paid
    data["user", "rev_paid_to", "transaction"].edge_index = paid.flip(0)

    data.validate()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, GRAPH_DIR / "hetero_data.pt")
    return data


if __name__ == "__main__":
    g = build_graph()
    print(g)
    print("fraud txns:", int(g["transaction"].y.sum()))
