"""Train HeteroGAT on the transaction graph (Milestone 4.4)."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudguard.models.hetero_gat import HeteroGAT

GRAPH_PT = Path("data/processed/graph/hetero_data.pt")
CKPT = Path("data/processed/graph/hetero_gat_best.pt")
RF_BASELINE_PR_AUC = 0.4162
SEED = 42


def set_seed(seed=SEED):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(y, mask, beta=1.0):
    yt = y[mask]
    counts = torch.bincount(yt, minlength=2).float()
    return (yt.numel() / (2.0 * counts)) ** beta


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    prob = F.softmax(model(data), dim=1)[:, 1]
    y_true = data["transaction"].y[mask].cpu().numpy()
    y_score = prob[mask].cpu().numpy()
    return (
        average_precision_score(y_true, y_score),
        roc_auc_score(y_true, y_score),
    )


def train(
    epochs=200,
    lr=1e-3,
    weight_decay=5e-4,
    patience=20,
    ckpt=CKPT,
    verbose=True,
    seed=SEED,
    weight_beta=0.75,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(GRAPH_PT, weights_only=False).to(device)
    model = HeteroGAT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y = data["transaction"].y
    train_mask = data["transaction"].train_mask
    val_mask = data["transaction"].val_mask
    w = class_weights(y, train_mask, weight_beta).to(device)

    tx = data["transaction"].x[train_mask]
    model.set_feature_stats(tx.mean(0), tx.std(0))

    best_pr, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=w)
        loss.backward()
        opt.step()

        val_pr, val_roc = evaluate(model, data, val_mask)
        if val_pr > best_pr:
            best_pr = val_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"epoch {epoch:3d}  loss {loss.item():.4f}  "
                f"val_pr {val_pr:.4f}  val_roc {val_roc:.4f}  best {best_pr:.4f}"
            )
        if bad >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (patience {patience})")
            break

    model.load_state_dict(best_state)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "val_pr_auc": best_pr}, ckpt)
    if verbose:
        delta = best_pr - RF_BASELINE_PR_AUC
        print(
            f"BEST val PR-AUC {best_pr:.4f}  vs RF {RF_BASELINE_PR_AUC:.4f}  "
            f"({'+' if delta >= 0 else ''}{delta:.4f})"
        )
    return best_pr


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=20)
    a = p.parse_args()
    train(a.epochs, a.lr, a.weight_decay, a.patience)
