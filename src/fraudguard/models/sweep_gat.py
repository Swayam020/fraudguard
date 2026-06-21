"""Sweep class-weight strength x seeds to stabilize the GNN (Milestone 4.5 fix)."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from fraudguard.models import train_gat
from fraudguard.models.eval_gat import rf_probs
from fraudguard.models.hetero_gat import HeteroGAT

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
OUT_MD = RESULTS_DIR / "gat_weight_sweep.md"
BETAS = [0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def ckpt_prauc(ckpt, data):
    model = HeteroGAT()
    blob = torch.load(ckpt, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(data), dim=1)[:, 1].numpy()
    y = data["transaction"].y.numpy()
    vm = data["transaction"].val_mask.numpy()
    tm = data["transaction"].test_mask.numpy()
    return (
        float(average_precision_score(y[vm], prob[vm])),
        float(average_precision_score(y[tm], prob[tm])),
    )


def main():
    data_cpu = torch.load(train_gat.GRAPH_PT, weights_only=False)
    _, _, y_test_rf, p_test_rf = rf_probs()
    rf_test = float(average_precision_score(y_test_rf, p_test_rf))
    print(f"RF test PR-AUC reference: {rf_test:.4f}\n", flush=True)

    results = {}
    with tempfile.TemporaryDirectory() as td:
        for beta in BETAS:
            vals, tests = [], []
            for seed in SEEDS:
                ckpt = Path(td) / f"b{beta}_s{seed}.pt"
                train_gat.train(
                    epochs=200,
                    patience=20,
                    ckpt=ckpt,
                    verbose=False,
                    seed=seed,
                    weight_beta=beta,
                )
                vpr, tpr = ckpt_prauc(ckpt, data_cpu)
                vals.append(vpr)
                tests.append(tpr)
                print(
                    f"  beta={beta} seed={seed}: val {vpr:.4f} test {tpr:.4f}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results[beta] = (np.array(vals), np.array(tests))

    best_beta = max(results, key=lambda b: results[b][0].mean())
    lines = [
        "# GNN Class-Weight Sweep (Milestone 4.5 fix)",
        "",
        f"{len(SEEDS)} seeds per setting. beta scales class weights "
        "((N/2Nc)^beta); beta=1.0 is the original aggressive weighting. "
        "Selection by mean validation PR-AUC; test shown for reference only.",
        "",
        f"RF test PR-AUC reference: **{rf_test:.4f}**",
        "",
        "| beta | val PR-AUC (mean +/- std) | test PR-AUC (mean +/- std) |",
        "|---|---|---|",
    ]
    for beta in BETAS:
        v, t = results[beta]
        mark = " **<- best val**" if beta == best_beta else ""
        lines.append(
            f"| {beta} | {v.mean():.4f} +/- {v.std():.4f} | "
            f"{t.mean():.4f} +/- {t.std():.4f} |{mark}"
        )
    bv, bt = results[best_beta]
    lines += [
        "",
        f"**Best beta by val: {best_beta}** -- test PR-AUC "
        f"{bt.mean():.4f} +/- {bt.std():.4f} vs RF {rf_test:.4f}.",
        "",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"\nBest beta by val mean: {best_beta}")
    print(f"  test PR-AUC {bt.mean():.4f} +/- {bt.std():.4f}  (RF {rf_test:.4f})")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
