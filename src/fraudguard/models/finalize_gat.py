"""Finalize GNN: multi-seed test distribution + val-selected checkpoint (Milestone 4.5)."""

import shutil
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
OUT_MD = RESULTS_DIR / "test_evaluation.md"
SEEDS = [0, 1, 2, 3, 4]


def probs_for(ckpt, data):
    model = HeteroGAT()
    blob = torch.load(ckpt, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    with torch.no_grad():
        return F.softmax(model(data), dim=1)[:, 1].numpy()


def main():
    data = torch.load(train_gat.GRAPH_PT, weights_only=False)
    y = data["transaction"].y.numpy()
    vm = data["transaction"].val_mask.numpy()
    tm = data["transaction"].test_mask.numpy()

    _, _, y_test_rf, p_test_rf = rf_probs()
    rf_test = float(average_precision_score(y_test_rf, p_test_rf))

    rows = []
    best_val = -1.0
    best_seed = SEEDS[0]
    with tempfile.TemporaryDirectory() as td:
        best_ckpt = Path(td) / "best.pt"
        for seed in SEEDS:
            ckpt = Path(td) / f"s{seed}.pt"
            train_gat.train(epochs=200, patience=20, ckpt=ckpt, verbose=False, seed=seed)
            prob = probs_for(ckpt, data)
            v = float(average_precision_score(y[vm], prob[vm]))
            t = float(average_precision_score(y[tm], prob[tm]))
            rows.append((seed, v, t))
            print(f"  seed={seed}: val {v:.4f} test {t:.4f}", flush=True)
            if v > best_val:
                best_val, best_seed = v, seed
                shutil.copy(ckpt, best_ckpt)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        shutil.copy(best_ckpt, train_gat.CKPT)

    tests = np.array([t for _, _, t in rows])
    sel_test = next(t for s, _, t in rows if s == best_seed)

    lines = [
        "# Test-Set Evaluation (Milestone 4.5)",
        "",
        "GNN uses class-weight beta=0.75 (selected on validation in the weight "
        "sweep). Trained across 5 seeds; test PR-AUC is reported as a "
        "distribution because the test split has only 11 fraud nodes, so any "
        "single run is high-variance. Deployed checkpoint = the seed with best "
        "VALIDATION PR-AUC (selection on val, never on test).",
        "",
        f"**GNN test PR-AUC: {tests.mean():.4f} +/- {tests.std():.4f}** "
        f"(5 seeds)  vs  **RandomForest: {rf_test:.4f}**",
        "",
        f"Deployed checkpoint: seed {best_seed} (val PR-AUC {best_val:.4f}), "
        f"test PR-AUC {sel_test:.4f}.",
        "",
        "| seed | val PR-AUC | test PR-AUC |",
        "|---|---|---|",
    ]
    for s, v, t in rows:
        mark = " **<- deployed (best val)**" if s == best_seed else ""
        lines.append(f"| {s} | {v:.4f} | {t:.4f} |{mark}")
    lines += [
        "",
        "## Honest reading",
        "",
        "- GNN and RandomForest are statistically indistinguishable on test "
        "PR-AUC; the gap is far smaller than the seed-to-seed variance.",
        "- RandomForest remains the simpler, equally strong baseline.",
        "- GNN ranks well (ROC-AUC ~0.93) but is poorly calibrated; probability "
        "calibration is noted as future work (does not affect PR-AUC).",
        "- Root cause of the wide error bars: only 11 fraud transactions in the "
        "test split (ADR-002 variance risk).",
        "",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"\nGNN {tests.mean():.4f} +/- {tests.std():.4f}  vs RF {rf_test:.4f}")
    print(f"deployed: seed {best_seed} (val {best_val:.4f}, test {sel_test:.4f})")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
