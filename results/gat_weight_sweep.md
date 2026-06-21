# GNN Class-Weight Sweep (Milestone 4.5 fix)

5 seeds per setting. beta scales class weights ((N/2Nc)^beta); beta=1.0 is the original aggressive weighting. Selection by mean validation PR-AUC; test shown for reference only.

RF test PR-AUC reference: **0.3570**

| beta | val PR-AUC (mean +/- std) | test PR-AUC (mean +/- std) |
|---|---|---|
| 0.5 | 0.4776 +/- 0.0192 | 0.3222 +/- 0.0649 |
| 0.75 | 0.4983 +/- 0.0114 | 0.3645 +/- 0.0087 | **<- best val**
| 1.0 | 0.4740 +/- 0.0125 | 0.3126 +/- 0.0390 |

**Best beta by val: 0.75** -- test PR-AUC 0.3645 +/- 0.0087 vs RF 0.3570.
