## 🔒 Irreversibility-Aware Navigation (New Contribution)

This repository includes a set of experiments studying **irreversibility constraints** in navigation under uncertainty, and their relation to risk-weighted planning.

### Key Components
- **Irreversibility map construction** from uncertainty and local geometry
- **Hard irreversibility-constrained planning** with feasibility thresholds
- **Safe-mode planning** with automatic threshold relaxation
- **Adaptive threshold selection** via minimax (global feasibility-aware) estimation
- **Failure taxonomy** distinguishing local vs global infeasibility modes

### Core Experiments

| Experiment | Script |
|---|---|
| Bottleneck feasibility sweep | `run_irreversibility_bottleneck_sweep.py` |
| Hard vs soft (risk-weighted) comparison | `plot_hard_vs_soft_comparison.py` |
| Safe-mode activation analysis | `run_irreversibility_safe_mode_sweep.py` |
| Minimax adaptive τ (multi-start) | `run_minimax_tau_multistart.py` |
| Failure taxonomy analysis | `plot_failure_taxonomy.py` |

### Formal Result

A formal proposition and counterexample showing that **irreversibility constraints are not reducible to risk weighting** is provided here:

📄 **[`PROPOSITION_Irreversibility_vs_Risk.md`](PROPOSITION_Irreversibility_vs_Risk.md)**

This document includes:
- a precise claim,
- a bottleneck counterexample,
- references to scripts and figures in this repository.
