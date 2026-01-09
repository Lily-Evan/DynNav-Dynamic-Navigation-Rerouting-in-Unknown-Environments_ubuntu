# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

## Summary
A research-oriented navigation framework for autonomous robots operating in unknown and evolving environments under uncertainty.  
The framework integrates **risk-aware replanning**, **uncertainty-aware exploration**, and **learning-augmented planning**, supported by extensive quantitative evaluation and reproducible experiments.

---

## Overview
This repository presents a unified research framework for **autonomous robotic navigation under uncertainty**.

It targets key real-world challenges:
- Partial observability and incrementally built maps  
- Visual odometry drift and feature sparsity  
- Dynamic obstacles requiring frequent replanning  
- Trade-offs between optimality, safety, coverage, energy/connectivity, and computational cost  

The pipeline combines:
- Classical planning algorithms  
- Probabilistic state estimation  
- Learning-based heuristics  
- Explicit modeling of uncertainty, risk, and irreversibility  

Developed as an **individual research project** at the  
School of Electrical and Computer Engineering,  
Democritus University of Thrace (D.U.Th.).

---

## Abstract
Autonomous navigation in unknown environments is fundamentally constrained by uncertainty arising from sensing, estimation, and environment dynamics.  
This work introduces a navigation pipeline that explicitly reasons about uncertainty and risk, supports dynamic replanning, and integrates learned heuristics into classical planners while preserving formal guarantees.

The framework is validated through quantitative evaluation, parameter sweeps, and ablation studies, with emphasis on experimental reproducibility.

---

## Repository Structure
```
.
├── contributions/
│   ├── 01_learned_astar/
│   ├── 02_uncertainty_calibration/
│   ├── 03_belief_risk_planning/
│   ├── 04_irreversibility_returnability/
│   ├── 05_safe_mode_navigation/
│   ├── 06_energy_connectivity/
│   ├── 07_nbv_exploration/
│   ├── 08_security_ids/
│   ├── 09_multi_robot/
│   ├── 10_human_language_ethics/
│   └── _unsorted/
├── docs/
├── requirements.txt
└── CITATION.cff
```

---

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Key Research Contributions
1. Uncertainty-aware dynamic navigation with online replanning  
2. Learned admissible A* heuristics with preserved optimality  
3. Belief–risk planning with adaptive self-trust  
4. Irreversibility- and returnability-aware navigation  
5. Security-aware estimation and planning hooks  
6. Human-, language-, and ethics-aware extensions  

---

## Reproducibility
- Multi-seed experiments  
- Ablation studies  
- CSV logs and plots  

---

## Citation
See `CITATION.cff`

---

## License
Apache License 2.0

---

## Author
**Panagiota Grosdouli**  
Electrical & Computer Engineering, D.U.Th.
