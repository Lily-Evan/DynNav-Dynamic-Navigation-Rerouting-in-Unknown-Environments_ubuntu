# Irreversibility-Aware Navigation Planning

This repository introduces **irreversibility-aware path planning** for autonomous robotic navigation in unknown and uncertain environments.

The core idea is to explicitly model **points of no return** and enforce them as **hard feasibility constraints** during planning, rather than treating uncertainty as a soft cost.

---

## 1. Motivation

In realistic navigation scenarios, certain regions are not merely risky but *irreversible*.  
Once entered, recovery may be impossible due to:

- Accumulated localization drift  
- Lack of visual features  
- Topological dead-ends  
- Loss of reliable state estimation  

Classical planners such as **A\***, **RRT\***, and risk-weighted planners do not distinguish between:

- *Recoverable risk*  
- *Irreversible commitment*  

This module explicitly models **irreversibility** and studies its impact on:

- Path feasibility  
- Environment connectivity  
- Search complexity  

---

## 2. Irreversibility Map

Each grid cell \( s \) is assigned an **irreversibility score**:

\[
I(s) \in [0, 1]
\]

Higher values indicate a higher probability that entering the cell constitutes a *point of no return*.

---

### 2.1 Irreversibility Model

Irreversibility is computed as a weighted combination of normalized factors:

\[
I(s) =
w_u \cdot \hat{U}(s)
+ w_f \cdot \bigl(1 - \hat{F}(s)\bigr)
+ w_d \cdot \hat{D}(s)
\]

where:

- \( \hat{U}(s) \): normalized localization uncertainty  
- \( \hat{F}(s) \): normalized feature density  
- \( \hat{D}(s) \): drift or instability indicator  
- \( w_u, w_f, w_d \): weighting coefficients  

All terms are normalized to \([0,1]\).  
Non-traversable cells are assigned:

\[
I(s) = 1
\]

---

## 3. Irreversibility-Constrained Planning

A path \( \pi \) is **admissible** if and only if:

\[
\max_{s \in \pi} I(s) \le \tau
\]

Cells with \( I(s) > \tau \) are treated as **hard obstacles**.

This induces a **phase transition** in feasibility:

- Feasible planning for \( \tau \ge \tau^* \)  
- Infeasible planning for \( \tau < \tau^* \)  

---

## 4. Experiment 1: Threshold Feasibility Sweep

### Script
```bash
python run_irreversibility_tau_sweep.py
```

### Setup

- Fixed start and goal states  
- Threshold sweep:

\[
\tau \in [0.30, 1.00]
\]

### Observations

- A sharp feasibility threshold emerges  
- Below the critical \( \tau^* \), planning fails due to irreversibility violations  

### Generated Plots

- Success vs \( \tau \)  
- Node expansions vs \( \tau \)  
- Path cost vs \( \tau \)  
- Maximum irreversibility along the path vs \( \tau \)  

---

## 5. Experiment 2: Bottleneck-Induced Disconnection

### Script
```bash
python run_irreversibility_bottleneck_sweep.py
```

### Setup

- Artificial high-irreversibility wall  
- Narrow low-irreversibility door  
- Fixed start and goal with low irreversibility  

### Result

- For:

\[
\tau < I_{door}
\]

no path exists.

- For:

\[
\tau \ge I_{door}
\]

feasibility is restored.

This experiment demonstrates that irreversibility constraints can **disconnect the environment**.

---

## 6. Failure Modes

Two distinct failure modes are observed:

### 1. Start or Goal Violation
The start or goal state violates the irreversibility threshold.

### 2. Constraint-Induced Disconnection
No path exists under the imposed irreversibility constraint.

---

## 7. Key Findings

- Irreversibility introduces **sharp feasibility transitions**  
- Planning complexity increases near the critical threshold  
- Hard irreversibility constraints fundamentally alter environment connectivity  
- Irreversibility-aware planning captures failure modes that risk-weighted planners cannot  

---

## 8. Files

- `irreversibility_map.py`  
- `irreversibility_planner.py`  
- `run_irreversibility_tau_sweep.py`  
- `run_irreversibility_bottleneck_sweep.py`  
- `plot_irreversibility_tau_sweep.py`  
- `plot_irreversibility_bottleneck_sweep.py`  

---

## 9. Intended Use

This module is intended for:

- Research and educational purposes  
- Autonomous navigation under uncertainty  
- Studies of feasibility, safety, and irreversibility  

---

© 2026 **Panagiota Grosdouli**  
School of Electrical & Computer Engineering  
Democritus University of Thrace
