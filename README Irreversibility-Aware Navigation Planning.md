# Irreversibility-Aware Navigation Planning

This repository presents an **irreversibility-aware path planning framework** for autonomous robotic navigation in unknown and uncertain environments.

The core idea is to explicitly model **points of no return** and enforce them as **hard feasibility constraints** during planning, rather than treating uncertainty as a soft or probabilistic cost.

---

## 1. Motivation

In realistic navigation scenarios, some regions are not merely risky but *irreversible*.  
Once entered, recovery may be impossible due to:

- accumulated localization drift,
- lack of visual features,
- topological dead-ends,
- loss of reliable state estimation.

Classical planners such as **A\***, **RRT\***, and risk-weighted planners do not explicitly distinguish between:

- **recoverable risk**, and  
- **irreversible commitment**.

This work explicitly models irreversibility and studies its impact on:

- path feasibility,
- environment connectivity,
- search complexity.

---

## 2. Irreversibility Map

Each grid cell `s` is assigned an **irreversibility score**:

$$
I(s) \in [0,1]
$$

Higher values indicate a higher probability that entering the cell constitutes a *point of no return*.

---

## 2.1 Irreversibility Model

Irreversibility is computed as a weighted combination of normalized factors:

$$
I(s)=
w_u\,\hat{U}(s)
+ w_f\,(1-\hat{F}(s))
+ w_d\,\hat{D}(s)
$$

where:

- $\hat{U}(s)$: normalized localization uncertainty  
- $\hat{F}(s)$: normalized feature density  
- $\hat{D}(s)$: drift or instability indicator  
- $w_u, w_f, w_d$: weighting coefficients  

All terms are normalized to $[0,1]$.

Non-traversable cells are assigned:

$$
I(s)=1
$$

---

## 3. Irreversibility-Constrained Planning

A path $\pi$ is **admissible if and only if**:

$$
\max_{s \in \pi} I(s) \le \tau
$$

Cells with $I(s) > \tau$ are treated as **hard obstacles**.

This induces a **phase transition** in feasibility:

- feasible planning for $\tau \ge \tau^*$,
- infeasible planning for $\tau < \tau^*$.

---

## 4. Experiment 1: Threshold Feasibility Sweep

### Script

```bash
python run_irreversibility_tau_sweep.py
```

### Setup

- Fixed start and goal states  
- Threshold sweep:

$$
\tau \in [0.30, 1.00]
$$

### Observations

- A sharp feasibility threshold emerges.
- Below the critical threshold $\tau^*$, planning fails due to irreversibility violations.

---

## 5. Experiment 2: Bottleneck-Induced Disconnection

### Script

```bash
python run_irreversibility_bottleneck_sweep.py
```

### Result

For:

$$
\tau < I_{\text{door}}
$$

no path exists.

For:

$$
\tau \ge I_{\text{door}}
$$

feasibility is restored.

---

## 6. Failure Modes

1. Start or goal irreversibility violation  
2. Constraint-induced environment disconnection  

---

## 7. Key Findings

- Irreversibility introduces sharp feasibility transitions.
- Hard constraints fundamentally alter environment connectivity.
- Complexity increases near the critical threshold.

---

© 2026 **Panagiota Grosdouli**  
School of Electrical & Computer Engineering  
Democritus University of Thrace
