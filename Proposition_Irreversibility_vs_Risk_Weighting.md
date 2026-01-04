# Proposition: Irreversibility Constraints Are Not Equivalent to Risk Weighting

## Claim (Informal Proposition)

There exist navigation environments in which a **risk-weighted planner** can always return a feasible path,  
while a **hard irreversibility-constrained planner** becomes infeasible below a critical threshold **τ\***.

Moreover, even when the risk-weighted planner reduces *mean* risk exposure by increasing **λ**,  
it may still traverse **high-irreversibility cells** (points of no return), i.e. it does not control  
the maximum irreversibility along the path.

---

## Definitions

Each cell `s` is associated with an irreversibility score:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?I(s)%5Cin%5B0,1%5D" />
</p>

---

### Hard Constraint Feasibility

A path `π` is admissible if and only if:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmax_%7Bs%5Cin%5Cpi%7D%20I(s)%20%5Cle%20%5Ctau" />
</p>

If no path satisfies this condition, planning is **infeasible**.

---

### Soft Risk-Weighted Planning

A typical soft risk-weighted objective is:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?J(%5Cpi)=%5Csum%20c(s)%2B%5Clambda%5Csum%20I(s)" />
</p>

This formulation trades geometric cost against **accumulated irreversibility exposure**.

---

## Counterexample Construction: Bottleneck Environment

We construct a grid environment with:

- a high-irreversibility wall (e.g. `I = 0.95`),
- a narrow door region with intermediate irreversibility (e.g. `I = 0.60`),
- start and goal states located in low-irreversibility regions (`I ≈ 0`).

In this environment:

- any path from start to goal must pass through the door,
- therefore the minimal feasible hard threshold satisfies:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Ctau%5E*%20%5Capprox%20%5Cmin_%7B%5Cpi%7D%20%5Cmax_%7Bs%5Cin%5Cpi%7D%20I(s)%20%5Capprox%20I(%5Ctext%7Bdoor%7D)" />
</p>

---

## Empirical Evidence in This Repository

### Hard Constraint Phase Transition

The script:

- `run_irreversibility_bottleneck_sweep.py`

demonstrates a **sharp feasibility transition**:

- planning is infeasible for `τ < τ*`,
- planning becomes feasible for `τ ≥ τ*`.

---

### Soft Risk Weighting Does Not Control Maximum Irreversibility

The script:

- `run_risk_weighted_lambda_sweep.py`

shows that increasing **λ** reduces *mean* irreversibility exposure,  
but does **not** reduce the maximum irreversibility along the path:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmax%20I(%5Cpi)%20%5Capprox%200.95%20%5Cquad%20%5Cforall%20%5Clambda" />
</p>

---

### Visual Confirmation: Hard vs Soft Planning

The script:

- `plot_path_overlay_hard_vs_soft.py`

produces the figure:

- `path_overlay_hard_vs_soft.png`

which visually shows:

- the **hard planner** refusing routes below `τ*`,
- the **soft planner** selecting paths that may cross high-irreversibility cells.

---

## Takeaway

Hard irreversibility constraints enforce safety by altering **feasibility** and **environment connectivity**.  
Soft risk weighting merely reshapes the objective function and cannot guarantee avoidance of  
high-irreversibility transitions.

Therefore, **irreversibility constraints are not reducible to risk weighting**.
