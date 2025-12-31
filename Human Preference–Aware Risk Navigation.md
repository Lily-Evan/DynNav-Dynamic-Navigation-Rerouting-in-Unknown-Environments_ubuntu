# Human Preference–Aware Risk Navigation

This module extends the existing navigation framework with a **human preference–aware risk model**.  
The goal is to combine:

- the robot’s internal risk reasoning (self-trust, OOD detection, drift, uncertainty), and  
- explicit human comfort and behavior preferences,

so that navigation is **both risk-aware and human-centered**.

---

## 1. Core Idea

We start from a standard risk-aware planner with a cost function:

```text
cost(path) = L(path) + λ * R(path)
Where:

L(path) = path length / time / energy

R(path) = risk metric (e.g., obstacle proximity, collision probability, uncertainty)

λ = risk weight

large λ → conservative, safe behavior

small λ → aggressive, performance-oriented behavior

1.1 Human risk attitude
We introduce a human risk preference:

text
Αντιγραφή κώδικα
h ∈ [0, 1]
h = 0   → very risk-averse
h = 0.5 → balanced
h = 1   → accepts high risk
The robot provides a baseline λ_robot (from self-trust, OOD, etc.).
Human preference modifies it into an effective λ:

text
Αντιγραφή κώδικα
λ_eff = λ_robot * f(h, α)
where α controls how strongly the human influences the final λ.
Intuitively:

safe human (low h) → λ_eff > λ_robot

aggressive human (high h) → λ_eff < λ_robot

The planner then uses:

text
Αντιγραφή κώδικα
cost = L + λ_eff * R
1.2 Semantic preferences
Humans can also express qualitative constraints, e.g.:

“Avoid dark areas”

“Avoid low-feature regions”

“Prefer well-mapped areas”

These are mapped to boolean flags:

avoid_dark_areas

avoid_low_feature_areas

prefer_well_mapped_areas

and applied as penalties on edges or cells that violate the requested preference.

2. Modules
2.1 Core human preference logic
user_preferences.py
Parses human preference text (English and Greek).

Produces a HumanPreference object with:

risk_preference: float (h in [0, 1])

avoid_dark_areas: bool

avoid_low_feature_areas: bool

prefer_well_mapped_areas: bool

This is the bridge between natural language and planner parameters.

human_risk_policy.py
Defines:

HumanRiskConfig
Configuration options:

human influence scale (how strong human input is),

min/max scaling factors for λ,

penalties for dark / low-feature areas.

HumanRiskPolicy
Main policy object with:

python
Αντιγραφή κώδικα
lambda_eff = policy.compute_lambda_effective(lambda_robot, human_pref)
cost = policy.edge_cost(
    base_length,
    edge_risk,
    lambda_effective=lambda_eff,
    human_pref=human_pref,
    is_dark=...,
    is_low_feature=...,
)
So the planner can stay simple and just call edge_cost(...).

risk_cost_utils.py
Contains a reusable helper:

python
Αντιγραφή κώδικα
human_aware_edge_cost(
    base_length,
    edge_risk,
    lambda_effective,
    human_pref,
    is_dark=False,
    is_low_feature=False,
    low_feature_penalty=5.0,
    dark_area_penalty=5.0,
)
This function is planner-agnostic and can be plugged into PRM, RRT, NBV, coverage planners, etc.

3. Toy Planners & Experiments
These are small, self-contained scripts used to validate the idea and generate simple results.

simple_risk_planner.py
Baseline risk-aware planner.

Three synthetic paths: A, B, C.

Cost model:

text
Αντιγραφή κώδικα
cost = length + lambda_robot * risk
Picks the path with lowest cost.

simple_risk_planner_human.py
Same three paths (A, B, C), but:

parses human preference text -> HumanPreference

computes lambda_eff using HumanRiskPolicy

adds penalties for dark / low-feature paths when requested by the human

Shows how different preferences change:

the effective λ

the selected path

save_simple_planner_results.py
Runs:

baseline planner

human-aware planner with multiple preference texts

Writes results to:

text
Αντιγραφή κώδικα
simple_planner_results.csv
Each row includes:

mode (baseline / human)

human_pref_text

path_name (A/B/C)

length, risk

is_dark, is_low_feature

lambda_robot, lambda_effective

cost

is_best (1 if this path is the best for that mode)

analyze_simple_planner_results.py
Reads simple_planner_results.csv.

Computes and prints statistics per (mode, human_pref_text):

how often each path wins,

average cost per path,

average effective λ.

Useful for tables and figures in a report or paper.

4. Real Planner Integration
human_aware_real_planner.py
Provides:

python
Αντιγραφή κώδικα
class HumanAwarePlannerWrapper:
    ...
This wrapper sits on top of any planner that has a plan(...) method.

Typical usage:

python
Αντιγραφή κώδικα
from modules.graph_planning.prm_planner import PRMPlanner
from human_aware_real_planner import HumanAwarePlannerWrapper

# 1. Build the existing planner
base_planner = PRMPlanner(...)

# 2. Wrap it with the human-aware layer
human_planner = HumanAwarePlannerWrapper(
    underlying_planner=base_planner,
    human_pref_text="Prefer safer route even if slower",
    human_influence_scale=1.0,
)

lambda_robot = 1.0  # from self-trust / OOD / drift / calibration

# 3. Use the wrapper instead of calling plan() directly
result, lambda_eff = human_planner.plan_with_human_lambda(
    lambda_robot=lambda_robot,
    start=start,
    goal=goal,
    # additional args/kwargs required by PRMPlanner.plan(...)
)
The wrapper:

parses human_pref_text → HumanPreference

computes lambda_eff

optionally sets planner.lambda_weight = lambda_eff

tries to pass λ to plan(...) as:

lambda_weight=lambda_eff, or

risk_weight=lambda_eff, or

lambda_risk=lambda_eff,

falls back to calling plan(...) without λ if none of those fit.

5. How to Run
From the project root (inside your virtual environment):

5.1 Toy examples
bash
Αντιγραφή κώδικα
python3 simple_risk_planner.py
python3 simple_risk_planner_human.py
5.2 Save and analyze results
bash
Αντιγραφή κώδικα
python3 save_simple_planner_results.py
python3 analyze_simple_planner_results.py
5.3 Real planner demo
bash
Αντιγραφή κώδικα
python3 run_human_preference_exp.py
python3 run_real_human_preference_demo.py
(Depending on your configuration and which planner you connect.)

6. Integrating with a Custom Planner
To use this framework with any custom planner:

Expose λ in your cost computation, for example:

python
Αντιγραφή κώδικα
cost = length + lambda_weight * risk
Optionally use semantic flags from HumanPreference:

python
Αντιγραφή κώδικα
if human_pref.avoid_dark_areas and is_dark:
    cost += dark_penalty
if human_pref.avoid_low_feature_areas and is_low_feature:
    cost += low_feature_penalty
Wrap the planner with HumanAwarePlannerWrapper and call:

python
Αντιγραφή κώδικα
plan_with_human_lambda(lambda_robot=..., ...)
instead of calling plan(...) directly.

7. Summary
This module introduces a human-centered risk adaptation layer on top of existing navigation planners:

Human intent is expressed in natural language.

It is converted into:

a continuous risk preference h ∈ [0, 1],

semantic constraints such as “avoid dark / low-feature areas”.

Robot-side risk reasoning (self-trust, OOD, drift) is combined with human preferences to produce an effective risk weight λ_eff.

The design is:

modular,

easy to plug into different planners,

validated with toy examples and CSV-based experiments,

ready for further integration into real robot navigation systems.

Navigation becomes not only risk-aware, but also aligned with human comfort and preferences.
