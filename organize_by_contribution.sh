#!/usr/bin/env bash
set -euo pipefail

# Use git mv only for tracked files; otherwise mv.
is_git_repo=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  is_git_repo=1
fi

is_tracked() {
  # returns 0 if tracked
  git ls-files --error-unmatch "$1" >/dev/null 2>&1
}

mv_smart() {
  local src="$1"
  local dst_dir="$2"
  mkdir -p "$dst_dir"

  # skip if source doesn't exist
  [[ -e "$src" ]] || return 0

  if [[ -d "$src" ]]; then
    # directory move: prefer git mv if directory tracked (best effort)
    if [[ $is_git_repo -eq 1 ]]; then
      git mv "$src" "$dst_dir/" 2>/dev/null || mv "$src" "$dst_dir/"
    else
      mv "$src" "$dst_dir/"
    fi
    return 0
  fi

  if [[ $is_git_repo -eq 1 ]] && is_tracked "$src"; then
    git mv "$src" "$dst_dir/"
  else
    mv "$src" "$dst_dir/"
  fi
}

move_glob() {
  local pattern="$1"
  local dst="$2"
  shopt -s nullglob
  local arr=( $pattern )
  shopt -u nullglob
  for f in "${arr[@]}"; do
    mv_smart "$f" "$dst"
  done
}

echo "==> Creating structure..."
mkdir -p docs contributions

mk_contrib () {
  local c="$1"
  mkdir -p "contributions/$c/code" \
           "contributions/$c/experiments" \
           "contributions/$c/results" \
           "contributions/$c/models" \
           "contributions/$c/demos"
}

mk_contrib "01_learned_astar"
mk_contrib "02_uncertainty_calibration"
mk_contrib "03_belief_risk_planning"
mk_contrib "04_irreversibility_returnability"
mk_contrib "05_safe_mode_navigation"
mk_contrib "06_energy_connectivity"
mk_contrib "07_nbv_exploration"
mk_contrib "08_security_ids"
mk_contrib "09_multi_robot"
mk_contrib "10_human_language_ethics"
mkdir -p "contributions/_unsorted/files" "contributions/_unsorted/results"

echo "==> Moving top-level Markdown docs into docs/ (keeping README.md)..."
shopt -s nullglob
for md in *.md; do
  [[ "$md" == "README.md" ]] && continue
  mv_smart "$md" "docs"
done
shopt -u nullglob

echo "==> [01] Learned A*"
mv_smart "eval_astar_learned.py" "contributions/01_learned_astar/experiments"
move_glob "train_heuristic*.py"  "contributions/01_learned_astar/code"
move_glob "learned_heuristic*.py" "contributions/01_learned_astar/code"
mv_smart "astar_learned_heuristic.py" "contributions/01_learned_astar/code"
move_glob "heuristic_*.py" "contributions/01_learned_astar/code"
move_glob "heuristic_*.pt" "contributions/01_learned_astar/models"
move_glob "heuristic_*.npz" "contributions/01_learned_astar/models"
move_glob "astar_*" "contributions/01_learned_astar/results"
move_glob "autotune_*" "contributions/01_learned_astar/results"
mv_smart "timing_benchmark.py" "contributions/01_learned_astar/experiments"
mv_smart "timing_results.csv" "contributions/01_learned_astar/results"

echo "==> [02] Uncertainty calibration / drift"
move_glob "drift_*.py" "contributions/02_uncertainty_calibration/code"
move_glob "train_drift_*.py" "contributions/02_uncertainty_calibration/code"
move_glob "build_*uncertainty_grid*.py" "contributions/02_uncertainty_calibration/code"
move_glob "analyze_*uncertainty*calibration*.py" "contributions/02_uncertainty_calibration/experiments"
move_glob "*uncertainty_grid*.csv" "contributions/02_uncertainty_calibration/results"
move_glob "drift_*dataset*.csv" "contributions/02_uncertainty_calibration/results"
move_glob "drift_model*.npz" "contributions/02_uncertainty_calibration/models"
move_glob "drift_uncertainty*.pt" "contributions/02_uncertainty_calibration/models"
move_glob "drift_uncertainty*.npz" "contributions/02_uncertainty_calibration/models"
move_glob "drift_*.png" "contributions/02_uncertainty_calibration/results"
move_glob "drift_vs_uncertainty*.png" "contributions/02_uncertainty_calibration/results"

echo "==> [03] Belief-risk planning"
move_glob "belief_risk_*.py" "contributions/03_belief_risk_planning/code"
mv_smart "risk_budget.py" "contributions/03_belief_risk_planning/code"
mv_smart "risk_cost_utils.py" "contributions/03_belief_risk_planning/code"
move_glob "select_*risk_budget*.py" "contributions/03_belief_risk_planning/experiments"
move_glob "run_*risk_budget*.py" "contributions/03_belief_risk_planning/experiments"
move_glob "lambda_sweep*.py" "contributions/03_belief_risk_planning/experiments"
move_glob "*lambda_sweep*.csv" "contributions/03_belief_risk_planning/results"
move_glob "*lambda_sweep*.png" "contributions/03_belief_risk_planning/results"

echo "==> [04] Irreversibility / returnability"
move_glob "irreversibility_*.py" "contributions/04_irreversibility_returnability/code"
move_glob "returnability_*.py" "contributions/04_irreversibility_returnability/code"
move_glob "run_irreversibility*.py" "contributions/04_irreversibility_returnability/experiments"
move_glob "run_returnability*.py" "contributions/04_irreversibility_returnability/experiments"
move_glob "plot_irreversibility*.py" "contributions/04_irreversibility_returnability/experiments"
move_glob "plot_returnability*.py" "contributions/04_irreversibility_returnability/experiments"
move_glob "*irreversibility*.csv" "contributions/04_irreversibility_returnability/results"
move_glob "*irreversibility*.png" "contributions/04_irreversibility_returnability/results"
move_glob "*returnability*.csv" "contributions/04_irreversibility_returnability/results"
move_glob "*returnability*.png" "contributions/04_irreversibility_returnability/results"
mv_smart "hard_vs_soft_comparison.png" "contributions/04_irreversibility_returnability/results"
mv_smart "path_overlay_hard_vs_soft.png" "contributions/04_irreversibility_returnability/results"

echo "==> [05] Safe mode"
move_glob "safe_mode_*.py" "contributions/05_safe_mode_navigation/code"
move_glob "run_*safe_mode*.py" "contributions/05_safe_mode_navigation/experiments"
move_glob "analyze_safe_mode*.py" "contributions/05_safe_mode_navigation/experiments"
move_glob "*safe_mode*.csv" "contributions/05_safe_mode_navigation/results"
move_glob "*safe_mode*.png" "contributions/05_safe_mode_navigation/results"

echo "==> [06] Energy + connectivity"
move_glob "energy_*.py" "contributions/06_energy_connectivity/code"
move_glob "connectivity_*.py" "contributions/06_energy_connectivity/code"
mv_smart "net_channel.py" "contributions/06_energy_connectivity/code"
move_glob "run_*energy*.py" "contributions/06_energy_connectivity/experiments"
move_glob "run_*connectivity*.py" "contributions/06_energy_connectivity/experiments"
move_glob "*energy*.csv" "contributions/06_energy_connectivity/results"
move_glob "*connectivity*.csv" "contributions/06_energy_connectivity/results"
move_glob "*energy*.png" "contributions/06_energy_connectivity/results"
move_glob "*connectivity*.png" "contributions/06_energy_connectivity/results"
move_glob "heatmap_*disconnect*.png" "contributions/06_energy_connectivity/results"
move_glob "joint_*disconnect*.png" "contributions/06_energy_connectivity/results"

echo "==> [07] NBV / frontier / IG"
move_glob "nbv_*.py" "contributions/07_nbv_exploration/code"
move_glob "frontier_*.py" "contributions/07_nbv_exploration/code"
mv_smart "info_gain_planner.py" "contributions/07_nbv_exploration/code"
move_glob "option_entropy*.py" "contributions/07_nbv_exploration/code"
move_glob "add_option_entropy*.py" "contributions/07_nbv_exploration/code"
move_glob "analyze_option_entropy*.py" "contributions/07_nbv_exploration/experiments"
move_glob "run_nbv_*.py" "contributions/07_nbv_exploration/experiments"
move_glob "plot_nbv_*.py" "contributions/07_nbv_exploration/experiments"
move_glob "bench_*" "contributions/07_nbv_exploration/results"
move_glob "nbv_*.csv" "contributions/07_nbv_exploration/results"
move_glob "nbv_*.png" "contributions/07_nbv_exploration/results"
move_glob "entropy_*.png" "contributions/07_nbv_exploration/results"

echo "==> [08] Security / IDS"
move_glob "attack_*.py" "contributions/08_security_ids/code"
move_glob "security_monitor*.py" "contributions/08_security_ids/code"
move_glob "ids_*.py" "contributions/08_security_ids/code"
move_glob "eval_ids_*.py" "contributions/08_security_ids/experiments"
mv_smart "calibrate_tf_cusum.py" "contributions/08_security_ids/experiments"
move_glob "tf_*" "contributions/08_security_ids/results"
move_glob "ids_*.csv" "contributions/08_security_ids/results"
move_glob "ids_*.png" "contributions/08_security_ids/results"
move_glob "attack_aware_*.png" "contributions/08_security_ids/results"

echo "==> [09] Multi-robot + memory"
move_glob "multi_robot_*.py" "contributions/09_multi_robot/code"
move_glob "*disagreement*.py" "contributions/09_multi_robot/code"
move_glob "memory_aware_*.py" "contributions/09_multi_robot/code"
move_glob "run_multi_robot_*.py" "contributions/09_multi_robot/experiments"
move_glob "analyze_multi_robot_*.py" "contributions/09_multi_robot/experiments"
move_glob "*multi_robot*.csv" "contributions/09_multi_robot/results"
move_glob "*multi_robot*.png" "contributions/09_multi_robot/results"
move_glob "memory_*.csv" "contributions/09_multi_robot/results"
move_glob "memory_*.png" "contributions/09_multi_robot/results"

echo "==> [10] Human / trust / language / ethics"
move_glob "human_*.py" "contributions/10_human_language_ethics/code"
mv_smart "user_preferences.py" "contributions/10_human_language_ethics/code"
move_glob "trust_*.py" "contributions/10_human_language_ethics/code"
move_glob "ethical_*.py" "contributions/10_human_language_ethics/code"
move_glob "language_safety*.py" "contributions/10_human_language_ethics/code"
move_glob "ask_for_help_*.py" "contributions/10_human_language_ethics/demos"
move_glob "*_demo.py" "contributions/10_human_language_ethics/demos"
move_glob "*trust*.png" "contributions/10_human_language_ethics/results"
move_glob "*ethical*.png" "contributions/10_human_language_ethics/results"
move_glob "*language*.png" "contributions/10_human_language_ethics/results"

echo "==> Sweeping leftover top-level png/csv/txt into _unsorted..."
shopt -s nullglob
for f in *.png *.csv *.txt; do
  [[ "$f" == "requirements.txt" ]] && continue
  mv_smart "$f" "contributions/_unsorted/results"
done
shopt -u nullglob

echo "==> Done."
echo "Run: git status"
