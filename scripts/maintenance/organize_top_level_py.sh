#!/usr/bin/env bash
set -euo pipefail

use_git=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  use_git=1
fi

mv_cmd() { if [[ $use_git -eq 1 ]]; then git mv "$@"; else mv "$@"; fi; }
mkdir -p scripts analysis generators core utils || true

# μόνο αρχεία .py που είναι στο root
shopt -s nullglob
for f in *.py; do
  case "$f" in
    organize_*|setup.py) continue ;;
  esac

  case "$f" in
    run_*|eval_*|demo_*|test_*)                 dest="scripts" ;;
    analyze_*|plot_*|viz_*|dashboard.py)        dest="analysis" ;;
    build_*|generate_*|export_*|save_*|collect_*|compute_*|log_*) dest="generators" ;;
    *_planner.py|*_policy.py|*_layer.py|*_models.py|*_model.py|*_utils.py|*_estimator.py|*_controller.py|*_fusion.py|*_monitor*.py|*_taxonomy.py|*_templates.py|abstract_planner.py|advanced_planners.py|multi_sampling_based_planners.py|multi_planners_grid.py|path_smoothing.py|theory_risk_models.py|continual_learning.py|learning_language_safety.py|online_update_heuristic.py) dest="core" ;;
    *) dest="utils" ;;
  esac

  mv_cmd "$f" "$dest/$f"
done

echo "Done. Check: git status"
