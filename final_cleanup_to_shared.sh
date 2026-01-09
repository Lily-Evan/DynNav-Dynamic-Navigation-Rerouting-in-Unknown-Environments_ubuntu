#!/usr/bin/env bash
set -euo pipefail

# Use git mv if inside git repo (preserves history)
use_git=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  use_git=1
fi
mv_cmd() { if [[ $use_git -eq 1 ]]; then git mv "$@"; else mv "$@"; fi; }

echo "==> Creating shared structure..."
mkdir -p contributions/00_shared/{code,experiments,analysis,utils} || true

# Root keepers: never move these
is_keeper() {
  case "$1" in
    README.md|README*.md|LICENSE|LICENSE.md|CITATION.cff|requirements.txt|setup.py|setup.cfg|pyproject.toml|package.xml|.gitignore)
      return 0 ;;
    *)
      return 1 ;;
  esac
}

echo "==> Moving top-level Python files into contributions/00_shared/..."
shopt -s nullglob
for f in *.py; do
  # skip keepers or organizer scripts
  if is_keeper "$f"; then
    continue
  fi
  case "$f" in
    organize_*|final_cleanup_to_shared.sh) continue ;;
  esac

  case "$f" in
    run_*|eval_*|demo_*|test_*)
      dest="contributions/00_shared/experiments" ;;
    analyze_*|plot_*|viz_*|dashboard.py)
      dest="contributions/00_shared/analysis" ;;
    build_*|generate_*|export_*|save_*|collect_*|compute_*|log_*|add_*)
      dest="contributions/00_shared/utils" ;;
    *_planner.py|*_policy.py|*_layer.py|*_model.py|*_models.py|*_utils.py|*_estimator.py|*_controller.py|*_fusion.py|*_monitor*.py|*_taxonomy.py|*_templates.py|abstract_planner.py|advanced_planners.py|multi_sampling_based_planners.py|multi_planners_grid.py|path_smoothing.py|theory_risk_models.py|continual_learning.py|learning_language_safety.py|online_update_heuristic.py)
      dest="contributions/00_shared/code" ;;
    *)
      dest="contributions/00_shared/utils" ;;
  esac

  mv_cmd "$f" "$dest/$f"
done

echo "==> Done. Run: git status"
