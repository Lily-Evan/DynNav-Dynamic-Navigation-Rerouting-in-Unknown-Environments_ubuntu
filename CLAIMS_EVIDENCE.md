# Claims → Evidence Map

_Auto-generated on 2026-01-08 12:07:37_

This document maps research claims to concrete evidence in the repository (scripts, logs, figures).

> Missing files are shown as strikethrough.

---

## Learned A* reduces search effort while preserving optimality

**Claim:** The learned admissible heuristic reduces node expansions compared to classical A* without increasing path cost.

**Scripts:**
- [eval_astar_learned.py](eval_astar_learned.py)
- [train_heuristic.py](train_heuristic.py)
- [astar_learned_heuristic.py](astar_learned_heuristic.py)

**CSV / Logs:**
- [astar_eval_results.csv](astar_eval_results.csv)

**Plots / Figures:**
- [astar_learned_vs_classic.png](astar_learned_vs_classic.png)
- [autotune_expansions.png](autotune_expansions.png)
- [autotune_ratio.png](autotune_ratio.png)

**Quick evidence (from CSVs):**
- `astar_eval_results.csv`
- Columns: `seed`, `classic_path_len`, `classic_expansions`, `learned_path_len`, `learned_expansions`, `expansion_ratio`, `delta_path`
- Head(5): `[{"seed": 0, "classic_path_len": 69, "classic_expansions": 1099, "learned_path_len": 69, "learned_expansions": 447, "expansion_ratio": 0.4067333939945405, "delta_path": 0}, {"seed": 1, "classic_path_len": 69, "classic_expansions": 1099, "learned_path_len": 69, "learned_expansions": 447, "expansion_ratio": 0.4067333939945405, "delta_path": 0}, {"seed": 2, "classic_path_len": 69, "classic_expansions": 1099, "learned_path_len": 69, "learned_expansions": 447, "expansion_ratio": 0.4067333939945405, "delta_path": 0}, {"seed": 3, "classic_path_len": 69, "classic_expansions": 1099, "learned_path_len": 69, "learned_expansions": 447, "expansion_ratio": 0.4067333939945405, "delta_path": 0}, {"seed": 4, "classic_path_len": 69, "classic_expansions": 1099, "learned_path_len": 69, "learned_expansions": 447, "expansion_ratio": 0.4067333939945405, "delta_path": 0}]`

---

## Irreversibility threshold exhibits a feasibility phase transition

**Claim:** As the irreversibility threshold τ increases, success/feasibility transitions sharply and expansions/cost change nonlinearly, indicating a critical τ* region.

**Scripts:**
- [run_irreversibility_tau_sweep.py](run_irreversibility_tau_sweep.py)
- [plot_irreversibility_tau_sweep.py](plot_irreversibility_tau_sweep.py)

**CSV / Logs:**
- [irreversibility_tau_sweep.csv](irreversibility_tau_sweep.csv)
- [irreversibility_bottleneck_tau_sweep.csv](irreversibility_bottleneck_tau_sweep.csv)

**Plots / Figures:**
- [irreversibility_success_vs_tau.png](irreversibility_success_vs_tau.png)
- [irreversibility_expansions_vs_tau.png](irreversibility_expansions_vs_tau.png)
- [irreversibility_cost_vs_tau.png](irreversibility_cost_vs_tau.png)
- [irreversibility_maxI_vs_tau.png](irreversibility_maxI_vs_tau.png)

**Quick evidence (from CSVs):**
- `irreversibility_tau_sweep.csv`
- tau range: [0.3000, 1.0000]
- mean success range: [0.0000, 1.0000]
- approx tau* (closest to success=0.5): 0.3000
- `irreversibility_bottleneck_tau_sweep.csv`
- tau range: [0.3000, 1.0000]
- mean success range: [0.0000, 1.0000]
- approx tau* (closest to success=0.5): 0.3000

---

## Risk-weighted planning yields a measurable risk–cost trade-off

**Claim:** Sweeping λ produces a consistent trade-off: increasing risk aversion reduces risk metrics at the expense of increased cost/length and/or expansions.

**Scripts:**
- [run_risk_weighted_lambda_sweep.py](run_risk_weighted_lambda_sweep.py)
- [plot_risk_weighted_lambda_sweep.py](plot_risk_weighted_lambda_sweep.py)

**CSV / Logs:**
- [risk_weighted_lambda_sweep.csv](risk_weighted_lambda_sweep.csv)

**Plots / Figures:**
- [risk_weighted_geocost_vs_lambda.png](risk_weighted_geocost_vs_lambda.png)
- [risk_weighted_expansions_vs_lambda.png](risk_weighted_expansions_vs_lambda.png)
- [risk_weighted_maxI_vs_lambda.png](risk_weighted_maxI_vs_lambda.png)
- [risk_weighted_meanI_vs_lambda.png](risk_weighted_meanI_vs_lambda.png)

**Quick evidence (from CSVs):**
- `risk_weighted_lambda_sweep.csv`
- lambda range: [0.0000, 12.0000]
- mean total_cost endpoints: 44.0000 → 74.8000

---

## Security monitoring detects estimation attacks and enables mitigation signals

**Claim:** Innovation-based IDS and TF integrity IDS detect injected attacks (ROC/PR) and can provide planner-facing signals (trust/alarms) for mitigation.

**Scripts:**
- [eval_ids_sweep.py](eval_ids_sweep.py)
- [eval_ids_replay.py](eval_ids_replay.py)
- [attack_injector.py](attack_injector.py)
- [security_monitor.py](security_monitor.py)
- [security_monitor_cusum.py](security_monitor_cusum.py)

**CSV / Logs:**
- [ids_roc.csv](ids_roc.csv)
- [ids_pr.csv](ids_pr.csv)
- [ids_replay_log.csv](ids_replay_log.csv)
- [ids_methods_summary.csv](ids_methods_summary.csv)
- [attack_aware_ukf_demo_log.csv](attack_aware_ukf_demo_log.csv)
- [ids_to_planner_hook_log.csv](ids_to_planner_hook_log.csv)

**Plots / Figures:**
- [ids_roc.png](ids_roc.png)
- [ids_pr.png](ids_pr.png)
- [ids_roc_compare.png](ids_roc_compare.png)
- [attack_aware_nis.png](attack_aware_nis.png)
- [attack_aware_trust.png](attack_aware_trust.png)
- [ids_mitigation_outputs.png](ids_mitigation_outputs.png)
- [ids_alarm_safe_mode.png](ids_alarm_safe_mode.png)

**Quick evidence (from CSVs):**
- `ids_roc.csv`
- Columns: `thr`, `FPR`, `TPR`, `TP`, `FP`, `TN`, `FN`
- Head(5): `[{"thr": 0.00036005842435893646, "FPR": 1.0, "TPR": 1.0, "TP": 5100.0, "FP": 2400.0, "TN": 0.0, "FN": 0.0}, {"thr": 0.0072863495283279345, "FPR": 0.9920833333333331, "TPR": 0.9976470588235294, "TP": 5088.0, "FP": 2381.0, "TN": 19.0, "FN": 12.0}, {"thr": 0.014452463807066272, "FPR": 0.9875, "TPR": 0.9939215686274508, "TP": 5069.0, "FP": 2370.0, "TN": 30.0, "FN": 31.0}, {"thr": 0.021361545386357338, "FPR": 0.9816666666666666, "TPR": 0.9907843137254903, "TP": 5053.0, "FP": 2356.0, "TN": 44.0, "FN": 47.0}, {"thr": 0.03059746302990957, "FPR": 0.9758333333333331, "TPR": 0.9876470588235294, "TP": 5037.0, "FP": 2342.0, "TN": 58.0, "FN": 63.0}]`
- `ids_pr.csv`
- Columns: `thr`, `precision`, `recall`, `TP`, `FP`, `TN`, `FN`
- Head(5): `[{"thr": 0.00036005842435893646, "precision": 0.68, "recall": 1.0, "TP": 5100.0, "FP": 2400.0, "TN": 0.0, "FN": 0.0}, {"thr": 0.0072863495283279345, "precision": 0.6812156915249699, "recall": 0.9976470588235294, "TP": 5088.0, "FP": 2381.0, "TN": 19.0, "FN": 12.0}, {"thr": 0.014452463807066272, "precision": 0.6814087915042344, "recall": 0.9939215686274508, "TP": 5069.0, "FP": 2370.0, "TN": 30.0, "FN": 31.0}, {"thr": 0.021361545386357338, "precision": 0.6820083682008368, "recall": 0.9907843137254903, "TP": 5053.0, "FP": 2356.0, "TN": 44.0, "FN": 47.0}, {"thr": 0.03059746302990957, "precision": 0.6826128201653341, "recall": 0.9876470588235294, "TP": 5037.0, "FP": 2342.0, "TN": 58.0, "FN": 63.0}]`
- `ids_replay_log.csv`
- Columns: `t`, `attack`, `d2`, `thr`, `ratio`, `scale`, `flagged`, `streak`, `triggered`, `alert`
- Head(5): `[{"t": 0, "attack": 0, "d2": 1.241781, "thr": 11.369058, "ratio": 0.109225, "scale": 1.0, "flagged": 0, "streak": 0, "triggered": 0, "alert": 0}, {"t": 1, "attack": 0, "d2": 1.285974, "thr": 11.369058, "ratio": 0.113112, "scale": 1.0, "flagged": 0, "streak": 0, "triggered": 0, "alert": 0}, {"t": 2, "attack": 0, "d2": 2.307891, "thr": 11.369058, "ratio": 0.202998, "scale": 1.0, "flagged": 0, "streak": 0, "triggered": 0, "alert": 0}, {"t": 3, "attack": 0, "d2": 0.136935, "thr": 11.369058, "ratio": 0.012045, "scale": 1.0, "flagged": 0, "streak": 0, "triggered": 0, "alert": 0}, {"t": 4, "attack": 0, "d2": 0.342968, "thr": 11.369058, "ratio": 0.030167, "scale": 1.0, "flagged": 0, "streak": 0, "triggered": 0, "alert": 0}]`
- `ids_methods_summary.csv`
- Columns: `method`, `thr_star`, `FPR_star`, `TPR_star`, `ROC_AUC`, `detect_rate`, `delay_mean_steps`, `delay_median_steps`, `delay_p90_steps`
- Head(5): `[{"method": "raw_nis", "thr_star": 8.36851333904353, "FPR_star": 0.0095833333333333, "TPR_star": 0.0201960784313725, "ROC_AUC": 0.5604361519607843, "detect_rate": 0.9666666666666668, "delay_mean_steps": 63.55172413793103, "delay_median_steps": 65.0, "delay_p90_steps": 107.99999999999996}, {"method": "ewma", "thr_star": 2.2979339096192786, "FPR_star": 0.0095833333333333, "TPR_star": 0.2856862745098039, "ROC_AUC": 0.8713002450980392, "detect_rate": 1.0, "delay_mean_steps": 63.53333333333333, "delay_median_steps": 60.0, "delay_p90_steps": 113.10000000000002}, {"method": "cusum", "thr_star": 49.286081422678514, "FPR_star": 0.0095833333333333, "TPR_star": 0.600392156862745, "ROC_AUC": 0.9317908088235294, "detect_rate": 1.0, "delay_mean_steps": 65.6, "delay_median_steps": 67.0, "delay_p90_steps": 113.1}]`
- `attack_aware_ukf_demo_log.csv`
- Columns: `t`, `est_px`, `est_py`, `vo_nis`, `vo_trust`, `vo_infl`, `w_nis`, `w_trust`, `w_infl`
- Head(5): `[{"t": 0.0, "est_px": 0.188405117661436, "est_py": 0.10965558073137809, "vo_nis": 0.07820283005166061, "vo_trust": 1.0, "vo_infl": 1.0, "w_nis": 3.374804377237268, "w_trust": 1.0, "w_infl": 1.0}, {"t": 1.0, "est_px": 0.29253641732889035, "est_py": 0.09157810544079825, "vo_nis": 1.8217806406449792, "vo_trust": 1.0, "vo_infl": 1.0, "w_nis": 0.00591453850814169, "w_trust": 1.0, "w_infl": 1.0}, {"t": 2.0, "est_px": 0.3305497297076546, "est_py": 0.16702080561203103, "vo_nis": 2.1152084887812803, "vo_trust": 1.0, "vo_infl": 1.0, "w_nis": 1.9345627171857565, "w_trust": 1.0, "w_infl": 1.0}, {"t": 3.0, "est_px": 0.42557501129028574, "est_py": 0.21189529199692245, "vo_nis": 0.2260889679225065, "vo_trust": 1.0, "vo_infl": 1.0, "w_nis": 0.09327624276289125, "w_trust": 1.0, "w_infl": 1.0}, {"t": 4.0, "est_px": 0.5406613443887691, "est_py": 0.2384646971920185, "vo_nis": 0.9572494520056984, "vo_trust": 1.0, "vo_infl": 1.0, "w_nis": 0.6266730561242935, "w_trust": 1.0, "w_infl": 1.0}]`
- `ids_to_planner_hook_log.csv`
- Columns: `t`, `nis`, `ewma`, `cusum`, `alarm`, `safe_mode`, `vo_trust`, `lambda_eff`
- Head(5): `[{"t": 0.0, "nis": 0.07820283005166061, "ewma": 0.07820283005166061, "cusum": 0.0, "alarm": 0.0, "safe_mode": 0.0, "vo_trust": 1.0, "lambda_eff": 0.2}, {"t": 1.0, "nis": 1.8217806406449792, "ewma": 0.11307438626352696, "cusum": 1.321780640644979, "alarm": 0.0, "safe_mode": 0.0, "vo_trust": 1.0, "lambda_eff": 0.2}, {"t": 2.0, "nis": 2.1152084887812803, "ewma": 0.15311706831388203, "cusum": 2.9369891294262596, "alarm": 0.0, "safe_mode": 0.0, "vo_trust": 1.0, "lambda_eff": 0.2}, {"t": 3.0, "nis": 0.2260889679225065, "ewma": 0.15457650630605455, "cusum": 2.6630780973487655, "alarm": 0.0, "safe_mode": 0.0, "vo_trust": 1.0, "lambda_eff": 0.2}, {"t": 4.0, "nis": 0.9572494520056984, "ewma": 0.1706299652200474, "cusum": 3.1203275493544647, "alarm": 0.0, "safe_mode": 0.0, "vo_trust": 1.0, "lambda_eff": 0.2}]`

---

## Ablations and statistical testing support robustness of conclusions

**Claim:** Multi-seed experiments, ablations, and hypothesis testing quantify the effect of each module and validate that key differences are statistically meaningful.

**Scripts:**
- [batch_run_30_seeds.py](batch_run_30_seeds.py)
- [batch_run_ablation_30_seeds.py](batch_run_ablation_30_seeds.py)
- [run_ablation_t_tests.py](run_ablation_t_tests.py)
- [analyze_statistical_validation.py](analyze_statistical_validation.py)

**CSV / Logs:**
- [ablation_results.csv](ablation_results.csv)
- [ablation_t_test_results.csv](ablation_t_test_results.csv)
- [t_test_results.csv](t_test_results.csv)
- [statistical_summary.csv](statistical_summary.csv)

**Plots / Figures:**
- [boxplot_total_cost.png](boxplot_total_cost.png)
- [boxplot_total_risk.png](boxplot_total_risk.png)
- [boxplot_total_distance.png](boxplot_total_distance.png)
- [boxplot_max_risk.png](boxplot_max_risk.png)

**Quick evidence (from CSVs):**
- `ablation_results.csv`
- Columns: `mode`, `w_ent`, `w_unc`, `w_cost`, `x`, `y`, `entropy_gain`, `uncertainty_gain`, `distance`, `score`
- Head(5): `[{"mode": "entropy_only", "w_ent": 1.0, "w_unc": 0.0, "w_cost": 0.2, "x": 10.125925925925928, "y": 2.5185185185185186, "entropy_gain": 23.576034179122303, "uncertainty_gain": 42.60000000000001, "distance": 10.44030650891055, "score": 0.8000000000000002}, {"mode": "uncertainty_only", "w_ent": 0.0, "w_unc": 1.0, "w_cost": 0.2, "x": 15.115776081424936, "y": 25.346055979643765, "entropy_gain": 26.523038451512587, "uncertainty_gain": 57.6, "distance": 29.154759474226505, "score": 0.8}, {"mode": "combined", "w_ent": 0.5, "w_unc": 0.3, "w_cost": 0.2, "x": 15.115776081424936, "y": 25.346055979643765, "entropy_gain": 26.523038451512587, "uncertainty_gain": 57.6, "distance": 29.154759474226505, "score": 0.6000000000000001}]`
- `ablation_t_test_results.csv`
- p=0.000000
- p=0.000000
- p=0.000000
- p=0.000000
- p=0.000000
- `t_test_results.csv`
- p=0.000000
- p=0.000000
- p=0.000008
- p=1.000000
- `statistical_summary.csv`
- Columns: `variant`, `metric`, `N`, `mean`, `std`, `ci_95`, `ci_low`, `ci_high`
- Head(5): `[{"variant": "astar_classic", "metric": "path_length", "N": 5, "mean": 14.2, "std": 0.1581138830084192, "ci_95": 0.1385929291125635, "ci_low": 14.061407070887435, "ci_high": 14.338592929112563}, {"variant": "astar_classic", "metric": "time_to_goal", "N": 5, "mean": 4.2, "std": 0.1581138830084191, "ci_95": 0.1385929291125634, "ci_low": 4.061407070887436, "ci_high": 4.338592929112564}, {"variant": "astar_classic", "metric": "num_expansions", "N": 5, "mean": 852.0, "std": 1.5811388300841898, "ci_95": 1.3859292911256331, "ci_low": 850.6140707088744, "ci_high": 853.3859292911256}, {"variant": "astar_classic", "metric": "num_replans", "N": 5, "mean": 2.0, "std": 1.5811388300841898, "ci_95": 1.3859292911256331, "ci_low": 0.6140707088743669, "ci_high": 3.385929291125633}, {"variant": "astar_classic", "metric": "coverage_percent", "N": 5, "mean": 85.2, "std": 0.1581138830084212, "ci_95": 0.1385929291125652, "ci_low": 85.06140707088744, "ci_high": 85.33859292911256}]`

---
