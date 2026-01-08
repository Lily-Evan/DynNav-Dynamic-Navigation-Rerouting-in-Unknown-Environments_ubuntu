#!/usr/bin/env python3
import os
import glob
import json
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pandas as pd

OUT_MD = "CLAIMS_EVIDENCE.md"


# -----------------------------
# Utilities
# -----------------------------
def exists(path: str) -> bool:
    return os.path.exists(path)

def expand_patterns(patterns: List[str]) -> List[str]:
    """Expand globs, keep stable order, unique."""
    out = []
    seen = set()
    for p in patterns:
        matches = glob.glob(p)
        if not matches:
            matches = [p]  # keep as literal (to show missing)
        for m in matches:
            if m not in seen:
                out.append(m)
                seen.add(m)
    return out

def md_code(s: str) -> str:
    return f"`{s}`"

def md_link_if_exists(path: str) -> str:
    # Markdown relative link works on GitHub for files in repo.
    if exists(path):
        return f"[{path}]({path})"
    return f"~~{path}~~ (missing)"

def read_csv_safely(path: str) -> Optional[pd.DataFrame]:
    try:
        if not exists(path):
            return None
        return pd.read_csv(path)
    except Exception:
        return None

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def fmt_float(x, nd=4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def summarize_csv_generic(path: str, max_rows: int = 5) -> str:
    df = read_csv_safely(path)
    if df is None:
        return f"- CSV not found: {md_code(path)}"
    # Show columns + head
    cols = ", ".join([md_code(c) for c in df.columns.tolist()])
    head = df.head(max_rows).to_dict(orient="records")
    return (
        f"- Columns: {cols}\n"
        f"- Head({max_rows}): `{json.dumps(head, ensure_ascii=False)}`"
    )


# -----------------------------
# Domain-specific summaries
# -----------------------------
def summarize_astar_eval(path: str) -> Optional[str]:
    """
    Try to extract expansions/cost from astar_eval_results.csv
    Works with common column names; falls back to generic summary.
    """
    df = read_csv_safely(path)
    if df is None:
        return None

    # Try common variants
    exp_col = pick_col(df, ["expansions", "node_expansions", "n_expansions"])
    cost_col = pick_col(df, ["cost", "path_cost", "g_cost", "total_cost"])
    method_col = pick_col(df, ["method", "planner", "variant", "type"])

    if method_col and (exp_col or cost_col):
        # Aggregate mean per method
        group = df.groupby(method_col).mean(numeric_only=True).reset_index()
        lines = []
        for _, row in group.iterrows():
            m = row[method_col]
            parts = []
            if exp_col:
                parts.append(f"mean expansions={fmt_float(row[exp_col], 2)}")
            if cost_col:
                parts.append(f"mean cost={fmt_float(row[cost_col], 4)}")
            lines.append(f"- {m}: " + ", ".join(parts))
        return "\n".join(lines)

    return summarize_csv_generic(path)


def summarize_tau_sweep(path: str) -> Optional[str]:
    df = read_csv_safely(path)
    if df is None:
        return None
    tau_col = pick_col(df, ["tau", "τ", "threshold_tau", "tau0"])
    succ_col = pick_col(df, ["success_rate", "success", "feasible_rate", "feasibility", "feasible"])

    if tau_col and succ_col:
        g = df.groupby(tau_col)[succ_col].mean().reset_index().sort_values(tau_col)
        # crude critical tau*: closest to 0.5 success if range crosses it; else max slope
        tau = g[tau_col].to_numpy(dtype=float)
        succ = g[succ_col].to_numpy(dtype=float)
        if succ.min() <= 0.5 <= succ.max():
            idx = int((abs(succ - 0.5)).argmin())
            tau_star = float(tau[idx])
            return (
                f"- tau range: [{fmt_float(tau.min(), 4)}, {fmt_float(tau.max(), 4)}]\n"
                f"- mean success range: [{fmt_float(succ.min(), 4)}, {fmt_float(succ.max(), 4)}]\n"
                f"- approx tau* (closest to success=0.5): {fmt_float(tau_star, 4)}"
            )
        # fallback: derivative peak
        import numpy as np
        d = np.gradient(succ, tau)
        idx = int(np.argmax(np.abs(d)))
        return (
            f"- tau range: [{fmt_float(tau.min(), 4)}, {fmt_float(tau.max(), 4)}]\n"
            f"- mean success range: [{fmt_float(succ.min(), 4)}, {fmt_float(succ.max(), 4)}]\n"
            f"- approx tau* (max |d(success)/d(tau)|): {fmt_float(float(tau[idx]), 4)}"
        )

    return summarize_csv_generic(path)


def summarize_lambda_sweep(path: str) -> Optional[str]:
    df = read_csv_safely(path)
    if df is None:
        return None
    lam_col = pick_col(df, ["lambda", "λ", "risk_lambda"])
    risk_col = pick_col(df, ["total_risk", "risk", "cum_risk", "aggregate_risk", "max_risk"])
    cost_col = pick_col(df, ["total_cost", "cost", "path_cost", "length", "distance"])

    if lam_col and (risk_col or cost_col):
        g = df.groupby(lam_col).mean(numeric_only=True).reset_index().sort_values(lam_col)
        # Take endpoints as summary
        first = g.iloc[0].to_dict()
        last = g.iloc[-1].to_dict()
        parts = [
            f"- lambda range: [{fmt_float(first[lam_col], 4)}, {fmt_float(last[lam_col], 4)}]"
        ]
        if risk_col:
            parts.append(f"- mean {risk_col} endpoints: {fmt_float(first[risk_col],4)} → {fmt_float(last[risk_col],4)}")
        if cost_col:
            parts.append(f"- mean {cost_col} endpoints: {fmt_float(first[cost_col],4)} → {fmt_float(last[cost_col],4)}")
        return "\n".join(parts)

    return summarize_csv_generic(path)


def summarize_ttests(path: str) -> Optional[str]:
    df = read_csv_safely(path)
    if df is None:
        return None
    # try common p-value columns
    p_col = pick_col(df, ["p_value", "p", "pval", "p-value"])
    eff_col = pick_col(df, ["cohens_d", "effect_size", "d"])
    if p_col:
        # show top 5 most significant
        dff = df.sort_values(p_col).head(5)
        rows = []
        for _, r in dff.iterrows():
            desc = []
            for c in df.columns:
                if c in [p_col, eff_col]:
                    continue
            rows.append(f"- p={fmt_float(r[p_col],6)}" + (f", d={fmt_float(r[eff_col],4)}" if eff_col else ""))
        return "\n".join(rows) if rows else summarize_csv_generic(path)
    return summarize_csv_generic(path)


# -----------------------------
# Claim structure
# -----------------------------
@dataclass
class Claim:
    title: str
    statement: str
    scripts: List[str] = field(default_factory=list)
    csvs: List[str] = field(default_factory=list)
    plots: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    summary_fn: Optional[str] = None  # name of summarizer


SUMMARIZERS = {
    "astar": summarize_astar_eval,
    "tau": summarize_tau_sweep,
    "lambda": summarize_lambda_sweep,
    "ttests": summarize_ttests,
    "generic": summarize_csv_generic,
}


def render_claim(cl: Claim) -> str:
    # Expand globs
    scripts = expand_patterns(cl.scripts)
    csvs = expand_patterns(cl.csvs)
    plots = expand_patterns(cl.plots)

    # Evidence lists
    md = []
    md.append(f"## {cl.title}\n")
    md.append(f"**Claim:** {cl.statement}\n")

    def list_block(title: str, items: List[str]) -> None:
        md.append(f"**{title}:**")
        if not items:
            md.append("- (none)\n")
            return
        for it in items:
            md.append(f"- {md_link_if_exists(it)}")
        md.append("")  # blank line

    list_block("Scripts", scripts)
    list_block("CSV / Logs", csvs)
    list_block("Plots / Figures", plots)

    # Quick summary from CSVs
    if csvs:
        md.append("**Quick evidence (from CSVs):**")
        summarizer = SUMMARIZERS.get(cl.summary_fn or "generic", summarize_csv_generic)
        any_summary = False
        for c in csvs:
            if exists(c) and c.lower().endswith(".csv"):
                s = summarizer(c)
                if s:
                    md.append(f"- {md_code(c)}")
                    md.append(s)
                    any_summary = True
        if not any_summary:
            md.append("- (no readable CSV summaries)")
        md.append("")

    if cl.notes:
        md.append(f"**Notes:** {cl.notes}\n")

    md.append("---\n")
    return "\n".join(md)


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    claims = [
        Claim(
            title="Learned A* reduces search effort while preserving optimality",
            statement="The learned admissible heuristic reduces node expansions compared to classical A* without increasing path cost.",
            scripts=["eval_astar_learned.py", "train_heuristic.py", "astar_learned_heuristic.py"],
            csvs=["astar_eval_results.csv"],
            plots=["astar_learned_vs_classic.png", "autotune_expansions.png", "autotune_ratio.png"],
            summary_fn="astar",
        ),
        Claim(
            title="Irreversibility threshold exhibits a feasibility phase transition",
            statement="As the irreversibility threshold τ increases, success/feasibility transitions sharply and expansions/cost change nonlinearly, indicating a critical τ* region.",
            scripts=["run_irreversibility_tau_sweep.py", "plot_irreversibility_tau_sweep.py"],
            csvs=["irreversibility_tau_sweep.csv", "irreversibility_bottleneck_tau_sweep.csv"],
            plots=[
                "irreversibility_success_vs_tau.png",
                "irreversibility_expansions_vs_tau.png",
                "irreversibility_cost_vs_tau.png",
                "irreversibility_maxI_vs_tau.png",
            ],
            summary_fn="tau",
        ),
        Claim(
            title="Risk-weighted planning yields a measurable risk–cost trade-off",
            statement="Sweeping λ produces a consistent trade-off: increasing risk aversion reduces risk metrics at the expense of increased cost/length and/or expansions.",
            scripts=["run_risk_weighted_lambda_sweep.py", "plot_risk_weighted_lambda_sweep.py"],
            csvs=["risk_weighted_lambda_sweep.csv"],
            plots=[
                "risk_weighted_geocost_vs_lambda.png",
                "risk_weighted_expansions_vs_lambda.png",
                "risk_weighted_maxI_vs_lambda.png",
                "risk_weighted_meanI_vs_lambda.png",
            ],
            summary_fn="lambda",
        ),
        Claim(
            title="Security monitoring detects estimation attacks and enables mitigation signals",
            statement="Innovation-based IDS and TF integrity IDS detect injected attacks (ROC/PR) and can provide planner-facing signals (trust/alarms) for mitigation.",
            scripts=[
                "eval_ids_sweep.py",
                "eval_ids_replay.py",
                "attack_injector.py",
                "security_monitor.py",
                "security_monitor_cusum.py",
            ],
            csvs=[
                "ids_roc.csv",
                "ids_pr.csv",
                "ids_replay_log.csv",
                "ids_methods_summary.csv",
                "attack_aware_ukf_demo_log.csv",
                "ids_to_planner_hook_log.csv",
            ],
            plots=[
                "ids_roc.png",
                "ids_pr.png",
                "ids_roc_compare.png",
                "attack_aware_nis.png",
                "attack_aware_trust.png",
                "ids_mitigation_outputs.png",
                "ids_alarm_safe_mode.png",
            ],
            summary_fn="generic",
        ),
        Claim(
            title="Ablations and statistical testing support robustness of conclusions",
            statement="Multi-seed experiments, ablations, and hypothesis testing quantify the effect of each module and validate that key differences are statistically meaningful.",
            scripts=[
                "batch_run_30_seeds.py",
                "batch_run_ablation_30_seeds.py",
                "run_ablation_t_tests.py",
                "analyze_statistical_validation.py",
            ],
            csvs=[
                "ablation_results.csv",
                "ablation_t_test_results.csv",
                "t_test_results.csv",
                "statistical_summary.csv",
            ],
            plots=[
                "boxplot_total_cost.png",
                "boxplot_total_risk.png",
                "boxplot_total_distance.png",
                "boxplot_max_risk.png",
            ],
            summary_fn="ttests",
        ),
    ]

    # Header
    md = []
    md.append("# Claims → Evidence Map\n")
    md.append(f"_Auto-generated on {now}_\n")
    md.append("This document maps research claims to concrete evidence in the repository (scripts, logs, figures).\n")
    md.append("> Missing files are shown as strikethrough.\n")
    md.append("---\n")

    for cl in claims:
        md.append(render_claim(cl))

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"[OK] Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
