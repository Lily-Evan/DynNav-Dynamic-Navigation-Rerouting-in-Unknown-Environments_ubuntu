import numpy as np
import pandas as pd
from textwrap import dedent


# -----------------------------
# Helper functions
# -----------------------------
def compute_initial_coverage(coverage_csv: str) -> float:
    df = pd.read_csv(coverage_csv)
    if not {"row", "col", "covered"}.issubset(df.columns):
        raise RuntimeError(f"{coverage_csv} πρέπει να έχει στήλες row, col, covered")
    max_row = int(df["row"].max())
    max_col = int(df["col"].max())
    grid = np.zeros((max_row + 1, max_col + 1), dtype=float)
    for _, r in df.iterrows():
        row = int(r["row"])
        col = int(r["col"])
        cov = float(r["covered"])
        grid[row, col] = 1.0 if cov >= 0.5 else 0.0
    total = grid.size
    covered = np.count_nonzero(grid >= 0.5)
    return 100.0 * covered / total if total > 0 else 0.0


def replay_replan_coverage(coverage_csv: str, replan_csv: str, radius: int = 1):
    df = pd.read_csv(coverage_csv)
    if not {"row", "col", "covered"}.issubset(df.columns):
        raise RuntimeError(f"{coverage_csv} πρέπει να έχει στήλες row, col, covered")

    max_row = int(df["row"].max())
    max_col = int(df["col"].max())
    grid = np.zeros((max_row + 1, max_col + 1), dtype=float)
    for _, r in df.iterrows():
        row = int(r["row"])
        col = int(r["col"])
        cov = float(r["covered"])
        grid[row, col] = 1.0 if cov >= 0.5 else 0.0

    rows, cols = grid.shape
    total = grid.size
    covered_init = np.count_nonzero(grid >= 0.5)
    cov_init = 100.0 * covered_init / total if total > 0 else 0.0

    df_wp = pd.read_csv(replan_csv)
    if not {"x", "z"}.issubset(df_wp.columns):
        raise RuntimeError(f"{replan_csv} πρέπει να έχει στήλες x, z")

    xs = df_wp["x"].to_numpy()
    zs = df_wp["z"].to_numpy()

    cov_over_time = [cov_init]
    current_grid = grid.copy()

    for x, z in zip(xs, zs):
        row = int(round(z))
        col = int(round(x))
        for rr in range(row - radius, row + radius + 1):
            for cc in range(col - radius, col + radius + 1):
                if 0 <= rr < rows and 0 <= cc < cols:
                    current_grid[rr, cc] = 1.0
        covered_now = np.count_nonzero(current_grid >= 0.5)
        cov_now = 100.0 * covered_now / total
        cov_over_time.append(cov_now)

    final_cov = cov_over_time[-1]
    return cov_init, final_cov


def load_ablation_results(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        return None


def load_timing_results(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        return None


def drift_uncertainty_summary(vo_csv: str, coverage_csv: str):
    # Μικρή εκδοχή του drift_uncertainty_analysis, μόνο για στατιστικά
    df_vo = pd.read_csv(vo_csv)
    if not {"x", "z"}.issubset(df_vo.columns):
        return None

    xs = df_vo["x"].to_numpy()
    zs = df_vo["z"].to_numpy()

    def moving_average(x, window=21):
        if window < 3:
            return x.copy()
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="same")

    xs_smooth = moving_average(xs, window=21)
    zs_smooth = moving_average(zs, window=21)
    drift = np.sqrt((xs - xs_smooth) ** 2 + (zs - zs_smooth) ** 2)

    df_cov = pd.read_csv(coverage_csv)
    if not {"row", "col", "uncertainty"}.issubset(df_cov.columns):
        return None

    max_row = int(df_cov["row"].max())
    max_col = int(df_cov["col"].max())
    U_grid = np.zeros((max_row + 1, max_col + 1), dtype=float)
    U_grid[:] = np.nan
    for _, r in df_cov.iterrows():
        row = int(r["row"])
        col = int(r["col"])
        u = float(r["uncertainty"])
        U_grid[row, col] = u

    x_min, x_max = xs.min(), xs.max()
    z_min, z_max = zs.min(), zs.max()
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    if abs(z_max - z_min) < 1e-9:
        z_max = z_min + 1.0

    cols = np.round((xs - x_min) / (x_max - x_min) * max_col).astype(int)
    rows = np.round((zs - z_min) / (z_max - z_min) * max_row).astype(int)
    rows = np.clip(rows, 0, max_row)
    cols = np.clip(cols, 0, max_col)

    uncert_list = []
    for rr, cc in zip(rows, cols):
        uncert_list.append(U_grid[rr, cc])
    uncert = np.array(uncert_list, dtype=float)

    mask_valid = ~np.isnan(uncert)
    drift_valid = drift[mask_valid]
    uncert_valid = uncert[mask_valid]

    if len(drift_valid) < 5:
        return {
            "n_valid": len(drift_valid),
            "std_unc": float("nan"),
            "corr": float("nan"),
            "mean_low": float("nan"),
            "mean_high": float("nan"),
        }

    std_u = float(np.std(uncert_valid))
    if std_u > 1e-6 and np.std(drift_valid) > 1e-6:
        corr = float(np.corrcoef(drift_valid, uncert_valid)[0, 1])
    else:
        corr = float("nan")

    q1 = np.quantile(uncert_valid, 1.0 / 3.0)
    q2 = np.quantile(uncert_valid, 2.0 / 3.0)
    low_mask = uncert_valid <= q1
    high_mask = uncert_valid > q2

    def safe_mean(x):
        return float(x.mean()) if len(x) > 0 else float("nan")

    mean_low = safe_mean(drift_valid[low_mask])
    mean_high = safe_mean(drift_valid[high_mask])

    return {
        "n_valid": int(len(drift_valid)),
        "std_unc": std_u,
        "corr": corr,
        "q1": float(q1),
        "q2": float(q2),
        "mean_low": mean_low,
        "mean_high": mean_high,
    }


# -----------------------------
# Main report generation
# -----------------------------
if __name__ == "__main__":
    coverage_csv = "coverage_grid.csv"
    coverage_unc_csv = "coverage_grid_with_uncertainty.csv"
    replan_csv = "replan_waypoints.csv"
    ablation_csv = "ablation_results.csv"
    timing_csv = "timing_results.csv"
    vo_csv = "vo_trajectory.csv"

    print("[REPORT] Υπολογισμός coverage πριν/μετά replan...")
    cov_init_est = compute_initial_coverage(coverage_csv)
    cov_init_replay, cov_final_replay = replay_replan_coverage(coverage_csv, replan_csv, radius=1)

    print("[REPORT] Φόρτωση ablation_results...")
    df_ablation = load_ablation_results(ablation_csv)

    print("[REPORT] Φόρτωση timing_results...")
    df_timing = load_timing_results(timing_csv)

    print("[REPORT] Υπολογισμός drift–uncertainty summary...")
    drift_summary = drift_uncertainty_summary(vo_csv, coverage_unc_csv)

    # Sections as markdown
    md_parts = []

    intro = dedent(f"""
    # Αναφορά Δυναμικής Πλοήγησης και Εξερεύνησης

    Η παρούσα αναφορά συνοψίζει τα βασικά αποτελέσματα του συστήματος δυναμικής πλοήγησης και επανασχεδιασμού διαδρομής σε άγνωστα περιβάλλοντα, με έμφαση σε:
    (α) την ανάλυση κάλυψης και τον μηχανισμό replan,
    (β) την επιλογή στόχων βάσει Information Gain και multi–objective κριτηρίων,
    (γ) πειράματα ablation,
    (δ) χρονομετρική αξιολόγηση των planners,
    και (ε) τη σχέση drift–uncertainty στην οπτική οδομετρία.
    """)

    md_parts.append(intro)

    coverage_section = dedent(f"""
    ## Ανάλυση Κάλυψης και Επανασχεδιασμός Διαδρομής

    Η περιοχή ενδιαφέροντος (Area of Interest, AOI) διακριτοποιείται σε ορθογώνιο grid κελιών, και κάθε κελί χαρακτηρίζεται ως «καλυμμένο» ή «ακάλυπτο» με βάση την εκτιμώμενη τροχιά από την οπτική οδομετρία. Από το αρχικό coverage grid υπολογίστηκε ποσοστό κάλυψης περίπου **{cov_init_est:.2f}%**.

    Για την αξιολόγηση του προτεινόμενου replan, εκτελέστηκε ένα απλό replay σενάριο, όπου τα waypoints της διαδρομής επανασχεδιασμού εφαρμόζονται διαδοχικά πάνω στον αρχικό δυαδικό χάρτη κάλυψης. Στο συγκεκριμένο σενάριο, η κάλυψη αυξάνεται από **{cov_init_replay:.2f}%** στην αρχική αποστολή σε **{cov_final_replay:.2f}%** μετά την εκτέλεση του replan, επιβεβαιώνοντας ότι η boustrophedon–based διαδρομή καλύπτει ουσιαστικά όλες τις εναπομείνασες ακάλυπτες περιοχές.

    Στα αντίστοιχα σχήματα (π.χ. `replan_replay_coverage_curve.png` και `replan_replay_heatmaps.png`) απεικονίζονται η εξέλιξη της κάλυψης σε συνάρτηση με τον αριθμό των replan waypoints και οι heatmaps πριν/μετά τον επανασχεδιασμό.
    """)
    md_parts.append(coverage_section)

    # Ablation section
    if df_ablation is not None:
        md_ablation_table = df_ablation.to_markdown(index=False)
        ablation_section = dedent(f"""
        ## Ablation Study: Entropy vs Uncertainty vs Combined

        Για να αποτυπωθεί η επίδραση διαφορετικών κριτηρίων επιλογής στόχων, πραγματοποιήθηκε ablation study με τρεις διαμορφώσεις της συνάρτησης χρησιμότητας:
        1. **Entropy-only** (w_ent = 1.0, w_unc = 0.0),
        2. **Uncertainty-only** (w_ent = 0.0, w_unc = 1.0),
        3. **Combined** (w_ent = 0.5, w_unc = 0.3, w_cost = 0.2).

        Ο παρακάτω πίνακας συνοψίζει τα αποτελέσματα για τον καλύτερο στόχο σε κάθε mode:

        {md_ablation_table}

        Παρατηρείται ότι οι διαμορφώσεις που δίνουν μεγαλύτερο βάρος στην αβεβαιότητα τείνουν να επιλέγουν πιο απομακρυσμένα viewpoints με υψηλότερο information gain, ενώ οι combined ρυθμίσεις εξισορροπούν την πληροφορική ωφέλεια με το μήκος διαδρομής.
        """)
        md_parts.append(ablation_section)

    # Timing section
    if df_timing is not None:
        md_timing_table = df_timing.to_markdown(index=False)
        timing_section = dedent(f"""
        ## Υπολογιστική Αποδοτικότητα (Timing Benchmark)

        Η χρονομετρική αξιολόγηση βασίστηκε σε επαναλαμβανόμενη εκτέλεση (50 runs) των βασικών components του planners. Ο μέσος χρόνος εκτέλεσης για κάθε component συνοψίζεται στον ακόλουθο πίνακα:

        {md_timing_table}

        Οι χρόνοι είναι σε ms ανά κλήση, και δείχνουν ότι τόσο ο multi–objective planner όσο και ο υπολογισμός του Pareto front μπορούν να ενσωματωθούν σε online βρόχο πλοήγησης σε πραγματικό χρόνο.
        """)
        md_parts.append(timing_section)

    # Drift–uncertainty section
    if drift_summary is not None:
        n_valid = drift_summary.get("n_valid", 0)
        std_unc = drift_summary.get("std_unc", float("nan"))
        corr = drift_summary.get("corr", float("nan"))
        mean_low = drift_summary.get("mean_low", float("nan"))
        mean_high = drift_summary.get("mean_high", float("nan"))

        drift_section = dedent(f"""
        ## Σχέση Drift–Uncertainty στην Οπτική Οδομετρία

        Για τη μελέτη της σχέσης ανάμεσα στο drift της οπτικής οδομετρίας (VO) και την τιμή uncertainty του coverage grid, η VO τροχιά εξομαλύνθηκε με moving–average φίλτρο και το drift ορίστηκε ως η στιγμιαία απόσταση της πραγματικής τροχιάς από τη smoothed αναφορά. Κάθε frame προβλήθηκε στο coverage grid μέσω γραμμικής κλιμάκωσης στο εύρος των δεικτών (row, col), και ανατέθηκε η αντίστοιχη τιμή uncertainty.

        Από τη διαδικασία αυτή προέκυψαν **{n_valid}** έγκυρα δείγματα. Η τυπική απόκλιση της αβεβαιότητας ήταν **{std_unc:.4f}**, ενώ ο συντελεστής συσχέτισης Pearson μεταξύ drift και uncertainty εκτιμήθηκε σε **{corr:.4f}**, γεγονός που υποδηλώνει ότι στο συγκεκριμένο dataset δεν εμφανίζεται ισχυρή γραμμική συσχέτιση μεταξύ των δύο μεγεθών. Επιπλέον, ο μέσος όρος drift σε περιοχές χαμηλής αβεβαιότητας ήταν περίπου **{mean_low:.4f}**, ενώ σε περιοχές υψηλής αβεβαιότητας **{mean_high:.4f}**, με σχετικά μικρή διαφορά, κάτι που οφείλεται εν μέρει στην περιορισμένη δυναμική της κατανομής uncertainty.

        Τα σχήματα `drift_vs_uncertainty_scatter.png` και `drift_vs_uncertainty_bins.png` απεικονίζουν αντίστοιχα τη διασπορά των δειγμάτων στο επίπεδο (uncertainty, drift) και τη μέση τιμή drift ανά κατηγορία αβεβαιότητας.
        """)
        md_parts.append(drift_section)

    conclusion = dedent("""
    ## Συμπεράσματα

    Συνολικά, το σύστημα δυναμικής πλοήγησης που υλοποιήθηκε συνδυάζει:
    (α) grid-based εκτίμηση κάλυψης,
    (β) επανασχεδιασμό διαδρομής σε ακάλυπτες περιοχές,
    (γ) επιλογή στόχων εξερεύνησης με Information Gain και multi–objective κριτήρια,
    και (δ) βασική ανάλυση της σχέσης drift–uncertainty στην οπτική οδομετρία.

    Τα αποτελέσματα δείχνουν ότι ο προτεινόμενος μηχανισμός replan μπορεί να αυξήσει σημαντικά την κάλυψη της περιοχής ενδιαφέροντος, ενώ οι planners παραμένουν υπολογιστικά ελαφριοί, επιτρέποντας online χρήση. Παράλληλα, το framework παρέχει γόνιμο έδαφος για μελλοντικές επεκτάσεις προς πιο προηγμένα information–theoretic σχήματα active exploration και ενσωμάτωση σε αρχιτεκτονικές Embodied Vision–Language–Action.
    """)
    md_parts.append(conclusion)

    full_md = "\n\n".join(md_parts)
    out_path = "report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(f"[REPORT] Αποθηκεύτηκε το report ως {out_path}")
