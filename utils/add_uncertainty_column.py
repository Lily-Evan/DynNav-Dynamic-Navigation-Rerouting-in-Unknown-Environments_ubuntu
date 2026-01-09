import pandas as pd
import numpy as np

INPUT_CSV = "coverage_grid.csv"
OUTPUT_CSV = "coverage_grid_with_uncertainty.csv"

df = pd.read_csv(INPUT_CSV)

if "covered" not in df.columns:
    raise RuntimeError("Το coverage_grid.csv δεν έχει στήλη 'covered'")

# Βασικό μοντέλο:
# covered = 1 → uncertainty ~ 0.2
# covered = 0 → uncertainty ~ 0.8
uncertainty = np.where(df["covered"] >= 0.5, 0.2, 0.8)

df["uncertainty"] = uncertainty

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved new CSV with uncertainty → {OUTPUT_CSV}")
