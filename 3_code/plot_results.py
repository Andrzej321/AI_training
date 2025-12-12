#!/usr/bin/env python3
"""
Minimal plotting script.

Edit the variables under USER SETTINGS and run:
    python 3_code/plot_results_simple.py

It will:
- Load the CSV
- Plot the selected y columns against the x column
- Try to parse x as datetime (if it looks like datetimes) but otherwise leaves it as-is
- Save a PNG or show interactively (set save_path)

If something is wrong (missing columns), it prints an error and exits.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ----------------------- USER SETTINGS (EDIT THESE) -----------------------
test_file = 8
csv_path = f"../2_trained_models/best_models_lon/ai_models/0_results_unified/results_unified_{test_file}.csv"   # Path to the CSV file
x_col    = "time"                 # Column to use for x-axis
y_cols   = ["veh_u_gps", "model_TCN_lon_129.pt", "veh_ref", "ekf_ai", "best_wheel"]  # Columns to plot as y lines
save_path = f"../4_other/plots/onepager_{test_file}.png"                 # Set to None to show interactively
figure_size = (10, 5)                   # Width, height in inches
line_width = 1
# --------------------------------------------------------------------------

if test_file == 8:
    title = "Steer until a_y is reached (snow mode on)"
elif test_file == 16:
    title = "Steer until a_y is reached (snow mode off)"
elif test_file == 31:
    title = "Accelerate until a_x is reached"
elif test_file == 37:
    title = "Brake until a_x is reached (steering wheel == 0)"
elif test_file == 43:
    title = "Brake until a_x is reached (steering wheel != 0)"
else:
    title = "Sine wave like steering"


SEC_PATTERN = re.compile(r'^\s*([0-9]*\.?[0-9]+)\s*sec\s*$', re.IGNORECASE)

def maybe_strip_sec(series: pd.Series) -> pd.Series:
    """
    If a pandas Series of dtype object contains entries like '12.5sec' or '300 sec',
    strip the 'sec' part and convert to float.
    Only modifies rows that match the pattern; others are returned unchanged.
    """
    if series.dtype != object:
        return series  # Not strings; nothing to do

    # Check if at least one value matches the pattern
    sample_matches = series.dropna().astype(str).map(lambda v: bool(SEC_PATTERN.match(v)))
    if not sample_matches.any():
        return series  # No 'sec'-style entries

    def convert(v):
        if pd.isna(v):
            return v
        m = SEC_PATTERN.match(str(v))
        if m:
            num_str = m.group(1)
            try:
                return float(num_str)
            except ValueError:
                return v  # fallback: leave original
        return v  # not matching pattern

    cleaned = series.map(convert)
    return cleaned


def main():
    # Validate file
    p = Path(csv_path)
    if not p.is_file():
        print(f"[ERROR] CSV file not found: {p}")
        return

    # Load CSV
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    # Check columns
    missing = [c for c in [x_col] + y_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Clean x column if it has 'sec' suffix values
    df[x_col] = maybe_strip_sec(df[x_col])

    # Attempt datetime parse only if not converted to float already and still object
    if df[x_col].dtype == object:
        try:
            df[x_col] = pd.to_datetime(df[x_col], errors="raise")
        except Exception:
            # Leave as original strings if not datetime
            pass

    # Plot
    plt.figure(figsize=figure_size)
    for col in y_cols:
        plt.plot(df[x_col], df[col], label=col, lw=line_width)



    plt.xlabel("Time [s]")
    plt.ylabel("Longitudinal speed [m/s]")
    plt.title(title)
    if len(y_cols) > 1:
        plt.legend()
    plt.tight_layout()

    plt.grid(True)

    if save_path:
        try:
            plt.savefig(save_path, dpi=500)
            print(f"[INFO] Saved plot to {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save plot: {e}")
    else:
        plt.show()


if __name__ == "__main__":
    main()