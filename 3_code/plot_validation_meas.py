
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ----------------------- USER SETTINGS -----------------------
csv_path = "../1_data/i7/it_1/it_1_100_norm/3_validation/i7_unified_veh_u.csv"
x_cols = ["i7_8_Time", "i7_16_Time", "i7_31_Time", "i7_37_Time", "i7_43_Time", "i7_48_Time"]
y_cols = ["i7_8_veh_u", "i7_16_veh_u", "i7_31_veh_u", "i7_37_veh_u", "i7_43_veh_u", "i7_48_veh_u"]
save_path = "../4_other/plots/validation_meas.png"  # Use .png for image
figure_size = (10, 5)
line_width = 1
title = "Validation measurements"
measurements = ["Steer until a_y is reached (Snow mode ON)", "Steer until a_y is reached (Snow mode OFF)"
    , "Accelerate until a_x is reached", "Brake until a_x is reached (steering wheel ~= 0)"
    , "Brake until a_x is reached (steering wheel != 0)", "Sine wave like steering"]
# -------------------------------------------------------------

SEC_PATTERN = re.compiler('^\s*([0-9]*\.?[0-9]+)\s*sec\s*$', re.IGNORECASE)

def maybe_strip_sec(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series
    sample_matches = series.dropna().astype(str).map(lambda v: bool(SEC_PATTERN.match(v)))
    if not sample_matches.any():
        return series
    def convert(v):
        if pd.isna(v):
            return v
        m = SEC_PATTERN.match(str(v))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return v
        return v
    return series.map(convert)

def main():
    p = Path(csv_path)
    if not p.is_file():
        print(f"[ERROR] CSV file not found: {p}")
        return

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    # Check columns
    missing = [c for c in x_cols + y_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Clean and parse each x column
    for col in x_cols:
        df[col] = maybe_strip_sec(df[col])
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
            except Exception:
                pass

    # Plot
    plt.figure(figsize=figure_size)
    for i in range(len(x_cols)):
        plt.plot(df[x_cols[i]], df[y_cols[i]], label=str(measurements[i]), lw=line_width)

    plt.xlabel("Time [s]")
    plt.ylabel("Longitudinal speed [m/s]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

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
