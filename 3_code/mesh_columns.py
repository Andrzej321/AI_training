#!/usr/bin/env python3
"""
Extract selected columns from 6 (or fewer/more) CSV files and merge them side-by-side.

HOW TO USE:
1. Edit FILE_SPECS: each entry has:
      path    -> path to your CSV file
      columns -> list of column names to extract from that file
      prefix  -> (optional) string added in front of each extracted column name to avoid collisions
2. Set OUTPUT_PATH to where you want the merged CSV written.
3. Run: python mesh6.py
4. Result: a single CSV with all selected columns.

ASSUMPTIONS:
- All CSVs have headers.
- Rows are aligned purely by their order (row 0 of file A with row 0 of file B, etc.).
- If lengths differ, missing rows are filled with empty strings.
- Duplicate final column names (after prefixing) are auto-disambiguated by appending _2, _3, ...

If you need key-based joining later, ask and we can extend—kept simple for now.
"""

import csv
import os
from collections import defaultdict

# ========== CONFIG (EDIT THIS) ==========
base_loc = "../2_trained_models/best_models_lon/ai_models/"

other_estimators_loc = "../2_trained_models/other_estimators/results/lon/i7_"

meas_num = 48

FILE_SPECS = [
    {"path": base_loc + "RNN/results/i7_" + str(meas_num) + "_norm.csv", "columns": ["time", "veh_u", "model_RNN_lon_2020.pt"], "prefix": ""},
    {"path": base_loc + "LSTM/results/i7_" + str(meas_num) + "_norm.csv", "columns": ["model_LSTM_lon_2139.pt"], "prefix": ""},
    {"path": base_loc + "GRU/results/i7_" + str(meas_num) + "_norm.csv", "columns": ["model_GRU_lon_2080.pt"], "prefix": ""},
    {"path": base_loc + "TCN/results/i7_" + str(meas_num) + "_norm.csv", "columns": ["model_TCN_lon_129.pt"], "prefix": ""},
    {"path": base_loc + "Transformer/results/i7_" + str(meas_num) + "_norm.csv", "columns": ["model_T_lon_1560.pt"], "prefix": ""},
    {"path": other_estimators_loc + str(meas_num) + ".csv", "columns": ["best_wheel", "model_based", "ekf_m_bw", "veh_ref", "ekf_ai"], "prefix": ""},
]
OUTPUT_PATH = base_loc + "0_results_unified/results_unified_" + str(meas_num) + ".csv"
FILL_MISSING = ""  # value used for missing cells
# ========================================


def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_unique(headers):
    counts = defaultdict(int)
    result = []
    for h in headers:
        counts[h] += 1
        if counts[h] == 1:
            result.append(h)
        else:
            # Find first available suffix
            new_h = f"{h}_{counts[h]}"
            while new_h in counts:
                counts[h] += 1
                new_h = f"{h}_{counts[h]}"
            counts[new_h] = 1
            result.append(new_h)
    return result


def main():
    if not FILE_SPECS:
        print("Please populate FILE_SPECS before running.")
        return

    all_data = []          # list of (rows, requested_columns, output_names)
    all_output_names = []  # flattened list for header construction

    for spec in FILE_SPECS:
        path = spec["path"]
        cols = spec["columns"]
        prefix = spec.get("prefix", "")
        rows = read_csv(path)
        if not rows:
            raise ValueError(f"No data rows in file: {path}")
        header_sample = rows[0].keys()
        missing = [c for c in cols if c not in header_sample]
        if missing:
            raise KeyError(f'Columns {missing} not found in "{path}". Available: {list(header_sample)}')
        out_names = [prefix + c for c in cols]
        all_output_names.extend(out_names)
        all_data.append((rows, cols, out_names))

    # Deduplicate column names if needed
    unique_header = make_unique(all_output_names)

    # Map original out_name to final unique header
    mapping = {}
    idx = 0
    for _, _, out_names in all_data:
        for o in out_names:
            mapping[o] = unique_header[idx]
            idx += 1

    max_rows = max(len(r[0]) for r in all_data)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(unique_header)
        for row_index in range(max_rows):
            row_cells = []
            for rows, cols, out_names in all_data:
                if row_index < len(rows):
                    row_dict = rows[row_index]
                    for col in cols:
                        row_cells.append(row_dict.get(col, FILL_MISSING))
                else:
                    # Pad missing rows
                    row_cells.extend([FILL_MISSING] * len(cols))
            writer.writerow(row_cells)

    print(f"Merged CSV written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()