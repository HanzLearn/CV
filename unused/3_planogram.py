import pandas as pd
import os
import re

# Directories
excel_dir = os.path.join("received_data", "EXCEL")
planogram_dir = os.path.join("received_data", "Planogram")

# Ensure planogram output directory exists
os.makedirs(planogram_dir, exist_ok=True)

# Pattern to match detection Excel files
pattern = re.compile(r"detection_results_colored_(\d+)\.xlsx")

# Columns to keep
columns_needed = ["frame_id", "rack_id", "track_id", "class_id", "class_name", "cx"]

# Scan for Excel files
for filename in os.listdir(excel_dir):
    match = pattern.match(filename)
    if not match:
        continue

    index = match.group(1)
    xlsx_path = os.path.join(excel_dir, filename)
    csv_name = f"planogram_{index}.csv"
    csv_path = os.path.join(planogram_dir, csv_name)

    # Skip if corresponding CSV already exists
    if os.path.exists(csv_path):
        print(f"Skipping {filename}: {csv_name} already exists.")
        continue

    # Load all sheets
    print(f"Processing {filename}...")
    sheets_dict = pd.read_excel(xlsx_path, sheet_name=None)
    df_list = []

    for sheet_df in sheets_dict.values():
        df_filtered = sheet_df[columns_needed]
        df_list.append(df_filtered)

    df = pd.concat(df_list, ignore_index=True)

    # Prepare valid entries
    valid_entries = []
    for track_id in df["track_id"].unique():
        print(f"Track ID : {track_id}")
        filtered_rows = df[df["track_id"] == track_id]
        if len(filtered_rows) >= 30:
            first_row = filtered_rows.iloc[0]
            valid_entries.append({
                "track_id": track_id,
                "rack_id": first_row["rack_id"],
                "class_id": first_row["class_id"],
                "class_name": first_row["class_name"]
            })

    planogram_df = pd.DataFrame(valid_entries)

    # Insert blank row between header and first item, and between rack changes
    rows_with_blanks = []

    # Add a blank row after header
    rows_with_blanks.append({col: "" for col in planogram_df.columns})

    last_rack = None
    for _, row in planogram_df.iterrows():
        current_rack = row['rack_id']
        if last_rack is not None and current_rack != last_rack:
            rows_with_blanks.append({col: "" for col in planogram_df.columns})
        rows_with_blanks.append(row)
        last_rack = current_rack

    final_df = pd.DataFrame(rows_with_blanks)
    final_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
