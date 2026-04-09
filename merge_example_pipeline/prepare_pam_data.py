import numpy as np
import pandas as pd
import torch
import pickle
import os
import json
from collections import Counter
from pamap_rus import load_pamap_data, preprocess_pamap_data, get_pamap_column_names

# ========== Configuration ==========
DATA_DIR = "/home/pham156/MoE/FuseMoE_poly/data/PAM_Protocol"
OUTPUT_DIR = "/home/pham156/MoE/FuseMoE_poly/data/PAM"
SUBJECT_IDS = list(range(1, 10))          # subjects 1 to 9
SEQ_LEN = 100                              # window length (T) – 600 as in paper
STEP = 50                                  # 50% overlap when SEQ_LEN=100
# Based on the feature EDA, use a cleaner protocol-only benchmark by:
# - removing activity 24, which is effectively tied to subject 9 only
# - removing orientation and temperature channels by default
# This keeps the core IMU channels plus heart rate.
SELECTED_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17]
FEATURE_SET = "no_orientation_no_temp"
# ====================================

def _summarize_subject(samples, subject_id, num_classes):
    class_counts = Counter([int(lbl) for _, lbl, _ in samples])
    total = len(samples)

    print(f"\n[SUBJECT {subject_id}]")
    print(f"  total windows: {total}")
    print("  class distribution:")
    for c in range(num_classes):
        cnt = class_counts.get(c, 0)
        pct = (100.0 * cnt / total) if total > 0 else 0.0
        print(f"    class {c}: {cnt} ({pct:.2f}%)")

    return {
        "total_windows": total,
        "class_distribution": {str(c): class_counts.get(c, 0) for c in range(num_classes)},
    }

def filter_sensor_columns(sensor_columns, feature_set):
    if feature_set == "full":
        return list(sensor_columns)
    if feature_set == "no_orientation":
        return [col for col in sensor_columns if "orient_" not in col]
    if feature_set == "no_orientation_no_temp":
        return [
            col for col in sensor_columns
            if "orient_" not in col and not col.startswith("temp_")
        ]
    raise ValueError(f"Unknown FEATURE_SET: {feature_set}")

def categorize_pamap_sensors(sensor_columns):
    """Categorize filtered sensor columns into chest, hand, ankle, heart_rate."""
    modality_sensors = {
        'chest': [],
        'hand': [],
        'ankle': [],
        'heart_rate': []
    }
    for col in sensor_columns:
        col_lower = col.lower()
        if 'heart' in col_lower or col_lower == 'heart_rate':
            modality_sensors['heart_rate'].append(col)
        elif 'chest' in col_lower:
            modality_sensors['chest'].append(col)
        elif 'hand' in col_lower:
            modality_sensors['hand'].append(col)
        elif 'ankle' in col_lower:
            modality_sensors['ankle'].append(col)
    return {k: v for k, v in modality_sensors.items() if v}

def create_windows(dataframe, modality_cols, seq_len, step, activity_map, subject_id):
    """Create sliding windows (non‑overlapping if step == seq_len)."""
    windows = []
    total_len = len(dataframe)
    for start in range(0, total_len - seq_len + 1, step):
        end = start + seq_len
        window_df = dataframe.iloc[start:end]
        # Most frequent activity in the window
        labels = window_df['activity_label'].values
        unique, counts = np.unique(labels, return_counts=True)
        label = unique[np.argmax(counts)]
        # For each modality, extract the sensor values
        mod_tensors = []
        for mod, cols in modality_cols.items():
            existing = [c for c in cols if c in window_df.columns]
            if existing:
                values = window_df[existing].values.astype(np.float32)
                mod_tensors.append(torch.tensor(values))   # shape (seq_len, n_features)
            else:
                # Should not happen; fill with zeros
                mod_tensors.append(torch.zeros(seq_len, 1))
        windows.append((mod_tensors, label, subject_id))
    return windows

def main():
    # Load first subject to get all sensor columns
    df0 = load_pamap_data(SUBJECT_IDS[0], DATA_DIR)
    df0_proc, all_sensors = preprocess_pamap_data(df0)
    filtered_sensors = filter_sensor_columns(all_sensors, FEATURE_SET)

    # Determine activity mapping for selected activities only
    unique_acts = sorted([a for a in df0_proc['activity_id'].unique() if a in SELECTED_ACTIVITIES])
    activity_map = {act: i for i, act in enumerate(unique_acts)}
    print(f"Selected activities: {unique_acts}")
    print(f"Mapping: {activity_map}")

    # Group filtered sensors into modalities.
    modality_cols = categorize_pamap_sensors(filtered_sensors)
    print(f"Modalities: {list(modality_cols.keys())}")
    for mod, cols in modality_cols.items():
        print(f"  {mod}: {len(cols)} sensors")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subject_meta = {}

    # Process and save each subject independently so splits can be formed at training time.
    for subj in SUBJECT_IDS:
        print(f"\nProcessing subject {subj}...")
        subject_path = os.path.join(OUTPUT_DIR, f"subject_{subj}.pkl")
        df = load_pamap_data(subj, DATA_DIR)
        df_proc, _ = preprocess_pamap_data(df)

        # Keep only rows with selected activities
        df_proc = df_proc[df_proc['activity_id'].isin(SELECTED_ACTIVITIES)].copy()
        if len(df_proc) == 0:
            print(f"  Subject {subj} has no selected activities, skipping.")
            with open(subject_path, "wb") as f:
                pickle.dump([], f)
            subject_meta[str(subj)] = {
                "total_windows": 0,
                "class_distribution": {str(c): 0 for c in range(len(activity_map))},
                "status": "no_selected_activities",
            }
            continue

        # Map activity IDs
        df_proc['activity_label'] = df_proc['activity_id'].map(activity_map)
        df_proc = df_proc.dropna(subset=['activity_label']).copy()
        df_proc['activity_label'] = df_proc['activity_label'].astype(int)

        if len(df_proc) < SEQ_LEN:
            print(f"  Subject {subj} too short after filtering, skipping.")
            with open(subject_path, "wb") as f:
                pickle.dump([], f)
            subject_meta[str(subj)] = {
                "total_windows": 0,
                "class_distribution": {str(c): 0 for c in range(len(activity_map))},
                "status": "too_short",
            }
            continue

        windows = create_windows(df_proc, modality_cols, SEQ_LEN, STEP, activity_map, subj)
        subject_data = [(m, l) for m, l, _ in windows]
        with open(subject_path, "wb") as f:
            pickle.dump(subject_data, f)
        print(f"  Saved {len(subject_data)} windows to {subject_path}")
        subject_meta[str(subj)] = _summarize_subject(windows, subj, len(activity_map))

    # Save diagnostic metadata for reproducibility and later split assembly.
    metadata = {
        "config": {
            "data_dir": DATA_DIR,
            "output_dir": OUTPUT_DIR,
            "subject_ids": SUBJECT_IDS,
            "seq_len": SEQ_LEN,
            "step": STEP,
            "feature_set": FEATURE_SET,
            "selected_activities": SELECTED_ACTIVITIES,
            "activity_map": {str(k): int(v) for k, v in activity_map.items()},
            "modality_dims": {mod: len(cols) for mod, cols in modality_cols.items()},
        },
        "subjects": subject_meta,
    }

    with open(os.path.join(OUTPUT_DIR, "subject_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Subject files saved to {OUTPUT_DIR}")
    print(f"Subject diagnostics saved to {os.path.join(OUTPUT_DIR, 'subject_metadata.json')}")

if __name__ == "__main__":
    main()
