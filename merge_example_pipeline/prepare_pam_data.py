import numpy as np
import pandas as pd
import torch
import pickle
import os
from pamap_rus import load_pamap_data, preprocess_pamap_data, get_pamap_column_names

# ========== Configuration ==========
DATA_DIR = "/home/pham156/MoE/FuseMoE_poly/data/PAM_Protocol"
OUTPUT_DIR = "/home/pham156/MoE/FuseMoE_poly/data/PAM"
SUBJECT_IDS = list(range(1, 10))          # subjects 1 to 9
SEQ_LEN = 100                              # window length (T) – 600 as in paper
STEP = 50                                 # no overlap (to match 5,333 total segments)
# Subject split (MERGE paper: train 1-6, val 7, test 8-9)
TRAIN_SUBJECTS = [1,2,3,4,5,6]
VAL_SUBJECTS   = [7]
TEST_SUBJECTS  = [8,9]

# Standard 8 activities used in many PAMAP2 benchmarks
SELECTED_ACTIVITIES = [1,2,3,17,16,12,13,4,7,6,5,24]
# ====================================

def categorize_pamap_sensors(sensor_columns):
    """Categorize all sensor columns into chest, hand, ankle, heart_rate,
    including orientation (quaternion) and temperature."""
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
    # Remove any empty modality (should not happen)
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

    # Determine activity mapping for selected activities only
    unique_acts = sorted([a for a in df0_proc['activity_id'].unique() if a in SELECTED_ACTIVITIES])
    activity_map = {act: i for i, act in enumerate(unique_acts)}
    print(f"Selected activities: {unique_acts}")
    print(f"Mapping: {activity_map}")

    # Group all sensors into four modalities (including orientation)
    modality_cols = categorize_pamap_sensors(all_sensors)
    print(f"Modalities: {list(modality_cols.keys())}")
    for mod, cols in modality_cols.items():
        print(f"  {mod}: {len(cols)} sensors")

    # Collect windows per subject
    all_windows = []   # list of (modality_tensors, label, subject_id)
    for subj in SUBJECT_IDS:
        print(f"\nProcessing subject {subj}...")
        df = load_pamap_data(subj, DATA_DIR)
        df_proc, _ = preprocess_pamap_data(df)

        # Keep only rows with selected activities
        df_proc = df_proc[df_proc['activity_id'].isin(SELECTED_ACTIVITIES)].copy()
        if len(df_proc) == 0:
            print(f"  Subject {subj} has no selected activities, skipping.")
            continue

        # Map activity IDs
        df_proc['activity_label'] = df_proc['activity_id'].map(activity_map)
        df_proc = df_proc.dropna(subset=['activity_label']).copy()
        df_proc['activity_label'] = df_proc['activity_label'].astype(int)

        if len(df_proc) < SEQ_LEN:
            print(f"  Subject {subj} too short after filtering, skipping.")
            continue

        windows = create_windows(df_proc, modality_cols, SEQ_LEN, STEP, activity_map, subj)
        all_windows.extend(windows)
        print(f"  Added {len(windows)} windows. Total now: {len(all_windows)}")

    # Split by subject
    train_samples = [w for w in all_windows if w[2] in TRAIN_SUBJECTS]
    val_samples   = [w for w in all_windows if w[2] in VAL_SUBJECTS]
    test_samples  = [w for w in all_windows if w[2] in TEST_SUBJECTS]

    print(f"\nSamples per split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Remove subject ID from stored tuples (only keep modality_tensors, label)
    train_data = [(m, l) for m, l, _ in train_samples]
    val_data   = [(m, l) for m, l, _ in val_samples]
    test_data  = [(m, l) for m, l, _ in test_samples]

    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "train_all_subjects.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(OUTPUT_DIR, "val_all_subjects.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    with open(os.path.join(OUTPUT_DIR, "test_all_subjects.pkl"), "wb") as f:
        pickle.dump(test_data, f)

    print(f"Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()