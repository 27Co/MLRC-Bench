# evaluate a method on eval/test dataset

### main() calls evaluate_method()
### evaluate_method() calls method.train() and method.predict()
### method.train() trains the model and saves checkpoint file
### method.predict() loads checkpoint file and saves predictions

from pathlib import Path

def evaluate_method(method, phase: str = "dev", base_path: str = "."):
    """Evaluate a method.
    Args:
        method: method to evaluate
        phase: Either "dev" or "test" phase
    Effects:
        saves predictions to "submission.csv"
    """
    dev_hdf5_file = f"{base_path}/data/Sherlock1/derivatives/serialised/sub-0_ses-11_task-Sherlock1_run-2_proc-bads+headpos+sss+notch+bp+ds_meg.h5"
    test_hdf5_file = f"{base_path}/data/Sherlock1/derivatives/serialised/sub-0_ses-12_task-Sherlock1_run-2_proc-bads+headpos+sss+notch+bp+ds_meg.h5"
    print(f"Starting evaluation for {method.get_name()} in {phase} phase")

    method.train()
    if phase == "dev":
        hdf5_file = dev_hdf5_file
    elif phase == "test":
        hdf5_file = test_hdf5_file
    elif phase == "debug":
        pass
    else:
        raise ValueError(f"Invalid phase: {phase}")

    if Path(hdf5_file).is_file():
        method.predict(hdf5_file, f"{base_path}/output/submission.csv")
    else:
        raise FileNotFoundError(f"H5 file not found: {hdf5_file}")

from sklearn.metrics import f1_score
import pandas as pd

def get_score(method, phase: str = "dev", base_path: str = "."):
    """Calculate F1-macro score for a method.
    Args:
        method: method to evaluate
        phase: Either "dev" or "test" phase
    Returns:
        F1-macro score
    """
    data_path = f"{base_path}/data"
    dev_tsv_file = f"{data_path}/Sherlock1/derivatives/events/sub-0_ses-11_task-Sherlock1_run-2_events.tsv"
    dev_hdf5_file = f"{data_path}/Sherlock1/derivatives/serialised/sub-0_ses-11_task-Sherlock1_run-2_proc-bads+headpos+sss+notch+bp+ds_meg.h5"

    test_tsv_file = f"{data_path}/Sherlock1/derivatives/events/sub-0_ses-12_task-Sherlock1_run-2_events.tsv"
    test_hdf5_file = f"{data_path}/Sherlock1/derivatives/serialised/sub-0_ses-12_task-Sherlock1_run-2_proc-bads+headpos+sss+notch+bp+ds_meg.h5"

    # Set filenames based on phase
    if phase == "dev":
        tsv_file = dev_tsv_file
        hdf5_file = dev_hdf5_file
        answer_file = f"{base_path}/output/dev_answer.csv"
    elif phase == "test":
        tsv_file = test_tsv_file
        hdf5_file = test_hdf5_file
        answer_file = f"{base_path}/output/test_answer.csv"
    else:
        raise ValueError(f"Invalid phase: {phase}")

    # check tsv and hdf5 files exist
    if not Path(tsv_file).is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")
    if not Path(hdf5_file).is_file():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

    # Generate the ground truth labels
    generate_speech_labels(tsv_file, hdf5_file, output_csv=answer_file)

    # Load the generated files
    print(f"Calculating F1-macro score for {phase} phase...")

    # method.predict() save predictions there
    try:
        df_pred = pd.read_csv(f"{base_path}/output/submission.csv")
        df_true = pd.read_csv(answer_file)
    except FileNotFoundError as e:
        print(f"Error: Missing files for scoring. {e}")
        return None

    # Alignment and Binarization
    # convert probabilities to binary labels (threshold 0.5)
    y_true = df_true['label'].values
    y_prob = df_pred['speech_prob'].values
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate F1-Macro Score
    score = f1_score(y_true, y_pred, average='macro')

    print(f"--- Evaluation Results ---")
    print(f"F1-Macro Score: {score:.4f}")
    print(f"--------------------------")

    return score

import numpy as np
import h5py
def generate_speech_labels(tsv_file_path, hdf5_file_path, output_csv):
    # create directory for output_csv if it doesn't exist
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    # 1. Load MEG metadata to determine total length
    print("loading HDF5 to get timepoints...")
    with h5py.File(hdf5_file_path, 'r') as f:
        n_samples = f['data'].shape[1] 
        times = f['times'][:]
    
    dt = times[1] - times[0]
    sfreq = 1.0 / dt  # Sampling frequency (expected 250 Hz)
    print(f"Total timepoints to label: {n_samples}")
    print(f"Start time: {times[0]:.2f}s, End time: {times[-1]:.2f}s, Sampling Frequency: {sfreq}Hz")
    # print first ten timepoints
    print(f"First ten timepoints: {times[:10]}")

    # 2. Initialize ground truth array with zeros (Silence = 0)
    y_true = np.zeros(n_samples, dtype=int)

    # 3. Load TSV and filter for speech events ('word' or 'phoneme')
    print("loading TSV and filtering for speech events...")
    tsv_data = pd.read_csv(tsv_file_path, sep='\t')
    speech_events = tsv_data[tsv_data['kind'].isin(['word', 'phoneme'])]

    # 4. Map TSV onsets/durations to discrete sample indices
    for _, row in speech_events.iterrows():
        # Convert seconds to sample indices using sfreq (250Hz)
        start_idx = int(round(row['timemeg'] * sfreq))
        end_idx = int(round((row['timemeg'] + row['duration']) * sfreq))
        
        # Boundary protection
        start_idx = max(0, start_idx)
        end_idx = min(n_samples, end_idx)
        
        # Fill range with 1 (Speech)
        y_true[start_idx:end_idx] = 1

    # 5. Save as CSV with an index column
    df_output = pd.DataFrame({
        'idx': np.arange(n_samples),
        'label': y_true
    })
    
    df_output.to_csv(output_csv, index=False)
    print(f"Success! Saved {len(y_true)} labels to {output_csv}")



