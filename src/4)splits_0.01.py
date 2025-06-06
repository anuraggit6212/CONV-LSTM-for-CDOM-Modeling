import numpy as np
import pandas as pd
import os
import psutil
import time
from concurrent.futures import ProcessPoolExecutor

# --------------------------- CONFIG --------------------------- #
input_npy_path = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/conv_lstm_inputs_fixed_range_all.npy"    # e.g., shape: (787, H, W, 15)
target_npy_path ="/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/conv_lstm_targets_fixed_range_all.npy"   # e.g., shape: (787, H, W, 1)
timestamps_csv_path = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/timestamps_0.01_withlatlon.csv"
output_dir = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/"
os.makedirs(output_dir, exist_ok=True)

# Memory usage threshold (in percent)
MEM_THRESHOLD = 80.0

# --------------------------- MEMORY MONITOR --------------------------- #
def check_memory(threshold=MEM_THRESHOLD):
    mem = psutil.virtual_memory()
    while mem.percent > threshold:
        print(f"Memory usage high: {mem.percent:.1f}%, sleeping for 5 sec")
        time.sleep(5)
        mem = psutil.virtual_memory()

# --------------------------- LOAD TIMESTAMPS & SPLIT INDICES --------------------------- #
timestamps_df = pd.read_csv(timestamps_csv_path)
timestamps_df["timestamp"] = pd.to_datetime(timestamps_df["timestamp"])
timestamps_df["year_month"] = timestamps_df["timestamp"].dt.to_period("M")
print("Timestamps per month:")
print(timestamps_df["year_month"].value_counts().sort_index())

# We assume that the rows in timestamps_df are in the same order as the samples in the npy arrays.
train_indices = []
val_indices = []
test_indices = []

for ym, group in timestamps_df.groupby("year_month"):
    indices = group.index.to_numpy()  # indices correspond to the sample order in npy files
    n = len(indices)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_indices.extend(indices[:train_end])
    val_indices.extend(indices[train_end:val_end])
    test_indices.extend(indices[val_end:])
    print(f"{ym}: Total = {n}, Train = {len(indices[:train_end])}, Val = {len(indices[train_end:val_end])}, Test = {len(indices[val_end:])}")

# Ensure indices are sorted to preserve temporal order
train_indices = np.sort(train_indices)
val_indices = np.sort(val_indices)
test_indices = np.sort(test_indices)

# --------------------------- FUNCTION TO PROCESS & SAVE A SPLIT --------------------------- #
def process_and_save(split_name, indices, input_npy_path, target_npy_path, output_dir):
    # Check memory usage before processing this split.
    check_memory()
    print(f"Processing '{split_name}' split with {len(indices)} samples...")
   
    # Load the memmapped arrays (in read-only mode) within this process.
    inputs = np.load(input_npy_path, mmap_mode='r')
    targets = np.load(target_npy_path, mmap_mode='r')
   
    # Slice the arrays using the computed indices.
    X_split = inputs[indices]
    y_split = targets[indices]
   
    # Save the splits in the output directory.
    np.save(os.path.join(output_dir, f"X_{split_name}.npy"), X_split)
    np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y_split)
   
    print(f"Finished processing '{split_name}': X shape {X_split.shape}, y shape {y_split.shape}")
    return X_split.shape, y_split.shape

# --------------------------- MAIN EXECUTION --------------------------- #
def main():
    # Confirm total number of samples without loading full data into memory.
    inputs = np.load(input_npy_path, mmap_mode='r')
    total_samples = inputs.shape[0]
    print("Total samples:", total_samples)
   
    # Use ProcessPoolExecutor to process splits concurrently.
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        futures.append(executor.submit(process_and_save, "train", train_indices, input_npy_path, target_npy_path, output_dir))
        futures.append(executor.submit(process_and_save, "val", val_indices, input_npy_path, target_npy_path, output_dir))
        futures.append(executor.submit(process_and_save, "test", test_indices, input_npy_path, target_npy_path, output_dir))
       
        for future in futures:
            split_shape = future.result()
            print("Saved split with shapes:", split_shape)
   
    print("Data splitting complete. Train/Val/Test arrays saved to:", output_dir)

if __name__ == '__main__':
    main()