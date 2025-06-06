import numpy as np
import os
import math
from concurrent.futures import ProcessPoolExecutor

def process_slice(inputs_slice, targets_slice, input_min, input_max, target_min, target_max):
    """
    Process a slice of the data.
    For inputs (first 6 channels), a pixel is valid only if all values are within [input_min, input_max].
    For targets (single channel), a pixel is valid only if the value is within [target_min, target_max].
    If either check fails for a pixel, all values for that pixel are set to 0.0.
    """
    # Create valid masks for the current slice.
    valid_inputs_mask = np.all((inputs_slice[..., :6] >= input_min) & (inputs_slice[..., :6] <= input_max), axis=-1)
    valid_targets_mask = (targets_slice[..., 0] >= target_min) & (targets_slice[..., 0] <= target_max)
    final_valid_mask = valid_inputs_mask & valid_targets_mask  # shape: (chunk, H, W)
   
    # Work on copies so that we don't modify the original slices.
    inputs_processed = inputs_slice.copy()
    targets_processed = targets_slice.copy()
   
    # Set pixels failing the check to 0.0 in both inputs and targets.
    inputs_processed[~final_valid_mask] = 0.0
    targets_processed[~final_valid_mask, 0] = np.nan
   
    return inputs_processed, targets_processed

def main():
    # --------------------------- CONFIG --------------------------- #
    input_npy_path = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/conv_lstm_inputs_withlatlon_0.01.npy"  # Expected shape: (N, H, W, 8)
    target_npy_path = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/conv_lstm_targets_withlatlon_0.01.npy"   # Expected shape: (N, H, W, 1)
    output_dir = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------- LOAD DATA --------------------------- #
    inputs = np.load(input_npy_path)    # shape: (N, H, W, 8)
    targets = np.load(target_npy_path)  # shape: (N, H, W, 1)
    print("Original inputs shape:", inputs.shape)
    print("Original targets shape:", targets.shape)
   
    # --------------------------- DEFINE FIXED RANGES --------------------------- #
    # For inputs, consider only the first 6 channels.
    input_min, input_max = 0.001, 1
    # For targets (single channel), valid range is:
    target_min, target_max = 0.01, 5

    # --------------------------- SPLIT DATA FOR PARALLEL PROCESSING --------------------------- #
    N = inputs.shape[0]
    num_workers = os.cpu_count() or 1
    chunk_size = math.ceil(N / num_workers)
   
    # Determine slices along the first dimension.
    slices = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, N)
        if start < end:
            slices.append((start, end))
   
    # --------------------------- PROCESS DATA IN PARALLEL --------------------------- #
    processed_inputs_list = []
    processed_targets_list = []
   
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for start, end in slices:
            # Extract slices.
            inputs_slice = inputs[start:end]
            targets_slice = targets[start:end]
            futures.append(executor.submit(
                process_slice,
                inputs_slice,
                targets_slice,
                input_min,
                input_max,
                target_min,
                target_max
            ))
       
        # Gather results from each process.
        for future in futures:
            inputs_chunk, targets_chunk = future.result()
            processed_inputs_list.append(inputs_chunk)
            processed_targets_list.append(targets_chunk)
   
    # --------------------------- COMBINE RESULTS --------------------------- #
    inputs_processed = np.concatenate(processed_inputs_list, axis=0)
    targets_processed = np.concatenate(processed_targets_list, axis=0)

    # --------------------------- SAVE THE RESULTS --------------------------- #
    np.save(os.path.join(output_dir, "conv_lstm_inputs_fixed_range_all.npy"), inputs_processed)
    np.save(os.path.join(output_dir, "conv_lstm_targets_fixed_range_all.npy"), targets_processed)
    print("Data with fixed ranges and synchronized validity (invalid values set to 0.0) saved to", output_dir)

if __name__ == '__main__':
    main()