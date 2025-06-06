import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import gc

# ------------------ Configuration ------------------ #
WINDOW_SIZE = 5
BATCH_SIZE = 4
HEIGHT, WIDTH = 2761, 5455

# ------------------ Custom Loss ------------------ #
def masked_mse(y_true, y_pred):
    """
    Loss function that expands the target dims to match predictions.
    y_true shape: (batch, time, height, width, 2) where the last dimension
    contains [target, mask]. y_pred shape: (batch, time, height, width, 1).
    """
    y = tf.expand_dims(y_true[..., 0], axis=-1)
    mask = tf.expand_dims(y_true[..., 1], axis=-1)
    squared_error = tf.square(y - y_pred)
    return tf.reduce_sum(squared_error * mask) / (tf.reduce_sum(mask) + 1e-8)

def masked_mape(y_true, y_pred):
    """
    Computes the masked Mean Absolute Percentage Error.

    y_true shape: (batch, time, height, width, 2) where the last dimension
    contains [target, mask]. y_pred shape: (batch, time, height, width, 1).
    """
    # Expand dims for target and mask
    y = tf.expand_dims(y_true[..., 0], axis=-1)
    mask = tf.expand_dims(y_true[..., 1], axis=-1)

    # Calculate absolute percentage error, adding a small epsilon to avoid division by zero
    ape = tf.abs((y - y_pred) / (y + 1e-8))

    # Return the masked MAPE (multiplied by 100 for percentage)
    return tf.reduce_sum(ape * mask) / (tf.reduce_sum(mask) + 1e-8) * 100

def masked_mae(y_true, y_pred):
    """
    Mask-aware MAE loss.

    y_true shape: (batch, time, height, width, 2) where the last dimension contains [target, mask].
    y_pred shape: (batch, time, height, width, 1).
    """
    # Expand dims so that y_true[..., 0] becomes (batch, time, height, width, 1)
    y = tf.expand_dims(y_true[..., 0], axis=-1)
    mask = tf.expand_dims(y_true[..., 1], axis=-1)
    absolute_error = tf.abs(y - y_pred)
    return tf.reduce_sum(absolute_error * mask) / (tf.reduce_sum(mask) + 1e-8)


def load_custom_model(model_path):
    custom_objects = {'masked_mae': masked_mae, 'masked_mse': masked_mse, 'masked_mape':masked_mape}
    return load_model(model_path, custom_objects=custom_objects)


# ------------------ Data Pipeline ------------------ #
class DataProcessor:
    def __init__(self, x_path, y_path):
        print(f"Initializing data processor:\nX: {x_path}\nY: {y_path}")
        self.x_memmap = np.load(x_path, mmap_mode='r')
        self.y_memmap = np.load(y_path, mmap_mode='r')
        self.num_images = self.x_memmap.shape[0]
        print(f"Loaded data with {self.num_images} images")

    def create_dataset(self, shuffle=False):
        print("Creating sliding window dataset...")
        num_windows = self.num_images - WINDOW_SIZE + 1
        indices = np.arange(num_windows)
        ds = tf.data.Dataset.from_tensor_slices(indices)
        if shuffle:
            ds = ds.shuffle(num_windows, reshuffle_each_iteration=True)
            print("Enabled shuffling")

        def process_window(idx):
            x = tf.numpy_function(self._get_window, [idx, False], tf.float32)
            # Only use the first 6 bands (channels) from input data
            x = x[..., :5]
            y = tf.numpy_function(self._get_window, [idx, True], tf.float32)
            # Create mask: nonzero in first 6 channels
            mask = tf.cast(tf.reduce_any(x[..., :5] != 0.0, axis=-1, keepdims=True), tf.float32)
            y = tf.where(tf.math.is_nan(y), 0.0, y)
            y_combined = tf.concat([y, mask], axis=-1)  # last dim holds target and mask
            return x, y_combined

        return (ds.map(process_window, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(BATCH_SIZE)
                  .prefetch(tf.data.AUTOTUNE))

    def _get_window(self, idx, is_y):
        arr = self.y_memmap if is_y else self.x_memmap
        return arr[idx:idx+WINDOW_SIZE]


# ------------------ Evaluation Function ------------------ #
def predict_and_evaluate(processor, model, save_path):
    print(f"\nStarting prediction for {processor.num_images} images...")
    # Initialize prediction array and counts for averaging
    full_pred = np.full((processor.num_images, HEIGHT, WIDTH, 1), np.nan)
    counts = np.zeros((processor.num_images, HEIGHT, WIDTH, 1))

    num_windows = processor.num_images - WINDOW_SIZE + 1

    for batch_idx, (x_batch, y_combined_batch) in enumerate(test_ds):
        print(f"Processing batch {batch_idx+1}...")
        preds = model.predict(x_batch, verbose=0)

        # Loop over each example in the batch
        for i in range(preds.shape[0]):
            window_idx = batch_idx * BATCH_SIZE + i
            if window_idx >= num_windows:
                break

            start = window_idx
            end = start + WINDOW_SIZE
            pred_window = preds[i]

            # Sum overlapping predictions
            full_pred[start:end] = np.where(
                np.isnan(full_pred[start:end]),
                pred_window,
                full_pred[start:end] + pred_window
            )
            counts[start:end] += 1

    print("Averaging overlapping predictions...")
    full_pred /= np.where(counts > 0, counts, 1)
    #Inverse Transform Apply only if needed
    full_pred = np.expm1(full_pred)

    print("Calculating metrics on original scale...")
    y_true_full = processor.y_memmap
    valid_mask = np.any(processor.x_memmap[..., :5] != 0.0, axis=-1, keepdims=True)
   
    y_true_valid = y_true_full[valid_mask]
    pred_valid = full_pred[valid_mask]
   
    mse = np.nanmean((y_true_valid - pred_valid) ** 2)
    rmse = np.sqrt(mse)
   
    # Compute R2 score: R2 = 1 - (SSE/SST)
    y_mean = np.nanmean(y_true_valid)
    sst = np.nansum((y_true_valid - y_mean) ** 2)
    sse = np.nansum((y_true_valid - pred_valid) ** 2)
    r2 = 1 - sse/sst if sst != 0 else np.nan
   
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
   
    print(f"Saving predictions to {save_path}...")
    full_pred[~valid_mask] = np.nan
    np.save(save_path, full_pred)
    return full_pred


# ------------------ Main Execution ------------------ #
if __name__ == "__main__":
    # Initialize components
    model = load_custom_model("/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/MODEL-CODES/best_model_logtransformed30epmae.h5")
    test_processor = DataProcessor(
        "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/conv_lstm_inputs_fixed_rangenext11_testsub2.npy",
        "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/conv_lstm_targets_fixed_rangenext11_testsub2.npy"
    )
    test_ds = test_processor.create_dataset()
    # Run evaluation
    print("Starting memory-constrained evaluation...")
    test_pred = predict_and_evaluate(test_processor, model, '/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/splits_latlon_0.01/test_predictions_logtransmaeep30.npy')
    print("\nProcess completed successfully!")

    # Cleanup
    del model
    tf.keras.backend.clear_session()
    gc.collect()