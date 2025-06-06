import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
import time
from tensorflow.keras.callbacks import ModelCheckpoint



# ------------------ Configuration ------------------ #
WINDOW_SIZE = 5
SLIDE_STEP = 1
BATCH_SIZE = 2
HEIGHT, WIDTH = 2761, 5455
PRINT_EVERY = 100


print("Configuration loaded:")
print(f"Window size: {WINDOW_SIZE}, Slide step: {SLIDE_STEP}, Batch size: {BATCH_SIZE}")

# ------------------ Model Definition ------------------ #
def create_sequence_model():
    print("Creating model architecture...")
    inputs = Input(shape=(WINDOW_SIZE, HEIGHT, WIDTH, 5))
   
    # Encoder: Process each time step independently
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
   
    # Temporal processing via ConvLSTM2D layer
    x = ConvLSTM2D(128, (3, 3), padding='same', return_sequences=True)(x)
    x = TimeDistributed(BatchNormalization())(x)
   
    # Decoder: Reconstruct output
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    outputs = TimeDistributed(Conv2D(1, (3, 3), padding='same', activation='linear'))(x)
   
    model = Model(inputs, outputs)
    print("Model created successfully")
    return model
    
    
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

def masked_mse(y_true, y_pred):
    """
    Loss function that expands the target dims to match predictions.
    y_true shape: (batch, time, height, width, 2) where the last dimension contains [target (log-transformed), mask].
    y_pred shape: (batch, time, height, width, 1).
    """
    # Expand dims so that y_true[..., 0] becomes (batch, time, height, width, 1)
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
    


checkpoint_h5 = ModelCheckpoint(
    'best_model_alldatalgt25-03.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    save_weights_only=False
)


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
            # Get input and target windows
            x = tf.numpy_function(self._get_window, [idx, False], tf.float32)
            x = x[..., :5]
            y = tf.numpy_function(self._get_window, [idx, True], tf.float32)
           
            # Create mask based on validity of first 6 channels in inputs
            mask = tf.cast(tf.reduce_any(x != 0.0, axis=-1, keepdims=True), tf.float32)
           
            # Replace NaNs in target with 0.0
            y = tf.where(tf.math.is_nan(y), 0.0, y)
            # Apply log1p transformation to positive target values (leaving non-positive as 0)
            y_log = tf.where(y > 0, tf.math.log1p(y), y)
           
            # Combine log-transformed target with mask
            y_combined = tf.concat([y_log, mask], axis=-1)  # Last dim holds [log-target, mask]
            return x, y_combined
       
        return (ds.map(process_window, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
   
    def _get_window(self, idx, is_y):
        arr = self.y_memmap if is_y else self.x_memmap
        return arr[idx:idx+WINDOW_SIZE]

# ------------------ Initialize Data Processors ------------------ #
print("\nInitializing data processors:")
train_processor = DataProcessor(
    "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/X_train.npy",
    "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/y_train.npy"
)
val_processor = DataProcessor(
    "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/X_val.npy",
    "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/y_val.npy"
)
test_processor = DataProcessor(
    "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/X_test.npy",
    "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/y_test.npy"
)

# ------------------ Create Datasets ------------------ #
print("\nCreating datasets:")
train_ds = train_processor.create_dataset(shuffle=True)
val_ds = val_processor.create_dataset()
test_ds = test_processor.create_dataset()

# ------------------ Model Setup ------------------ #
print("\nInitializing model:")

model = create_sequence_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=masked_mae,
    metrics=[masked_mse, masked_mape])
                  

model.summary()

# ------------------ Training ------------------ #
print("\nStarting training:")
start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=1,
    callbacks=[checkpoint_h5]
)
print(f"Training completed in {(time.time()-start_time)/60:.2f} minutes")

# ------------------ Save Final Model ------------------ #
print("\nSaving model...")
model.save('final_modelalldatalgt25-03.h5')
print("Model saved to final_model.h5")