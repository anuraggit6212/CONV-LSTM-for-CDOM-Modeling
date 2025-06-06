import netCDF4 as nc
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import psutil
import pandas as pd

# --------------------- CONFIGURATION --------------------- #
# Input npy file (shape: (N, H, W, 1))
target_npy_path = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/fixed_range/conv_lstm_targets_fixed_rangenext11_testsub2.npy"
# Global grid CSV file; expected to have columns "latitude" and "longitude"
global_grid_csv_path = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/global_grid_0.01.csv"
# Output directory where NetCDF files will be saved.
output_dir = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM-OUTPUT-MODEL-1/Final_true/"

os.makedirs(output_dir, exist_ok=True)

# --------------------- LOAD DATA --------------------- #
# Load targets; expected shape: (N, H, W, 1)
targets = np.load(target_npy_path)
print("Targets shape:", targets.shape)

# Determine spatial dimensions from the first sample.
H, W, _ = targets[0].shape

# Load the global grid CSV file using pandas.
grid_df = pd.read_csv(global_grid_csv_path)
# Reshape the latitude and longitude arrays to the grid shape.
lat = grid_df["latitude"].values.reshape(H, W)
lon = grid_df["longitude"].values.reshape(H, W)

# --------------------- DEFINE SAVE FUNCTION --------------------- #
def save_nc_file_separate_bands(tgt, file_path, lat, lon):
    """
    Saves a single time step's target data into one NetCDF file.
    The file will contain:
      - Dimensions: time (singleton), y, and x.
      - Variables: time (a simple index), 'CDOM' for the target data,
        'latitude' and 'longitude' as grid coordinates.
    """
    # Print memory usage before saving
    proc = psutil.Process(os.getpid())
    mem_usage_mb = proc.memory_info().rss / (1024**2)
    print(f"Saving file {file_path}; current memory usage: {mem_usage_mb:.2f} MB")
   
    # Create the NetCDF file using NETCDF4 format.
    ds = nc.Dataset(file_path, "w", format="NETCDF4")
   
    # Create dimensions: time (singleton), y, and x.
    ds.createDimension("time", 1)
    ds.createDimension("y", H)
    ds.createDimension("x", W)
   
    # Create the time coordinate variable (using a simple index).
    time_nc = ds.createVariable("time", "f8", ("time",))
    time_nc[:] = [0]
   
    # Create the target variable named "CDOM".
    target_var = ds.createVariable("CDOM", "f4", ("time", "y", "x"))
    target_var[0, :, :] = tgt[:, :, 0]
   
    # Add latitude and longitude variables as common grid coordinates.
    lat_var = ds.createVariable("latitude", "f4", ("y", "x"))
    lon_var = ds.createVariable("longitude", "f4", ("y", "x"))
    lat_var[:, :] = lat
    lon_var[:, :] = lon
   
    ds.close()
    print(f"Saved file to: {file_path}")
    return file_path

# --------------------- SAVE FILES CONCURRENTLY --------------------- #
if __name__ == '__main__':
    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
   
    # Use ProcessPoolExecutor to save files concurrently.
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, sample in enumerate(targets):
            # Construct file name with a numeric prefix (1-indexed) and a base name.
            output_nc_file = os.path.join(output_dir, f"{i+1}_maelogtransep30.nc")
            futures.append(executor.submit(save_nc_file_separate_bands, sample, output_nc_file, lat, lon))
       
        # Wait for all files to be saved.
        for future in futures:
            result = future.result()
            print("Saved file:", result)