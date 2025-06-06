import netCDF4 as nc
import numpy as np
import os
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt
from dask import delayed, compute
import psutil
import time
import gc

# --------------------------- CONFIG --------------------------- #
data_dir = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/EOS-06-OCM-3_LSTM_Input/"    # Folder containing your 786 NetCDF files
output_dir = "/sachome1/usr/sahay/Anurag_Mahalpure_Workspace/LSTM_Input_Model-1/" # Folder where processed data will be saved
os.makedirs(output_dir, exist_ok=True)

# Define the 6 input bands and 1 target band
input_band_names = ['RRS01', 'RRS02', 'RRS03', 'RRS04', 'RRS05',
                    'RRS06']
target_band_name = 'CDOM'


# Desired grid resolution in degrees (high resolution)
res = 0.01

# Memory usage threshold (in percent)
MEM_THRESHOLD = 80.0

def check_memory(threshold=MEM_THRESHOLD):
    mem = psutil.virtual_memory()
    while mem.percent > threshold:
        print(f"Memory usage high ({mem.percent:.1f}%). Waiting for memory to free up...")
        time.sleep(5)
        mem = psutil.virtual_memory()

# ------------------ STEP 1: EXTRACT TIMESTAMPS AND SORT FILES ------------------ #
def extract_timestamp(file_path):
    """
    Extracts the full timestamp (date+time) from a NetCDF file.
    Assumes the file contains a 'time' variable.
    """
    ds = nc.Dataset(file_path)
    date = ds.getncattr('DateOfPass')
    raw_time = ds.getncattr('Scene_End_Time')
    time=':'.join(raw_time.split(':')[:3])
    timestamp = f"{date} {time}"
    ds.close()
    return pd.to_datetime(timestamp,dayfirst=True)

def get_sorted_files():
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')]
    print(f"Total files found: {len(all_files)}")
    files_with_time = [(f, extract_timestamp(f)) for f in all_files]
    files_with_time.sort(key=lambda x: x[1])
    return [item[0] for item in files_with_time]

def compute_global_grid(sorted_files):
    grid_files = sorted_files[:200]  # Use first 100 files for grid computation
    lat_min, lat_max = float('inf'), float('-inf')
    lon_min, lon_max = float('inf'), float('-inf')
    for file in grid_files:
        ds = nc.Dataset(file)
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        lat_min = min(lat_min, lat.min())
        lat_max = max(lat_max, lat.max())
        lon_min = min(lon_min, lon.min())
        lon_max = max(lon_max, lon.max())
        ds.close()
    #new_lat = np.arange(lat_min, lat_max, res)
    new_lat = np.arange(lat_max, lat_min, -res)
    new_lon = np.arange(lon_min, lon_max, res)
    return new_lat, new_lon

def regrid_band(ds, band_var, new_lat, new_lon, lat_var="latitude", lon_var="longitude"):
    # Convert lat and lon to float32 for memory efficiency
    lat = ds.variables[lat_var][:].astype(np.float32)
    lon = ds.variables[lon_var][:].astype(np.float32)
    # If the original latitude is descending, reverse to get ascending order for interpolation
    #if lat[0] > lat[-1]:
    #lat = lat[::-1]
    #data = ds.variables[band_var][:][::-1, :]
    #else:
    data = ds.variables[band_var][:]
    orig_lon_mesh, orig_lat_mesh = np.meshgrid(lon, lat)
    points = np.column_stack((orig_lat_mesh.flatten(), orig_lon_mesh.flatten())).astype(np.float32)
    values = data.flatten()
    new_lon_mesh, new_lat_mesh = np.meshgrid(new_lon, new_lat)
    mapped = griddata(points, values, (new_lat_mesh, new_lon_mesh),
                        method='nearest', fill_value=np.nan)
    valid_mask = griddata(points, np.ones_like(values), (new_lat_mesh, new_lon_mesh),
                          method='nearest', fill_value=np.nan)
    new_data = np.full((len(new_lat), len(new_lon)), np.nan, dtype=np.float32)
    new_data[~np.isnan(valid_mask)] = mapped[~np.isnan(valid_mask)]
    # No vertical flip is needed because new_lat is already in descending order.
    return new_data

def regrid_input_and_target(file_path, new_lat, new_lon, lat_var="latitude", lon_var="longitude"):
    """
    Regrids all 13 input bands and the target variable for a given file.
    Then, appends the geographic coordinate grids as extra channels.
    Returns:
      input_image: NumPy array of shape (new_height, new_width, 15)
                   where 15 = 13 spectral bands + 1 latitude + 1 longitude.
      target_image: NumPy array of shape (new_height, new_width, 1)
    """
    ds = nc.Dataset(file_path)
    input_bands = []
    for band in input_band_names:
        band_regrid = regrid_band(ds, band, new_lat, new_lon, lat_var, lon_var)
        input_bands.append(band_regrid)
    input_image = np.stack(input_bands, axis=-1)
    target_image = regrid_band(ds, target_band_name, new_lat, new_lon, lat_var, lon_var)
    target_image = target_image[..., np.newaxis]
    ds.close()
   
    # Create geographic coordinate channels from new_lat and new_lon:
    new_lat_grid, new_lon_grid = np.meshgrid(new_lat, new_lon, indexing="ij")
    lat_channel = new_lat_grid[..., np.newaxis].astype(np.float32)
    lon_channel = new_lon_grid[..., np.newaxis].astype(np.float32)
   
    # Concatenate the lat and lon channels to the input_image along the last axis.
    # Final shape becomes: (H, W, 13 + 1 + 1) = (H, W, 15)
    input_image = np.concatenate([input_image, lat_channel, lon_channel], axis=-1)
   
    return input_image, target_image

@delayed
def process_file(file_path, new_lat, new_lon):
    check_memory()
    inp_img, tgt_img = regrid_input_and_target(file_path, new_lat, new_lon)
    ts = extract_timestamp(file_path)
    return inp_img, tgt_img, ts

if __name__ == '__main__':
    sorted_files = get_sorted_files()
    new_lat, new_lon = compute_global_grid(sorted_files)
    new_height, new_width = len(new_lat), len(new_lon)
    print("Common grid shape:", new_height, "x", new_width)
    
    # Save the global grid coordinates to one CSV file with two columns (lat and lon)
    new_lat_grid, new_lon_grid = np.meshgrid(new_lat, new_lon, indexing="ij")
    df_coords = pd.DataFrame({
        "latitude": new_lat_grid.flatten(),
        "longitude": new_lon_grid.flatten()
    })
    df_coords.to_csv(os.path.join(output_dir, "global_grid_0.01.csv"), index=False)
    print("Global grid coordinates saved to global_grid.csv.")
   
    total_files = len(sorted_files)
    chunk_size = 250  # Adjust chunk size as needed
    all_inputs = []
    all_targets = []
    all_timestamps = []
   
    for i in range(0, total_files, chunk_size):
        chunk_files = sorted_files[i:i+chunk_size]
        delayed_tasks = [process_file(f, new_lat, new_lon) for f in chunk_files]
        # Use the processes scheduler (ProcessPoolExecutor)
        results = compute(*delayed_tasks, scheduler="processes")
        for j, (inp_img, tgt_img, ts) in enumerate(results):
            all_inputs.append(inp_img)
            all_targets.append(tgt_img)
            all_timestamps.append(ts)
        print(f"Processed files {i+1} to {i+len(chunk_files)} / {total_files}")
        gc.collect()
   
    final_inputs = np.array(all_inputs)   # Shape: (num_files, new_height, new_width, 15)
    final_targets = np.array(all_targets)  # Shape: (num_files, new_height, new_width, 1)
    print("Final Data bank inputs shape:", final_inputs.shape)
    print("Final Data bank targets shape:", final_targets.shape)
   
    np.save(os.path.join(output_dir, "conv_lstm_inputs_withlatlon_0.01.npy"), final_inputs)
    np.save(os.path.join(output_dir, "conv_lstm_targets_withlatlon_0.01.npy"), final_targets)
    timestamps_df = pd.DataFrame({
        "file": sorted_files,
        "timestamp": all_timestamps
    })
    timestamps_df.to_csv(os.path.join(output_dir, "timestamps_0.01_withlatlon.csv"), index=False)
    print(f"Processed data and timestamps saved to {output_dir}")