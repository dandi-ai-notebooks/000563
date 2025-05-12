# Objective: Plot a smoothed segment of running speed over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # For rolling average
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()
print("NWB file loaded.")

# Access running speed data
running_speed_ts = nwb.processing['running']['running_speed']
running_speed_data = running_speed_ts.data
running_speed_timestamps = running_speed_ts.timestamps

# Select a segment of data (e.g., first 5000 points for better smoothing context)
num_points_to_plot = 5000
if len(running_speed_data) > num_points_to_plot:
    segment_running_speed_raw = running_speed_data[:num_points_to_plot]
    segment_timestamps = running_speed_timestamps[:num_points_to_plot]
    print(f"Using first {num_points_to_plot} points of running speed for smoothing.")
else:
    segment_running_speed_raw = running_speed_data[:]
    segment_timestamps = running_speed_timestamps[:]
    print(f"Using all {len(running_speed_data)} points of running speed for smoothing.")

# Smooth the data using a moving average
window_size = 50
# Ensure segment_running_speed_raw is a NumPy array for pd.Series
segment_running_speed_series = pd.Series(np.array(segment_running_speed_raw))
smoothed_running_speed = segment_running_speed_series.rolling(window=window_size, center=True).mean().to_numpy()
print(f"Smoothed running speed with window size {window_size}.")

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(segment_timestamps, segment_running_speed_raw, label='Raw Data', alpha=0.5)
plt.plot(segment_timestamps, smoothed_running_speed, label=f'Smoothed (window={window_size})', color='red')
plt.xlabel("Time (s)")
plt.ylabel(f"Running Speed ({running_speed_ts.unit})")
plt.title(f"Smoothed Running Speed Over Time (First {len(segment_running_speed_raw)} points)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
output_path = "explore/running_speed_smoothed_segment.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

io.close()
print("Script finished.")