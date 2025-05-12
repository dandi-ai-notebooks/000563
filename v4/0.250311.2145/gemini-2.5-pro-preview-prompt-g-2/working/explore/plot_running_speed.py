# explore/plot_running_speed.py
# This script loads a subset of running speed data from the NWB file
# and plots it to visualize changes over a short time period.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access running speed data
running_speed_ts = nwb.processing['running']['running_speed']
running_speed_data = running_speed_ts.data
running_speed_timestamps = running_speed_ts.timestamps

# Select a subset of data to plot (e.g., first 2000 points)
num_points_to_plot = 2000
subset_running_speed = running_speed_data[:num_points_to_plot]
subset_timestamps = running_speed_timestamps[:num_points_to_plot]

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(subset_timestamps, subset_running_speed)
plt.xlabel("Time (s)")
plt.ylabel(f"Running Speed ({running_speed_ts.unit})")
plt.title(f"Running Speed over Time (First {num_points_to_plot} points)")
plt.grid(True)
plt.savefig("explore/running_speed_plot.png")
plt.close()

print("Running speed plot saved to explore/running_speed_plot.png")

# Print some info about the data
print(f"Shape of running_speed_data: {running_speed_data.shape}")
print(f"Shape of running_speed_timestamps: {running_speed_timestamps.shape}")
if running_speed_data.shape[0] > 0:
    print(f"First 5 running_speed values: {running_speed_data[:5]}")
    print(f"First 5 timestamps: {running_speed_timestamps[:5]}")
else:
    print("Running speed data is empty.")

io.close()