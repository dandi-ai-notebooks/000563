# Objective: Plot a segment of running speed over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
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

# Select a segment of data (e.g., first 1000 points)
num_points_to_plot = 1000
if len(running_speed_data) > num_points_to_plot:
    segment_running_speed = running_speed_data[:num_points_to_plot]
    segment_timestamps = running_speed_timestamps[:num_points_to_plot]
    print(f"Plotting first {num_points_to_plot} points of running speed.")
else:
    segment_running_speed = running_speed_data[:]
    segment_timestamps = running_speed_timestamps[:]
    print(f"Plotting all {len(running_speed_data)} points of running speed (less than {num_points_to_plot}).")

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(segment_timestamps, segment_running_speed)
plt.xlabel("Time (s)")
plt.ylabel(f"Running Speed ({running_speed_ts.unit})")
plt.title(f"Running Speed Over Time (First {len(segment_running_speed)} points)")
plt.grid(True)
plt.tight_layout()

# Save plot
output_path = "explore/running_speed_segment.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

io.close()
print("Script finished.")