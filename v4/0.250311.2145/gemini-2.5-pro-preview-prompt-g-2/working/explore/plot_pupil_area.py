# explore/plot_pupil_area.py
# This script loads a subset of pupil area data from the NWB file
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
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Add mode='r'
nwb = io.read()

# Access pupil tracking data
pupil_tracking = nwb.acquisition['EyeTracking'].spatial_series['pupil_tracking']
pupil_area_data = pupil_tracking.area
pupil_timestamps = pupil_tracking.timestamps

# Select a subset of data to plot (e.g., first 1000 points)
num_points_to_plot = 1000
subset_pupil_area = pupil_area_data[:num_points_to_plot]
subset_timestamps = pupil_timestamps[:num_points_to_plot]

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(subset_timestamps, subset_pupil_area)
plt.xlabel("Time (s)")
plt.ylabel(f"Pupil Area ({pupil_tracking.area.attrs.get('unit', 'unknown unit')})")
plt.title(f"Pupil Area over Time (First {num_points_to_plot} points)")
plt.grid(True)
plt.savefig("explore/pupil_area_plot.png")
plt.close()

print("Pupil area plot saved to explore/pupil_area_plot.png")

# Print some info about the data
print(f"Shape of pupil_area_data: {pupil_area_data.shape}")
print(f"Shape of pupil_timestamps: {pupil_timestamps.shape}")
if pupil_area_data.shape[0] > 0:
    print(f"First 5 pupil area values: {pupil_area_data[:5]}")
    print(f"First 5 timestamps: {pupil_timestamps[:5]}")
else:
    print("Pupil area data is empty.")

io.close()