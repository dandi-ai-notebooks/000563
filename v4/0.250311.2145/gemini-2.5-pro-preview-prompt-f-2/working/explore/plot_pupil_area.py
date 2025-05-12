# Objective: Plot a segment of pupil area over time.

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
h5_file = h5py.File(remote_file, 'r') # Added 'r' for read mode
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Added mode='r'
nwb = io.read()
print("NWB file loaded.")

# Access pupil tracking data
pupil_tracking = nwb.acquisition['EyeTracking'].pupil_tracking
pupil_area = pupil_tracking.area
pupil_timestamps = pupil_tracking.timestamps

# Select a segment of data (e.g., first 1000 points)
num_points_to_plot = 1000
if len(pupil_area) > num_points_to_plot:
    segment_pupil_area = pupil_area[:num_points_to_plot]
    segment_pupil_timestamps = pupil_timestamps[:num_points_to_plot]
    print(f"Plotting first {num_points_to_plot} points of pupil area.")
else:
    segment_pupil_area = pupil_area[:]
    segment_pupil_timestamps = pupil_timestamps[:]
    print(f"Plotting all {len(pupil_area)} points of pupil area (less than {num_points_to_plot}).")


# Create plot
plt.figure(figsize=(12, 6))
plt.plot(segment_pupil_timestamps, segment_pupil_area)
plt.xlabel("Time (s)")
plt.ylabel(f"Pupil Area ({pupil_tracking.unit})")
plt.title(f"Pupil Area Over Time (First {len(segment_pupil_area)} points)")
plt.grid(True)
plt.tight_layout()

# Save plot
output_path = "explore/pupil_area_segment.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

io.close()
print("Script finished.")