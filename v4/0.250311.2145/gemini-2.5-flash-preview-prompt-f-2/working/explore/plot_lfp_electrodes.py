# This script loads a small chunk of LFP data and electrode positions from an NWB file
# and generates time series and scatter plots, saving them as PNG files.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load
url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data and timestamps
lfp_data = nwb.acquisition["probe_0_lfp_data"].data
timestamps = nwb.acquisition["probe_0_lfp_data"].timestamps

# Get electrodes table
electrodes_table = nwb.electrodes.to_dataframe()

# Select a small segment of LFP data (first 1000 time points) for a single electrode
num_time_points = 1000
electrode_index_to_plot = 0
lfp_segment = lfp_data[0:num_time_points, electrode_index_to_plot]

# Plot LFP data for the selected electrode
plt.figure(figsize=(12, 6))
plt.plot(lfp_segment)

plt.xlabel("Time (samples)")
plt.ylabel("LFP (volts)")
plt.title(f"LFP data segment for electrode index {electrode_index_to_plot}")
plt.tight_layout()
plt.savefig('explore/lfp_time_series.png')
plt.close()

# Plot electrode positions (vertical vs horizontal position)
plt.figure(figsize=(8, 8))
plt.scatter(electrodes_table['probe_horizontal_position'], electrodes_table['probe_vertical_position'])
plt.xlabel("Horizontal position (microns)")
plt.ylabel("Vertical position (microns)")
plt.title("Electrode positions on the probe")
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/electrode_positions.png')
plt.close()

# Close the NWB file
io.close()