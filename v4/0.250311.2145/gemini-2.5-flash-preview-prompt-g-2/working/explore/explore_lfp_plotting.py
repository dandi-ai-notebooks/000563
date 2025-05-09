# This script loads a subset of LFP data and plots it.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access LFP data and timestamps
probe_0_lfp_data = nwb.acquisition["probe_0_lfp_data"]
lfp_data = probe_0_lfp_data.data
timestamps = probe_0_lfp_data.timestamps

# Load a subset of data (e.g., first 10000 time points for the first 5 channels)
num_timepoints = 10000
num_channels = 5
lfp_subset = lfp_data[0:num_timepoints, 0:num_channels]
timestamps_subset = timestamps[0:num_timepoints]

# Load electrode locations for the selected channels
electrodes_table = nwb.electrodes.to_dataframe()
electrode_locations = electrodes_table.iloc[0:num_channels]['location'].tolist()

# Plot the LFP data subset
plt.figure(figsize=(12, 6))
for i in range(num_channels):
    plt.plot(timestamps_subset, lfp_subset[:, i] + i * 500, label=f'Channel {i+1} ({electrode_locations[i]})') # Offset channels for visibility
plt.xlabel("Time (seconds)")
plt.ylabel("LFP (volts)")
plt.title(f"Subset of LFP Data (First {num_channels} Channels)")
plt.legend()
plt.savefig('explore/lfp_subset_plot.png')
plt.close()

print("LFP subset plot generated: explore/lfp_subset_plot.png")