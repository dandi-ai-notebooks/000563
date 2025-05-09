# Plot LFP traces from the first 5 seconds for the first 5 LFP channels.
# This will help visualize the structure and temporal dynamics of the data.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

probe_0_lfp = nwb.acquisition["probe_0_lfp"]
electrical_series = probe_0_lfp.electrical_series
probe_0_lfp_data = electrical_series["probe_0_lfp_data"]

# Sampling rate for probe 0 LFP is 625 Hz
fs = 625
n_channels = 5
# Plot first 5 seconds
n_samples = fs * 5

# Read a subset only for plotting efficiency
lfp = probe_0_lfp_data.data[:n_samples, :n_channels]
timestamps = probe_0_lfp_data.timestamps[:n_samples]

plt.figure(figsize=(10, 6))
offset = 200e-6  # Offset for visualization (0.2 mV)
for ch in range(n_channels):
    plt.plot(timestamps, lfp[:, ch] + ch * offset, label=f'Channel {ch}')
plt.xlabel('Time (s)')
plt.ylabel('LFP signal + offset (V)')
plt.title('LFP (first 5 s, first 5 channels)')
plt.legend()
plt.tight_layout()
plt.savefig("explore/lfp_short_segment.png")