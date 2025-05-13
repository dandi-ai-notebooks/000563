# This script loads and plots a small segment of LFP from 5 example channels.
# It now uses a fixed sample rate (625 Hz) from metadata and samples the first 5 seconds accordingly.

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

lfp = nwb.acquisition["probe_0_lfp"]
elec_series = lfp.electrical_series["probe_0_lfp_data"]

n_channels = elec_series.data.shape[1]
channels_to_plot = np.linspace(0, n_channels - 1, 5, dtype=int)

n_seconds = 5
sample_rate = 625
n_samples = n_seconds * sample_rate
t = np.arange(n_samples) / sample_rate

print(f"data shape: {elec_series.data.shape}, n_samples: {n_samples}")

data = np.array(elec_series.data[:n_samples, channels_to_plot])

plt.figure(figsize=(10, 6))
for i, ch in enumerate(channels_to_plot):
    plt.plot(t, data[:, i] * 1e3 + i * 2, label=f"Ch {ch}")  # Offset for clarity
plt.xlabel("Time (s)")
plt.ylabel("LFP (mV, offset for visibility)")
plt.title("Example LFP traces (first 5 seconds, 5 channels)")
plt.legend()
plt.tight_layout()
plt.savefig("explore/02_lfp_example.png")