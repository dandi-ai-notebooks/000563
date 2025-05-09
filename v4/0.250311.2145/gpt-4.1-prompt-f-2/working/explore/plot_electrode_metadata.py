# Visualize electrode metadata from the NWB electrode table: distribution of locations and vertical positions

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

probe_0_lfp = nwb.acquisition["probe_0_lfp"]
electrical_series = probe_0_lfp.electrical_series
probe_0_lfp_data = electrical_series["probe_0_lfp_data"]
df_electrodes = probe_0_lfp_data.electrodes.table.to_dataframe()

# Plot 1: Count of electrodes per location
plt.figure(figsize=(7,4))
df_electrodes['location'].value_counts().plot(kind='bar')
plt.title('Electrode count by location')
plt.xlabel('Location')
plt.ylabel('Number of electrodes')
plt.tight_layout()
plt.savefig("explore/electrode_locations.png")
plt.close()

# Plot 2: Vertical position along the probe
plt.figure(figsize=(7,4))
plt.scatter(df_electrodes.index, df_electrodes['probe_vertical_position'])
plt.title('Vertical position vs. electrode index')
plt.xlabel('Electrode index')
plt.ylabel('Vertical position (microns)')
plt.tight_layout()
plt.savefig("explore/electrode_vertical_position.png")
plt.close()