# Preview the structure and metadata of the selected NWB file for Dandiset 000563.
# This script extracts key metadata, acquisition LFP data shape, and electrode table summary.

import pynwb
import h5py
import remfile
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("Session description:", nwb.session_description)
print("Identifier:", nwb.identifier)
print("Session start time:", nwb.session_start_time)
print("Institution:", getattr(nwb, 'institution', 'N/A'))
print("Stimulus notes:", getattr(nwb, 'stimulus_notes', 'N/A'))
print("Subject ID:", nwb.subject.subject_id if nwb.subject else "N/A")
print("Subject:", {k: getattr(nwb.subject, k, 'N/A') for k in ['age', 'sex', 'species', 'strain', 'genotype']})

probe_0_lfp = nwb.acquisition["probe_0_lfp"]
electrical_series = probe_0_lfp.electrical_series
probe_0_lfp_data = electrical_series["probe_0_lfp_data"]

print("LFP data shape (samples, channels):", probe_0_lfp_data.data.shape)
print("LFP data dtype:", probe_0_lfp_data.data.dtype)
print("LFP unit:", probe_0_lfp_data.unit)
print("LFP timestamps shape:", probe_0_lfp_data.timestamps.shape)

print("\nElectrodes table description:", probe_0_lfp_data.electrodes.table.description)
df_electrodes = probe_0_lfp_data.electrodes.table.to_dataframe()
print("Electrodes table columns:", df_electrodes.columns.tolist())
print("First 5 rows of electrodes table:")
print(df_electrodes.head())