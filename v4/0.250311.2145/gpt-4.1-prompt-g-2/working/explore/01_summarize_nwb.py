# This script summarizes the top-level structure and metadata of the NWB file:
#   - Subject info, session info
#   - LFP shape and metadata
#   - Channel/electrode table summary
# The purpose is to guide further exploration and identify the structure of the data for the notebook.

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

with open("explore/01_metadata_summary.txt", "w") as f:
    # Session/subject info
    f.write(f"Session description: {nwb.session_description}\n")
    f.write(f"Session start time: {nwb.session_start_time}\n")
    f.write(f"Subject: {nwb.subject.specimen_name} ({nwb.subject.subject_id}), Sex: {nwb.subject.sex}, Genotype: {nwb.subject.genotype}, Strain: {nwb.subject.strain}\n")
    f.write(f"Institution: {getattr(nwb, 'institution', None)}\n")
    # Electrode groups
    f.write("\nElectrode groups:\n")
    egroups = nwb.electrode_groups
    for k in egroups:
        g = egroups[k]
        f.write(f"  {k}: {g.description}, probe_id: {getattr(g, 'probe_id', None)}, sampling rate: {getattr(g, 'lfp_sampling_rate', None)}\n")
    # Devices
    f.write("\nDevices:\n")
    devs = nwb.devices
    for k in devs:
        d = devs[k]
        f.write(f"  {k}: {d.description}, manufacturer: {d.manufacturer}, probe_id: {d.probe_id}, sampling_rate: {d.sampling_rate}\n")
    # LFP and acquisition data
    f.write("\nAcquisition:\n")
    aq = nwb.acquisition
    lfp_keys = [k for k in aq if 'lfp' in k]
    for k in lfp_keys:
        lfp = aq[k]
        f.write(f"  {k}: Type {type(lfp)}\n")
        if hasattr(lfp, 'electrical_series'):
            es = lfp.electrical_series
            for esk in es:
                e = es[esk]
                f.write(f"    ElectricalSeries: {esk}: data shape: {e.data.shape}, data dtype: {e.data.dtype}, unit: {e.unit}\n")
                f.write(f"      (# chans: {e.data.shape[1] if len(e.data.shape) > 1 else 1})\n")
                f.write(f"      timestamps shape: {e.timestamps.shape}, ts unit: {e.timestamps_unit}\n")
        if hasattr(lfp, 'data'):
            f.write(f"    data shape: {lfp.data.shape}, unit: {lfp.unit}\n")
    # Electrode table
    elec_tbl = nwb.electrodes
    df = elec_tbl.to_dataframe()
    f.write(f"\nElectrode table: {len(df)} channels\n")
    f.write(f"Columns: {list(df.columns)}\n")
    f.write(f"First 5 rows:\n{df.head().to_string()}\n")
    if 'location' in df.columns:
        f.write(f"Channels by location:\n{df['location'].value_counts().to_string()}\n")