# %% [markdown]
# # Exploring Dandiset 000563: Allen Institute Openscope - Barcoding
#
# **Warning:** This notebook was AI-generated and has not been fully verified. Please interpret the code and results with caution before drawing conclusions or basing further work on this analysis.
#
# This notebook introduces and explores data from [Dandiset 000563, version 0.250311.2145](https://dandiarchive.org/dandiset/000563/0.250311.2145), made available by the Allen Institute's OpenScope project. This dataset includes large-scale, multisite neural recordings using Neuropixels probes in mouse neocortex, focusing on temporally precise neural responses to visual stimuli ("barcoding").
#
# ## Notebook Overview
# - **Dandiset/project metadata**: Context and summary
# - **How to access**: Description and code to load the Dandiset via the DANDI API
# - **NWB file structure**: Exploring acquisition, LFP data, and electrode information
# - **Visualizations**: Example LFP trace, anatomical mapping of electrodes, and probe geometry
# - **Next steps**: Comments on potential further analyses

# %% [markdown]
# ## About the Dandiset and Project
#
# - **Title:** Allen Institute Openscope - Barcoding
# - **DOI/Citation:** Reinagel, Pamela; Lecoq, Jérôme; Durand, Séverine; et al. (2025) Allen Institute Openscope - Barcoding (Version 0.250311.2145) [Data set]. DANDI Archive. https://doi.org/10.48324/dandi.000563/0.250311.2145
# - **Keywords:** mouse, neuropixel, extracellular electrophysiology, neocortex, barcoding, temporal precision
# - **Description:**  
#   This experiment used the OpenScope Neuropixels passive viewing protocol, with visually modulated white noise and sinusoidal grating stimuli. It aimed to discover whether barcode-like population neural response patterns are present throughout the mouse brain, enabling cell type discrimination.
#
# **Dandiset link:**  
# https://dandiarchive.org/dandiset/000563/0.250311.2145

# %% [markdown]
# ## What this notebook covers
#
# This notebook demonstrates how to:
# 1. Access DANDI datasets programmatically.
# 2. Explore the structure and metadata of an NWB file from this Dandiset.
# 3. Visualize example time series data (LFP) and map electrodes to anatomical locations and probe geometry.
# 4. Understand the types of information and analysis possible with this dataset.
#
# **Note:** The code assumes that you have the following packages pre-installed in your environment:
# 
# - `dandi`, `pynwb`, `h5py`, `remfile`, `numpy`, `pandas`, `matplotlib`

# %% [markdown]
# ## Loading the Dandiset via the DANDI API
#
# Below we show how to connect to the DANDI archive, retrieve Dandiset metadata, and list available assets.
# **You do not need to download the entire dataset; NWB files can be streamed remotely.**

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000563", "0.250311.2145")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ---
# ## Accessing and Exploring a Sample NWB File
#
# For illustration, we will examine one extracellular electrophysiology NWB file:  
# **sub-681446/sub-681446_ses-1290510496_probe-0_ecephys.nwb**
#
# **Direct asset URL:**  
# https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/
#
# *To explore this file interactively in a browser, try [Neurosift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/&dandisetId=000563&dandisetVersion=draft).*

# %% [markdown]
# ### Loading the NWB file (streamed from DANDI)
#
# NWB files can be loaded remotely using PyNWB and remfile for efficient streaming.  
# *This avoids the need to download large files locally.*  
# The following code demonstrates how to load the selected NWB file and introspect its top-level structure and session/subject metadata.

# %%
import pynwb
import h5py
import remfile
import pandas as pd

nwb_url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(nwb_url)
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

# %% [markdown]
# ---
# ### NWB File Structure and Electrode Metadata
#
# The main LFP data for this probe is in the file as `acquisition['probe_0_lfp']['electrical_series']['probe_0_lfp_data']`.
#
# **Shape of LFP data:** Rows = time points, Columns = LFP channels  
# The electrode metadata table (shown below) maps channels to location, device, spatial coordinates, and other relevant metadata.
#
# Let's display the shape, units, and a preview of the electrode DataFrame:

# %%
probe_0_lfp = nwb.acquisition["probe_0_lfp"]
electrical_series = probe_0_lfp.electrical_series
probe_0_lfp_data = electrical_series["probe_0_lfp_data"]
print("LFP data shape (samples, channels):", probe_0_lfp_data.data.shape)
print("LFP data dtype:", probe_0_lfp_data.data.dtype)
print("LFP unit:", probe_0_lfp_data.unit)
print("LFP timestamps shape:", probe_0_lfp_data.timestamps.shape)

df_electrodes = probe_0_lfp_data.electrodes.table.to_dataframe()
print("Electrodes table columns:", df_electrodes.columns.tolist())
print("First 5 rows of electrodes table:")
display(df_electrodes.head())

# %% [markdown]
# ---
# ## Visualization 1: Example LFP Traces
#
# Below, we plot the first 5 seconds of LFP from the first 5 channels. Channel signals are offset for clarity.  
# *(Acquisition is at 625 Hz; 5 seconds × 625 Hz = 3125 timepoints.)*

# %%
import numpy as np
import matplotlib.pyplot as plt

fs = 625
n_channels = 5
n_samples = fs * 5

lfp = probe_0_lfp_data.data[:n_samples, :n_channels]
timestamps = probe_0_lfp_data.timestamps[:n_samples]

plt.figure(figsize=(10, 6))
offset = 200e-6  # 0.2 mV offset for clarity
for ch in range(n_channels):
    plt.plot(timestamps, lfp[:, ch] + ch * offset, label=f'Channel {ch}')
plt.xlabel('Time (s)')
plt.ylabel('LFP signal + offset (V)')
plt.title('LFP (first 5 s, first 5 channels)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Visualization 2: Electrode Count by Location
#
# This bar plot illustrates the distribution of electrodes across identified anatomical locations.

# %%
df_electrodes['location'].value_counts().plot(kind='bar', figsize=(7,4))
plt.title('Electrode count by location')
plt.xlabel('Location')
plt.ylabel('Number of electrodes')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Visualization 3: Electrode Vertical Position Along the Probe
#
# Each point represents an electrode, showing its position along the vertical axis of the Neuropixels probe.

# %%
plt.figure(figsize=(7,4))
plt.scatter(df_electrodes.index, df_electrodes['probe_vertical_position'])
plt.title('Vertical position vs. electrode index')
plt.xlabel('Electrode index')
plt.ylabel('Vertical position (microns)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Summary and Next Steps
# - **We have explored the structure and content of Dandiset 000563 using streaming NWB file access.**
# - **Visualizations highlight LFP signal structure and probe/electrode mapping, providing a strong foundation for further analysis.**
# 
# *Recommended next steps:*
# - Investigate spike trains, units, and stimulus-response relationships
# - Subset recordings by anatomical location or probe group
# - Cross-reference LFP features with behavioral events or stimulus identity
# - Explore additional NWB files in this Dandiset (e.g., from other probes/mice)
# 
# For in-depth, interactive exploration, consider using [Neurosift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/&dandisetId=000563&dandisetVersion=draft) or developing custom workflows that follow the patterns presented in this notebook.