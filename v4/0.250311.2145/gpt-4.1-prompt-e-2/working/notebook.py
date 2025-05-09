# %% [markdown]
# # Exploring Dandiset 000563: Allen Institute Openscope - Barcoding
#
# **Note: This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting code or results.**
#
# ---
#
# ## Overview
#
# This notebook helps you explore [Dandiset 000563, version 0.250311.2145](https://dandiarchive.org/dandiset/000563/0.250311.2145), titled _Allen Institute Openscope - Barcoding_. The Dandiset comprises extracellular electrophysiological recordings and related data from mouse brain visual areas responding to white noise visual stimuli, enabling investigation into neural "barcode" patterns and precise spike timing.
#
# **Citation:**  
# Reinagel, Pamela; Lecoq, Jérôme; Durand, Séverine; Gillis, Ryan; ...; Howard, Robert (2025) Allen Institute Openscope - Barcoding (Version 0.250311.2145) [Data set]. DANDI Archive. https://doi.org/10.48324/dandi.000563/0.250311.2145
#
# **Keywords:** mouse, neuropixel, extracellular electrophysiology, neocortex, barcoding, temporal precision
#
# ---
#
# ## What this notebook covers:
# - Summarizes dataset contents and metadata.
# - Shows how to access Dandiset assets using the DANDI API.
# - Demonstrates how to load and inspect a remote NWB file.
# - Visualizes Local Field Potential (LFP) data across electrodes.
# - Illustrates examining metadata tables.
# - Provides links for external exploration (Neurosift).
#
# _**Tip:** The datasets may be large; this notebook demonstrates working with manageable data slices for efficient exploration._

# %% [markdown]
# ## Required packages
# This notebook requires the following Python packages (assumed to be pre-installed):
#
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `pandas`
# - `matplotlib`
# - `seaborn`
#
# _Please ensure these are installed before running the notebook._

# %%
# Import packages
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py
import pynwb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For visualization aesthetics
sns.set_theme()
plt.rcParams['figure.figsize'] = (12, 5)

# %% [markdown]
# ## 1. Explore Dandiset Metadata

# %%
# Connect to DANDI archive and get Dandiset
dandiset_id = "000563"
dandiset_version = "0.250311.2145"
client = DandiAPIClient()
dandiset = client.get_dandiset(dandiset_id, dandiset_version)
metadata = dandiset.get_raw_metadata()

print(f"Dandiset ID: {metadata.get('identifier', '')}")
print(f"Name: {metadata.get('name', '')}")
print(f"Version: {metadata.get('version', '')}")
print(f"Description:\n{metadata.get('description', '')}\n")
# Display contributor names if possible
contributors = metadata.get('contributor', [])
if contributors and isinstance(contributors[0], dict):
    contributors_str = "; ".join(str(c.get("name", c)) for c in contributors)
else:
    contributors_str = "; ".join(str(c) for c in contributors)
print(f"Contributors: {contributors_str}\n")
print("Keywords:", ", ".join(metadata.get('keywords', [])))
print("\nCitation:")
print(metadata.get('citation', ''))

# %% [markdown]
# ---
# ## 2. List Dandiset Assets
#
# Let's list the first several files to get a sense of what types of assets are present.

# %%
# List the first 10 assets in the Dandiset
assets = list(islice(dandiset.get_assets(), 10))
df_assets = pd.DataFrame([{'path': asset.path, 'size_MB': asset.size / 1e6, 'asset_id': asset.identifier} for asset in assets])
display(df_assets.style.hide(axis='index'))

# %% [markdown]
# For this exploration, we'll use the following NWB file:
#
# - **Path:** `sub-681446/sub-681446_ses-1290510496_probe-0_ecephys.nwb`
# - **Asset ID:** `1f158fe0-f8ef-495e-b031-da25316a335c`
# - **Download URL:** [Link](https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/)
#
# You can also explore this file on [Neurosift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/&dandisetId=000563&dandisetVersion=draft).
#
# _Note: Only a portion of the file will be loaded at a time, as these are very large datasets!_

# %% [markdown]
# ---
# ## 3. Load and Inspect NWB File Metadata

# %%
# Load the NWB file remotely following DANDI/CLI usage
nwb_url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Show session-level metadata
print('Session Description:', nwb.session_description)
print('Identifier:', nwb.identifier)
print('Session Start:', nwb.session_start_time)
print('Institution:', getattr(nwb, 'institution', '?'))
print('Subject:')
subject = nwb.subject
print('  Subject ID:', getattr(subject, 'subject_id', '?'))
print('  Species:', getattr(subject, 'species', '?'))
print('  Genotype:', getattr(subject, 'genotype', '?'))
print('  Sex:', getattr(subject, 'sex', '?'))
print('  Age (days):', getattr(subject, 'age_in_days', '?'))

# %% [markdown]
# Let's summarize the tree of key objects in this NWB file:

# %% [markdown]
# ```
# NWB file
# ├── acquisition
# │   └── probe_0_lfp (LFP)
# │       └── electrical_series
# │           └── probe_0_lfp_data (ElectricalSeries)
# │               ├── data: shape (10168076, 73) [float32, volts]
# │               ├── timestamps: (10168076,) [float64, seconds]
# │               └── electrodes: references electrodes table
# ├── electrodes (metadata table)
# |    ├── 73 rows × 13 columns (locations, impedance, etc.)
# ├── electrode_groups (e.g. probeA)
# ├── devices (includes Neuropixels probe details)
# └── subject (mouse metadata)
# ```
#
# We'll now look visually at the electrode metadata and load a sample of LFP data.

# %% [markdown]
# ---
# ## 4. Electrode Metadata Table

# %%
# Electrode metadata as pandas DataFrame (first 7 rows)
electrodes_table = nwb.electrodes.to_dataframe()
display(electrodes_table.head(7))

# Inspect what columns are present
print("Columns in electrode table:", electrodes_table.columns.tolist())

# %% [markdown]
# ---
# ## 5. Explore LFP Data Shape and Preview

# %%
probe_0_lfp = nwb.acquisition["probe_0_lfp"]
electrical_series = probe_0_lfp.electrical_series["probe_0_lfp_data"]

# LFP data shape: timepoints × electrodes
data_shape = electrical_series.data.shape
print("LFP data shape (time, channels):", data_shape)
print("LFP channel unit:", electrical_series.unit)
print("Timestamps unit:", electrical_series.timestamps_unit)

# %% [markdown]
# ---
# ## 6. Visualize LFP from a Subset of Channels
#
# To avoid streaming too much data, let’s plot ~2 seconds for a few channels.

# %%
# How many time points correspond to 2 seconds?
n_seconds = 2
sampling_rate = 625  # Hz, from probeA.lfp_sampling_rate
n_samples = n_seconds * sampling_rate
n_channels = min(5, data_shape[1])

data_subset = electrical_series.data[:n_samples, :n_channels]  # shape: (samples, channels)
timestamps_subset = electrical_series.timestamps[:n_samples]

plt.figure(figsize=(12, 6))
for i in range(n_channels):
    plt.plot(timestamps_subset, data_subset[:, i] * 1e3 + i*400, label=f'Channel {i}')
plt.xlabel(f'Time ({electrical_series.timestamps_unit})')
plt.ylabel('LFP (mV, offset per channel)')
plt.title('LFP: First 2 seconds, 5 channels')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 7. Advanced: Electrode Location Mapping
#
# Let's see how channels distribute by their assigned reported brain region:

# %%
# Value counts of locations
if "location" in electrodes_table.columns:
    location_counts = electrodes_table["location"].value_counts()
    location_counts.plot(kind='bar')
    plt.ylabel("Number of electrodes")
    plt.title("Electrodes by brain region ('location')")
    plt.show()
else:
    print("No 'location' column found in electrode table.")

# %% [markdown]
# ---
# ## 8. Neurosift Exploration
#
# You can browse this NWB file interactively on [Neurosift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/&dandisetId=000563&dandisetVersion=draft).
#
# Neurosift provides a web interface to visualize and inspect various aspects of NWB files.

# %% [markdown]
# ---
# ## 9. Summary and Future Directions
#
# This notebook demonstrated how to:
# - Discover and inspect assets in a Dandiset via the DANDI API
# - Load NWB files remotely using PyNWB and remfile
# - Summarize and visualize LFP recording channels and electrode metadata
#
# Possible next steps for analysis:
# - Explore additional NWB files from the Dandiset (different sessions, probes, or types)
# - Analyze spike data or other modalities present (if available)
# - Compare responses across brain regions or experimental conditions
# - Develop advanced visualizations, e.g., cross-channel correlations, time-frequency analyses
#
# _Remember: This is a large, rich dataset—explore with care and patience!_