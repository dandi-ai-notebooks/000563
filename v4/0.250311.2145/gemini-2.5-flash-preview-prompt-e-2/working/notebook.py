# %% [markdown]
# # Exploring Dandiset 000563: Allen Institute Openscope - Barcoding
#
# This notebook explores Dandiset 000563, which contains extracellular electrophysiology data related to "barcoding" neural responses to visual stimuli in the mouse brain.
#
# **Note:** This notebook was AI-generated and has not been fully verified. Please exercise caution when interpreting the code or results.

# %% [markdown]
# ## Dandiset Overview
#
# Dandiset 000563, titled "Allen Institute Openscope - Barcoding", focuses on neural responses to white noise flicker visual stimuli, which can produce "barcode"-like patterns in spike rasters. This dataset explores whether these patterns could be used as identifiers of discrete cell types.
#
# The Dandiset can be accessed at: https://dandiarchive.org/dandiset/000563/0.250311.2145
#
# The data includes Neuropixels recordings throughout the mouse brain.
#
# This notebook will demonstrate how to:
# - Load basic information about the Dandiset using the DANDI API.
# - Access one of the NWB files in the Dandiset using a remote URL.
# - Explore the metadata and structure of the NWB file.
# - Load and visualize some of the electrophysiology data (LFP).

# %% [markdown]
# ## Required Packages
#
# This notebook requires the following Python packages:
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `numpy`
# - `matplotlib`
# - `seaborn`
# - `pandas`
# - `itertools`

# %%
# Load necessary libraries
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set seaborn theme for visualizations (excluding images)
sns.set_theme()

# %% [markdown]
# ## Loading the Dandiset
#
# We can use the DandiAPIClient to connect to the DANDI archive and retrieve information about the Dandiset.

# %%
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
for i, asset in enumerate(islice(assets, 5)):
    print(f"- {asset.path} (ID: {asset.identifier})")
    if i == 0:
        # Store the first asset path and ID for later use
        first_asset_path = asset.path
        first_asset_id = asset.identifier

# %% [markdown]
# ## Loading an NWB file
#
# This Dandiset contains multiple NWB files, primarily containing electrophysiology (`_ecephys.nwb`) and optogenetics (`_ogen.nwb`) data from different subjects and sessions.
#
# We will load the following NWB file for demonstration:
#
# `{first_asset_path}`
#
# We can access this file directly using its remote URL derived from its Asset ID.

# %%
# Construct the URL for the selected asset
nwb_url = f"https://api.dandiarchive.org/api/assets/{first_asset_id}/download/"
print(f"Loading NWB file from URL: {nwb_url}")

# Load the NWB file remotely
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# We won't display the full nwb object as it can be very large.
print("\nNWB file loaded successfully.")
print(f"Session description: {nwb.session_description}")
print(f"Session ID: {nwb.session_id}")

# %% [markdown]
# ## Exploring NWB File Contents and Metadata
#
# Let's look at some of the key sections and metadata available in this NWB file.

# %% [markdown]
# ### Acquisition
#
# The `acquisition` section contains the raw data recorded during the experiment. In this file, we expect to find the LFP data.

# %%
# Explore the acquisition section
print("Acquisition keys:")
for key in nwb.acquisition.keys():
    print(f"- {key}")

# Access the LFP data ElectricalSeries
if "probe_0_lfp" in nwb.acquisition and "probe_0_lfp_data" in nwb.acquisition["probe_0_lfp"].electrical_series:
    lfp_electrical_series = nwb.acquisition["probe_0_lfp"].electrical_series["probe_0_lfp_data"]
    print(f"\nLFP data shape: {lfp_electrical_series.data.shape}")
    print(f"LFP data units: {lfp_electrical_series.unit}")
    print(f"LFP timestamps shape: {lfp_electrical_series.timestamps.shape}")
else:
    print("\nCould not find expected LFP data in acquisition.")
    lfp_electrical_series = None # Ensure lfp_electrical_series is None if not found

# %% [markdown]
# ### Electrodes
#
# The `electrodes` table provides metadata about each recording channel, such as their location and grouping.

# %%
# Explore the electrodes table
if nwb.electrodes:
    print("\nElectrodes table columns:")
    for col in nwb.electrodes.colnames:
        print(f"- {col}")

    # Display the first few rows of the electrodes table
    electrodes_df = nwb.electrodes.to_dataframe()
    print("\nFirst 5 rows of the electrodes table:")
    print(electrodes_df.head())

# %% [markdown]
# ### Subject
#
# Information about the experimental subject is stored in the `subject` section.

# %%
# Explore subject metadata
if nwb.subject:
    print("\nSubject metadata:")
    print(f"- Subject ID: {nwb.subject.subject_id}")
    print(f"- Species: {nwb.subject.species}")
    print(f"- Genotype: {nwb.subject.genotype}")
    print(f"- Sex: {nwb.subject.sex}")
    print(f"- Age: {nwb.subject.age}")
    print(f"- Strain: {nwb.subject.strain}")

# %% [markdown]
# ## Exploring this NWB file on Neurosift
#
# For a different way to explore the contents and structure of this NWB file directly in your web browser, you can use Neurosift:
#
# https://neurosift.app/nwb?url={nwb_url}&dandisetId=000563&dandisetVersion=0.250311.2145

# %% [markdown]
# ## Visualizing LFP Data
#
# We can load a subset of the LFP data and its corresponding timestamps to visualize the activity over time. Since the data is large, we will load only a short segment for a few channels.

# %%
# Check if LFP data was successfully loaded
if lfp_electrical_series is not None:
    # Define the start and end index for the data subset
    start_index = 10000
    end_index = start_index + 5000 # Load 5000 samples
    
    # Define the channel indices to visualize
    channel_indices = [0, 10, 20, 30] # Visualize 4 channels

    # Load the data subset
    # We select rows by index and columns by index
    lfp_data_subset = lfp_electrical_series.data[start_index:end_index, channel_indices]

    # Load the corresponding timestamps subset
    lfp_timestamps_subset = lfp_electrical_series.timestamps[start_index:end_index]

    # Plot the LFP data subset
    plt.figure(figsize=(12, 6))
    for i, channel_index in enumerate(channel_indices):
        # Offset the channels vertically for better visibility
        offset = i * 500 # Adjust offset as needed
        plt.plot(lfp_timestamps_subset, lfp_data_subset[:, i] + offset, label=f'Channel Index {channel_index}')

    plt.xlabel("Time (s)")
    plt.ylabel(f"LFP Signal ({lfp_electrical_series.unit}) + Offset")
    plt.title("Subset of LFP Data from Selected Channels")
    plt.legend()
    plt.show()
else:
    print("LSP data not available for visualization.")

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to access and perform basic exploration and visualization of electrophysiology data from Dandiset 000563. We loaded Dandiset metadata, explored assets, accessed a specific NWB file remotely, examined its structure and metadata, and visualized a subset of the LFP data.
#
# Possible future directions for analysis include:
# - Exploring other NWB files in the Dandiset (e.g., optogenetics data).
# - Downloading larger subsets of data for more extensive analysis.
# - Applying signal processing techniques (e.g., filtering, spectral analysis) to the LFP data.
# - Investigating the relationship between neural activity and the visual stimuli.
# - Comparing data across different subjects or sessions.
# - Analyzing spike times (if available in other NWB files in the dandiset) and their "barcode" patterns.