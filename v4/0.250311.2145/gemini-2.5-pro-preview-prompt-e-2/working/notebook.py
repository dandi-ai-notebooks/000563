# %% [markdown]
# # Exploring Dandiset 000563: Allen Institute Openscope - Barcoding (Version 0.250311.2145)

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This Dandiset, titled "Allen Institute Openscope - Barcoding," contains data related to how visual neurons respond to white noise flicker visual stimuli. The experiments used the OpenScope Neuropixels passive viewing protocol with mice. The visual stimuli were either a spatially uniform field whose luminance was modulated in time (Full Field Flicker) or a standing sinusoidal grating whose contrast was modulated in time (Static Gratings). The dataset aims to provide "barcodes" for visually responsive neurons throughout the mouse brain, which could potentially be used as identifiers of discrete cell types.
#
# You can find more information about this Dandiset and access its data at:
# [https://dandiarchive.org/dandiset/000563/0.250311.2145](https://dandiarchive.org/dandiset/000563/0.250311.2145)

# %% [markdown]
# ## Purpose of this Notebook
#
# This notebook will guide you through:
# 1. Listing the required Python packages.
# 2. Connecting to the DANDI archive and retrieving basic information about the Dandiset.
# 3. Listing some of the assets (files) available in the Dandiset.
# 4. Loading a specific NWB (Neurodata Without Borders) file from the Dandiset.
# 5. Exploring the metadata and contents of the loaded NWB file.
# 6. Visualizing some example data from the NWB file, such as eye tracking and running speed.
# 7. Summarizing the findings and suggesting potential future directions for analysis.

# %% [markdown]
# ## Required Packages
#
# To run this notebook, you will need the following Python packages. Please ensure they are installed in your environment.
#
# - `dandi` (for interacting with the DANDI Archive)
# - `pynwb` (for reading NWB files)
# - `h5py` (dependency for pynwb, for HDF5 file access)
# - `remfile` (for streaming remote files)
# - `numpy` (for numerical operations)
# - `matplotlib` (for plotting)
# - `seaborn` (for enhanced visualizations)
# - `pandas` (for data manipulation, especially with NWB tables)
#
# This notebook assumes these packages are already installed. No `pip install` commands are included.

# %%
# Import necessary libraries
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Apply a seaborn style for plots
sns.set_theme()

# %% [markdown]
# ## Connecting to DANDI and Loading Dandiset Information
#
# We will use the `DandiAPIClient` to connect to the DANDI archive and retrieve information about our target Dandiset (000563, version 0.250311.2145).

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset_id = "000563"
dandiset_version = "0.250311.2145"
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# Print basic information about the Dandiset from its raw metadata
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata.get('name', 'N/A')}")
print(f"Dandiset URL: {metadata.get('url', 'N/A')}")
print(f"Dandiset description: {metadata.get('description', 'N/A')[:500]}...") # Print first 500 chars of description

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})") # asset.identifier is correct for RemoteBlobAsset

# %% [markdown]
# ## Loading an NWB File
#
# Now, let's load one of the NWB files from this Dandiset. We will use the file `sub-681446/sub-681446_ses-1290510496_ogen.nwb`.
# The `tools_cli.py nwb-file-info` command (which is not run in this notebook) provides the direct download URL and Python code snippets for loading. We will use that information here.
#
# The asset ID for this file is `2f2ac304-83a3-4352-8612-5f34b68062a0`.
# The download URL is: `https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/`

# %%
# Define the URL for the NWB file
nwb_asset_id = "2f2ac304-83a3-4352-8612-5f34b68062a0"
nwb_file_url = f"https://api.dandiarchive.org/api/assets/{nwb_asset_id}/download/"
selected_file_path = "sub-681446/sub-681446_ses-1290510496_ogen.nwb" # For display purposes

print(f"Loading NWB file: {selected_file_path}")
print(f"From URL: {nwb_file_url}")

# Load the NWB file using remfile and pynwb
# This code is based on the output of `tools_cli.py nwb-file-info`
remote_file_stream = remfile.File(nwb_file_url)
h5_file = h5py.File(remote_file_stream, 'r') # Specify read-only mode
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Specify read-only mode for IO
nwbfile = io.read()

print("\nNWB file loaded successfully.")
print(f"Session ID: {nwbfile.session_id}")
print(f"Session Start Time: {nwbfile.session_start_time}")
print(f"Subject ID: {nwbfile.subject.subject_id if nwbfile.subject else 'N/A'}")

# %% [markdown]
# ### Neurosift Link for Interactive Exploration
#
# You can explore this NWB file interactively using Neurosift:
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/&dandisetId=000563&dandisetVersion=0.250311.2145](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/&dandisetId=000563&dandisetVersion=0.250311.2145)

# %% [markdown]
# ## Exploring the NWB File Contents
#
# Let's look at some of the metadata and data containers within the loaded NWB file.
# The `nwb-file-info` tool output (not shown here) gives a detailed tree structure of the file. We will explore some common NWB components.

# %% [markdown]
# ### Acquisition Data
#
# The `nwbfile.acquisition` object often contains raw acquired data streams.

# %%
print("Contents of nwbfile.acquisition:")
if nwbfile.acquisition:
    for item_name, item_data in nwbfile.acquisition.items():
        print(f"- {item_name}: ({type(item_data).__name__})")
        if hasattr(item_data, 'description'):
            print(f"  Description: {item_data.description}")
else:
    print("No acquisition data found.")

# %% [markdown]
# Let's specifically look at `EyeTracking` data if available.

# %%
if "EyeTracking" in nwbfile.acquisition:
    eye_tracking_data = nwbfile.acquisition["EyeTracking"]
    print("\nEyeTracking data details:")
    if hasattr(eye_tracking_data, 'spatial_series') and eye_tracking_data.spatial_series:
        for series_name, series_obj in eye_tracking_data.spatial_series.items():
            print(f"  - {series_name} ({type(series_obj).__name__}):")
            if hasattr(series_obj, 'data'):
                 print(f"    Data shape: {series_obj.data.shape}, Data dtype: {series_obj.data.dtype}")
            if hasattr(series_obj, 'timestamps') and series_obj.timestamps is not None:
                 print(f"    Timestamps shape: {series_obj.timestamps.shape if hasattr(series_obj.timestamps, 'shape') else 'N/A (likely linked)'}")
    else:
        print("  No spatial series found in EyeTracking.")
else:
    print("\nNo 'EyeTracking' data found in nwbfile.acquisition.")

# %% [markdown]
# ### Processing Modules
#
# The `nwbfile.processing` object often contains processed data derived from raw signals.

# %%
print("\nContents of nwbfile.processing:")
if nwbfile.processing:
    for module_name, processing_module in nwbfile.processing.items():
        print(f"- {module_name}: ({type(processing_module).__name__}) - {processing_module.description}")
        if hasattr(processing_module, 'data_interfaces') and processing_module.data_interfaces:
            print("  Data interfaces:")
            for di_name, di_obj in processing_module.data_interfaces.items():
                print(f"    - {di_name} ({type(di_obj).__name__})")
else:
    print("No processing modules found.")

# %% [markdown]
# Let's look at `running` speed data from the `running` processing module, if available.

# %%
if "running" in nwbfile.processing and "running_speed" in nwbfile.processing["running"].data_interfaces:
    running_speed_ts = nwbfile.processing["running"].data_interfaces["running_speed"]
    print("\nRunning speed TimeSeries details:")
    print(f"  Data shape: {running_speed_ts.data.shape}, Data dtype: {running_speed_ts.data.dtype}")
    print(f"  Timestamps shape: {running_speed_ts.timestamps.shape}")
    print(f"  Unit: {running_speed_ts.unit}")
else:
    print("\n'running_speed' TimeSeries not found in nwbfile.processing['running'].")

# %% [markdown]
# ### Intervals
#
# The `nwbfile.intervals` object can store information about epochs or experimental trials.

# %%
print("\nContents of nwbfile.intervals:")
if nwbfile.intervals:
    for interval_name, time_intervals in nwbfile.intervals.items():
        print(f"- {interval_name}: ({type(time_intervals).__name__}) - {time_intervals.description[:100]}...")
        print(f"  Columns: {list(time_intervals.colnames)}")
        # Accessing .id to get a count of intervals without loading full table to dataframe
        # This relies on .id being populated and representative of row count.
        # For TimeIntervals, len(time_intervals.id) gives the number of intervals.
        if hasattr(time_intervals, 'id') and time_intervals.id is not None:
            print(f"  Number of intervals: {len(time_intervals.id)}")
        else:
            print("  Number of intervals: (Could not be determined without loading fully)")
else:
    print("No intervals found.")

# %% [markdown]
# ### Units (Spike Data)
#
# The `nwbfile.units` table contains information about sorted spike units, if present.

# %%
print("\nContents of nwbfile.units (spike data):")
if nwbfile.units:
    print(f"  Description: {nwbfile.units.description}")
    print(f"  Columns: {list(nwbfile.units.colnames)}")
    # Get number of units from the length of the 'id' column, which is usually efficient
    print(f"  Number of units: {len(nwbfile.units.id)}")
    
    # For displaying a few units, convert to DataFrame but only take head.
    # This might still be slow if underlying data access is not optimized.
    # We're assuming .to_dataframe().head() is reasonably efficient for a preview.
    try:
        df_units_head = nwbfile.units.to_dataframe().head()
        if not df_units_head.empty:
            print("\n  Example of first few units (ID and quality):")
            # Displaying 'quality' if it exists, otherwise just the index (unit ID)
            if 'quality' in df_units_head.columns:
                print(df_units_head[['quality']])
            else:
                print(df_units_head.index.to_frame(name="Unit ID"))
        else:
            print("  Units table is empty or head could not be retrieved.")
    except Exception as e:
        print(f"  Could not display head of units table due to: {e}")
else:
    print("No units (spike data) found.")


# %% [markdown]
# ## Visualizing Data from the NWB File
#
# Let's plot some of the data we've identified. We'll be careful to load only subsets of data if the full datasets are too large, to avoid long loading times over the network.

# %% [markdown]
# ### Visualizing Pupil Tracking Area
#
# If pupil tracking data is available under `nwbfile.acquisition['EyeTracking'].spatial_series['pupil_tracking']`, let's plot a segment of the pupil area over time.

# %%
if "EyeTracking" in nwbfile.acquisition and \
   "pupil_tracking" in nwbfile.acquisition["EyeTracking"].spatial_series:
    pupil_tracking = nwbfile.acquisition["EyeTracking"].spatial_series["pupil_tracking"]

    if hasattr(pupil_tracking, 'area') and hasattr(pupil_tracking, 'timestamps'):
        print("Plotting pupil tracking area...")
        # Load a subset of data to keep it manageable
        num_points_to_plot = 1000
        pupil_area_data = pupil_tracking.area[:num_points_to_plot]
        pupil_timestamps_data = pupil_tracking.timestamps[:num_points_to_plot]

        plt.figure(figsize=(12, 6))
        sns.set_theme() # Ensure seaborn theme is applied
        plt.plot(pupil_timestamps_data, pupil_area_data)
        plt.xlabel(f"Time ({pupil_tracking.timestamps_unit})")
        plt.ylabel(f"Pupil Area ({pupil_tracking.unit} relative)") # Unit might be 'pixels' or relative
        plt.title(f"Pupil Area (First {num_points_to_plot} Points)")
        plt.show()
    else:
        print("Pupil tracking area or timestamps data not found.")
else:
    print("Pupil tracking data ('EyeTracking' or 'pupil_tracking' series) not found in acquisition.")

# %% [markdown]
# ### Visualizing Running Speed
#
# If running speed data is available under `nwbfile.processing['running'].data_interfaces['running_speed']`, let's plot a segment of it.

# %%
if "running" in nwbfile.processing and \
   "running_speed" in nwbfile.processing["running"].data_interfaces:
    running_speed_ts = nwbfile.processing["running"].data_interfaces["running_speed"]

    if hasattr(running_speed_ts, 'data') and hasattr(running_speed_ts, 'timestamps'):
        print("Plotting running speed...")
        # Load a subset of data
        num_points_to_plot = 2000
        running_speed_data = running_speed_ts.data[:num_points_to_plot]
        running_timestamps_data = running_speed_ts.timestamps[:num_points_to_plot]

        plt.figure(figsize=(12, 6))
        sns.set_theme() # Ensure seaborn theme is applied
        plt.plot(running_timestamps_data, running_speed_data)
        plt.xlabel(f"Time ({running_speed_ts.timestamps_unit})")
        plt.ylabel(f"Running Speed ({running_speed_ts.unit})")
        plt.title(f"Running Speed (First {num_points_to_plot} Points)")
        plt.show()
    else:
        print("Running speed data or timestamps not found in the TimeSeries.")
else:
    print("Running speed TimeSeries ('running_speed') not found in processing module 'running'.")

# %% [markdown]
# ### Visualizing Spike Times from a Single Unit
#
# If spike data (`nwbfile.units`) is available, let's select the first unit and plot its spike times as a raster plot over a short interval.

# %%
if nwbfile.units is not None and len(nwbfile.units.id) > 0:
    print("Plotting spike times for the first unit...")
    units_df = nwbfile.units.to_dataframe()
    
    # Select the spike times for the first unit (index 0)
    # The spike_times column in the DataFrame contains numpy arrays of spike times for each unit
    first_unit_spike_times = units_df.iloc[0]['spike_times']
    first_unit_id = units_df.index[0] # This gets the actual unit ID from the DataFrame index

    if first_unit_spike_times is not None and len(first_unit_spike_times) > 0:
        # Select a time window for plotting, e.g., the first 10 seconds of spikes
        time_window_end = min(first_unit_spike_times[0] + 10, first_unit_spike_times[-1]) # Max 10s or end of recording
        spikes_in_window = first_unit_spike_times[
            (first_unit_spike_times >= first_unit_spike_times[0]) & (first_unit_spike_times <= time_window_end)
        ]
        
        if len(spikes_in_window) > 0:
            plt.figure(figsize=(12, 3))
            sns.set_theme() # Ensure seaborn theme is applied
            plt.eventplot(spikes_in_window, lineoffset=0, linelength=0.8)
            plt.xlabel("Time (s)")
            plt.ylabel(f"Unit ID: {first_unit_id}")
            plt.title(f"Spike Raster for Unit {first_unit_id} (up to {time_window_end:.2f}s)")
            plt.yticks([]) # No y-ticks needed for a single unit raster
            plt.show()
        else:
            print(f"Unit {first_unit_id} has no spikes in the selected initial window.")
    else:
        print(f"Unit {first_unit_id} has no spike times recorded or spike_times array is empty.")
else:
    print("No units data found, or no units available to plot spike times.")

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to:
# - Connect to the DANDI archive and retrieve Dandiset metadata.
# - List assets within a Dandiset.
# - Load a specific NWB file using its DANDI asset URL.
# - Explore basic NWB file structure, including acquisition data, processing modules, intervals, and units.
# - Visualize example data such as pupil area, running speed, and spike times for a single unit.
#
# ### Possible Future Directions for Analysis:
#
# 1.  **Detailed Stimulus Correlation:** Explore the `nwbfile.intervals` (e.g., `RepeatFFF_presentations`, `UniqueFFF_presentations`, `static_block_presentations`) to correlate neural activity (spike times from `nwbfile.units`) with specific visual stimulus presentations.
2.  **Population Analysis:** Analyze spike data from multiple units simultaneously. This could involve calculating population firing rates, cross-correlations, or applying dimensionality reduction techniques.
3.  **Behavioral State Correlation:** Correlate neural activity with behavioral data, such as running speed or pupil metrics, to understand how brain state influences neural responses.
4.  **Across-Animal/Across-Session Comparisons:** If other NWB files in this Dandiset follow a similar structure, the methods shown here can be adapted to compare data across different subjects or recording sessions.
5.  **Advanced Visualizations:** Create more sophisticated visualizations, such as peristimulus time histograms (PSTHs) aligned to stimulus onsets, or heatmaps of neural activity across different conditions or trials.
6.  **Explore other NWB files:** This notebook focused on an `_ogen.nwb` file. The Dandiset also contains `_ecephys.nwb` files which likely contain more detailed electrophysiology data (e.g., raw voltage traces, LFP), which could be explored similarly. Remember to always check the `nwb-file-info` output (externally) to understand the structure of those files first.
#
# Remember to consult the DANDI archive page for this Dandiset and any associated publications for more context on the experimental design and data collection, which will inform more targeted analyses.
# %% [markdown]
# ---
# End of AI-Generated Notebook.