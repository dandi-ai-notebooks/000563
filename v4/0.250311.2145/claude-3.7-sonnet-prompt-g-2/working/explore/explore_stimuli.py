# This script explores the stimulus presentation information
# to understand the types of stimuli that evoke the barcode patterns

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

print("Starting stimulus information analysis...")
start_time = time.time()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
print(f"Loading file from {url}...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
print(f"File loaded in {time.time() - start_time:.2f} seconds")

# Function to print basic info about stimulus presentations
def summarize_stimuli(presentations, name):
    """Print summary info about a stimulus presentation type"""
    print(f"\n{name} Summary:")
    print(f"Number of presentations: {len(presentations.id[:])}")
    print(f"Column names: {presentations.colnames}")
    
    # Get the dataframe for the first few examples
    df = presentations.to_dataframe().head(10)
    
    # Print key information
    print("\nFirst few presentations:")
    print(df[['start_time', 'stop_time', 'stimulus_name', 'stimulus_block']])
    
    # If there's contrast information, summarize it
    if 'contrast' in presentations.colnames:
        contrasts = presentations['contrast'][:]
        unique_contrasts = np.unique(contrasts)
        print(f"\nUnique contrast values: {unique_contrasts}")
    
    # If there's spatial frequency information, summarize it
    if 'spatial_frequency' in presentations.colnames:
        sf = presentations['spatial_frequency'][:]
        unique_sf = np.unique(sf)
        print(f"Unique spatial frequency values: {unique_sf}")
    
    # Get unique stimulus names
    stim_names = presentations['stimulus_name'][:]
    unique_names = np.unique(stim_names)
    print(f"\nUnique stimulus names: {unique_names}")
    
    return df

# Get the various stimulus presentation intervals
print("\nExamining stimulus presentations...")

# RepeatFFF (Repeated Full Field Flicker)
repeat_fff = nwb.intervals['RepeatFFF_presentations']
repeat_fff_df = summarize_stimuli(repeat_fff, "RepeatFFF (Repeated Full Field Flicker)")

# UniqueFFF (Unique Full Field Flicker)
unique_fff = nwb.intervals['UniqueFFF_presentations']
unique_fff_df = summarize_stimuli(unique_fff, "UniqueFFF (Unique Full Field Flicker)")

# Static gratings
static_block = nwb.intervals['static_block_presentations']
static_block_df = summarize_stimuli(static_block, "Static Block (Gratings)")

# Receptive field mapping
rf_block = nwb.intervals['receptive_field_block_presentations']
rf_block_df = summarize_stimuli(rf_block, "Receptive Field Block")

# Plot the distribution of stimulus durations for RepeatFFF
repeat_fff_durations = repeat_fff['stop_time'][:] - repeat_fff['start_time'][:]
plt.figure(figsize=(10, 5))
plt.hist(repeat_fff_durations, bins=50)
plt.title('RepeatFFF Stimulus Durations')
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.savefig('explore/repeat_fff_durations.png')
print(f"Saved stimulus duration histogram to explore/repeat_fff_durations.png")

# Plot the distribution of inter-stimulus intervals for RepeatFFF
isi = np.diff(repeat_fff['start_time'][:])
plt.figure(figsize=(10, 5))
plt.hist(isi, bins=50)
plt.title('RepeatFFF Inter-Stimulus Intervals')
plt.xlabel('Interval (s)')
plt.ylabel('Count')
plt.savefig('explore/repeat_fff_isi.png')
print(f"Saved inter-stimulus interval histogram to explore/repeat_fff_isi.png")

# Compare unique vs repeat FFF stimuli
print("\nComparing Unique vs Repeated FFF stimuli:")
print(f"RepeatFFF presentations: {len(repeat_fff.id[:])}")
print(f"UniqueFFF presentations: {len(unique_fff.id[:])}")

# Create a plot showing the timing of different stimuli
plt.figure(figsize=(12, 6))

# Get start times for each stimulus type
repeat_fff_starts = repeat_fff['start_time'][:]
unique_fff_starts = unique_fff['start_time'][:]
static_block_starts = static_block['start_time'][:]
rf_block_starts = rf_block['start_time'][:]

# Plot the stimulus presentation times
plt.scatter(repeat_fff_starts[:1000], np.ones_like(repeat_fff_starts[:1000]), label='RepeatFFF', s=3, alpha=0.5)
plt.scatter(unique_fff_starts[:1000], 2*np.ones_like(unique_fff_starts[:1000]), label='UniqueFFF', s=3, alpha=0.5)
plt.scatter(static_block_starts[:1000], 3*np.ones_like(static_block_starts[:1000]), label='Static Block', s=3, alpha=0.5)
plt.scatter(rf_block_starts[:1000], 4*np.ones_like(rf_block_starts[:1000]), label='RF Block', s=3, alpha=0.5)

plt.yticks([1, 2, 3, 4], ['RepeatFFF', 'UniqueFFF', 'Static Block', 'RF Block'])
plt.xlabel('Time (s)')
plt.title('Stimulus Presentation Timeline (first 1000 per type)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('explore/stimulus_timeline.png')
print(f"Saved stimulus timeline plot to explore/stimulus_timeline.png")

print(f"Script completed in {time.time() - start_time:.2f} seconds")