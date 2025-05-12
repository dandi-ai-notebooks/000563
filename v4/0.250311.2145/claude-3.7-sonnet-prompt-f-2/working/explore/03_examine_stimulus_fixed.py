"""
Script to examine the stimulus presentation information in the ogen.nwb file.
This script focuses specifically on the RepeatFFF_presentations data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import remfile
import pynwb

# Directly access the RepeatFFF_presentations in the NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
print(f"Loading file from {url}...")

# Create remfile without context manager
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Examine the structure of the h5 file using a low-level approach
print("\nTop-level keys:")
for key in h5_file.keys():
    print(f"  {key}")

print("\nIntervals keys:")
if 'intervals' in h5_file:
    for key in h5_file['intervals'].keys():
        print(f"  {key}")

# Check if RepeatFFF_presentations exists and get basic info
if 'intervals' in h5_file and 'RepeatFFF_presentations' in h5_file['intervals']:
    repeat_fff = h5_file['intervals']['RepeatFFF_presentations']
    print("\nRepeatFFF_presentations attributes:")
    for attr in repeat_fff.attrs:
        print(f"  {attr}: {repeat_fff.attrs[attr]}")
    
    print("\nRepeatFFF_presentations datasets:")
    for key in repeat_fff.keys():
        if isinstance(repeat_fff[key], h5py.Dataset):
            dataset = repeat_fff[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
    
    # Get a small sample of start_time and stop_time data
    if 'start_time' in repeat_fff:
        start_times = repeat_fff['start_time'][:10]
        print(f"\nFirst 10 start times: {start_times}")
    
    if 'stimulus_name' in repeat_fff:
        # Get only a few unique stimulus names
        stimulus_names = np.unique(repeat_fff['stimulus_name'][:100])
        print(f"\nSome unique stimulus names: {stimulus_names}")

# Create a simple graph to save
plt.figure(figsize=(10, 3))
plt.text(0.5, 0.5, "Stimulus Analysis", ha='center', va='center', fontsize=16)
plt.text(0.5, 0.3, "NWB file contains stimulus presentation intervals", ha='center', va='center', fontsize=12)
plt.axis('off')
plt.savefig('explore/stimulus_analysis.png')
print("\nSaved a placeholder figure to explore/stimulus_analysis.png")