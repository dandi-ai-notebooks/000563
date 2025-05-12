# This script examines the units (neuron) data from the Dandiset
# We're interested in the spiking patterns that create the "barcode" patterns described in the dataset
# Optimized version that only looks at a small subset of data

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import time

print("Starting script...")
start_time = time.time()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
print(f"Loading file from {url}...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
print(f"File loaded in {time.time() - start_time:.2f} seconds")

# Get a quick overview of units
unit_ids = nwb.units.id[:]
print(f"Number of units: {len(unit_ids)}")
print(f"Unit columns: {nwb.units.colnames}")

# Get stimulus information
print("Getting stimulus information...")
try:
    repeat_fff_presentations = nwb.intervals['RepeatFFF_presentations']
    # Just get the first few trials
    trial_starts = repeat_fff_presentations.start_time[:50]
    print(f"Number of trial starts: {len(trial_starts)}")
    print(f"First 5 trial starts: {trial_starts[:5]}")
except Exception as e:
    print(f"Error getting stimulus info: {e}")
    # Use a placeholder for trial starts if needed
    trial_starts = np.linspace(0, 100, 50)

# Look at the first few good units
print("Examining good quality units...")
# First get all the quality values
unit_quality = nwb.units['quality'][:]
# Find the indices of good quality units
good_unit_indices = np.where(np.array(unit_quality) == 'good')[0]
print(f"Number of good quality units: {len(good_unit_indices)}")

# Only examine the first 3 good units 
num_units_to_examine = min(3, len(good_unit_indices))
examined_units = good_unit_indices[:num_units_to_examine]

plt.figure(figsize=(12, 8))
for i, unit_idx in enumerate(examined_units):
    unit_id = unit_ids[unit_idx]
    print(f"Processing unit {unit_id}...")
    
    # Get spike times for this unit
    spike_times = nwb.units['spike_times'][unit_idx]
    
    # Get basic unit info
    firing_rate = nwb.units['firing_rate'][unit_idx]
    
    # Instead of complex raster plot, let's just histogram the spike times
    plt.subplot(num_units_to_examine, 1, i+1)
    
    # If spikes exist, plot them
    if len(spike_times) > 0:
        # Create a simple histogram
        hist, bins = np.histogram(spike_times, bins=100)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.bar(bin_centers, hist, width=(bins[1]-bins[0]), alpha=0.7)
    else:
        plt.text(0.5, 0.5, "No spikes", ha='center', va='center')
    
    plt.title(f"Unit {unit_id} - Firing rate: {firing_rate:.2f} Hz")
    plt.ylabel("Spike count")
    
    if i == num_units_to_examine - 1:
        plt.xlabel("Time (s)")

plt.tight_layout()
plt.savefig('explore/unit_histograms.png')
print(f"Saved histogram figure to explore/unit_histograms.png")

print(f"Script completed in {time.time() - start_time:.2f} seconds")