"""
Script to visualize the "barcode" pattern of neuronal responses to repeated stimuli.
This script will:
1. Find neurons that respond to the RepeatFFF stimulus
2. Create a raster plot showing spike responses aligned to stimulus onsets
3. Visualize the "barcode" pattern described in the dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile

# Connect to the NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
print(f"Loading file from {url}...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Get stimulus presentation times for RepeatFFF
print("Extracting RepeatFFF stimulus times...")
repeat_fff = h5_file['intervals']['RepeatFFF_presentations']
stim_start_times = repeat_fff['start_time'][:]
stim_stop_times = repeat_fff['stop_time'][:]
stim_blocks = repeat_fff['stimulus_block'][:]

# Find a block of repeated stimuli
unique_blocks = np.unique(stim_blocks)
block = unique_blocks[0]  # Use the first block
block_mask = stim_blocks == block
block_start_times = stim_start_times[block_mask]
block_duration = np.median(stim_stop_times[block_mask] - block_start_times)

print(f"Using stimulus block {block} with {len(block_start_times)} presentations")
print(f"Average stimulus duration: {block_duration * 1000:.2f} ms")

# Let's use a smaller subset of presentations for the visualization (e.g., 100)
stim_subset_start = 0
stim_subset_size = 100
stim_times_subset = block_start_times[stim_subset_start:stim_subset_start+stim_subset_size]

# Get all units
units = h5_file['units']
num_units = units['id'].shape[0]
spike_times = units['spike_times'][:]
spike_times_index = units['spike_times_index'][:]

# Compute pre and post stimulus window for alignment
pre_stim = 0  # No time before stimulus onset
post_stim = block_duration * 1.5  # 1.5x the stimulus duration

# Helper function to get spikes aligned to stimulus onsets
def get_aligned_spikes(unit_spikes, stim_times, pre_stim, post_stim):
    aligned_spikes = []
    for stim_time in stim_times:
        # Find spikes that occur within window around stimulus
        mask = (unit_spikes >= stim_time - pre_stim) & (unit_spikes <= stim_time + post_stim)
        if np.sum(mask) > 0:
            # Align spike times to stimulus onset
            aligned_times = unit_spikes[mask] - stim_time
            aligned_spikes.append(aligned_times)
        else:
            aligned_spikes.append(np.array([]))
    return aligned_spikes

# Try to find responsive units
print("Searching for responsive units...")
responsive_units = []

# Check a sample of units for responsiveness (limit to 200 for time)
for unit_idx in range(min(200, num_units)):
    if unit_idx % 20 == 0:  # Progress update
        print(f"Checking unit {unit_idx}...")
    
    # Get spike times for this unit
    index_start = spike_times_index[unit_idx]
    if unit_idx < num_units - 1:
        index_end = spike_times_index[unit_idx + 1]
    else:
        index_end = len(spike_times)
    
    unit_spike_times = spike_times[index_start:index_end]
    
    # Only consider units with good firing rate (0.5-50 Hz)
    if len(unit_spike_times) < 50 or len(unit_spike_times) > 500000:
        continue
    
    # Test responsiveness on a small subset of stimuli
    test_stims = stim_times_subset[:20]
    aligned_spikes = get_aligned_spikes(unit_spike_times, test_stims, pre_stim, post_stim)
    
    # Count spikes during stimulus
    total_spikes = sum(len(spikes) for spikes in aligned_spikes)
    
    # If unit has enough spikes during stimulus presentations, consider it responsive
    if total_spikes > 10:
        responsive_units.append((unit_idx, total_spikes, unit_spike_times))

print(f"Found {len(responsive_units)} potentially responsive units")

# Sort responsive units by number of spikes
responsive_units.sort(key=lambda x: x[1], reverse=True)

# Create barcode plot for top responsive unit
if responsive_units:
    top_unit = responsive_units[0]
    unit_idx, total_spikes, unit_spike_times = top_unit
    
    print(f"\nCreating barcode plot for unit {unit_idx} with {total_spikes} stimulus-aligned spikes")
    
    # Get all aligned spikes for this unit across stimulus presentations
    aligned_spikes = get_aligned_spikes(unit_spike_times, stim_times_subset, pre_stim, post_stim)
    
    # Plot the barcodes (raster plot)
    plt.figure(figsize=(10, 8))
    
    # Plot each stimulus presentation as a row
    for i, spikes in enumerate(aligned_spikes):
        if len(spikes) > 0:
            plt.scatter(spikes, np.ones_like(spikes) * i, marker='|', s=10, color='black')
    
    # Add a vertical line at stimulus onset
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
    
    # Draw stimulus duration
    plt.axvspan(0, block_duration, color='lightgray', alpha=0.3, label='Stimulus Duration')
    
    plt.xlabel('Time from Stimulus Onset (s)')
    plt.ylabel('Stimulus Presentation Number')
    plt.title(f'Barcode Pattern: Unit {unit_idx} Response to Repeated Visual Stimuli')
    plt.xlim(pre_stim - 0.01, post_stim + 0.01)
    plt.ylim(-1, stim_subset_size + 1)
    plt.legend()
    plt.savefig('explore/barcode_pattern.png')
    print("Saved barcode pattern visualization to explore/barcode_pattern.png")
    
    # Also create an average PSTH to show the overall response pattern
    bin_width = 0.002  # 2 ms bins
    bins = np.arange(pre_stim, post_stim + bin_width, bin_width)
    
    # Combine all spikes across presentations
    all_spikes = np.concatenate(aligned_spikes)
    
    # Compute PSTH
    psth, bin_edges = np.histogram(all_spikes, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize by number of presentations and bin width to get firing rate
    firing_rate = psth / (stim_subset_size * bin_width)
    
    plt.figure(figsize=(10, 5))
    plt.bar(bin_centers, firing_rate, width=bin_width, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
    plt.axvspan(0, block_duration, color='lightgray', alpha=0.3, label='Stimulus Duration')
    plt.xlabel('Time from Stimulus Onset (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'PSTH: Unit {unit_idx} Average Response to Repeated Visual Stimuli')
    plt.xlim(pre_stim - 0.01, post_stim + 0.01)
    plt.legend()
    plt.savefig('explore/barcode_psth.png')
    print("Saved PSTH visualization to explore/barcode_psth.png")
else:
    print("No responsive units found")