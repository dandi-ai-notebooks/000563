"""
Script to find and compare the "barcode" patterns across multiple neurons.
This will demonstrate how different neurons have distinctive temporal response patterns
to the same repeated white noise stimuli.
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

# Let's use a smaller subset of presentations for the visualization
stim_subset_start = 0
stim_subset_size = 50
stim_times_subset = block_start_times[stim_subset_start:stim_subset_start+stim_subset_size]

# Get all units
units = h5_file['units']
num_units = units['id'].shape[0]
spike_times = units['spike_times'][:]
spike_times_index = units['spike_times_index'][:]

# We'll look at quality and firing rate
if 'quality' in units and 'firing_rate' in units:
    quality = units['quality'][:]
    firing_rate = units['firing_rate'][:]
    print(f"Number of units: {num_units}")
    # Get count of good units
    good_mask = np.array([q == b'good' for q in quality])
    print(f"Number of good units: {np.sum(good_mask)}")

# Compute pre and post stimulus window for alignment
pre_stim = 0.003  # 3ms before stimulus
post_stim = block_duration * 2  # 2x the stimulus duration

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

# Function to implement a simple metric for stimulus responsiveness
def compute_responsiveness(aligned_spikes, pre_stim, stim_duration):
    # Count spikes in pre-stimulus window vs during stimulus
    pre_count = 0
    stim_count = 0
    
    for spikes in aligned_spikes:
        pre_mask = (spikes >= -pre_stim) & (spikes < 0)
        stim_mask = (spikes >= 0) & (spikes < stim_duration)
        
        pre_count += np.sum(pre_mask)
        stim_count += np.sum(stim_mask)
    
    # If there are no pre-stimulus spikes, use a small value to avoid division by zero
    pre_rate = pre_count / pre_stim if pre_count > 0 else 0.1
    stim_rate = stim_count / stim_duration
    
    # Simple ratio of firing rates
    ratio = stim_rate / pre_rate if pre_rate > 0 else stim_rate
    
    return ratio, stim_count

# Try to find responsive units - only check a subset to save time
print("Searching for responsive units...")
responsive_units = []

# Limit to 300 good units with firing rates in range 1-30 Hz for efficiency
good_unit_indices = np.where((good_mask) & (firing_rate >= 1) & (firing_rate <= 30))[0][:300]
print(f"Checking {len(good_unit_indices)} candidate units...")

for i, unit_idx in enumerate(good_unit_indices):
    if i % 20 == 0:  # Progress update
        print(f"Checked {i}/{len(good_unit_indices)} units...")
    
    # Get spike times for this unit
    index_start = spike_times_index[unit_idx]
    if unit_idx < num_units - 1:
        index_end = spike_times_index[unit_idx + 1]
    else:
        index_end = len(spike_times)
    
    unit_spike_times = spike_times[index_start:index_end]
    
    # Test responsiveness on a subset of stimuli
    test_stims = stim_times_subset[:30]  # Try 30 presentations
    aligned_spikes = get_aligned_spikes(unit_spike_times, test_stims, pre_stim, post_stim)
    
    # Compute responsiveness
    resp_ratio, stim_count = compute_responsiveness(aligned_spikes, pre_stim, block_duration)
    
    # If unit is responsive, add it to our list
    if resp_ratio > 2 and stim_count > 10:  # At least 2x increase and 10 spikes
        responsive_units.append((unit_idx, resp_ratio, stim_count, unit_spike_times))

print(f"Found {len(responsive_units)} responsive units")

# Sort by responsiveness
responsive_units.sort(key=lambda x: x[1] * x[2], reverse=True)  # Sort by combo of ratio and count

# Take top units for visualization
top_units = responsive_units[:min(4, len(responsive_units))]

if top_units:
    # Create a figure to compare barcode patterns
    fig, axes = plt.subplots(len(top_units), 1, figsize=(12, 4*len(top_units)), sharex=True)
    
    # If only one unit, make sure axes is still indexable
    if len(top_units) == 1:
        axes = [axes]
    
    for i, (unit_idx, resp_ratio, stim_count, unit_spike_times) in enumerate(top_units):
        print(f"Creating barcode for unit {unit_idx} (responsiveness: {resp_ratio:.2f}x, spikes: {stim_count})")
        
        # Get all aligned spikes for presentations
        aligned_spikes = get_aligned_spikes(unit_spike_times, stim_times_subset, pre_stim, post_stim)
        
        # Create raster plot
        ax = axes[i]
        for j, spikes in enumerate(aligned_spikes):
            if len(spikes) > 0:
                ax.scatter(spikes, np.ones_like(spikes) * j, marker='|', s=10, color='black')
        
        # Add stimulus markers
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
        ax.axvspan(0, block_duration, color='lightgray', alpha=0.3, label='Stimulus Duration')
        
        # Label the plot
        ax.set_ylabel(f'Unit {unit_idx}\nTrial #')
        ax.set_title(f'Unit {unit_idx} Barcode Pattern (Resp: {resp_ratio:.1f}x)')
        ax.set_ylim(-1, stim_subset_size + 1)
        
        # Only show legend for first subplot
        if i == 0:
            ax.legend(loc='upper right')
    
    # Set common x-axis label
    axes[-1].set_xlabel('Time from Stimulus Onset (s)')
    axes[-1].set_xlim(-pre_stim - 0.001, post_stim + 0.001)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig('explore/barcode_comparison.png')
    print("Saved barcode comparison plot to explore/barcode_comparison.png")
    
    # Create correlation plot to compare barcode patterns between neurons
    if len(top_units) > 1:
        print("\nComputing correlation between barcode patterns...")
        
        # Create PSTHs for each unit to compare
        bin_width = 0.001  # 1ms bins
        bins = np.arange(-pre_stim, post_stim + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Compute PSTH for each unit
        psth_data = []
        for unit_idx, resp_ratio, stim_count, unit_spike_times in top_units:
            aligned_spikes = get_aligned_spikes(unit_spike_times, stim_times_subset, pre_stim, post_stim)
            all_spikes = np.concatenate(aligned_spikes)
            hist, _ = np.histogram(all_spikes, bins=bins)
            # Normalize by number of presentations and bin width to get firing rate
            firing_rate = hist / (len(stim_times_subset) * bin_width)
            psth_data.append(firing_rate)
        
        # Compute correlation matrix between PSTHs
        corr_matrix = np.corrcoef(psth_data)
        
        # Visualization of PSTH correlation
        plt.figure(figsize=(8, 8))
        plt.imshow(corr_matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title('Correlation Between Barcode Patterns')
        
        unit_labels = [f'Unit {unit[0]}' for unit in top_units]
        plt.xticks(np.arange(len(unit_labels)), unit_labels, rotation=45)
        plt.yticks(np.arange(len(unit_labels)), unit_labels)
        
        plt.tight_layout()
        plt.savefig('explore/barcode_correlation.png')
        print("Saved barcode correlation matrix to explore/barcode_correlation.png")
        
        # Also plot the PSTHs together for comparison
        plt.figure(figsize=(12, 6))
        for i, fr in enumerate(psth_data):
            plt.plot(bin_centers, fr, label=f'Unit {top_units[i][0]}')
        
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
        plt.axvspan(0, block_duration, color='lightgray', alpha=0.3)
        plt.xlabel('Time from Stimulus Onset (s)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title('Comparison of PSTHs')
        plt.legend()
        plt.tight_layout()
        plt.savefig('explore/psth_comparison.png')
        print("Saved PSTH comparison plot to explore/psth_comparison.png")
else:
    print("No responsive units found for comparison")