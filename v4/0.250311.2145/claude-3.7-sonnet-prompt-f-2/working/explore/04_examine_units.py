"""
Script to examine the units (neural responses) in the dataset.
This script focuses on looking at unit properties and spike times.
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

# Examine the units structure
print("\nExamining units structure...")
if 'units' in h5_file:
    units = h5_file['units']
    print("Units attributes:")
    for attr in units.attrs:
        print(f"  {attr}: {units.attrs[attr]}")
    
    print("\nUnit datasets:")
    for key in units.keys():
        if isinstance(units[key], h5py.Dataset):
            dataset = units[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
    
    # Get some basic statistics about the units
    if 'quality' in units:
        # Get quality values
        qualities = units['quality'][:]
        unique_qualities, counts = np.unique(qualities, return_counts=True)
        print("\nUnit quality counts:")
        for quality, count in zip(unique_qualities, counts):
            print(f"  {quality}: {count}")
    
    if 'firing_rate' in units:
        # Get firing rate statistics
        firing_rates = units['firing_rate'][:]
        print("\nFiring rate statistics:")
        print(f"  Mean: {np.mean(firing_rates):.2f} Hz")
        print(f"  Median: {np.median(firing_rates):.2f} Hz")
        print(f"  Min: {np.min(firing_rates):.2f} Hz")
        print(f"  Max: {np.max(firing_rates):.2f} Hz")
        
        # Plot firing rate distribution
        plt.figure(figsize=(10, 6))
        plt.hist(firing_rates, bins=50)
        plt.xlabel('Firing Rate (Hz)')
        plt.ylabel('Number of Units')
        plt.title('Distribution of Firing Rates')
        plt.savefig('explore/firing_rate_distribution.png')
        print("\nSaved firing rate distribution to explore/firing_rate_distribution.png")

    # Look at spike times for a few units
    if 'spike_times_index' in units and 'spike_times' in units:
        print("\nExamining spike times for a few units...")
        # Look at the first 5 units
        for unit_idx in range(min(5, units['id'].shape[0])):
            # Get spike times for this unit
            index = units['spike_times_index'][unit_idx]
            if isinstance(index, (tuple, list)) and len(index) == 2:
                start_idx, count = index
                if count > 0:
                    # Retrieve the spike times
                    spike_times = units['spike_times'][start_idx:start_idx+count]
                    print(f"  Unit {unit_idx} has {count} spikes, first 5 spike times: {spike_times[:5]}")
                    
                    # Plot spike times for one example unit
                    if unit_idx == 0:
                        plt.figure(figsize=(12, 4))
                        # Let's look at just the first 60 seconds of data
                        mask = spike_times < 60  # First 60 seconds
                        plt.plot(spike_times[mask], np.ones_like(spike_times[mask]), '|', markersize=10)
                        plt.xlabel('Time (s)')
                        plt.ylabel('Spikes')
                        plt.title(f'Spike Times for Unit {unit_idx} (First 60 seconds)')
                        plt.grid(True, alpha=0.3)
                        plt.savefig('explore/unit_spike_times.png')
                        print(f"  Saved spike time plot for unit {unit_idx} to explore/unit_spike_times.png")
            
    # Examine if we can see the "barcode" pattern in the neural responses
    # To do this, we need to look at spike responses aligned to stimulus presentations
    if 'intervals' in h5_file and 'RepeatFFF_presentations' in h5_file['intervals']:
        print("\nLooking for barcode patterns in response to RepeatFFF stimulus...")
        repeat_fff = h5_file['intervals']['RepeatFFF_presentations']
        
        # Get stimulus times for RepeatFFF
        stim_start_times = repeat_fff['start_time'][:]
        stim_stop_times = repeat_fff['stop_time'][:]
        
        # Get stimulus blocks
        stim_blocks = repeat_fff['stimulus_block'][:] if 'stimulus_block' in repeat_fff else None
        
        # Find a block of repeated stimuli to analyze
        if stim_blocks is not None:
            unique_blocks = np.unique(stim_blocks)
            if len(unique_blocks) > 0:
                # Get start and stop times for the first block
                block = unique_blocks[0]
                block_mask = stim_blocks == block
                block_start_times = stim_start_times[block_mask]
                
                if len(block_start_times) > 0:
                    print(f"  Found block {block} with {len(block_start_times)} stimulus presentations")
                    print(f"  First 5 stimulus times in this block: {block_start_times[:5]}")
                    
                    # Plot the stimulus times for this block
                    plt.figure(figsize=(12, 3))
                    for i, start_time in enumerate(block_start_times[:50]):  # Plot first 50
                        plt.axvline(start_time, color='gray', alpha=0.5)
                    plt.xlim(block_start_times[0] - 1, block_start_times[49] + 1)
                    plt.xlabel('Time (s)')
                    plt.title(f'Stimulus Times for Block {block}')
                    plt.grid(True, alpha=0.3)
                    plt.savefig('explore/stimulus_times.png')
                    print(f"  Saved stimulus times plot to explore/stimulus_times.png")
else:
    print("No units found in the file")