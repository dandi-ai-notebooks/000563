# This script examines the units (neuron) data from the Dandiset
# We're interested in the spiking patterns that create the "barcode" patterns described in the dataset

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import islice

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Function to convert spike times to raster format
def get_raster_data(spike_times, trial_start_times, trial_window=1.0):
    """
    Convert spike times to raster format for visualization
    
    Args:
        spike_times: array of spike times for a unit
        trial_start_times: array of trial start times
        trial_window: duration of each trial in seconds
    
    Returns:
        trial_indices, spike_times_rel: Arrays for raster plot
    """
    trial_indices = []
    spike_times_rel = []
    
    for i, start_time in enumerate(trial_start_times):
        end_time = start_time + trial_window
        mask = (spike_times >= start_time) & (spike_times < end_time)
        if np.sum(mask) > 0:
            spikes_in_window = spike_times[mask]
            trial_indices.extend([i] * len(spikes_in_window))
            spike_times_rel.extend(spikes_in_window - start_time)
    
    return np.array(trial_indices), np.array(spike_times_rel)

# General info about units
print(f"Number of units: {len(nwb.units.id[:])}")
print(f"Unit columns: {nwb.units.colnames}")
print(f"Available regions: {np.unique(nwb.electrodes.location[:])}")

# Get stimulus information (needed to analyze barcoding)
# Get repeated trials for Full Field Flicker (RepeatFFF presentatinos)
repeat_fff_presentations = nwb.intervals['RepeatFFF_presentations']
repeat_fff_df = repeat_fff_presentations.to_dataframe().head(20)
print("\nRepeat FFF presentation example:")
print(repeat_fff_df[['start_time', 'stop_time', 'stimulus_name']])

# Get units info - focusing on high quality units
units_df = nwb.units.to_dataframe()
quality_units = units_df[units_df['quality'] == 'good']
print(f"\nNumber of good quality units: {len(quality_units)}")

# Get some example units to plot
sampled_units = quality_units.sample(n=min(5, len(quality_units)), random_state=42)

# Get trial start times for stimuli
trial_starts = repeat_fff_presentations.start_time[:][:100]  # First 100 trials

# Create a figure with raster plots for the sampled units
plt.figure(figsize=(15, 10))
for i, (unit_id, unit) in enumerate(sampled_units.iterrows()):
    plt.subplot(len(sampled_units), 1, i+1)
    
    # Get spike times for this unit
    spike_times = nwb.units['spike_times'][unit_id]
    
    # Convert to raster format
    trial_indices, spike_times_rel = get_raster_data(spike_times, trial_starts)
    
    # Plot raster
    plt.scatter(spike_times_rel, trial_indices, s=1, c='black')
    plt.ylabel(f"Unit {unit['cluster_id']}\nTrial")
    if i == len(sampled_units) - 1:
        plt.xlabel("Time from stimulus onset (s)")
    
    # Add unit info
    plt.title(f"Unit {unit['cluster_id']} - Firing rate: {unit['firing_rate']:.2f} Hz")
    
plt.tight_layout()
plt.savefig('explore/unit_rasters.png')

# Now let's analyze different brain regions
regions = np.unique(nwb.electrodes.location[:])
print("\nBrain regions in the dataset:", regions)

# Find good units from different brain regions
region_units = {}
for region in regions:
    # Get electrodes for this region
    region_electrodes = np.where(nwb.electrodes.location[:] == region)[0]
    
    # Find units from these electrodes
    region_unit_ids = []
    for unit_id, unit in quality_units.iterrows():
        if unit['peak_channel_id'] in region_electrodes:
            region_unit_ids.append(unit_id)
    
    region_units[region] = region_unit_ids
    print(f"Region {region}: {len(region_unit_ids)} good units")

# Plot example units from different regions
plt.figure(figsize=(15, 15))
plot_count = 1
max_regions = min(5, len(regions))

for i, region in enumerate(list(regions)[:max_regions]):
    unit_ids = region_units[region]
    if not unit_ids:
        continue  # Skip if no units in this region
        
    # Sample up to 2 units from each region
    sample_size = min(2, len(unit_ids))
    sampled_ids = np.random.choice(unit_ids, sample_size, replace=False)
    
    for unit_id in sampled_ids:
        plt.subplot(max_regions*2, 1, plot_count)
        
        # Get spike times
        spike_times = nwb.units['spike_times'][unit_id]
        
        # Convert to raster format
        trial_indices, spike_times_rel = get_raster_data(spike_times, trial_starts)
        
        # Plot raster
        plt.scatter(spike_times_rel, trial_indices, s=1, c='black')
        plt.ylabel(f"{region}\nUnit {units_df.loc[unit_id]['cluster_id']}\nTrial")
        
        # Add unit info
        plt.title(f"Region: {region} - Unit {units_df.loc[unit_id]['cluster_id']} - FR: {units_df.loc[unit_id]['firing_rate']:.2f} Hz")
        
        plot_count += 1

plt.tight_layout()
plt.savefig('explore/region_unit_rasters.png')

# Now let's examine the PSTH (Peri-Stimulus Time Histogram) for a few units
plt.figure(figsize=(15, 10))
for i, (unit_id, unit) in enumerate(sampled_units.iterrows()):
    plt.subplot(len(sampled_units), 1, i+1)
    
    # Get spike times
    spike_times = nwb.units['spike_times'][unit_id]
    
    # Calculate PSTH
    all_spike_times_rel = []
    for start_time in trial_starts:
        end_time = start_time + 1.0  # 1 second window
        mask = (spike_times >= start_time) & (spike_times < end_time)
        if np.sum(mask) > 0:
            spikes_in_window = spike_times[mask] - start_time
            all_spike_times_rel.extend(spikes_in_window)
    
    # Create histogram
    bins = np.linspace(0, 1, 101)  # 100 bins for 1 second
    hist, _ = np.histogram(all_spike_times_rel, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot
    plt.bar(bin_centers, hist, width=bins[1]-bins[0], alpha=0.7)
    plt.title(f"Unit {unit['cluster_id']} - PSTH")
    plt.ylabel("Spike count")
    
    if i == len(sampled_units) - 1:
        plt.xlabel("Time from stimulus onset (s)")

plt.tight_layout()
plt.savefig('explore/unit_psth.png')