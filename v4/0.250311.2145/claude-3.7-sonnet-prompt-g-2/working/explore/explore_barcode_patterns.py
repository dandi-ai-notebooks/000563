# This script examines the barcode patterns in neural responses
# It focuses on responses to repeated stimuli to see time-locked patterns

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import time

print("Starting barcode pattern analysis...")
start_time = time.time()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
print(f"Loading file from {url}...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
print(f"File loaded in {time.time() - start_time:.2f} seconds")

# Get the repeated Full Field Flicker stimulus presentations
print("Getting stimulus information...")
repeat_fff_presentations = nwb.intervals['RepeatFFF_presentations']
trial_starts = repeat_fff_presentations.start_time[:]
print(f"Number of RepeatFFF trials: {len(trial_starts)}")

# Function to create a raster plot for a single unit
def create_raster_plot(unit_id, spike_times, trial_starts, window_size=0.25, max_trials=50):
    """
    Create a raster plot showing spikes aligned to stimulus onset
    
    Args:
        unit_id: ID of the unit being plotted
        spike_times: Array of spike times for this unit
        trial_starts: Array of stimulus onset times
        window_size: Time window to plot after stimulus onset (in seconds)
        max_trials: Maximum number of trials to plot
    """
    trials_to_plot = min(max_trials, len(trial_starts))
    trial_indices = []
    spike_times_rel = []
    
    for i in range(trials_to_plot):
        start_time = trial_starts[i]
        end_time = start_time + window_size
        
        # Find spikes in this window
        mask = (spike_times >= start_time) & (spike_times < end_time)
        these_spikes = spike_times[mask]
        
        # Convert to relative time and add to lists
        if len(these_spikes) > 0:
            rel_times = these_spikes - start_time
            trial_indices.extend([i] * len(rel_times))
            spike_times_rel.extend(rel_times)
    
    # Plot
    plt.scatter(spike_times_rel, trial_indices, s=2, c='black')
    plt.xlim(0, window_size)
    plt.ylim(-1, trials_to_plot)
    plt.ylabel("Trial #")
    plt.title(f"Unit {unit_id} response to repeated stimuli")

# Find good units with high firing rates
print("Finding units with good quality and high firing rates...")
unit_ids = nwb.units.id[:]
firing_rates = nwb.units['firing_rate'][:]
quality = nwb.units['quality'][:]

# Get indices of good quality units
good_indices = np.where(np.array(quality) == 'good')[0]
print(f"Number of good quality units: {len(good_indices)}")

# Sort by firing rate
good_unit_indices = good_indices[np.argsort(-np.array(firing_rates)[good_indices])]
print(f"Top firing rates: {[firing_rates[i] for i in good_unit_indices[:5]]}")

# Select 6 units to examine
units_to_examine = good_unit_indices[:6]

# Create raster plots for selected units
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, unit_idx in enumerate(units_to_examine):
    unit_id = unit_ids[unit_idx]
    print(f"Processing unit {unit_id} (index {unit_idx})...")
    
    # Get spike times for this unit
    spike_times = nwb.units['spike_times'][unit_idx]
    
    # Plot in the corresponding subplot
    plt.sca(axes[i])
    create_raster_plot(unit_id, spike_times, trial_starts)
    
    # Add firing rate information
    fr = firing_rates[unit_idx]
    plt.xlabel(f"Time from stimulus onset (s) - FR: {fr:.2f} Hz")

plt.tight_layout()
plt.savefig('explore/barcode_raster_plots.png')
print(f"Saved raster plots to explore/barcode_raster_plots.png")

# Create plot showing repeated trials with same stimulus
print("Creating plot for repeated trials...")
plt.figure(figsize=(10, 6))

# Select just one good unit with high firing rate
selected_unit_idx = good_unit_indices[0]  # Highest firing rate unit
unit_id = unit_ids[selected_unit_idx]
spike_times = nwb.units['spike_times'][selected_unit_idx]
fr = firing_rates[selected_unit_idx]

# Plot raster for this unit with more detailed time window
window_size = 1.0  # 1 second after stimulus
max_trials = 20    # Show more trials

trial_indices = []
spike_times_rel = []

print(f"Creating detailed raster for unit {unit_id} with firing rate {fr:.2f} Hz...")
for i in range(max_trials):
    start_time = trial_starts[i]
    end_time = start_time + window_size
    
    # Find spikes in this window
    mask = (spike_times >= start_time) & (spike_times < end_time)
    these_spikes = spike_times[mask]
    
    # Convert to relative time and add to lists
    if len(these_spikes) > 0:
        rel_times = these_spikes - start_time
        trial_indices.extend([i] * len(rel_times))
        spike_times_rel.extend(rel_times)

# Plot
plt.scatter(spike_times_rel, trial_indices, s=4, c='black')
plt.xlim(0, window_size)
plt.ylim(-1, max_trials)
plt.ylabel("Trial #")
plt.xlabel("Time from stimulus onset (s)")
plt.title(f"Unit {unit_id} 'Barcode' Pattern - Firing Rate: {fr:.2f} Hz")
plt.grid(alpha=0.3)
plt.savefig('explore/detailed_barcode_pattern.png')
print(f"Saved detailed barcode pattern to explore/detailed_barcode_pattern.png")

print(f"Script completed in {time.time() - start_time:.2f} seconds")