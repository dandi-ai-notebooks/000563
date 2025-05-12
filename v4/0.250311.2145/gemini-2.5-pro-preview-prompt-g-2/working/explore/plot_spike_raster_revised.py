# explore/plot_spike_raster_revised.py
# This script loads spike times for fewer units from the NWB file
# and creates a raster plot for a shorter time interval.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
# Increase cache size for potentially better performance with repeated access
# remote_file = remfile.File(url, _cache_size=50*1024*1024)
# h5_file = h5py.File(remote_file, rdcc_nbytes=50*1024*1024)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access units table and get IDs
all_unit_ids = nwb.units.id[:] # Get all unit IDs directly

# Select a few units for plotting (e.g., first 3 units)
num_units_to_plot = 3
if len(all_unit_ids) < num_units_to_plot:
    num_units_to_plot = len(all_unit_ids) # Adjust if fewer units exist

if num_units_to_plot > 0:
    unit_ids_to_plot = all_unit_ids[:num_units_to_plot]
    print(f"Selected unit IDs for raster plot: {unit_ids_to_plot.tolist()}")

    # Define time interval for plotting
    start_time = 50.0
    end_time = 55.0  # Plot 5 seconds of data

    # Extract spike times for selected units within the interval
    spike_times_list = []
    valid_unit_indices = []
    for i, unit_id in enumerate(unit_ids_to_plot):
        try:
            # Get all spike times for the unit
            # Accessing the spike_times VectorData directly
            spike_times_vd = nwb.units['spike_times']
            # Get the data corresponding to the index for this unit_id
            unit_idx = np.where(all_unit_ids == unit_id)[0][0]
            idx_start = spike_times_vd.data.ind[unit_idx]
            idx_end = spike_times_vd.data.ind[unit_idx+1]
            all_spike_times = spike_times_vd.data.data[idx_start:idx_end]

            # Filter spike times within the desired interval
            interval_spike_times = all_spike_times[(all_spike_times >= start_time) & (all_spike_times < end_time)]
            if len(interval_spike_times) > 0:
                spike_times_list.append(interval_spike_times)
                valid_unit_indices.append(i) # Keep track of units that had spikes
        except Exception as e:
            print(f"Error processing unit {unit_id}: {e}")
            continue # Skip this unit if there's an error


    # Prepare data for eventplot
    plot_unit_ids = unit_ids_to_plot[valid_unit_indices]
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_unit_ids)))

    # Create raster plot
    plt.figure(figsize=(12, 4)) # Adjust figure size potentially
    if not spike_times_list:
        print(f"No spikes found for selected units {unit_ids_to_plot.tolist()} in the interval {start_time}-{end_time}s.")
        plt.text(0.5, 0.5, 'No spikes in interval', horizontalalignment='center', verticalalignment='center')
    else:
        plt.eventplot(spike_times_list, colors=colors, lineoffsets=np.arange(len(plot_unit_ids)), linelengths=0.8)
        plt.yticks(np.arange(len(plot_unit_ids)), plot_unit_ids)
        plt.ylabel("Unit ID")

    plt.xlabel("Time (s)")
    plt.title(f"Spike Raster Plot ({start_time}s - {end_time}s)")
    plt.grid(True, axis='x')
    plt.xlim(start_time, end_time)
    plt.savefig("explore/spike_raster_plot.png") # Overwrite previous attempt
    plt.close()

    print(f"Spike raster plot saved to explore/spike_raster_plot.png")

else:
    print("No units found in the file.")

io.close()