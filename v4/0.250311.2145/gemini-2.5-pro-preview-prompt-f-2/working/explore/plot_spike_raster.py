# Objective: Plot a spike raster for a few selected units.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()
print("NWB file loaded.")

# Access units table
units_df = nwb.units.to_dataframe()
print(f"Units table loaded with {len(units_df)} units.")

# Select a few units to plot (e.g., first 5 units that have a reasonable number of spikes)
# We'll also get their actual IDs for y-axis labeling
selected_spike_times = []
selected_unit_ids = []
num_units_to_plot = 0
max_units_to_try = min(20, len(units_df)) # Try up to 20 units to find 5 good ones
target_num_units = 5

for i in range(max_units_to_try):
    if num_units_to_plot >= target_num_units:
        break
    try:
        # Get spike times for the i-th unit in the original table
        # The index for spike_times_index corresponds to the row in the Units table
        unit_spike_times = nwb.units.spike_times_index[i][:]
        unit_id = nwb.units.id[i] # Get the actual unit ID
        if len(unit_spike_times) > 10: # Only plot units with at least 10 spikes
            selected_spike_times.append(unit_spike_times)
            selected_unit_ids.append(str(unit_id)) # Use string for categorical plotting
            num_units_to_plot += 1
            print(f"Selected unit ID {unit_id} (index {i}) with {len(unit_spike_times)} spikes.")
    except IndexError:
        print(f"Could not access spike times for unit at index {i}. Skipping.")
        continue # Should not happen if iterating based on len(units_df)
    except Exception as e:
        print(f"Error processing unit at index {i} (ID: {nwb.units.id[i]}): {e}")
        continue


if not selected_spike_times:
    print("No units with sufficient spikes found to plot.")
else:
    print(f"Plotting raster for {len(selected_unit_ids)} units: {selected_unit_ids}")
    # Create raster plot
    plt.figure(figsize=(12, 6))
    # Eventplot expects a list of lists/arrays, where each inner list/array contains the event times for one series.
    plt.eventplot(selected_spike_times, linelengths=0.75, colors='black')
    plt.yticks(np.arange(len(selected_unit_ids)), selected_unit_ids) # Use actual unit IDs
    plt.xlabel("Time (s)")
    plt.ylabel("Unit ID")
    plt.title(f"Spike Raster for Selected Units")

    # Zoom in on a time segment if the overall duration is too long
    # For example, if max spike time > 60s, plot first 60s.
    max_time = 0
    for st in selected_spike_times:
        if len(st) > 0:
            max_time = max(max_time, np.max(st))
    
    if max_time > 60:
        plt.xlim(0, 60)
        plt.title(f"Spike Raster for Selected Units (First 60s)")
        print("Zoomed into first 60s of spike raster.")
    
    plt.grid(True, axis='x', linestyle=':')
    plt.tight_layout()

    # Save plot
    output_path = "explore/spike_raster_segment.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

io.close()
print("Script finished.")