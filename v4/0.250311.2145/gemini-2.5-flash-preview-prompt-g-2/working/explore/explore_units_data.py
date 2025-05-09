# This script explores the units data in the NWB file.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/1f158fe0-f8ef-495e-b031-da25316a335c/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access units data
units = nwb.units

# Print information about the units table
print("Units table columns:", units.colnames)
print("Number of units:", len(units))

# Get a sample of unit spike times (e.g., for the first 5 units)
num_units_to_show = min(5, len(units))
for i in range(num_units_to_show):
    unit_id = units.id[i]
    spike_times = units['spike_times'][i][:] # Load all spike times for this unit
    print(f"\nUnit {unit_id} has {len(spike_times)} spikes.")
    if len(spike_times) > 10:
        print(f"First 10 spike times: {spike_times[:10]}")
    else:
        print(f"Spike times: {spike_times}")

# Plot a raster plot for a few units (if available)
if len(units) > 0:
    plt.figure(figsize=(12, 6))
    unit_indices_to_plot = range(min(10, len(units))) # Plot up to 10 units
    for i in unit_indices_to_plot:
        unit_id = units.id[i]
        spike_times = units['spike_times'][i][:]
        plt.vlines(spike_times, i + 0.5, i + 1.5, lw=0.5)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Unit ID")
    plt.title("Raster plot for a subset of units")
    plt.yticks(np.arange(len(unit_indices_to_plot)) + 1, [units.id[i] for i in unit_indices_to_plot])
    plt.ylim(0.5, len(unit_indices_to_plot) + 0.5)
    plt.savefig('explore/units_raster_plot.png')
    plt.close()
    print("Units raster plot generated: explore/units_raster_plot.png")
else:
    print("No units data found in this NWB file.")