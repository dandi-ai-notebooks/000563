# explore/plot_spike_raster.py
# This script loads spike times for a few units from the NWB file
# and creates a raster plot for a specific time interval.

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
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access units table
units_df = nwb.units.to_dataframe()

# Select a few units for plotting (e.g., first 5 units)
num_units_to_plot = 5
unit_ids_to_plot = units_df.index[:num_units_to_plot]
print(f"Selected unit IDs for raster plot: {unit_ids_to_plot.tolist()}")

# Define time interval for plotting
start_time = 50.0
end_time = 60.0  # Plot 10 seconds of data

# Extract spike times for selected units within the interval
spike_times_list = []
valid_unit_indices = []
for i, unit_id in enumerate(unit_ids_to_plot):
    # Get all spike times for the unit
    all_spike_times = nwb.units['spike_times'][unit_id]
    # Filter spike times within the desired interval
    interval_spike_times = all_spike_times[(all_spike_times >= start_time) & (all_spike_times < end_time)]
    if len(interval_spike_times) > 0:
        spike_times_list.append(interval_spike_times)
        valid_unit_indices.append(i) # Keep track of units that had spikes in the interval

# Prepare data for eventplot
plot_unit_ids = unit_ids_to_plot[valid_unit_indices]
colors = plt.cm.viridis(np.linspace(0, 1, len(plot_unit_ids)))

# Create raster plot
plt.figure(figsize=(12, 6))
if not spike_times_list:
    print(f"No spikes found for selected units in the interval {start_time}-{end_time}s.")
    plt.text(0.5, 0.5, 'No spikes in interval', horizontalalignment='center', verticalalignment='center')
else:
    plt.eventplot(spike_times_list, colors=colors, lineoffsets=np.arange(len(plot_unit_ids)), linelengths=0.8)
    plt.yticks(np.arange(len(plot_unit_ids)), plot_unit_ids)
    plt.ylabel("Unit ID")

plt.xlabel("Time (s)")
plt.title(f"Spike Raster Plot ({start_time}s - {end_time}s)")
plt.grid(True, axis='x')
plt.xlim(start_time, end_time)
plt.savefig("explore/spike_raster_plot.png")
plt.close()

print(f"Spike raster plot saved to explore/spike_raster_plot.png")

io.close()