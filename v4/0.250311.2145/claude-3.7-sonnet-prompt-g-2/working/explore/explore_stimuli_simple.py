# This script provides a simple overview of the stimulus information
# without attempting to extract large dataframes

import pynwb
import h5py
import remfile
import numpy as np
import time

print("Starting simplified stimulus analysis...")
start_time = time.time()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"
print(f"Loading file from {url}...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
print(f"File loaded in {time.time() - start_time:.2f} seconds")

# Function to get basic info without full dataframe extraction
def get_stimulus_info(stimulus_presentations, name):
    """Get basic info about the stimulus presentations without extracting full dataframes"""
    print(f"\n{name} Summary:")
    num_presentations = len(stimulus_presentations.id[:])
    print(f"Number of presentations: {num_presentations}")
    
    # Get column names 
    print(f"Available fields: {stimulus_presentations.colnames}")
    
    # Get sample start/stop times (just first 10)
    start_times = stimulus_presentations['start_time'][:10]
    stop_times = stimulus_presentations['stop_time'][:10]
    
    print(f"Sample start times: {start_times}")
    print(f"Sample stop times: {stop_times}")
    
    # Calculate average duration
    durations = stop_times - start_times
    avg_duration = np.mean(durations)
    print(f"Average stimulus duration (from sample): {avg_duration:.6f} seconds")
    
    # Try to get unique stimulus names if available
    try:
        stim_names = stimulus_presentations['stimulus_name'][:20]  # Just get first 20
        unique_names = np.unique(stim_names)
        print(f"Sample stimulus names: {unique_names}")
    except Exception as e:
        print(f"Could not get stimulus names: {e}")
    
    # Try to get contrast information if available
    try:
        if 'contrast' in stimulus_presentations.colnames:
            contrasts = stimulus_presentations['contrast'][:20]  # Just get first 20
            unique_contrasts = np.unique(contrasts)
            print(f"Sample contrast values: {unique_contrasts}")
    except Exception as e:
        print(f"Could not get contrast information: {e}")
        
    return num_presentations

# Get stimulus presentation counts
print("\nExamining stimulus intervals...")
all_intervals = nwb.intervals

# Print all available interval types
print(f"Available interval types: {list(all_intervals.keys())}")

# RepeatFFF (Repeated Full Field Flicker)
repeat_fff_count = get_stimulus_info(all_intervals['RepeatFFF_presentations'], "RepeatFFF (Repeated Full Field Flicker)")

# UniqueFFF (Unique Full Field Flicker)
unique_fff_count = get_stimulus_info(all_intervals['UniqueFFF_presentations'], "UniqueFFF (Unique Full Field Flicker)")

# Static gratings
static_block_count = get_stimulus_info(all_intervals['static_block_presentations'], "Static Block (Gratings)")

# Receptive field mapping
rf_block_count = get_stimulus_info(all_intervals['receptive_field_block_presentations'], "Receptive Field Block")

# Summarize stimulus totals
print("\nSummary of stimulus presentations:")
print(f"RepeatFFF presentations: {repeat_fff_count}")
print(f"UniqueFFF presentations: {unique_fff_count}")
print(f"Static Block presentations: {static_block_count}")
print(f"RF Block presentations: {rf_block_count}")

# Get some units information for reference
print("\nReference unit information:")
unit_ids = nwb.units.id[:]
print(f"Total number of units: {len(unit_ids)}")

# Count good quality units
try:
    quality = nwb.units['quality'][:]
    good_count = np.sum(np.array(quality) == 'good')
    print(f"Number of good quality units: {good_count}")
except Exception as e:
    print(f"Could not count good quality units: {e}")

print(f"Script completed in {time.time() - start_time:.2f} seconds")