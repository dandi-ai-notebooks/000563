"""
This script explores the structure of a specific NWB file from Dandiset 000563 to understand
what data is available and how it's organized.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import remfile
import pynwb

# URL for the ogen.nwb file for subject 681446
url = "https://api.dandiarchive.org/api/assets/2f2ac304-83a3-4352-8612-5f34b68062a0/download/"

print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the NWB file
print("\nNWB File Information:")
print(f"Session ID: {nwb.session_id}")
print(f"Institution: {nwb.institution}")
print(f"Stimulus notes: {nwb.stimulus_notes}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")

# Subject information
print("\nSubject Information:")
subject = nwb.subject
print(f"Species: {subject.species}")
print(f"Subject ID: {subject.subject_id}")
print(f"Age: {subject.age}")
print(f"Sex: {subject.sex}")
print(f"Genotype: {subject.genotype}")
print(f"Strain: {subject.strain}")

# Explore stimulus presentations
print("\nStimulus Presentations:")
stimulus_intervals = {}
for interval_name, interval in nwb.intervals.items():
    if "presentations" in interval_name:
        try:
            num_presentations = len(interval.id.data)
            stimulus_intervals[interval_name] = num_presentations
            print(f"  {interval_name}: {num_presentations} presentations")
        except Exception as e:
            print(f"  Error with {interval_name}: {str(e)}")

# Get more details about RepeatFFF_presentations
print("\nDetails about RepeatFFF_presentations:")
if "RepeatFFF_presentations" in nwb.intervals:
    repeat_fff = nwb.intervals["RepeatFFF_presentations"]
    try:
        # Get a sample of the data
        df = repeat_fff.to_dataframe().head(5)
        print(f"Columns: {df.columns.tolist()}")
        print("\nSample data:")
        print(df[["start_time", "stop_time", "stimulus_name", "contrast", "stimulus_block", "index_repeat"]].head())
    except Exception as e:
        print(f"Error: {str(e)}")

# Explore units data
print("\nUnits Information:")
try:
    units_df = nwb.units.to_dataframe()
    print(f"Number of units: {len(units_df)}")
    print(f"Units properties: {', '.join(units_df.columns[:10])}...")
    
    # Get count of units by quality if available
    if 'quality' in units_df.columns:
        quality_counts = units_df['quality'].value_counts()
        print("\nUnits by quality:")
        for q, count in quality_counts.items():
            print(f"  {q}: {count}")
    
    # Get some basic statistics on firing rate
    if 'firing_rate' in units_df.columns:
        print("\nFiring rate statistics:")
        print(f"  Mean: {units_df['firing_rate'].mean():.2f} Hz")
        print(f"  Median: {units_df['firing_rate'].median():.2f} Hz")
        print(f"  Min: {units_df['firing_rate'].min():.2f} Hz")
        print(f"  Max: {units_df['firing_rate'].max():.2f} Hz")
except Exception as e:
    print(f"Error accessing units: {str(e)}")

# Create a histogram of firing rates
try:
    if 'firing_rate' in units_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(units_df['firing_rate'], bins=50)
        plt.xlabel('Firing Rate (Hz)')
        plt.ylabel('Number of Units')
        plt.title('Distribution of Firing Rates')
        plt.savefig('explore/firing_rate_distribution.png')
        print("\nSaved firing rate distribution to explore/firing_rate_distribution.png")
except Exception as e:
    print(f"Error creating firing rate histogram: {str(e)}")

# Create a pie chart of unit quality
try:
    if 'quality' in units_df.columns:
        plt.figure(figsize=(8, 8))
        units_df['quality'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Units by Quality')
        plt.savefig('explore/units_by_quality.png')
        print("Saved units by quality pie chart to explore/units_by_quality.png")
except Exception as e:
    print(f"Error creating quality pie chart: {str(e)}")

# Plot number of presentations per stimulus type
plt.figure(figsize=(10, 6))
plt.bar(stimulus_intervals.keys(), stimulus_intervals.values())
plt.ylabel("Number of Presentations")
plt.title("Stimulus Presentations")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('explore/stimulus_presentations.png')
print("Saved stimulus presentations plot to explore/stimulus_presentations.png")