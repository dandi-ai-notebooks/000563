"""
This script explores the overall structure of Dandiset 000563 (Allen Institute Openscope - Barcoding)
to understand what data is available and how it's organized.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dandi.dandiapi import DandiAPIClient
import h5py
import remfile
import pynwb

# Connect to DANDI archive
print("Connecting to DANDI archive...")
client = DandiAPIClient()
dandiset = client.get_dandiset("000563", "0.250311.2145")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Description: {metadata['description'][:500]}...")  # Truncate for brevity

# Get the assets
print("\nGetting assets...")
assets = list(dandiset.get_assets())
print(f"Total number of assets: {len(assets)}")

# Group assets by subject and type
subjects = {}
for asset in assets:
    path_parts = asset.path.split('/')
    if len(path_parts) >= 2:
        subject = path_parts[0]
        if subject not in subjects:
            subjects[subject] = {"ogen": None, "probes": []}
        
        if "ogen.nwb" in asset.path:
            subjects[subject]["ogen"] = asset
        elif "ecephys.nwb" in asset.path:
            subjects[subject]["probes"].append(asset)

# Print summary of subjects and assets
print(f"\nNumber of subjects: {len(subjects)}")
for subject, data in subjects.items():
    num_probes = len(data["probes"])
    print(f"{subject}: {num_probes} probes + {'1 ogen file' if data['ogen'] else 'no ogen file'}")

# Choose one subject for further exploration
subject_to_explore = list(subjects.keys())[0]
print(f"\nExploring subject {subject_to_explore}...")
selected_ogen = subjects[subject_to_explore]["ogen"]

# Load the ogen file
print(f"Loading ogen file: {selected_ogen.path}")
url = f"https://api.dandiarchive.org/api/assets/{selected_ogen.identifier}/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Explore stimulus presentations
print("\nExploring stimulus presentations:")
for interval_name, interval in nwb.intervals.items():
    if "presentations" in interval_name:
        try:
            df = interval.to_dataframe().head(2)
            num_presentations = len(interval.id.data)
            print(f"  {interval_name}: {num_presentations} presentations")
            print(f"  First few columns: {', '.join(df.columns[:5])}")
        except Exception as e:
            print(f"  Error with {interval_name}: {str(e)}")

# Look at units data
print("\nExploring units data:")
try:
    units_df = nwb.units.to_dataframe()
    print(f"  Number of units: {len(units_df)}")
    print(f"  Unit properties: {', '.join(units_df.columns[:10])}...")
    # Count units by quality
    if 'quality' in units_df.columns:
        quality_counts = units_df['quality'].value_counts()
        print("  Units by quality:")
        for q, count in quality_counts.items():
            print(f"    {q}: {count}")
except Exception as e:
    print(f"  Error accessing units: {str(e)}")

# Save a figure showing basic info about the dataset
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.9, f"Dandiset: {metadata['name']}", ha='center', fontsize=16, transform=plt.gca().transAxes)
plt.text(0.5, 0.8, f"Total subjects: {len(subjects)}", ha='center', fontsize=14, transform=plt.gca().transAxes)

# Plot number of probes per subject
subject_names = list(subjects.keys())
probe_counts = [len(subjects[s]["probes"]) for s in subject_names]

plt.bar(subject_names, probe_counts)
plt.ylabel("Number of probes")
plt.title("Probes per subject")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('explore/dataset_summary.png')
print("\nSaved dataset summary figure to explore/dataset_summary.png")