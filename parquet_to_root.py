'''
Script to convert an apache arrow parquet file into a root tree.

Turns files created in the rust spsevb_cebra eventbuilder into root files (for people who like using root). 
This assumes that every columnn in the parquet file is the same length and an f64.

Run: python3 parquet_to_root.py FILENAME.parquet

Output: FILENAME.root
 
'''

import pyarrow.parquet as pq
import ROOT
import ctypes
import os
import sys
from tqdm import tqdm
import glob

# Ensure at least one file pattern is provided
if len(sys.argv) < 2:
    print("Usage: python3 parquet_to_root.py FILENAME_PATTERN.parquet")
    sys.exit(1)

# Get a list of all Parquet files matching the provided pattern
parquet_file_patterns = sys.argv[1:]
parquet_file_paths = []
for pattern in parquet_file_patterns:
    parquet_file_paths.extend(glob.glob(pattern))

# Loop over each Parquet file and convert it to ROOT
for parquet_file_path in parquet_file_paths:
    root_file_name = parquet_file_path.replace('.parquet', '.root')

    # Remove the previous ROOT file if it exists to avoid appending to it
    if os.path.exists(root_file_name):
        os.remove(root_file_name)

    # Read the Parquet file
    table = pq.read_table(parquet_file_path)

    # Create a ROOT file and a TTree
    root_file = ROOT.TFile(root_file_name, 'RECREATE')
    sps_tree = ROOT.TTree('SPSTree', 'SPSTree converted from Parquet file')

    # Dictionary for branch buffers
    branch_buffers = {}

    # Create branches
    for field in table.schema:
        branch_buffers[field.name] = ctypes.c_double()
        sps_tree.Branch(field.name, ctypes.addressof(branch_buffers[field.name]), f"{field.name}/D")

    # Fill the TTree with a progress bar
    num_rows = table.num_rows
    with tqdm(total=num_rows, desc=f"Converting {parquet_file_path}", unit="row") as pbar:
        for row in range(num_rows):
            for field in table.schema:
                branch_buffers[field.name].value = table.column(field.name)[row].as_py()
            sps_tree.Fill()
            pbar.update(1)

    # Write the TTree to the ROOT file and close
    sps_tree.Write()
    root_file.Close()

    print(f"Conversion complete! Output file for {parquet_file_path} is:", root_file_name)
