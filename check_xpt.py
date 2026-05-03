"""
check_xpt.py — inspect a SAS XPT file

Usage
-----
# Auto mode — picks first .xpt from Outputs/ (errors if none found)
python check_xpt.py

# Explicit path
python check_xpt.py Outputs/myfile.xpt
"""
import sys
import os
import pandas as pd
from pathlib import Path

# Accept path as CLI arg, else search Outputs/, else fall back to current dir
if len(sys.argv) > 1:
    xpt_path = sys.argv[1]
else:
    outputs_dir = Path("Outputs")
    candidates = sorted(outputs_dir.glob("*.xpt")) if outputs_dir.exists() else []
    if candidates:
        xpt_path = str(candidates[0])
        print(f"Using: {xpt_path}")
    else:
        sys.exit("ERROR: No .xpt files found in Outputs/")

if not os.path.isfile(xpt_path):
    sys.exit(f"ERROR: File not found: {xpt_path}")

df = pd.read_sas(xpt_path, format="xport")
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nFirst rows:\n", df.head())
