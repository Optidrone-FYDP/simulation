import pandas as pd
import glob
import os

DATA_PATH = "old_data"
OUTPUT_PATH = "raw_data"

def extract_pot_values(filename):
    parts = os.path.basename(filename).split('-')
    potX = int(parts[3][1:])  # extracts x064 -> 64
    potY = int(parts[4][1:])
    potZ = int(parts[5][1:])
    return potX, potY, potZ

for file in glob.glob(os.path.join(DATA_PATH, "*.csv")):
    potX, potY, potZ = extract_pot_values(file)
    df = pd.read_csv(file, skiprows=[0, 1, 2, 4])  # non-data, non-header rows
    df = df[['Frame', 'Sub Frame', 'RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']]
    
    df['potX'] = potX
    df['potY'] = potY
    df['potZ'] = potZ
    
    df = df[['Frame', 'Sub Frame', 'RX', 'RY', 'RZ', 'TX', 'TY', 'TZ', 'potX', 'potY', 'potZ']]
    
    parts = os.path.basename(file).split('-')
    new_filename = '-'.join(parts[:3] + parts[6:])
    df.to_csv(os.path.join(OUTPUT_PATH, new_filename), index=False, header=False)
    print(f"updated {os.path.basename(file)} with potX={potX}, potY={potY}, potZ={potZ}")