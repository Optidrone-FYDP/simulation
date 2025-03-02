import pandas as pd

# Replace 'data.csv' with your input CSV file name
df = pd.read_csv("processed_data_v2/test_file_1.csv")

if "duration" not in df.columns:
    df["duration"] = 1
else:
    df["duration"] = df["duration"].fillna(1)

# Select only the potentiometer input columns along with duration.
# Based on your CSV header, these are: potZ, potRot, potY, potX, and duration.
pot_inputs = df[["potZ", "potRot", "potY", "potX", "duration"]]

# Write the selected columns to a new CSV file
pot_inputs.to_csv("simA.csv", index=False)

print("Potentiometer inputs saved to pot_inputs.csv")
