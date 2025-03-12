import pandas as pd

def round_to_nearest(value, rounding_values):
    return min(rounding_values, key=lambda x: abs(x - value))

def round_csv_values(input_file, output_file, rounding_values):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Apply rounding to numerical columns
    df = df.apply(lambda x: x.map(lambda y: round_to_nearest(y, rounding_values) if isinstance(y, (int, float)) else y))

    # Save to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Rounded values saved to {output_file}")

# Example usage
input_csv = "output_path_plan1.csv"  # Change this to your file
output_csv = "output_path_plan1_ideal.csv"  # Change this to your desired output file
rounding_values = [0, -950, 1200, 1600, -1600]  # Define the discrete rounding values
round_csv_values(input_csv, output_csv, rounding_values)