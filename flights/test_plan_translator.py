import csv
import json
import os

POSTION_NN_DIR = "../position_nn"


def expand_values(value, duration):
    """Expands a single value into a list repeated for the given duration."""
    return [value] * duration


def convert_csv_to_json(input_csv, output_json):
    result = {"ud": [], "rt": [], "fb": [], "lr": []}

    try:
        with open(input_csv, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                duration = int(row["duration"])

                result["ud"].extend(expand_values(int(row["potX"]), duration))
                result["rt"].extend(expand_values(int(row["potRot"]), duration))
                result["fb"].extend(expand_values(int(row["potY"]), duration))
                result["lr"].extend(expand_values(int(row["potZ"]), duration))

        with open(output_json, "w") as json_file:
            json.dump(result, json_file, indent=4)

        print(f"Converted {input_csv} to {output_json}")

    except FileNotFoundError:
        print(f"Error: Input file {input_csv} not found.")
    except KeyError as e:
        print(f"Error: Missing column in CSV file: {e}")
    except ValueError:
        print("Error: Invalid value in the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


os.makedirs(POSTION_NN_DIR, exist_ok=True)

input_csv = os.path.join(POSTION_NN_DIR, "simulation.csv")
output_json = os.path.join("plans", "simulation_input.json")

if not os.path.exists(output_json):
    with open(output_json, "w") as f:
        pass

convert_csv_to_json(input_csv, output_json)
