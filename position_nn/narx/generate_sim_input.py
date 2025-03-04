import csv
import argparse

DEFAULT_POT_ROT = 64


def pot_values_match(val1, val2, tol):
    return all(abs(a - b) <= tol for a, b in zip(val1, val2))


def compress_pot_values(input_filename, output_filename, frame_duration, tolerance):
    groups = []
    prev_vals = None
    count = 0

    with open(input_filename, newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                potX = int(round(float(row["potX"])))
                potY = int(round(float(row["potY"])))
                potZ = int(round(float(row["potZ"])))
            except KeyError:
                raise ValueError("Input CSV must have columns: potX, potY, potZ")

            current_vals = (potX, potY, potZ)

            if prev_vals is None:
                prev_vals = current_vals
                count = 1
            elif pot_values_match(current_vals, prev_vals, tolerance):
                count += 1
            else:
                groups.append(prev_vals + (DEFAULT_POT_ROT, count))
                prev_vals = current_vals
                count = 1

        if prev_vals is not None:
            groups.append(prev_vals + (DEFAULT_POT_ROT, count))

    with open(output_filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["potX", "potY", "potZ", "potRot", "duration"])
        writer.writerows(groups)

    print(f"Compressed simulation input saved to '{output_filename}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Compress an input CSV by grouping consecutive frames with similar pot values."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output CSV file."
    )
    parser.add_argument(
        "--duration", type=int, default=10, help="Duration per frame (default: 10)."
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Tolerance for grouping pot values (default: 0).",
    )
    args = parser.parse_args()

    compress_pot_values(args.input, args.output, args.duration, args.tolerance)


if __name__ == "__main__":
    main()
