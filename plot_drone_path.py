import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Plot drone path from CSV file in 3D space"
    )
    parser.add_argument(
        "csv_file", nargs="?", help="Path to the CSV file containing drone path data"
    )
    parser.add_argument("--output", "-o", help="Path to save the plot image")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display the plot"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all CSV files in flights/output directory",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Process all CSV files in flights/synthetic directory",
    )
    parser.add_argument(
        "--output-dir",
        default="flights/3d_output",
        help="Directory to save output plots when processing all files",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Enable interactive view toggling"
    )

    args = parser.parse_args()

    if args.synthetic:
        input_dir = "flights/synthetic"
        output_dir = "flights/synthetic_3d_output"
        process_all_files(input_dir, output_dir, not args.no_show, args.interactive)
        return

    if args.all:
        input_dir = "flights/output"
        output_dir = args.output_dir
        process_all_files(input_dir, output_dir, not args.no_show, args.interactive)
        return

    if not args.csv_file:
        # ... existing code ...

if __name__ == "__main__":
    main() 