import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import csv
import os
import glob
from matplotlib.widgets import Button, RadioButtons
import matplotlib as mpl


def plot_drone_path(
    csv_file, output_file=None, show_plot=True, global_stats=None, interactive=False
):
    """
    Plot a drone path from a CSV file in 3D space.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing drone path data
    output_file : str, optional
        Path to save the plot image. If None, the plot is not saved.
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    global_stats : dict, optional
        Dictionary to store global statistics across multiple files
    interactive : bool, optional
        Whether to enable interactive view toggling. Default is False.
    """
    with open(csv_file, "r") as f:
        lines = f.readlines()

    header_line = -1
    for i, line in enumerate(lines):
        if "Frame" in line and "TX" in line and "TY" in line and "TZ" in line:
            header_line = i
            break

    if header_line == -1:
        raise ValueError(f"Could not find header line in CSV file: {csv_file}")

    headers = lines[header_line].strip().split(",")
    tx_idx, ty_idx, tz_idx = -1, -1, -1

    for i, header in enumerate(headers):
        if header == "TX":
            tx_idx = i
        elif header == "TY":
            ty_idx = i
        elif header == "TZ":
            tz_idx = i

    if tx_idx == -1 or ty_idx == -1 or tz_idx == -1:
        raise ValueError(
            f"Could not find TX, TY, TZ columns in the CSV file: {csv_file}"
        )

    x_data, y_data, z_data = [], [], []

    for i in range(header_line + 2, len(lines)):  # Skip header and units line
        line = lines[i].strip()
        if not line:  # Skip empty lines
            continue

        parts = line.split(",")
        if len(parts) <= max(tx_idx, ty_idx, tz_idx):
            continue  # Skip lines with insufficient columns

        try:
            x = float(parts[tx_idx])
            y = float(parts[ty_idx])
            z = float(parts[tz_idx])

            # Skip rows with missing data
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)
        except (ValueError, IndexError):
            # Skip rows that can't be converted to float
            continue

    if not x_data:
        print(f"Warning: No valid position data found in {csv_file}. Skipping.")
        return None, None

    # Convert to numpy arrays
    x = np.array(x_data)
    y = np.array(y_data)
    z = np.array(z_data)

    # Print the maximum values in each axis for this file
    filename = os.path.basename(csv_file)
    print(f"\nFile: {filename}")
    print(
        f"X-axis (TX): Min = {np.min(x):.2f} mm, Max = {np.max(x):.2f} mm, Range = {np.ptp(x):.2f} mm"
    )
    print(
        f"Y-axis (TY): Min = {np.min(y):.2f} mm, Max = {np.max(y):.2f} mm, Range = {np.ptp(y):.2f} mm"
    )
    print(
        f"Z-axis (TZ): Min = {np.min(z):.2f} mm, Max = {np.max(z):.2f} mm, Range = {np.ptp(z):.2f} mm"
    )

    # Update global stats if provided
    if global_stats is not None:
        global_stats["x_min"] = min(global_stats["x_min"], np.min(x))
        global_stats["x_max"] = max(global_stats["x_max"], np.max(x))
        global_stats["y_min"] = min(global_stats["y_min"], np.min(y))
        global_stats["y_max"] = max(global_stats["y_max"], np.max(y))
        global_stats["z_min"] = min(global_stats["z_min"], np.min(z))
        global_stats["z_max"] = max(global_stats["z_max"], np.max(z))

    if interactive and show_plot:
        plt.rcParams["figure.figsize"] = [16, 9]
        mpl.rcParams["toolbar"] = "None"

        fig = plt.figure(figsize=(16, 9))
        fig.canvas.manager.full_screen_toggle()

        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax_xy = fig.add_subplot(2, 2, 2)
        ax_xz = fig.add_subplot(2, 2, 3)
        ax_yz = fig.add_subplot(2, 2, 4)

        ax_3d.plot(x, y, z, "b-", linewidth=2, label="Drone Path")
        ax_3d.scatter(x[0], y[0], z[0], c="green", s=100, label="Start")
        ax_3d.scatter(x[-1], y[-1], z[-1], c="red", s=100, label="End")
        ax_3d.set_xlabel("X Position (mm)")
        ax_3d.set_ylabel("Y Position (mm)")
        ax_3d.set_zlabel("Z Position (mm)")
        ax_3d.set_title(f"3D View: {filename}")
        ax_3d.legend()
        ax_3d.grid(True)

        ax_xy.plot(x, y, "b-", linewidth=2)
        ax_xy.scatter(x[0], y[0], c="green", s=100, label="Start")
        ax_xy.scatter(x[-1], y[-1], c="red", s=100, label="End")
        ax_xy.set_xlabel("X Position (mm)")
        ax_xy.set_ylabel("Y Position (mm)")
        ax_xy.set_title("XY Plane (Top View)")
        ax_xy.grid(True)
        ax_xy.axis("equal")

        ax_xz.plot(x, z, "b-", linewidth=2)
        ax_xz.scatter(x[0], z[0], c="green", s=100, label="Start")
        ax_xz.scatter(x[-1], z[-1], c="red", s=100, label="End")
        ax_xz.set_xlabel("X Position (mm)")
        ax_xz.set_ylabel("Z Position (mm)")
        ax_xz.set_title("XZ Plane (Front View)")
        ax_xz.grid(True)
        ax_xz.axis("equal")

        ax_yz.plot(y, z, "b-", linewidth=2)
        ax_yz.scatter(y[0], z[0], c="green", s=100, label="Start")
        ax_yz.scatter(y[-1], z[-1], c="red", s=100, label="End")
        ax_yz.set_xlabel("Y Position (mm)")
        ax_yz.set_ylabel("Z Position (mm)")
        ax_yz.set_title("YZ Plane (Side View)")
        ax_yz.grid(True)
        ax_yz.axis("equal")

        ax_radio = plt.axes([0.02, 0.8, 0.12, 0.15])
        radio = RadioButtons(
            ax_radio, ("All Views", "XY Plane", "XZ Plane", "YZ Plane", "3D Only")
        )

        def view_selection(label):
            if label == "All Views":
                ax_3d.set_visible(True)
                ax_xy.set_visible(True)
                ax_xz.set_visible(True)
                ax_yz.set_visible(True)
            elif label == "XY Plane":
                ax_3d.set_visible(False)
                ax_xy.set_visible(True)
                ax_xz.set_visible(False)
                ax_yz.set_visible(False)
                ax_xy.set_position([0.1, 0.1, 0.8, 0.8])
            elif label == "XZ Plane":
                ax_3d.set_visible(False)
                ax_xy.set_visible(False)
                ax_xz.set_visible(True)
                ax_yz.set_visible(False)
                ax_xz.set_position([0.1, 0.1, 0.8, 0.8])
            elif label == "YZ Plane":
                ax_3d.set_visible(False)
                ax_xy.set_visible(False)
                ax_xz.set_visible(False)
                ax_yz.set_visible(True)
                ax_yz.set_position([0.1, 0.1, 0.8, 0.8])
            elif label == "3D Only":
                ax_3d.set_visible(True)
                ax_xy.set_visible(False)
                ax_xz.set_visible(False)
                ax_yz.set_visible(False)
                ax_3d.set_position([0.1, 0.1, 0.8, 0.8])

            plt.draw()

        radio.on_clicked(view_selection)

        ax_reset = plt.axes([0.02, 0.7, 0.12, 0.05])
        reset_button = Button(ax_reset, "Reset Layout")

        def reset_layout(event):
            ax_3d.set_visible(True)
            ax_xy.set_visible(True)
            ax_xz.set_visible(True)
            ax_yz.set_visible(True)

            ax_3d.set_position([0.125, 0.525, 0.35, 0.35])
            ax_xy.set_position([0.55, 0.525, 0.35, 0.35])
            ax_xz.set_position([0.125, 0.1, 0.35, 0.35])
            ax_yz.set_position([0.55, 0.1, 0.35, 0.35])

            radio.set_active(0)
            plt.draw()

        reset_button.on_clicked(reset_layout)

        plt.tight_layout(rect=[0, 0, 1, 1])

        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")

        plt.show()
        return fig, [ax_3d, ax_xy, ax_xz, ax_yz]
    else:
        plt.rcParams["figure.figsize"] = [16, 9]
        mpl.rcParams["toolbar"] = "None"

        fig = plt.figure(figsize=(16, 9))
        if show_plot:
            fig.canvas.manager.full_screen_toggle()

        ax = fig.add_subplot(111, projection="3d")

        ax.plot(x, y, z, "b-", linewidth=2, label="Drone Path")

        ax.scatter(x[0], y[0], z[0], c="green", s=100, label="Start")
        ax.scatter(x[-1], y[-1], z[-1], c="red", s=100, label="End")

        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("Y Position (mm)")
        ax.set_zlabel("Z Position (mm)")

        ax.set_title(f"Drone Flight Path: {filename}")

        ax.legend()

        ax.grid(True)

        try:
            max_range = np.array([np.ptp(x), np.ptp(y), np.ptp(z)]).max() / 2.0

            mid_x = (np.max(x) + np.min(x)) / 2.0
            mid_y = (np.max(y) + np.min(y)) / 2.0
            mid_z = (np.max(z) + np.min(z)) / 2.0

            if np.isnan(max_range) or np.isinf(max_range) or max_range == 0:
                print(
                    f"Warning: Could not determine appropriate axis limits for {filename}. Using auto-scaling."
                )
            else:
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
        except Exception as e:
            print(
                f"Warning: Error setting axis limits for {filename}: {e}. Using auto-scaling."
            )

        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax


def process_all_files(input_dir, output_dir, show_plot=False, interactive=False):
    """
    Process all CSV files in the input directory and save plots to the output directory.

    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files to process
    output_dir : str
        Directory to save output plots
    show_plot : bool, optional
        Whether to display the plots. Default is False.
    interactive : bool, optional
        Whether to enable interactive view toggling. Default is False.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    global_stats = {
        "x_min": float("inf"),
        "x_max": float("-inf"),
        "y_min": float("inf"),
        "y_max": float("-inf"),
        "z_min": float("inf"),
        "z_max": float("-inf"),
    }

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}_3d_plot.png")

        try:
            plot_drone_path(csv_file, output_file, show_plot, global_stats, interactive)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n" + "=" * 50)
    print("MAXIMUM VALUES OBSERVED ACROSS ALL FILES:")
    print("=" * 50)
    print(
        f"X-axis (TX): Min = {global_stats['x_min']:.2f} mm, Max = {global_stats['x_max']:.2f} mm, Range = {global_stats['x_max'] - global_stats['x_min']:.2f} mm"
    )
    print(
        f"Y-axis (TY): Min = {global_stats['y_min']:.2f} mm, Max = {global_stats['y_max']:.2f} mm, Range = {global_stats['y_max'] - global_stats['y_min']:.2f} mm"
    )
    print(
        f"Z-axis (TZ): Min = {global_stats['z_min']:.2f} mm, Max = {global_stats['z_max']:.2f} mm, Range = {global_stats['z_max'] - global_stats['z_min']:.2f} mm"
    )
    print("=" * 50)


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
        parser.error("csv_file is required unless --all or --synthetic flag is set")

    if not args.output:
        input_filename = os.path.basename(args.csv_file)
        base_name = os.path.splitext(input_filename)[0]
        output_dir = "flights/3d_output"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{base_name}_3d_plot.png")

    plot_drone_path(
        args.csv_file, args.output, not args.no_show, interactive=args.interactive
    )


if __name__ == "__main__":
    main()
