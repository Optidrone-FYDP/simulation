import numpy as np
import pandas as pd
import os
import math
from datetime import datetime


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_straight_flight(
    num_frames=200, plane="xy", speed=5, noise_level=0.5, pot_values=None
):
    """
    Generate a straight flight path in the specified plane.
    Speed is in mm/frame at 30fps.
    """
    frames = np.arange(num_frames)

    # Initialize rotation angles with small random noise
    rx = np.random.normal(0, noise_level * 0.01, num_frames)
    ry = np.random.normal(0, noise_level * 0.01, num_frames)
    rz = np.random.normal(0, noise_level * 0.01, num_frames)

    # Initialize position coordinates
    tx = np.zeros(num_frames)
    ty = np.zeros(num_frames)
    tz = np.zeros(num_frames)

    # Set default pot values if not provided
    if pot_values is None:
        pot_values = {"potX": 64, "potY": 64, "potZ": 64, "potRot": 64}

    # Generate position data based on the specified plane
    if plane == "xy":
        tx = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        ty = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potX" not in pot_values:
            pot_values["potX"] = 96  # Forward
        if "potY" not in pot_values:
            pot_values["potY"] = 96  # Right
    elif plane == "xz":
        tx = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        tz = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potX" not in pot_values:
            pot_values["potX"] = 96  # Forward
        if "potZ" not in pot_values:
            pot_values["potZ"] = 96  # Up
    elif plane == "yz":
        ty = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        tz = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potY" not in pot_values:
            pot_values["potY"] = 96  # Right
        if "potZ" not in pot_values:
            pot_values["potZ"] = 96  # Up
    elif plane == "x":
        tx = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potX" not in pot_values:
            pot_values["potX"] = 96  # Forward
    elif plane == "y":
        ty = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potY" not in pot_values:
            pot_values["potY"] = 96  # Right
    elif plane == "z":
        tz = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potZ" not in pot_values:
            pot_values["potZ"] = 96  # Up

    # Create arrays for pot values
    potX = np.ones(num_frames) * pot_values.get("potX", 64)
    potY = np.ones(num_frames) * pot_values.get("potY", 64)
    potZ = np.ones(num_frames) * pot_values.get("potZ", 64)
    potRot = np.ones(num_frames) * pot_values.get("potRot", 64)

    # Create DataFrame with the correct column order
    df = pd.DataFrame(
        {
            "Frame": frames,
            "RX": rx,
            "RY": ry,
            "RZ": rz,
            "TX": tx,
            "TY": ty,
            "TZ": tz,
            "potZ": potZ,
            "potRot": potRot,
            "potY": potY,
            "potX": potX,
        }
    )

    return df


def generate_diagonal_flight(
    num_frames=200, plane="xyz", speed=5, noise_level=0.5, pot_values=None
):
    """
    Generate a diagonal flight path in the specified plane.
    Speed is in mm/frame at 30fps.
    """
    frames = np.arange(num_frames)

    # Initialize rotation angles with small random noise
    rx = np.random.normal(0, noise_level * 0.01, num_frames)
    ry = np.random.normal(0, noise_level * 0.01, num_frames)
    rz = np.random.normal(0, noise_level * 0.01, num_frames)

    # Initialize position coordinates
    tx = np.zeros(num_frames)
    ty = np.zeros(num_frames)
    tz = np.zeros(num_frames)

    # Set default pot values if not provided
    if pot_values is None:
        pot_values = {"potX": 64, "potY": 64, "potZ": 64, "potRot": 64}

    # Generate position data based on the specified plane
    if plane == "xyz":
        tx = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        ty = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        tz = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potX" not in pot_values:
            pot_values["potX"] = 96  # Forward
        if "potY" not in pot_values:
            pot_values["potY"] = 96  # Right
        if "potZ" not in pot_values:
            pot_values["potZ"] = 96  # Up
    elif plane == "xy":
        tx = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        ty = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potX" not in pot_values:
            pot_values["potX"] = 96  # Forward
        if "potY" not in pot_values:
            pot_values["potY"] = 96  # Right
    elif plane == "xz":
        tx = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        tz = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potX" not in pot_values:
            pot_values["potX"] = 96  # Forward
        if "potZ" not in pot_values:
            pot_values["potZ"] = 96  # Up
    elif plane == "yz":
        ty = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        tz = np.linspace(0, speed * num_frames, num_frames) + np.random.normal(
            0, noise_level, num_frames
        )
        if "potY" not in pot_values:
            pot_values["potY"] = 96  # Right
        if "potZ" not in pot_values:
            pot_values["potZ"] = 96  # Up

    # Create arrays for pot values
    potX = np.ones(num_frames) * pot_values.get("potX", 64)
    potY = np.ones(num_frames) * pot_values.get("potY", 64)
    potZ = np.ones(num_frames) * pot_values.get("potZ", 64)
    potRot = np.ones(num_frames) * pot_values.get("potRot", 64)

    # Create DataFrame with the correct column order
    df = pd.DataFrame(
        {
            "Frame": frames,
            "RX": rx,
            "RY": ry,
            "RZ": rz,
            "TX": tx,
            "TY": ty,
            "TZ": tz,
            "potZ": potZ,
            "potRot": potRot,
            "potY": potY,
            "potX": potX,
        }
    )

    return df


def generate_circle_flight(
    num_frames=200, plane="xy", radius=100, noise_level=0.5, pot_values=None
):
    """
    Generate a circular flight path in the specified plane.
    Radius is in mm.
    """
    frames = np.arange(num_frames)

    # Initialize rotation angles with small random noise
    rx = np.random.normal(0, noise_level * 0.01, num_frames)
    ry = np.random.normal(0, noise_level * 0.01, num_frames)
    rz = np.random.normal(0, noise_level * 0.01, num_frames)

    # Initialize position coordinates
    tx = np.zeros(num_frames)
    ty = np.zeros(num_frames)
    tz = np.zeros(num_frames)

    # Set default pot values if not provided
    if pot_values is None:
        pot_values = {
            "potX": 64,
            "potY": 64,
            "potZ": 64,
            "potRot": 96,
        }  # Default rotation for circles

    # Generate angles for circular motion
    angles = np.linspace(0, 2 * np.pi, num_frames)

    # Generate position data based on the specified plane
    if plane == "xy":
        tx = radius * np.cos(angles) + np.random.normal(0, noise_level, num_frames)
        ty = radius * np.sin(angles) + np.random.normal(0, noise_level, num_frames)
    elif plane == "xz":
        tx = radius * np.cos(angles) + np.random.normal(0, noise_level, num_frames)
        tz = radius * np.sin(angles) + np.random.normal(0, noise_level, num_frames)
    elif plane == "yz":
        ty = radius * np.cos(angles) + np.random.normal(0, noise_level, num_frames)
        tz = radius * np.sin(angles) + np.random.normal(0, noise_level, num_frames)

    # For circle flights, we can vary the pot values to simulate the changing control inputs
    # needed to maintain a circular path
    if pot_values.get("vary", False):
        potX = 64 + 32 * np.cos(angles)
        potY = 64 + 32 * np.sin(angles)
        potZ = np.ones(num_frames) * pot_values.get("potZ", 64)
        potRot = np.ones(num_frames) * pot_values.get(
            "potRot", 96
        )  # Rotation for circles
    else:
        potX = np.ones(num_frames) * pot_values.get("potX", 64)
        potY = np.ones(num_frames) * pot_values.get("potY", 64)
        potZ = np.ones(num_frames) * pot_values.get("potZ", 64)
        potRot = np.ones(num_frames) * pot_values.get(
            "potRot", 96
        )  # Rotation for circles

    # Create DataFrame with the correct column order
    df = pd.DataFrame(
        {
            "Frame": frames,
            "RX": rx,
            "RY": ry,
            "RZ": rz,
            "TX": tx,
            "TY": ty,
            "TZ": tz,
            "potZ": potZ,
            "potRot": potRot,
            "potY": potY,
            "potX": potX,
        }
    )

    return df


def generate_random_walk(num_frames=200, dimensions=3, step_size=2, noise_level=0.5):
    """
    Generate a predictable pattern (like zigzag) instead of a truly random walk.
    This creates more learnable patterns for a NARX model.
    Step size is in mm/frame at 30fps.
    """
    frames = np.arange(num_frames)

    # Initialize rotation angles with small random noise
    rx = np.random.normal(0, noise_level * 0.01, num_frames)
    ry = np.random.normal(0, noise_level * 0.01, num_frames)
    rz = np.random.normal(0, noise_level * 0.01, num_frames)

    # Initialize position coordinates
    tx = np.zeros(num_frames)
    ty = np.zeros(num_frames)
    tz = np.zeros(num_frames)

    # Generate zigzag or simple patterns instead of random walks
    pattern_type = np.random.choice(["zigzag", "sine", "square", "triangle"])

    if dimensions >= 1:
        if pattern_type == "zigzag":
            # Create a zigzag pattern with period of about 30-60 frames
            period = np.random.randint(30, 60)
            direction = 1
            tx[0] = 0
            for i in range(1, num_frames):
                if i % period == 0:
                    direction *= -1
                tx[i] = tx[i - 1] + direction * step_size
        elif pattern_type == "sine":
            # Create a sine wave pattern
            period = np.random.randint(30, 60)
            amplitude = step_size * period / 4  # Reasonable amplitude
            tx = amplitude * np.sin(2 * np.pi * frames / period)
        elif pattern_type == "square":
            # Create a square wave pattern
            period = np.random.randint(40, 80)
            amplitude = step_size * 10
            tx = amplitude * np.sign(np.sin(2 * np.pi * frames / period))
        elif pattern_type == "triangle":
            # Create a triangle wave pattern
            period = np.random.randint(40, 80)
            amplitude = step_size * 15
            tx = amplitude * (
                2 * np.abs(2 * (frames / period - np.floor(frames / period + 0.5))) - 1
            )

        # Add small noise to make it more realistic
        tx += np.random.normal(0, noise_level, num_frames)

    if dimensions >= 2:
        # Use a different pattern for Y to create interesting combinations
        pattern_type_y = np.random.choice(["zigzag", "sine", "square", "triangle"])

        if pattern_type_y == "zigzag":
            period = np.random.randint(30, 60)
            direction = 1
            ty[0] = 0
            for i in range(1, num_frames):
                if i % period == 0:
                    direction *= -1
                ty[i] = ty[i - 1] + direction * step_size
        elif pattern_type_y == "sine":
            period = np.random.randint(30, 60)
            amplitude = step_size * period / 4
            ty = amplitude * np.sin(2 * np.pi * frames / period)
        elif pattern_type_y == "square":
            period = np.random.randint(40, 80)
            amplitude = step_size * 10
            ty = amplitude * np.sign(np.sin(2 * np.pi * frames / period))
        elif pattern_type_y == "triangle":
            period = np.random.randint(40, 80)
            amplitude = step_size * 15
            ty = amplitude * (
                2 * np.abs(2 * (frames / period - np.floor(frames / period + 0.5))) - 1
            )

        # Add small noise to make it more realistic
        ty += np.random.normal(0, noise_level, num_frames)

    if dimensions >= 3:
        # Use a different pattern for Z to create interesting combinations
        pattern_type_z = np.random.choice(["zigzag", "sine", "square", "triangle"])

        if pattern_type_z == "zigzag":
            period = np.random.randint(30, 60)
            direction = 1
            tz[0] = 0
            for i in range(1, num_frames):
                if i % period == 0:
                    direction *= -1
                tz[i] = tz[i - 1] + direction * step_size
        elif pattern_type_z == "sine":
            period = np.random.randint(30, 60)
            amplitude = step_size * period / 4
            tz = amplitude * np.sin(2 * np.pi * frames / period)
        elif pattern_type_z == "square":
            period = np.random.randint(40, 80)
            amplitude = step_size * 10
            tz = amplitude * np.sign(np.sin(2 * np.pi * frames / period))
        elif pattern_type_z == "triangle":
            period = np.random.randint(40, 80)
            amplitude = step_size * 15
            tz = amplitude * (
                2 * np.abs(2 * (frames / period - np.floor(frames / period + 0.5))) - 1
            )

        # Add small noise to make it more realistic
        tz += np.random.normal(0, noise_level, num_frames)

    # Initialize pot values at neutral
    potX = np.ones(num_frames) * 64
    potY = np.ones(num_frames) * 64
    potZ = np.ones(num_frames) * 64
    potRot = np.ones(num_frames) * 64

    # Adjust pot values based on movement
    for i in range(num_frames):
        if i > 0:
            # potX - Forward/Backward
            if tx[i] > tx[i - 1]:
                potX[i] = 96  # Forward
            elif tx[i] < tx[i - 1]:
                potX[i] = 32  # Backward

            # potY - Right/Left
            if ty[i] > ty[i - 1]:
                potY[i] = 96  # Right
            elif ty[i] < ty[i - 1]:
                potY[i] = 32  # Left

            # potZ - Up/Down
            if tz[i] > tz[i - 1]:
                potZ[i] = 96  # Up
            elif tz[i] < tz[i - 1]:
                potZ[i] = 32  # Down

    # Create DataFrame with the correct column order
    df = pd.DataFrame(
        {
            "Frame": frames,
            "RX": rx,
            "RY": ry,
            "RZ": rz,
            "TX": tx,
            "TY": ty,
            "TZ": tz,
            "potZ": potZ,
            "potRot": potRot,
            "potY": potY,
            "potX": potX,
        }
    )

    return df


def main():
    output_dir = "flights/synthetic"
    create_directory_if_not_exists(output_dir)

    # Generate 50 flights with different patterns
    flight_id = 1

    # Straight flights in single plane (15 flights)
    planes = ["x", "y", "z", "xy", "xz", "yz"]
    for i, plane in enumerate(planes):
        for j in range(max(1, 15 // len(planes))):
            # Calculate realistic speed for a drone at 30fps (between 50-200 mm/s)
            # Convert to mm/frame by dividing by 30
            speed_mm_per_second = np.random.uniform(50, 200)
            speed = speed_mm_per_second / 30  # mm per frame

            noise = np.random.uniform(0.2, 1.0)
            num_frames = np.random.randint(150, 300)

            # Vary pot values slightly for each flight
            potX = 64 + (32 if "x" in plane else 0) + np.random.randint(-5, 5)
            potY = 64 + (32 if "y" in plane else 0) + np.random.randint(-5, 5)
            potZ = 64 + (32 if "z" in plane else 0) + np.random.randint(-5, 5)
            potRot = 64  # Keep rotation neutral

            pot_values = {"potX": potX, "potY": potY, "potZ": potZ, "potRot": potRot}

            df = generate_straight_flight(
                num_frames=num_frames,
                plane=plane,
                speed=speed,
                noise_level=noise,
                pot_values=pot_values,
            )

            filename = f"synthetic_straight_{plane}_{flight_id:02d}.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"Generated {filename}")
            flight_id += 1

    # Diagonal flights (10 flights)
    planes = ["xy", "xz", "yz", "xyz"]
    for i, plane in enumerate(planes):
        for j in range(max(1, 10 // len(planes))):
            # Calculate realistic speed for a drone at 30fps (between 50-150 mm/s)
            # Convert to mm/frame by dividing by 30
            speed_mm_per_second = np.random.uniform(50, 150)
            speed = speed_mm_per_second / 30  # mm per frame

            noise = np.random.uniform(0.2, 1.0)
            num_frames = np.random.randint(150, 300)

            # Vary pot values slightly for each flight
            potX = 64 + (32 if "x" in plane else 0) + np.random.randint(-5, 5)
            potY = 64 + (32 if "y" in plane else 0) + np.random.randint(-5, 5)
            potZ = 64 + (32 if "z" in plane else 0) + np.random.randint(-5, 5)
            potRot = 64  # Keep rotation neutral

            pot_values = {"potX": potX, "potY": potY, "potZ": potZ, "potRot": potRot}

            df = generate_diagonal_flight(
                num_frames=num_frames,
                plane=plane,
                speed=speed,
                noise_level=noise,
                pot_values=pot_values,
            )

            filename = f"synthetic_diagonal_{plane}_{flight_id:02d}.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"Generated {filename}")
            flight_id += 1

    # Circle flights (15 flights)
    planes = ["xy", "xz", "yz"]
    for i, plane in enumerate(planes):
        for j in range(max(1, 15 // len(planes))):
            # Realistic radius for a drone circle (between 50-200 mm)
            radius = np.random.uniform(50, 200)
            noise = np.random.uniform(0.2, 1.0)
            num_frames = np.random.randint(200, 400)

            # For circle flights, we'll use varying pot values for some
            vary = j % 2 == 0
            potRot = np.random.randint(90, 110)  # Rotation control for circles

            pot_values = {
                "potX": 64,
                "potY": 64,
                "potZ": 64,
                "potRot": potRot,
                "vary": vary,
            }

            df = generate_circle_flight(
                num_frames=num_frames,
                plane=plane,
                radius=radius,
                noise_level=noise,
                pot_values=pot_values,
            )

            filename = f"synthetic_circle_{plane}_{flight_id:02d}.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"Generated {filename}")
            flight_id += 1

    # Random walk flights (10 flights)
    dimensions = [1, 2, 3]
    for i, dim in enumerate(dimensions):
        for j in range(max(1, 10 // len(dimensions))):
            # Calculate realistic step size for a drone at 30fps (between 30-100 mm/s)
            # Convert to mm/frame by dividing by 30
            step_size_mm_per_second = np.random.uniform(30, 100)
            step_size = step_size_mm_per_second / 30  # mm per frame

            noise = np.random.uniform(0.2, 1.0)
            num_frames = np.random.randint(200, 500)

            df = generate_random_walk(
                num_frames=num_frames,
                dimensions=dim,
                step_size=step_size,
                noise_level=noise,
            )

            filename = f"synthetic_random_walk_{dim}d_{flight_id:02d}.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"Generated {filename}")
            flight_id += 1

    print(f"\nGenerated {flight_id-1} synthetic flight files in {output_dir}")


if __name__ == "__main__":
    main()
