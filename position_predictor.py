import json
import sys


class PositionPredictor:
    def __init__(self, initial_position):
        """
        Initialize with user input initial position as tuple (x0, y0, z0)
        Initial velocity of flight is assumed to be 0, 0 , 0 the PositionPredictor with initial position and velocity set to zero
        """
        self.x0, self.y0, self.z0 = initial_position
        self.vx, self.vy, self.vz = 0, 0, 0

    def update_position(self, ax, ay, az, dt):
        """
        Update the current position and velocity based on new acceleration and time interval
        Return final predicted position as a tuple (x, y, z)
        """
        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt

        # just a basic kinematics formula applied to all axis
        self.x0 += self.vx * dt + 0.5 * ax * dt**2
        self.y0 += self.vy * dt + 0.5 * ay * dt**2
        self.z0 += self.vz * dt + 0.5 * az * dt**2

        return self.x0, self.y0, self.z0


class PositionPredictorApp:
    def __init__(self, json_file, initial_position):
        self.json_file = json_file
        self.initial_position = initial_position

    def load_data(self):
        """
        Load and return data from the specified JSON file.
        """
        try:
            with open(self.json_file, "r") as file:
                data = json.load(file)
                if not data:
                    raise ValueError(
                        "Go add some data in the file, its a desert in here"
                    )
                return data
        except json.JSONDecodeError:
            print("Error: Invalid JSON")
            return None
        except FileNotFoundError:
            print(f"Error: Mapping data file {self.json_file} was not found")
            return None
        except ValueError as e:
            print(f"Error: {e}")
            return None

    def run(self, time_window):
        data = self.load_data()
        if data is None:
            return

        predictor = PositionPredictor(self.initial_position)

        # dont need potentiometer values to calculate final position?
        for _, acc in data.items():
            acc = list(map(float, acc))
            predictor.update_position(acc[0], acc[1], acc[2], time_window)

        final_position = (predictor.x0, predictor.y0, predictor.z0)
        print(f"Final position: {final_position}")


if __name__ == "__main__":
    print(f"User input is: {sys.argv}")

    if len(sys.argv) != 6:
        print(
            "pls use arg format <json_file> <time_window> <initial_x> <initial_y> <initial_z>"
        )
        sys.exit(1)

    else:
        json_file = sys.argv[1]
        try:
            time_window = float(sys.argv[2])
            initial_x = float(sys.argv[3])
            initial_y = float(sys.argv[4])
            initial_z = float(sys.argv[5])
            initial_position = (initial_x, initial_y, initial_z)
            position_predictor_app = PositionPredictorApp(json_file, initial_position)
            position_predictor_app.run(time_window)
        except ValueError:
            print("Error: enter numbers for time window and initial coords")
            sys.exit(1)
