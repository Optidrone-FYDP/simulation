import json
from scipy.interpolate import interp1d

"""
uses kinematic equations to calculate the new positions and velocities, and outputs final x y z position in mm:
position is updated using formula: position_new[i] = position_old[i] + velocity[i] * duration + 0.5 * acceleration[i] * (duration^2)
velocity is updated using formula: velocity_new[i] = velocity_old[i] + acceleration[i] * duration
"""


class PositionPredictor:

    def __init__(self, start_pos, potentiometer_map):
        self.position = start_pos  # user input
        self.velocity = [0, 0, 0]  # assuming drone flight starts stationary
        self.potentiometer_map = potentiometer_map  # from ingestion output json
        self.setup_interpolation()  # only used if user input has unmapped pot values

    def setup_interpolation(self):
        """
        set up interpolation based on the potentiometer map provided by ingestion step
        """
        self.interpolators = {}
        for i in range(3):  # range 3 = 3 axis
            pot = []
            accel = []
            for key, value in self.potentiometer_map.items():
                pot.append(int(key, 10))
                accel.append(value[i])
            self.interpolators[i] = interp1d(pot, accel, fill_value="extrapolate")

    def interpolate_acceleration(self, pot_input):
        """
        dataset to interpolate on is in setup_interpolation()
        """
        try:
            pot_value = int(pot_input, 10)
            return [self.interpolators[i](pot_value) for i in range(3)]
        except Exception as e:
            print(f"Interpolation error: {e}")
            return [0, 0, 0]

    def update_position(self, pot_input, duration):
        """
        Runs per each command in flight user input
        """
        if pot_input in self.potentiometer_map:
            acceleration = self.potentiometer_map[pot_input]
        else:
            print(
                f"User potentiometer input '{pot_input}' could not be mapped to a known value, to use interpolation."
            )
            acceleration = self.interpolate_acceleration(pot_input)

        for i in range(3):
            # calculate new pos
            self.position[i] += self.velocity[i] * duration + 0.5 * acceleration[i] * (
                duration**2
            )
            # calculate new velocity
            self.velocity[i] += acceleration[i] * duration

        print(
            f"New position (x,y,z) in mm after user potentiometer input (xxxyyyzzz) '{pot_input}' applied for {duration} seconds: {self.position}"
        )

    def get_position(self):  # simple get
        return self.position


def load_json(filename):
    """
    helper function for json file and validation
    """
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File '{filename}' not found error")
        exit(1)
    except json.JSONDecodeError:
        print(f"File '{filename}' contains invalid JSON error")
        exit(1)


def validate_potentiometer_map(potentiometer_map):
    """
    probably unnecessary but i left it in from debugging
    """
    for key, value in potentiometer_map.items():
        if len(value) != 3 or not isinstance(key, str) or not isinstance(value, list):
            print(f"Invalid potentiometer map entry '{key}: {value}'.")
            exit(1)


def validate_commands(data):
    """
    command validation in user_input.json, just to make sure all inputs are provided in JSON
    """
    if (
        "initial_position" not in data or "commands" not in data
    ):  # mandatory json fields
        print("JSON file missing 'initial_position' and 'commands' fields")
        exit(1)
    if (  # initial position format check
        not isinstance(data["initial_position"], list)
        or len(data["initial_position"]) != 3
    ):
        print("Provide 'initial_position' as 3 numbers of x y z")
        exit(1)

    for key, value in data["commands"].items():  # type check
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            print(f"Invalid input command types '{key}: {value}'.")
            exit(1)


def main():
    potentiometer_map = load_json("potentiometer_map.json")
    validate_potentiometer_map(potentiometer_map)

    # json validation
    data = load_json("user_input.json")
    validate_commands(data)

    start_pos = data["initial_position"]
    commands = data["commands"]

    print(f"Initial position (x,y,z) in mm: {start_pos}")
    predictor = PositionPredictor(start_pos, potentiometer_map)

    for pot_input, duration in commands.items():
        predictor.update_position(pot_input, duration)

    final_position = predictor.get_position()
    print(f"Final position (x,y,z) in mm is: {final_position}")


if __name__ == "__main__":
    main()
