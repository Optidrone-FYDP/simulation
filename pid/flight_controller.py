from pid_controller import dronePID
from flight_sim_pid import *
from predict_pid import *
import keyboard

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pid = dronePID()
    pred = Predictor(DroneMovementModel.to(dev), "drone_movement_model_5k.pt", 10, dev)

    path = pd.read_csv("paths/path_to_flightplan.csv")

    if len(sys.argv) != 8:
        print(
            f"Usage: python {sys.argv[0]} <simulation_inputs.csv> <start_RX> <start_RY> <start_RZ> <start_TX> <start_TY> <start_TZ>"
        )
    
    pred.start_predictor(sys.argv[2:8])

    for index, row in path.iterrows():
        pid.setpoint(row['x'], row['y'], row['z'], row['rot'])
        while True:
            if keyboard.is_pressed('k'):
                break
            pid.get_pos(pred.current_position)
            next_pots = pid.update()
            pred.predict_next(next_pots)    #TODO: need to figure out what format the lstm model accepts for pot inputs

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())
    return
