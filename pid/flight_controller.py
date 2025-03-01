from pid_controller import dronePID
from flight_sim_pid import *
from predict_pid import *
import keyboard

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pid = dronePID()
    pred = Predictor(DroneMovementModel().to(dev), "models/drone_movement_model_lstm_0.4.pt", 20, dev)

    path = pd.read_csv("paths/plan1.csv")

    #if len(sys.argv) != 5:
    #    print(
    #        f"Usage: python {sys.argv[0]} <pot_inputs.csv> <start_TX> <start_TY> <start_TZ>"
    #    )

    if len(sys.argv) < 5:
        inputs = [0, 0, 1000]
    else:
        inputs = sys.argv[2:5]

    print("starting predictor")
    pred.start_predictor(inputs)

    print("beginning flight controller")
    print(path)
    for index, row in path.iterrows():
        pid.get_pos(pred.current_position)
        pid.setpoint(row['x'], row['y'], row['z'], row['rot'])
        while True:
            #print("running")
            input("press enter")
            '''
            if keyboard.is_pressed('k'):
                print("k is pressed")
                break
            '''
            for i in range(100):
                pid.get_pos(pred.current_position)
                next_pots = pid.update()
                print(pred.current_position)
                print(next_pots)
                pred.predict_next(next_pots)

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())
