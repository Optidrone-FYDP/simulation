from pid_controller import dronePID
#from flight_sim_pid import *
from predict_pid import *
from uart_handler import *
from vicon_handler import *
import keyboard
import time

if __name__ == "__main__":

    mode = "vicon"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pid = dronePID(mode=mode)
    pred = Predictor(DroneMovementModel().to(dev), "models/drone_movement_model_lstm_0.4.pt", 20, dev)

    path = pd.read_csv("paths/plan1.csv")

    vicon = Vicon()
    vicon.connect()
    vicon.setViconMode(ViconDataStream.Client.StreamMode.EClientPull)

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

    if mode == "vicon":
        open_serial()
        print("controller on")
        controller_on()

    time.sleep(2.2)

    print("beginning flight controller")
    print(path)
    start_flight()
    time.sleep(6)
    for index, row in path.iterrows():
        if mode == "sim":
            pid.get_pos(pred.current_position)
        else:
            pid.get_pos(vicon.get_frame())
        pid.setpoint(row['x'], row['y'], row['z'], row['rot'])
        while True:
            #print("running")
            if mode == "sim":
                input("press enter")
            
            if keyboard.is_pressed('k'):
                print("k is pressed")
                land()
                sys.exit(0)
            
            if mode == "sim":
                pid.get_pos(pred.current_position)
            else:
                pid.get_pos(vicon.get_frame())
            next_pots = pid.update()
            if pred.reached_target == True or keyboard.is_pressed('k'):
                break
            print(pred.current_position)
            print(next_pots)
            if mode == "sim":
                pred.predict_next(next_pots)
            else:
                send_to_controller(next_pots)
        time.sleep(0.5)

    land()
    #app = QApplication(sys.argv)
    #mainWindow = MainWindow()
    #sys.exit(app.exec_())
    sys.exit(0)
