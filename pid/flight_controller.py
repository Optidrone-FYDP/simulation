from pid_controller import dronePID
from predict_pid import *
from uart_handler_2 import *
from vicon_handler import *
import keyboard
import time
import serial
import pandas

if __name__ == "__main__":

    mode = "vicon"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pid = dronePID(mode=mode)
    pred = Predictor(DroneMovementModel().to(dev), "models/drone_movement_model_lstm_0.4.pt", 20, dev)

    path = pd.read_csv("paths/plan3.csv")

    path_plot = []

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
        time.sleep(5)
        print("controller on")
        controller_on()

    time.sleep(3)

    print("beginning flight controller")
    print(path)
    start_flight()
    time.sleep(10)
    for index, row in path.iterrows():
        if mode == "sim":
            pid.get_pos(pred.current_position)
        else:
            pid.get_pos(vicon.get_frame())
            path_plot.append(pid.current_pos)
        pid.setpoint(row['x'], row['y'], row['z'], row['rot'])
        while True:
            #print("running")
            #if mode == "sim":
            #input("press enter")
            
            if mode == "sim":
                pid.get_pos(pred.current_position)
            else:
                pid.get_pos(vicon.get_frame())
                path_plot.append(pid.current_pos)
            next_pots = pid.update()
            if pid.reached_target == True or keyboard.is_pressed('k'):
                print("k is pressed, aborting")
                land()
                time.sleep(4)
                controller_on()
                time.sleep(5)
                df = pd.DataFrame(path_plot, columns=['x','y','z','rot'])
                df.to_csv("output/output_path.csv", index=False)
                sys.exit()
            if keyboard.is_pressed('n'):
                print("n is pressed, moving to next setpoint")
                all_neutral()
                break
            if mode == "sim":
                print(pred.current_position)
            else:
                print(pid.current_pos)
            print(next_pots)
            if mode == "sim":
                pred.predict_next(next_pots)
            else:
                send_to_controller(next_pots)
        time.sleep(0.5)

    print("not more setpoints, path done!")
    land()
    time.sleep(4)
    controller_on()
    time.sleep(5)
    df = pd.DataFrame(path_plot, columns=['x','y','z','rot'])
    df.to_csv("output/output_path.csv", index=False)
    sys.exit()
