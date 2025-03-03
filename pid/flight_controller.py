from pid_controller import dronePID
#from flight_sim_pid import *
from predict_pid import *
from uart_handler_2 import *
from vicon_handler import *
import keyboard
import time
import serial

# uart_port = serial.Serial("COM5", 9600, timeout=5)

# def controller_on():
#     uart_port.write(bytes.fromhex("31"))

# def start_flight():
#     uart_port.write(bytes.fromhex("32"))
#     uart_port.write(bytes.fromhex("33"))
#     uart_port.write(bytes.fromhex("34"))

# def send_to_controller(next_inputs):
#     uart_port.write(bytes.fromhex("04"))    #left/right
#     uart_port.write(hex_helper(next_inputs[0]))
#     uart_port.write(bytes.fromhex("03"))    #front/back
#     uart_port.write(hex_helper(next_inputs[1]))
#     uart_port.write(bytes.fromhex("01"))    #up/down
#     uart_port.write(hex_helper(next_inputs[2]))
#     uart_port.write(bytes.fromhex("02"))    #rotation
#     uart_port.write(hex_helper(next_inputs[3]))

# def land():
#     uart_port.write(bytes.fromhex("10"))

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
        time.sleep(10)
        print("controller on")
        controller_on()

    time.sleep(3)

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
            if pid.reached_target == True or keyboard.is_pressed('k'):
                print("k is pressed, aborting")
                break
            if mode == "sim":
                print(pred.current_position)
            else:
                print(vicon.curr_pos)
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
