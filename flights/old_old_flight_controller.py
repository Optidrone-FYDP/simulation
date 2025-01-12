import tkinter as tk
import serial
import time
import json
import pandas as pd

uart_port = serial.Serial("COM3", 9600, timeout=5)
fps = 30

def hex_helper(int_in):
    hex_str = hex(int_in)[hex(int_in).find('x')+1:]
    if(len(hex_str)%2 != 0):
        hex_str = "0"+hex_str
    return bytes.fromhex(hex_str)

on = False    

# FUNCTIONS
def controller_on():
    global on
    uart_port.write(bytes.fromhex("31"))
    on = not on

def start_flight():
    global on
    if(on):
        uart_port.write(bytes.fromhex("32"))
        uart_port.write(bytes.fromhex("33"))
        uart_port.write(bytes.fromhex("34"))

def start_routine():
    plan_name = plan_name_strvar.get()
    with open(plan_name + ".json", "r") as file:
        plan = json.load(file)
    plan_size = len(plan["ud"])
    start_time = time.time()
    previous_time = start_time
    pots = []
    i = 0
    while(True):
        if i >= plan_size:
            break
        timestamp = time.time()
        if (timestamp - previous_time) > (1/fps):
            uart_port.write(bytes.fromhex("01"))    #up/down
            #print(plan['ud'][i])
            #print(hex_helper(plan['ud'][i]))
            uart_port.write(hex_helper(plan['ud'][i]))
            #print("immediately returned value")
            #print(uart_port.read())
            uart_port.write(bytes.fromhex("02"))    #rotation
            uart_port.write(hex_helper(plan['rt'][i]))
            uart_port.write(bytes.fromhex("03"))    #front/back
            uart_port.write(hex_helper(plan['fb'][i]))
            uart_port.write(bytes.fromhex("04"))    #left/right
            uart_port.write(hex_helper(plan['lr'][i]))
            i += 1
            previous_time = timestamp
        pot_status = [timestamp-start_time] + joy_status()
        pots.append(pot_status)
    for i in range(len(pots)):
        pots[i][1] = int.from_bytes(pots[i][1], 'big')
        pots[i][2] = int.from_bytes(pots[i][2], 'big')
        pots[i][3] = int.from_bytes(pots[i][3], 'big')
        pots[i][4] = int.from_bytes(pots[i][4], 'big')
        pots[i][5] = int.from_bytes(pots[i][5], 'big')
    uart_port.write(bytes.fromhex("0f"))
    df = pd.DataFrame(pots, columns = ["timestamp", "potZ", "potRot", "potY", "potX", "potCam"])
    df.to_csv(plan_name + "_out.csv")

def update_flight():
    ud = up_down_intvar.get()
    lr = left_right_intvar.get()
    fb = forward_backward_intvar.get()
    print("FLYING ", ud, " JOYSTICK UP/DOWN")
    print("FLYING ", lr, " JOYSTICK LEFT/RIGHT")
    print("FLYING ", fb, " JOYSTICK FORWARD/BACKWARD")
    print("======================================")
    uart_port.write(bytes.fromhex("01"))
    uart_port.write(hex_helper(ud))
    uart_port.write(bytes.fromhex("03"))
    uart_port.write(hex_helper(fb))
    uart_port.write(bytes.fromhex("04"))
    uart_port.write(hex_helper(lr))

def land():
    uart_port.write(bytes.fromhex("10"))

def joy_status():
    uart_port.write(bytes.fromhex("11"))
    status = []
    for i in range(5):
        val = uart_port.read()
        print(val)
        status.append(val)
    return status

def clear_entries():
    up_down_intvar.set(64)
    left_right_intvar.set(64)
    forward_backward_intvar.set(64)
    plan_name_strvar.set("")


# GUI

# Create root window and main title
root = tk.Tk()
root.title("OptiDrone Test Flight Controls")
title_lbl = tk.Label(root, text="OptiDrone Test Flight Controls")

# Create labels
up_down_lbl = tk.Label(root, text="UP/DOWN (joystick n)")
left_right_lbl = tk.Label(root, text="LEFT/RIGHT (joystick n)")
forward_backward_lbl = tk.Label(root, text="FORWARD/BACKWARD (joystick n)")
plan_name_lbl = tk.Label(root, text="Flight Plan Name:")

# Create intvariables
up_down_intvar = tk.IntVar(root, value=0)
left_right_intvar = tk.IntVar(root, value=0)
forward_backward_intvar = tk.IntVar(root, value=0)
plan_name_strvar = tk.StringVar(root, value="test_plan_1")

# Create entry boxes for user input
up_down_ent = tk.Entry(root, textvariable=up_down_intvar)
left_right_ent = tk.Entry(root, textvariable=left_right_intvar)
forward_backward_ent = tk.Entry(root, textvariable=forward_backward_intvar)
plan_name_ent = tk.Entry(root, textvariable=plan_name_strvar)

# Create start flight button
controller_on_btn = tk.Button(root, text="Controller On/Off", command=controller_on)
start_flight_btn = tk.Button(root, text="Takeoff", command=start_flight)
update_flight_btn = tk.Button(root, text="Update Flight", command=update_flight)
land_btn = tk.Button(root, text="Land", command=land)
get_state_btn = tk.Button(root, text="Get Joystick States", command=joy_status)
fly_routine_btn = tk.Button(root, text="Fly Routine", command=start_routine)

# Create clear button
clear_btn = tk.Button(root, text="Clear Entries", command=clear_entries)

# Grid widgets onto root window
title_lbl.grid(column=0, row=0)
up_down_lbl.grid(column=0, row=1)
left_right_lbl.grid(column=0, row=2)
forward_backward_lbl.grid(column=0, row=3)
plan_name_lbl.grid(column=0, row=4)

up_down_ent.grid(column=1, row=1)
left_right_ent.grid(column=1, row=2)
forward_backward_ent.grid(column=1, row=3)
plan_name_ent.grid(column=1, row=4)

controller_on_btn.grid(column=0, row=5, columnspan=2)
start_flight_btn.grid(column=0, row=6, columnspan=2)
update_flight_btn.grid(column=0, row=7, columnspan=2)
fly_routine_btn.grid(column=0, row=8, columnspan=2)
land_btn.grid(column=0, row=9, columnspan=2)
clear_btn.grid(column=0, row=10, columnspan=2)
get_state_btn.grid(column=0, row=11, columnspan=2)

# Run the GUI
root.mainloop()
