import tkinter as tk
import serial
import time
import json
import pandas as pd
import os

UART_PORT = serial.Serial("COM5", 9600, timeout=5)
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
    UART_PORT.write(bytes.fromhex("31"))
    on = not on

def start_flight():
    global on
    if(on):
        UART_PORT.write(bytes.fromhex("32"))
        UART_PORT.write(bytes.fromhex("33"))
        UART_PORT.write(bytes.fromhex("34"))

def start_routine():
    plan_name = plan_name_strvar.get() + ".csv"
    plan_file = os.path.join("plans", plan_name)

    # print(plan_file)
    if not os.path.isfile(plan_file):
        print(f"Error: The test plan file '{plan_file}' does not exist.")
        return

    try:
        plan_df = pd.read_csv(plan_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    required_columns = {"potX", "potY", "potZ", "potRot", "duration"}
    if not required_columns.issubset(plan_df.columns):
        print(f"Error: CSV file must contain columns: {required_columns}")
        return
    
    plan_size = len(plan_df)
    start_time = time.time()
    previous_time = start_time
    # pots = []

    i = 0
    while i < plan_size:
        row = plan_df.iloc[i]
        step_duration_frames = row['duration']
        step_duration_seconds = step_duration_frames / fps
        step_end_time = previous_time + step_duration_seconds

        potZ = int(row['potZ'])
        potX = int(row['potX'])
        potY = int(row['potY'])
        potRot = int(row['potRot'])

        UART_PORT.write(bytes.fromhex("01"))    # up/down -> Z
        UART_PORT.write(hex_helper(int(row['potZ'])))

        UART_PORT.write(bytes.fromhex("02"))    # rotation -> Rot
        UART_PORT.write(hex_helper(potRot))

        UART_PORT.write(bytes.fromhex("03"))    # front/back -> Y
        UART_PORT.write(hex_helper(potY))

        UART_PORT.write(bytes.fromhex("04"))    # left/right -> X
        UART_PORT.write(hex_helper(potX))

        current_time = time.perf_counter()
        # pot_status = [current_time - start_time] + joy_status()
        # pots.append(pot_status)

        print(f"Step {i}: potX {potX} - potY {potY} - potZ {potZ} - potRot - {potRot}")

        while True:
            current_time = time.perf_counter()
            remaining_time = step_end_time - current_time
            if remaining_time <= 0:
                break
            time.sleep(min(remaining_time, 0.001)) # 1 ms sleep

        previous_time = step_end_time
        i += 1

        print(f"Step {i} completed after {step_duration_frames} frames")
    
    # for i in range(len(pots)):
    #     pots[i][1] = int.from_bytes(pots[i][1], 'big')
    #     pots[i][2] = int.from_bytes(pots[i][2], 'big')
    #     pots[i][3] = int.from_bytes(pots[i][3], 'big')
    #     pots[i][4] = int.from_bytes(pots[i][4], 'big')
    #     pots[i][5] = int.from_bytes(pots[i][5], 'big')
    UART_PORT.write(bytes.fromhex("0f")) # stop drone


def update_flight():
    ud = up_down_intvar.get()
    lr = left_right_intvar.get()
    fb = forward_backward_intvar.get()
    print("FLYING ", ud, " JOYSTICK UP/DOWN")
    print("FLYING ", lr, " JOYSTICK LEFT/RIGHT")
    print("FLYING ", fb, " JOYSTICK FORWARD/BACKWARD")
    print("======================================")
    UART_PORT.write(bytes.fromhex("01"))
    UART_PORT.write(hex_helper(ud))
    UART_PORT.write(bytes.fromhex("03"))
    UART_PORT.write(hex_helper(fb))
    UART_PORT.write(bytes.fromhex("04"))
    UART_PORT.write(hex_helper(lr))

def land():
    UART_PORT.write(bytes.fromhex("10"))

def joy_status():
    UART_PORT.write(bytes.fromhex("11"))
    status = []
    for i in range(5):
        val = UART_PORT.read()
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