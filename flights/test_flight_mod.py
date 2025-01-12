import tkinter as tk
import serial
import time
import json
import pandas as pd
import os

UART_PORT = serial.Serial("COM6", 9600, timeout=5)
FPS = 30

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
    plan_name = flight_plan_var.get() + ".csv"
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
        step_duration_seconds = step_duration_frames / FPS
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
    for _ in range(5):
        val = UART_PORT.read()
        status.append(val)
    return status

# def clear_entries():
#     up_down_intvar.set(64)
#     left_right_intvar.set(64)
#     forward_backward_intvar.set(64)
#     plan_name_strvar.set("")

# GUI

# # Create root window and main title
# root = tk.Tk()
# root.title("OptiDrone Test Flight Controls")
# title_lbl = tk.Label(root, text="OptiDrone Test Flight Controls")

# # Create labels
# up_down_lbl = tk.Label(root, text="UP/DOWN (joystick n)")
# left_right_lbl = tk.Label(root, text="LEFT/RIGHT (joystick n)")
# forward_backward_lbl = tk.Label(root, text="FORWARD/BACKWARD (joystick n)")
# plan_name_lbl = tk.Label(root, text="Flight Plan Name:")

# # Create intvariables
# up_down_intvar = tk.IntVar(root, value=0)
# left_right_intvar = tk.IntVar(root, value=0)
# forward_backward_intvar = tk.IntVar(root, value=0)
# plan_name_strvar = tk.StringVar(root, value="04_circle")

# # Create entry boxes for user input
# up_down_ent = tk.Entry(root, textvariable=up_down_intvar)
# left_right_ent = tk.Entry(root, textvariable=left_right_intvar)
# forward_backward_ent = tk.Entry(root, textvariable=forward_backward_intvar)
# plan_name_ent = tk.Entry(root, textvariable=plan_name_strvar)

# # Create start flight button
# controller_on_btn = tk.Button(root, text="Controller On/Off", command=controller_on)
# start_flight_btn = tk.Button(root, text="Takeoff", command=start_flight)
# update_flight_btn = tk.Button(root, text="Update Flight", command=update_flight)
# land_btn = tk.Button(root, text="Land", command=land)
# get_state_btn = tk.Button(root, text="Get Joystick States", command=joy_status)
# fly_routine_btn = tk.Button(root, text="Fly Routine", command=start_routine)

# # Create clear button
# clear_btn = tk.Button(root, text="Clear Entries", command=clear_entries)

# # Grid widgets onto root window
# title_lbl.grid(column=0, row=0)
# up_down_lbl.grid(column=0, row=1)
# left_right_lbl.grid(column=0, row=2)
# forward_backward_lbl.grid(column=0, row=3)
# plan_name_lbl.grid(column=0, row=4)

# up_down_ent.grid(column=1, row=1)
# left_right_ent.grid(column=1, row=2)
# forward_backward_ent.grid(column=1, row=3)
# plan_name_ent.grid(column=1, row=4)

# controller_on_btn.grid(column=0, row=5, columnspan=2)
# start_flight_btn.grid(column=0, row=6, columnspan=2)
# update_flight_btn.grid(column=0, row=7, columnspan=2)
# fly_routine_btn.grid(column=0, row=8, columnspan=2)
# land_btn.grid(column=0, row=9, columnspan=2)
# clear_btn.grid(column=0, row=10, columnspan=2)
# get_state_btn.grid(column=0, row=11, columnspan=2)

# # Run the GUI
# root.mainloop()

import threading

def start_routine_threaded():
    thread = threading.Thread(target=start_routine)
    thread.start()

def refresh_flight_plans():
    flight_plans = [f for f in os.listdir("plans") if f.endswith(".csv")]
    flight_plan_menu['menu'].delete(0, 'end')
    for plan in flight_plans:
        flight_plan_menu['menu'].add_command(label=plan, command=tk._setit(flight_plan_var, plan))
    if flight_plans:
        flight_plan_var.set(flight_plans[0])
    else:
        flight_plan_var.set("")

root = tk.Tk()
root.title("OptiDrone Test Flight Controls")
root.geometry("400x500")

status_frame = tk.Frame(root)
status_frame.pack(pady=10)

controller_status = tk.StringVar(value="Off")
flight_status = tk.StringVar(value="Idle")

controller_lbl = tk.Label(status_frame, text="Controller Status:")
controller_val = tk.Label(status_frame, textvariable=controller_status)
flight_lbl = tk.Label(status_frame, text="Flight Status:")
flight_val = tk.Label(status_frame, textvariable=flight_status)

controller_lbl.grid(row=0, column=0, padx=5, pady=5, sticky='e')
controller_val.grid(row=0, column=1, padx=5, pady=5, sticky='w')
flight_lbl.grid(row=1, column=0, padx=5, pady=5, sticky='e')
flight_val.grid(row=1, column=1, padx=5, pady=5, sticky='w')

plan_frame = tk.Frame(root)
plan_frame.pack(pady=10)

flight_plan_var = tk.StringVar()
flight_plan_label = tk.Label(plan_frame, text="Flight Plan:")
flight_plan_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

flight_plan_menu = tk.OptionMenu(plan_frame, flight_plan_var, "")
flight_plan_menu.config(width=20)
flight_plan_menu.grid(row=0, column=1, padx=5, pady=5)

refresh_btn = tk.Button(plan_frame, text="Refresh Plans", command=refresh_flight_plans)
refresh_btn.grid(row=0, column=2, padx=5, pady=5)

refresh_flight_plans()

# Control Variables Frame
control_vars_frame = tk.Frame(root)
control_vars_frame.pack(pady=10)

up_down_var = tk.IntVar(value=64)
left_right_var = tk.IntVar(value=64)
forward_backward_var = tk.IntVar(value=64)

up_down_lbl = tk.Label(control_vars_frame, text="UP/DOWN (potZ):")
left_right_lbl = tk.Label(control_vars_frame, text="LEFT/RIGHT (potX):")
forward_backward_lbl = tk.Label(control_vars_frame, text="FORWARD/BACKWARD (potY):")

up_down_entry = tk.Entry(control_vars_frame, textvariable=up_down_var, width=10)
left_right_entry = tk.Entry(control_vars_frame, textvariable=left_right_var, width=10)
forward_backward_entry = tk.Entry(control_vars_frame, textvariable=forward_backward_var, width=10)

up_down_lbl.grid(row=0, column=0, padx=5, pady=5, sticky='e')
up_down_entry.grid(row=0, column=1, padx=5, pady=5)
left_right_lbl.grid(row=1, column=0, padx=5, pady=5, sticky='e')
left_right_entry.grid(row=1, column=1, padx=5, pady=5)
forward_backward_lbl.grid(row=2, column=0, padx=5, pady=5, sticky='e')
forward_backward_entry.grid(row=2, column=1, padx=5, pady=5)

buttons_frame = tk.Frame(root)
buttons_frame.pack(pady=20)

controller_on_btn = tk.Button(buttons_frame, text="Controller On/Off", command=controller_on, width=20)
start_flight_btn = tk.Button(buttons_frame, text="Takeoff", command=start_flight, width=20)
update_flight_btn = tk.Button(buttons_frame, text="Update Flight", command=update_flight, width=20)
fly_routine_btn = tk.Button(buttons_frame, text="Fly Routine", command=start_routine_threaded, width=20)
land_btn = tk.Button(buttons_frame, text="Land", command=land, width=20)
get_state_btn = tk.Button(buttons_frame, text="Get Joystick States", command=joy_status, width=20)

controller_on_btn.pack(pady=5)
start_flight_btn.pack(pady=5)
update_flight_btn.pack(pady=5)
fly_routine_btn.pack(pady=5)
land_btn.pack(pady=5)
get_state_btn.pack(pady=5)

root.mainloop()
