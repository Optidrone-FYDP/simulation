import tkinter as tk
import serial
import time
import pandas as pd
import os


UART_PORT = serial.Serial("COM5", 9600, timeout=5)
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
    UART_PORT.write(bytes.fromhex("0f")) # stop drone

def update_flight():
    ud = up_down_var.get()
    lr = left_right_var.get()
    fb = forward_backward_var.get()
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
fly_routine_btn = tk.Button(buttons_frame, text="Fly Routine", command=start_routine, width=20)
land_btn = tk.Button(buttons_frame, text="Land", command=land, width=20)
get_state_btn = tk.Button(buttons_frame, text="Get Joystick States", command=joy_status, width=20)

controller_on_btn.pack(pady=5)
start_flight_btn.pack(pady=5)
update_flight_btn.pack(pady=5)
fly_routine_btn.pack(pady=5)
land_btn.pack(pady=5)
get_state_btn.pack(pady=5)

root.mainloop()
