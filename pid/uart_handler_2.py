import tkinter as tk
import serial
import time
import pandas as pd
import os
import threading

class Serial:
    def __init__(self, port, baudrate, timeout, mock=False):
        self.mock = mock
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        if self.mock:
            print(f"[Serial] Opening MOCK serial on port={port} baudrate={baudrate} timeout={timeout}")
        else:
            print(f"[Serial] Opening REAL serial on port={port}")
            self.real_serial = serial.Serial(port, baudrate, timeout=timeout)

    def write(self, data: bytes):
        if self.mock:
            print(f"[Serial] WRITE (mock): {data.hex()}")
        else:
            #print(f"[Serial] WRITE (real): {data.hex()}")
            self.real_serial.write(data)

    def read(self, size=1) -> bytes:
        if self.mock:
            print(f"[Serial] READ (mock) size={size}")
            return b'\x00' * size
        else:
            print(f"[Serial] READ (real) size={size}")
            return self.real_serial.read(size)

    def close(self):
        if self.mock:
            print("[Serial] Mock serial port closed.")
        else:
            self.real_serial.close()
            print("[Serial] Real serial port closed.")

MOCK = False
PORT = "COM5"
FPS = 30
UART_PORT = Serial(PORT, 9600, 5, mock=MOCK)

def hex_helper(int_in):
    hex_str = hex(int_in)[hex(int_in).find('x')+1:]
    if(len(hex_str)%2 != 0):
        hex_str = "0"+hex_str
    return bytes.fromhex(hex_str)

on = False

def controller_on():
    global on
    UART_PORT.write(bytes.fromhex("31"))
    on = not on
    #controller_status.set("TRUE" if on else "FALSE")
    print(f"Drone is ON: {on}")

def start_flight():
    global on
    if(on):
        #flight_status.set("Taking off/READY")
        print(f"Taking off...")
        UART_PORT.write(bytes.fromhex("32"))
        UART_PORT.write(bytes.fromhex("33"))
        UART_PORT.write(bytes.fromhex("34"))
    else:
        print(f"Controller is off. Takeoff aborted.")


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

def send_to_controller(next_inputs):
    UART_PORT.write(bytes.fromhex("04"))    #left/right
    UART_PORT.write(hex_helper(next_inputs[0]))
    UART_PORT.write(bytes.fromhex("03"))    #front/back
    UART_PORT.write(hex_helper(next_inputs[1]))
    UART_PORT.write(bytes.fromhex("01"))    #up/down
    UART_PORT.write(hex_helper(next_inputs[2]))
    UART_PORT.write(bytes.fromhex("02"))    #rotation
    UART_PORT.write(hex_helper(next_inputs[3]))

def all_neutral():
    UART_PORT.write(bytes.fromhex("04"))    #left/right
    UART_PORT.write(hex_helper(64))
    UART_PORT.write(bytes.fromhex("03"))    #front/back
    UART_PORT.write(hex_helper(64))
    UART_PORT.write(bytes.fromhex("01"))    #up/down
    UART_PORT.write(hex_helper(64))
    UART_PORT.write(bytes.fromhex("02"))    #rotation
    UART_PORT.write(hex_helper(64))

def land():
    #flight_status.set("Landing")
    print("Landing...")
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