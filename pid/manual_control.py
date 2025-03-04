from uart_handler_2 import *
from vicon_handler import *
import keyboard
import time

if __name__ == "__main__":
    pot = [64, 64, 64, 64]
    w_pressed = False
    a_pressed = False
    s_pressed = False
    d_pressed = False
    r_pressed = False
    w_prev_pressed = False
    a_prev_pressed = False
    s_prev_pressed = False
    d_prev_pressed = False
    r_prev_pressed = False
    time.sleep(5)
    print("controller on")
    controller_on()
    time.sleep(4)
    print("starting flight")
    start_flight()
    time.sleep(6)
    while True:
        if keyboard.is_pressed('w'):
            w_pressed = True
        else:
            w_pressed = False
        if keyboard.is_pressed('a'):
            a_pressed = True
        else:
            a_pressed = False
        if keyboard.is_pressed('s'):
            s_pressed = True
        else:
            s_pressed = False
        if keyboard.is_pressed('d'):
            d_pressed = True
        else:
            d_pressed = False
        if keyboard.is_pressed('r'):
            r_pressed = True
        else:
            r_pressed = False
            
        if w_pressed != w_prev_pressed and w_pressed:
            pot[0] += 1
            send_to_controller(pot)
            print(pot)
        if a_pressed != a_prev_pressed and a_pressed:
            pot[1] -= 1
            send_to_controller(pot)
            print(pot)
        if s_pressed != s_prev_pressed and s_pressed:
            pot[0] -= 1
            send_to_controller(pot)
            print(pot)
        if d_pressed != d_prev_pressed and d_pressed:
            pot[1] += 1
            send_to_controller(pot)
            print(pot)
        if r_pressed != r_prev_pressed and r_pressed:
            pot = [64, 64, 64, 64]
            send_to_controller(pot)
            print(pot)
        w_prev_pressed = w_pressed
        a_prev_pressed = a_pressed
        s_prev_pressed = s_pressed
        d_prev_pressed = d_pressed
        r_prev_pressed = r_pressed
        time.sleep(0.05)