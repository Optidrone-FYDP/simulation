from time import time

class dronePID:
    def __init__(self, Kp=1.2, Ki=0.1, Kd=0.5, max_a = 500):
        self.last_five_vel = []
        self.current_pos = [0, 0, 0]
        self.prev_pos = [0, 0, 0]
        self.current_acc = [0, 0, 0]
        self.target_pos = [0, 0, 0]
        self.target_acc = [0, 0, 0]
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error = [0, 0, 0]
        self.prev_error = [0, 0, 0]
        self.p = [0, 0, 0]
        self.i = [0, 0, 0]
        self.d = [0, 0, 0]
        self.restoring = [0, 0, 0]
        self.Kr = 0.02
        self.max_a = max_a #mm/s^2
        self.accel = 220
        self.travel_time = 0
        self.profiled_path = 0
        self.profiling_step = 0
    def setpoint(self, x_pos, y_pos, z_pos):
        self.target = [x_acc, y_acc, z_acc]
        self.profiled_path = profile()
        self.profiling_step = 0
    def profile(self):
        # math to calculate profiled accelerations
        return  # should return 2d array of acceleration values to hit for each axis per frame
    def update(self):
        self.prev_pos = self.current_pos
        self.current_pos = get_pos()
        dt = get_dt()
        last_five_vel.append([(self.current_pos[0]-self.prev_pos[0])/dt, (self.current_pos[1]-self.prev_pos[1])/dt, (self.current_pos[2]-self.prev_pos[2])/dt])
        out = [0, 0, 0]
        for i in range(len(3)):
            self.error[i] =  abs(self.target_acc[i] - self.current_acc[i])
            self.p[i] = Kp * self.error[i]
            self.i[i] += Ki * self.error[i]
            self.d[i] = Kd * self.error[i] - self.prev_error[i]
            self.restoring = self.Kr * (self.target_pos[i] - self.current_pos[i])
            self.prev_error = self.error
            out[i] = self.p[i] + self.i[i] + self.d[i] + self.restoring
    def get_pos():
        return #do nothing
    def get_dt():
        return #do nothing
