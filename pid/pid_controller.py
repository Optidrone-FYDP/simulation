from time import time
import math

mode = "sim"

class dronePID:
    def __init__(self, Kp=1.2, Ki=0.1, Kd=0.5, max_a = 150):
        self.last_five_vel = []
        self.current_pos = [0, 0, 0, 0]
        self.prev_pos = [0, 0, 0, 0]
        self.current_acc = [0, 0, 0, 0]
        self.target_pos = [0, 0, 0, 0]
        self.target_acc = [0, 0, 0, 0]
        self.Kp = [Kp, Kp, Kp, Kp]
        self.Ki = [Ki, Ki, Ki, Ki]
        self.Kd = [Kd, Kd, Kd, Kd]
        self.error = [0, 0, 0, 0]
        self.max_i = 300
        self.prev_error = [0, 0, 0, 0]
        self.p = [0, 0, 0, 0]
        self.i = [0, 0, 0, 0]
        self.d = [0, 0, 0, 0]
        self.restoring = [0, 0, 0, 0]
        self.Kr = 1
        self.max_a = max_a #mm/s^2
        self.accel = 0
        self.accel_time = 0
        self.travel_time = 0
        self.profiled_path = 0
        self.setpoint_step = 0
        self.current_time = 0
        self.last_time = 0
        self.state = "hover" # or "follow"
        self.imm_target = [0,0]
    
    def setpoint(self, x_pos, y_pos, z_pos, rot):
        self.target_pos = [x_pos, y_pos, z_pos, rot]
        self.profiled_path = profile(self.target_pos)
        self.setpoint_step = -1
        self.state = "follow"
        next_setpoint()

    def next_setpoint(self):
        if self.state == "follow":
            self.setpoint_step += 1
            if self.setpoint_step >= len(self.profiled_path):
                self.state == "hover"
            else:
                self.imm_target[0] = self.profiled_path[self.stepoint_step][0][0]
                self.target_acc[0] = self.profiled_path[self.setpoint_step][0][1]
                self.imm_target[1] = self.profiled_path[self.stepoint_step][1][0]
                self.target_acc[1] = self.profiled_path[self.setpoint_step][1][1]
    
    def profile(self, target_pos):  # only x-y plane is profiled as z and rot have special cases
        # math to calculate profiled accelerations
        out = []
        total_dist = math.sqrt( (target_pos[0]-self.current_pos[0])**2 + (target_pos[1]-self.current_pos[1])**2 )
        self.travel_time = 2 + 0.01*total_dist/3
        self.accel_time = self.travel_time/2
        self.accel = [ (target_pos[0]-self.current_pos[0])/(self.accel_time**2), (target_pos[1]-self.current_pos[1])/(self.accel_time**2) ]
        s1 = [ [ (target_pos[0]-self.current_pos[0])/2 + self.current_pos[0], self.accel[0] ] , [ (target_pos[1]-self.current_pos[1])/2 + self.current_pos[1], self.accel[1] ] ]    # format is : [position, acceleration]
        s2 = [ [ target_pos[0], -self.accel[0] ], [ target_pos[1], -self.accel[1] ] ]
        out.append(s1)
        out.append(s2)
        return out  # should return 3d array of acceleration values with corresponding positions and accelerations we should be hitting
    
    def update(self):   # TODO: need to add code that goes to next set point if we are within tolerable range of setpoint (3cm ish), can ignore acceleration if we already reached the setpoint
        self.prev_pos = self.current_pos
        self.current_pos = get_pos()
        dt = get_dt()
        last_five_vel.append([(self.current_pos[0]-self.prev_pos[0])/dt, (self.current_pos[1]-self.prev_pos[1])/dt, (self.current_pos[2]-self.prev_pos[2])/dt])     # don't need to track rot velocity
        out = [0, 0, 0, 0]
        for i in range(len(2)): # x and y pid control
            pos_error = self.target_pos[i] - self.current_pos[i]
            if abs(pos_error) <= 30 and state == "follow":
                next_setpoint()
            self.error[i] =  self.target_acc[i] - self.current_acc[i]
            self.p[i] = self.Kp[i] * self.error[i]
            self.i[i] += self.Ki[i] * self.error[i]
            if self.i[i] > self.max_i:
                self.i[i] = self.max_i
            elif self.i[i] < -self.max_i:
                self.i[i] = -self.max_i
            self.d[i] = self.Kd[i] * (self.error[i] - self.prev_error[i])/dt
            self.restoring = self.Kr * pos_error
            self.prev_error = self.error
            out[i] = (state == "follow" ? self.p[i] + self.i[i] + self.d[i] : 0) + (abs(restoring_error) < 30 ? self.restoring : 0)
            if out[i] > 150:
                out[i] = 150
            elif out[i] < -150:
                out[i] = -150
            if out[i] > 0:
                out[i] = out[i]*63/150
            else:
                out[i] = out[i]*-64/-150
            out[i] = round(out[i] + 64)
        #TODO: special z-axis control (z-axis has higher accel and stopping input basically maintains position so we can do this one off position alone)
        self.error[2] = self.target_pos[2] - self.current_pos[2]
        self.p[2] = self.Kp[2] * self.error[2]
        self.i[2] += self.Ki[2] * self.error[2]
        if self.i[2] > self.max_i:
            self.i[2] = self.max_i
        elif self.i[2] < -self.max_i:
            self.i[2] = -self.max_i
        self.d[2] = self.Kd[2] * (self.error[2] - self.prev_error[2])/dt
        self.prev_error = self.error
        out[i] = self.p[i] + self.i[i] + self.d[i]
        if out[i] > 63:
            out[i] = 63
        elif out[i] < -64:
            out[i] = -64
        out[i] = round(out[i] + 64)
        #TODO: special rot control (rot moves too slow and moves at a very constant rate so just position pid is enough), we should also only do rot at the end of each path as changing the rot will force us to recalculate our path
        if state == "hover":
            self.error[3] = self.target_pos[3] - self.current_pos[3]
            self.p[i] = self.Kp[3] * self.error[3]
            self.i[i] += self.Ki[3] * self.error[3]
            if self.i[i] > self.max_i:
                self.i[i] = self.max_i
            elif self.i[i] < -self.max_i:
                self.i[i] = -self.max_i
            self.d[i] = self.Kd[3] * (self.error[3] - self.prev_error[3])/dt
            self.prev_error = self.error
            out[i] = self.p[i] + self.i[i] + self.d[i]
            if out[i] > 63:
                out[i] = 63
            elif out[i] < -64:
                out[i] = -64
            out[i] = round(out[i] + 64)
        else:
            out[i] = 64
        return out  # output format: [ x, y, z, rot ]
    
    def get_pos(self):  # in this func we need to preprocess data with rot reported by model/vicon, this will let us calculate velocity and accel correctly according to the drone's orientation, converting from absolute coordinates to coordinates relative to drone's axis
        # exception is the drone's rotation which must be collected and expressed in relation to the absolute coordinates (calibration coordinates)
        global mode
        if mode=="sim":
            continue #get feedback from lstm model
        elif mode=="vicon":
            continue #get position from vicon cameras
        else:
            print("invalid mode")
            exit()
        return
    
    def get_dt(self):
        self.current_time = time.now()
        dt = self.current_time - self.last_time
        self.last_time = self.current_time
        return dt
