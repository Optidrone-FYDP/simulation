import time
import math
import sys

# x is left/right
# y is forward/backward

class dronePID:
    def __init__(self, Kp=0.1, Ki=0.001, Kd=0.05, max_a = 150, mode = "sim"):
        self.current_pos = [0, 0, 0, 0]
        self.prev_pos = [0, 0, 0, 0]
        self.target_pos = [0, 0, 0, 0]
        self.starting_pos = [0, 0, 0, 0]
        self.Kp = [Kp, Kp, Kp, 0.1]
        self.Ki = [Ki, Ki, Ki, 0.001]
        self.Kd = [Kd, Kd, Kd, 0.01]
        self.error = [0, 0, 0, 0]
        self.max_i = 300
        self.prev_error = [0, 0, 0, 0]
        self.p = [0, 0, 0, 0]
        self.i = [0, 0, 0, 0]
        self.d = [0, 0, 0, 0]
        self.restoring = [0, 0, 0, 0]
        self.Kr = 1
        self.max_a = max_a #mm/s^2
        self.current_time = 0
        self.last_time = 0
        self.state = "hover"
        self.mode = mode # sim or vicon
        self.reached_target = False
        self.rot_target = 0
    
    def setpoint(self, x_pos, y_pos, z_pos, rot):
        self.target_pos = [x_pos, y_pos, z_pos, rot]
        self.state = "start"
        self.next_setpoint()

    def find_travel_angle():
        o = (self.target_pos[1] - self.starting_pos[1])
        a = (self.target_pos[0] - self.starting_pos[0])

        rot_target = 0
        if a = 0 and o > 0:
            self.rot_target = 3*math.pi/2
        elif a = 0 and o < 0:
            self.rot_target = math.pi/2
        elif a = 0 and o = 0:
            self.rot_target = 0
        elif o > 0 and a > 0 or o < 0 and a > 0:
            self.rot_target = math.pi + math.atan(o/a)
        elif o > 0 and a < 0:
            self.rot_target = 2*math.pi + math.atan(o/a)
        elif o < 0 and a < 0:
            self.rot_target = math.atan(o/a)
        else:
            self.rot_target = 0

    def next_setpoint(self):
        ''' states:
            start -> start here
            starting_rotation -> do rotation to line up with target
            follow -> do x/y pid to reach target, y will do most of the travelling and x will be error correction if drift is present
            ending_rotation -> rotation at end to match desired rotation
            hover -> all states done and holding position
        '''
        if self.state is "start":
            self.reached_target = False
            self.starting_pos = self.current_pos
            find_travel_angle()
            self.state = "starting_rotation"
        elif self.state is "starting_rotation":
            if math.fabs(self.current_pos[3] - self.rot_target) < 0.0087:
                self.state = "follow"
        elif self.state is "follow":
            if dist(self.target_pos[0], self.current_pos[0], self.target_pos[1], self.current_pos[1]) < 30:
                self.state = "ending_rotation"
        elif self.state is "ending_rotation":
            if math.fabs(self.current_pos[3] - self.target_pos[3]) < 0.0087:
                self.state = "hover"
            #if dist(self.target_pos[0], self.current_pos[0], self.target_pos[1], self.current_pos[1]) > 30:
            #    self.state = "follow"
        elif self.state is "hover":
            self.reached_target = True

    def find_nearest_point():
        t = ((self.current_pos[0] - self.starting_pos[0])*(self.target_pos[0] - self.starting_pos[0]) + (self.current_pos[1] - self.starting_pos[1])*(self.target_pos[1] - self.starting_pos[1]))/((self.target_pos[0]-self.starting_pos[0])**2+(self.target_pos[1]-self.starting_pos[1])**2)
        closest_x = self.starting_pos[0] + t*(self.target_pos[0] - self.starting_pos[0])
        closest_y = self.starting_pos[1] + t*(self.target_pos[1] - self.starting_pos[1])
        return [closest_x, closest_y]

    def above_below():
        sign = (self.target_pos[0] - self.starting_pos[0])(self.current_pos[1] - self.starting_pos[1]) - (self.target_pos[1] - self.starting_pos[1])(self.current_pos[0] - self.starting_pos[0])
        if sign < 0:
            return 1
        elif sign > 0:
            return -1
        else:
            return 0

    def dist(x1, x2, y1, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def update(self):
        dt = self.get_dt()

        nearest = self.find_nearest_point()

        out = [64, 64, 64, 64]
        if not self.state is "starting_rotation" or not self.state is "ending_rotation":        # do not run xyz pids when adjusting rotation, could be bad bad bad

            # x-axis
            self.error[0] =  above_below()*dist(nearest[0], self.current_pos[0], nearest[1], self.current_pos[1])
            
            self.p[0] = self.Kp[0] * self.error[0]
            
            self.i[0] += self.Ki[0] * self.error[0]
            if self.i[0] > self.max_i:
                self.i[0] = self.max_i
            elif self.i[0] < -self.max_i:
                self.i[0] = -self.max_i
                
            self.d[0] = self.Kd[0] * (self.error[0] - self.prev_error[0])/dt
            
            self.prev_error = self.error
            
            out[0] = self.p[0] + self.i[0] + self.d[0]
            
            if out[0] > 150:
                out[0] = 150
            elif out[0] < -150:
                out[0] = -150
            if out[0] > 0:
                out[0] = out[0]*63/150
            else:
                out[0] = out[0]*-64/-150
            if out[0] > 25:
                out[0] = 25
            elif out[0] < -25:
                out[0] = -25
            out[0] = round(out[0] + 64)

            # y-axis
            self.error[1] = dist(self.target_pos[0], nearest[0], self.target_pos[1], nearest[1])
            
            self.p[1] = self.Kp[1] * self.error[1]
            
            self.i[1] += self.Ki[1] * self.error[1]
            if self.i[1] > self.max_i:
                self.i[1] = self.max_i
            elif self.i[1] < -self.max_i:
                self.i[1] = -self.max_i
                
            self.d[1] = self.Kd[1] * (self.error[1] - self.prev_error[1])/dt
            
            self.prev_error = self.error
            
            out[1] = self.p[1] + self.i[1] + self.d[1]
            
            if out[1] > 150:
                out[1] = 150
            elif out[1] < -150:
                out[1] = -150
            if out[1] > 0:
                out[1] = out[0]*63/150
            else:
                out[1] = out[0]*-64/-150
            if out[1] > 25:
                out[1] = 25
            elif out[1] < -25:
                out[1] = -25
            out[1] = round(out[0] + 64)

            # special z-axis control (z-axis has higher accel and stopping input basically maintains position so we can do this one off position alone)
            self.error[2] = self.target_pos[2] - self.current_pos[2]
            
            self.p[2] = self.Kp[2] * self.error[2]
            
            self.i[2] += self.Ki[2] * self.error[2]
            if self.i[2] > self.max_i:
                self.i[2] = self.max_i
            elif self.i[2] < -self.max_i:
                self.i[2] = -self.max_i
                
            self.d[2] = self.Kd[2] * (self.error[2] - self.prev_error[2])/dt
            
            self.prev_error = self.error
            
            out[2] = self.p[2] + self.i[2] + self.d[2]
            
            if out[2] > 16:
                out[2] = 16
            elif out[2] < -16:
                out[2] = -16
            out[2] = round(out[2] + 64)

        # special rot control (rot moves too slow and moves at a very constant rate so just position pid is enough), we should also only do rot at the end of each path as changing the rot will force us to recalculate our path
        if self.state is "ending_rotation":
            target = self.target_pos[3]
        else:
            target = self.rot_target
        self.error[3] = target - self.current_pos[3]
        
        self.p[3] = self.Kp[3] * self.error[3]
        
        self.i[3] += self.Ki[3] * self.error[3]
        if self.i[3] > self.max_i:
            self.i[3] = self.max_i
        elif self.i[3] < -self.max_i:
            self.i[3] = -self.max_i
            
        self.d[3] = self.Kd[3] * (self.error[3] - self.prev_error[3])/dt
        
        self.prev_error = self.error
        
        out[3] = self.p[3] + self.i[3] + self.d[3]
        if out[3] > 63:
            out[3] = 63
        elif out[3] < -64:
            out[3] = -64
        out[3] = round(out[3] + 64)

        self.next_setpoint()
        
        self.prev_pos = self.current_pos
        
        return out  # output format: [ x, y, z, rot ]
    
    def get_pos(self, new_pos):  # in this func we need to preprocess data with rot reported by model/vicon, this will let us calculate velocity and accel correctly according to the drone's orientation, converting from absolute coordinates to coordinates relative to drone's axis
        # exception is the drone's rotation which must be collected and expressed in relation to the absolute coordinates (calibration coordinates)
        #print(f"current_position fed to drone: {new_pos}")

        if self.mode=="sim":
            self.current_pos = [new_pos[0], new_pos[1], new_pos[2]]

        elif self.mode=="vicon":
            self.current_pos = [new_pos[0], new_pos[1], new_pos[2], (new_pos[5] if new_pos[5] >= 0 else 2*math.pi + new_pos[5])]

        else:
            print("invalid mode")
            sys.exit(1)
    
    def get_dt(self):
        if self.mode=="sim":
            dt = 1/30
        elif self.mode=="vicon":
            self.current_time = time.time()
            dt = self.current_time - self.last_time
            self.last_time = self.current_time
        else:
            print("invalid mode")
            sys.exit(1)
        return dt
