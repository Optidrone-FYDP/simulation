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
        self.Kp = [0.2, 70, 0.1, 0.3]
        self.Ki = [0, 0, 0, 0.002]
        self.Kd = [0, 0, 0.05, 0.25]
        self.error = [0, 0, 0, 0]
        self.max_i = 200
        self.prev_error = [0, 0, 0, 0]
        self.p = [0, 0, 0, 0]
        self.i = [0, 0, 0, 0]
        self.d = [0, 0, 0, 0]
        self.current_time = 0
        self.last_time = 0
        self.state = "hover"
        self.mode = mode # sim or vicon
        self.reached_target = False
        self.rot_target = 0
        self.dead_zone_pos = 81
        self.dead_zone_neg = 47
        self.settling_frames = 30
        self.settling = 0
        self.total_dist = 0
        self.calc_stop = True
        self.stopping_point = [0, 0]
        self.max_vel = 0
    
    def setpoint(self, x_pos, y_pos, z_pos, rot):
        self.target_pos = [x_pos, y_pos, z_pos, rot]
        self.state = "start"
        self.next_setpoint()

    def find_travel_angle(self):
        self.rot_target = self.find_angle(self.target_pos, self.starting_pos)

    def find_angle(self, target_point, start_point):
        o = (target_point[1] - start_point[1])
        a = (target_point[0] - start_point[0])

        if a == 0 and o > 0:
            return 0
        elif a == 0 and o < 0:
            return math.pi
        elif a == 0 and o == 0:
            return 0
        elif a > 0:
            return 3*math.pi/2 + math.atan(o/a)
        elif a < 0:
            return math.pi/2 + math.atan(o/a)
        else:
            return 0

    def next_setpoint(self):
        ''' states:
            start -> start here
            starting_rotation -> do rotation to line up with target
            follow -> do x/y pid to reach target, y will do most of the travelling and x will be error correction if drift is present
            ending_rotation -> rotation at end to match desired rotation
            hover -> all states done and holding position
        '''
        if self.state == "start":
            self.reached_target = False
            self.starting_pos = self.current_pos
            self.total_dist = self.dist(self.target_pos[0], self.starting_pos[0], self.target_pos[1], self.starting_pos[1])
            self.find_travel_angle()
            self.settling = 0
            self.calc_stop = True
            self.state = "starting_rotation"
        elif self.state == "starting_rotation":
            if math.fabs(self.current_pos[3] - self.rot_target) < 0.0131:
                self.settling += 1
            else:
                self.settling = 0
            if self.settling > self.settling_frames:
                self.state = "follow"
                self.settling = 0
        elif self.state == "follow":
            if self.dist(self.target_pos[0], self.current_pos[0], self.target_pos[1], self.current_pos[1]) < 50 and math.fabs(self.target_pos[2]-self.current_pos[2] < 100):
                self.settling += 1
            else:
                self.settling = 0
            if self.settling > self.settling_frames:
                self.state = "ending_rotation"
                self.settling = 0
        elif self.state == "ending_rotation":
            if math.fabs(self.current_pos[3] - self.target_pos[3]) < 0.0174:
                self.settling += 1
            else:
                self.settling = 0
            if self.settling > self.settling_frames:
                self.state = "hover"
                self.settling = 0
            #if self.dist(self.target_pos[0], self.current_pos[0], self.target_pos[1], self.current_pos[1]) > 30:
            #    self.state = "follow"
        elif self.state == "hover":
            self.reached_target = True

    def find_nearest_point(self):
        t = ((self.current_pos[0] - self.starting_pos[0])*(self.target_pos[0] - self.starting_pos[0]) + (self.current_pos[1] - self.starting_pos[1])*(self.target_pos[1] - self.starting_pos[1]))/((self.target_pos[0]-self.starting_pos[0])**2+(self.target_pos[1]-self.starting_pos[1])**2)
        closest_x = self.starting_pos[0] + t*(self.target_pos[0] - self.starting_pos[0])
        closest_y = self.starting_pos[1] + t*(self.target_pos[1] - self.starting_pos[1])
        return [closest_x, closest_y]

    def above_below(self):
        sign = (self.target_pos[0] - self.starting_pos[0])*(self.current_pos[1] - self.starting_pos[1]) - (self.target_pos[1] - self.starting_pos[1])*(self.current_pos[0] - self.starting_pos[0])
        if sign < 0:
            return -1
        elif sign > 0:
            return 1
        else:
            return 0

    def dist(self, x1, x2, y1, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def update(self):

        dt = self.get_dt()

        print(self.state)

        nearest = self.find_nearest_point()
        print(f"target pos: {self.target_pos}")

        out = [64, 64, 64, 64]
        if (not self.state == "starting_rotation" and not self.state == "ending_rotation") and self.calc_stop:        # do not run xyz pids when adjusting rotation, could be bad bad bad

            #if self.state == "hover":
            #    self.starting_pos = self.current_pos
            #    nearest = self.current_pos

            # x-axis
            self.error[0] =  self.above_below()*self.dist(nearest[0], self.current_pos[0], nearest[1], self.current_pos[1])
            
            #print(self.error[0])

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
                out[0] = out[0]*5/150
            else:
                out[0] = out[0]*-5/-150
            if out[0] > 5:
                out[0] = 5
            elif out[0] < -5:
                out[0] = -5
            if math.fabs(out[0]) < 0.3:
                out[0] = 64
            else:
                out[0] = math.ceil(out[0] + self.dead_zone_pos) if out[0] > 0 else math.ceil(out[0] + self.dead_zone_neg)
            #if math.fabs(self.error[0]) < 30:
            #    out[0] = 64

            # y-axis

            vel = self.dist(self.current_pos[0], self.prev_pos[0], self.current_pos[1], self.prev_pos[1])/dt

            if vel > self.max_vel:
                self.max_vel = vel

            ref_vel = 1050
            ref_stopping_dist = 260
            offset = 75
            power = 1.4
            stopping_dist = ( (ref_stopping_dist-offset)/(ref_vel**power) ) * (vel**power) + offset
            self.stopping_point = [self.target_pos[0] - stopping_dist * -math.sin(self.rot_target), self.target_pos[1] - stopping_dist * math.cos(self.rot_target)]

            print(f"vel: {vel}")
            print(f"stopping point: {self.stopping_point}")

            print(f"nearest point: {nearest}")
            dist_error = self.dist(self.target_pos[0], nearest[0], self.target_pos[1], nearest[1])
            dist_to_stop = self.dist(self.stopping_point[0], nearest[0], self.stopping_point[1], nearest[1])
            direction_to_stop = self.find_angle(self.stopping_point, nearest)
            direction = self.find_angle(self.target_pos, nearest)
            print(f"direction: {direction}")
            print(f"target rotation: {self.rot_target}")
            if math.fabs(direction_to_stop - self.rot_target) < 0.000001:
                self.error[1] = dist_error
            else:
                self.calc_stop = False
                self.error[1] = -dist_error
            
            print(f"y-axis error: {self.error[1]}")
            print(f"x-axis error: {self.error[0]}")

            self.p[1] = self.Kp[1] * self.error[1]
            
            self.i[1] += self.Ki[1] * self.error[1]
            if self.i[1] > self.max_i:
                self.i[1] = self.max_i
            elif self.i[1] < -self.max_i:
                self.i[1] = -self.max_i
                
            self.d[1] = self.Kd[1] * (self.error[1] - self.prev_error[1])/dt
            
            self.prev_error = self.error
            
            out[1] = self.p[1] + self.i[1] - self.d[1]
            
            if out[1] > 63:
                out[1] = 63
            elif out[1] < 0:
                out[1] = 0
            out[1] = math.ceil(out[1] + 64)

            # if out[1] > 150:
            #     out[1] = 150
            # elif out[1] < -150:
            #     out[1] = -150
            # if out[1] > 0:
            #     out[1] = out[1]*(127-self.dead_zone_pos)/150
            # else:
            #     out[1] = out[1]*-self.dead_zone_neg/-150
            # if math.fabs(out[1]) < 0.01:
            #     out[1] = 64
            # else:
            #     out[1] = math.ceil(out[1] + self.dead_zone_pos) if out[1] > 0 else math.ceil(out[1] + self.dead_zone_neg)


        # special z-axis control (z-axis has higher accel and stopping input basically maintains position so we can do this one off position alone)
        self.error[2] = self.target_pos[2] - self.current_pos[2]
        
        self.p[2] = self.Kp[2] * self.error[2]
        
        self.i[2] += self.Ki[2] * self.error[2]
        if self.i[2] > 1:
            self.i[2] = 1
        elif self.i[2] < -1:
            self.i[2] = -1
            
        self.d[2] = self.Kd[2] * (self.error[2] - self.prev_error[2])/dt
        
        self.prev_error = self.error
        
        out[2] = self.p[2] + self.i[2] + self.d[2]
        
        if out[2] > 150:
            out[2] = 150
        elif out[2] < -150:
            out[2] = -150
        if out[2] > 0:
            out[2] = out[2]*10/150
        else:
            out[2] = out[2]*-10/-150
        if out[2] > 10:
            out[2] = 10
        elif out[2] < -10:
            out[2] = -10
        if math.fabs(out[2]) < 0.9:
            out[2] = 64
        else:
            out[2] = math.ceil(out[2] + self.dead_zone_pos) if out[2] > 0 else math.ceil(out[2] + self.dead_zone_neg)
        # if math.fabs(self.error[2]) < 100:
        #     out[2] = 64

        # special rot control (rot moves too slow and moves at a very constant rate so just position pid is enough), we should also only do rot at the end of each path as changing the rot will force us to recalculate our path
        target = self.rot_target

        print(f"rotation target: {target}")

        if self.current_pos[3] > target:
            left_way_error = 2*math.pi - self.current_pos[3] + target
            right_way_error = self.current_pos[3] - target
        else:
            right_way_error = 2*math.pi - target + self.current_pos[3]
            left_way_error = target - self.current_pos[3]

        if left_way_error < right_way_error:
            self.error[3] = -left_way_error
        else:
            self.error[3] = right_way_error
        
        print(f"rotational error: {self.error[3]}")

        self.p[3] = self.Kp[3] * self.error[3]
        
        self.i[3] += self.Ki[3] * self.error[3]
        if self.i[3] > 3:
            self.i[3] = 3
        elif self.i[3] < -3:
            self.i[3] = -3
            
        self.d[3] = self.Kd[3] * (self.error[3] - self.prev_error[3])/dt
        
        self.prev_error = self.error

        out[3] = self.p[3] + self.i[3] + self.d[3]
        if out[3] > 5:
            out[3] = 5
        elif out[3] < -5:
            out[3] = -5
        if math.fabs(out[3]) < 0.017:
            out[3] = 64
        else:
            out[3] = math.ceil(out[3] + self.dead_zone_pos) if out[3] > 0 else math.ceil(out[3] + self.dead_zone_neg)
        #if math.fabs(self.error[3]) < 0.0174:
        #    out[3] = 64

        self.next_setpoint()
        
        return out  # output format: [ x, y, z, rot ]
    
    def get_pos(self, new_pos):  # in this func we need to preprocess data with rot reported by model/vicon, this will let us calculate velocity and accel correctly according to the drone's orientation, converting from absolute coordinates to coordinates relative to drone's axis
        # exception is the drone's rotation which must be collected and expressed in relation to the absolute coordinates (calibration coordinates)
        #print(f"current_position fed to drone: {new_pos}")

        if self.mode=="sim":
            self.current_pos = [new_pos[0], new_pos[1], new_pos[2]]

        elif self.mode=="vicon":
            self.prev_pos = self.current_pos
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
