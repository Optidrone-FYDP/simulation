from time import time
import math
import sys

class dronePID:
    def __init__(self, Kp=0.5, Ki=0.01, Kd=0.1, max_a = 150, mode = "sim"):
        self.last_five_vel = [[0]*3,[0]*3,[0]*3,[0]*3,[0]*3]
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
        self.state = ["hover", "hover", "hover"] # or "follow" or "wait"
        self.imm_target = [0,0,0]
        self.mode = mode # sim or vicon
    
    def setpoint(self, x_pos, y_pos, z_pos, rot):
        self.target_pos = [x_pos, y_pos, z_pos, rot]
        self.profiled_path = self.profile(self.target_pos)
        #print("profiled path:")
        #print(self.profiled_path)
        for i in range(len(self.state)):
            self.state[i] = "start"
            self.next_setpoint(i)

    def next_setpoint(self, axis):
        if self.state[axis] == "start":
            self.setpoint_step = 0
            self.imm_target[0] = self.profiled_path[self.setpoint_step][0][0]
            self.target_acc[0] = self.profiled_path[self.setpoint_step][0][1]
            self.imm_target[1] = self.profiled_path[self.setpoint_step][1][0]
            self.target_acc[1] = self.profiled_path[self.setpoint_step][1][1]
            self.imm_target[2] = self.profiled_path[self.setpoint_step][2]
            self.state[axis] = "follow"
        elif self.state[axis] == "follow":
            self.state[axis] = "wait"
            if "follow" in self.state:
                return
            self.setpoint_step += 1
            if self.setpoint_step >= len(self.profiled_path):
                for i in range(len(self.state)):
                    self.state[i] = "hover"
            else:
                for i in range(len(self.state)):
                    self.state[i] = "follow"
                self.imm_target[0] = self.profiled_path[self.setpoint_step][0][0]
                self.target_acc[0] = self.profiled_path[self.setpoint_step][0][1]
                self.imm_target[1] = self.profiled_path[self.setpoint_step][1][0]
                self.target_acc[1] = self.profiled_path[self.setpoint_step][1][1]
                self.imm_target[2] = self.profiled_path[self.setpoint_step][2]
    
    def profile(self, target_pos):  # only x-y plane, z is semi-profiled only a target pos is provided, rot have special case so it's not profiled
        # math to calculate profiled accelerations
        out = []
        total_dist = math.sqrt( (target_pos[0]-self.current_pos[0])**2 + (target_pos[1]-self.current_pos[1])**2 )
        self.travel_time = 2 + 0.01*total_dist/3
        self.accel_time = self.travel_time/2
        self.accel = [ (target_pos[0]-self.current_pos[0])/(self.accel_time**2), (target_pos[1]-self.current_pos[1])/(self.accel_time**2) ]
        s1 = [ [ (target_pos[0]-self.current_pos[0])/2 + self.current_pos[0], self.accel[0] ] , [ (target_pos[1]-self.current_pos[1])/2 + self.current_pos[1], self.accel[1] ], (target_pos[2]-self.current_pos[2])/2 + self.current_pos[2] ]    # format is : [position, acceleration]
        s2 = [ [ target_pos[0], -self.accel[0] ], [ target_pos[1], -self.accel[1] ], target_pos[2] ]
        out.append(s1)
        out.append(s2)
        return out  # should return 3d array of path steps with corresponding positions and accelerations we should be hitting
    
    def update(self):
        dt = self.get_dt()

        for i in range(4):
            self.last_five_vel[i] = self.last_five_vel[i+1]

        abs_x_vel = (self.current_pos[0]-self.prev_pos[0])/dt
        abs_y_vel = (self.current_pos[1]-self.prev_pos[1])/dt

        # need to refine math that converts absolute vectors into the reference (drone) frame
        #super_vel_vec = math.sqrt( abs_x_vel**2 + abs_y_vel**2 )
        #if abs_y_vel == 0:
        #    if abs_x_vel > 0:
        #        vel_angle = math.pi/2
        #    elif abs_x_vel < 0:
        #        vel_angle = -math.pi/2
        #    else:
        #        vel_angle = 0
        #else:
        #    vel_angle = math.atan(abs_x_vel/abs_y_vel)
        #relative_angle = self.current_pos[3] - vel_angle
        #print(f"current pos: {self.current_pos}")
        #print(f"target pos: {self.target_pos}")
        #print(f"abs_x_vel: {abs_x_vel}")
        #print(f"abs_y_vel: {abs_y_vel}")
        #print(f"super_vel_vec: {super_vel_vec}")
        #print(f"vel_angle: {vel_angle}")
        #print(f"relative_angle: {relative_angle}")

        self.last_five_vel[4] = [ abs_x_vel , abs_y_vel , (self.current_pos[2]-self.prev_pos[2])/dt ]    # don't need to track rot velocity
        
        mean_v = [0]*3
        for i in range(3):
            for j in range(5):
                mean_v[i] += self.last_five_vel[j][i]/5
        
        accel = [0]*3
        
        for i in range(len(self.last_five_vel)):
            accel[0] += ( ( i-2 ) * dt ) * ( self.last_five_vel[i][0] - mean_v[0] ) / (10 * dt**2)
            accel[1] += ( ( i-2 ) * dt ) * ( self.last_five_vel[i][1] - mean_v[1] ) / (10 * dt**2)
            accel[2] += ( ( i-2 ) * dt ) * ( self.last_five_vel[i][2] - mean_v[2] ) / (10 * dt**2)

        self.current_acc = accel

        #TODO: add step that does least squares estimate of acceleration based on last 5 velocities

        out = [0, 0, 0, 0]
        for i in range(2): # x and y pid control
            pos_error = self.target_pos[i] - self.current_pos[i]
            #print("x-y pid")
            #print(pos_error)
            #print(self.target_pos[i])
            #print(self.current_pos[i])
            #print(self.state[i])
            if abs(pos_error) <= 30 and self.state[i] == "follow":
                self.next_setpoint(i)
            self.error[i] =  self.target_acc[i] - self.current_acc[i]
            self.p[i] = self.Kp[i] * self.error[i]
            self.i[i] += self.Ki[i] * self.error[i]
            if self.i[i] > self.max_i:
                self.i[i] = self.max_i
            elif self.i[i] < -self.max_i:
                self.i[i] = -self.max_i
            self.d[i] = self.Kd[i] * (self.error[i] - self.prev_error[i])/dt
            self.restoring[i] = self.Kr * pos_error
            self.prev_error = self.error
            out[i] = (self.p[i] + self.i[i] + self.d[i] if self.state[i]=="follow" else 0) + (self.restoring[i] if abs(pos_error) < 100 else 0)
            if out[i] > 150:
                out[i] = 150
            elif out[i] < -150:
                out[i] = -150
            if out[i] > 0:
                out[i] = -out[i]*63/150
            else:
                out[i] = -out[i]*-64/-150
            out[i] = round(out[i] + 64)

        # special z-axis control (z-axis has higher accel and stopping input basically maintains position so we can do this one off position alone)
        self.error[2] = self.imm_target[2] - self.current_pos[2]
        if abs(self.error[2]) < 30 and self.state[i] == "follow":
            next_setpoint(i)
        self.p[2] = self.Kp[2] * self.error[2]
        self.i[2] += self.Ki[2] * self.error[2]
        if self.i[2] > self.max_i:
            self.i[2] = self.max_i
        elif self.i[2] < -self.max_i:
            self.i[2] = -self.max_i
        self.d[2] = self.Kd[2] * (self.error[2] - self.prev_error[2])/dt
        self.prev_error = self.error
        out[2] = self.p[2] + self.i[2] + self.d[2]
        if out[2] > 63:
            out[2] = 63
        elif out[2] < -64:
            out[2] = -64
        out[2] = round(out[2] + 64)

        # special rot control (rot moves too slow and moves at a very constant rate so just position pid is enough), we should also only do rot at the end of each path as changing the rot will force us to recalculate our path
        if not ("follow" in self.state):
            self.error[3] = self.target_pos[3] - self.current_pos[3]
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
        else:
            out[3] = 64
        
        self.prev_pos = self.current_pos
        
        return out  # output format: [ x, y, z, rot ]
    
    def get_pos(self, new_pos):  # in this func we need to preprocess data with rot reported by model/vicon, this will let us calculate velocity and accel correctly according to the drone's orientation, converting from absolute coordinates to coordinates relative to drone's axis
        # exception is the drone's rotation which must be collected and expressed in relation to the absolute coordinates (calibration coordinates)
        #print(f"current_position fed to drone: {new_pos}")
        if self.mode=="sim":
            self.current_pos = [new_pos[0], new_pos[1], new_pos[2]]
            #print(f"current_position taken by drone: {self.current_pos}")
        elif self.mode=="vicon":
            return #get position from vicon cameras
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
