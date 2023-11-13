import numpy as np
from scipy.optimize import minimize
import copy
import math


class Car_Dynamics:
    def __init__(self, x_0, y_0, v_0, psi_0, length, dt):
        self.dt = dt             # sampling time
        self.L = length          # vehicle length
        self.x = x_0
        self.y = y_0
        self.v = v_0
        self.psi = psi_0
        self.state = np.array([[self.x, self.y, self.v, self.psi]]).T

    def move(self, accelerate, delta):
        x_dot = self.v*np.cos(self.psi)
        y_dot = self.v*np.sin(self.psi)
        v_dot = accelerate
        psi_dot = self.v*np.tan(delta)/self.L
        return np.array([[x_dot, y_dot, v_dot, psi_dot]]).T

    def update_state(self, state_dot):
        # self.u_k = command
        # self.z_k = state
        self.state = self.state + self.dt*state_dot
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.v = self.state[2,0]
        self.psi = self.state[3,0]
        
    def process_sensor_data(self, sensor_data):
        """
        Process sensor data to update the environment and re-plan the path if necessary.
        sensor_data: list of tuples containing obstacle positions relative to the car's position
        """
        new_obstacles = []
        for data in sensor_data:
            # Assuming data is a tuple (distance, angle) relative to the car
            distance, angle = data

            # Filter based on the 120-degree view and distance
            if -60 <= angle <= 60 and distance <= 20:
                # Convert relative obstacle position to global coordinates
                global_x, global_y = self.convert_to_global_coordinates(distance, angle)
                new_obstacles.append((global_x, global_y))

        # Update the environment with the new obstacles
        self.update_environment(new_obstacles)

    def convert_to_global_coordinates(self, distance, angle):
        """
        Convert relative position to global coordinates.
        Assumes angle is in degrees.
        """
        # Convert angle to radians for computation
        angle_rad = math.radians(angle)

        # Assuming access to car's current position (self.x, self.y) and orientation (self.psi)
        global_x = self.x + distance * math.cos(self.psi + angle_rad)
        global_y = self.y + distance * math.sin(self.psi + angle_rad)

        return global_x, global_y

    def detect_obstacles(car_pos, car_orientation, environment, max_distance=20, fov_deg=120):
        """
        Detects obstacles within the field of view of the car.

        :param car_pos: Tuple (x, y) representing the car's position.
        :param car_orientation: Car's orientation in degrees (0 degrees is east, 90 is north).
        :param environment: 2D list or array representing the environment, where obstacles are marked.
        :param max_distance: Maximum distance the sensor can detect.
        :param fov_deg: Field of view in degrees.
        :return: List of tuples with (distance, angle) for each detected obstacle.
        """
        sensor_data = []
        car_x, car_y = car_pos

        # Convert car orientation to radians and calculate FOV boundaries
        car_orientation_rad = math.radians(car_orientation)
        fov_start = car_orientation_rad - math.radians(fov_deg / 2)
        fov_end = car_orientation_rad + math.radians(fov_deg / 2)

        for y in range(len(environment)):
            for x in range(len(environment[y])):
                if environment[y][x] == 1:  # Assuming 1 represents an obstacle
                    obstacle_pos = (x, y)
                    rel_x, rel_y = obstacle_pos[0] - car_x, obstacle_pos[1] - car_y
                    distance = math.sqrt(rel_x**2 + rel_y**2)

                    if distance <= max_distance:
                        angle = math.atan2(rel_y, rel_x)
                        # Normalize the angle
                        if fov_start <= angle <= fov_end:
                            # Convert angle to relative to car orientation
                            rel_angle = math.degrees(angle - car_orientation_rad)
                            sensor_data.append((distance, rel_angle))
        
        #sensor data from here will be passed into process_sensor_data() to update the map
        return sensor_data

#To optimise steering based off wheere you want to go    
class MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix

    def mpc_cost(self, u_k, my_car, points):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
    
        desired_state = points.T
        cost = 0.0

        for i in range(self.horiz):
            state_dot = mpc_car.move(u_k[0,i], u_k[1,i])
            mpc_car.update_state(state_dot)
        
            z_k[:,i] = [mpc_car.x, mpc_car.y]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]



######################################################################################################################################################################

class Linear_MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix
        self.dt=0.2   
        self.L=4                          

    def make_model(self, v, psi, delta):        
        # matrices
        # 4*4
        A = np.array([[1, 0, self.dt*np.cos(psi)         , -self.dt*v*np.sin(psi)],
                    [0, 1, self.dt*np.sin(psi)         , self.dt*v*np.cos(psi) ],
                    [0, 0, 1                           , 0                     ],
                    [0, 0, self.dt*np.tan(delta)/self.L, 1                     ]])
        # 4*2 
        B = np.array([[0      , 0                                  ],
                    [0      , 0                                  ],
                    [self.dt, 0                                  ],
                    [0      , self.dt*v/(self.L*np.cos(delta)**2)]])

        # 4*1
        C = np.array([[self.dt*v* np.sin(psi)*psi                ],
                    [-self.dt*v*np.cos(psi)*psi                ],
                    [0                                         ],
                    [-self.dt*v*delta/(self.L*np.cos(delta)**2)]])
        
        return A, B, C

    def mpc_cost(self, u_k, my_car, points):
        
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
        desired_state = points.T
        cost = 0.0
        old_state = np.array([my_car.x, my_car.y, my_car.v, my_car.psi]).reshape(4,1)

        for i in range(self.horiz):
            delta = u_k[1,i]
            A,B,C = self.make_model(my_car.v, my_car.psi, delta)
            new_state = A@old_state + B@u_k + C
        
            z_k[:,i] = [new_state[0,0], new_state[1,0]]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
            
            old_state = new_state
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]