import numpy as np
from scipy.optimize import minimize
import copy
import math
# from environment_v3 import Environment, Parking1


class Car_Dynamics:
    def __init__(self, x_0, y_0, v_0, psi_0, length, dt): 
        self.dt = dt             # sampling time
        self.L = length          # vehicle length
        self.x = x_0             # initial x position
        self.y = y_0             # initial y position
        self.v = v_0             # initial velocity
        self.psi = psi_0         # initial angle
        self.state = np.array([[self.x, self.y, self.v, self.psi]]).T # initial state vector (4x1)

    def move(self, accelerate, delta):
        x_dot = self.v*np.cos(self.psi) #Vel in the x direction
        y_dot = self.v*np.sin(self.psi) #Vel in the y direction
        v_dot = accelerate #acceleration 
        psi_dot = self.v*np.tan(delta)/self.L #change in angle; yaw rate
        return np.array([[x_dot, y_dot, v_dot, psi_dot]]).T #return the state vector as a 4x1 matrix

    def update_state(self, state_dot):
        # self.u_k = command 
        # self.z_k = state
        self.state = self.state + self.dt*state_dot #update the state vector with based on the changes with time
        self.x = self.state[0,0] #update the x position 
        self.y = self.state[1,0] #update the y position
        self.v = self.state[2,0] #update the velocity
        self.psi = self.state[3,0] #update the angle
        
    def process_sensor_data(self, sensor_data):
        """
        Process sensor data to update the environment and re-plan the path if necessary.
        sensor_data: list of tuples containing obstacle positions relative to the car's position
        """
        new_obstacles = []
        for data in sensor_data:
            #Data is a tuple (distance, angle) relative to the car
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

    def detect_obstacles_and_spaces(self, car_pos, car_orientation, environment, obstacles, empty_spaces, max_distance=100, fov_deg=120):
        """
        Detects obstacles within the field of view of the car and identifies empty spaces.

        :param car_pos: Tuple (x, y) representing the car's position.
        :param car_orientation: Car's orientation in degrees.
        :param environment: 2D list or array representing the environment, where obstacles are marked.
        :param max_distance: Maximum distance the sensor can detect.
        :param fov_deg: Field of view in degrees.
        :return: Tuple of two lists:
                - List of tuples with (distance, angle) for each detected obstacle.
                - List of tuples with (distance, angle) for each detected empty space.
        """
        car_x, car_y = car_pos

        # Convert car orientation to radians and calculate FOV boundaries
        car_orientation_rad = math.radians(car_orientation)
        fov_start = car_orientation_rad - math.radians(fov_deg / 2)
        fov_end = car_orientation_rad + math.radians(fov_deg / 2)
        
        # Iterate through each cell in the environment, where environment in this case will probably be called as Environment.background
        for y in range(car_y-20,car_y+100):
            for x in range(car_x-20,car_y+100):
                if x>=0 and y>=0:
                    obstacle_pos = (x, y)
                    rel_x, rel_y = obstacle_pos[0] - car_x, obstacle_pos[1] - car_y
                    distance = math.sqrt(rel_x**2 + rel_y**2)
                    angle = math.atan2(rel_y, rel_x)

                    # Check if the point is within the sensor's FOV and within the max distance
                    if distance <= max_distance and fov_start <= angle <= fov_end:
                        # Normalize the angle
                        rel_angle = math.degrees(angle - car_orientation_rad)
                        # Check if the point is an obstacle or empty space
                        if environment[y][x] == [0,0,0]:  # Obstacle detected
                            obstacles.append((distance, rel_angle))
                        elif environment[y][x] == [1,1,1]:  # Empty space detected
                            empty_spaces.append((distance, rel_angle))

            return obstacles, empty_spaces
    
    def find_key_by_value(self, dictionary, target_value):
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None
    
    def detect_empty_parking(self, car_positions, parking_slots, empty_spaces):
        #car_positions = Parking1.cars.values()  # Accessing cars attribute
        #parking_slots = Parking1.parking_slots  # Accessing parking_slots attribute
        
        for car in car_positions:
            if car in empty_spaces:
                slot_number = self.find_key_by_value(parking_slots, car)
                if slot_number is not None:
                    parking_slots[slot_number] = False
                

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
    
        desired_state = points.T #desired state is the path for a*star
        cost = 0.0 #initial cost is 0

        for i in range(self.horiz):
            state_dot = mpc_car.move(u_k[0,i], u_k[1,i]) #state_dot is the change in state
            mpc_car.update_state(state_dot) #update the state of the car based on the new change
        
            z_k[:,i] = [mpc_car.x, mpc_car.y] #update the state vector
            cost += np.sum(self.R@(u_k[:,i]**2)) #cost is the sum of the input cost matrix multiplied by the input
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2)) #cost is the sum of the state cost matrix multiplied by the difference between the desired state and the actual state
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2)) #cost is the sum of the input difference cost matrix multiplied by the difference between the current input and the next input
        return cost 

    def optimize(self, my_car, points):
        # Set the prediction horizon based on the number of desired states
        self.horiz = points.shape[0]
        
        # Define bounds for the control inputs
        # Example: steering angles between -60 and 60 degrees, accelerations between -5 and 5
        bnd = [(-5, 5), (-np.deg2rad(60), np.deg2rad(60))] * self.horiz
        
        # Define the initial guess for the optimization as a vector of zeros
        x0 = np.zeros((2 * self.horiz,))
        
        # Perform the optimization to minimize the cost function
        result = minimize(self.mpc_cost, x0, args=(my_car, points), method='SLSQP', bounds=bnd)
        
        # Return the first optimal input only
        return result.x[0], result.x[1]



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
                    [0, 0, self.dt*np.tan(delta)/self.L, 1                     ]]) #linearized model of state transition matrix
        # 4*2 
        B = np.array([[0      , 0                                  ],
                    [0      , 0                                  ],
                    [self.dt, 0                                  ],
                    [0      , self.dt*v/(self.L*np.cos(delta)**2)]]) #linearized model of control input matrix

        # 4*1
        C = np.array([[self.dt*v* np.sin(psi)*psi                ],
                    [-self.dt*v*np.cos(psi)*psi                ],
                    [0                                         ],
                    [-self.dt*v*delta/(self.L*np.cos(delta)**2)]]) #linearized model of offset matrix
        
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
        # Set the prediction horizon based on the number of desired states
        self.horiz = points.shape[0]
        
        # Define bounds for the control inputs
        # Example: steering angles between -60 and 60 degrees, accelerations between -5 and 5
        bnd = [(-5, 5), (-np.deg2rad(60), np.deg2rad(60))] * self.horiz
        
        # Define the initial guess for the optimization as a vector of zeros
        x0 = np.zeros((2 * self.horiz,))
        
        # Perform the optimization to minimize the cost function
        result = minimize(self.mpc_cost, x0, args=(my_car, points), method='SLSQP', bounds=bnd)
        
        # Return the first optimal input only
        return result.x[0], result.x[1]