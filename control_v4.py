import numpy as np
from scipy.optimize import minimize
import copy
import math
import bisect
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
        
    def limit_sensor(self, x, y):
        """
        Limit Sensor Range
        """
        if x < 5:
            x = 5
        elif x > 105:
            x = 105
        
        if y < 5:
            y = 5
        elif y> 105:
            y = 105

        return x,y
    
    '''
    def binary_search(self, lst, item):
        # Assuming lst is a sorted list of lists (or tuples) and item is also a list (or tuple)
        left, right = 0, len(lst) - 1
        while left <= right:
            mid = (left + right) // 2
            if lst[mid] < item:
                left = mid + 1
            elif lst[mid] > item:
                right = mid - 1
            else:
                return True  # Item found
        return False  # Item not found
    '''

    def process_sensor_data(self, car_x, car_y, environment, obstacles, empty_spaces, distance=20):
        """
        Process sensor data to update the environment and re-plan the path if necessary.
        sensor_data: list of tuples containing obstacle positions relative to the car's position
        """
        for y in range(int(car_y-(distance/2)), int(car_y+(distance/2 + 1))):
            for x in range (int(car_x-(distance/2)), int(car_x+(distance/2 + 1))):
                x , y = self.limit_sensor(x,y)
                if environment[y][x] == False and [y,x] in empty_spaces:
                    empty_spaces.append([y,x])
                elif environment[y][x] == True and [y,x] not in obstacles:
                    obstacles.append([y,x])
        return obstacles
    
    def find_key_by_value(self, dictionary, target_value):
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None
    
    def detect_empty_parking(self, car_positions, empty_spaces):
        #check if the parking slot is empty
        for car in car_positions:
            return car in empty_spaces


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