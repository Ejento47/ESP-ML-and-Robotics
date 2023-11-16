import numpy as np
from scipy.optimize import minimize
import copy
import math
import bisect
#from environment_v4 import Environment, Parking1


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
        
    def limit_sensor(self, x, y):#if matrix was what I thought it was
        """
        Limit Sensor Range
        """
        if x < 0:
            x = 0
        elif x > 100:
            x = 100
        else:
            x=x
        
        if y < 0:
            y = 0
        elif y> 100:
            y = 100
        else:
            y=y

        return x,y
    

    def process_sensor_data(self, car_x, car_y, og_obstacles, distance):
        """
        Process sensor data to update the environment and re-plan the path if necessary.
        sensor_data: list of tuples containing obstacle positions relative to the car's position
        """
        more_empty_spaces = []
        more_obstacles = []

        for y in range(int(car_y-(distance/2)), int(car_y+(distance/2 + 1))):
            for x in range (int(car_x-(distance/2)), int(car_x+(distance/2 + 1))):
                new_x , new_y = self.limit_sensor(x,y)
                #if using path_planner.a_star.obstacle_map
                
                #new_x+=5
                #new_y+=5

                if og_obstacles[new_y][new_x] == False:
                    more_empty_spaces.append([new_y,new_x])

                elif og_obstacles[new_y-5][new_x-5] == True:
                    more_obstacles.append([new_y-5,new_x-5])       

                '''
                #new_x-=5
                #new_y-=5
                #using env_obs
                if [new_y,new_x] not in og_obstacles:
                    more_empty_spaces.append([new_y,new_x])
                    print('mek')
                elif [new_y,new_x] in og_obstacles:
                    more_obstacles.append([new_y,new_x])
                    #print('kfc')
                '''      
        return more_empty_spaces, more_obstacles

    def sensor_update(self, car_x, car_y, og_obstacles, obstacles, empty_spaces, distance):
        extra_empty_spaces, extra_obstacles = self.process_sensor_data(car_x, car_y, og_obstacles, distance)
        for empty_space in extra_empty_spaces:
            if empty_space not in empty_spaces:
                empty_spaces = np.append(empty_spaces, [empty_space], axis=0)
                #empty_spaces.append(empty_space)
        for obstacle in extra_obstacles:
            if not any((obstacle == o).all() for o in obstacles):
                #np.append(obstacles,obstacle)
                obstacles = np.append(obstacles, [obstacle], axis=0)

    

    def find_key_by_value(self, dictionary, target_value):
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None
    
    def detect_empty_parking(self, car_positions, empty_spaces):
        #check if the parking slot is empty
        for cars in car_positions:
            for car in cars:
                print(car in empty_spaces)
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