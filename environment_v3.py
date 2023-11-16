import cv2
import numpy as np
import matplotlib.pyplot as plt
# from control_v3 import Car_Dynamics, MPC_Controller, Linear_MPC_Controller

class Environment:
    def __init__(self,obstacles):
        self.margin = 5 
        #coordinates are in [x,y] format
        self.car_length = 80
        self.car_width = 40
        self.wheel_length = 15
        self.wheel_width = 7
        self.wheel_positions = np.array([[25,15],[25,-15],[-25,15],[-25,-15]])
        
        self.color = np.array([0,0,255])/255
        self.wheel_color = np.array([20,20,20])/255

        self.car_struct = np.array([[+self.car_length/2, +self.car_width/2],
                                    [+self.car_length/2, -self.car_width/2],  
                                    [-self.car_length/2, -self.car_width/2],
                                    [-self.car_length/2, +self.car_width/2]], 
                                    np.int32)
        
        self.wheel_struct = np.array([[+self.wheel_length/2, +self.wheel_width/2],
                                      [+self.wheel_length/2, -self.wheel_width/2],  
                                      [-self.wheel_length/2, -self.wheel_width/2],
                                      [-self.wheel_length/2, +self.wheel_width/2]], 
                                      np.int32)

        #height and width
        
        self.background = np.ones((1000+20*self.margin,1000+20*self.margin,3)) #creating background image of environment in white
        # self.background[10:1000+20*self.margin:10,:] = np.array([200,200,200])/255 #setting of gridlines by setting every 10th pixel to grey in x direction
        # self.background[:,10:1000+20*self.margin:10] = np.array([200,200,200])/255 #setting of gridlines by setting every 10th pixel to grey in y direction
        self.place_obstacles(obstacles)
        ##plot background in matlab for testing
        # plt.imshow(self.background)
        # plt.show()

                
    def place_obstacles(self, obs): 
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*10 #adding margin to obstacles and converting to size of 1000 coordinates
        for ob in obstacles:
            self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]=0 #setting of obstacles in black

            
    def draw_path(self, path): #taken from visualisation of exisiting codebase
        path = np.array(path)*10 #convert path coordinates to 1000 coordinates
        color = np.random.randint(0,150,3)/255
        path = path.astype(int)
        for p in path:
            self.background[p[1]+10*self.margin:p[1]+10*self.margin+3,p[0]+10*self.margin:p[0]+10*self.margin+3]=color

    def rotate_car(self, pts, angle=0): #taken from visualisation of exisiting codebase
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        return ((R @ pts.T).T).astype(int)

    def render(self, x, y, psi, delta): #taken from visualisation of exisiting codebase
        # x,y in 100 coordinates
        x = int(10*x)
        y = int(10*y)
        # x,y in 1000 coordinates
        # adding car body
        rotated_struct = self.rotate_car(self.car_struct, angle=psi)
        rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)
        for i,wheel in enumerate(rotated_wheel_center):
            
            if i <2:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
            else:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
            rotated_wheel += np.array([x,y]) + wheel + np.array([10*self.margin,10*self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        # gel
        gel = np.vstack([np.random.randint(-50,-30,16),np.hstack([np.random.randint(-20,-10,8),np.random.randint(10,20,8)])]).T
        gel = self.rotate_car(gel, angle=psi)
        gel += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        gel = np.vstack([gel,gel+[1,0],gel+[0,1],gel+[1,1]])
        rendered[gel[:,1],gel[:,0]] = np.array([60,60,135])/255

        new_center = np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        self.background = cv2.circle(self.background, (new_center[0],new_center[1]), 2, [255/255, 150/255, 100/255], -1)

        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))
        return rendered


class Parking1:
    def __init__(self, car_pos,end):
        self.car_obstacle = self.make_car()
        self.walls = [[70,i] for i in range(-5,90) ]+\
                     [[30,i] for i in range(10,105)]+\
                     [[i,10] for i in range(30,36) ]+\
                     [[i,90] for i in range(70,76) ] #+ [[i,20] for i in range(-5,50)]
        # self.car_obs = np.array(self.walls)  #create an obstacle
        self.env_obs = np.array(self.walls) #obstacles for env to draw
        self.cars = {1 : [[35,20]], 2 : [[65,20]], 3 : [[75,20]], 4 : [[95,20]],
                     5 : [[35,32]], 6 : [[65,32]], 7 : [[75,32]], 8 : [[95,32]],
                     9 : [[35,44]], 10: [[65,44]], 11: [[75,44]], 12: [[95,44]],
                     13: [[35,56]], 14: [[65,56]], 15: [[75,56]], 16: [[95,56]],
                     17: [[35,68]], 18: [[65,68]], 19: [[75,68]], 20: [[95,68]],
                     21: [[35,80]], 22: [[65,80]], 23: [[75,80]], 24: [[95,80]]} #parking slots location
        
        self.parking_slots ={1 : True, 2 : True, 3 : True, 4 : True,
                            5 : True, 6 : True, 7 : True, 8 : True,
                            9 : True, 10: True, 11: True, 12: True,
                            13: True, 14: True, 15: True, 16: True,
                            17: True, 18: True, 19: True, 20: True,
                            21: True, 22: True, 23: True, 24: True} #parking slots occupancy status (True = occupied , False = empty)
        if car_pos in self.cars.keys(): #if car_pos is a valid parking slot
            self.end = self.cars[car_pos][0]
            self.parking_slots[car_pos] = False
            self.cars.pop(car_pos)
        else:
            self.end =  end

    # def discover_obstacles(self,car_pos,car_orientation,environment):
    #     """
    #     Updates the car_obstacle array with new sensor data.
        
    #     :param sensor_data_function: A function from the control system that returns obstacle coordinates.
    #     :return: Updated array of obstacles.
    #     """
    #     # Call the sensor data function to get new obstacles
        
    #     new_obstacles = Car_Dy.detect_obstacles(car_pos, car_orientation, environment, max_distance=20, fov_deg=120)
    #     for obstacle in new_obstacles:
    #         if obstacle not in self.car_obs:
    #             self.car_obs.append(obstacle)
    #     print(self.car_obs)
    #     return self.car_obs        

    def generate_obstacles(self):
        for i in self.cars.keys():
            for j in range(len(self.cars[i])):
                obstacle = self.car_obstacle + self.cars[i] #adding car obstacles to the environment
                self.env_obs = np.append(self.env_obs, obstacle) #adding car obstacles to the environment
        self.env_obs = np.array(self.env_obs).reshape(-1,2) #convert 1 column array to 2 column array
        self.car_obs = self.env_obs #FSFBFEFIBFEIB
        return self.end, self.car_obs, self.env_obs

    def make_car(self): #to make car obstacle
        car_obstacle_x, car_obstacle_y = np.meshgrid(np.arange(-2,2), np.arange(-4,4)) #create a meshgrid of car obstacle
        car_obstacle = np.dstack([car_obstacle_x, car_obstacle_y]).reshape(-1,2) # add x and y coordinates together and store in 2 columns
        return car_obstacle
    
