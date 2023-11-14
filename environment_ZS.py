import cv2
import numpy as np
import random

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
        self.background = np.ones((1000+20*self.margin,1000+20*self.margin,3))
        self.background[10:1000+20*self.margin:10,:] = np.array([200,200,200])/255
        self.background[:,10:1000+20*self.margin:10] = np.array([200,200,200])/255
        self.place_obstacles(obstacles)
                
    def place_obstacles(self, obs):
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*10
        for ob in obstacles:
            self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]=0
    
    def draw_path(self, path):
        path = np.array(path)*10
        color = np.random.randint(0,150,3)/255
        path = path.astype(int)
        for p in path:
            self.background[p[1]+10*self.margin:p[1]+10*self.margin+3,p[0]+10*self.margin:p[0]+10*self.margin+3]=color

    def rotate_car(self, pts, angle=0):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        return ((R @ pts.T).T).astype(int)
    
    def draw_parking_slots(self, parking_slots):
        slot_width = 44  # car width is 40, plus 2 units on each side for the parking slot
        slot_length = 84  # car length is 80, plus 2 units on each side for the parking slot
        half_slot_width = slot_width // 2
        half_slot_length = slot_length // 2
        slot_color = (0, 255, 0)  # green color for the parking slot
        thickness = 2 

        for center in parking_slots:
            center_x, center_y = center
            top_left = (int((center_x - half_slot_width) * 3), int((center_y - half_slot_length) * 3))
            bottom_right = (int((center_x + half_slot_width) * 3), int((center_y + half_slot_length) * 3))
            self.background = cv2.rectangle(self.background, top_left, bottom_right, color=slot_color, thickness=thickness)

    def render(self, x, y, psi, delta, parking_slots):
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
        # Draw parking slots
        self.draw_parking_slots(parking_slots)

        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))

        return rendered


class Parking1:
    def __init__(self, car_pos):
        self.car_obstacle = self.make_car()
        self.walls = self.define_walls()

        # Define parking slots along walls
        self.parking_slots = self.define_parking_slots()

        # Randomly place cars in parking slots
        self.cars = self.randomly_place_cars()

        self.end = self.cars[car_pos][0] if car_pos in self.cars else None
        if self.end:
            self.cars.pop(car_pos)

    def define_walls(self):
        # Define walls
        walls = [[70, i] for i in range(-5, 90)] + \
                [[30, i] for i in range(10, 105)] + \
                [[i, 10] for i in range(30, 36)] + \
                [[i, 90] for i in range(70, 76)]
        return walls

    def define_parking_slots(self):
        # Define parking slots along the walls

        parking_slots = [[35, 20], [65, 20], [75, 20], [95, 20], 
        [35, 32], [65, 32], [75, 32], [95, 32], 
        [35, 44], [65, 44], [75, 44], [95, 44], 
        [35, 56], [65, 56], [75, 56], [95, 56], 
        [35, 68], [65, 68], [75, 68], [95, 68], 
        [35, 80], [65, 80], [75, 80], [95, 80]]

        return parking_slots
    
    def get_parking_slots(self):
        # Return the list of parking slots
        return self.parking_slots  # If parking_slots is an attribute containing this info

    def randomly_place_cars(self):
        # Randomly place cars in the defined parking slots
        cars = {}
        available_slots = self.parking_slots.copy()  # Ensure this is a list

        # Make sure available_slots is a list before this point
        if not available_slots:
            raise ValueError("No available parking slots to place cars.")

        num_cars = random.randint(0, len(available_slots) - 1)
        for i in range(1, num_cars + 1):
            chosen_slot = random.choice(available_slots)
            cars[i] = [chosen_slot]
            available_slots.remove(chosen_slot)  # This should work if available_slots is a list
        return cars

    def generate_obstacles(self):
        # Generate the obstacle array
        obs = np.array(self.walls)
        for car in self.cars.values():
            for pos in car:
                obstacle = self.car_obstacle + pos + 1
                obs = np.concatenate((obs, obstacle), axis=0)
        
        # Calculate the end position
        end_position = self.calculate_end_position()

        # Ensure that only two values are returned
        return end_position, obs

    def calculate_end_position(self):
        # Implement the logic to determine the end position
        # Example: return the last parking slot or a specific target position
        return self.end  # or any other logic to determine the end position

    def make_car(self):
        car_obstacle_x, car_obstacle_y = np.meshgrid(np.arange(-2, 2), np.arange(-4, 4))
        car_obstacle = np.dstack([car_obstacle_x, car_obstacle_y]).reshape(-1, 2)
        return car_obstacle