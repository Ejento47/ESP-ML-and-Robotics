import numpy as np
from scipy.optimize import minimize
import copy
import math


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
        
        # Iterate through each cell in the environment
        for y in range(len(environment)):
            for x in range(len(environment[y])):
                if environment[y][x] == 1:  #  1 represents an obstacle
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