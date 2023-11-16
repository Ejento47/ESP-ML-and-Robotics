import cv2
import numpy as np
import matplotlib.pyplot as plt
from qlearningpark import QLearningAgent
import numpy as np

class ParkingEnvironment:
    def __init__(self,initial_pos,goal_pos,obstacles):
        self.margin = 5
        self.initial_pos = initial_pos
        self.backgroundQL = np.ones((100+2*self.margin,100+2*self.margin,1)) #creating background image of environment in white for QL 
        self.current_pos = initial_pos #initial position of car which should be current_pos in the main code
        self.goal_pos = goal_pos #goal position of car which should be a parking slot in main code
        self.obstacles = self.place_obstacles_ql(obstacles)
        self.grid_width = 100+2*self.margin

    def place_obstacles_ql(self, obs): #cteates a grid map specifically for the Q-learning algorithm
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*1 #adding margin to obstacles wihtout scaling
        for ob in obstacles:
            # Flipping the y coordinate
            flipped_y = (100+2*self.margin-1) - ob[1] #reverse from index 0 to 109 to 109 to 0
            # Set the obstacle in the backgroundQL
            self.backgroundQL[flipped_y, ob[0]] = 0 #setting of obstacles in black
        return set([tuple(obstacle) for obstacle in obstacles])
            
    def reset(self):
        # print("Current position:", self.current_pos)
        # Reset the environment to an initial state
        self.current_pos = self.initial_pos  # Set to initial position
        return self.current_pos[0],self.current_pos[1]   # Return the initial state in the environment as x,y coordinates

    def step(self, actions):
        reward = 0
        done = False

        # Define movement directions
        movements = {
            0: (0, -1),   # Up
            1: (0, 1),    # Down
            2: (-1, 0),   # Left
            3: (1, 0),    # Right
            4: (-1, -1),  # Upper-left
            5: (1, -1),   # Upper-right
            6: (-1, 1),   # Lower-left
            7: (1, 1)     # Lower-right
        }

            # Update current position based on the action
        move = movements[actions]
        new_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])

        # Check for boundaries, obstacles, and goal
        if  new_pos in self.obstacles:
            reward = -100  # Penalty for collision or out of bounds
            done = True
            return self.current_pos, reward, done
        if new_pos[0] < 0 or new_pos[0] >= self.grid_width or new_pos[1] < 0 or new_pos[1] >= self.grid_width:
            reward = -100
            done = True
            return self.current_pos, reward, done
        
        elif new_pos == self.goal_pos:
            reward = 100  # Reward for reaching the goal
            self.current_pos = new_pos
            done = True
            return self.current_pos, reward,done
        else:
            reward = -1  # Penalty for each step taken
            done = False
            self.current_pos = new_pos  # Update position if safe to move
            return self.current_pos, reward,done

    def get_state_index(self): # Convert the current position to a state index
        return round(self.current_pos[1] * self.grid_width + self.current_pos[0])

    def get_current_state(self):
        return self.current_pos[0],self.current_pos[1]

    def get_position_from_state_index(self, state_index):
        y = state_index // self.grid_width
        x = state_index % self.grid_width
        return x,y