import numpy as np
import math
import scipy.interpolate as scipy_interpolate
from utils import angle_of_line
# from control_v3 import Car_Dynamics, MPC_Controller, Linear_MPC_Controller

############################################## Functions ######################################################

def interpolate_b_spline_path(x, y, n_path_points, degree=3):
    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)
    travel = np.linspace(0.0, len(x) - 1, n_path_points)
    return spl_i_x(travel), spl_i_y(travel)

def interpolate_path(path, sample_rate):
    choices = np.arange(0,len(path),sample_rate)
    if len(path)-1 not in choices:
            choices =  np.append(choices , len(path)-1)
    way_point_x = path[choices,0]
    way_point_y = path[choices,1]
    n_course_point = len(path)*3
    rix, riy = interpolate_b_spline_path(way_point_x, way_point_y, n_course_point)
    new_path = np.vstack([rix,riy]).T
    # new_path[new_path<0] = 0
    return new_path

################################################ Path Planner ################################################

class AStarPlanner:

    def __init__(self, ox, oy, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        rr: robot radius[m] #radius of the car for collision checking
        
        """
        
        self.rr = rr
        self.min_x, self.min_y = 0, 0 #min x and y coordinates of the grid map
        self.max_x, self.max_y = 0, 0 #max x and y coordinates of the grid map
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0 #width of the grid map in x and y direction
        self.motion = self.get_motion_model() #8-connected grid map meaning that the car can move in 8 directions # 3 columns: x,y,cost
        self.calc_obstacle_map(ox, oy) #calculate the obstacle map based on the obstacles
        
        #attributes for re-planning
        self.start = None
        self.goal = None
        self.path = []

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost #cost of the node
            self.parent_index = parent_index 

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        self.start = self.Node(sx, sy, 0.0, -1) #start node, 0 cost , -1 parent index
        self.goal = self.Node(gx, gy, 0.0, -1) #goal node, 0 cost , -1 parent index
 
        open_set, closed_set = dict(), dict() #open set and closed set dictionaries
        open_set[self.calc_grid_index(self.start)] = self.start #add start node to open set dictionary

        while open_set is not None: #while open set is not empty

            # Initialize variables for finding the node with the lowest cost
            current_node_id = None #current node id is none cause we haven't found the lowest cost node yet
            lowest_cost = float('inf') #inf cause we want the minimum cost

            # Iterate through each node in the open set
            for node_id, node in open_set.items():
                # Calculate the total cost for the node (cost + heuristic)
                total_cost = node.cost + self.manhattan_heuristic(self.goal, node)

                # Check if this node has the lowest total cost and replace the lowest cost node if so
                if total_cost < lowest_cost: 
                    lowest_cost = total_cost
                    current_node_id = node_id

            # Retrieve the current node using the lowest cost node ID
            current_node = open_set[current_node_id] #current node is the node with the lowest cost

            # Check if the current node is the goal
            if current_node.x == self.goal.x and current_node.y == self.goal.y:
                self.goal.parent_index = current_node.parent_index
                self.goal.cost = current_node.cost
                break

            # Move the current node from the open set to the closed set
            del open_set[current_node_id]
            closed_set[current_node_id] = current_node

            # Explore the neighboring nodes
            for motion in self.motion: #remember that motion is a list of lists of 8 directions with x,y,cost
                #check if the neighbor node is safe is 8 directions
                neighbor_node = self.Node(current_node.x + motion[0],
                                        current_node.y + motion[1],
                                        current_node.cost + motion[2], current_node_id)
                neighbor_node_id = self.calc_grid_index(neighbor_node) #index of neighbor node in the grid map (done by x*y)

                # Skip if the neighbor is not safe or already in the closed set
                if not self.verify_node(neighbor_node) or neighbor_node_id in closed_set:
                    continue

                # If the neighbor is not in the open set or offers a lower cost, add/update it
                if neighbor_node_id not in open_set or open_set[neighbor_node_id].cost > neighbor_node.cost:
                    open_set[neighbor_node_id] = neighbor_node

        # Reconstruct the final path from the goal node to the start node
        path_x, path_y = self.calc_final_path(self.goal, closed_set)
        return path_x, path_y

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def manhattan_heuristic(self,n1, n2):
        w = 1.0  # weight of heuristic
        d = w * (abs(n1.x - n2.x) + abs(n1.y - n2.y))
        return d
    
    def euclidean_heuristic(self,n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos))

    def calc_grid_index(self, node): #index of node in the grid map
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)  # index = y*width + x, because we are using a 1D array

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        
        # check if node is outside the grid map
        if px < self.min_x: 
            return False
        elif py < self.min_y: 
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        #check if the node is in the obstacle map which is  collision
        if self.obstacle_map[node.x][node.y]: 
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        
        #calculate the min and max x and y coordinates of the grid map
        self.min_x = round(min(ox))  
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x)) #width of the grid map in x direction
        self.y_width = round((self.max_y - self.min_y)) #width of the grid map in y direction

        #this is the obstacle map which is a 2D array of the grid map
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        
        #fill in the obstacle map with True if there is an obstacle in the grid map
        for ix in range(self.x_width): #iterate through the x coordinates of the grid map
            x = self.calc_grid_position(ix, self.min_x) #should have no change since min_x is 0
            for iy in range(self.y_width): #iterate through the y coordinates of the grid map
                y = self.calc_grid_position(iy, self.min_y) #should have no change since min_y is 0
                for iox, ioy in zip(ox, oy): #iterate through the obstacles, zip is to iterate through 2 lists at the same time since ox and oy are the same length
                    d = math.hypot(iox - x, ioy - y) #distance between the obstacle and the grid map
                    if d < self.rr: # means car is placed in the obstacle to near the obstacle or is on the obstacle which will cause collision and impractical
                        self.obstacle_map[ix][iy] = True  
                        break

    def get_motion_model(self): # up,down,left,right,up-left,up-right,down-left,down-right
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]] #8-connected grid map meaning that the car can move in 8 directions # 3 columns: x,y,cost

        return motion

############################################### Park Path Planner #################################################

class ParkPathPlanning:
    def __init__(self,obstacles):
        self.margin = 5
        #scale obstacles from env margin to pathplanning margin
        obstacles = obstacles + np.array([self.margin,self.margin])
        obstacles = obstacles[(obstacles[:,0]>=0) & (obstacles[:,1]>=0)]

        self.obs = np.concatenate([np.array([[0,i] for i in range(100+self.margin)]),
                                  np.array([[100+2*self.margin,i] for i in range(100+2*self.margin)]),
                                  np.array([[i,0] for i in range(100+self.margin)]),
                                  np.array([[i,100+2*self.margin] for i in range(100+2*self.margin)]),
                                  obstacles])

        self.ox = [int(item) for item in self.obs[:,0]]
        self.oy = [int(item) for item in self.obs[:,1]]
        self.robot_radius = 4
        self.a_star = AStarPlanner(self.ox, self.oy,self.robot_radius)
        self.q_table = np.zeros((100, 100, 4))  # Initialize Q-table with zeros
    
    def plan_path(self,sx, sy, gx, gy): #A star planning to the goal
        rx, ry = self.a_star.planning(sx+self.margin, sy+self.margin, gx+self.margin, gy+self.margin)
        rx = np.array(rx)-self.margin+0.5
        ry = np.array(ry)-self.margin+0.5
        path = np.vstack([rx,ry]).T #transpose to [x,y] format
        return path[::-1] #return path in reverse order

    # def update_q_table(self, state, action, reward, next_state):
    #     alpha = 0.5  # learning rate
    #     gamma = 0.9  # discount factor
    #     self.q_table[state[0], state[1], action] = (1 - alpha) * self.q_table[state[0], state[1], action] + \
    #                                                 alpha * (reward + gamma * np.max(self.q_table[next_state[0], next_state[1], :]))

    # def get_action(self, state, epsilon):
    #     if np.random.uniform(0, 1) < epsilon:
    #         action = np.random.choice(4)  # Explore: select a random action
    #     else:
    #         action = np.argmax(self.q_table[state[0], state[1], :])  # Exploit: select the action with max value (Q) for the current state
    #     return action

    # def generate_park_scenario(self, sx, sy, gx, gy):
    #     # Initialize parameters
    #     epsilon = 0.3  # Exploration rate
    #     num_episodes = 10000  # Number of episodes to train over

    #     # Initialize state
    #     state = [sx, sy]

    #     for episode in range(num_episodes):
    #         # Reset state at start of each episode
    #         state = [sx, sy]

    #         while state != [gx, gy]:  # While goal state not reached
    #             # Choose action
    #             action = self.get_action(state, epsilon)

    #             # Take action and get reward, transit to next state
    #             next_state, reward = self.take_action(state, action)

    #             # Update Q-table
    #             self.update_q_table(state, action, reward, next_state)

    #             # Go to the next state
    #             state = next_state

    #     # After training, generate the optimal path
    #     state = [sx, sy]
    #     path = []

    #     while state != [gx, gy]:
    #         # Choose the best action
    #         action = np.argmax(self.q_table[state[0], state[1], :])

    #         # Take action and transit to next state
    #         next_state, _ = self.take_action(state, action)

    #         # Add state to path
    #         path.append(state)

    #         # Go to the next state
    #         state = next_state

    #     return path

    #### PARKING SCENARIO BY EXISITING CODEBASE ####
    def generate_park_scenario(self,sx, sy, gx, gy):    
        rx, ry = self.a_star.planning(sx+self.margin, sy+self.margin, gx+self.margin, gy+self.margin)
        rx = np.array(rx)-self.margin+0.5
        ry = np.array(ry)-self.margin+0.5
        path = np.vstack([rx,ry]).T
        path = path[::-1]
        computed_angle = angle_of_line(path[-10][0],path[-10][1],path[-1][0],path[-1][1])

        s = 4
        l = 8
        d = 2
        w = 4

        if -math.atan2(0,-1) < computed_angle <= math.atan2(-1,0):
            x_ensure2 = gx
            y_ensure2 = gy
            x_ensure1 = x_ensure2 + d + w
            y_ensure1 = y_ensure2 - l - s
            ensure_path1 = np.vstack([np.repeat(x_ensure1,3/0.25), np.arange(y_ensure1-3,y_ensure1,0.25)[::-1]]).T
            ensure_path2 = np.vstack([np.repeat(x_ensure2,3/0.25), np.arange(y_ensure2,y_ensure2+3,0.25)[::-1]]).T
            park_path = self.plan_park_down_right(x_ensure2, y_ensure2)

        elif math.atan2(-1,0) <= computed_angle <= math.atan2(0,1):
            x_ensure2 = gx
            y_ensure2 = gy
            x_ensure1 = x_ensure2 - d - w
            y_ensure1 = y_ensure2 - l - s 
            ensure_path1 = np.vstack([np.repeat(x_ensure1,3/0.25), np.arange(y_ensure1-3,y_ensure1,0.25)[::-1]]).T
            ensure_path2 = np.vstack([np.repeat(x_ensure2,3/0.25), np.arange(y_ensure2,y_ensure2+3,0.25)[::-1]]).T
            park_path = self.plan_park_down_left(x_ensure2, y_ensure2)

        elif math.atan2(0,1) < computed_angle <= math.atan2(1,0):
            x_ensure2 = gx
            y_ensure2 = gy
            x_ensure1 = x_ensure2 - d - w
            y_ensure1 = y_ensure2 + l + s
            ensure_path1 = np.vstack([np.repeat(x_ensure1,3/0.25), np.arange(y_ensure1,y_ensure1+3,0.25)]).T
            ensure_path2 = np.vstack([np.repeat(x_ensure2,3/0.25), np.arange(y_ensure2-3,y_ensure2,0.25)]).T
            park_path = self.plan_park_up_left(x_ensure2, y_ensure2)

        elif math.atan2(1,0) < computed_angle <= math.atan2(0,-1):
            x_ensure2 = gx
            y_ensure2 = gy
            x_ensure1 = x_ensure2 + d + w
            y_ensure1 = y_ensure2 + l + s
            ensure_path1 = np.vstack([np.repeat(x_ensure1,3/0.25), np.arange(y_ensure1,y_ensure1+3,0.25)]).T
            ensure_path2 = np.vstack([np.repeat(x_ensure2,3/0.25), np.arange(y_ensure2-3,y_ensure2,0.25)]).T
            park_path = self.plan_park_up_right(x_ensure2, y_ensure2)

        return np.array([x_ensure1, y_ensure1]), park_path, ensure_path1, ensure_path2


    def plan_park_up_right(self, x1, y1):       
            s = 4
            l = 8
            d = 2
            w = 4

            x0 = x1 + d + w
            y0 = y1 + l + s
            
            curve_x = np.array([])
            curve_y = np.array([])
            y = np.arange(y1,y0+1)
            circle_fun = (6.9**2 - (y-y0)**2)
            x = (np.sqrt(circle_fun[circle_fun>=0]) + x0-6.9)
            y = y[circle_fun>=0]
            choices = x>x0-6.9/2
            x=x[choices]
            y=y[choices]
            curve_x = np.append(curve_x, x[::-1])
            curve_y = np.append(curve_y, y[::-1])
            
            y = np.arange(y1,y0+1)
            circle_fun = (6.9**2 - (y-y1)**2)
            x = (np.sqrt(circle_fun[circle_fun>=0]) + x1+6.9)
            y = y[circle_fun>=0]
            x = (x - 2*(x-(x1+6.9)))
            choices = x<x1+6.9/2
            x=x[choices]
            y=y[choices]
            curve_x = np.append(curve_x, x[::-1])
            curve_y = np.append(curve_y, y[::-1])

            park_path = np.vstack([curve_x, curve_y]).T
            return park_path

    def plan_park_up_left(self, x1, y1):       
            s = 4
            l = 8
            d = 2
            w = 4

            x0 = x1 - d - w
            y0 = y1 + l + s
            
            curve_x = np.array([])
            curve_y = np.array([])
            y = np.arange(y1,y0+1)
            circle_fun = (6.9**2 - (y-y0)**2)
            x = (np.sqrt(circle_fun[circle_fun>=0]) + x0+6.9)
            y = y[circle_fun>=0]
            x = (x - 2*(x-(x0+6.9)))
            choices = x<x0+6.9/2
            x=x[choices]
            y=y[choices]
            curve_x = np.append(curve_x, x[::-1])
            curve_y = np.append(curve_y, y[::-1])
            
            y = np.arange(y1,y0+1)
            circle_fun = (6.9**2 - (y-y1)**2)
            x = (np.sqrt(circle_fun[circle_fun>=0]) + x1-6.9)
            y = y[circle_fun>=0]
            choices = x>x1-6.9/2
            x=x[choices]
            y=y[choices]
            curve_x = np.append(curve_x, x[::-1])
            curve_y = np.append(curve_y, y[::-1])

            park_path = np.vstack([curve_x, curve_y]).T
            return park_path


    def plan_park_down_right(self, x1,y1):
            s = 4
            l = 8
            d = 2
            w = 4

            x0 = x1 + d + w
            y0 = y1 - l - s
            
            curve_x = np.array([])
            curve_y = np.array([])
            y = np.arange(y0,y1+1)
            circle_fun = (6.9**2 - (y-y0)**2)
            x = (np.sqrt(circle_fun[circle_fun>=0]) + x0-6.9)
            y = y[circle_fun>=0]
            choices = x>x0-6.9/2
            x=x[choices]
            y=y[choices]
            
            curve_x = np.append(curve_x, x)
            curve_y = np.append(curve_y, y)
            
            y = np.arange(y0,y1+1)
            circle_fun = (6.9**2 - (y-y1)**2)
            x = (np.sqrt(circle_fun[circle_fun>=0]) + x1+6.9)
            x = (x - 2*(x-(x1+6.9)))
            y = y[circle_fun>=0]
            choices = x<x1+6.9/2
            x=x[choices]
            y=y[choices]
            curve_x = np.append(curve_x, x)
            curve_y = np.append(curve_y, y)
            
            park_path = np.vstack([curve_x, curve_y]).T
            return park_path


    def plan_park_down_left(self, x1,y1):
            s = 4
            l = 8
            d = 2
            w = 4

            x0 = x1 - d - w
            y0 = y1 - l - s
            
            curve_x = np.array([])
            curve_y = np.array([])
            y = np.arange(y0,y1+1)
            circle_fun = (6.9**2 - (y-y0)**2)
            x = (np.sqrt(circle_fun[circle_fun>=0]) + x0+6.9)
            y = y[circle_fun>=0]
            x = (x - 2*(x-(x0+6.9)))
            choices = x<x0+6.9/2
            x=x[choices]
            y=y[choices]
            curve_x = np.append(curve_x, x)
            curve_y = np.append(curve_y, y)
            
            y = np.arange(y0,y1+1)
            circle_fun = (6.9**2 - (y-y1)**2)
            x = (np.sqrt(circle_fun[circle_fun>=0]) + x1-6.9)
            y = y[circle_fun>=0]
            choices = x>x1-6.9/2
            x=x[choices]
            y=y[choices]
            curve_x = np.append(curve_x, x)
            curve_y = np.append(curve_y, y)
            
            park_path = np.vstack([curve_x, curve_y]).T
            return park_path