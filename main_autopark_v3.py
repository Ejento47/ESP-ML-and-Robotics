import cv2
import numpy as np
from time import sleep
import argparse
import random

from environment_v3 import Environment, Parking1
from pathplanning_V3 import  ParkPathPlanning, interpolate_b_spline_path, interpolate_path
from control_v3 import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger
from qlenvironment import ParkingEnvironment
from qlearningpark import QLearningAgent
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0)
    parser.add_argument('--y_start', type=int, default=90)
    parser.add_argument('--psi_start', type=int, default=0)
    parser.add_argument('--parking', type=int, default=1) #to remvoe since parking slot randomized

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables for start and end ##############################
    start = np.array([args.x_start, args.y_start])
    original_end   = np.array([85, 10]) #to indicate of carpark end
    #############################################################################################

    # environment margin  : 5
    # pathplanning margin : 5

    ########################## defining obstacles ###############################################
    # park_slot = args.parking
    # park_slot = np.random.randint(-10,24)
    parking1 = Parking1(1,original_end) # random parking slot selection ## of can be args.parking
    end,car_obs,env_obs = parking1.generate_obstacles() #car_obs is what car can see and env_obs is what see


 
    # Adding of obstables to the environment
    square1 = make_square(10,65,20)
    square2 = make_square(15,30,20)
    square3 = make_square(50,50,10)
    env_obs = np.vstack([env_obs,square1,square2,square3])
    car_obs = np.vstack([car_obs,square1,square2,square3]) ####TO CHANGE BACK TO CAR_OBS
    new_obs = np.array([[78,78],[79,79],[78,79]])
    env_obs = np.vstack([env_obs,new_obs])
    car_obs = np.vstack([car_obs,new_obs])
    #############################################################################################

    ########################### initialization for map and car and booleans for Astar and QL ##################################################
    env = Environment(env_obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    MPC_HORIZON = 5
    controller = MPC_Controller()
    # controller = Linear_MPC_Controller()

    #zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    
    #Astar
    can_park = False #boolean to check if car can park
    parkIsAstar = False #boolean to check if parking is done using A star
    current_pos = [] #to store the current position of the car
    counter = 0 #counter to check if car can park
    
    #QL
    trainQL = False #boolean to check if training of QL is true
    envQL = ParkingEnvironment(current_pos, end, env_obs)
    state_space_size = envQL.grid_width * envQL.grid_width  # 110 x 110 = 12100
    action_space_size = 8  # 8 directions of movement
    learning_rate = 0.1
    discount_factor = 0.99
    exploration_rate = 1.0
    agent = QLearningAgent(state_space_size, action_space_size, learning_rate, discount_factor, exploration_rate)
    ####################################### Booleans and setting ######################################################

    ############################# path planning to Original Goal ###############################
    path_planner = ParkPathPlanning(car_obs) #path planner class to take in the obstacles visible to the car only
    print('routing to end of carpark ...') 
    path = path_planner.plan_path(int(start[0]),int(start[1]),int(original_end[0]),int(original_end[1])) #path planning
    env.draw_path(path)
    ################################## Travel to original goal or empty parking slot ##################################################

    print('driving to destination ...')  
    while len(path) > 0: #while path is not empty

        acc, delta = controller.optimize(my_car, path[:MPC_HORIZON]) #get acc and delta from controller of the first 5 points in path
        my_car.update_state(my_car.move(acc,  delta)) #update car state
        res = env.render(my_car.x, my_car.y, my_car.psi, delta) #render environment of the car 
        logger.log(path[0], my_car, acc, delta) #log the data of current car state
        cv2.imshow('environment', res) #show the environment of car
        key = cv2.waitKey(1) 
        
        #if path has reached end of orinal end
        if path[0][0] -0.5 == original_end[0] and path[0][1] -0.5 == original_end[1]:
            print('~ No parking slots found, driver to take over ~')
            break
        
        #Rerouting to empty parking slot using QL planning as parking
        if counter > 120:
            print('rerouting to empty parking slot ...')
            if parkIsAstar: #using A* path planning to park car
                current_pos = path[0] #get current position of car
                # end = end #replace new end with empty parking slot
                break
            else: #using Q learning to park car
                current_pos = path[0]
                # end = end #replace new end with empty parking slot
                break
        if key == ord('s'):  #to save the imagee
            cv2.imwrite('res.png', res*255)
            
        path = path[1:] #remove the first point in path
        counter += 1

    ################################## Parking of car using Astar ##################################################
    if parkIsAstar: #if  park to be done using A star is true
        new_end, park_path, ensure_path1, ensure_path2 = path_planner.replan_park(int(current_pos[0]),int(current_pos[1]),int(end[0]),int(end[1]))
        path = path_planner.plan_path(int(current_pos[0]),int(current_pos[1]),int(new_end[0]),int(new_end[1]))
        path = np.vstack([path, ensure_path1])
        env.draw_path(path)
        env.draw_path(park_path)
        final_path = np.vstack([path, park_path, ensure_path2])
        for i,point in enumerate(final_path):
            acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
            my_car.update_state(my_car.move(acc,  delta))
            res = env.render(my_car.x, my_car.y, my_car.psi, delta)
            logger.log(point, my_car, acc, delta)
            cv2.imshow('environment', res)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite('res.png', res*255)
        print('~ Parking completed ~')

    ################################## Training QL or testing QL to park car ##################################################
    
    else: #if  park to be done using Q learning is true
        if trainQL:
            print('training park scenario using QL ...')
            # Training parameters
            num_episodes = 1000
            max_steps_per_episode = 100
            
            # Training loop
            for episode in range(num_episodes):
                # Reset environment and get initial state
                x, y = envQL.reset()  # Reset the environment to start state
                state_index = envQL.get_state_index()  # Get the current state

                for step in range(max_steps_per_episode):
                    action = agent.choose_action(state_index)
                    new_state, reward, done = envQL.step(action)
                    new_state_index = envQL.get_state_index()
                    agent.update_q_table(state_index, action, reward, new_state_index)
                    
                    state_index = new_state_index
                    
                    if done:
                        break

            # Save the trained Q-table
            np.save("trained_q_table.npy", agent.q_table)
        else:
            print('testing park scenario using QL ...')
            # Load the trained Q-table
            q_table = np.load("trained_q_table.npy")
            agent.exploration_rate = 0  # Disable exploration
            state = envQL.reset()  # Reset the environment to start state
            state_index = envQL.get_state_index(state.x, state.y, env.grid_width)

            done = False
            while not done:
                action = agent.choose_action(state_index)  # Choose best action based on Q-table
                new_state, reward, done = env.step(action)  # Take the action
                new_state_index = envQL.get_state_index(new_state.x, new_state.y, env.grid_width)

                state_index = new_state_index
        
    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    cv2.destroyAllWindows()
