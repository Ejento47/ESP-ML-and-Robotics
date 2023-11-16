import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import argparse
import random

from environment_v3 import Environment, Parking1
from pathplanning_V3 import  ParkPathPlanning, interpolate_b_spline_path, interpolate_path
from control_v3 import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=90, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    # # parser.add_argument('--x_end', type=int, default=90, help='X of end') #to be removed since hard-coded
    # # parser.add_argument('--y_end', type=int, default=80, help='Y of end') #to be removed since hard-coded
    # parser.add_argument('--parking', type=int, default=1, help='park position in parking1 out of 24') #to remvoe since parking slot randomized

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables for start and end ##############################
    start = np.array([args.x_start, args.y_start])
    original_end   = np.array([85, 10])
    #############################################################################################

    # environment margin  : 5
    # pathplanning margin : 5

    ########################## defining obstacles ###############################################
    random_emptyslot = np.random.randint(-10,24)
    parking1 = Parking1(24,original_end) # random parking slot selection ## of can be args.parking
    end,car_obs,env_obs = parking1.generate_obstacles() #car_obs is what car can see and env_obs is what see
    print(end)
    print(car_obs)
    print(env_obs)


 
    # Adding of obstables to the environment
    square1 = make_square(10,65,20)
    square2 = make_square(15,30,20)
    square3 = make_square(50,50,10)
    env_obs = np.vstack([env_obs,square1,square2,square3])
    car_obs = np.vstack([car_obs,square1,square2,square3]) ####TO CHANGE BACK TO CAR_OBS
    new_obs = np.array([[78,78],[79,79],[78,79]])
    env_obs = np.vstack([env_obs,new_obs])
    car_obs = np.vstack([car_obs,new_obs])
    print(car_obs)
    print(env_obs)
    #############################################################################################

    ########################### initialization for map and car ##################################################
    env = Environment(env_obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    MPC_HORIZON = 5
    controller = MPC_Controller()
    # controller = Linear_MPC_Controller()

    #zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################
    
    ############################# path planning to Original Goal ###############################
    path_planner = ParkPathPlanning(car_obs) #path planner class to take in the obstacles visible to the car only

    path = path_planner.plan_path(int(start[0]),int(start[1]),int(original_end[0]),int(original_end[1])) #path planning
    env.draw_path(path)
    ################################## Travel to Original Goal ##################################################
    can_park = False #boolean to check if car can park
    obstacle_found = False #boolean to check if obstacle is found
    counter = 0 #counter to check if car can park
    print('driving to destination ...')
    
    # print(path_planner.a_star.obstacle_map)
    
    while len(path) > 0: #while path is not empty

        acc, delta = controller.optimize(my_car, path[:MPC_HORIZON]) #get acc and delta from controller of the first 5 points in path
        my_car.update_state(my_car.move(acc,  delta)) #update car state
        res = env.render(my_car.x, my_car.y, my_car.psi, delta) #render environment of the car 
        logger.log(path[0], my_car, acc, delta) #log the data of car
        cv2.imshow('environment', res) #show the environment of car
        key = cv2.waitKey(1) 
        if path[0][0] -0.5 == original_end[0] and path[0][1] -0.5 == original_end[1]:
            print('~ No parking slots found, driver to take over ~')
            break
        
        #TEWSITNTSNT SFWSDSDWEER
        if counter > 150: ###TESTTTT NEEDD TO BE REPLACED WHEN CAN PARK IS TRUE
            can_park = True #set can_park to true
            current_pos = path[0]           
            path = [] #clear path array
            path = path_planner.plan_path(int(current_pos[0]),int(current_pos[1]),int(end[0]),int(end[1])) #path planning
            env.draw_path(path)
            
        path = path[1:] #remove the first point in path
        counter += 1
        
        if key == ord('s'):  #to save the imagee
            cv2.imwrite('res.png', res*255)

    
    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    #############################################################################################
    ############################# path planning #################################################


    # print('planning park scenario ...')
    # new_end, park_path, ensure_path1, ensure_path2 = path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    
    # print('routing to destination ...')
    # path = park_path_planner.plan_path(int(start[0]),int(start[1]),int(new_end[0]),int(new_end[1]))
    # path = np.vstack([path, ensure_path1])

    # # print('interpolating ...')
    # # interpolated_path = interpolate_path(path, sample_rate=5)
    # # interpolated_park_path = interpolate_path(park_path, sample_rate=2)
    # # interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])
    
    # park_path = np.vstack([ensure_path1[::-1], park_path, ensure_path2[::-1]])

    # env.draw_path(path)
    # env.draw_path(park_path)

    # final_path = np.vstack([path, park_path, ensure_path2])

    # env.draw_path(park_path)
    # # env.draw_path(interpolated_park_path)

    # # final_path = path
    # final_path = np.vstack([path, park_path])

    #############################################################################################

    # ################################## control ##################################################
    # print('driving to destination ...')
    # for i,point in enumerate(final_path):

            
    #         acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
    #         my_car.update_state(my_car.move(acc,  delta))
            
    #         #sensor update
    #         res = env.render(my_car.x, my_car.y, my_car.psi, delta)
    #         logger.log(point, my_car, acc, delta)
    #         cv2.imshow('environment', res)
    #         key = cv2.waitKey(1)
    #         if key == ord('s'):
    #             cv2.imwrite('res.png', res*255)
    #         if point[0] -0.5 == original_end[0] and point[1] -0.5 == original_end[1]:
    #             print('~ No parking slots found, driver to take over ~')
    #             # break

    
    # # zeroing car steer
    # res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    # logger.save_data()
    # cv2.imshow('environment', res)
    # key = cv2.waitKey()
    # #############################################################################################

    cv2.destroyAllWindows()


#comment this while loop out if useless
'''
    # Start the main loop
    while not point[0] -0.5 == original_end[0] and point[1] -0.5 == original_end[1]:
        # Your code for updating obstacles, path planning, and moving the car goes here

        acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
        my_car.update_state(my_car.move(acc,  delta))
        #sensor update
        res = env.render(my_car.x, my_car.y, my_car.psi, delta)
        logger.log(point, my_car, acc, delta)
        cv2.imshow('environment', res)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('res.png', res*255)


        print('~ No parking slots found, driver to take over ~')

'''