import cv2
import numpy as np
from time import sleep
import argparse
import random

from environment_v1 import Environment, Parking1
from pathplanning_V1 import PathPlanning, ParkPathPlanning, interpolate_path
from control_v1 import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
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

    ########################## default variables for start and end ################################################
    start = np.array([args.x_start, args.y_start])
    original_end   = np.array([85, 10])
    #############################################################################################

    # environment margin  : 5
    # pathplanning margin : 5

    ########################## defining obstacles ###############################################
    random_emptyslot = np.random.randint(-10,24)
    parking1 = Parking1(5,original_end) # random parking slot selection ## of can be args.parking
    end,car_obs,env_obs = parking1.generate_obstacles()



    # Adding of obstables to the environment
    square1 = make_square(10,65,20)
    square2 = make_square(15,30,20)
    square3 = make_square(50,50,10)
    env_obs = np.vstack([env_obs,square1,square2,square3])
    new_obs = np.array([[78,78],[79,79],[78,79]])
    env_obs = np.vstack([env_obs,new_obs])
    #############################################################################################

    ########################### initialization ##################################################
    env = Environment(env_obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    MPC_HORIZON = 5
    controller = MPC_Controller()
    # controller = Linear_MPC_Controller()

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ############################# path planning #################################################
    park_path_planner = ParkPathPlanning(car_obs)
    path_planner = PathPlanning(car_obs)

    # print('planning park scenario ...')
    # new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    
    print('routing to destination ...')
    path = path_planner.plan_path(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    # path = np.vstack([path, ensure_path1])

    print('interpolating ...')
    interpolated_path = interpolate_path(path, sample_rate=5)
    # interpolated_park_path = interpolate_path(park_path, sample_rate=2) 
    # interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])

    env.draw_path(interpolated_path)
    # env.draw_path(interpolated_park_path)

    final_path = interpolated_path
    # final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

    #############################################################################################

    ################################## control ##################################################
    print('driving to destination ...')
    for i,point in enumerate(final_path):

            
            acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
            my_car.update_state(my_car.move(acc,  delta))
            res = env.render(my_car.x, my_car.y, my_car.psi, delta)
            logger.log(point, my_car, acc, delta)
            cv2.imshow('environment', res)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite('res.png', res*255)
            if point[0] -0.5 == original_end[0] and point[1] -0.5 == original_end[1]:
                print('~ No parking slots found, driver to take over ~')

    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    #############################################################################################

    cv2.destroyAllWindows()

