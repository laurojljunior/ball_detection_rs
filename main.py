# First import library
from cv2 import circle
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
import configparser

import math
import time
import ast
from udp import UdpServer

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

def read_config(file):
	config = configparser.ConfigParser()
	config.read(file)

	return config

def get_warp_transform(input, warp_size, offset):
	warp_width = warp_size[0]
	warp_height = warp_size[1]
	
	input_pts = np.float32(input)
	output_pts = np.float32([[offset,offset], [warp_width-offset, offset], [warp_width-offset,warp_height-offset], [offset,warp_height-offset]])
	
	M = cv2.getPerspectiveTransform(input_pts, output_pts)

	return M

def warp(frame, M, warp_size):
	out = cv2.warpPerspective(frame, M, (warp_size[0], warp_size[1]), flags=cv2.INTER_LINEAR)

	return out

def euclideanDistance(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def isEnteringGoal(point_list, img, warp_offset):

    first_speed = euclideanDistance(point_list[0][0], point_list[1][0]) / 1.0
    if first_speed < 15.0:
        return False

    if point_list[-1][0][1] - (img.shape[0] - warp_offset) > (0.2 * warp_offset) or point_list[-1][0][1] - warp_offset < (0.2 * warp_offset):
        return False

    x = np.array(list(range(1, len(point_list)+1)))

    list_y = []
    for p in point_list:
        list_y.append(p[0][1])

    y = np.array(list_y)
	
    a, b = np.polyfit(x, y, 1)

    if a <= 0:
        return True
    
    return False
		
def refineBallPosition(pos, mask, radius):
    ball_mask = np.zeros(mask.shape, np.uint8) 
    cv2.circle(ball_mask, (int(pos[0]), int(pos[1])), radius, (255), cv2.FILLED)

    kernel = np.ones((5, 5), np.uint8)
    ball_mask = ball_mask & mask;

    ball_mask = cv2.dilate(ball_mask, kernel, anchor=(2, 2))
    ball_mask = cv2.erode(ball_mask, kernel, anchor=(2, 2))

    contours, hierarchy = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:            
        c = max(contours, key = cv2.contourArea)
        ccircle = cv2.minEnclosingCircle(c)
    
        return ccircle[0], ccircle[1]

    return None, None

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Ball Detection using Realsense D455")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file", required=False)
parser.add_argument("-c", "--config", type=str, help="Path to the config file", required=True)
parser.add_argument("-d", "--debug", type=bool, help="Debug Flag", required=False)
# Parse the command line arguments to an object
args = parser.parse_args()
config_args = read_config(args.config)

try:
    udp_server = UdpServer(UDP_IP, UDP_PORT)

    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    
    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)    

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    if args.input:
        rs.config.enable_device_from_file(config, args.input)
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)
    else:
        profile = pipeline.start(config)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.laser_power, 360)   

    align_to = rs.stream.color
    align = rs.align(align_to)

    backSubColor = cv2.createBackgroundSubtractorKNN(history = 30, dist2Threshold=2000.0, detectShadows=True)
    
    projection_points = ast.literal_eval(config_args["Params"]["ProjectionPoints"])
    warp_offset = int(config_args["Params"]["WarpOffset"])
    warp_width = int(config_args["Params"]["WarpWidth"])
    warp_height = int(config_args["Params"]["WarpHeight"])
    M = get_warp_transform(projection_points, (warp_width, warp_height), warp_offset)

    ball_tracking_list = []
    ball_current_pose = None
    ball_previous_pose = None
    ball_goal_pose = None
    
    is_entering_goal = False
    count_frames = 0
    show_time = 0
    send_data = False

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Streaming loop
    start_time = time.time();
    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        warp_color_image = warp(color_image, M, (warp_width, warp_height))
        
        kernel = np.ones((5, 5), np.uint8)
        color_mask = backSubColor.apply(warp_color_image)
        _, color_mask = cv2.threshold(color_mask, 200, 255, cv2.THRESH_BINARY);
        color_mask = cv2.erode(color_mask, kernel, anchor=(2, 2))
        color_mask = cv2.dilate(color_mask, kernel, anchor=(2, 2), iterations=2)

        foreground_mask = color_mask
        
        if count_frames > 30:
            contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:            
                c = max(contours, key = cv2.contourArea)
                c_area = cv2.contourArea(c)
                c_bbox = cv2.boundingRect(c)

                #print("c_area: " + str(c_area))
                #print("c_bbox: " + str(c_bbox))

                if c_area > (6.4 * 6.4) and c_area < (48 * 48):
                    c_circle = cv2.minEnclosingCircle(c)
                    #isOnGoal = checkBallIsOnGoal(ccircle, warp_depth_image, int(config_args["Params"]["BallIsOnGoalSensibility"]))
                    ball_current_pose = c_circle

                    if ball_previous_pose is not None:
                        ball_speed = euclideanDistance(ball_previous_pose[0], ball_current_pose[0]) / 1.0
                        
                        #print(ball_speed)
                        if len(ball_tracking_list) >=3 and ball_speed < 10.0:
                            #print(ball_tracking_list)
                            is_entering_goal = isEnteringGoal(ball_tracking_list, warp_color_image, warp_offset)
                            ball_tracking_list.clear()

                            if is_entering_goal:
                                send_data = True
                                ball_goal_pose = ball_current_pose

                        else:
                            ball_tracking_list.append(ball_current_pose)

                        if ball_speed < 10.0:
                            ball_tracking_list.clear()


                    ball_previous_pose = ball_current_pose

            else:
                ball_tracking_list.clear()

            if is_entering_goal:
                scale_x = warp_color_image.shape[1] / (warp_color_image.shape[1] - 2 * warp_offset)
                scale_y = warp_color_image.shape[0] / (warp_color_image.shape[0] - 2 * warp_offset)

                if send_data:
                    #print("send data")
                    send_data = False
                    udp_server.send_message(str(((((ball_goal_pose[0][0]-warp_offset)*scale_x) / warp_color_image.shape[1], ((ball_goal_pose[0][1]-warp_offset)*scale_y) / warp_color_image.shape[0]) , ball_goal_pose[1]))+"\n")

                show_time += 1
                if show_time < 5:
                    cv2.circle(warp_color_image, (int(ball_goal_pose[0][0]), int(ball_goal_pose[0][1])), 17, (0, 255, 0), 3)
                    cv2.circle(warp_color_image, (int(ball_goal_pose[0][0]), int(ball_goal_pose[0][1])), 20, (0, 0, 255), 3)
                else:
                    show_time = 0
                    is_entering_goal = False

        count_frames += 1

        if args.debug:
            # Render image in opencv window 
            #cv2.imshow("Depth 8Bit", depth_image_8u) 
            #cv2.imshow("Color Stream", color_image)
            cv2.imshow("Warp Color",  warp_color_image)
            #cv2.imshow("Warp Depth", warp_depth_image_8u)
            #cv2.imshow("Foreground Mask", foreground_mask)
            key = cv2.waitKey(1)
            #processing_time_end = time.time()
            #print("Processing Time: " + str(processing_time_end-processing_time_start))
            # if pressed escape exit program
            if key == 27:    
                cv2.destroyAllWindows()
                break

finally:
    pass