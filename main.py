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

def isEnteringGoal(point_list, img):
    if math.fabs(point_list[-1][0][1] - img.shape[0]) < 32:
        return False

    x = np.array(list(range(1, len(point_list)+1)))

    list_y = []
    for p in point_list:
        list_y.append(p[0][1])

    y = np.array(list_y)
	
    a, b = np.polyfit(x, y, 1)

    if a <= 3:
        return True
    
    return False

def checkBallIsOnGoal(ccircle, depth_map, thresh):
    pos = ccircle[0]
    rad = 1.2 * ccircle[1]
    depth_center = depth_map[int(pos[1]), int(pos[0])]

    x = 0 if pos[0]-rad < 0 else pos[0]-rad
    depth_left = depth_map[int(pos[1]), int(x)]

    x = (depth_map.shape[1]-1) if pos[0]+rad >= depth_map.shape[1] else pos[0]+rad
    depth_right = depth_map[int(pos[1]), int(x)]

    y = 0 if pos[1]-rad < 0 else pos[1]-rad
    depth_top = depth_map[int(y), int(pos[0])]

    y = (depth_map.shape[0]-1) if pos[1]+rad >= depth_map.shape[0] else pos[1]+rad
    depth_bottom = depth_map[int(y), int(pos[0])]

    x = 0 if pos[0]-rad < 0 else pos[0]-rad
    y = 0 if pos[1]-rad < 0 else pos[1]-rad
    depth_top_left = depth_map[int(y), int(x)]

    x = (depth_map.shape[1]-1) if pos[0]+rad >= depth_map.shape[1] else pos[0]+rad
    y = 0 if pos[1]-rad < 0 else pos[1]-rad
    depth_top_right = depth_map[int(y), int(x)]

    x = 0 if pos[0]-rad < 0 else pos[0]-rad
    y = (depth_map.shape[0]-1) if pos[1]+rad >= depth_map.shape[0] else pos[1]+rad
    depth_bottom_left = depth_map[int(y), int(x)]

    x = (depth_map.shape[1]-1) if pos[0]+rad >= depth_map.shape[1] else pos[0]+rad
    y = (depth_map.shape[0]-1) if pos[1]+rad >= depth_map.shape[0] else pos[1]+rad
    depth_bottom_right = depth_map[int(y), int(x)]

    std = np.std(np.array([depth_center, depth_left, depth_right, depth_top, depth_bottom, depth_top_left, depth_top_right, depth_bottom_left, depth_bottom_right]))

    if std < thresh:
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
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, rs.format.y8, 30);     

    depth_to_disparity =  rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    dec_filter = rs.decimation_filter()
    temp_filter = rs.temporal_filter()

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

    backSubColor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    backSubDepth = cv2.createBackgroundSubtractorKNN(dist2Threshold=150.0, detectShadows=False)
    backSubIR = cv2.createBackgroundSubtractorKNN(dist2Threshold=300.0, detectShadows=False)
    

    projection_points = ast.literal_eval(config_args["Params"]["ProjectionPoints"])
    warp_offset = int(config_args["Params"]["WarpOffset"])
    warp_width = int(config_args["Params"]["WarpWidth"])
    warp_height = int(config_args["Params"]["WarpHeight"])
    M = get_warp_transform(projection_points, (warp_width, warp_height), warp_offset)

    loc_list = []
    ball_estimated_pos = None
    ball_estimated_rad = None
    is_entering_goal = False
    count_frames = 0
    send_data = False

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Streaming loop
    start_time = time.time();
    while True:
        #processing_time_start = time.time()
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        # Get depth frame
        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = dec_filter.process(depth_frame)
        depth_frame = temp_filter.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        color_frame = aligned_frames.get_color_frame()
        infrared_frame = aligned_frames.get_infrared_frame()
		
		# Convert depth_frame to numpy array to render image in opencv
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(infrared_frame.get_data())

        depth_image_8u = cv2.convertScaleAbs(depth_image, alpha=255.0/6000.0, beta=0)

        warp_color_image = warp(color_image, M, (warp_width, warp_height))
        warp_depth_image = warp(depth_image, M, (warp_width, warp_height))
        warp_depth_image = cv2.GaussianBlur(warp_depth_image, (5, 5), 0);

        ir_image  = cv2.resize(ir_image, (depth_image.shape[1], depth_image.shape[0]))
        warp_ir_image = warp(ir_image, M, (warp_width, warp_height))
        warp_ir_image = clahe.apply(warp_ir_image)
        
        warp_depth_image_8u = cv2.convertScaleAbs(warp_depth_image, alpha=255.0/6000.0, beta=0)
        #warp_depth_image_8u = cv2.GaussianBlur(warp_depth_image_8u, (5, 5), 0);

        kernel = np.ones((7, 7), np.uint8)
        color_mask = backSubColor.apply(warp_color_image)
        _, color_mask = cv2.threshold(color_mask, 200, 255, cv2.THRESH_BINARY);
        color_mask_orig = color_mask
        color_mask = cv2.erode(color_mask, kernel, anchor=(3, 3))
        color_mask = cv2.dilate(color_mask, kernel, anchor=(3, 3), iterations=3)

        #cv2.imshow("color_mask", color_mask_orig)

        depth_mask = backSubDepth.apply(warp_depth_image_8u)
        depth_mask = cv2.dilate(depth_mask, kernel, anchor=(3, 3))
        depth_mask = cv2.erode(depth_mask, kernel, anchor=(3, 3))

        #cv2.imshow("depth_mask", depth_mask)

        kernel = np.ones((17, 17), np.uint8)
        ir_mask = backSubIR.apply(warp_ir_image)
        ir_mask = cv2.dilate(ir_mask, kernel, anchor=(8, 8))
        #ir_mask = cv2.erode(ir_mask, kernel)

        #cv2.imshow("ir_mask", ir_mask)

        foreground_mask = color_mask
        
        if count_frames > 30:
            foreground_mask = (depth_mask & ir_mask) | (ir_mask & color_mask)
            contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            balls = []
            if len(contours) > 0:            
                c = max(contours, key = cv2.contourArea)
                ccircle = cv2.minEnclosingCircle(c)
                isOnGoal = checkBallIsOnGoal(ccircle, warp_depth_image, int(config_args["Params"]["BallIsOnGoalSensibility"]))
                if not isOnGoal:
                    balls.append(ccircle)

            if len(balls) > 0:
                ball = balls[0]
                depth = warp_depth_image[int(ball[0][1]), int(ball[0][0])]
                if depth > 1000:			  
                    start_time = time.time();
                    loc_list.append(ball)
                    ball_estimated_pos = None
            else:
                end_time = time.time()
                if (end_time - start_time) >= 0.025:
                    start_time = end_time
                    if len(loc_list) > 1:
                        ball_current_pos = loc_list[-1][0]
                        ball_previous_pos = loc_list[-2][0]

                        is_entering_goal = isEnteringGoal(loc_list, color_mask_orig)
                        if is_entering_goal:
                            send_data = True
                            ball_estimated_pos = ball_current_pos
                            ball_estimated_rad = loc_list[-1][1]
                            
                            refined_ball_pos, refined_ball_rad = refineBallPosition(ball_current_pos, color_mask_orig, 32) #[ball_current_pos[0] + (ball_current_pos[0] - ball_previous_pos[0]), ball_current_pos[1] + (ball_current_pos[1] - ball_previous_pos[1])]

                            if refined_ball_pos is not None and refined_ball_rad is not None:
                                ball_estimated_pos = refined_ball_pos
                                ball_estimated_rad = refined_ball_rad
                        
                    loc_list.clear()

            if ball_estimated_pos != None:
                if is_entering_goal:
                    scale_x = warp_color_image.shape[1] / (warp_color_image.shape[1] - 2 * warp_offset)
                    scale_y = warp_color_image.shape[0] / (warp_color_image.shape[0] - 2 * warp_offset)

                    if send_data:
                        send_data = False
                        udp_server.send_message(str(((((ball_estimated_pos[0]-warp_offset)*scale_x) / warp_color_image.shape[1], ((ball_estimated_pos[1]-warp_offset)*scale_y) / warp_color_image.shape[0]) , ball_estimated_rad))+"\n")

                    cv2.circle(warp_color_image, (int(ball_estimated_pos[0]), int(ball_estimated_pos[1])), 17, (0, 255, 0), 3)
                    cv2.circle(warp_color_image, (int(ball_estimated_pos[0]), int(ball_estimated_pos[1])), 20, (0, 0, 255), 3)
                    

        count_frames += 1

        if args.debug:
            # Render image in opencv window 
            # cv2.imshow("Depth 8Bit", depth_image_8u) 
            # cv2.imshow("Color Stream", color_image)
            cv2.imshow("Warp Color",  warp_color_image)
            # cv2.imshow("Warp IR",  warp_ir_image)
            # cv2.imshow("Warp Depth", warp_depth_image_8u)
            # cv2.imshow("Foreground Mask", foreground_mask)
            key = cv2.waitKey(1)
            #processing_time_end = time.time()
            #print("Processing Time: " + str(processing_time_end-processing_time_start))
            # if pressed escape exit program
            if key == 27:    
                cv2.destroyAllWindows()
                break

finally:
    pass