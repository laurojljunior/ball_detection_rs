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
    #print(first_speed)
    if first_speed <= 10.0:
        return False

    if point_list[-1][0][1] - (img.shape[0] - warp_offset) > (0.2 * warp_offset):
        #print("eita")
        return False

    x = np.array(list(range(1, len(point_list)+1)))

    list_y = []
    for p in point_list:
        list_y.append(p[0][1])

    y = np.array(list_y)
	
    a, b = np.polyfit(x, y, 1)

    travel_dist = euclideanDistance(point_list[0][0], point_list[-1][0])

    # print("a: " + str(a))
    # print("travel: " + str(travel_dist))
    if a <= 3 and travel_dist >= 30.0:
        return True
    
    return False

def contourCloserToPose(contours, pose):
    minDist = 9999
    minIdx = 0
    for i, c in enumerate(contours):
        c_circle = cv2.minEnclosingCircle(c)
        dist = euclideanDistance(c_circle[0], pose[0])
        if dist < minDist:
            minDist = dist
            minIdx = i

    return contours[minIdx]

		
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
    hole_filter = rs.hole_filling_filter()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    if args.input:
        rs.config.enable_device_from_file(config, args.input)
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
    else:
        profile = pipeline.start(config)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.laser_power, 360)   

    align_to = rs.stream.color
    align = rs.align(align_to)

    backSubColor = cv2.createBackgroundSubtractorKNN(history = 30, dist2Threshold=int(config_args["Params"]["BgsSensibility"]), detectShadows=True)
    #backSubDepth = cv2.createBackgroundSubtractorKNN(dist2Threshold=150.0, detectShadows=False)
    backSubIR = cv2.createBackgroundSubtractorKNN(history = 30, dist2Threshold=int(config_args["Params"]["BgsSensibility"]), detectShadows=False)

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
        aligned_frames = align.process(frames)

        # depth_frame = aligned_frames.get_depth_frame()
        # depth_frame = depth_to_disparity.process(depth_frame)
        # depth_frame = dec_filter.process(depth_frame)
        # depth_frame = temp_filter.process(depth_frame)
        # #depth_frame = hole_filter.process(depth_frame)
        # depth_frame = disparity_to_depth.process(depth_frame)
        # depth_frame = depth_frame.as_depth_frame()

        color_frame = aligned_frames.get_color_frame()
        infrared_frame = aligned_frames.get_infrared_frame()

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        warp_color_image = warp(color_image, M, (warp_width, warp_height))
        warp_color_image = cv2.addWeighted( warp_color_image, 1.1, warp_color_image, 0, 0)
        warp_gray_image = cv2.cvtColor(warp_color_image, cv2.COLOR_BGR2GRAY)

        warp_canny_image = cv2.Canny(warp_gray_image, 50, 200)

        #cv2.imshow("warp_gray_image", warp_gray_image)
        #cv2.imshow("warp_canny_image", warp_canny_image)

        # depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image_8u = cv2.convertScaleAbs(depth_image, alpha=255.0/6000.0, beta=0)
        # warp_depth_image = warp(depth_image, M, (warp_width, warp_height))
        # warp_depth_image = cv2.GaussianBlur(warp_depth_image, (5, 5), 0);
        # warp_depth_image_8u = cv2.convertScaleAbs(warp_depth_image, alpha=255.0/6000.0, beta=0)

        ir_image = np.asanyarray(infrared_frame.get_data())
        ir_image  = cv2.resize(ir_image, (color_image.shape[1], color_image.shape[0]))
        warp_ir_image = warp(ir_image, M, (warp_width, warp_height))
        warp_ir_image = clahe.apply(warp_ir_image)

        kernel = np.ones((5, 5), np.uint8)  
        
        lab_image = cv2.cvtColor(warp_color_image, cv2.COLOR_BGR2LAB)
        #orange
        # minLAB = np.array([75, 150, 125])
        # maxLAB = np.array([255, 255, 255])

        #white 
        minLAB = np.array([75, 81, 125])
        maxLAB = np.array([245, 255, 255])

        maskLAB = cv2.inRange(lab_image, minLAB, maxLAB)
        maskLAB = cv2.erode(maskLAB, kernel, anchor=(2, 2))
        maskLAB = cv2.dilate(maskLAB, kernel, anchor=(2, 2), iterations=2)

        cv2.imshow("maskLAB", maskLAB)

        color_mask = backSubColor.apply(warp_color_image)
        _, color_mask = cv2.threshold(color_mask, 200, 255, cv2.THRESH_BINARY);
        color_mask = cv2.erode(color_mask, kernel, anchor=(2, 2))
        color_mask = cv2.dilate(color_mask, kernel, anchor=(2, 2), iterations=2)

        cv2.imshow("color_mask", color_mask)

        ir_mask = backSubIR.apply(warp_ir_image)
        _, ir_mask = cv2.threshold(ir_mask, 200, 255, cv2.THRESH_BINARY);
        ir_mask = cv2.erode(ir_mask, kernel, anchor=(2, 2))
        ir_mask = cv2.dilate(ir_mask, kernel, anchor=(2, 2), iterations=3)

        cv2.imshow("ir_mask", ir_mask)

        foreground_mask = (color_mask | ir_mask) & maskLAB

        ball_future_mask = np.ones(maskLAB.shape, np.uint8)  
        if len(ball_tracking_list) >= 2:
            ball_prev_pos = ball_tracking_list[-2][0]
            ball_curr_pos = ball_tracking_list[-1][0]
            ball_estimated_pos = (ball_curr_pos[0] + (ball_curr_pos[0] - ball_prev_pos[0]), ball_curr_pos[1] + (ball_curr_pos[1] - ball_prev_pos[1]))
            angle = math.atan2((ball_prev_pos[1] - ball_curr_pos[1]), (ball_prev_pos[0] - ball_curr_pos[0])) * 180.0 / math.pi + 90.0 
            #print("angle: " + str(angle))
            cv2.ellipse(ball_future_mask, (int(ball_estimated_pos[0]), int(ball_estimated_pos[1])), (30, 60), angle, angle, angle+360, (255), cv2.FILLED)
            #cv2.circle(ball_future_mask, (int(ball_estimated_pos[0]), int(ball_estimated_pos[1])), , (255), cv2.FILLED)
            foreground_mask = foreground_mask & ball_future_mask

        cv2.imshow("ball_future_mask", ball_future_mask)                    
        
        if count_frames > 30:
            contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                c = None
                if len(ball_tracking_list) == 0:   
                    c = max(contours, key = cv2.contourArea)
                else:
                    c = contourCloserToPose(contours, ball_previous_pose)

                c_area = cv2.contourArea(c)
                c_bbox = cv2.boundingRect(c)

                if c_area > (5 * 5) and c_area < (100 * 100):
                    c_circle = cv2.minEnclosingCircle(c)
                    ball_current_pose = c_circle

                    if ball_previous_pose is not None:
                        ball_speed = euclideanDistance(ball_previous_pose[0], ball_current_pose[0]) / 1.0
                        
                        #print("vel: " + str(ball_speed))
                        #print("size: " + str(len(ball_tracking_list)))
                        if len(ball_tracking_list) >=3 and ball_speed < 10.0:
                            is_entering_goal = isEnteringGoal(ball_tracking_list, warp_color_image, warp_offset)
                            ball_tracking_list.clear()

                            if is_entering_goal:
                                send_data = True
                                ball_goal_pose = ball_current_pose

                        else:
                            ball_tracking_list.append(ball_current_pose)

                        if ball_speed < 10.0:
                            ball_previous_pose = None
                            ball_tracking_list.clear()


                    ball_previous_pose = ball_current_pose

            else:
                if len(ball_tracking_list) >=3:
                    #print(ball_tracking_list)
                    is_entering_goal = isEnteringGoal(ball_tracking_list, warp_color_image, warp_offset)
                    if is_entering_goal:
                        send_data = True
                        #print("speed: " + str(ball_speed))
                        ball_goal_pos = (ball_tracking_list[-1][0][0] + (ball_tracking_list[-1][0][0] - ball_tracking_list[-2][0][0]), ball_tracking_list[-1][0][1] + (ball_tracking_list[-1][0][1] - ball_tracking_list[-2][0][1]))
                        ball_goal_rad = ball_tracking_list[-1][1]
                        ball_goal_pose = (ball_goal_pos, ball_goal_rad)

                    ball_tracking_list.clear()
                else:
                    ball_previous_pose = None
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

        cv2.putText(warp_color_image, "Frame: " + str(count_frames), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        count_frames += 1

        if args.debug:
            # Render image in opencv window 
            #cv2.imshow("Depth 8Bit", depth_image_8u) 
            #cv2.imshow("Color Stream", color_image)
            cv2.imshow("Warp Color",  warp_color_image)
            #cv2.imshow("Warp Depth",  warp_depth_image_8u)
            cv2.imshow("Warp IR",  warp_ir_image)
            cv2.imshow("Foreground Mask", foreground_mask)
            key = cv2.waitKey(0)
            #processing_time_end = time.time()
            #print("Processing Time: " + str(processing_time_end-processing_time_start))
            # if pressed escape exit program
            if key == 27:    
                cv2.destroyAllWindows()
                break

finally:
    pass