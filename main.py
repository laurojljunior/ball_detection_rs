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
import projection
import util
import vision

import math
import time
import ast
from udp import UdpServer

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

depth_to_disparity =  rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
dec_filter = rs.decimation_filter()
temp_filter = rs.temporal_filter()
hole_filter = rs.hole_filling_filter()

def configure_realsense_pipeline(input_file):
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    
    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)    
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, rs.format.y8, 30);

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    if args.input:
        rs.config.enable_device_from_file(config, input_file, repeat_playback=False)
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

    return pipeline, profile, align

def read_args():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Ball Detection using Realsense D455")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file", required=False)
    parser.add_argument("-c", "--config", type=str, help="Path to the config file", required=True)
    parser.add_argument("-d", "--debug", type=bool, help="Debug Flag", required=False)
    # Parse the command line arguments to an object
    args = parser.parse_args()
    return args

def read_config(file):
	config = configparser.ConfigParser()
	config.read(file)
	return config

def main(args, config_args):
    udp_server = UdpServer(UDP_IP, UDP_PORT)
    pipeline, profile, align = configure_realsense_pipeline(args.input)

    #Configuration file parameters
    projection_points_param = ast.literal_eval(config_args["Params"]["ProjectionPoints"])
    warp_offset_param = int(config_args["Params"]["SideOffset"])
    warp_width_param = 640
    warp_height_param = 360
    bgs_sensibility_param = int(config_args["Params"]["BgsSensibility"])
    min_lab_color_param = ast.literal_eval(config_args["Params"]["MinLabColor"])
    max_lab_color_param = ast.literal_eval(config_args["Params"]["MaxLabColor"])

    projection.compute_warp_transform(projection_points_param, (warp_width_param, warp_height_param), warp_offset_param)

    ball_tracking_list = []
    ball_current_pose = None
    ball_previous_pose = None
    ball_goal_pose = None
    
    is_entering_goal = False
    count_frames = 0
    show_time = 0
    send_data = False

    backSubColor = cv2.createBackgroundSubtractorKNN(history = 30, dist2Threshold=bgs_sensibility_param, detectShadows=True)
    #backSubDepth = cv2.createBackgroundSubtractorKNN(dist2Threshold=150.0, detectShadows=False)
    backSubIR = cv2.createBackgroundSubtractorKNN(history = 30, dist2Threshold=bgs_sensibility_param, detectShadows=False)
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

        #warp_depth_image = projection.get_depth_warped(depth_frame, warp_width_param, warp_height_param)
        #warp_depth_image_8u = cv2.convertScaleAbs(warp_depth_image, alpha=255.0/6000.0, beta=0)
        warp_color_image, color_image = projection.get_color_warped(color_frame)
        warp_ir_image, ir_image = projection.get_ir_warped(infrared_frame, clahe, color_image.shape)

        kernel = np.ones((5, 5), np.uint8)  
        
        lab_mask = vision.get_lab_mask(warp_color_image, kernel, min_lab_color_param, max_lab_color_param)
        cv2.imshow("lab_mask", lab_mask)

        color_mask = vision.get_color_mask(warp_color_image, kernel, backSubColor)
        cv2.imshow("color_mask", color_mask)

        ir_mask = vision.get_ir_mask(warp_ir_image, kernel, backSubIR)
        cv2.imshow("ir_mask", ir_mask)

        foreground_mask = (color_mask | ir_mask) & lab_mask

        ball_future_mask = np.ones(lab_mask.shape, np.uint8)  
        if len(ball_tracking_list) >= 2:
            ball_prev_pos = ball_tracking_list[-2][0]
            ball_curr_pos = ball_tracking_list[-1][0]
            ball_estimated_pos = (ball_curr_pos[0] + (ball_curr_pos[0] - ball_prev_pos[0]), ball_curr_pos[1] + (ball_curr_pos[1] - ball_prev_pos[1]))
            angle = math.atan2((ball_prev_pos[1] - ball_curr_pos[1]), (ball_prev_pos[0] - ball_curr_pos[0])) * 180.0 / math.pi + 90.0 
            cv2.ellipse(ball_future_mask, (int(ball_estimated_pos[0]), int(ball_estimated_pos[1])), (30, 60), angle, angle, angle+360, (255), cv2.FILLED)
            foreground_mask = foreground_mask & ball_future_mask

        cv2.imshow("ball_future_mask", ball_future_mask)                    
        
        if count_frames > 30:
            contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                c = None
                if len(ball_tracking_list) == 0:   
                    c = max(contours, key = cv2.contourArea)
                else:
                    c = util.contourCloserToPose(contours, ball_previous_pose)

                c_area = cv2.contourArea(c)
                c_bbox = cv2.boundingRect(c)

                if c_area > (5 * 5) and c_area < (100 * 100):
                    c_circle = cv2.minEnclosingCircle(c)
                    ball_current_pose = c_circle

                    if ball_previous_pose is not None:
                        ball_speed = util.euclideanDistance(ball_previous_pose[0], ball_current_pose[0]) / 1.0
                        
                        #print("vel: " + str(ball_speed))
                        #print("size: " + str(len(ball_tracking_list)))
                        if len(ball_tracking_list) >=3 and ball_speed < 10.0:
                            is_entering_goal = util.isEnteringGoal(ball_tracking_list, warp_color_image, warp_offset_param)
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
                    is_entering_goal = util.isEnteringGoal(ball_tracking_list, warp_color_image, warp_offset_param)
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
                scale_x = warp_color_image.shape[1] / (warp_color_image.shape[1] - 2 * warp_offset_param)
                scale_y = warp_color_image.shape[0] / (warp_color_image.shape[0] - 2 * warp_offset_param)

                if send_data:
                    #print("send data")
                    send_data = False
                    udp_server.send_message(str(((((ball_goal_pose[0][0]-warp_offset_param)*scale_x) / warp_color_image.shape[1], ((ball_goal_pose[0][1]-warp_offset_param)*scale_y) / warp_color_image.shape[0]) , ball_goal_pose[1]))+"\n")

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
            cv2.imshow("Color Stream", color_image)
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


if __name__ == "__main__":
    args = read_args()
    config_args = read_config(args.config)
    main(args, config_args)