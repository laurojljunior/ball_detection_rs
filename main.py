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

warp_offset_param = None
warp_offset_param_has_changed = True
bgs_sensibility_param = None
bgs_sensibility_param_has_changed = True
ball_hit_threshold_param = None
ball_hit_threshold_param_has_changed = True
travel_dist_threshold_param = None
travel_dist_threshold_param_has_changed = True
ground_line_threshold_param = None
ground_line_threshold_param_has_changed = True

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
        playback.set_real_time(True)
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

def on_change_warp_offset_param(value):
    global warp_offset_param, warp_offset_param_has_changed
    warp_offset_param = value
    warp_offset_param_has_changed = True

def on_change_bgs_sensibility_param(value):
    global bgs_sensibility_param, bgs_sensibility_param_has_changed
    bgs_sensibility_param = value
    bgs_sensibility_param_has_changed = True

def on_change_ball_hit_threshold_param(value):
    global ball_hit_threshold_param, ball_hit_threshold_param_has_changed
    ball_hit_threshold_param = value
    ball_hit_threshold_param_has_changed = True

def on_change_travel_dist_threshold_param(value):
    global travel_dist_threshold_param, travel_dist_threshold_param_has_changed
    travel_dist_threshold_param = value
    travel_dist_threshold_param_has_changed = True

def on_change_ground_line_threshold_param(value):
    global ground_line_threshold_param, ground_line_threshold_param_has_changed
    ground_line_threshold_param = value
    ground_line_threshold_param_has_changed = True

def write_config_file(filename, config):
    global warp_offset_param, bgs_sensibility_param, ball_hit_threshold_param, travel_dist_threshold_param
    with open(filename, 'w') as configfile:
        config_args["Params"]["SideOffset"] = str(warp_offset_param)
        config_args["Params"]["BgsSensibility"] = str(bgs_sensibility_param)
        config_args["Params"]["BallHitThreshold"] = str(ball_hit_threshold_param)
        config_args["Params"]["TravelDistanceThreshold"] = str(travel_dist_threshold_param)
        config_args["Params"]["GroundLineThreshold"] = str(ground_line_threshold_param)
        config.write(configfile)

def main(args, config_args):
    global warp_offset_param, bgs_sensibility_param, ball_hit_threshold_param, travel_dist_threshold_param, ground_line_threshold_param
    global warp_offset_param_has_changed, bgs_sensibility_param_has_changed, ball_hit_threshold_param_has_changed, travel_dist_threshold_param_has_changed, ground_line_threshold_param_has_changed
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
    ball_hit_threshold_param = int(config_args["Params"]["BallHitThreshold"])
    travel_dist_threshold_param = int(config_args["Params"]["TravelDistanceThreshold"])
    ground_line_threshold_param = int(config_args["Params"]["GroundLineThreshold"])

    
    if args.debug is not None:
        cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar("Side Offset", "Frame", warp_offset_param, 100, on_change_warp_offset_param)
        cv2.setTrackbarMin("Side Offset", "Frame", 0)

        cv2.createTrackbar("Background Subtraction Sensibility", "Frame", bgs_sensibility_param, 2000, on_change_bgs_sensibility_param)
        cv2.setTrackbarMin("Background Subtraction Sensibility", "Frame", 300)

        cv2.createTrackbar("Ball Hit Speed Threshold", "Frame", ball_hit_threshold_param, 50, on_change_ball_hit_threshold_param)
        cv2.setTrackbarMin("Ball Hit Speed Threshold", "Frame", 10)

        cv2.createTrackbar("Ball Traveling Distance Threshold", "Frame", travel_dist_threshold_param, 50, on_change_travel_dist_threshold_param)
        cv2.setTrackbarMin("Ball Traveling Distance Threshold", "Frame", 1)

        cv2.createTrackbar("Ground Line Threshold", "Frame", ground_line_threshold_param, warp_height_param, on_change_ground_line_threshold_param)
        cv2.setTrackbarMin("Ground Line Threshold", "Frame", int(warp_height_param / 2))
    
    ball_tracking_list = []
    ball_current_pose = None
    ball_previous_pose = None
    ball_goal_pose = None
    
    is_entering_goal = False
    count_frames = 0
    count_frames_fps = 0
    fps = 0
    show_time = 0
    send_data = False

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    net_plane = None
    net_plane_memory = 0
    depth_image_list = []

    depth_intrinsics = None
    start_time = time.time()
    # Streaming loop
    while True:
        if warp_offset_param_has_changed:
            warp_offset_param_has_changed = False
            projection.compute_warp_transform(projection_points_param, (warp_width_param, warp_height_param), warp_offset_param)

        if bgs_sensibility_param_has_changed:
            bgs_sensibility_param_has_changed = False
            backSubColor = cv2.createBackgroundSubtractorKNN(history = 30, dist2Threshold=bgs_sensibility_param, detectShadows=True)
            backSubDepth = cv2.createBackgroundSubtractorKNN(history = 30, dist2Threshold=100.0, detectShadows=False)
            backSubIR = cv2.createBackgroundSubtractorKNN(history = 30, dist2Threshold=bgs_sensibility_param, detectShadows=False)

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = dec_filter.process(depth_frame)
        depth_frame = temp_filter.process(depth_frame)
        depth_frame = hole_filter.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        infrared_frame = aligned_frames.get_infrared_frame()

        if depth_intrinsics is None:
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            print("Depth Intrinsics: " + str(depth_intrinsics))

        warp_depth_image, depth_image = projection.get_depth_warped(depth_frame)
        depth_image_8u = cv2.convertScaleAbs(depth_image, alpha=255.0/6000.0, beta=0)
        warp_depth_image_8u = cv2.convertScaleAbs(warp_depth_image, alpha=255.0/6000.0, beta=0)
        warp_color_image, color_image = projection.get_color_warped(color_frame)
        warp_ir_image, ir_image = projection.get_ir_warped(infrared_frame, color_image.shape)
        warp_ir_image = clahe.apply(warp_ir_image)
        
        lab_mask = vision.get_lab_mask(warp_color_image, min_lab_color_param, max_lab_color_param)
        #cv2.imshow("lab_mask", lab_mask)

        color_mask, color_mask_orig = vision.get_color_mask(warp_color_image, backSubColor)
        #cv2.imshow("color_mask", color_mask)

        # depth_mask = vision.get_depth_mask(warp_depth_image_8u, backSubDepth)
        # cv2.imshow("depth_mask", depth_mask)

        ir_mask = vision.get_ir_mask(warp_ir_image, backSubIR)
        #cv2.imshow("ir_mask", ir_mask)

        foreground_mask = (color_mask | ir_mask) & lab_mask

        if len(ball_tracking_list) >= 2:
            ball_future_mask = vision.get_ball_future_mask(ball_tracking_list, foreground_mask.shape)  
            foreground_mask = foreground_mask & ball_future_mask
            #cv2.imshow("ball_future_mask", ball_future_mask)

        #define net plane based on first 30 depth samples
        if net_plane is None:
            if net_plane_memory <= 30:
                net_plane_memory +=1
                depth_image_list.append(depth_image.copy())

                if net_plane_memory == 30:
                    depth_image_mean = np.mean(depth_image_list, axis=0)
                    depth_image_mean = depth_image_mean.astype(np.uint16)
                    net_plane = util.get_net_plane_from_projection_points(projection_points_param, depth_image_mean, depth_intrinsics)
        
        if count_frames > 30:
            contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                c = None
                if len(ball_tracking_list) == 0:   
                    c = max(contours, key = cv2.contourArea)
                else:
                    c = util.contourCloserToPose(contours, ball_previous_pose)

                c_area = cv2.contourArea(c)
                if c_area > (5 * 5) and c_area < (100 * 100):
	
                    c_circle = cv2.minEnclosingCircle(c)
                    ball_current_pose = c_circle

                    ball_position_unwarped = projection.unwarp_point(ball_current_pose[0])
                    ball_position_pt = util.get_xyz_from_neighbors(depth_intrinsics, depth_image, ball_position_unwarped)
                    ball_dist_to_net_plane = util.distance_to_plane(ball_position_pt, net_plane)
                    ball_current_pose += (ball_dist_to_net_plane, )

                    if ball_previous_pose is not None:
                        ball_speed = util.euclideanDistance(ball_previous_pose[0], ball_current_pose[0]) / 1.0

                        # print("vel: " + str(ball_speed))
                        # print("size: " + str(len(ball_tracking_list)))
                        if ball_current_pose[0][1] < ground_line_threshold_param:
                            if len(ball_tracking_list) >=2 and ball_speed < ball_hit_threshold_param:
                                is_entering_goal = util.isEnteringGoal(ball_tracking_list, warp_color_image, warp_offset_param, travel_dist_threshold_param)
                                ball_tracking_list.clear()

                                if is_entering_goal:
                                    send_data = True
                                    show_time = 0

                                    ball_refined_pose = util.refineBallPosition(ball_current_pose[0], ir_mask, 20)
                                    if ball_refined_pose is not None:
                                        ball_goal_pose = ball_refined_pose
                                    else:
                                        ball_goal_pose = ball_current_pose
                            else:
                                ball_tracking_list.append(ball_current_pose)
                        else:
                            gl_dist = math.fabs(ball_current_pose[0][1] - ground_line_threshold_param)
                            if len(ball_tracking_list) >= 2 and gl_dist < 10 and ball_speed < ball_hit_threshold_param:
                                is_entering_goal = util.isEnteringGoal(ball_tracking_list, warp_color_image, warp_offset_param, 0)
                                ball_tracking_list.clear()

                                if is_entering_goal:
                                    send_data = True
                                    show_time = 0
                                    
                                    ball_refined_pose = util.refineBallPosition(ball_current_pose[0], ir_mask, 20)
                                    if ball_refined_pose is not None:
                                        ball_goal_pose = ball_refined_pose
                                    else:
                                        ball_goal_pose = ball_current_pose
                            else:
                                ball_tracking_list.append(ball_current_pose)

                        bottom_dist = math.fabs(ball_current_pose[0][1] - warp_color_image.shape[0])
                        if bottom_dist < 10:
                            ball_previous_pose = None
                            ball_tracking_list.clear()

                    ball_previous_pose = ball_current_pose

            else:
                if len(ball_tracking_list) >=2:
                    #print(ball_tracking_list)
                    is_entering_goal = util.isEnteringGoal(ball_tracking_list, warp_color_image, warp_offset_param, travel_dist_threshold_param)
                    if is_entering_goal:
                        send_data = True
                        show_time = 0
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
        cv2.line(warp_color_image, (0, ground_line_threshold_param), (warp_color_image.shape[1]-1, ground_line_threshold_param), (255, 0, 255), 1)


        count_frames += 1
        count_frames_fps+=1
        if (time.time() - start_time) > 1.0 :
            fps = count_frames_fps / (time.time() - start_time)
            count_frames_fps = 0
            start_time = time.time()

        cv2.putText(warp_color_image, "FPS: " + str(int(fps + 0.5)), (warp_color_image.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if args.debug is not None:
            # Render image in opencv window 
            #cv2.imshow("Color Stream", color_image)
            #cv2.imshow("Depth Stream", depth_image_8u)
            cv2.imshow("Frame",  warp_color_image)
            #cv2.imshow("Warp Depth", warp_depth_image_8u)
            #cv2.imshow("Warp IR",  warp_ir_image)
            #cv2.imshow("Foreground Mask", foreground_mask)
            key = cv2.waitKey(1)
            
            # if pressed escape exit program
            if key == 27:
                write_config_file(args.config, config_args)
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    args = read_args()
    config_args = read_config(args.config)
    main(args, config_args)