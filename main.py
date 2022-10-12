# First import library
from cv2 import circle
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

import threading
import time
from udp import UdpServer

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

def get_warp_transform(input, warp_size):
	warp_width = warp_size[0]
	warp_height = warp_size[1]
	
	input_pts = np.float32(input)
	output_pts = np.float32([[0,0], [warp_width, 0], [warp_width,warp_height], [0,warp_height]])
	
	M = cv2.getPerspectiveTransform(input_pts, output_pts)

	return M

def warp(frame, M, warp_size):
	out = cv2.warpPerspective(frame, M, (warp_size[0], warp_size[1]), flags=cv2.INTER_LINEAR)

	return out

def isEnteringGoal(first_pos, last_pos):
	if (last_pos[1] - first_pos[1]) <= 0:
		return True
	
	return False
		

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file", required=False)
parser.add_argument("-d", "--debug", type=bool, help="Debug Flag", required=False)
# Parse the command line arguments to an object
args = parser.parse_args()

try:
    udp_server = UdpServer(UDP_IP, UDP_PORT)

    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    depth_to_disparity =  rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    dec_filter = rs.decimation_filter()
    temp_filter = rs.temporal_filter()
    spat_filter = rs.spatial_filter()
    hole_filter = rs.hole_filling_filter()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    if args.input:
        rs.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    # Start streaming from file
    profile = pipeline.start(config)
    if args.input:
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Create colorizer object
    colorizer = rs.colorizer()

    backSubColor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    backSubDepth = cv2.createBackgroundSubtractorKNN(dist2Threshold=150.0, detectShadows=False)
    #projection_points = [[214, 181], [1046, 187], [980, 465], [282, 465]]
    projection_points = [[130, 290], [1130, 274], [1065, 620], [215, 627]]
    #projection_points = [[100, 150], [1150, 153], [1032, 487], [230, 480]]

    M = get_warp_transform(projection_points, (640, 360))

    loc_list = []
    ball_estimated_pos = None
    ball_estimated_rad = None
    is_entering_goal = False
    count_frames = 0
    send_data = False

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
        # depth_frame = spat_filter.process(depth_frame)
        # depth_frame = hole_filter.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        color_frame = aligned_frames.get_color_frame()
		
		# Convert depth_frame to numpy array to render image in opencv
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        # print(depth_intrinsics)
        # for p in projection_points:
        #     cv2.circle(color_image, p, 3, (255, 0, 0), 3)
        #     text = "Distance: " + "{:10.4f}".format(depth_frame.get_distance(p[0], p[1]))
        #     print(text)
        #     cv2.putText(color_image, text, (p[0], p[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     p_xyz = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p[1], p[0]], depth_image[p[1], p[0]])
        #     print("Point: " + str(p_xyz))

        depth_image_8u = cv2.convertScaleAbs(depth_image, alpha=255.0/6000.0, beta=0)

        warp_color_image = warp(color_image, M, (640, 360))
        warp_depth_image = warp(depth_image, M, (640, 360))
        
        warp_depth_image_8u = cv2.convertScaleAbs(warp_depth_image, alpha=255.0/6000.0, beta=0)
        #warp_depth_image_8u = cv2.addWeighted( warp_depth_image_8u, 1.2, warp_depth_image_8u, 0, 0)

        warp_depth_image_8u = cv2.GaussianBlur(warp_depth_image_8u, (5, 5), 0);

        kernel = np.ones((5, 5), np.uint8)

        foreground_mask = backSubColor.apply(warp_color_image)
        _, foreground_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY);
        foreground_mask = cv2.erode(foreground_mask, kernel)
        foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=2)

        depth_mask = backSubDepth.apply(warp_depth_image_8u)
        depth_mask = cv2.dilate(depth_mask, kernel)
        depth_mask = cv2.erode(depth_mask, kernel)
        
        if count_frames > 30:
            foreground_mask = depth_mask & foreground_mask
            contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            balls = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > (10*10):
                    ccircle = cv2.minEnclosingCircle(c)
                    balls.append(ccircle)

            if len(balls) > 0:
                ball = balls[0]
                if warp_depth_image[int(ball[0][1]), int(ball[0][0])] > 1000:			  
                    start_time = time.time();

                    loc_list.append(ball)
                    ball_estimated_pos = None
            else:
                end_time = time.time()
                if (end_time - start_time) >= 0.025:
                    start_time = end_time
                    if len(loc_list) > 1:
                        ball_current_pos = loc_list[len(loc_list)-1][0]
                        ball_previous_pos = loc_list[len(loc_list)-2][0]

                        is_entering_goal = isEnteringGoal(loc_list[0][0], loc_list[len(loc_list)-1][0])
                        if is_entering_goal:
                            send_data = True
                            ball_estimated_pos = [ball_current_pos[0] + (ball_current_pos[0] - ball_previous_pos[0]), ball_current_pos[1] + (ball_current_pos[1] - ball_previous_pos[1])]
                            ball_estimated_rad = int(loc_list[len(loc_list)-1][1])
                        
                    loc_list.clear()

            if ball_estimated_pos != None:
                if is_entering_goal:
                    if send_data:
                        send_data = False
                        udp_server.send_message(str(((ball_estimated_pos[0] / warp_color_image.shape[1], ball_estimated_pos[1] / warp_color_image.shape[0]) , ball_estimated_rad))+"\n")

                    cv2.circle(warp_color_image, (int(ball_estimated_pos[0]), int(ball_estimated_pos[1])), 17, (0, 255, 0), 3)
                    cv2.circle(warp_color_image, (int(ball_estimated_pos[0]), int(ball_estimated_pos[1])), 20, (0, 0, 255), 3)

        count_frames += 1

        if args.debug:
            # Render image in opencv window 
            #cv2.imshow("Depth 8Bit", depth_image_8u) 
            #cv2.imshow("Color Stream", color_image)
            #cv2.imshow("Warp Color",  warp_color_image)
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