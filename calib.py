# First import library
from cv2 import circle
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse

display_roi = []
def click_event(event, x, y, flags, params):
	global display_roi
	if event == cv2.EVENT_LBUTTONDOWN:
		print("x: " + str(x) + ", y:" + str(y))
		display_roi.append([x, y])

		if len(display_roi) == 4:
			print("ROI: " + str(display_roi))
			display_roi = []

	if event == cv2.EVENT_RBUTTONDOWN:
		display_roi = []
		print("ROI: " + str(display_roi))

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Ball Detection using Realsense D455")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file", required=False)
parser.add_argument("-d", "--debug", type=bool, help="Debug Flag", required=False)
# Parse the command line arguments to an object
args = parser.parse_args()

cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Frame', click_event)

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

# Streaming loop
while True:
	#processing_time_start = time.time()
	# Get frameset of depth
	frames = pipeline.wait_for_frames()

	aligned_frames = align.process(frames)
	color_frame = aligned_frames.get_color_frame()
	color_image = np.asanyarray(color_frame.get_data())
	
	for p in display_roi:
		cv2.circle(color_image, p, 3, (255, 0, 0), 2)
	
	cv2.imshow("Frame", color_image)
	key = cv2.waitKey(1)

	if key == 27:    
		cv2.destroyAllWindows()
		break