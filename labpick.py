import cv2
import numpy as np
import argparse
import time
import pyrealsense2 as rs

cap = None
lab = None

image_width = 1280
image_height = 720

minLAB = None
maxLAB = None

# Define and parse input arguments
def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Input Video or Webcam (0 for Webcam)', required=True)
	
	args = parser.parse_args()

	return args

display_roi = []
def click_event(event, x, y, flags, params):
	global display_roi
	if event == cv2.EVENT_LBUTTONDOWN:
		print("x: " + str(x) + ", y:" + str(y))
		display_roi.append([[x, y]])

		if len(display_roi) == 4:
			print("ROI: " + str(display_roi))
			display_roi = []



def filter_event(x):
	global minLAB, maxLAB
	# Get current positions of all trackbars
	if lab is not None:
		lMin = cv2.getTrackbarPos('LMin', 'image')
		aMin = cv2.getTrackbarPos('AMin', 'image')
		bMin = cv2.getTrackbarPos('BMin', 'image')
		lMax = cv2.getTrackbarPos('LMax', 'image')
		aMax = cv2.getTrackbarPos('AMax', 'image')
		bMax = cv2.getTrackbarPos('BMax', 'image')

		minLAB = np.array([lMin, aMin, bMin])
		maxLAB = np.array([lMax, aMax, bMax])

		maskLAB = cv2.inRange(lab, minLAB, maxLAB)
		cv2.imshow("mask", maskLAB)

def main(args):
	global cap
	global lab

	# Create pipeline
	pipeline = rs.pipeline()

    # Create a config object
	config = rs.config()
    
    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
	config.enable_stream(rs.stream.color, rs.format.rgb8, 30)    

	rs.config.enable_device_from_file(config, args.input)
	profile = pipeline.start(config)
	playback = profile.get_device().as_playback()
	playback.set_real_time(False)

	cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
	cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)

	# Create trackbars for color change
	cv2.createTrackbar('LMin', 'image', 0, 255, filter_event)
	cv2.createTrackbar('AMin', 'image', 0, 255, filter_event)
	cv2.createTrackbar('BMin', 'image', 0, 255, filter_event)
	cv2.createTrackbar('LMax', 'image', 0, 255, filter_event)
	cv2.createTrackbar('AMax', 'image', 0, 255, filter_event)
	cv2.createTrackbar('BMax', 'image', 0, 255, filter_event)

	# Set default value for Max LAB trackbars
	cv2.setTrackbarPos('LMax', 'image', 255)
	cv2.setTrackbarPos('AMax', 'image', 255)
	cv2.setTrackbarPos('BMax', 'image', 255)

	cv2.setMouseCallback('frame', click_event)

	while True:
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		color_image = np.asanyarray(color_frame.get_data())
		lab = cv2.cvtColor(color_image, cv2.COLOR_RGB2LAB)
		
		color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
		cv2.imshow("frame", color_image)

		time.sleep(0.033)
		if cv2.waitKey(0) == ord('q'):
			print("Min LAB: " + str(minLAB))
			print("Max LAB: " + str(maxLAB))
			cv2.destroyAllWindows()
			break	

if __name__ == "__main__":
    args = read_args()
    main(args)
