import cv2
import numpy as np

M = None
warp_width = None
warp_height = None

def compute_warp_transform(input, warp_size, offset):
	global M, warp_width, warp_height
	warp_width = warp_size[0]
	warp_height = warp_size[1]
	
	input_pts = np.float32(input)
	output_pts = np.float32([[offset,offset], [warp_width-offset, offset], [warp_width-offset,warp_height-offset], [offset,warp_height-offset]])
	
	M = cv2.getPerspectiveTransform(input_pts, output_pts)

def warp(frame):
	global M
	out = cv2.warpPerspective(frame, M, (warp_width, warp_height), flags=cv2.INTER_LINEAR)

	return out

def get_color_warped(color_frame):
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    warp_color_image = warp(color_image)
    warp_color_image = cv2.addWeighted( warp_color_image, 1.1, warp_color_image, 0, 0)

    return warp_color_image, color_image

def get_depth_warped(depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    #depth_image_8u = cv2.convertScaleAbs(depth_image, alpha=255.0/6000.0, beta=0)0*
    warp_depth_image = warp(depth_image)
    warp_depth_image = cv2.GaussianBlur(warp_depth_image, (5, 5), 0);

    return warp_depth_image, depth_image

def get_ir_warped(infrared_frame, clahe, size):
    ir_image = np.asanyarray(infrared_frame.get_data())
    ir_image  = cv2.resize(ir_image, (size[1],size[0]))
    warp_ir_image = warp(ir_image)
    warp_ir_image = clahe.apply(warp_ir_image)

    return warp_ir_image, ir_image