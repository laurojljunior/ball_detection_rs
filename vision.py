import cv2
import numpy as np
import math

def get_lab_mask(warp_color_image, kernel, min, max):
	lab_image = cv2.cvtColor(warp_color_image, cv2.COLOR_BGR2LAB)

	minLAB = np.array(min)
	maxLAB = np.array(max)

	maskLAB = cv2.inRange(lab_image, minLAB, maxLAB)
	maskLAB = cv2.erode(maskLAB, kernel, anchor=(2, 2))
	maskLAB = cv2.dilate(maskLAB, kernel, anchor=(2, 2), iterations=2)

	return maskLAB

def get_color_mask(warp_color_image, kernel, backSubColor):
	color_mask_orig = backSubColor.apply(warp_color_image)
	_, color_mask_orig = cv2.threshold(color_mask_orig, 200, 255, cv2.THRESH_BINARY);
	color_mask = cv2.erode(color_mask_orig, kernel, anchor=(2, 2))
	color_mask = cv2.dilate(color_mask, kernel, anchor=(2, 2), iterations=2)

	return color_mask, color_mask_orig

def get_ir_mask(warp_ir_image, kernel, backSubIR):
	ir_mask = backSubIR.apply(warp_ir_image)
	_, ir_mask = cv2.threshold(ir_mask, 200, 255, cv2.THRESH_BINARY);
	ir_mask = cv2.erode(ir_mask, kernel, anchor=(2, 2))
	ir_mask = cv2.dilate(ir_mask, kernel, anchor=(2, 2), iterations=3)

	return ir_mask

def get_ball_future_mask(ball_tracking_list, size):
	ball_future_mask = np.ones(size, np.uint8)  
	if len(ball_tracking_list) >= 2:
		ball_prev_pos = ball_tracking_list[-2][0]
		ball_curr_pos = ball_tracking_list[-1][0]
		ball_estimated_pos = (ball_curr_pos[0] + (ball_curr_pos[0] - ball_prev_pos[0]), ball_curr_pos[1] + (ball_curr_pos[1] - ball_prev_pos[1]))
		angle = math.atan2((ball_prev_pos[1] - ball_curr_pos[1]), (ball_prev_pos[0] - ball_curr_pos[0])) * 180.0 / math.pi + 90.0 
		cv2.ellipse(ball_future_mask, (int(ball_estimated_pos[0]), int(ball_estimated_pos[1])), (30, 60), angle, angle, angle+360, (255), cv2.FILLED)

	return ball_future_mask
	
