import cv2
import numpy as np

def get_lab_mask(warp_color_image, kernel, min, max):
	lab_image = cv2.cvtColor(warp_color_image, cv2.COLOR_BGR2LAB)

	minLAB = np.array(min)
	maxLAB = np.array(max)

	maskLAB = cv2.inRange(lab_image, minLAB, maxLAB)
	maskLAB = cv2.erode(maskLAB, kernel, anchor=(2, 2))
	maskLAB = cv2.dilate(maskLAB, kernel, anchor=(2, 2), iterations=2)

	return maskLAB

def get_color_mask(warp_color_image, kernel, backSubColor):
	color_mask = backSubColor.apply(warp_color_image)
	_, color_mask = cv2.threshold(color_mask, 200, 255, cv2.THRESH_BINARY);
	color_mask = cv2.erode(color_mask, kernel, anchor=(2, 2))
	color_mask = cv2.dilate(color_mask, kernel, anchor=(2, 2), iterations=2)

	return color_mask

def get_ir_mask(warp_ir_image, kernel, backSubIR):
	ir_mask = backSubIR.apply(warp_ir_image)
	_, ir_mask = cv2.threshold(ir_mask, 200, 255, cv2.THRESH_BINARY);
	ir_mask = cv2.erode(ir_mask, kernel, anchor=(2, 2))
	ir_mask = cv2.dilate(ir_mask, kernel, anchor=(2, 2), iterations=3)

	return ir_mask
