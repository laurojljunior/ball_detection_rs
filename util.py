import cv2
import numpy as np

def euclideanDistance(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def isEnteringGoal(point_list, img, warp_offset, travel_dist_thresh):

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

    #print("a: " + str(a))
    #print("travel: " + str(travel_dist))
    if a <= 1 and travel_dist >= travel_dist_thresh:
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
