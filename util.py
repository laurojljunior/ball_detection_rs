import cv2
import numpy as np
import math
import pyrealsense2 as rs

def euclideanDistance(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def isEnteringGoal(point_list, img, warp_offset, travel_dist_thresh):
    first_speed = euclideanDistance(point_list[0][0], point_list[1][0]) / 1.0
    #print("first_speed: " + str(first_speed))
    if travel_dist_thresh > 0 and first_speed <= 10.0:
        return False

    if point_list[-1][0][1] - (img.shape[0] - warp_offset) > (0.2 * warp_offset):
        #print("eita")
        return False

    x = np.array(list(range(1, len(point_list)+1)))

    list_y = []
    for p in point_list:
        list_y.append(p[0][1])

    list_d = []
    for p in point_list:
        list_d.append(p[2])

    y = np.array(list_y)
    y_slope, b = np.polyfit(x, y, 1)

    y = np.array(list_d)
    d_slope, b = np.polyfit(x, y, 1)

    travel_dist = euclideanDistance(point_list[0][0], point_list[-1][0])

    #print("y_slope: " + str(y_slope))
    #print("d_slope: " + str(d_slope))
    #print("travel: " + str(travel_dist))
    if y_slope <= 1 and d_slope > 10 and travel_dist >= travel_dist_thresh:
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

def plane_equation(p1, p2, p3):     
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = p2[2] - p1[2]
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = p3[2] - p1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * p1[0] - b * p1[1] - c * p1[2])
    return (a, b, c, d)

def distance_to_plane(p1, plane):
     
    d = abs((plane[0] * p1[0] + plane[1] * p1[1] + plane[2] * p1[2] + plane[3]))
    e = (math.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]))

    return d/e

def get_xyz_from_point(depth_intrinsics, depth_image, p):
    xyz = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p[1], p[0]], depth_image[p[1], p[0]])
    return xyz

def get_xyz_from_neighbors(depth_intrinsics, depth_image, p):
    xyz = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [p[1], p[0]], depth_image[p[1], p[0]])
    z1 = depth_image[p[1]-1, p[0]-1]
    z2 = depth_image[p[1]-1, p[0]]
    z3 = depth_image[p[1]-1, p[0]+1]
    z4 = depth_image[p[1], p[0]-1]
    z5 = depth_image[p[1], p[0]+1]
    z6 = depth_image[p[1]+1, p[0]-1]
    z7 = depth_image[p[1]+1, p[0]]
    z8 = depth_image[p[1]+1, p[0]+1]

    z_mean = (xyz[2] + z1 + z2 + z3 + z4 + z5 + z6 + z7 + z8) / 9.0

    return (xyz[0], xyz[1], z_mean)
		
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

# depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
# projection_points_param1 = (projection_points_param[0][0] + 50, projection_points_param[0][1] + 50)
# projection_points_param2 = (projection_points_param[1][0] - 50, projection_points_param[1][1] + 50)
# projection_points_param3 = (projection_points_param[2][0] - 50, projection_points_param[2][1] - 50)

# p1 = util.get_xyz_from_neighbors(depth_intrinsics, depth_image, projection_points_param1)
# p2 = util.get_xyz_from_neighbors(depth_intrinsics, depth_image, projection_points_param2)
# p3 = util.get_xyz_from_neighbors(depth_intrinsics, depth_image, projection_points_param3)

# if plane is None:
#     plane = util.plane_equation(p1, p2, p3)

# ball_position_unwarped = projection.unwarp_point(ball_current_pose[0])
# print(ball_position_unwarped)
# p = util.get_xyz_from_neighbors(depth_intrinsics, depth_image, ball_position_unwarped)
# dist = util.distance_to_plane(p, plane)
# print(dist)