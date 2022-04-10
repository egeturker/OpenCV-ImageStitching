import cv2 as cv
import numpy as np
import math
import sys


def sift(img):
    # for non rgb images
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    # key points and sift descriptor
    kp, des = sift.detectAndCompute(gray_img, None)

    return kp, des


def pixel_descriptor(local_feature, img):
    # half of the diameter of local feature
    half_length = local_feature.size / 2

    # x, y coordinates of the center
    center_x, center_y = int(local_feature.pt[0]), int(local_feature.pt[1])

    angle = local_feature.angle

    # corners of the square by the clockwise order, starting from left top
    # 0 -> left top 1 -> right top 2 -> right bottom 3-> left bottom
    corners_x = list()
    corners_x.append(center_x - half_length)
    corners_x.append(center_x + half_length)
    corners_x.append(center_x + half_length)
    corners_x.append(center_x - half_length)

    corners_y = list()
    corners_y.append(center_y + half_length)
    corners_y.append(center_y + half_length)
    corners_y.append(center_y - half_length)
    corners_y.append(center_y - half_length)


    # angle rotation of corners
    angle_rad = math.radians(360 - angle)
    corners_x_rotated = list()
    corners_y_rotated = list()

    for i in range(len(corners_x)):
        corners_x_rotated.append((corners_x[i] - center_x) * math.cos(angle_rad)
                                 - (corners_y[i] - center_y) * math.sin(angle_rad) + center_x )
    for i in range(len(corners_y)):
        corners_y_rotated.append((corners_x[i] - center_x) * math.sin(angle_rad)
                                 + (corners_y[i] - center_y) * math.cos(angle_rad) + center_y )

    # get points inside the square
    mask = np.zeros(img.shape, dtype=np.uint8)

    pts = np.array([[[corners_x_rotated[0], corners_y_rotated[0]],
                     [corners_x_rotated[1], corners_y_rotated[1]],
                     [corners_x_rotated[2], corners_y_rotated[2]],
                     [corners_x_rotated[3], corners_y_rotated[3]]]], dtype=np.int32)
    cv.fillPoly(mask, pts, (255, 255, 255))
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    values = np.argwhere(mask == 255)

    h = np.full(256, 0)
    for i in range(len(values)):
        x = values[i][0]
        y = values[i][1]
        pixel_value = img[x][y]
        h[pixel_value] += 1
    return h


def feature_match_np(sift_des1, sift_des2):
    matches = np.zeros(shape=[len(sift_des1), 2], dtype=np.uint32)

    for i in range(len(sift_des1)):
        euclidian_dist = np.zeros(shape=[len(sift_des2), 2])
        for j in range(len(sift_des2)):
            euclidian_dist[j] = [j, np.linalg.norm(sift_des1[i] - sift_des2[j])]
        euclidian_dist = euclidian_dist[np.argsort(euclidian_dist[:, 1])]
        if euclidian_dist[0][1] < euclidian_dist[1][1] * 0.7  and\
                not any(euclidian_dist[0][0] == match[1] for match in matches):
            matches[i] = [i, euclidian_dist[0][0]]
    matches = matches[~(matches == 0).all(1)]

    return matches


def stitch(img1, img2):

    kp1, sift_des1 = sift(img1)
    kp2, sift_des2 = sift(img2)

    matches = feature_match_np(sift_des1, sift_des2)

    src_pts = list()
    dest_pts = list()
    for i in range(len(matches)):
        src_pts.append([int(kp1[matches[i][0]].pt[0]), int(kp1[matches[i][0]].pt[1])])
        dest_pts.append([int(kp2[matches[i][1]].pt[0]), int(kp2[matches[i][1]].pt[1])])
    src_pts = np.array(src_pts)
    dest_pts = np.array(dest_pts)

    h, status = cv.findHomography(src_pts, dest_pts, cv.RANSAC, ransacReprojThreshold=5.0)

    dst = cv.warpPerspective(img1, h, (img1.shape[1] + img2.shape[1], img1.shape[0]))  # warped image
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2

    return dst

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=sys.maxsize)

    right_img = cv.imread(sys.argv[1])
    left_img = cv.imread(sys.argv[2])

    img_final = stitch(right_img, left_img)
    cv.imwrite('final.png', img_final)



