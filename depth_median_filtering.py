import cv2
import numpy as np

depth_image1 = cv2.imread("./images/depth_img1.png", -1)  # read in as 1 channel
depth_image2 = cv2.imread("./images/depth_img2.png", -1)  # read in as 1 channel

cv2.imshow('Original Image', (depth_image1))
filtered_depth1 = cv2.medianBlur((depth_image1/257).astype(np.uint8), 9)
cv2.imshow('Filtered Image', filtered_depth1)
cv2.waitKey(0)
