import cv2
from fuse_pointclouds import get_pointcloud_from_depth_image, align_pointclouds, \
    get_pointcloud_from_rgbd_image
from segmentation_matching_helpers import *
import open3d as o3d
import numpy as np
import random
import open3d as o3d


def homogenous_transform(R, t):
    homogeneous_matrix = np.eye(4, dtype=np.float64)
    homogeneous_matrix[0:3, 0:3] = R
    homogeneous_matrix[0:3, 3:4] = t

    return homogeneous_matrix


def inverse_transform(transform_matrix):
    rotation = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]

    inverse_rotation = np.transpose(rotation)  # Transpose of the rotation matrix
    inverse_translation = -inverse_rotation @ translation

    inverse_transform_matrix = np.identity(4)
    inverse_transform_matrix[:3, :3] = inverse_rotation
    inverse_transform_matrix[:3, 3] = inverse_translation

    return inverse_transform_matrix


if __name__ == "__main__":
    rotations = {"camera1": np.array([[0.40201686, 0.7393932, -0.54007419],  # (weiter unten)
                                      [-0.9156228, 0.32195245, -0.24079349],
                                      [-0.00416286, 0.59130729, 0.80643559]]),

                 "camera0": np.array([[0.92302632, 0.33048885, -0.19697598],
                                      [-0.38312593, 0.74275817, -0.54911276],
                                      [-0.03517012, 0.58231214, 0.81220418]])}

    translations = {"camera0": np.array([[0.17342514], [0.66141277], [-1.00268682]]),
                    "camera1": np.array([[0.58776885], [0.4904326], [-0.66038956]])}
    # camera intrinsics for ZED cam
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720, fx=946.026, fy=946.026, cx=652.250,
                                                      cy=351.917)
    fx = 946.026
    fy = 946.026
    cx = 652.250
    cy = 351.917

    max_depths = {"camera0": 3.1, "camera1": 8.251155445575714}
    min_depths = {"camera0": 0.5, "camera1": 0.5051345419883728}

    T_S_c1 = homogenous_transform(rotations["camera1"], translations["camera1"])
    T_S_c2 = homogenous_transform(rotations["camera0"], translations["camera0"])
    T_c2_S = inverse_transform(T_S_c2)
    T_c1_S = inverse_transform(T_S_c1)
    T_c1_c2 = T_c1_S @ T_S_c2
    T_c2_c1 = T_c2_S @ T_S_c1
    depth_image1 = cv2.imread("./images/depth_avg1.png", -1)[:, :, 0:3]  # read in as 3 channel
    depth_image2 = cv2.imread("./images/depth_avg2.png", -1)[:, :, 0:3]  # read in as 3 channel
    color_image1 = cv2.imread("./images/l_ws1.jpg")
    color_image2 = cv2.imread("./images/l_ws2.jpg")

    points1 = get_pointcloud_from_rgbd_image(color_image1, depth_image1, intrinsic=o3d_intrinsic,
                                             min_val=min_depths["camera0"],
                                             max_val=max_depths["camera0"], cutoff=5.0)
    points1 = np.asarray(points1.points)
    points1 = np.hstack([points1, np.ones((points1.shape[0], 1))])  # homogenize (will be nx4)

    c2_points1 = T_c2_c1 @ np.transpose(points1)  # 4x4 times 4xn
    u2 = (np.round(c2_points1[0, :] * fx / c2_points1[2, :] + cx)).astype(int)
    v2 = (np.round(c2_points1[1, :] * fy / c2_points1[2, :] + cy)).astype(int)  # both are row vectors n
    print("max in u is ", np.max(u2))
    print("max in v is ", np.max(v2))
    mask_u = np.asarray([u2 > 1280])
    mask_v = np.asarray([v2 > 720])
    total_mask = np.logical_or(mask_u, mask_v)
    total_mask.flatten()
    valid_u = np.delete(u2, total_mask)
    valid_v = np.delete(v2, total_mask)
    print("max in valid u is ", np.max(valid_u))
    print("max in valid v is ", np.max(valid_v))
