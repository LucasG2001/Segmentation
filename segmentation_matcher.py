import numpy as np
import open3d.cpu.pybind.geometry
import open3d.visualization
import cv2
from tryout import visualize_segmentation
from segmentation_matching_helpers import *
import pickle
import time


class SegmentationParameters:
    def __init__(self, image_size=736, conf=0.6, iou=0.9):
        self.image_size = image_size
        self.confidence = conf
        self.iou = iou

    def set_parameters(self, imgsz, conf, iou):
        self.image_size = imgsz
        self.confidence = conf
        self.iou = iou


class SegmentationMatcher:
    """
    Note that this Matcher works with tuples or lists of images, camera parameters and so on
    """

    def __init__(self, segmentation_parameters, cutoff, model_path='FastSAM-x.pt'):
        self.color_images = []
        self.depth_images = []
        self.intrinsics = []
        self.transforms = []
        self.mask_arrays = []
        self.seg_params = segmentation_parameters
        self.nn_model = FastSAM(model_path)
        self.max_depth = cutoff  # truncation depth
        self.pc_array_1 = []
        self.pc_array_2 = []

    def set_camera_params(self, intrinsics, transforms):
        self.intrinsics = intrinsics
        self.transforms = transforms

    def set_segmentation_model(self, model_path):
        self.nn_model = FastSAM(model_path)

    def set_images(self, color_images, depth_images):
        self.color_images = color_images
        self.depth_images = depth_images

    def preprocess_images(self):
        for i, (depth_image, color_image) in enumerate(zip(self.depth_images, self.color_images)):
            depth_mask = depth_image > self.max_depth * 10000
            depth_image[depth_mask] = 0  # should make depth image black at these points -> non-valid pointcloud
            # Set corresponding color image pixels to 0
            color_image[np.stack([depth_mask] * 3, axis=-1)] = 0
            cv2.imshow('color_img', self.color_images[i])
            cv2.waitKey(0)


    def set_segmentation_params(self, segmentation_params):
        self.seg_params = segmentation_params

    def segment_color_images(self, device="cpu", filter_masks=True):
        # ToDo: test if one can scam runtime of the model by combining the them at the same time
        DEVICE = device
        print("loaded NN model")
        color_images = self.color_images
        for image in color_images:
            everything_results = self.nn_model(image, device=DEVICE, retina_masks=True,
                                               imgsz=self.seg_params.image_size, conf=self.seg_params.confidence,
                                               iou=self.seg_params.iou)
            prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)
            # everything prompt
            # mask_array = prompt_process.everything_prompt()  # results.mask.data
            annotations = prompt_process._format_results(result=everything_results[0], filter=0)
            if filter_masks:
                annotations, _ = prompt_process.filter_masks(annotations)
            mask_array = [ann["segmentation"] for ann in annotations]  # is of type np_array

            self.mask_arrays.append(mask_array)

        return self.mask_arrays

    def generate_pointclouds_from_masks(self):
        postprocessing = 0
        pc_creating = 0
        rgbd_creating = 0
        point_cloud_arrays = [self.pc_array_1, self.pc_array_2]
        # ToDo: Find a way to generate way less pointclouds
        #  maybe by blacking the image where depth is no more interesting
        for i, (mask_array, depth_image, color_image, intrinsic) in enumerate(
                zip(self.mask_arrays, self.depth_images, self.color_images,
                    self.intrinsics)):
            for mask in mask_array:
                start_time = time.time()
                local_depth = o3d.geometry.Image(depth_image * mask)
                local_color = o3d.geometry.Image(color_image.astype(np.uint8))
                # Done: Check if creating pointcloud from depth image only is faster and feasible
                # It speeds up the function call by ca. 1s but makes it impossible to use colored ICP
                rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(local_color, local_depth,
                                                                              depth_scale=10000,
                                                                              depth_trunc=self.max_depth,
                                                                              convert_rgb_to_intensity=False)

                pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=intrinsic)
                rgbd_time = time.time() - start_time
                rgbd_creating += rgbd_time
                # pc = open3d.cpu.pybind.geometry.PointCloud.create_from_depth_image(depth=local_depth,
                # intrinsic=intrinsic,
                # depth_scale=10000,
                # depth_trunc=self.max_depth)

                pc_creation_time = time.time() - rgbd_time - start_time
                pc_creating += pc_creation_time

                # pc.paint_uniform_color(np.divide(colors_ocv1[i], 255))
                pc = pc.uniform_down_sample(every_k_points=4)
                # pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.99)
                # ToDO: PC filtering is very costly and make sup 60% of the function call.
                #  However, the quality of the matches is shit without it
                pc, _ = pc.remove_radius_outlier(nb_points=30, radius=0.05)
                pc_postprocess_time = time.time() - pc_creation_time - rgbd_time - start_time
                postprocessing += pc_postprocess_time

                # o3d.visualization.draw_geometries([pc], width=1280, height=720)
                if len(pc.points) > 100:  # delete all pointclouds with less than 40 points
                    point_cloud_arrays[i].append(pc)

        print("time for creating rgbd image is ", rgbd_creating)
        print("creating pointcloud took ", pc_creating)
        print("cleaning pointcloud took ", postprocessing)

    def project_pointclouds_to_global(self):
        for pc_array, transform in zip([self.pc_array_1, self.pc_array_2], self.transforms):
            for pc in pc_array:
                pc.transform(transform)

        return self.pc_array_1, self.pc_array_2

    def match_segmentations(self, voxel_size=0.05, threshold=0.0):
        correspondences, scores = match_segmentations_3d(self.pc_array_1, self.pc_array_2, voxel_size=voxel_size,
                                                         threshold=threshold)

        return correspondences, scores

    def align_corresponding_objects(self, correspondences, scores):
        # ToDo: (Here or in other function) -> take correspondences and create single pointcloud
        #  out of them for ROS publishing
        corresponding_pointclouds = []
        for pc_tuple, iou in zip(correspondences, scores):
            # align both pointclouds
            max_dist = 2 * np.linalg.norm(pc_tuple[0].get_center() - pc_tuple[1].get_center())
            # Estimate normals for both point clouds
            pc_tuple[0].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pc_tuple[1].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            try:
                # use default colored icp parameters
                icp = o3d.pipelines.registration.registration_colored_icp(pc_tuple[0], pc_tuple[1],
                                                                          max_correspondence_distance=max_dist)
                # transform point cloud 1 onto point cloud 2
                pc_tuple[0].transform(icp.transformation)
            except RuntimeError as e:
                # sometimes no correspondence is found. Then we simply overlay the untransformed point-clouds to avoid a
                # complete stop of the program
                print(f"Open3D Error: {e}")
                print("proceeding by overlaying point-clouds without transformation")

            corresponding_pointclouds.append(pc_tuple)
            o3d.visualization.draw_geometries(pc_tuple)

        return corresponding_pointclouds
