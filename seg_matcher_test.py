import numpy as np
import open3d.cpu.pybind.io
import open3d.cpu.pybind.t.geometry
import open3d.visualization
import cv2
from tryout import visualize_segmentation
from segmentation_matching_helpers import *
import pickle
import open3d.core as o3c
from segmentation_matcher import SegmentationParameters, SegmentationMatcher


segmentation_parameters=SegmentationParameters(736, conf=0.6, iou=0.9)
segmenter = SegmentationMatcher(segmentation_parameters, cutoff=2.0, model_path='FastSAM-x.pt')

mask_arrays = segmenter.segment_color_images(device="cpu")
segmenter.generate_pointclouds_from_masks()
global_pointclouds = segmenter.project_pointclouds_to_global()
correspondences, scores = segmenter.match_segmentations(voxel_size=0.05, threshold=0.0)
corresponding_pointclouds = segmenter.align_corresponding_objects(correspondences, scores)
