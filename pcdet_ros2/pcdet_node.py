"""! @brief Defines the PCDetROS Class.
The package subscribes to the pointcloud message and publishes instances of object detection.
"""
##
# @file pcdet_node.py
#
# @brief Defines the PCDetROS classes.
#
# @section description_pcdet_ros Description
# Defines the class to connect PCDet with the ROS 2 Environment.
# - PCDetROS
#
# @section libraries_pcdet_ros Libraries/Modules
# - ROS 2 Humble (https://docs.ros.org/en/humble/index.html)
# - OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
#
# @section author_pcdet_ROS Author(s)
# - Created by Shrijal Pradhan on 16/03/2023.
#
# Copyright (c) 2023 Shrijal Pradhan.  All rights reserved.
import sys
sys.path.append(f'/home/iat/anaconda3/envs/ros2/lib/python3.10/site-packages/') 

# Imports
import rclpy 
from rclpy.node import Node
import ros2_numpy
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2

import argparse
import numpy as np
from typing import List
import torch

from pyquaternion import Quaternion

from .config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from .pc_roi import create_bbox_marker

class PCDetROS(Node):
    """! The PCDetROS class.
    Defines the ROS 2 Wrapper class for PCDet.
    """
    def __init__(self):
        """! The PCDetROS class initializer.
        @param config_file Path to the configuration file for OpenPCDet.
        @param package_folder_path Path to the configuration folder, generally inside the ROS 2 Package.
        @param model_file Path to model used for Detection.
        @param allow_memory_fractioning Boolean to activate fraction CUDA Memory.
        @param allow_score_thresholding Boolean to activate score thresholding.
        @param num_features Number of features in each pointcloud data. 4 for Kitti. 5 for NuScenes
        @param device_id CUDA Device ID.
        @param device_memory_fraction Use only the input fraction of the allowed CUDA Memory.
        @param threshold_array Cutoff threshold array for detections. Even values for detection id, odd values for detection score. Sample: [0, 0.7, 1, 0.5, 2, 0.7].
        """
        super().__init__('pcdet')
        self.__initParams__()
        self.__initObjects__()
    
    def __cloudCB__(self, cloud_msg):
        out_msg = MarkerArray()
        cloud_array = ros2_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
        np_points = self.__convertCloudFormat__(cloud_array)

        #check if the roi is empty
        x_min, x_max = self.pcrange[0], self.pcrange[3]
        y_min, y_max = self.pcrange[1], self.pcrange[4]
        z_min, z_max = self.pcrange[2], self.pcrange[5]
                    
        region_points =np_points[(np_points[:, 0] >= x_min) & (np_points[:, 0] <= x_max) &
                            (np_points[:, 1] >= y_min) & (np_points[:, 1] <= y_max) &
                            (np_points[:, 2] >= z_min) & (np_points[:, 2] <= z_max)]
        if region_points.size == 0:
            print("roi have no points, skip")
            return

        import time
        start_time = time.time()
        scores, dt_box_lidar, types = self.__runTorch__(np_points)
        end_time = time.time()
        procesing_time = end_time - start_time
        print("Time taken: ", procesing_time, "seconds")

        print_str = f"Frame: {cloud_msg.header.frame_id}. Time: {cloud_msg.header.stamp}. Prediction results: \n"
        for i in range(len(types)):
            print_str += f"Type: {types[i]:d} Prob: {scores[i]:.2f}\n"
        print(print_str)

        current_marker_ids = set()

        if scores.size != 0:
            for i in range(scores.size):
                allow_publishing = self.__getPubState__(int(types[i]), scores[i])
                if allow_publishing:
                    marker = Marker()
                    marker.header.frame_id = cloud_msg.header.frame_id
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.id = i
                    marker.pose.position.x = float(dt_box_lidar[i][0])
                    marker.pose.position.y = float(dt_box_lidar[i][1])
                    marker.pose.position.z = float(dt_box_lidar[i][2])
                    # Use Quaternion message for orientation
                    quat = self.__yawToQuaternion__(float(dt_box_lidar[i][6]))
                    marker.pose.orientation.x = quat[1]
                    marker.pose.orientation.y = quat[2]
                    marker.pose.orientation.z = quat[3]
                    marker.pose.orientation.w = quat[0]
                    marker.scale.x = float(dt_box_lidar[i][3])
                    marker.scale.y = float(dt_box_lidar[i][4])
                    marker.scale.z = float(dt_box_lidar[i][5])
                    marker.color.a = 1.0
                    marker.color.r = 1.0  # Set color as needed
                    marker.color.g = 0.0
                    marker.color.b = 0.0

                    out_msg.markers.append(marker)
                    current_marker_ids.add(marker.id)
                    # Delete markers that are no longer valid
        for marker_id in self.active_marker_ids - current_marker_ids:
            marker = Marker()
            marker.header.frame_id = cloud_msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.CUBE
            marker.action = Marker.DELETE
            marker.id = marker_id
            out_msg.markers.append(marker)
        
        # Update the set of active marker IDs
        self.active_marker_ids = current_marker_ids

        if len(out_msg.markers) != 0:
            self.__pub_det__.publish(out_msg)
        else:
            # If no detections, publish an empty MarkerArray
            self.__pub_det__.publish(MarkerArray())

        #publish pc roi
        pcRangeX = [self.pcrange[0], self.pcrange[3]]
        pcRangeY = [self.pcrange[1], self.pcrange[4]]
        pcRangeZ = [self.pcrange[2], self.pcrange[5]]
        PCRange = [pcRangeX, pcRangeY, pcRangeZ]

        length = float(PCRange[0][1] - PCRange[0][0])
        width  = float(PCRange[1][1] - PCRange[1][0])
        height = float(PCRange[2][1] - PCRange[2][0])
        center_x = float(PCRange[0][1] + PCRange[0][0]) / 2
        center_y  = float(PCRange[1][1] + PCRange[1][0]) / 2
        center_z = float(PCRange[2][1] + PCRange[2][0]) / 2

        # Marker for Box
        markerBox = Marker()
        markerBox.header.frame_id = cloud_msg.header.frame_id
        markerBox.header.stamp = self.get_clock().now().to_msg()
        markerBox.type = Marker.CUBE
        markerBox.action = Marker.ADD
        markerBox.pose.position.x = center_x
        markerBox.pose.position.y = center_y
        markerBox.pose.position.z = center_z
        markerBox.pose.orientation.x = 0.0
        markerBox.pose.orientation.y = 0.0
        markerBox.pose.orientation.z = 0.0
        markerBox.pose.orientation.w = 1.0
        markerBox.scale.x = length
        markerBox.scale.y = width
        markerBox.scale.z = height
        markerBox.color.a = 0.1
        markerBox.color.r = 0.0
        markerBox.color.g = 1.0
        markerBox.color.b = 0.0
        print(markerBox.header.stamp)
        self.publisherRangeBox.publish(markerBox)

    def __convertCloudFormat__(self, cloud_array, remove_nans=True, dtype=float):
        '''
        '''
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]
        
        points = np.zeros(cloud_array.shape + (self.__num_features__,), dtype=dtype)
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']
        return points

    def __runTorch__(self, points):
        if len(points) == 0:
            return 0, 0, 0
        
        self.__points__ = points.reshape([-1, self.__num_features__])

        input_dict = {
            'points': self.__points__
        }
        with torch.no_grad():
            data_dict = self.__online_detection__.prepare_data(data_dict=input_dict)
            data_dict = self.__online_detection__.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            torch.cuda.synchronize()
            pred_dicts, _ = self.__net__.forward(data_dict)

            torch.cuda.synchronize()

            #pred = self.__removeLowScore__(pred_dicts[0])
            boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
            scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
            types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()

        return scores, boxes_lidar, types
    
    def __yawToQuaternion__(self, yaw: float) -> Quaternion:
        return Quaternion(axis=[0, 0, 1], radians=yaw)
    
    def __getPubState__(self, id, score) -> bool:
        if(not self.__allow_score_thresholding__):
           return True
        
        for i in range(len(self.__thr_arr__)):
            if(i + 1 == id):
                if(self.__thr_arr__[i] > score):
                    return False
                else:
                    return True
        
        return True

    def __readConfig__(self):
        cfg_from_yaml_file(self.__config_file__, cfg, self.__package_folder_path__)
        cfg.DATA_CONFIG._BASE_CONFIG_ = self.__package_folder_path__ + cfg.DATA_CONFIG._BASE_CONFIG_
        self.__online_detection__ = DatasetTemplate(dataset_cfg=cfg.DATA_CONFIG, 
                                                    class_names=cfg.CLASS_NAMES, 
                                                    training=False, 
                                                    root_path=self.__package_folder_path__,
                                                    logger=self.__logger__)
        
        torch.cuda.set_device(self.__device_id__)
        torch.backends.cudnn.benchmark = False
        self.__device__ = torch.device('cuda:'+ str(self.__device_id__) if torch.cuda.is_available() else "cpu")
        if(self.__allow_memory_fractioning__):
            torch.cuda.set_per_process_memory_fraction(self.__device_memory_fraction__, device=self.__device_id__)
        
        self.__net__ = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.__online_detection__)
        self.__net__.load_params_from_file(filename=self.__model_file__, logger=self.__logger__, to_cpu=True)
        self.__net__ = self.__net__.to(self.__device__).eval()
        self.pcrange = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    
    def __initParams__(self):
        self.declare_parameter("config_file", rclpy.Parameter.Type.STRING)
        self.declare_parameter("package_folder_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("model_file", rclpy.Parameter.Type.STRING)
        self.declare_parameter("allow_memory_fractioning", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("allow_score_thresholding", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("num_features", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("device_id", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("device_memory_fraction", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("threshold_array", rclpy.Parameter.Type.DOUBLE_ARRAY)

        self.__config_file__ = self.get_parameter("config_file").value
        self.__package_folder_path__ = self.get_parameter("package_folder_path").value
        self.__model_file__ = self.get_parameter("model_file").value
        self.__allow_memory_fractioning__ = self.get_parameter("allow_memory_fractioning").value
        self.__allow_score_thresholding__ = self.get_parameter("allow_score_thresholding").value
        self.__num_features__ = self.get_parameter("num_features").value
        self.__device_id__ = self.get_parameter("device_id").value
        self.__device_memory_fraction__ = self.get_parameter("device_memory_fraction").value
        self.__thr_arr__ = self.get_parameter("threshold_array").value

        self.__config_file__ = self.__package_folder_path__ + "/" + self.__config_file__
        self.__model_file__ = self.__package_folder_path__ + "/" + self.__model_file__
        self.__points__ = None
        self.__logger__ = common_utils.create_logger()
        self.__readConfig__()
    
    def __initObjects__(self):
        self.sub_cloud = self.create_subscription(PointCloud2, 
                                                  "input", 
                                                  self.__cloudCB__, 
                                                  10)
        self.__pub_det__ = self.create_publisher(MarkerArray,
                                                 "output",
                                                 1)
        self.publisherRangeBox = self.create_publisher(Marker,
                                                       "pc_range",
                                                       10)
        self.active_marker_ids = set()

def main(args=None):
    rclpy.init(args=args)
    pcdet_node = PCDetROS()
    rclpy.spin(pcdet_node)
    pcdet_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()