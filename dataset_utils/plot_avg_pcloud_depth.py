"""
Loads nuScenes dataset and plots the average depth of the Radar PointCloud
(distance from the radar sensor)
"""

import matplotlib
import os
#import os.path as osp
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import random
from scipy.sparse import csr_matrix
import errno
import shutil
import json
import argparse

from PIL import Image
from pyquaternion import Quaternion
from typing import Tuple, List, Dict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix


def load_keyframe_rad_tokens(nusc: NuScenes) -> (List[str]):
    """
    This method takes a Nuscenes instance and returns two lists with the
    sample_tokens of all RADAR_FRONT sample_data which
    have (almost) the same timestamp as their corresponding sample
    (is_key_frame = True).
    :param nusc: Nuscenes instance
    :return: rad_sd_tokens List of radar sample data tokens
    """


    rad_sd_tokens = []
    for scene_rec in nusc.scene:
        print('Loading samples of scene %s....' % scene_rec['name'], end = '')
        start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])

        rad_front_sd_rec = nusc.get('sample_data', start_sample_rec['data']['RADAR_FRONT'])

        cur_rad_front_sd_rec = rad_front_sd_rec
        rad_sd_tokens.append(cur_rad_front_sd_rec['token'])

        #Append all keyframe radar sample tokens in list
        while cur_rad_front_sd_rec['next'] != '':
            cur_rad_front_sd_rec = nusc.get('sample_data', cur_rad_front_sd_rec['next'])
            if cur_rad_front_sd_rec['is_key_frame']:
                rad_sd_tokens.append(cur_rad_front_sd_rec['token'])
        print("done!")

    return rad_sd_tokens

def tokens_to_data(nusc: NuScenes, rad_sd_tokens: List[str]) -> List:
    """
    This method takes a list with the Radar sample_data tokens and
    loads the actual data in a new list
    :param nusc: Nuscenes instance
    :param rad_sd_tokens: List with all the radar sample_data tokens
    :return: list((np.array) List of numpy array with the radar data
    """

    radar_pcl_list = []
    for i in range(len(rad_sd_tokens)):
        rad_sd_path = nusc.get_sample_data_path(rad_sd_tokens[i])
        if not os.path.isfile(rad_sd_path):
            continue
        radar_pcl = RadarPointCloud.from_file(rad_sd_path)
        #nuScenes RadarPointCloud has shape (18, num_points)
        #RADNET expects (num_points, 4)
        radar_pcl.points = radar_pcl.points.transpose()
        radar_pcl.points = radar_pcl.points[:, :3]
        radar_pcl.points = np.hstack((radar_pcl.points, np.ones((radar_pcl.points.shape[0], 1), dtype=radar_pcl.points.dtype)))
        radar_pcl_list.append(radar_pcl)

    return radar_pcl_list


def main():

    #Instantiate an object of the NuScenes dataset class
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/odysseas/thesis/data/sets/nuscenes_mini/', verbose=True)

    #Load front_rad sample_data tokens
    rad_sd_tokens = load_keyframe_rad_tokens(nusc)

    #Load actual sample_data
    radar_pointclouds = tokens_to_data(nusc, rad_sd_tokens)

    depths_list = []
    #Load x (front/depth) value of all radar data
    depths_sum = 0.0
    total_number_of_detections = 0
    for pcl in range(len(radar_pointclouds)):
        number_of_detections = len(radar_pointclouds[pcl].points)
        total_number_of_detections += number_of_detections
        for detection in range(len(radar_pointclouds[pcl].points)):
            #radar_pointclouds[pcl].points = np.delete(radar_pointclouds[pcl].points, np.argwhere(radar_pointclouds[pcl].points[0] > 100))
            depth = radar_pointclouds[pcl].points[detection][0]
            depths_list.append(depth)
            depths_sum += depth
    depths_avg = depths_sum / total_number_of_detections

    print('Total number of radar points detected across all scenes: {}'.format(total_number_of_detections))
    print('Average depth for all radar detections across all scenes: {}'.format(depths_avg))

    plt.hist(depths_list, bins = 100)
    plt.title("Histogram of radar depths")
    plt.ylabel("Number of detections")
    plt.xlabel("Depth")
    plt.xticks((np.arange(0, max(depths_list)+1, 20)))
    plt.savefig('Radar_Depths_Histogram.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
