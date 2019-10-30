"""
Loads nuScenes dataset and de-calibrates the radar_front to cam_front calibration data.
Creates a dataset with the following structure:
    1 npz file per sample containing:
        rgb_image: numpy array with shape (150, 240, 3)
        radar_detections: numpy array with shape (number_of_detections, 4) containing position vectors [x,y,z,1] of all detections in radar coordinate frame
        projections_groundtruth: radar detections projected into image using h_gt, stored as csr_matrix in ndarray
        projections_decalib: radar detections projected into image using h_gt and decalibration, stored as csr_matrix in ndarray
        K: intrinsic camera calibration matrix K (numpy array with shape (3,4))
        H_gt: homogeneous transformation matrix from radar to camera frame (numpy array with shape (4,4))
        decalib: inverted transformation between camera and radar rotation as quaternions (idx 0-3) and translation concatinated (4-6) (numpy array with shape (7,))
        rgb_image_orig_dim : ndarray (2,1) with height ([0]) and width ([1]) in pixel of original image (req. for transformation with P)
    dataset_info.txt:
        width and height of image / projection data
        minimum number of radar detections per sample
    debug_rgb_images:
        images of every sample with plotted radar projections
    debug_projections_groundtruth:
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
import create_delta_calib_list as decalibration #creating decalib
import DualQuaternion.transformations as quattransform
import DualQuaternion.DualQuaternion as dualquat

from PIL import Image
from pyquaternion import Quaternion
from typing import Tuple, List, Dict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix


#Config parameters
ROTATION_STD = 10. # Degrees
TRANSLATION_STD = 0 # meters
static_decalib = False

IMAGE_WIDTH = 240
IMAGE_HEIGHT = 150
debug_circle_size = 2

ORIGINAL_WIDTH = 1600
ORIGINAL_HEIGHT = 900
#Each image frame must contain at least 10 radar detections
#in order to be added in the dataset
REQUIRED_NUM_POINTS = 10
h_gt = None
counter_for_num_images = 0
counter_total_correspondences = 0

projections_decalibrated = []
projections_groundtruth = []



def load_keyframe_rad_cam_data(nusc: NuScenes) -> (List[str], List[str], List[str]):
    """
    This method takes a Nuscenes instance and returns two lists with the
    sample_tokens of all CAM_FRONT and RADAR_FRONT sample_data which
    have (almost) the same timestamp as their corresponding sample
    (is_key_frame = True). In addition, it returns the sample_names which are set
    equal to the filename of each CAM_FRONT sample_data.
    :param nusc: Nuscenes instance
    :return: (cam_sd_tokens, rad_sd_tokens, sample_names).
    Tuple with lists of camera and radar tokens as well as sample_names
    """

    #Lists to hold tokens of all cam and rad sample_data that have is_key_frame = True
    #These have (almost) the same timestamp as their corresponding sample and
    #correspond to the files in the ..sets/nuscenes/samples/ folder
    cam_sd_tokens = []
    rad_sd_tokens = []
    sample_names = []
    for scene_rec in nusc.scene:
        #scene_name = scene_rec["name"] + "_sample_"
        print('Loading samples of scene %s....' % scene_rec['name'], end = '')
        start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        #sample_name = scene_name + str(start_sample_rec["timestamp"])

        cam_front_sd_rec = nusc.get('sample_data', start_sample_rec['data']['CAM_FRONT'])
        rad_front_sd_rec = nusc.get('sample_data', start_sample_rec['data']['RADAR_FRONT'])

        cur_cam_front_sd_rec = cam_front_sd_rec
        cur_rad_front_sd_rec = rad_front_sd_rec
        sample_name = cur_cam_front_sd_rec["filename"].replace('samples/CAM_FRONT/', '').replace('.jpg', '')
        #Append the first sample_name, cam and rad sample_data tokens in lists
        sample_names.append(sample_name)
        cam_sd_tokens.append(cur_cam_front_sd_rec['token'])
        rad_sd_tokens.append(cur_rad_front_sd_rec['token'])

        #Append all keyframe sample_names and camera sample tokens in list
        while cur_cam_front_sd_rec['next'] != '':
            cur_cam_front_sd_rec = nusc.get('sample_data', cur_cam_front_sd_rec['next'])
            sample_name = cur_cam_front_sd_rec["filename"].replace('samples/CAM_FRONT/', '').replace('.jpg', '')
            if cur_cam_front_sd_rec['is_key_frame']:
                sample_names.append(sample_name)
                cam_sd_tokens.append(cur_cam_front_sd_rec['token'])

        #Append all keyframe radar sample tokens in list
        while cur_rad_front_sd_rec['next'] != '':
            cur_rad_front_sd_rec = nusc.get('sample_data', cur_rad_front_sd_rec['next'])
            if cur_rad_front_sd_rec['is_key_frame']:
                rad_sd_tokens.append(cur_rad_front_sd_rec['token'])
        print("done!")

    assert(len(cam_sd_tokens) == len(rad_sd_tokens) == len(sample_names))

    return (cam_sd_tokens, rad_sd_tokens, sample_names)

def tokens_to_data_pairs(nusc: NuScenes,
                         cam_sd_tokens: List[str],
                         rad_sd_tokens: List[str]) -> list(zip()):
    """
    This method takes a pair of lists with the Camera and Radar sample_data tokens,
    loads the actual data in two corresponding lists and returns the zipped lists
    :param nusc: Nuscenes instance
    :param cam_sd_tokens: List with all the camera sample_data tokens
    :param rad_sd_tokens: List with all the radar sample_data tokens
    :return: list(zip(np.array, np.array)) List of zipped array lists with the data
    """
    rgb_images_list = []
    for i in range(len(cam_sd_tokens)):
        cam_sd_path = nusc.get_sample_data_path(cam_sd_tokens[i])
        if not os.path.isfile(cam_sd_path):
            continue
        #im = Image.open(cam_sd_path)
        #im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
        img = cv2.imread(cam_sd_path)
        #Resize with Bilinear interpolation
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_images_list.append(img)

    radar_pcl_list = []
    for i in range(len(rad_sd_tokens)):
        rad_sd_path = nusc.get_sample_data_path(rad_sd_tokens[i])
        if not os.path.isfile(rad_sd_path):
            continue
        #radar_pcl = RadarPointCloud.from_file(rad_sd_path, invalid_states = range(18), dynprop_states = range(18), ambig_states = range(18))
        #radar_pcl = RadarPointCloud.from_file(rad_sd_path, invalid_states = None, dynprop_states = [0,2,6], ambig_states = None)
        radar_pcl = RadarPointCloud.from_file(rad_sd_path)
        #nuScenes RadarPointCloud has shape (18, num_points)
        #RADNET expects (num_points, 4)
        radar_pcl.points = radar_pcl.points.transpose()
        radar_pcl.points = radar_pcl.points[:, :3]
        radar_pcl.points = np.hstack((radar_pcl.points, np.ones((radar_pcl.points.shape[0], 1), dtype=radar_pcl.points.dtype)))
        radar_pcl_list.append(radar_pcl)

    assert(len(rgb_images_list) == len(radar_pcl_list))
    image_radar_pairs = list(zip(rgb_images_list, radar_pcl_list))

    del rgb_images_list
    del radar_pcl_list

    return image_radar_pairs

def get_rad_to_cam(nusc: NuScenes, cam_sd_token: str, rad_sd_token: str):
    """
    Method to get the extrinsic calibration matrix from radar_front to camera_front
    for a specific sample.
    Every sample_data has a record on which calibrated - sensor the
    data is collected from ("calibrated_sensor_token" key)
    The calibrated_sensor record consists of the definition of a
    particular sensor (lidar/radar/camera) as calibrated on a particular vehicle
    :param nusc: Nuscenes instance
    :param cam_sd_token : A token of a specific camera_front sample_data
    :param rad_sd_token : A token of a specific radar_front sample_data
    :return: rad_to_cam <np.float: 4, 4> Returns homogeneous transform matrix from radar to camera
    """
    cam_cs_token = nusc.get('sample_data', cam_sd_token)["calibrated_sensor_token"]
    cam_cs_rec = nusc.get('calibrated_sensor', cam_cs_token)

    rad_cs_token = nusc.get('sample_data', rad_sd_token)["calibrated_sensor_token"]
    rad_cs_rec = nusc.get('calibrated_sensor', rad_cs_token)

    #Based on how transforms are handled in nuScenes scripts like scripts/export_kitti.py
    rad_to_ego = transform_matrix(rad_cs_rec['translation'], Quaternion(rad_cs_rec['rotation']), inverse = False)
    ego_to_cam = transform_matrix(cam_cs_rec['translation'], Quaternion(cam_cs_rec['rotation']), inverse = True)
    rad_to_cam = np.dot(ego_to_cam, rad_to_ego)
    return rad_to_cam


def invert_homogeneous_matrix(mat):
    mat_inverse = np.zeros(mat.shape)
    mat_inverse[:3,:3] = np.linalg.inv(mat[:3,:3])
    mat_inverse[:3,3] = mat[:3,3] * -1.
    return mat_inverse

def comp_uv_depth(K, h_gt, decalib, point):
    '''
    Compute pixels coordinates and inverted radar depth.
    '''
    # Project on image plane with: z * (u, v, 1)^T = K * H * x
    tmp = np.matmul(K, decalib)
    tmp = np.matmul(tmp, h_gt)
    point = np.matmul(tmp, point.transpose())
    if point[2] != 0:
        # return np.array([int(point[0]/point[2]), int(point[1]/point[2]), 1./point[2]])
        #return [point[0]/point[2], point[1]/point[2], 1./point[2]]
        return [point[0] / point[2], point[1] / point[2], point[2]]

    else:
        return None

def valid_pixel_coordinates(u, v, IMAGE_HEIGHT, IMAGE_WIDTH):
    """
    Checks whether the provided pixel coordinates are valid.
    """
    return (u >= 0 and v >= 0 and v < IMAGE_HEIGHT and u < IMAGE_WIDTH)


def create_and_store_samples(image_radar_pairs: List,
                             sample_names: List,
                             rad_to_cam_calibration_matrices: List,
                             cam_intrinsics: List ):

    global counter_total_correspondences
    global counter_for_num_images
    #global received_rgb_images
    #global received_radar_frames

    print("Number of rgb radar pairs: " + str(len(image_radar_pairs)))

    cnt_images = 0

    if static_decalib == True:
        decalib = decalibration.create_decalib_transformation(1, ROTATION_STD, TRANSLATION_STD)
        print(decalib)

    for i in range(len(image_radar_pairs)):
    #for img, radar_pcl in image_radar_pairs:

        img, radar_pcl = image_radar_pairs[i]
        sample_name = sample_names[i]
        h_gt = rad_to_cam_calibration_matrices[i]
        K = cam_intrinsics[i]

        retry_counter = 0
        counter_points = 0 # count number of projected vehicles
        print("Processing image {}".format(cnt_images))
        while(counter_points < REQUIRED_NUM_POINTS):
            if(retry_counter > 5):
                break
            retry_counter += 1
            counter_points = 0 # count number of projected vehicles
            if static_decalib != True:
                decalib = decalibration.create_decalib_transformation(1, ROTATION_STD, TRANSLATION_STD)

            projection_groundtruth = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH])
            projection_decalibrated = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH])

            img_copy_debug = img.copy()
            # store radar detections as vectors in list
            radar_detections = list()
            # Project radar detections in matrices.
            for radar_detection in radar_pcl.points:
                radar_detections.append(radar_detection)
                u, v, depth = comp_uv_depth(K, h_gt, decalib, radar_detection)
                u_true, v_true, depth_true = comp_uv_depth(K, h_gt, np.identity(4), radar_detection )

                # Convert to pixel coordinates, write in the matrix and draw dots on debug image.
                if (u, v, depth, u_true, v_true, depth_true) != None:
                    v /= (ORIGINAL_HEIGHT/IMAGE_HEIGHT)
                    u /= (ORIGINAL_WIDTH/IMAGE_WIDTH)
                    v_true /= (ORIGINAL_HEIGHT/IMAGE_HEIGHT)
                    u_true /= (ORIGINAL_WIDTH/IMAGE_WIDTH)
                    u, v, v_true, u_true = int(u), int(v), int(v_true), int(u_true)
                    if valid_pixel_coordinates(u, v, IMAGE_HEIGHT, IMAGE_WIDTH):
                        projection_decalibrated[v][u] = depth
                        cv2.circle(img_copy_debug,(u,v), debug_circle_size, (255,0,0), -1) # Red
                        counter_points += 1
                    if valid_pixel_coordinates(u_true, v_true, IMAGE_HEIGHT, IMAGE_WIDTH):
                        projection_groundtruth[v_true][u_true] = depth_true
                        cv2.circle(img_copy_debug,(u_true,v_true), debug_circle_size, (0,0,255), -1) # Blue

        # Store data only if there are more projections_groundtruth than the required amount
        if(counter_points < REQUIRED_NUM_POINTS):
            cnt_images += 1
            continue

        # num_detections_lower_image_half = np.sum((projection_groundtruth > 0.0).astype(int)[len(projection_groundtruth)/3:])
        # if num_detections_lower_image_half < 3:
        #     break
        counter_total_correspondences += len(radar_pcl.points)
        global counter_for_num_images
        # sample_name = str(counter_for_num_images).zfill(6)+"_"+measurement_name+"_"+str(rgb_timestamps[cnt_images])
        #sample_name = measurement_name+"_"+str(rgb_timestamps[cnt_images])

        # Save debug images.
        cv2.cvtColor(img_copy_debug, cv2.COLOR_RGB2BGR, img_copy_debug)
        cv2.imwrite(debug_images_path + sample_name + ".jpg", img_copy_debug)
        projection_decalib_debug = (projection_decalibrated > 0) * 255.
        cv2.imwrite(projections_decalib_debug_path + sample_name + ".jpg", projection_decalib_debug)
        projection_groundtruth_debug = (projection_groundtruth > 0) * 255.
        cv2.imwrite(projections_groundtruth_debug_path + sample_name + ".jpg", projection_groundtruth_debug)
        counter_for_num_images += 1


        decalib_inverted = invert_homogeneous_matrix(decalib) # Invert for training label.
        rotation = decalib_inverted[:3, :3]
        translation = decalib_inverted[:3, 3]
        quat = quattransform.quaternion_from_matrix(rotation)
        decalib_quat = np.concatenate((quat, translation))

        # Store sample
        store_sample(sample_name, img, radar_detections, csr_matrix(projection_groundtruth), csr_matrix(projection_decalibrated), decalib_quat, h_gt, K)

        cnt_images += 1


def store_sample(sample_name, image, radar_detections, projection, projection_decalib, decalib, h_gt, K):
    try:
        print(" *** Trying to convert to numpy *** ")
        image = np.array(image)
        projection = np.array(projection)
        radar_detections = np.array(radar_detections)
        rgb_image_orig_dim = np.array([ORIGINAL_HEIGHT, ORIGINAL_WIDTH])
        print(" *** Conversion completed *** ")
        try:
            print(" *** Trying to store results *** ")
            np.savez_compressed(data_samples_path + str(sample_name), rgb_image=image, radar_detections=radar_detections, projections_groundtruth=projection, \
                projections_decalib=projection_decalib, K=K, H_gt=h_gt, decalib=decalib, rgb_image_orig_dim=rgb_image_orig_dim)
            print(" *** Completed *** ")
        except Exception as e:
            print(" *** ERROR WHILE STORING ***: {}".format(str(e)))
    except Exception as e:
        print(" *** CONVERSION ERROR ***: {}".format(str(e)))



def store_data(image_radar_pairs: List,
               sample_names: List,
               rad_to_cam_calibration_matrices: List,
               cam_intrinsics: List ):

    print(" *** Creating samples ***")
    create_and_store_samples(image_radar_pairs, sample_names, rad_to_cam_calibration_matrices, cam_intrinsics)

    if counter_for_num_images != 0:
        try:
            print(" *** Trying to store dataset parameter *** ")
            # Save parameters.
            with open(output_path + "/" +'dataset_info.txt', 'w') as params_file:
                params_file.write('rotation_std: {}\n'.format(ROTATION_STD))
                params_file.write('translation_std: {}\n'.format(TRANSLATION_STD))
                params_file.write('image_width: {}\n'.format(IMAGE_WIDTH))
                params_file.write('image_height: {}\n'.format(IMAGE_HEIGHT))
                params_file.write('average number of correspondences: {}\n'.format(counter_total_correspondences/counter_for_num_images))
                params_file.write('minimum number of radar projections per sample: {}\n'.format(REQUIRED_NUM_POINTS))
            print(" *** Completed *** ")
        except Exception as e:
            print(" *** ERROR WHILE STORING ***: {}".format(str(e)))
    else:
        print("No samples generated.")

def main():
    # Read input parameters
    parser = argparse.ArgumentParser(description='Load nuScenes dataset, decalibrate radar - camera calibration and store samples in RADNET format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out_dir', default='/home/jupyter/thesis/data/sets/nuscenes_RADNET/nuscenes_08_RADNET', type=str, help='Output folder')
    parser.add_argument('--static_decalib', default = False, type = bool, help='Option for static decalibration between all samples')

    args = parser.parse_args()
    global static_decalib
    static_decalib = args.static_decalib

    #Create output directory and subdirectories
    global output_path
    output_path = args.out_dir
    global debug_images_path
    debug_images_path = output_path + "/debug_rgb_images/" # inside output folder we will a sub-folder called debug images
    global projections_decalib_debug_path
    projections_decalib_debug_path = output_path + "/debug_projections_decalibrated/"
    global projections_groundtruth_debug_path
    projections_groundtruth_debug_path = output_path + "/debug_projections_groundtruth/"
    global data_samples_path
    data_samples_path = output_path + "/samples/"

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except OSError as exc:
            print(exc)
            if exc.errno != errno.EEXIST:
                raise

    # Create debug_images directory and overwrite if it exists
    if os.path.exists(debug_images_path):
        shutil.rmtree(debug_images_path) # Remove all subdirectories.
    try:
        os.makedirs(debug_images_path)
    except OSError as exc:
        print(exc)
        if exc.errno != errno.EEXIST:
            raise

    # Create projections_groundtruth directory and overwrite if it exists
    if os.path.exists(projections_groundtruth_debug_path):
        shutil.rmtree(projections_groundtruth_debug_path) # Remove all subdirectories.
    try:
        os.makedirs(projections_groundtruth_debug_path)
    except OSError as exc:
        print(exc)
        if exc.errno != errno.EEXIST:
            raise

    # Create projections_decalibrated directory and overwrite if it exists
    if os.path.exists(projections_decalib_debug_path):
        shutil.rmtree(projections_decalib_debug_path) # Remove all subdirectories.
    try:
        os.makedirs(projections_decalib_debug_path)
    except OSError as exc:
        print(exc)
        if exc.errno != errno.EEXIST:
            raise

    # Create data samples directory and overwrite if it exists
    if os.path.exists(data_samples_path):
        shutil.rmtree(data_samples_path) # Remove all subdirectories.
    try:
        os.makedirs(data_samples_path)
    except OSError as exc:
        print(exc)
        if exc.errno != errno.EEXIST:
            raise

    #Instantiate an object of the NuScenes dataset class
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/jupyter/thesis/data/sets/nuscenes/', verbose=True)

    #Load front_cam and front_rad sample_data info in respective lists
    cam_sd_tokens, rad_sd_tokens, sample_names = load_keyframe_rad_cam_data(nusc)

    #Load actual sample_data from files in a list of zipped lists
    image_radar_pairs = tokens_to_data_pairs(nusc, cam_sd_tokens, rad_sd_tokens)

    #Load H_gt and K matrices for each sample_data record
    rad_to_cam_calibration_matrices = []
    cam_intrinsics = []
    #print(len(image_radar_pairs))
    # scale factor for camera intrinsics accordingly to the resize we perform on the images
    scale_factor = (IMAGE_HEIGHT / ORIGINAL_HEIGHT )
    for i in range(len(image_radar_pairs)):
        cam_cs_token = nusc.get('sample_data', cam_sd_tokens[i])["calibrated_sensor_token"]
        cam_cs_rec = nusc.get('calibrated_sensor', cam_cs_token)
        K = np.array(cam_cs_rec["camera_intrinsic"])
        # K_scaled = K
        # K_scaled[0][0] *= scale_factor
        # K_scaled[0][2] *= scale_factor
        # K_scaled[1][1] *= scale_factor
        # K_scaled[1][2] *= scale_factor
        #nuscenes K is 3x3 and we augment it to 3x4 with an extra zero column
        #since it will be used for mult witl 4x4 H_gt matrix
        K = np.hstack((K, np.zeros((K.shape[0], 1), dtype=K.dtype)))
        cam_intrinsics.append(K)
        H_gt = get_rad_to_cam(nusc, cam_sd_tokens[i], rad_sd_tokens[i])
        rad_to_cam_calibration_matrices.append(H_gt)

    store_data(image_radar_pairs, sample_names, rad_to_cam_calibration_matrices, cam_intrinsics)

if __name__ == '__main__':
    main()
