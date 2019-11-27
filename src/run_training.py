# Software versions needed: Keras 2.1, Tensorflow 1.13, Cuda 10.0, Cudnn 6.0

# Fixed seeding for repeatability and finding good starting points.
# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

import math
import os, sys, subprocess, glob
import datetime
from time import time
import logging
import argparse
import shutil
import json

import numpy as np
np.set_printoptions(suppress=True) # No scientific number notation.
from scipy import sparse

from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model
from keras import metrics
from keras import losses
import keras.backend as K
from keras.layers.advanced_activations import PReLU
from keras import activations
from keras.utils.generic_utils import CustomObjectScope

from util.tee import Tee
from util.data_generator import DataGenerator
import util.metrics as met
import util.data_wrangling as dw
import util.dataloading as dl
import util.quaternion_ops as quatops
import util.dual_quaternion.transformations as quattransform
import util.plotting as plotting
import util.create_decalibration as dec
from util.radar_reprojection_manager import RadarReprojectionManager
from util.radar_reprojection_manager_batch import RadarBatchReprojectionManager
import paper_visualizations as pv
import custom_loss_functions as loss_fn
import pandas as pd

from rad_net import RadNet
from run_config import RunConfig


os.environ['CUDA_VISIBLE_DEVICES'] = ''
run_config = RunConfig()
experiments_path = "../experiments/"
model_1_name = "model_1"
model_2_name = "model_2"



def save_minimum_loss_and_metrics(history, output_folder):
    min_value_file = open(output_folder + 'loss_metrics_minima.txt', 'w')
    for key in sorted(history.keys()):
        min_value = np.min(np.array(history[key]))
        if not 'val' in key:
            min_value_file.write('train_' + key + ": " + str(min_value) + '\n')
        else:
            min_value_file.write(key + ": " + str(min_value) + '\n')
    min_value_file.close()

def compute_example_predictions(model, sample_file_names, num_prints):
    np.set_printoptions(suppress=True)
    for i in range(num_prints):
        [input_1, input_2, input_3, input_4], label = dl.load_radnet_training_sample_with_intrinsics_gt_decalib(str(sample_file_names[i]))
        # Expand dimensions to account for expected batch dimension
        input_1 = np.expand_dims(input_1, axis=0)
        input_2 = np.expand_dims(input_2, axis=0)
        input_3 = np.expand_dims(input_3, axis=0)
        input_4 = np.expand_dims(input_4, axis=0)
        print("Label: " + str(label[:4]))
        output = model.predict([input_1, input_2, input_3, input_4])
        # Normalize quaternions.
        quats = output[0]
        #quats = output[:,:4]
        inv_mags = 1. / np.sqrt(np.sum(np.square(quats), axis=1))
        quats_normalized = np.transpose(np.transpose(quats) * inv_mags)
        # print("Output unnormalized: " + str(quats[0]))
        print("Output: " + str(quats_normalized[0]))

def create_callbacks(model_name):
    callbacks = []
    # checkpoint = ModelCheckpoint(experiments_path + model_name + '.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1) # Saves best model.
    # callbacks.append(checkpoint)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0, verbose=1, mode='min', min_delta=1e-5)
    callbacks.append(reduce_lr)
    earls_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='min', restore_best_weights=True)
    callbacks.append(earls_stopping)
    return callbacks

def get_metrics():
    metrics = []
    # Separate losses.
    # metrics.append(rot_loss)
    # metrics.append(trans_loss)
    # Metrics.
    metrics.append(met.rot_angle_error)
    metrics.append(met.tilt_error)
    metrics.append(met.pan_error)
    metrics.append(met.roll_error)
    # metrics.append(trans_error)
    # metrics.append(trans_error_x)
    # metrics.append(trans_error_y)
    # metrics.append(trans_error_z)
    return metrics

def create_model():
    init_model_start = time()
    network_model = RadNet(run_config.input_shape, drop_rate=run_config.drop_rate, l2_reg=run_config.l2_reg)
    print("Time to setup model: " + str(time() - init_model_start) + "s")
    visualize_model(network_model.model)
    return network_model.model

def train_model(samples_list_train, samples_list_val, model_name):
    model = create_model()
    # Create generators.

    if model_name == model_2_name:
        params_train = {'batch_size': run_config.batch_size,
            'dim': (150, 240),
            'shuffle': True,
            'radar_input_factor': run_config.radar_input_factor,
            'path_augmented_data': str(experiments_path + model_1_name + "/aug_samples_train/")}
        params_val = {'batch_size': run_config.batch_size,
            'dim': (150, 240),
            'shuffle': True,
            'radar_input_factor': run_config.radar_input_factor,
            'path_augmented_data': str(experiments_path + model_1_name + "/aug_samples_val/")}
    else:
        params_train = {'batch_size': run_config.batch_size,
            'dim': (150, 240),
            'radar_input_factor': run_config.radar_input_factor,
            'shuffle': True}
        params_val = params_train

    training_generator = DataGenerator(samples_list_train, **params_train)
    validation_generator = DataGenerator(samples_list_val, **params_val)

    # Train model.
    optimizer = optimizers.Adam(lr=run_config.lr)
    #model.compile(loss=keras.losses.mean_squared_error, optimizer=optimizer, metrics=get_metrics())
    # Use a specific loss and metric for each specific output,
    # based on the name of the output Layer
    losses_dic = {
        'quat': loss_fn.keras_weighted_quaternion_translation_loss(run_config.length_error_weight),
        'cloud': loss_fn.keras_photometric_and_3d_pointcloud_loss(model.input[1], model.input[2],
            model.output[1], model.output[2], run_config.photometric_loss_factor, run_config.point_cloud_loss_factor)
    }
    loss_weights_dict = {
        'quat': 0.0, #100
        'cloud': 1.0
    }
    metrics_dict = {
        'quat': get_metrics()
    }
    model.compile(loss=losses_dic, loss_weights=loss_weights_dict, optimizer=optimizer, metrics=metrics_dict)
    callback_list = create_callbacks(model_name)
    history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                  epochs=run_config.epochs, callbacks=callback_list, use_multiprocessing=True, workers=6, verbose =2)
    # Generate training visualizations.
    model_output_folder = experiments_path + model_name + '/'
    model.save(experiments_path + model_name + '.h5')
    plotting.plot_history(history.history, model_output_folder)
    save_minimum_loss_and_metrics(history.history, model_output_folder)
    # Load best weights and return model.
    model.load_weights(experiments_path + model_name + '.h5') # Load best weights into model.
    return model


def visualize_corrected_projections(rgb_images, proj_gt, proj_init, proj_model_1, proj_model_2):
    plotting.visualize_corrected_projection(rgb_images, proj_gt, proj_init, os.path.join(experiments_path, 'projections_init'))
    plotting.visualize_corrected_projection(rgb_images, proj_gt, proj_model_1, os.path.join(experiments_path, 'projections_model_1'))
    plotting.visualize_corrected_projection(rgb_images, proj_gt, proj_model_2, os.path.join(experiments_path, 'projections_model_2'))

def start_training(samples_list):
    global run_config
    # Shuffle before splitting.
    samples_list = shuffle(samples_list)
    samples_list_train, samples_list_val = dw.split_validation(samples_list, run_config.val_split)
    print("Data split into {} training and {} validation samples.".format(str(len(samples_list_train)), str(len(samples_list_val))))

    # Instantiate reprojection manager.
    reprojection_manager = RadarReprojectionManager(run_config.original_resolution, run_config.input_shape, np.identity(4), np.identity(4))
    reprojection_manager_batch = RadarBatchReprojectionManager(run_config.original_resolution, run_config.input_shape)

    # Train model 1.
    print("Starting training of {}".format(model_1_name))
    model_1 = train_model(samples_list_train, samples_list_val, model_1_name)
    compute_example_predictions(model_1, samples_list_val, 10)

    # Load validation set
    print("Loading validation data")
    images_val, projections_decalibrated_val, projections_groundtruth_val, decalibs, H_gts, Ks, radar_detections, dims = dl.load_dataset(samples_list_val)

    # Augment data with model_1
    print("Reprojecting corrections of {}...".format(model_1_name))
    start_time = time()
    # Generate new training data for model_2
    output_path_train = experiments_path + model_1_name + "/aug_samples_train/"
    output_path_val = experiments_path + model_1_name + "/aug_samples_val/"
    print("Reprojecting and storing training data")
    reprojection_manager.compute_and_save_corrected_projections_labels(samples_list_train, model_1, output_path_train)
    print("Reprojecting and storing validation data")
    #print(K.int_shape(decalibs))
    #print(K.int_shape(Ks))
    trans_labels = decalibs[:, 4:]
    #k_mats = Ks[:, :, :3]
    #print(K.int_shape(trans_labels))
    #print(K.int_shape(k_mats))
    projections_val_1, labels_val_1 = reprojection_manager_batch.compute_projections_and_labels([images_val, projections_decalibrated_val, Ks, trans_labels], decalibs, radar_detections, H_gts, Ks, model_1)
    # save augmented validation samples
    if not os.path.exists(output_path_val):
        os.makedirs(output_path_val)

    for i, file_path in enumerate(samples_list_val):
        dl.save_augmented_projection_sample(output_path_val + str(file_path).split("/")[-1], projections_val_1[i,], labels_val_1[i,])

    print("Reprojection time: " + str(time() - start_time))

    # Reset noise parameter.
    #run_config = RunConfig()

    # Train model 2.
    print("Starting training of {}".format(model_2_name))
    model_2 = train_model(samples_list_train, samples_list_val, model_2_name)

    # Augment data with model_2
    print("Reprojecting corrections of {}...".format(model_2_name))
    start_time = time()
    print("Reprojecting validation data")
    # models do not predict trans so labels_val_1 contain only quat label
    # use original trans decalibs as labels
    trans_labels_val_1 = trans_labels #labels_val_1[:, 4:]
    projections_val_2, labels_val_2 = reprojection_manager_batch.compute_projections_and_labels([images_val, projections_val_1, Ks, trans_labels_val_1], labels_val_1, radar_detections, H_gts, Ks, model_2)
    print("Reprojection time: " + str(time() - start_time))

    # Delete unused variables / data
    del radar_detections
    del H_gts
    del Ks
    start_time = time()
    # Delete augmented samples from training model 1
    try:
        shutil.rmtree(output_path_train)
        shutil.rmtree(output_path_val)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    # Visualize results of first model.
    print("Visualizing results...")
    #Creating all the images is time-consuming so in case the val samples are many,
    #create visualization results for the first 100 only
    if (len(images_val)>100):
        visualize_corrected_projections(images_val[:100], projections_groundtruth_val[:100], projections_decalibrated_val[:100], projections_val_1[:100], projections_val_2[:100])
    else:
        visualize_corrected_projections(images_val, projections_groundtruth_val, projections_decalibrated_val, projections_val_1, projections_val_2)
    print("Time to visualize corrected projections: " + str(time() - start_time))
    # Create paper plots
    start_time = time()
    print("Create paper plots...")
    pv.create_paper_plots(experiments_path, projections_decalibrated_val, labels_val_2, decalibs)
    print("Time to make paper plots: " + str(time() - start_time))


def visualize_model(model):
    plot_model(model, to_file=experiments_path + 'model.png', show_shapes=True, show_layer_names=True)
    print(model.summary())

def parse_commandline():
    parser = argparse.ArgumentParser(description='Run training of the radar calibration network.')
    parser.add_argument('--run-name', '-n', type=str, help='Name of the experiment to be saved for this run.')
    parser.add_argument('--info', '-i', type=str, help='Experiment information text, will be witten to an info.txt file.')
    parser.add_argument('--gpu', type=str, default='0', help='ID of the gpu to run the training on. Default: 0')
    parser.add_argument('--weights-path', type=str, help='Path to model weights in case of inference and not training.')
    parser.add_argument('--static-decalib', type=bool, default=False, help='Boolean enabling evaluation of dataset with static decalibration.')
    parser.add_argument('--static-analysis', type=bool, default=False, help='Boolean enabling automatic generation and evaluation of datasets with static decalibration.')
    args = parser.parse_args()
    return args

def copy_code(output_path):
    """
    Creates a copy of the code used to run in the experiments directory.
    """
    # Create backup directory.
    code_backup_path = experiments_path + 'code'
    if not os.path.exists(code_backup_path): # Create code directory.
        os.makedirs(code_backup_path)
    sources_path = os.path.dirname(sys.argv[0])
    sources_path = sources_path if sources_path != '' else './'
    # Copy over source code and create subfolders if necessary.
    for root, dirs, files in os.walk(sources_path):
        cropped_root = root[2:] if (root[:2] == './') else root # Strip leading local directory.
        dest_dir = os.path.join(code_backup_path, cropped_root)
        # Create target folder.
        if not '__pycache__' in dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # Copy files.
        for filename in filter(lambda x: '.py' in x and not '.pyc' in x, files):
            source_file_path = os.path.join(root, filename)
            dest_file_path = os.path.join(dest_dir, filename)
            shutil.copyfile(source_file_path, dest_file_path)

def retrieve_git_hash():
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return git_hash
    except subprocess.CalledProcessError as e:
        print(e.output)
    return False

def save_run_params_in_file():
    with open(experiments_path + 'run_params.conf', 'w') as run_param_file:
        for attr, value in sorted(run_config.__dict__.items()):
                run_param_file.write(attr + ': ' +  str(value) + '\n')

def load_models(models_path):
    model_1 = load_model(os.path.join(models_path, model_1_name + '.h5'), compile=False)
    model_2 = load_model(os.path.join(models_path, model_2_name + '.h5'), compile=False)
    # model_1 = RadNet(run_config.input_shape)
    # model_2 = RadNet(run_config.input_shape)
    # model_1.model.load_weights(os.path.join(models_path, model_1_name + '.h5'))
    # model_2.model.load_weights(os.path.join(models_path, model_2_name + '.h5'))
    # return model_1.model, model_2.model
    return model_1, model_2

def cross_evaluate_models(models_path, samples_list, static_decalib=False):
    # Initialize models.
    start_time = time()
    print("Loading model weights...")
    model_1, model_2 = load_models(models_path)
    # Load data.
    print("Loading data...")
    images, projections_decalib, projections_groundtruth, decalibs, H_gts, Ks, radar_detections, dims = dl.load_dataset(samples_list)
    # Instantiate reprojection manager.
    reprojection_manager = RadarReprojectionManager(run_config.original_resolution, run_config.input_shape, np.identity(4), np.identity(4))
    reprojection_manager_batch = RadarBatchReprojectionManager(run_config.original_resolution, run_config.input_shape)
    trans_labels = decalibs[:, 4:]
    #k_mats = Ks[:, :, :3]
    # Apply model.
    print("Applying model...")
    if static_decalib == True:
        print("Static decalibration assumed. Samples get corrected with last estimate before model prediction.")
        projections_1, labels_1 = reprojection_manager.compute_projections_and_labels_static_decalib([images, projections_decalib, Ks, trans_labels], decalibs, radar_detections, H_gts, Ks, dims, model_1)
        trans_labels_1 = trans_labels #labels_1[:, 4:]
        projections_2, labels_2 = reprojection_manager.compute_projections_and_labels_static_decalib([images, projections_1, Ks, trans_labels_1], labels_1, radar_detections, H_gts, Ks, dims, model_2)
    else:
        projections_1, labels_1 = reprojection_manager_batch.compute_projections_and_labels([images, projections_decalib, Ks, trans_labels], decalibs, radar_detections, H_gts, Ks, model_1)
        trans_labels_1 = labels_1[:, 4:]
        projections_2, labels_2 = reprojection_manager_batch.compute_projections_and_labels([images, projections_1, Ks, trans_labels_1], labels_1, radar_detections, H_gts, Ks, model_2)

    print("Reprojection time: " + str(time() - start_time))
    # Print errors.
    print("Cross Evaluation:")
    start_time = time()
    print("Initial:")
    pv.print_angular_errors(decalibs)
    print("Model 1:")
    pv.print_angular_errors(labels_1)
    print("Model 2:")
    pv.print_angular_errors(labels_2)
    print("Generating corrected projections...")
    pv.create_paper_plots(experiments_path, projections_decalib, labels_2, decalibs)
    visualize_corrected_projections(images, projections_groundtruth, projections_decalib, projections_1, projections_2)
    print("Plotting time: " + str(time() - start_time))

def cross_evaluate_models_static_decalib(models_path, samples_list):
    # Initialize models.
    start_time = time()
    print("Loading model weights...")
    model_1, model_2 = load_models(models_path)
    # Load data.
    print("Loading data...")
    images_init, projections_decalib, projections_groundtruth, decalibs, H_gts_init, Ks_init, radar_detections_init, dims = dl.load_dataset(samples_list)
    # Instantiate reprojection manager.
    reprojection_manager_batch = RadarBatchReprojectionManager(run_config.original_resolution, run_config.input_shape)

    mean_errors_init_tilt = []
    mean_errors_init_pan = []
    mean_errors_init_roll = []
    mean_errors_init_tilt_abs = []
    mean_errors_init_pan_abs = []
    mean_errors_init_roll_abs = []
    mean_errors_init_total_abs = []
    mean_errors_tilt = []
    mean_errors_pan = []
    mean_errors_roll = []
    mean_errors_tilt_abs = []
    mean_errors_pan_abs = []
    mean_errors_roll_abs = []
    mean_errors_total_abs = []

    num_runs = 100
    cnt_runs = 0
    while cnt_runs < num_runs:
        print("Starting test run {} of {}".format(cnt_runs+1, num_runs))
        images = images_init
        H_gts = H_gts_init
        Ks = Ks_init
        radar_detections = radar_detections_init

        decalib = dec.create_decalib_transformation(10., 0.1)

        decalib_inverted = dec.invert_homogeneous_matrix(decalib) # Invert for training label.
        rotation = decalib_inverted[:3, :3]
        translation = decalib_inverted[:3, 3]
        quat = quattransform.quaternion_from_matrix(rotation)
        label = np.concatenate((quat, translation))
        label = np.expand_dims(label, axis=0)
        labels = np.repeat(label, len(samples_list), axis=0)

        h_inits = np.matmul(decalib, H_gts)
        projections = reprojection_manager_batch._project_radar_detections(h_inits, Ks, radar_detections)
        img_list = []
        proj_list = []
        labels_list = []
        detections_list = []
        H_gts_list = []
        Ks_list = []
        samples = zip(images, projections, labels, radar_detections, H_gts, Ks)
        for img, proj, label, detections, H_gt, K in samples:
            if np.count_nonzero(proj) > 9:
                img_list.append(img)
                proj_list.append(proj)
                labels_list.append(label)
                detections_list.append(detections)
                H_gts_list.append(H_gt)
                Ks_list.append(K)
        print("Number of samples: {}".format(len(img_list)))
        if len(img_list) < 300:
            continue

        images = np.array(img_list)
        projections = np.array(proj_list)
        labels = np.array(labels_list)
        radar_detections = np.array(detections_list)
        H_gts = np.array(H_gts_list)
        Ks = np.array(Ks_list)

        # Apply model.
        print("Applying model...")
        trans_labels = decalibs[:, 4:]
        #k_mats = Ks[:, :, :3]
        projections_1, labels_1 = reprojection_manager_batch.compute_projections_and_labels([images, projections, Ks, trans_labels], labels, radar_detections, H_gts, Ks, model_1)
        trans_labels_1 = trans_labels #labels_1[:, 4:]
        projections_2, labels_2 = reprojection_manager_batch.compute_projections_and_labels([images, projections_1, Ks, trans_labels_1], labels_1, radar_detections, H_gts, Ks, model_2)
        print("Reprojection time: " + str(time() - start_time))

        # add mean errors.
        mean_errors_init_tilt.append(np.mean(pv.quat_tilt_angles(labels)))
        mean_errors_init_pan.append(np.mean(pv.quat_pan_angles(labels)))
        mean_errors_init_roll.append(np.mean(pv.quat_roll_angles(labels)))

        mean_errors_init_total_abs.append(np.mean(np.absolute(pv.comp_abs_quat_angles(labels))))
        mean_errors_init_tilt_abs.append(np.mean(np.absolute(pv.quat_tilt_angles(labels))))
        mean_errors_init_pan_abs.append(np.mean(np.absolute(pv.quat_pan_angles(labels))))
        mean_errors_init_roll_abs.append(np.mean(np.absolute(pv.quat_roll_angles(labels))))

       # mean_errors_tilt.append(np.mean(pv.quat_tilt_angles(labels_2)))
       # mean_errors_pan.append(np.mean(pv.quat_pan_angles(labels_2)))
       # mean_errors_roll.append(np.mean(pv.quat_roll_angles(labels_2)))

        mean_errors_total_abs.append(np.mean(np.absolute(pv.comp_abs_quat_angles(labels_2))))
        mean_errors_tilt_abs.append(np.mean(np.absolute(pv.quat_tilt_angles(labels_2))))
        mean_errors_pan_abs.append(np.mean(np.absolute(pv.quat_pan_angles(labels_2))))
        mean_errors_roll_abs.append(np.mean(np.absolute(pv.quat_roll_angles(labels_2))))

        print("Evaluation:")
        print("Initial:")
        pv.print_angular_errors(labels)
        print("Model 2:")
        pv.print_angular_errors(labels_2)
        # print("Generating corrected projections...")
        pv.create_paper_plots(str(experiments_path + "{}/".format(cnt_runs)), projections, labels_2, labels)
        # visualize_corrected_projections(images, projections_groundtruth, projections, projections_1, projections_2)
        # print("Plotting time: " + str(time() - start_time))
        cnt_runs = cnt_runs + 1

    mean_errors_init_tilt = np.array(mean_errors_init_tilt)
    mean_errors_init_pan = np.array(mean_errors_init_pan)
    mean_errors_init_roll = np.array(mean_errors_init_roll)
    mean_errors_init_tilt_abs = np.array(mean_errors_init_tilt_abs)
    mean_errors_init_pan_abs = np.array(mean_errors_init_pan_abs)
    mean_errors_init_roll_abs = np.array(mean_errors_init_roll_abs)
    mean_errors_init_total_abs = np.array(mean_errors_init_total_abs)
    mean_errors_tilt = np.array(mean_errors_tilt)
    mean_errors_pan = np.array(mean_errors_pan)
    mean_errors_roll = np.array(mean_errors_roll)
    mean_errors_tilt_abs = np.array(mean_errors_tilt_abs)
    mean_errors_pan_abs = np.array(mean_errors_pan_abs)
    mean_errors_roll_abs = np.array(mean_errors_roll_abs)
    mean_errors_total_abs = np.array(mean_errors_total_abs)

    print("Overall Mean Errors:")
    print("Initial Tilt: {}".format(np.mean(mean_errors_init_tilt)))
    print("Initial Pan: {}".format(np.mean(mean_errors_init_pan)))
    print("Initial Roll: {}".format(np.mean(mean_errors_init_roll)))
    print("")
    print("Initial Tilt abs: {}".format(np.mean(mean_errors_init_tilt_abs)))
    print("Initial Pan abs: {}".format(np.mean(mean_errors_init_pan_abs)))
    print("Initial Roll abs: {}".format(np.mean(mean_errors_init_roll_abs)))
    print("Initial Total abs: {}".format(np.mean(mean_errors_init_total_abs)))
    print("")
    print("Tilt: {}".format(np.mean(mean_errors_tilt)))
    print("Pan: {}".format(np.mean(mean_errors_pan)))
    print("Roll: {}".format(np.mean(mean_errors_roll)))
    print("")
    print("Tilt abs: {}".format(np.mean(mean_errors_tilt_abs)))
    print("Pan abs: {}".format(np.mean(mean_errors_pan_abs)))
    print("Roll abs: {}".format(np.mean(mean_errors_roll_abs)))
    print("Total abs: {}".format(np.mean(mean_errors_total_abs)))

    data = pd.DataFrame({'Tilt':mean_errors_init_tilt, 'Pan':mean_errors_init_pan, 'Roll':mean_errors_init_roll})
    pv.save_boxplot(data, 'Rotation Axis', 'Error in Degree', experiments_path + "init_rotation_boxplot.png")
    data = pd.DataFrame({'Tilt':mean_errors_init_tilt_abs, 'Pan':mean_errors_init_pan_abs, 'Roll':mean_errors_init_roll_abs})
    rot_data = pd.DataFrame({'Total Rotation':mean_errors_init_total_abs})
    pv.save_boxplot(data, 'Rotation Axis', 'Absolute Error in Degree', experiments_path + "init_rotation_boxplot_abs.png")
    pv.save_boxplot(rot_data, '', 'Absolute Error in Degree', experiments_path + "init_rot_boxplot_abs.png", width=0.5)

    data = pd.DataFrame({'Tilt':mean_errors_tilt, 'Pan':mean_errors_pan, 'Roll':mean_errors_roll})
    pv.save_boxplot(data, 'Rotation Axis', 'Error in Degree', experiments_path + "rotation_boxplot.png")

    data = pd.DataFrame({'Tilt':mean_errors_tilt_abs, 'Pan':mean_errors_pan_abs, 'Roll':mean_errors_roll_abs})
    rot_data = pd.DataFrame({'Total Rotation':mean_errors_total_abs})
    pv.save_boxplot(data, 'Rotation Axis', 'Absolute Error in Degree', experiments_path + "rotation_boxplot_abs.png")
    pv.save_boxplot(rot_data, '', 'Error in Degree', experiments_path + "rot_boxplot_abs.png", width=0.5)

    data = pd.DataFrame({'Initial':mean_errors_init_total_abs, 'Corrected':mean_errors_total_abs})
    pv.save_boxplot(data, 'Total Rotation', 'Absolute Error in Degree', experiments_path + "rotation_boxplot_total.png", width=0.6)


def main():
    global experiments_path
    # Parse user arguments.
    args = parse_commandline()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # Set gpu to run on.
    # Optimizations for Intel CPU usage
    # os.environ["KMP_BLOCKTIME"] = "0"
    # os.environ["KMP_SETTINGS"] = "0"
    # os.environ["KMP_AFFINITY"] = 'granularity=fine,verbose,compact,1,0'
    # config = tf.ConfigProto()
    # config.inter_op_parallelism_threads = 2
    # config.intra_op_parallelism_threads = 8
    # sess = tf.Session(config=config)
    # K.set_session(sess)

    # Construct experiment folder name.
    date_time = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')
    experiments_path += date_time + ("_" + args.run_name if args.run_name else "") + "/"
    if not os.path.exists(experiments_path): # Create experiments directory.
        os.makedirs(experiments_path)
        os.makedirs(experiments_path + model_1_name)
        os.makedirs(experiments_path + model_2_name)

    # Save current git commit hash and info string if provided.
    with open(experiments_path+ 'info.txt', 'w') as info_file:
        git_hash = retrieve_git_hash()
        if git_hash:
            info_file.write('GIT commit hash: ' + str(git_hash) + '\n\n')
        if args.info:
            info_file.write("Experiment info:\n" + args.info + '\n')

    # Save run configuration in file.
    save_run_params_in_file()

    # Write output to file.
    tee = Tee(experiments_path + 'console_output.log', 'w')

    # Store code for reproducibility.
    copy_code(experiments_path)

    # Prepare data.
    sample_files_list = []
    print("Loading datasets:")
    for dataset_path in run_config.dataset_paths:
        print(dataset_path)
        temp = [f for f in glob.glob(dataset_path + "*.npz")] # get list of all sample file names in dataset folder
        sample_files_list.extend(temp)

    sample_files_list.sort()
    print("List of data samples loaded: {} samples found.".format(str(len(sample_files_list))))

    if args.weights_path:
        eval_start = time()
        print("Weights path specified: {}".format(args.weights_path))
        print("Starting inference...")
        if args.static_analysis:
            cross_evaluate_models_static_decalib(args.weights_path, sample_files_list)
        else:
            cross_evaluate_models(args.weights_path, sample_files_list, args.static_decalib)
        print("Evaluation time: " + str(time() - eval_start))
    else: # Training
        print("Starting training...")
        train_start = time()
        start_training(sample_files_list)
        #start_training(data, labels, radar_detections, projection_params, projections_groundtruth_sparse)
        print("Training time: " + str(time() - train_start))

if __name__ == '__main__':
    main()
