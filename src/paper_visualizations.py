import math
import os

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def create_paper_plots(experiments_path, init_projections, residual_decalib_quats, init_decalib_quats):
    plots_path = experiments_path + 'paper_plots/'
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    plot_correspondences_error_scatter(init_projections, residual_decalib_quats, plots_path)
    plot_angle_error_distributions(residual_decalib_quats, plots_path)
    plots_path_inits = plots_path + 'initial/'
    if not os.path.exists(plots_path_inits):
        os.makedirs(plots_path_inits)
    plot_angle_error_distributions(init_decalib_quats, plots_path_inits)

def plot_correspondences_error_scatter(init_projections, residual_decalib_quats, experiments_path, color="g"):
    plt.tight_layout()
    cnt_projections = np.sum((init_projections > 0), axis=(1,2))
    decalib_angles = comp_abs_quat_angles(residual_decalib_quats)
    ax = sns.regplot(x=cnt_projections, y=decalib_angles, fit_reg=False, color=color)
    ax.set(xlabel="Number Correspondences", ylabel="Decalibration Error")
    sns.set(font_scale=1.5)
    ax.get_figure().savefig(experiments_path + "error_correspondences.png")
    plt.clf()

    data_json = json.dumps({'correspondences': cnt_projections.tolist(), 'angles': decalib_angles.tolist()})
    with open(experiments_path + 'angles_corrs.json', 'w') as f:
        f.write(data_json)


def plot_angle_error_distributions(residual_decalib_quats, experiments_path):
    # Compute errors.
    rot_errors = comp_total_quat_angles(residual_decalib_quats)
    tilt_errors = quat_tilt_angles(residual_decalib_quats)
    pan_errors = quat_pan_angles(residual_decalib_quats)
    roll_errors = quat_roll_angles(residual_decalib_quats)
    # Plot boxplots
    errors = np.array([tilt_errors, pan_errors, roll_errors, rot_errors])
    data = pd.DataFrame({'Tilt':errors[0,:], 'Pan':errors[1,:], 'Roll':errors[2,:]})
    tilt_data = pd.DataFrame({'Tilt':errors[0,:]})
    pan_data = pd.DataFrame({'Pan':errors[1,:]})
    roll_data = pd.DataFrame({'Roll':errors[2,:]})
    rot_data = pd.DataFrame({'Total Rotation':errors[3,:]})
    save_boxplot(data, 'Rotation Axis', 'Error in Degree', experiments_path + "rotation_boxplot.png")
    save_boxplot(tilt_data, '', 'Error in Degree', experiments_path + "tilt_boxplot.png", figsize=(7,7), width=0.5)
    save_boxplot(pan_data, '', 'Error in Degree', experiments_path + "pan_boxplot.png", figsize=(7,7), width=0.5)
    save_boxplot(roll_data, '', 'Error in Degree', experiments_path + "roll_boxplot.png", figsize=(7,7), width=0.5)
    save_boxplot(rot_data, '', 'Error in Degree', experiments_path + "rot_boxplot.png")

    # Plot histograms.
    num_bins = 50
    label_y = 'Count'
    save_hist_plot(tilt_errors, "Degree Error", label_y, num_bins, experiments_path + "tilt_error_distribution.png")
    save_hist_plot(pan_errors, "Degree Error", label_y, num_bins, experiments_path + "pan_error_distribution.png")
    save_hist_plot(roll_errors, "Degree Error", label_y, num_bins, experiments_path + "roll_error_distribution.png")
    save_hist_plot(rot_errors, "Degree Error", label_y, num_bins, experiments_path + "rot_error_distribution.png")

    data_json = json.dumps({'tilt': tilt_errors.tolist(), 'pan': pan_errors.tolist(), 'roll': roll_errors.tolist(), 'total': rot_errors.tolist()})
    with open(experiments_path + 'angle_hists.json', 'w') as f:
        f.write(data_json)

def save_boxplot(data, label_x, label_y, output_path, color="g", figsize=(12,9), width=0.7):
    # plt.figure()
    plt.figure(figsize=figsize)
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    ax = sns.boxplot(data=data, palette="Blues", showfliers=False, width=width)
    ax.set(xlabel=label_x, ylabel=label_y)
    # Add transparency to colors
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))
    fig = ax.get_figure()
    fig.savefig(output_path)
    plt.close(fig)

def save_hist_plot(x, label_x, label_y, bins, output_path, kde=False, color="b"):
    # plt.figure()
    plt.figure(figsize=(8,6))
    sns.set(font_scale=1.5)
    sns.set_style('white')
    ax = sns.distplot(x, kde=kde, bins=bins, color=color)
    ax.grid(False)
    ax.set(xlabel=label_x, ylabel=label_y)
    ax.set(xlim=(-7., 7))
    fig = ax.get_figure()
    fig.savefig(output_path)
    plt.close(fig)

def quat_tilt_angles(quat):
    w, x, y, z = split_quat(quat)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    x_angles = np.degrees(np.arctan2(t0, t1))
    return x_angles

def quat_pan_angles(quat):
    w, x, y, z = split_quat(quat)
    t = +2.0 * (w * y - z * x)
    t = np.clip(t, -0.999999999, 0.999999999)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    y_angles = np.degrees(np.arcsin(t))
    return y_angles

def quat_roll_angles(quat):
    w, x, y, z = split_quat(quat)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    z_angles = np.degrees(np.arctan2(t3, t4))
    return z_angles

def split_quat(quat):
    w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    return w, x, y, z

def comp_abs_quat_angles(quats):
    angles = 2 * np.arccos(quats[:,0])
    angles = abs(angles * 180 / math.pi)
    return angles

def comp_total_quat_angles(quats):
    angles = np.degrees(2.0 * np.arccos(quats[:,0]))
    return angles

def print_angular_errors(quats):
    total_errors = comp_abs_quat_angles(quats)
    tilt_errors = quat_tilt_angles(quats)
    pan_errors = quat_pan_angles(quats)
    roll_errors = quat_roll_angles(quats)
    tilt_errors_abs = np.absolute(tilt_errors)
    pan_errors_abs = np.absolute(pan_errors)
    roll_errors_abs = np.absolute(roll_errors)
    print("- Total angle error (abs): {}".format(np.mean(total_errors)))
    print("- Tilt error (abs): {}".format(np.mean(tilt_errors_abs)))
    print("- Pan error (abs): {}".format(np.mean(pan_errors_abs)))
    print("- Roll error (abs): {}".format(np.mean(roll_errors_abs)))
    print("- Tilt error: {}".format(np.mean(tilt_errors)))
    print("- Pan error: {}".format(np.mean(pan_errors)))
    print("- Roll error: {}".format(np.mean(roll_errors)))
    print("- Tilt error median: {}".format(np.median(tilt_errors)))
    print("- Pan error median: {}".format(np.median(pan_errors)))
    print("- Roll error median: {}".format(np.median(roll_errors)))
    # rot_avg, rot_max, rot_min = comp_rot_errors(labels[:, :4])
    # print("- Average rotational error: {}".format(str(rot_avg)))
    # print("- Max rotational error: {}".format(str(rot_max)))
    # print("- Min rotational error: {}".format(str(rot_min)))


# def comp_rot_errors(quats):
#     angles = comp_abs_quat_angles(quats)
#     return np.mean(angles), np.max(angles), np.min(angles)