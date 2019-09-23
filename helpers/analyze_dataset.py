import argparse
import numpy as np
import os.path as path
import transformations as transf
import math

def to_angle_axis(quaternion):
    rot_mat = transf.quaternion_matrix(quaternion)
    angle, direction, point = transf.rotation_from_matrix(rot_mat)
    return np.array([angle, direction[0], direction[1], direction[2]])

def comp_absmean_min_max(values):
    return (np.mean(np.abs(values)), np.min(values), np.max(values))

def analyze_translation(translations):
    mean_euclidean_distance = np.mean(np.sqrt(np.sum(np.square(translations), axis=1)))
    x_values, y_values, z_values = translations[:,0], translations[:,1], translations[:,2]
    absmean_x, min_x, max_x = comp_absmean_min_max(x_values)
    absmean_y, min_y, max_y = comp_absmean_min_max(y_values)
    absmean_z, min_z, max_z = comp_absmean_min_max(z_values)
    print("== Translation ==")
    print("Mean euclidean distance: " + str(mean_euclidean_distance))
    print("absmean_x, min_x, max_x: {}, {}, {}".format(absmean_x, min_x, max_x))
    print("absmean_y, min_y, max_y: {}, {}, {}".format(absmean_y, min_y, max_y))
    print("absmean_z, min_z, max_z: {}, {}, {}".format(absmean_z, min_z, max_z))

def to_angle_axis_matrix(quaternions):
    angle_axis_matrix = []
    for quaternion in quaternions:
        angle_axis_matrix.append(to_angle_axis(quaternion))
    angle_axis_matrix = np.array(angle_axis_matrix)
    return angle_axis_matrix

def analyze_rotation(quaternions):
    aa_mat = to_angle_axis_matrix(quaternions)
    angles, x_axis, y_axis, z_axis = aa_mat[:,0], aa_mat[:,1], aa_mat[:,2], aa_mat[:,3]
    absmean_angle, max_angle, min_angle = comp_absmean_min_max(angles)
    absmean_x_axis, max_x_axis, min_x_axis = comp_absmean_min_max(x_axis)
    absmean_y_axis, max_y_axis, min_y_axis = comp_absmean_min_max(y_axis)
    absmean_z_axis, max_z_axis, min_z_axis = comp_absmean_min_max(z_axis)
    print("== Rotation ==")
    print("absmean_angle, max_angle, min_angle: {}, {}, {}".format(absmean_angle, max_angle, min_angle))
    print("absmean_x_axis, max_x_axis, min_x_axis: {}, {}, {}".format(absmean_x_axis, max_x_axis, min_x_axis))
    print("absmean_y_axis, max_y_axis, min_y_axis: {}, {}, {}".format(absmean_y_axis, max_y_axis, min_y_axis))
    print("absmean_z_axis, max_z_axis, min_z_axis: {}, {}, {}".format(absmean_z_axis, max_z_axis, min_z_axis))

def run_analyzation(folder_path):
    decalibs_inv = np.load(path.join(folder_path, 'decalibrations_inverted.npz'), encoding='latin1')['decalibrations_inverted']
    quaternions = decalibs_inv[:,:4]
    translations = decalibs_inv[:,4:]
    analyze_rotation(quaternions)
    analyze_translation(translations)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyzes a dataset descriptively.")
    parser.add_argument('folder_path', type=str, help="Path to the dataset folder.")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    run_analyzation(args.folder_path)

if __name__ == '__main__':
    main()


# angle = math.pi / 2.
# direction = [0.7, -0.2, 1.1]
# print(angle, direction)
# rot_mat = transf.rotation_matrix(angle, direction)
# quat = transf.quaternion_from_matrix(rot_mat)
# new_rot_mat = transf.quaternion_matrix(quat)
# new_angle, new_direction, new_point = transf.rotation_from_matrix(new_rot_mat)
# print(rot_mat)
# print(quat)
# print(new_rot_mat)
# print(new_angle, new_direction)

# # Assemble rot mat.
# deg = 30
# rad = deg * math.pi / 180.
# rot_mat = transf.rotation_matrix(rad, [1, 0, 0])
# rot_mat[:3, 3] = np.array([3, -5, 2])

# # Invert homogeneous matrix.
# def invert_homogeneous_matrix(mat):
#     mat_inverse = np.zeros(mat.shape)
#     mat_inverse[:3,:3] = np.linalg.inv(mat[:3,:3])
#     mat_inverse[:3,3] = mat[:3,3] * -1.
#     return mat_inverse

# print(rot_mat)
# print ('\n')
# print(rot_mat_inv)
