import numpy as np
import math

def create_decalib_transformation(train_size, angle_std_deviation, translation_std_deviation):
    sampled_rotation_matrix = create_random_rotation_matrix(angle_std_deviation)
    sampled_translation_vector = create_random_translation_vector(translation_std_deviation)
    return create_homogeneous_transformation(sampled_rotation_matrix, sampled_translation_vector)

def create_homogeneous_transformation(rotation_matrix, translation_vector):
    """
    Create an homogeneous transformation matrix with given rotation matrix and translation vector
    """
    homogeneous_matrix = np.identity(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation_vector
    return homogeneous_matrix

def create_random_rotation_matrix(angle_std_dev):
    # Create random roll rotation (rotation around z-axis)
    roll_angle = np.random.uniform(low=-angle_std_dev*0.5, high=angle_std_dev*0.5)
    # print("Roll angle: {}".format(roll_angle))
    roll_angle = np.radians(roll_angle)
    # roll_angle = np.radians(angle_std_dev * np.random.randn()) * 0.5 # Assume smaller roll error.
    roll_c, roll_s = np.cos(roll_angle), np.sin(roll_angle)
    roll_matrix = np.array([[roll_c, -roll_s, 0], [roll_s, roll_c, 0], [0, 0, 1]])

    # Create random yaw rotation (rotation around y-axis)
    yaw_angle = np.random.uniform(low=-angle_std_dev, high=angle_std_dev)
    # print("Pan/Yaw angle: {}".format(yaw_angle))
    yaw_angle = np.radians(yaw_angle)
    # yaw_angle = np.radians(angle_std_dev * np.random.randn())
    yaw_c, yaw_s = np.cos(yaw_angle), np.sin(yaw_angle)
    yaw_matrix = np.array([[yaw_c, 0, yaw_s],[0, 1, 0], [-yaw_s, 0, yaw_c]])

    # Create random pitch rotation (rotation around x-axis)
    # pitch_angle = np.radians(np.random.uniform(low=-angle_std_dev, high=angle_std_dev))
    pitch_angle = np.random.uniform(low=-angle_std_dev, high=angle_std_dev)
    # print("Pitch/Tilt angle: {}".format(pitch_angle))
    pitch_angle = np.radians(pitch_angle)
    # pitch_angle = np.radians(angle_std_dev * np.random.randn())
    pitch_c, pitch_s = np.cos(pitch_angle), np.sin(pitch_angle)
    pitch_matrix = np.array([[1, 0, 0], [0, pitch_c, -pitch_s], [0, pitch_s, pitch_c]])

    # Create overall rotation matrix
    return (roll_matrix.dot(pitch_matrix.dot(yaw_matrix)))

def create_random_translation_vector(translation_std_dev):
    """
    Create random translation vector in 3D from standard normal distribution.
    """
    x = np.random.uniform(low=-translation_std_dev, high=translation_std_dev)
    y = np.random.uniform(low=-translation_std_dev, high=translation_std_dev)
    z = np.random.uniform(low=-translation_std_dev, high=translation_std_dev)
    # x = np.random.randn() * translation_std_dev
    # y = np.random.randn() * translation_std_dev
    # z = np.random.randn() * translation_std_dev
    return np.array([x, y, z])
