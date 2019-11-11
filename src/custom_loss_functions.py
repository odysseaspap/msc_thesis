import tensorflow as tf
import keras.backend as K
from util import all_transformer as at3
from util import quaternion_ops as qt_ops
#from util import model_utils
from util import metrics as met

def keras_photometric_and_3d_pointcloud_loss(radar_input, k_mat, depth_maps_predicted, cloud_pred, alpha = 1.0, beta = 1.0):
    """
    Keras wrapper function for using weighted dual quaternion distance.
    Added extra term to penalize roll angle errors more than the rest,
    because automotive radar detections lie on the same line making roll angle
    changes not so important which leads to poor accuracy for this angle
    """

    def loss(y_true, y_pred):
        return photometric_and_3d_pointcloud_loss(y_true, y_pred, radar_input, k_mat, depth_maps_predicted, cloud_pred, alpha, beta)

    return loss

def photometric_and_3d_pointcloud_loss(y_true, y_pred, radar_input, k_mat, depth_maps_predicted, cloud_pred, alpha, beta):
    """
    :param y_true: (batch_size, 7, 1) -  ground truth de-calibration
    quaternion (indexes 0-3) and translation (indexes 4-6) vectors
    :param y_pred: Output of RadNet [(batch_size, H, W), (batch_size, num_points, 3), (batch_size, H, W, 1)]
    [depth_map_predicted, cloud_predicted, radar_input]
    :param alpha: float value - photometric loss weight
    :param beta: float value - 3D point cloud loss weight

    :return: float value - Final loss value
    """
    # In Keras, the i-th loss function uses the i-th pair of output, label as y_pred, y_true.

    batch_size = tf.shape(radar_input)[0]
    y_true = tf.reshape(y_true, (batch_size, 4))

    yaw_expected = y_true[:, 0]
    trans_expected = y_true[:, 1:]
    trans_expected = tf.reshape(trans_expected, (batch_size, 3, 1))

    # rotation_matrix_3d.from_euler expects  [x - pitch, y - yaw, z- roll] vectors in Radians
    paddings = tf.constant([[0, 0], [1, 1]])
    expected_decalib_pyr = tf.pad(yaw_expected, paddings, constant_values=0.0)
    T_expected = qt_ops.transform_from_pyr_and_trans(expected_decalib_pyr, trans_expected)

    depth_maps_expected, cloud_exp = tf.map_fn(lambda x: at3._simple_transformer(radar_input[x, :, :, 0], T_expected[x], k_mat[x]), elems=tf.range(0, batch_size, 1), dtype=(tf.float32, tf.float32))


    # photometric loss between predicted and expected transformation
    # Note that here they have to re-normalize the depth maps since they de-normalized them before the ST layers!!!
    # plus, they measure the photometric loss only in a 10x10 area in the center of the image!!!
    #photometric_loss = tf.nn.l2_loss(tf.subtract(depth_maps_expected[:, 10:-10, 10:-10], depth_maps_predicted[:, 10:-10, 10:-10]))

    # earth mover's distance between point clouds
    #cloud_loss = model_utils.get_emd_loss(cloud_pred, cloud_exp)

    # Average of squared distance between point clouds
    cloud_loss = tf.reduce_sum((cloud_pred - cloud_exp) ** 2, axis=-1)
    cloud_loss = tf.reduce_mean(cloud_loss)

    # final loss term
    #predicted_loss_train = alpha * photometric_loss #+ beta * cloud_loss
    predicted_loss_train = cloud_loss

    return predicted_loss_train



def keras_weighted_yaw_loss(alpha):
    """
    Keras wrapper function for using weighted dual quaternion distance.
    Added extra term to penalize roll angle errors more than the rest,
    because automotive radar detections lie on the same line making roll angle
    changes not so important which leads to poor accuracy for this angle
    """

    def loss(y_true, y_pred):
        return weighted_yaw_loss(y_true, y_pred, alpha)

    return loss


def weighted_yaw_loss(y_true, y_pred, alpha):

    return met.pan_error(y_true, y_pred)


def keras_weighted_yaw__euclidean_loss(alpha):
    """
    Keras wrapper function for using weighted dual quaternion distance.
    """

    def loss(y_true, y_pred):
        return weighted_yaw_euclidean_loss(y_true, y_pred, alpha)

    return loss


def weighted_yaw_euclidean_loss(y_true, y_pred, alpha):
    y_true = y_true[:, 0]
    diff = (y_true - y_pred) ** 2
    # mean_squared_error = tf.reduce_mean(tf.reduce_sum(diff, 1))
    eucl_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff, 1)))
    return eucl_dist
