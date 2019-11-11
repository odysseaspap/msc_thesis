import math
import tensorflow as tf
import util.quaternion_ops as quat
import keras.backend as K

def rot_angle_error(y_true, y_pred):
    """
    Computes the angular error between the quaternions by converting their difference
    to angle axis representation. Formula used to compute the angular difference in radians:
    2 * arcos(|q1 * q2|)
    Taken from: 
    https://math.stackexchange.com/questions/90081/quaternion-distance and verified in
    https://link.springer.com/article/10.1007%2Fs10851-009-0161-2 (Du Q.Huynh, Section 3.4)
    """
    quat_true = y_true[:, :4]
    quat_pred = y_pred
    #quat_pred = tf.split(y_pred, [4, 1], axis=-1)[0]
    # quat_true = tf.split(y_true, [4, 3], axis=1)[0]
    # quat_pred = tf.split(y_pred, [4, 3], axis=1)[0]
    quat_true = quat.normalize_quaternions(quat_true)
    quat_pred = quat.normalize_quaternions(quat_pred)

    quat_dot_product = tf.abs(quat.batchwise_dot_product(quat_true, quat_pred))
    quat_dot_product = tf.clip_by_value(quat_dot_product, -0.99999, 0.99999)
    check_greater_min = tf.assert_greater_equal(quat_dot_product, -1.)
    check_smaller_max = tf.assert_less_equal(quat_dot_product, 1.)
    with tf.control_dependencies([check_greater_min, check_smaller_max]):
        diffs_rad = 2. * tf.acos(quat_dot_product)
        diffs_deg = (diffs_rad * 180.) / math.pi
        return tf.reduce_mean(diffs_deg)

def tilt_error(y_true, y_pred):
    y_true = y_true[:, :4]
    delta_quaternion = quat.compute_delta_quaternion(y_true, y_pred)
    q0, q1, q2, q3 = tf.split(delta_quaternion, [1, 1, 1, 1], axis=1)
    t0 = 2.*(q0*q1 + q2*q3)
    t1 = 1. - 2.*(tf.square(q1) + tf.square(q2))
    tilt_deg = tf.atan2(t0, t1) * (180./math.pi)
    tilt_deg = tf.abs(tilt_deg)
    return tf.reduce_mean(tilt_deg)
	# t0 = +2.0 * (w * x + y * z)
	# t1 = +1.0 - 2.0 * (x * x + ysqr)
	# X = math.degrees(math.atan2(t0, t1))

def pan_error(y_true, y_pred):
    y_true = y_true[:, :0]
    #delta_quaternion = quat.compute_delta_quaternion(y_true, y_pred)
    #q0, q1, q2, q3 = tf.split(delta_quaternion, [1, 1, 1, 1], axis=1)
    #t0 = 2.*(q0*q2 + q3*q1)
    #t0 = tf.clip_by_value(t0, clip_value_min=-0.999999999, clip_value_max=0.999999999)
    pan_diff = y_true - y_pred
    pan_deg = pan_diff * (180./math.pi)
    pan_deg = tf.abs(pan_deg)
    #mse_error = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis = 1))
    return tf.reduce_mean(pan_deg)
	# t2 = +2.0 * (w * y - z * x)
	# t2 = +1.0 if t2 > +1.0 else t2
	# t2 = -1.0 if t2 < -1.0 else t2
	# Y = math.degrees(math.asin(t2))

def roll_error(y_true, y_pred):
    y_true = y_true[:, :4]
    delta_quaternion = quat.compute_delta_quaternion(y_true, y_pred)
    q0, q1, q2, q3 = tf.split(delta_quaternion, [1, 1, 1, 1], axis=1)
    t0 = 2.*(q0*q3 + q1*q2)
    t1 = 1. - 2.*(tf.square(q2) + tf.square(q3))
    roll_deg = tf.atan2(t0, t1) * (180./math.pi)
    roll_deg = tf.abs(roll_deg)
    return tf.reduce_mean(roll_deg)
	# t3 = +2.0 * (w * z + x * y)
	# t4 = +1.0 - 2.0 * (ysqr + z * z)
	# Z = math.degrees(math.atan2(t3, t4))

def trans_error(y_true, y_pred):
#     y_true = tf.split(y_true, [4, 3], axis=1)
#     y_pred = tf.split(y_pred, [4, 3], axis=1)
#     trans_true, trans_pred = y_true[1], y_pred[1]
    trans_true = y_true
    trans_pred = y_pred
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(trans_true - trans_pred), axis=1))) # Euclidean distance.

def trans_error_x(y_true, y_pred):
#     y_true = tf.split(y_true, [4, 3], axis=1)
#     y_pred = tf.split(y_pred, [4, 3], axis=1)
#     trans_true, trans_pred = y_true[1], y_pred[1]
    trans_true = y_true
    trans_pred = y_pred
    return tf.reduce_mean(tf.sqrt(tf.square(trans_true[:, 0] - trans_pred[:, 0]))) # Euclidean distance.

def trans_error_y(y_true, y_pred):
#     y_true = tf.split(y_true, [4, 3], axis=1)
#     y_pred = tf.split(y_pred, [4, 3], axis=1)
#     trans_true, trans_pred = y_true[1], y_pred[1]
    trans_true = y_true
    trans_pred = y_pred
    return tf.reduce_mean(tf.sqrt(tf.square(trans_true[:, 1] - trans_pred[:, 1]))) # Euclidean distance.

def trans_error_z(y_true, y_pred):
#     y_true = tf.split(y_true, [4, 3], axis=1)
#     y_pred = tf.split(y_pred, [4, 3], axis=1)
#     trans_true, trans_pred = y_true[1], y_pred[1]
    trans_true = y_true
    trans_pred = y_pred
    return tf.reduce_mean(tf.sqrt(tf.square(trans_true[:, 2] - trans_pred[:, 2]))) # Euclidean distance.