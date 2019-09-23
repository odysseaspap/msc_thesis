import tensorflow as tf


def split_dual_quaternions(quat):
    """
    Splits the given tensor of dual quaternions in real and dual parts.
    """
    quat_split = tf.split(quat, [4, 4], axis=1)
    return quat_split[0], quat_split[1]

def batchwise_dot_product(x, y):
    return tf.reduce_sum(tf.multiply(x, y), 1, keepdims=True)

def conjugate_quaternions(quaternions):
    return tf.multiply(quaternions, [1.0, -1.0, -1.0, -1.0])

def multiply_quaternions(quat_1, quat_2):
    """
    Multiplies two quaternion tensors (1D vector shape) according to the formula:
    q_1*q_2 = ((w_1*w_2 - v_1*v_2), w_1*v_1 + w_2*v-1 + cross(v_1, v_2))
    with q = (w, v) and v = [x, y, z]^T
    """
    w1, x1, y1, z1 = tf.unstack(quat_1, axis=-1, num=4)
    w2, x2, y2, z2 = tf.unstack(quat_2, axis=-1, num=4)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return tf.squeeze(tf.stack((w, x, y, z), axis=-1))

def normalize_quaternions(quats): # Reduces to euclidean norm.
    inverse_mags = tf.constant(1, dtype=tf.float32) / tf.sqrt(tf.reduce_sum(tf.square(quats), axis=1))
    return tf.transpose(tf.multiply(tf.transpose(quats), inverse_mags))

def compute_delta_quaternion(y_true, y_pred):
    quat_true = y_true
    quat_pred = y_pred
    # quat_pred = tf.split(y_pred, [4, 1], axis=1)[0]
    quat_true = normalize_quaternions(quat_true)
    quat_pred = normalize_quaternions(quat_pred)
    decalib_hat_inverse = quat_pred
    decalib = conjugate_quaternions(quat_true)
    delta_quaternion = multiply_quaternions(decalib_hat_inverse, decalib)
    return delta_quaternion