import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion as tfg_quaternion
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d as tfg_rot_mat

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

def transform_from_quat_and_trans(quaternion, trans_vector):
    """
    Method that creates an augmented transform which includes the batch size
    in the output shape
    :param quaternion: (batch_size, 4) quaternion vectors
    :param trans_vector: (batch_size, 3, 1) translation vectors
    :return: transform_augm (batch_size, 4, 4) augmented transform matrix
    """
    #We use [w, x, y, z] quaternion notation but TF Geometry lib expects [x, y, z, w]
    #quaternion = tf.concat([quaternion2[:, 1:], tf.expand_dims(quaternion2[:, 0], axis=1)], axis=-1)

    #TODO: if I don't normalize here, quat_loss drops even when only cloud_loss is used!
    # However, cloud_loss is much bigger!
    quat_normalized = tfg_quaternion.normalize(quaternion)
    #predicted_rot_mat = tfg_rot_mat.from_quaternion(quat_normalized)
    predicted_rot_mat = rot_matrix_from_quat_wxyz(quat_normalized)
    paddings = tf.constant([[0, 0], [0, 1], [0, 0]])
    predicted_rot_mat_augm = tf.pad(predicted_rot_mat, paddings, constant_values=0)
    decalib_qt_trans_augm = tf.pad(trans_vector, paddings, constant_values=1)
    transform_augm = tf.concat([predicted_rot_mat_augm, decalib_qt_trans_augm], axis=-1)

    return transform_augm


# Modified the below method from TF graphics library
# https://www.tensorflow.org/graphics/api_docs/python/tfg/geometry/transformation/rotation_matrix_3d/from_quaternion
# The original supported xyzw format but our dataset uses wxyz
#Input must be a NORMALIZED quaternion
def rot_matrix_from_quat_wxyz(quaternion):
    w, x, y, z = tf.unstack(quaternion, axis=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                      txy + twz, 1.0 - (txx + tzz), tyz - twx,
                      txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                     axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
    out_matrix = tf.reshape(matrix, shape=output_shape)
    return out_matrix



if __name__ == '__main__': # For debugging.
    quat = tf.constant([[1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    trans_vector = tf.constant([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]])
    transform_augm = transform_from_quat_and_trans(quat, trans_vector)
    with tf.Session() as sess:
        out = sess.run(transform_augm)
        print(out)
