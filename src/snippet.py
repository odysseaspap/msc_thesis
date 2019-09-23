#import tensorflow as tf

#sess = tf.InteractiveSession()

# x = tf.constant([1,2,3,4,5,6,7,8], dtype=tf.float32)
# print(tf.slice(x, [0, 4], [4]).eval())

# print(x[0:4].eval())
# print(x[4:8].eval())

# alpha_param = tf.constant(alpha, dtype=tf.float32)
# p_true, q_true = y_true[0:4], y_true[4:8]
# p_pred, q_pred = y_pred[0:4], y_pred[4:8]
# p_true = tf.nn.l2_normalize(p_true)
# p_pred = tf.nn.l2_normalize(p_pred)
# return tf.sum(tf.square((p_true - p_pred)) + alpha_param * tf.square((q_true - q_pred)))

# def batchwise_dot_product(x, y):
#     return tf.reduce_sum(tf.multiply(x, y), 1, keep_dims=True)

# def split_dual_quaternions(quat):
#     quat_split = tf.split(quat, [4, 4], axis=1)
#     return quat_split[0], quat_split[1]

# def normalize_dual_quaternions(dual_quaternions):
#         q_reals, q_duals = split_dual_quaternions(dual_quaternions)
#         quat_dot = tf.nn.relu(batchwise_dot_product(q_reals, q_reals)) # Only > 0
#         inverse_mag = tf.where(tf.greater(quat_dot, 0), 1./quat_dot, quat_dot)
#         normalized_dual_quaternions = dual_quaternions * inverse_mag
#         print(quat_dot.eval())
#         print(inverse_mag.eval())
#         print(normalized_dual_quaternions.eval())


# x = tf.constant([[1, -3, 4, 2, 6, 7, 5, 8], [2, -3, 6, 9, 7, 3, 4, 1], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
# normalize_dual_quaternions(x)


# x = tf.constant([[1, -3, 2, 4, 5, 6, 7], [2, -2, 3, 5, 6, 7, 8]], dtype=tf.float32)
# x = tf.constant([[1, -3, 4, 2], [6, 7, 5, 8]], dtype=tf.float32)

# print(x[:,1:].eval())

#y = tf.constant([[2, -2], [9, 3]], dtype=tf.float32)
# conj = tf.multiply(x, [1.0, -1.0, -1.0, -1.0])

#d = tf.reduce_sum(tf.multiply(x, y), 1, keep_dims=True)
#print(d.eval())

# x = tf.split(x, [4, 3], axis=1)
# print(x[0].eval())
# print(x[1].eval())


# inverse_mags = tf.constant(1, dtype=tf.float32) / tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
# normalized = tf.transpose(tf.multiply(tf.transpose(x), inverse_mags))
# print(x.eval())
# print(inverse_mags.eval())
# print(normalized.eval())


# y = tf.split(x, [4, 3], axis=1)
# print(y[0].eval())
# print(y[1].eval())



# import dual_quaternion.transformations as transformations
# import numpy as np
# import math

# def to_quaternion(mat):
#     quat = transformations.quaternion_from_matrix(mat)
#     return np.array(quat)

# degree_1 = 30
# rad_1 = degree_1 * math.pi / 180.
# rot1 = transformations.rotation_matrix(rad_1, [1,0,0])[:3,:3]
# quat1 = to_quaternion(rot1)

# degree_2 = 250
# rad_2 = degree_2 * math.pi / 180.
# rot2 = transformations.rotation_matrix(rad_2, [1,0,0])[:3,:3]
# quat2 = to_quaternion(rot2)

# print("Quaternions:")
# print(quat1)
# print(quat2)

# print("Metrics:")
# metric1 = 1. - abs(np.dot(quat1, quat2))
# metric2 = 2 * min(math.acos(np.dot(quat1, quat2)), math.pi - math.acos((np.dot(quat1, quat2))))
# metric3 = 2 * math.acos(abs(np.dot(quat1, quat2)))

# metrics = []
# metrics.append(metric1)
# metrics.append(metric2)
# metrics.append(metric3)

# angle1 = metric1 * 180.
# print(angle1)
# angle2 = ((metric3 * 180.) / math.pi)
# print(angle2)

# print(metrics)


# noise = tf.random_normal((15,3), mean=0.0, stddev=0.1)
# print(noise.eval())


import numpy as np

def split_stuff(x, idx):
    split_2, split_1 = x[idx:], x[:idx]
    del x
    return split_1, split_2

x = np.random.randint(10, size=[5,8])
print(x)
split_1, split_2 = split_stuff(x, 3)
x += 1
print(x)
del x
print("split_1:")
print(split_1)
print("split_2:")
print(split_2)