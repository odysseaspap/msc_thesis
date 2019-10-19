import tensorflow as tf
from util import quaternion_ops as qt_ops
import keras
from keras import backend as K, optimizers
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Dense, Input, Flatten, MaxPooling2D, Conv2D, Lambda
from keras.layers import concatenate
from keras.models import Model, load_model  # basic class for specifying and training a neural network
#from keras.utils import plot_model
from scipy import ndimage
from keras.utils.vis_utils import plot_model

from keras.applications import mobilenet
from mlpconv_layer import *
from util import all_transformer as at3
import custom_loss_functions as loss_fn


class RadNet:

    def __init__(self, input_shape, weight_init='glorot_normal', drop_rate=0.0, l2_reg=0.0, mid_layer_activations='relu'):
        # Input dimensions.
        self._rgb_shape = input_shape
        self._radar_shape = input_shape[:]
        self._radar_shape[-1] = 1 # 1 radar channel
        # Initialization.
        self._weight_init = weight_init
        self._bias_init = keras.initializers.Constant(0.01)
        # Regularization.
        self._drop_rate = drop_rate
        self._l2_reg = keras.regularizers.l2(l2_reg)
        self._ls_bias_reg = None
        # Activation functions.
        self._mid_layer_activation_type = mid_layer_activations
        self._model = self._build_model()
    
    def _get_activation_instance(self):
        """
        Instantiates activation object if not provided as a string. Workaround for keras.
        """
        return self._mid_layer_activation_type if type(self._mid_layer_activation_type) == str else self._mid_layer_activation_type()

    def _build_model(self):

        with tf.name_scope('rgb_stream'):
            rgb_input = Input(shape=self._rgb_shape)
            rgb_stream_out = self._rgb_stream(rgb_input)

        with tf.name_scope('radar_stream'):
            radar_input = Input(shape=self._radar_shape)
            radar_stream_out = MaxPooling2D(pool_size=(4,4))(radar_input)
        
        with tf.name_scope('calibration_block'):
            predicted_decalib_quat = self._calibration_block(rgb_stream_out, radar_stream_out)

        with tf.name_scope('se3_block'):
            # has to take as argument either this or radar_stream_out (the max pooled version)
            # radar_input = Input(shape=self._radar_shape)
            k_mat = Input(shape=(3, 3))
            decalib_gt_trans = Input(shape=(3, 1))
            output = self._se3_block(predicted_decalib_quat, radar_input, k_mat, decalib_gt_trans)

        # Compose model.
        return Model(inputs=[rgb_input, radar_input, k_mat, decalib_gt_trans], outputs= output[0] )

    def _rgb_stream(self, rgb_input):
        pretrained_out = self._pretrained_block(rgb_input)
        nin_1 = MlpConv(pretrained_out, filter_maps=16, kernel_size=(5, 5), activation=self._mid_layer_activation_type, kernel_initializer=self._weight_init, bias_initializer=self._bias_init)
        nin_2 = MlpConv(nin_1.output, filter_maps=16, kernel_size=(5, 5), activation=self._mid_layer_activation_type, kernel_initializer=self._weight_init, bias_initializer=self._bias_init)
        return nin_2.output

    def _pretrained_block(self, rgb_input):
        rgb_mobile = self._load_mobilenet(rgb_input)
        rgb_mobile_out = rgb_mobile.output
        return rgb_mobile_out

    def _load_mobilenet(self, input_tensor):
        mobilenet_model = mobilenet.MobileNet(input_tensor=input_tensor, weights='imagenet')#, alpha=0.75)
        cropped_model = Model(inputs=mobilenet_model.input, outputs=mobilenet_model.get_layer(index=20).output)
        #mobilenet_model.layers = mobilenet_model.layers[0:22] # Crop model.
        #mobilenet_model.outputs = [mobilenet_model.layers[-1].output]
        # mobilenet_model.outputs = [mobilenet_model.layers[21].output]
        # mobilenet_model.layers[0].inbound_nodes = [] # Cut inbound node connections.
        # mobilenet_model.layers[21].outbound_nodes = [] # Cut outbound node connections.
        # #mobilenet_model.layers[-1].outbound_nodes = [] # Cut outbound node connections.
        cropped_model.layers[0].inbound_nodes = []
        cropped_model.layers[-1].outbound_nodes = []
        return cropped_model

    def _calibration_block(self, input_rgb, input_radar):
        with tf.name_scope('rgb_compression'):
            rgb_flatten = Flatten()(input_rgb)
            rgb_dense_1 = Dense(50, activation=self._get_activation_instance(), kernel_initializer=self._weight_init, bias_initializer=self._bias_init, kernel_regularizer=self._l2_reg, bias_regularizer=self._ls_bias_reg)(rgb_flatten)

        with tf.name_scope('radar_compression'):
            radar_flatten = Flatten()(input_radar)
            radar_dense_1 = Dense(50, activation=self._get_activation_instance(), kernel_initializer=self._weight_init, bias_initializer=self._bias_init, kernel_regularizer=self._l2_reg, bias_regularizer=self._ls_bias_reg)(radar_flatten)

        with tf.name_scope('transformation_regression'):
            concatenated = concatenate([rgb_dense_1, radar_dense_1])
            # drop_0 = keras.layers.Dropout(0.2)(concatenated)
            fc_1 = Dense(512, activation=self._get_activation_instance(), kernel_initializer=self._weight_init, bias_initializer=self._bias_init, kernel_regularizer=self._l2_reg, bias_regularizer=self._ls_bias_reg)(concatenated)
            drop_1 = keras.layers.Dropout(self._drop_rate)(fc_1)
            fc_2 = Dense(256, activation=self._get_activation_instance(), kernel_initializer=self._weight_init, bias_initializer=self._bias_init, kernel_regularizer=self._l2_reg, bias_regularizer=self._ls_bias_reg)(drop_1)
            #gaussian_noise_1 = keras.layers.GaussianNoise(1e-05)(fc_2)
            predicted_decalib_quat = Dense(4, activation='linear', kernel_initializer=self._weight_init, bias_initializer=self._bias_init)(fc_2)
        return predicted_decalib_quat

    def _spatial_transformer_layers(self, input_list):
        """
        Creates the predicted transform matrix from the predicted quaternion and the ground truth translation decalib vector.
        Then, it applies the transform to the depth image (radar_input) and extracts the predicted depth map and the
        predicted radar point cloud via a 3D Sampling Grid and Bilinear Interpolation.

        :param input_list: [predicted_decalib_quat, radar_input, k_mat, decalib_gt_trans] =
        [(batch_size, 4, 1), (batch_size, 150, 240, 1), (batch_size, 3, 3), (batch_size, 3, 1)]
        :return: List with [predicted_depth_map, cloud_pred, radar_input]

        """
        # TODO: Modify the following operations from CalibNet
        predicted_decalib_quat = input_list[0]
        radar_input = input_list[1]
        k_mat = input_list[2]
        decalib_gt_trans = input_list[3]

        # se(3) -> SE(3) (for the whole batch)
        # Create augmented transform matrix from predicted quaternion and ground truth translation vector
        predicted_transform_augm = qt_ops.transform_from_quat_and_trans(predicted_decalib_quat, decalib_gt_trans)

        # transforms depth maps by the predicted transformation
        batch_size = tf.shape(radar_input)[0]
        #print(batch_size)
        depth_maps_predicted, cloud_pred = tf.map_fn(lambda x:at3._simple_transformer(radar_input[x,:,:,0], predicted_transform_augm[x], k_mat[x]), elems = tf.range(0, batch_size, 1), dtype = (tf.float32, tf.float32))
        #depth_maps_pred, cloud_pred = Lambda(at3._simple_transformer(radar_input, predicted_transform_augm, k_mat))

        # transforms depth maps by the expected transformation
        # depth_maps_expected, cloud_exp = tf.map_fn(lambda x:at3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, expected_transforms[x], K_

        #Return radar input to use it in loss function

        #return [predicted_decalib_quat, depth_maps_predicted, cloud_pred, radar_input, k_mat]

        return [predicted_decalib_quat, depth_maps_predicted, cloud_pred, radar_input, k_mat]


    # TODO: Because using multiple outputs on a single loss function is a hurdle/impossible in keras, try the following:
    # create the loss here and get the result in a Tensor. Then, add the loss in the model
    # before returning via model.add_loss()

    def _photometric_and_3d_pointcloud_loss(self, y_true, y_pred, alpha, beta):
        """
        :param y_true: (batch_size, 7, 1) -  ground truth de-calibration
        quaternion (indexes 0-3) and translation (indexes 4-6) vectors
        :param y_pred: Output of RadNet++  [(batch_size, H, W), (batch_size, num_points, 3), (batch_size, H, W, 1)]
        [depth_map_predicted, cloud_predicted, radar_input]
        :param alpha: float value - photometric loss weight
        :param beta: float value - 3D point cloud loss weight

        :return: float value - Final loss value
        """
        print(K.int_shape(y_pred))
        print(K.int_shape(y_true))
        depth_maps_predicted = y_pred[0]
        print(K.int_shape(depth_maps_predicted))
        cloud_pred = y_pred[2]
        radar_input = y_pred[3]
        k_mat = y_pred[4]

        quat_expected = y_true[:, 4]
        print(K.int_shape(quat_expected))
        quat_expected = tf.convert_to_tensor(quat_expected)
        trans_expected = y_true[:, 4:]
        trans_expected = tf.convert_to_tensor(trans_expected)

        # T_expected = qt_ops.transform_from_quat_and_trans(quat_expected, trans_expected)
        batch_size = tf.shape(radar_input)[0]
        T_expected = tf.identity((batch_size, 4, 4))
        depth_maps_expected, cloud_exp = tf.map_fn(
            lambda x: at3._simple_transformer(radar_input[x, :, :, 0], T_expected[x], k_mat[x]),
            elems=tf.range(0, batch_size, 1), dtype=(tf.float32, tf.float32))

        # photometric loss between predicted and expected transformation
        # Note that here they have to re-normalize the depth maps since they de-normalized them before the ST layers!!!
        # plus, they measure the photometric loss only in a 10x10 area in the center of the image!!!
        photometric_loss = tf.nn.l2_loss(
            tf.subtract(depth_maps_expected[:, 10:-10, 10:-10], depth_maps_predicted[:, 10:-10, 10:-10]))

        # earth mover's distance between point clouds
        # cloud_loss = model_utils.get_emd_loss(cloud_pred, cloud_exp)

        # final loss term
        predicted_loss_train = alpha * photometric_loss  # + beta * cloud_loss

        return predicted_loss_train


    def _se3_block(self, predicted_decalib_quat, radar_input, k_mat, decalib_gt_trans):
        # TODO: Check if I should create a new custom Layer instead of using Lambda
        # The below has 0 trainable parameters - is this correct?
        # From gvnn paper pg10 (where CalibNet took 3D ST layers from) :
        """
        In this work, we bridge the gap between learning and geometry
        based methods with our 3D Spatial Transformer module which explicitly defines
        these operations as layers that act as computational blocks with no learning
        parameters but allow backpropagation from the cost function to the input layers
        """
        # https://github.com/keras-team/keras/issues/7078
        output = Lambda(self._spatial_transformer_layers)([predicted_decalib_quat, radar_input, k_mat, decalib_gt_trans])

        return output

    @property
    def model(self):
        return self._model



from keras.layers import PReLU
if __name__ == '__main__': # For debugging.
    input_shape = [150, 240, 3]
    rad_net = RadNet(input_shape, weight_init=keras.initializers.Orthogonal(), mid_layer_activations=PReLU)
    print(rad_net.model.summary())
    plot_model(rad_net.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
