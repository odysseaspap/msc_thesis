import tensorflow as tf
import keras
from keras import backend, optimizers
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Dense, Input, Flatten, MaxPooling2D, Conv2D
from keras.layers import concatenate
from keras.models import Model, load_model  # basic class for specifying and training a neural network
#from keras.utils import plot_model
from scipy import ndimage
from keras.utils.vis_utils import plot_model

from keras.applications import mobilenet
from mlpconv_layer import *


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
            predicted_decalib_transform = self._calibration_block(rgb_stream_out, radar_stream_out)

        with tf.name_scope('se3_block'):
            k_mat = Input(shape=(3, 4))
            output = self._se3_block(predicted_decalib_transform, k_mat)

        # Compose model.
        return Model(inputs=[rgb_input, radar_input, k_mat], outputs=output)

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
            predicted_decalib_transform = Dense(4, activation='linear', kernel_initializer=self._weight_init, bias_initializer=self._bias_init)(fc_2)
        return predicted_decalib_transform

    def _se3_block(self, predicted_decalib_transform, k_mat):
        """
        TODO: Modify the following operations from CalibNet
        # se(3) -> SE(3) for the whole batch
        predicted_transforms = tf.map_fn(lambda x:exponential_map_single(output_vectors[x]), elems=tf.range(0, batch_size, 1), dtype=tf.float32)

        # transforms depth maps by the predicted transformation
        depth_maps_predicted, cloud_pred = tf.map_fn(lambda x:at3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, predicted_transforms[x], K_final, small_transform), elems = tf.range(0, batch_size, 1), dtype = (tf.float32, tf.float32))

        # transforms depth maps by the expected transformation
        depth_maps_expected, cloud_exp = tf.map_fn(lambda x:at3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, expected_transforms[x], K_
        """
        output = predicted_decalib_transform

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
