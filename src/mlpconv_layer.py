import keras
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Dropout, Activation, Flatten
from keras.utils import np_utils
import tensorflow as tf

class MlpConv:
    """
    MlpConv Layer according to NiN Paper by Lin et al.
    """
    def __init__(self, x, filter_maps, kernel_size, activation, strides=(1,1), kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        self._kernel_x, self._kernel_y = kernel_size
        self._activation = activation
        self._strides = strides
        self._kernel_init = kernel_initializer
        self._bias_init = bias_initializer
        
        with tf.name_scope('mlpconv'):
            self.conv_1 = Convolution2D(filter_maps, (self._kernel_x, self._kernel_y), strides=self._strides, padding='same', activation=self._get_activation(), kernel_initializer=self._kernel_init, bias_initializer=self._bias_init)(x)
            self.conv_2 = Convolution2D(filter_maps, (1, 1), padding='same', activation=self._get_activation(), kernel_initializer=self._kernel_init, bias_initializer=self._bias_init)(self.conv_1)
            self.y = Convolution2D(filter_maps, (1, 1), padding='same', activation=self._get_activation(), kernel_initializer=self._kernel_init, bias_initializer=self._bias_init)(self.conv_2)

    def _get_activation(self):
        return self._activation if type(self._activation) == str else self._activation()

    @property
    def output(self):
        return self.y