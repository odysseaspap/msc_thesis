import keras
import keras.backend as K

# class LearningRateLossCallback(keras.callbacks.Callback):

#     # self.model
#     # self.param

#     def on_epoch_begin(self, epoch, logs=None):
#         return

#     def on_epoch_end(self, epoch, logs=None):
#         return

#     def on_batch_begin(self, batch, logs=None):
#         lr = K.get_value(self.model.optimizer.lr)
#         print(lr)

#     def on_batch_end(self, batch, logs=None):
#         return

#     def on_train_begin(self, logs=None):
#         return

#     def on_train_end(self, logs=None):
#         return


class SGDLearningRateTracker(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        # current_lr = self._get_current_lr()
        # print("\nbeing_lr: " + str(current_lr))
        return

    def on_epoch_end(self, epoch, logs={}):
        current_lr = self._get_current_lr()
        print("\nend_lr: " + str(current_lr))
        
    def _get_current_lr(self):
        optimizer = self.model.optimizer
        lr, decay, iterations = self._get_param_values([optimizer.lr, optimizer.decay, optimizer.iterations])
        return (lr * (1. / (1. + decay * iterations)))

    def _get_param_values(self, params):
        return [K.get_value(param) for param in params]
