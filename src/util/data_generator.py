import numpy as np
import keras
from sklearn.utils import shuffle
#from scipy import sparse
import util.data_wrangling as dw
import util.dataloading as dl

class DataGenerator(keras.utils.Sequence):
    """
    DataGenerator class for batchwise loading of samples from the calibration data set.

    Parameters
    ----------
    sample_files_list: list
        list of all sample file names including .npz ending
    batch_size: int
        Batch size
    dim: tuple of ints
        Dimension of image / radar input data
    shuffle: boolean
        If true, the samples are shuffled for each epoch
    radar_input_factor : int
        The inverse depth of the projected radar detections gets multiplied by this factor for training improvement
    path_augmented_data : string
        Path to the folder with augmented projections and labels. if None, no augmented projections are used.
    """

    def __init__(self, sample_files_list, batch_size=32, dim=(150,240), radar_input_factor=1, shuffle=True, path_augmented_data=None):
        print("Initializing DataGenerator...")
        self.sample_files_list = sample_files_list
        self.batch_size = batch_size
        self.dim = dim
        self.image_channels = 3
        self.radar_channels = 1
        self.shuffle = shuffle
        self._shuffled_indices = list(range(len(self.sample_files_list)))
        self.radar_input_factor = 1
        self.path_augmented_data = path_augmented_data
        self.use_augmented_data = False
        if (path_augmented_data != None):
            self.use_augmented_data = True
            print("DataGenerator will use augmented projections from folder {}.".format(str(self.path_augmented_data)))
        
        print("DataGenerator initialized.")

    def __len__(self):
        """
        Computes the number of batches per epoch.
        """
        return int(np.floor(len(self.sample_files_list) / self.batch_size)) # model sees training samples at most once per epoch

    def __getitem__(self, index):
        X, y = self._prepare_batch(index)
        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            self._shuffled_indices = shuffle(self._shuffled_indices)

    def _prepare_batch(self, batch_idx):
        # Compute batch boundaries.
        batch_start = batch_idx * self.batch_size
        batch_end  = batch_start + self.batch_size
        batch_indices = self._shuffled_indices[batch_start:batch_end]
        # Get IDs of batch samples
        list_IDs_batch = [self.sample_files_list[k] for k in batch_indices]
        # Initialization
        batch_rgb_input = np.empty((self.batch_size, *self.dim, self.image_channels))
        batch_radar_input = np.empty((self.batch_size, *self.dim, self.radar_channels))
        batch_labels = np.empty((self.batch_size, 4))
        # Load and augment data samples
        for i, ID in enumerate(list_IDs_batch):
            [batch_rgb_input[i,], batch_radar_input[i,]], batch_labels[i,] = dl.load_radnet_training_sample(str(ID))
            # exchange radar input and label in case augmented data has to be used
            if self.use_augmented_data == True:
                projection_aug, label_aug = dl.load_augmented_projection_sample(self.path_augmented_data + str(ID).split("/")[-1])
                batch_radar_input[i,] = projection_aug
                batch_labels[i,] = label_aug
            # [batch_rgb_input[i,], batch_radar_input[i,]], batch_labels[i,] = self._augment_data(sample["rgb_image"], sample["radar_detections"], sample["K"], sample["H_gt"], sample["rgb_image_orig_dim"][0], sample["rgb_image_orig_dim"][1])
            # increase stored depth values in radar input for better training performance
            batch_radar_input[i,] = batch_radar_input[i,] * self.radar_input_factor
        dw.standardize_images(batch_rgb_input)
        assert(len(batch_rgb_input) == len(batch_labels))
        assert(len(batch_radar_input) == len(batch_labels))
        return [batch_rgb_input, batch_radar_input], batch_labels

    # def _augment_data(self, rgb_image, radar_detections, K, H_gt, image_original_height, image_original_width):
    #     """
    #     Returns augmented data sample: create decalibration and transform radar detections

    #     A augmented data sample is only accepted if the minimum number of projections is reached. Otherwise
    #     it will retry with a different decalibration up to 10 times. Afterwards, no decalibration is used to 
    #     create a valid sample.

    #     Parameters
    #     ----------
    #     rgb_image : ndarray
    #         rgb image of data sample
    #     radar_detections : ndarray 
    #         with shape (4,1,number_of_detections) containing position vectors of all detections in radar coordinate frame
    #     K : ndarray
    #         intrinsic calibration matrix of camera with shape (3,4)
    #     H_gt : ndarray
    #         Homogeneous transformation matrix of ground truth calibration between camera and radar
    #     image_original_height : int
    #         Height in pixel of original image (necessary for transformation with K)
    #     image_original_width : int
    #         Width in pixel of original image (necessary for trasformation with K)

    #     Returns
    #     -------
    #     [rgb_image, radar_input] : [ndarray, ndarray]
    #         touple of rgb image and projected radar detections (both the same shape)
    #     inv_decalibration_matrix : ndarray
    #         inverse decalibration matrix as homogeneous transformation matrix
    #     """
    #     rgb_input = rgb_image
    #     image_height = rgb_image.shape[0]
    #     image_width = rgb_image.shape[1]
        
    #     counter_points = 0 # count number of projected vehicles
    #     retry = 0 # counter of retry if too less projections available
    #     while(counter_points < self.min_number_projections):
    #         if retry < 10: # retry up to 10 times until minimum required projections reached, then use no decalibration
    #             decalibration_matrix = create_decalib_transformation(self.angle_std_deviation, self.translation_std_deviation)
    #         else:
    #             decalibration_matrix = np.identity(4)
                
    #         radar_input = np.zeros(image_height, image_width)
    #         counter_points = 0
    #         for index in range(radar_detections.shape[2]):
    #             point = radar_detections[:,:,index]
    #             (u, v, inv_depth) = self._comp_uv_invdepth(K, H_gt, decalibration_matrix, point)
    #             if (u, v, inv_depth) != None:
    #                 v /= (image_original_height/image_height)
    #                 u /= (image_original_width/image_width)
    #                 v, u = int(v), int(u)
    #                 if self._valid_pixel_coordinates(u, v, image_height, image_width):
    #                     radar_input[v][u] = inv_depth
    #                     counter_points += 1
    #         retry += 1

    #     inv_decalibration_matrix = invert_homogeneous_matrix(decalibration_matrix)
    #     return [rgb_input, radar_input], inv_decalibration_matrix


    # def _comp_uv_invdepth(self, K, h_gt, decalib, point):
    #     '''
    #     Compute pixels coordinates and inverted radar depth.
    #     '''
    #     # Project on image plane with: z * (u, v, 1)^T = K * H * x
    #     tmp = np.matmul(K, decalib)
    #     tmp = np.matmul(tmp, h_gt)
    #     point = np.matmul(tmp, point.transpose())
    #     if point[2] != 0:
    #         # return np.array([int(point[0]/point[2]), int(point[1]/point[2]), 1./point[2]])
    #         return [point[0]/point[2], point[1]/point[2], 1./point[2]]
    #     else:
    #         return None

    # def _valid_pixel_coordinates(self, u, v, IMAGE_HEIGHT, IMAGE_WIDTH):
    #     """
    #     Checks whether the provided pixel coordinates are valid.
    #     """
    #     return (u >= 0 and v >= 0 and v < IMAGE_HEIGHT and u < IMAGE_WIDTH)

# def __init__(self, sample_files_list, angle_std_deviation, translation_std_deviation, min_number_projections, batch_size=32, dim=(150,240), shuffle=True):
#         self.sample_files_list = sample_files_list
#         self.batch_size = batch_size
#         self.dim = dim
#         self.image_channels = 3
#         self.radar_channels = 1
#         self.angle_std_deviation = angle_std_deviation
#         self.translation_std_deviation = translation_std_deviation
#         self.min_number_projections = min_number_projections # minimum number of radar projections required to use sample
#         self.shuffle = shuffle
#         self._shuffled_indices = list(range(len(self.sample_files_list)))


# class DataGenerator(keras.utils.Sequence):

#     def __init__(self, data, labels, batch_size=32, shuffle=True):
#         self.data_1 = data[0]
#         self.data_2 = data[1]
#         self.labels = labels
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         assert(len(self.data_1) == len(self.labels))
#         assert(len(self.data_2) == len(self.labels))
#         self._shuffled_indices = list(range(len(self.labels)))

#     def __len__(self):
#         """
#         Computes the number of batches per epoch.
#         """
#         return int(np.floor(len(self.labels) / self.batch_size)) # model sees training samples at most once per epoch

#     def __getitem__(self, index):
#         X, y = self._prepare_batch(index)
#         return X, y

#     def on_epoch_end(self):
#         if self.shuffle == True:
#             self._shuffled_indices = shuffle(self._shuffled_indices)

#     def _prepare_batch(self, batch_idx):
#         # Compute batch boundaries.
#         batch_start = batch_idx * self.batch_size
#         batch_end  = batch_start + self.batch_size
#         batch_indices = self._shuffled_indices[batch_start:batch_end]

#         # Get batch.
#         batch_rgb_input = np.take(self.data_1, batch_indices, axis=0)
#         batch_radar_input = np.take(self.data_2, batch_indices, axis=0)
#         batch_labels = np.take(self.labels, batch_indices, axis=0)
#         # Augment data.
#         standardize_images(batch_rgb_input)
#         return [batch_rgb_input, batch_radar_input], batch_labels