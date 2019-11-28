import keras
import keras.backend as K
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU

class RunConfig:
    def __init__(self):
        self.lr = 0.002
        self.epochs = 32
        self.batch_size = 16
        self.val_split = 0.1
        self.original_resolution = [900, 1600, 3] #[1200, 1920, 3]
        self.input_shape = [150, 240, 3]

        # Increasing the inverse depth of the radar projections with this factor
        self.radar_input_factor = 1

        # Decalibration params.
        self.rotation_std_dev = 10 # in degree
        self.translation_std_dev = 0.0 # in meter
        self.min_number_projections = 10

        # Network params.
        self.length_error_weight = 0.005

        self.photometric_loss_factor = 1.0
        self.point_cloud_loss_factor = 1.0

        self.drop_rate = 0.5
        self.l2_reg = 0.00 #0.002
        #self.weight_init = 'glorot_normal'
        #self.mid_layer_activations = 'relu'
        self.weight_init = keras.initializers.Orthogonal()
        self.mid_layer_activations = PReLU
        # self.mid_layer_activations = ELU
        # self.mid_layer_activations = LeakyReLU

        # Data.
        #Gcloud folders
        self.general_dataset_folder_path = "/home/jupyter/thesis/data/sets/nuscenes_RADNET/"
        self.dataset_names = ["nuscenes_01_04_RADNET","nuscenes_05_RADNET","nuscenes_06_RADNET","nuscenes_07_RADNET","nuscenes_08_RADNET","nuscenes_09_RADNET"]

        # Providentia data
        #self.general_dataset_folder_path = "/home/jupyter/thesis/data/providentia/"
        #self.dataset_names = ["mp10_near_bag_00_30","mp10_near_bag_31_40","mp10_near_bag_41_50","mp10_near_bag_51_60","mp10_near_bag_61_70","mp10_near_bag_76_78"]
        
        #Local PC folders
        #self.general_dataset_folder_path = "/home/odysseas/thesis/data/sets/other/nuscenes_RADNET_mini_stored_depth/"
        #self.dataset_names = [""]
        
        # Datasets for testing
        # self.dataset_names = ["mp10_near_bag_71_75"]
        # self.dataset_names = ["testset"]

        self.dataset_paths = []
        for dataset in self.dataset_names:
            self.dataset_paths.append(str(self.general_dataset_folder_path + dataset + "/samples/"))
