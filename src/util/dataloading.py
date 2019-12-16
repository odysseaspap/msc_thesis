import numpy as np
from scipy import sparse

def load_np_file(file_path):
    """
    Returns instance of NpzFile class (should be closed after usage)
    """
    return np.load(file_path, encoding='latin1', allow_pickle = True)

def load_nparray_from_file(file_path, key):
    """
    Returns nparray stored with key in file at given path
    """
    with load_np_file(file_path) as npfile:
        data = npfile[key]
    return data

def load_nparrays_from_file(file_path, keys):
    """
    Returns list of nparrays stored with given keys in file at given path
    """
    data = []
    with load_np_file(file_path) as npfile:
        for key in keys:
            data.append(npfile[key])
    return data

def load_radnet_training_sample(sample_file):
    """
    Returns content of calibration data sample 

    Parameters
    ----------
    sample_file : string
        Path to sample file

    Returns
    -------
    [rgb_image, radar_input] : [ndarray, ndarray]
        touple of rgb image and projected radar detections
    label : ndarray
        Quaternions of inverse decalibration rotation
    """
    with load_np_file(sample_file) as sample:
        rgb_image = sample["rgb_image"]
        radar_input = get_projections_from_npz_file(sample, "projections_decalib")
        label = sample["decalib"][:4] # crop translation from decalibration

    return [rgb_image, radar_input], label


def load_radnet_training_sample_with_intrinsics_gt_decalib(sample_file):
    """
    Returns content of calibration data sample

    Parameters
    ----------
    sample_file : string
        Path to sample file

    Returns
    -------
    [rgb_image, radar_input, k_mat, label] : [ndarray, ndarray, ndarray, ndarray]
        tuple of rgb image, projected radar detections, camera intrinsics matrix
        and ground truth quaternion and translation inverse decalibration
    label : ndarray
        7x1 vector: 1x4 Quaternions of inverse decalibration rotation
        and 1x3 translation vector which represents ground truth (inverse)
        translation decalibration
    """
    with load_np_file(sample_file) as sample:
        rgb_image = sample["rgb_image"]
        radar_input = get_projections_from_npz_file(sample, "projections_decalib")
        k_mat = sample["K"] #[:, :3]

        trans_label = sample["decalib"][4:]
        label = sample["decalib"]

    return [rgb_image, radar_input, k_mat, trans_label], label

def load_data_from_samples(dataset_path, file_list, keys):
    """
    Returns a list of ndarrays corresponding to the list of keys
    """
    data = []
    for i in range(len(keys)):
        data[i] = []
    for i, file_name in enumerate(file_list):
        sample_data = load_nparrays_from_file(str(dataset_path)+str(file_name), keys)
        for i in range(len(sample_data)):
            data[i].append(sample_data[i])
    
    for i in range(len(data)):
        data[i] = np.array(data[i])
    return data

def get_csr_matrix_from_npz_file(file, key):
    return np.expand_dims(file[key], axis=-1)[0] # expand dimensions to access csr matrix

def get_projections_from_npz_file(file, key):
    """
    Returns dense matrix of projections with channel size 1 stored as sparse CSR matrix

    Parameters
    ----------
    file : NpzFile
        file object of npz file
    key : string
        key of the projections stored as ndarray of csr matrix

    Returns
    -------
    projections : ndarray
        array of projections with extended last dimension as channel size 1
    """
    csr = get_csr_matrix_from_npz_file(file, key)
    dense_matrix = sparse.csr_matrix.todense(csr)
    return np.expand_dims(dense_matrix, axis=-1) # expand dimensions for channels size of 1

def load_radnet_training_sample_batchdim(sample_file):
    """
    Returns content of calibration data sample with additional batch dimension at axis 0 of data for direct model input

    Parameters
    ----------
    sample_file : string
        Path to sample file

    Returns
    -------
    [rgb_image, radar_input] : [ndarray, ndarray]
        touple of rgb image and projected radar detections
    label : ndarray
        Quaternions of inverse decalibration rotation
    """
    data, label = load_radnet_training_sample(sample_file)
    # Expand dimensions to account for expected batch dimension
    data = expand_input_data_batchdim(data)
    return [data[0], data[1]], label

def expand_input_data_batchdim(data):
    for i in range(len(data)):
        data[i] = np.expand_dims(data[i], axis=0)
    return data

def load_complete_sample(sample_file):
    """
    Returns complete content of sample file

    Returns
    -------
    rgb_image : ndarray (height, width, 3)

    projections_decalib : ndarray (height, width, 1)(with channel size 1)

    projections_gt : ndarray (height, width, 1)(with channel size 1)

    radar_detections : ndarray (4, 1, number_of_detections)

    decalib : ndarray (7,)
        inverted transformation between camera and radar rotation as quaternions (idx 0-3) and translation concatinated (4-6) 
    K : ndarray (3, 4)

    H_gt : ndarray (4, 4)

    rgb_img_orig_dim : ndarray (2, 1)

    """
    with load_np_file(sample_file) as sample:
        rgb_image = sample["rgb_image"]
        projections_decalib = get_projections_from_npz_file(sample, "projections_decalib")
        projections_gt = get_projections_from_npz_file(sample, "projections_groundtruth")
        radar_detections = sample["radar_detections"]
        decalib = sample["decalib"]
        K = sample["K"]
        H_gt = sample["H_gt"]
        rgb_img_orig_dim = sample["rgb_image_orig_dim"]

    return rgb_image, projections_decalib, projections_gt, radar_detections, decalib, K, H_gt, rgb_img_orig_dim

def save_augmented_projection_sample(abs_file_path, projections, label):
    """
    Savez projections and corresponding label into npz file

    Parameters
    ----------
    abs_file_path : string
        Absolute path with filename
    projections : ndarray
        Projections in dense format
    label : ndarray
        Corresponding label (decalibration)
    """
    if len(projections.shape) == 3:
        projections = projections[:,:,0]

    projections = sparse.csr_matrix(projections)
    np.savez_compressed(str(abs_file_path), projections=projections, label=label)

def load_augmented_projection_sample(abs_file_name):
    """
    Returns projections and corresponding label of augmented sample file

    Returns
    -------
    projections : ndarray
        array of projections with extended last dimension as channel size 1
    label : ndarray
        Corresponding label
    
    Parameters
    ----------
    abs_file_name : string
        Absolute path to file including ending
    """
    with load_np_file(abs_file_name) as sample:
        label = sample["label"]
        projections = get_projections_from_npz_file(sample, "projections")

    return projections, label

def load_dataset(file_list):
    """
    Returns a complete dataset of the given sample file list

    Returns
    -------
    images : ndarray

    projections_decalibrated : ndarray

    projections_groundtruth : ndarray

    decalibs : ndarray

    H_gts : ndarray

    Ks : ndarray

    radar_detections : ndarray
    
    dims : ndarray
    """
    images = []
    projections_decalibrated = []
    projections_groundtruth = []
    decalibs = []
    H_gts = []
    Ks = []
    radar_detections = []
    dims = []
    for file_name in file_list:
        img, proj_decalib, proj_gt, radar_detection, decalib, K, H_gt, dim = load_complete_sample(file_name)
        images.append(img)
        projections_decalibrated.append(proj_decalib)
        projections_groundtruth.append(proj_gt)
        decalibs.append(decalib)
        H_gts.append(H_gt)
        Ks.append(K)
        radar_detections.append(radar_detection)
        dims.append(dim)
    
    images = np.array(images)
    projections_decalibrated = np.array(projections_decalibrated)
    projections_groundtruth = np.array(projections_groundtruth)
    decalibs = np.array(decalibs)
    H_gts = np.array(H_gts)
    Ks = np.array(Ks)
    radar_detections = np.array(radar_detections)
    dims = np.array(dims)
    return images, projections_decalibrated, projections_groundtruth, decalibs, H_gts, Ks, radar_detections, dims
