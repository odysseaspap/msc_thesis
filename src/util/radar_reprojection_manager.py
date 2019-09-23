import numpy as np
import os

from .dual_quaternion import transformations
import util.dataloading as dl
from scipy.sparse import csr_matrix

class RadarReprojectionManager:

    def __init__(self, original_resolution, projection_resolution, p, h_gt):
        self._orig_height = original_resolution[0]
        self._orig_width = original_resolution[1]
        self._proj_height = projection_resolution[0]
        self._proj_width = projection_resolution[1]
        self._P = p
        self._H_gt = h_gt

    def _convert_to_matrices(self, quaternions):
        inv_decalibs = []
        for quaternion in quaternions:
            rot_matrix = transformations.quaternion_matrix(quaternion)
            inv_decalibs.append(rot_matrix)
        return np.array(inv_decalibs)

    def _invert_homogeneous_matrix(self, mat):
        mant_inv = np.zeros_like(mat)
        mant_inv[:3,:3] = np.linalg.inv(mat[:3,:3])
        mant_inv[:3,3] = mat[:3,3] * -1.
        return mant_inv

    def _invert_homogeneous_matrices(self, mats):
        inverted_matrices = []
        for i in range(len(mats)):
            inverted_matrices.append(self._invert_homogeneous_matrix(mats[i]))
        return np.array(inverted_matrices)

    def _convert_to_quaternions(self, mats):
        quaternions = []
        for mat in mats:
            rotation = mat[:3, :3]
            quat = transformations.quaternion_from_matrix(rotation)
            quaternions.append(quat)
        return np.array(quaternions)

    def _valid_pixel_coordinates(self, u, v):
        return (u >= 0 and v >= 0 and v < self._proj_height and u < self._proj_width)

    def _compute_pixel_coordinates_and_invdepth(self, H, point):
        # Project point.
        tmp = np.matmul(H, point.transpose())
        plane_point = np.matmul(self._P, tmp)

        # Compute image coordinates.
        u = plane_point[0]/plane_point[2]
        v = plane_point[1]/plane_point[2]
        inv_depth = 1./plane_point[2]

        # Scale coordinates.
        v /= (self._orig_height / self._proj_height)
        u /= (self._orig_width / self._proj_width)
        return u, v, inv_depth

    def _project_radar_detections_sample(self, h_init, radar_detections):
        projection_image = np.zeros([self._proj_height, self._proj_width])
        for detection in radar_detections:
            u, v, inv_depth = self._compute_pixel_coordinates_and_invdepth(h_init, detection)
            u, v = int(u), int(v)
            if self._valid_pixel_coordinates(u, v):
                projection_image[v][u] = inv_depth

        projection_image = np.expand_dims(projection_image, axis=-1)
        return projection_image

    def _project_radar_detections(self, h_inits, radar_detections):
        projections = []
        for H_init, detections in zip(h_inits, radar_detections):
            projection_image = np.zeros([self._proj_height, self._proj_width])
            for detection in detections:
                u, v, inv_depth = self._compute_pixel_coordinates_and_invdepth(H_init, detection)
                u, v = int(u), int(v)
                if self._valid_pixel_coordinates(u, v):
                    projection_image[v][u] = inv_depth

            projection_image = np.expand_dims(projection_image, axis=-1)
            projections.append(projection_image)

        return np.array(projections)

    def _normalize_quaternions(self, quats):
        inv_mags = 1. / np.sqrt(np.sum(np.square(quats), axis=1))
        quats_normalized = np.transpose(np.transpose(quats) * inv_mags)
        return quats_normalized

    def compute_projections_and_labels(self, data, labels, radar_detections, model):
        # Fake model output.
        # fake_output = np.array([0.9998477, 0, 0.0174524, 0])
        # output_quats = np.tile(fake_output, (len(data[0]),1)) # Repeat output N times.

        # Generate model output.
        model_outputs = model.predict(data)
        output_quats = model_outputs[:,:4] # Crop noise output.
        output_quats = self._normalize_quaternions(output_quats)

        # Convert to matrices.
        inv_decalib = self._convert_to_matrices(labels)
        decalib = self._invert_homogeneous_matrices(inv_decalib)
        
        inv_decalib_hat = self._convert_to_matrices(output_quats)
        decalibs_hat = self._invert_homogeneous_matrices(inv_decalib_hat)

        # Recompute projections.
        tmp = np.matmul(decalib, self._H_gt)
        h_inits_new = np.matmul(inv_decalib_hat, tmp)
        projections_new = self._project_radar_detections(h_inits_new, radar_detections)
        # tmp = np.matmul(decalib, self._H_gt)
        # projections_new = self._project_radar_detections(tmp, radar_detections)

        # Compute new labels
        inv_decalibs_hat_hat = np.matmul(inv_decalib, decalibs_hat)
        labels_new = self._convert_to_quaternions(inv_decalibs_hat_hat)

        return projections_new, labels_new

    def get_corrected_projections_and_labels(self, sample_files_list, model):
        """
        Returns the new projections and residual labels after correction with given model
        """
        projections_new = np.empty((len(sample_files_list), self._proj_height, self._proj_width, 1))
        labels_new = np.empty((len(sample_files_list), 4))
        for i, ID in enumerate(sample_files_list):
            projections_new[i,], labels_new[i,] = self.compute_projection_and_label(str(ID), model)

        return projections_new, labels_new

    def get_corrected_projections_and_labels_and_save(self, sample_files_list, model, storage_folder):
        """
        Returns the new projections and residual labels after correction with given model and stores them as pairs in npz files in given folder
        """
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)

        projections_new, labels_new = self.get_corrected_projections_and_labels(sample_files_list, model)
        for i, ID in enumerate(sample_files_list):
            dl.save_augmented_projection_sample(str(storage_folder) + str(ID).split("/")[-1], projections_new[i,], labels_new[i,])

        return projections_new, labels_new
    
    def compute_and_save_corrected_projections_labels(self, sample_files_list, model, storage_folder):
        """
        Computes the new projections and residual labels with given model and stores pais in npz files in given folder

        Parameters
        ----------
        sample_files_list : list of strings
            List of absolute file names
        model : Keras model
            Model used to correct the samples
        storage_folder : string
            path to folder where new npz files are saved in
        """
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)

        for i, file_path in enumerate(sample_files_list):
            projections_new, labels_new = self.compute_projection_and_label(str(file_path), model)
            dl.save_augmented_projection_sample(str(storage_folder) + str(file_path).split("/")[-1], projections_new, labels_new)


    def compute_projection_and_label(self, sample_file, model):
        """
        Returns the projections corrected by the given model and the resulting label (decalibration)

        Parameters
        ----------
        sample_file : string
            Path to sample file
        model : Keras model
            Model used to correct the given sample

        Returns
        -------
        projection_new : ndarray
            New projected radar detections as input for model
        label_new : ndarray
            Resulting label: residual decalibration
        """
        img, proj_decalib, proj_gt, radar_detections, decalib, K, H_gt, dims = dl.load_complete_sample(sample_file)
        data = [img, proj_decalib]
        label = decalib[:4]
        self._H_gt = H_gt
        self._P = K
        self._orig_height = dims[0]
        self._orig_width = dims[1]
        data = dl.expand_input_data_batchdim(data)
        # Generate model output.
        model_output = model.predict(data)
        output_quats = model_output[:,:4] # Crop noise output.
        output_quats = self._normalize_quaternions(output_quats)

        # Convert to matrices.
        inv_decalib = transformations.quaternion_matrix(label)
        #inv_decalib = self._convert_to_matrices(label)
        decalib = self._invert_homogeneous_matrix(inv_decalib)
        
        inv_decalib_hat = transformations.quaternion_matrix(output_quats[0,:])
        decalibs_hat = self._invert_homogeneous_matrix(inv_decalib_hat)

        # Recompute projections.
        tmp = np.matmul(decalib, H_gt)
        h_inits_new = np.matmul(inv_decalib_hat, tmp)
        projection_new = self._project_radar_detections_sample(h_inits_new, radar_detections)
        # tmp = np.matmul(decalib, self._H_gt)
        # projections_new = self._project_radar_detections(tmp, radar_detections)

        # Compute new labels
        inv_decalibs_hat_hat = np.matmul(inv_decalib, decalibs_hat)
        rotation = inv_decalibs_hat_hat[:3, :3]
        label_new = transformations.quaternion_from_matrix(rotation)

        return projection_new, label_new

    def compute_projections_and_labels_static_decalib(self, data, labels, radar_detections, h_gt, Ks, orig_dims, model):
        """
        Returns the projections corrected by the given model and the resulting labels (decalibration) for a dataset with static decalibration.
        The model output is used as a correction for the next sample before prediction.

        Parameters
        ----------
        data : [ndarray, ndarray]
            Input data of network
        labels : ndarray
            Labels of data samples with shape: (#_of_samples, 4)
        radar_detections : list of ndarrays
            List of ndarrays with radar detections of each sample
        h_gt : ndarray
            H_gt matrices of all samples, ndarray shape (#_of_samples, 4, 4)
        Ks : ndarray
            K matrices of all samples ndarray shape (#_of_samples, 4, 3)
        orig_dims : ndarray
            original dimensions of sample images (#_of_samples, 2, 1)
        model : Keras model
            Model used to correct the given sample

        Returns
        -------
        projections_new : ndarray
            New projected radar detections as input for model
        labels_new : ndarray
            Resulting label: residual decalibration
        """
        
        # Fake model output.
        # fake_output = np.array([0.9998477, 0, 0.0174524, 0])
        # output_quats = np.tile(fake_output, (len(data[0]),1)) # Repeat output N times.
        projections_new = []
        labels_new = []
        inv_decalib_hat = None
        for input_img, input_radar, label, H_gt, K, dims, detections in zip(data[0], data[1], labels, h_gt, Ks, orig_dims, radar_detections):
            self._H_gt = H_gt
            self._P = K
            self._orig_height = dims[0]
            self._orig_width = dims[1]
            if inv_decalib_hat is not None:
                input_radar, label = self._correct_projections(label, inv_decalib_hat, H_gt, detections)
                # projection_corrected = self._project_radar_detections_sample(h_inits_new, detections)
            
            model_input = dl.expand_input_data_batchdim([input_img, input_radar])
            model_output = model.predict(model_input)

            output_quats = model_output[:,:4] # Crop noise output.
            output_quats = self._normalize_quaternions(output_quats)
            inv_decalib_hat = transformations.quaternion_matrix(output_quats[0,:])

            projection_new, label_new = self._correct_projections(label, inv_decalib_hat, H_gt, detections)            
            # # Convert to matrices.
            # inv_decalib = transformations.quaternion_matrix(label)
            # #inv_decalib = self._convert_to_matrices(label)
            # decalib = self._invert_homogeneous_matrix(inv_decalib)
        
            # decalibs_hat = self._invert_homogeneous_matrix(inv_decalib_hat)

            # # Recompute projections.
            # tmp = np.matmul(decalib, H_gt)
            # h_inits_new = np.matmul(inv_decalib_hat, tmp)
            # projection_new = self._project_radar_detections_sample(h_inits_new, detections)
            # # tmp = np.matmul(decalib, self._H_gt)
            #  # projections_new = self._project_radar_detections(tmp, radar_detections)

            # # Compute new labels
            # inv_decalibs_hat_hat = np.matmul(inv_decalib, decalibs_hat)
            # rotation = inv_decalibs_hat_hat[:3, :3]
            # label_new = transformations.quaternion_from_matrix(rotation)

            # Add to overall lists
            projections_new.append(projection_new)
            labels_new.append(label_new)
        
        projections_new = np.array(projections_new)
        labels_new = np.array(labels_new)

        return projections_new, labels_new

    def _correct_projections(self, label, inv_decalib_hat, H_gt, radar_detections):
        # Convert to matrices.
        inv_decalib = transformations.quaternion_matrix(label)
        decalib = self._invert_homogeneous_matrix(inv_decalib)
        decalibs_hat = self._invert_homogeneous_matrix(inv_decalib_hat)
         # Recompute projections.
        tmp = np.matmul(decalib, H_gt)
        h_inits_new = np.matmul(inv_decalib_hat, tmp)
        projection_corrected = self._project_radar_detections_sample(h_inits_new, radar_detections)

        # Compute new labels
        inv_decalibs_hat_hat = np.matmul(inv_decalib, decalibs_hat)
        rotation = inv_decalibs_hat_hat[:3, :3]
        label_new = transformations.quaternion_from_matrix(rotation)

        return projection_corrected, label_new


# if __name__ == '__main__': # For debugging.
#     proj = RadarReprojectionManager([], [100, 100], [5, 5], np.random.normal(size=[3,3]), np.random.normal(size=[3,3]))
