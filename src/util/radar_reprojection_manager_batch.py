import numpy as np

from .dual_quaternion import transformations

class RadarBatchReprojectionManager:

    def __init__(self, original_resolution, projection_resolution):
        self._orig_height = original_resolution[0]
        self._orig_width = original_resolution[1]
        self._proj_height = projection_resolution[0]
        self._proj_width = projection_resolution[1]

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

    def _compute_pixel_coordinates_and_depth(self, H, K, point):
        # Project point.
        tmp = np.matmul(H, point.transpose())
        plane_point = np.matmul(K, tmp)

        # Compute image coordinates.
        u = plane_point[0]/plane_point[2]
        v = plane_point[1]/plane_point[2]
        depth = plane_point[2]

        # Scale coordinates.
        v /= (self._orig_height / self._proj_height)
        u /= (self._orig_width / self._proj_width)
        return u, v, depth

    def _project_radar_detections(self, h_inits, Ks, radar_detections):
        projections = []
        for H_init, K, detections in zip(h_inits, Ks, radar_detections):

            projection_image = np.zeros([self._proj_height, self._proj_width])
            for detection in detections:
                u, v, depth = self._compute_pixel_coordinates_and_depth(H_init, K, detection)
                u, v = int(u), int(v)
                if self._valid_pixel_coordinates(u, v):
                    projection_image[v][u] = depth

            projection_image = np.expand_dims(projection_image, axis=-1)
            projections.append(projection_image)

        return np.array(projections)

    def _normalize_quaternions(self, quats):
        inv_mags = 1. / np.sqrt(np.sum(np.square(quats), axis=1))
        quats_normalized = np.transpose(np.transpose(quats) * inv_mags)
        return quats_normalized

    def compute_projections_and_labels(self, data, labels, radar_detections, h_gt, Ks, model):
        """
        Returns the projections corrected by the given model and the resulting label (decalibration)

        Parameters
        ----------
        data : [ndarray, ndarray]
            Input data of network
        labels : ndarray
            Labels of data samples with shape: (#_of_samples, 1)
        radar_detections : list of ndarrays
            List of ndarrays with radar detections of each sample
        h_gt : ndarray
            H_gt matrices of all samples, ndarray shape (#_of_samples, 4, 4)
        Ks : ndarray
            K matrices of all samples ndarray shape (#_of_samples, 4, 3)
        model : Keras model
            Model used to correct the given sample

        Returns
        -------
        projection_new : ndarray
            New projected radar detections as input for model
        label_new : ndarray
            Resulting label: residual decalibration
        """
        
        # Fake model output.
        # fake_output = np.array([0.9998477, 0, 0.0174524, 0])
        # output_quats = np.tile(fake_output, (len(data[0]),1)) # Repeat output N times.

        # Generate model output.
        model_outputs = model.predict(data)
        output_quats = model_outputs[0]
        #output_quats = model_outputs[:,:4] # Crop noise output.
        output_quats = self._normalize_quaternions(output_quats)

        # Convert to matrices.
        inv_decalib = self._convert_to_matrices(labels)
        decalib = self._invert_homogeneous_matrices(inv_decalib)
        
        inv_decalib_hat = self._convert_to_matrices(output_quats)
        decalibs_hat = self._invert_homogeneous_matrices(inv_decalib_hat)

        # Recompute projections.
        tmp = np.matmul(decalib, h_gt)
        h_inits_new = np.matmul(inv_decalib_hat, tmp)
        projections_new = self._project_radar_detections(h_inits_new, Ks, radar_detections)
        # tmp = np.matmul(decalib, self._H_gt)
        # projections_new = self._project_radar_detections(tmp, radar_detections)

        # Compute new labels
        inv_decalibs_hat_hat = np.matmul(inv_decalib, decalibs_hat)
        labels_new = self._convert_to_quaternions(inv_decalibs_hat_hat)

        return projections_new, labels_new



# if __name__ == '__main__': # For debugging.
#     proj = RadarReprojectionManager([], [100, 100], [5, 5], np.random.normal(size=[3,3]), np.random.normal(size=[3,3]))
