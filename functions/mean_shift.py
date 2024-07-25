import numpy as np
import cv2

class MeanShift:
    """
    A class representing the Mean Shift clustering algorithm.
    Parameters:
        bandwidth (float): The bandwidth parameter for the Mean Shift algorithm.
        threshold (float): The convergence threshold for the Mean Shift algorithm.
        sigma (float): The standard deviation parameter for the Gaussian kernel used in Mean Shift.
    Methods:
        fit(image): Performs Mean Shift clustering on the input image.
    """
    def __init__(self, bandwidth, threshold, sigma):
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.sigma = sigma

    def fit(self, image):
        """
        Apply Mean Shift clustering to the input image.
        Parameters:
            image (numpy.ndarray): The input image to be clustered.
        Returns:
            list: A list of dictionaries representing the clusters found by Mean Shift.
                Each dictionary contains keys 'points' and 'center', where:
                - 'points' is a boolean array indicating points belonging to the cluster.
                - 'center' is the centroid of the cluster.
        """
        img = np.array(image, copy=True, dtype=float)
        features = img.reshape(-1, img.shape[2])
        num_points = len(features)
        visited = np.full(num_points, False, dtype=bool)
        meanshift_clusters = []
        while np.sum(visited) < num_points:
            initial_idx = np.random.choice(np.arange(num_points)[~visited])
            initial_mean = features[initial_idx]
            while True:
                distances = np.linalg.norm(initial_mean - features, axis=1)
                gauss_weight = self.gaussian_kernel(distances)
                indx_within_bandwidth = np.where(distances <= self.bandwidth / 2)[0]
                marked_points_within_bandwidth= np.full(num_points, False, dtype=bool)
                marked_points_within_bandwidth[indx_within_bandwidth] = True
                points_within_bandwidth = features[indx_within_bandwidth]
                new_mean = np.average(points_within_bandwidth, axis=0, weights=gauss_weight[indx_within_bandwidth])
                if np.linalg.norm(new_mean - initial_mean) < self.threshold:
                    merged = False
                    for cluster in meanshift_clusters:
                        if np.linalg.norm(cluster['center'] - new_mean) < 0.5 * self.bandwidth:
                            cluster['points'] = cluster['points'] + marked_points_within_bandwidth
                            cluster['center'] = 0.5 * (cluster['center'] + new_mean)
                            merged = True
                            break
                    if not merged:
                        meanshift_clusters.append({'points': marked_points_within_bandwidth, 'center': new_mean})
                    visited[indx_within_bandwidth] = True
                    break
                initial_mean = new_mean
        return meanshift_clusters
    
    def gaussian_kernel(self, distances):
        """
        Compute the Gaussian kernel weights.
        Parameters:
            distances (numpy.ndarray): Array of distances.
        Returns:
            numpy.ndarray: Array of Gaussian kernel weights.
        """
        return np.exp(-0.5 * (distances / self.sigma) ** 2)










def apply_mean_shift(img,bandwidth, threshold, sigma):
    mean_shift = MeanShift(bandwidth, threshold, sigma)
    meanshift_clusters = mean_shift.fit(img)
    clustered_image = np.zeros_like(img)
    for cluster in meanshift_clusters:
        points = cluster['points'].reshape(img.shape[:2])
        clustered_image[points, :] = cluster["center"] 
    return clustered_image