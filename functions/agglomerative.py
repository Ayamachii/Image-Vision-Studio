import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Agglomerative:
    def __init__(self):
        pass

    def calculate_distance_bet_clusters(self, c1, c2):
        # print(f"c1 {c1, type(c1)}")
        return math.sqrt(sum((c1[i] - c2[i]) ** 2 for i in range(3)))

    def get_new_center_for_cluster(self, c1, c2):
        l = (c1[0] + c2[0]) // 2
        u = (c1[1] + c2[1]) // 2
        v = (c1[2] + c2[2]) // 2
        return (l, u, v)

    def apply_agg(self, input_image, number_clusters):

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2Luv)
        # output_image = np.zeros_like(input_image)
        output_image_gs = np.zeros((input_image.shape[0], input_image.shape[1], 1))
        clusters = [input_image[i, j] for i in range(input_image.shape[0]) for j in range(input_image.shape[1])]

        while len(clusters) > number_clusters:
            print(len(clusters))
            min_distance = float('inf')
            first_cluster_index, second_cluster_index = 0, 0
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self.calculate_distance_bet_clusters(clusters[i], clusters[j])
                    if dist < min_distance:
                        min_distance = dist
                        first_cluster_index, second_cluster_index = i, j

            clusters[first_cluster_index] = self.get_new_center_for_cluster(clusters[first_cluster_index], clusters[second_cluster_index])
            clusters.pop(second_cluster_index)

        labels = np.zeros(input_image.shape[:2], dtype=np.int32)
        for i in range(input_image.shape[0]):
            for j in range(input_image.shape[1]):
                min_cluster = 0
                min_distance = float('inf')

                for cluster_index in range(len(clusters)):
                    dist = self.calculate_distance_bet_clusters(input_image[i, j], clusters[cluster_index])
                    if dist < min_distance:
                        min_distance = dist
                        min_cluster = cluster_index
                labels[i, j] = min_cluster

        # Generate a list of random colors
        # colors = np.random.randint(0, 255, size=(len(clusters), 3), dtype=np.uint8)
        colors_gs = [i/len(clusters)*255 for i in range(len(clusters))]

        flattened_array = labels.flatten()
        unique_values = np.unique(flattened_array)
        for index, value in enumerate(unique_values):
            indices = np.argwhere(labels == value)
            output_image_gs[indices[:, 0], indices[:, 1]] = colors_gs[index]
                
        # output_image_gs = cv2.cvtColor(output_image_gs, cv2.COLOR_Luv2RGB)
        return output_image_gs




# image_size = 20 # 20 is good
# K = 4

# output_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

# input_image = cv2.imread('watermelon.jpg')
# resized_image = cv2.resize(input_image, (image_size, image_size))

# agg = Agglomerative()
# output_image = agg.apply_agg(resized_image, K)

# cv2.imwrite("before_h.jpg", resized_image)
# cv2.imwrite("after_h.jpg", output_image)

# plt.figure(figsize=(12, 6))  # Adjust figure size as needed

# plt.subplot(1, 2, 1)
# plt.imshow(resized_image)
# plt.title('Image 1')
# plt.axis('off')  

# plt.subplot(1, 2, 2)
# plt.imshow(output_image)
# plt.title('Image 2')
# plt.axis('off')  

# plt.show()

