import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognizer")
        self.setGeometry(100, 100, 800, 600)

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.vlayout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)

        # Create an ImageView widget
        self.imgv1 = pg.ImageView()
        self.imgv2 = pg.ImageView()
        self.hlayout.addWidget(self.imgv1)
        self.hlayout.addWidget(self.imgv2)

        self.vlayout.addLayout(self.hlayout)
        self.images = np.zeros((8 * 40, 112, 92), dtype='float64')
        self.test_images = []
        self.test_labels = []
        
        self.dataset_path = r'dataset'
        # get training images
        self.images = self.construct_data_matrix()

        # perform PCA and calc eigenfaces to be used on data
        self.init_PCA()

        # 8 images out of each folder is in train folder and 2 in test
        # these are the labels for the  training
        self.labels = np.array([i for i in range(1, 41)])
        self.labels = np.repeat(self.labels, 8)

        self.train_KNN()

        # predict the class using the trained knn model on the test image
        # test_image_path = "dataset/s24/test/5.pgm"
        test_image_path = "test-images/Einstein.jpg."
        print(f"The true class is {test_image_path.split('.')[-2].split('/')[-1]}")
        test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

        # predict function and returns the distance between the image and its neighbors to check
        # for outsiders
        pred, dist = self.recognise_face(test_image)
        print(f"The predicted class is {pred}")
        # print(f"Test image distance {dist}")

        if dist > 224000000:
            print("Outsider")
        
        acc = self.calc_accuracy()
        print(f"The accuracy for the whole test set is {acc}")
    
    def train_KNN(self):
        # Create a KNN classifier
        self.knn = KNeighborsClassifier(n_neighbors=5)

        # Train the classifier on the training data
        self.knn.fit(self.reduced_data, self.labels)

    def create_test_datasets(self):
        # get test data from folders
        self.test_images = self.construct_data_matrix("test")
        # the labels for the test data, each subject has 2 test images
        self.test_labels = np.repeat(np.array([i for i in range(1, 41)]), 2)

    def calc_accuracy(self):
        self.create_test_datasets()
        true_count = 0
        for true_y, tst_img in zip(self.test_labels, self.test_images):
            pred, _ = self.recognise_face(tst_img)
            # print(f"true y {true_y}")
            # print(f"pred {pred}")
            if true_y == pred:
                true_count += 1
        acc = true_count / self.test_labels.shape[0]
        return acc

    def recognise_face(self, test_img):

        test_img = np.array(test_img)
        # project example on eigenfaces
        test_reduced_data = self.apply_PCA(test_img)

        # Make predictions on the testing data
        predicted_label = self.knn.predict(test_reduced_data)[0]
        # get the distance sum to check for outsiders
        distances, _ = self.knn.kneighbors(test_reduced_data)
        total_dist = distances.flatten().sum()
        # print(f"The total_dist is {total_dist}")
        return predicted_label, total_dist

    def construct_data_matrix(self, extension="train"):
        # Iterate through each folder (s1, s2, ..., s40)
        images = []
        i = 0
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, f"{folder_name}/{extension}")
            
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
                images.append(image)
                i += 1
        return np.array(images)
        # print(np.count_nonzero(self.images))
        
    def init_PCA(self):
        # Reshape the whole images to 1D vectors and construct data matrix
        self.matrix = np.resize(self.images, (self.images.shape[0], self.images.shape[1] * self.images.shape[2]))

        # Get the Mean Image list (10304,)
        self.mean_list = np.mean(self.matrix, axis=0, dtype='float64')
  
        # Subtract the mean image from All images
        mean_subtracted_matrix = self.matrix - self.mean_list

        # Compute Covariance Matrix (400, 400)
        covariance_matrix = np.cov(mean_subtracted_matrix)
        
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        sorted_eigenvalues_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_eigenvalues_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_eigenvalues_indices]

        # Calculate Cumulative Sum of Eigenvalues
        cumulative_sum = np.cumsum(sorted_eigenvalues)
        total_sum = np.sum(sorted_eigenvalues)
        ratio = cumulative_sum / total_sum

        # Determine the Number of Eigenvectors to Keep
        num_eigenfaces = np.argmax(ratio >= 0.9) + 1

        print(num_eigenfaces)

        # Select Top Eigenvectors -eigenfaces- all vectors summing up eigen values to 90%
        selected_eigenvectors = sorted_eigenvectors[:, :num_eigenfaces]

        # Project Data (selected_eigenvectors, 10304)
        self.projected_data = np.dot(selected_eigenvectors.T, mean_subtracted_matrix)

        self.reduced_data = self.matrix @ self.projected_data.T

    # only for ONE image
    def apply_PCA(self, data):
        # flatten
        if len(data.shape) == 2:
            data = np.resize(data, (1, 112 * 92))
        else:
            raise ValueError("Not supported")
        # project on eigen faces
        reduced_data = data @ self.projected_data.T
        return reduced_data


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    # mainWindow.show()
    sys.exit(app.exec())
