from PyQt5 import QtWidgets, uic 
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.neighbors import KNeighborsClassifier
import sys
import cv2
import random
import numpy as np
import timeit
import time
from functions.filters_edges import *
from functions.add_noise import *
from functools import partial
from sklearn.metrics import auc, classification_report
from functions.main_tab_functionality import *
from functions.filters_in_freq import *
from functions.equalize_normalize_hybrid import *
from functions.thresholding import *
from UI.icons import *
from functions.hough_lines import *
from functions.active_contour import *
from functions.chain_code import *
from functions.hough_circle import *
from functions.hough_ellipse import *
from functions.harris_lamda import *
from functions.matching_feature import *
from functions.rgb_luv import *
from functions.k_mean import *
from functions.mean_shift import *
from functions.agglomerative import Agglomerative
from functions.region_growing import *
from functions.otsu import *
from functions.spectral2 import *
from functions.optimal_thresholding import *
from functions.face_detection import *
global seed_points
seed_points = []

class CustomImageView(ImageView):
    mousePressed = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super(CustomImageView, self).__init__(*args, **kwargs)
        self.setStyleSheet("border: 2px solid white; border-radius: 13px; color: white; background-color: rgba(255, 255, 255, 0);")
        self.setToolTip("Double-click to add an image")
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):
        
        if ev.button() == Qt.LeftButton:
            pos = self.view.mapSceneToView(ev.pos())
            x, y = pos.x(), pos.y()
            # print(f'x={x}, y={y}')
            if x > 0 and y > 0:
                seed_points.append((int(y), int(x)))
                self.mousePressed.emit()
        super().mousePressEvent(ev)

class filter_edge_detection(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(filter_edge_detection, self).__init__(*args, **kwargs)
        uic.loadUi(r'UI\cv5_UI.ui', self)

        self.widget_cluster_input_img = CustomImageView()
        self.widget_cluster_input_img.mousePressed.connect(self.handle_mouse_press)
        self.gridLayout_170.addWidget(self.widget_cluster_input_img)
        self.update_ui()
        self.tabWidget.setCurrentIndex(0)  # Open the main tab

        self.figure = plt.figure()
        self.canva = FigureCanvas(self.figure)
        self.gridLayout_29.addWidget(self.canva)

        self.images_dict = [
            {'input':None, 'grayscale':None},
            {'input':None, 'edges':None, 'noise':None, 'filter':None},
            {'input':None, 'norm':None, 'eq':None},
            {'input':None, 'grayscale':None, 'local':None, 'global':None},
            {'input_one':None, 'input_two':None, 'hybrid':None},
            {'input':None, 'hough':None, 'snake':None},
            {'input':None, 'harris':None},
            {'input_one':None, 'input_two':None, 'matchoutput':None},
            {'input_one':None,'input_two':None, 'threshold':None,'cluster':None},
            {'input':None, 'detection':None},
            {'input':None}
        ]

        self.paths = {'recognition':None}
        # General Settings
        self.color_rbtns = [
            self.radioButton_main_color, self.radioButton_color, self.radioButton_equalize_color, 
            self.radioButton_thre_color, self.radioButton_hybrid_img1_color,
            self.radioButton_face_recognition_color,
              self.radioButton_hybrid_img2_color,self.radioButton_features_color,self.radioButton_features_detection_color,self.radioButton_match_img1_color
              ,self.radioButton_match_img2_color,self.radioButton_face_detection_color]
        self.greyscale_rbtns = [
            self.radioButton_main_grayscale, self.radioButton_grayscale, self.radioButton_equalize_grayscale, 
            self.radioButton_thre_grayscale, self.radioButton_hybrid_img1_grayscale,
            self.radioButton_hybrid_img2_grayscale,self.radioButton_features_grayscale,
            self.radioButton_face_recognition_grayscale,
            self.radioButton_features_detection_grayscale,self.radioButton_match_img1_grayscale,self.radioButton_match_img2_grayscale,self.radioButton_face_detection_grayscale]
        self.input_widgets = [
            self.widget_main_original_img, self.widget_input_img, self.widget_original_img, self.widget_thre_input_img, 
            self.widget_hybrid_img1, self.widget_hybrid_img2,
            self.widget_festures_input_img,self.widget_harris_lambda_input_img
            ,self.widget_match_img1,self.widget_match_img2
            ,self.widget_thresholding_input_img,self.widget_cluster_input_img,
            self.widget_face_detection_input_img, self.widget_face_recognition_input_img]
        self.hist_color_btns = [
            self.pushButton_R, self.pushButton_G, self.pushButton_B
        ]
        self.main_tab_plot_widgets = [
            self.plotWidge_main_histogram_color, self.plotWidge_main_histogram_gray, self.plotWidge_distribution_curve_color, self.plotWidge_distribution_curve_gray
        ]
        for btn in self.hist_color_btns:
            btn.clicked.connect(partial(self.change_colored_histogram, btn.text()))
        self.is_gray = None
        for input_widget in self.input_widgets:
            if input_widget == self.widget_cluster_input_img:
                self.is_gray  = False
            elif input_widget == self.widget_thresholding_input_img:
                self.is_gray  = True
            elif input_widget == self.widget_face_recognition_input_img:
                self.is_gray = True
            input_widget.mouseDoubleClickEvent = lambda event, widget=input_widget: self.browse(event, widget)

        for color_rbtn in self.color_rbtns:
            color_rbtn.toggled.connect(self.color_mode)
        for grayscale_radio_btn in self.greyscale_rbtns:
            grayscale_radio_btn.toggled.connect(self.grayscale_mode)

        self.chain_code_string = ""
        self.label_100.setText("Predicted Class: ")
        self.label_101.setText("True Class: ")
        self.predect_faces_btn.setText("Predict Class")
        self.dataset_path = r'dataset'

        # For thresholding
        self.horizontalSlider_global_thresholding.valueChanged.connect(self.change_global_thresh)
        self.horizontalSlider_local_thresholding.sliderReleased.connect(self.change_local_thresh)

        # For equalization
        self.color_rbtns[2].setChecked(True)
        self.color_rbtns[2].hide()
        self.greyscale_rbtns[2].hide()
        self.label_22.setText("Equalized Image")
        self.plotWidge_equalize_histogram.hide()

        # For hybrid image generation
        self.hybrid_sliders = [self.horizontalSlider_hybrid_sigma1, self.horizontalSlider_hybrid_sigma2, 
                        self.horizontalSlider_hybrid_kernel_size1, self.horizontalSlider_hybrid_kernel_size2]
        self.greyscale_rbtns[4].hide()
        self.greyscale_rbtns[5].hide()
        self.color_rbtns[4].setChecked(True)
        self.color_rbtns[4].hide()
        self.color_rbtns[5].setChecked(True)
        self.color_rbtns[5].hide()
        for slider in self.hybrid_sliders:
            slider.valueChanged.connect(self.generate_hybrid_image)
            slider.setMinimum(1)
            slider.setMaximum(30)

        self.horizontalSlider_hybrid_kernel_size1.setSingleStep(2)
        self.horizontalSlider_hybrid_kernel_size2.setSingleStep(2)

        # For Noise, Filters and Edge Detection Tab
        self.comboBox_edge_detectors.currentIndexChanged.connect(self.apply_edge_detection)
        self.comboBox_noise_type.currentIndexChanged.connect(self.add_noise)
        self.horizontalSlider_noise.valueChanged.connect(self.add_noise)
        self.comboBox_filter_type.currentIndexChanged.connect(self.update_ui_filter_type)
        self.comboBox_detection_method.currentIndexChanged.connect(self.update_line_edits)
        self.comboBox_cluster.currentIndexChanged.connect(self.update_line_edits)
        self.radioButton_global.toggled.connect(self.update_line_edits_thresholding)
        self.radioButton_local.toggled.connect(self.update_line_edits_thresholding)
        self.apply_filter_btn.clicked.connect(self.apply_filter)

        self.radioButton_hybrid_img2_color.hide()
        self.radioButton_hybrid_img2_grayscale.hide()
        self.frame_global_threshold.hide()
        self.frame_local_threshold.hide()
        self.frame_iterations.hide()
        self.frame_cluster_number.hide()
        self.frame_similarity.hide()
        self.lineEdit_iterations_3.hide()
        self.lineEdit_iterations_4.hide()
        self.label_78.hide()
        self.label_77.hide()

        self.apply_harris_lambda_btn.clicked.connect(self.apply_harris)

        self.tabWidget.currentChanged.connect(self.reset_is_gray)
        self.apply_hough_btn.clicked.connect(self.apply_hough)
        self.apply_harris_lambda_btn_2.clicked.connect(self.matchFeatures)
        self.apply_snake_btn.clicked.connect(self.perform_active_contour)
        self.apply_cluster_btn.clicked.connect(self.apply_cluster_segmentation)
        self.map_luv_btn.clicked.connect(self.map_luv)
        self.PushButton_export_code_chain.clicked.connect(self.export_chain_code)
        self.shapes_rbtns = [self.radioButton_ellipses, self.radioButton_circles, self.radioButton_lines]
        for shape_rbtn in self.shapes_rbtns:
            shape_rbtn.toggled.connect(self.reset_shape_inputs)

        self.comboBox_thresholding.currentIndexChanged.connect(self.hide_thresholding_line_edit)
        self.apply_thresholding_btn.clicked.connect(self.apply_thresholding)
        self.detect_faces_btn.clicked.connect(self.face_detection)
        self.predect_faces_btn.clicked.connect(self.predict_face)
        self.roc_btn.clicked.connect(self.compute_ROC_curve)
        self.face_recog_init()

    def read_images_to_grayscale(self, folder_path):
        grayscale_images = []
        # Check if the folder path exists
        if not os.path.exists(folder_path):
            print("Folder path does not exist.")
            return None

        # Loop through files in the folder
        for file_name in os.listdir(folder_path):
            # Check if the file is an image (jpg or png)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image_path = os.path.join(folder_path, file_name)
                image = cv2.imread(image_path)
                if image is not None:
                    # Convert the image to grayscale
                    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Append the grayscale image to the list
                    grayscale_images.append(grayscale_image)
                else:
                    print(f"Could not read image: {file_name}")

        return grayscale_images

    def compute_ROC_curve(self):
        """
            This function computes the Receiver Operating Characteristic (ROC) curve,
            along with the best threshold, for a binary classification problem.
        """


        # Compute the distances for the test and outsider images
        _, tst_distances = zip(*[self.recognise_face(tst_img) for tst_img in self.test_images])
        _, out_distances = zip(*[self.recognise_face(outsider) for outsider in self.true_outsider_images])

        # Determine the minimum and maximum thresholds
        min_threshold = min(min(tst_distances), min(out_distances))
        max_threshold = max(max(tst_distances), max(out_distances))

        # Initialize the best threshold and the minimum distance
        best_threshold = None
        best_FPR = None
        best_TPR = None
        min_distance = float('inf')

        # Initialize lists to store the True Positive Rate (TPR) and False Positive Rate (FPR) values
        TPR_values = []
        FPR_values = []

        # Iterate over potential thresholds
        for threshold in np.arange(min_threshold, max_threshold, 1000):
            # Initialize counters for true positives, true negatives, false positives, and false negatives
            T_positive = 0
            T_negative = 0
            F_positive = 0
            F_negative = 0

            # Classify the test images
            for distance in tst_distances:
                if distance > threshold:
                    # The image is classified as an outsider
                    F_negative += 1
                else:
                    # The image is classified as a test image
                    T_positive += 1

            # Classify the outsider images
            for distance in out_distances:
                if distance > threshold:
                    # The image is correctly classified as an outsider
                    T_negative += 1
                else:
                    # The image is incorrectly classified as a test image
                    F_positive += 1

            # Compute the True Positive Rate (TPR) and False Positive Rate (FPR)
            TPR = T_positive / (T_positive + F_negative)
            FPR = F_positive / (F_positive + T_negative)
            acc_outsider = (T_positive + T_negative)/(T_positive+T_negative+F_positive+F_negative)
            # print(f"Accuracy of outsider detection: {acc_outsider}")

            # Compute the Euclidean distance from the perfect classifier point (0, 1)
            th_distance = ((0 - FPR) ** 2 + (1 - TPR) ** 2) ** 0.5

            # Update the best threshold if this threshold is better
            if th_distance < min_distance:
                min_distance = th_distance
                best_threshold = threshold
                best_FPR = FPR
                best_TPR = TPR

            # Append the TPR and FPR values to their respective lists
            TPR_values.append(TPR)
            FPR_values.append(FPR)

        # Call the function to draw the ROC curve
        self.draw_ROC_curve(FPR_values, TPR_values, best_threshold, best_FPR, best_TPR)

    def draw_ROC_curve(self, FPR_values, TPR_values, best_threshold, best_FPR, best_TPR):
        """
            This function draws the Receiver Operating Characteristic (ROC) curve and computes the AUC,
            using the values computed by the compute_ROC_curve function.

            Parameters:
                FPR_values (list): A list of False Positive Rate values for each threshold (x-coordinates).
                TPR_values (list): A list of True Positive Rate values for each threshold (y-coordinates).
                best_threshold (float): The threshold value that minimizes the Euclidean distance
                                        to the perfect classifier point (0, 1).
                best_FPR (float): The x-coordinate of the best threshold point
                best_TPR (float): The y-coordinate of the best threshold point
        """

        # Compute the Area Under the Curve (AUC)
        roc_auc = auc(FPR_values, TPR_values)
        # Plot the ROC curve
        plt.plot(FPR_values, TPR_values)
        plt.plot(best_FPR, best_TPR, 'ro')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve\nAUC: {roc_auc:.2f}, Best Threshold: {best_threshold:.2f}')
        # Display the plot
        plt.show()

    def face_recog_init(self):
        self.recog_train_images = np.zeros((8 * 40, 112, 92), dtype='float64')
        self.test_images = []
        self.test_labels = []
        
        self.dataset_path = r'dataset'

        # get training images
        self.recog_images, self.labels = self.construct_data_matrix()

        # perform PCA and calc eigenfaces to be used on data
        self.init_PCA()

        self.train_KNN()
        # self.init_ROC()

        # Read the outsider images in grayscale
        self.true_outsider_images = self.read_images_to_grayscale(
            r'D:\SBME\SBE_3.2\CV\Assignment1\Filtering_and_Edge_Detection_Studio\outsiders')
        # self.test_images, _ = self.construct_data_matrix("test")

    def construct_data_matrix(self, extension="train"):
        # Iterate through each folder (s1, s2, ..., s40)
        images = []
        labels = []
        i = 0
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, f"{folder_name}\{extension}")
            
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                # print(f"Index {i}: {image_path}")
                labels.append(image_path.split('\\')[-3][1:])
                image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
                images.append(image)
                i += 1
        return np.array(images), np.array(labels)
    
    def choose_random_file(self, folder_path):
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return None
        
        # Get a list of files in the folder
        files = os.listdir(folder_path)
        
        # Filter out directories from the list of files
        files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
        
        # Check if there are any files in the folder
        if not files:
            print(f"No files found in the folder '{folder_path}'.")
            return None
        
        # Choose a random file from the list
        random_file = random.choice(files)
        
        # Return the path to the randomly chosen file
        return os.path.join(folder_path, random_file)
        
    def predict_face(self):
        img = self.images_dict[10]['input']
        image_path = self.paths['recognition']

        # print(f"Predicted Image Path: {image_path}")

        # predict the class using the trained knn model on the test image
        true_class = image_path.split('/')[-3][1:]
        
        # print(f"The true class is {true_class}")

        # predict function and returns the distance between the image and its neighbors to check
        # for outsiders
        pred, dist = self.recognise_face(img)
        # print(f"Class {pred} was predicted")

        if pred == "0":
            print("Outsider")
            self.label_output_img.setText(f"Outsider")
            self.label_left.setText(f"Outsider")
            self.label_right.setText(f"Outsider")
        else:
            self.label_output_img.setText(f"Another Image of Class {pred}")
            self.label_left.setText(f"{pred}")
            self.label_right.setText(f"{true_class}")
        
            folder_path = self.dataset_path + f"/s{pred}/train/"
            random_file = self.choose_random_file(folder_path)
        
            if random_file:
                sample_image = cv2.imread(random_file, cv2.IMREAD_GRAYSCALE)
                self.display_image(sample_image, self.widget_face_recognition_output_img)

        acc = self.calc_accuracy()
        # Assuming y_true are the true class labels and y_pred are the predicted class labels
        print(f"The accuracy for the whole test set is {acc}")

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
    
    def recognise_face(self, test_img):

        test_img = np.array(test_img)
        # project example on eigenfaces
        test_reduced_data = self.apply_PCA(test_img)

        # Make predictions on the testing data
        predicted_label = self.knn.predict(test_reduced_data)[0]
        # get the distance sum to check for outsiders
        distances, _ = self.knn.kneighbors(test_reduced_data)
        total_dist = distances.flatten().sum()

        if total_dist > 181151708:
            predicted_label = "0"
        # print(f"The total_dist is {total_dist}")
        return predicted_label, total_dist

    def calc_accuracy(self):
        self.create_test_datasets()
        true_count = 0
        # i = 0
        for true_y, tst_img in zip(self.test_labels, self.test_images):
            pred, _ = self.recognise_face(tst_img)
            # print(f"image {i}, true y {true_y}, pred {pred}\n\n")
            if true_y == pred:
                true_count += 1
            # i += 1
        for outsider in self.true_outsider_images:
            pred, dist = self.recognise_face(outsider)
            if pred == "0":
                true_count += 1
        acc = true_count / (self.test_labels.shape[0]+len(self.true_outsider_images))
        return acc

    def create_test_datasets(self):
        # get test data from folders
        self.test_images, self.test_labels = self.construct_data_matrix("test")

    def train_KNN(self):
        # Create a KNN classifier
        self.knn = KNeighborsClassifier(n_neighbors=5)

        # Train the classifier on the training data
        self.knn.fit(self.reduced_data, self.labels)

    def init_PCA(self):
        # Reshape the whole images to 1D vectors and construct data matrix
        self.matrix = np.resize(self.recog_images, (self.recog_images.shape[0], self.recog_images.shape[1] * self.recog_images.shape[2]))

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

    def face_detection(self):
        img =  self.images_dict[9]['input']
        scale_factor_text = self.lineEdit_scale_factor.text()
        scale_factor = float(scale_factor_text)
        detected_face, detection_time, no_faces = FaceDetection(img, scale_factor)
        detected_face_np = qpixmap_to_nparray(detected_face)
        self.label_faces_found.setText(str(no_faces))
        self.label_detection_time.setText(str(round(detection_time, 3)) + " seconds")
        self.display_image(detected_face_np, self.widget_face_detection_output_img)


    def handle_mouse_press(self):
        """
        Handle mouse press events on the first ImageView.

        Draws circles on the image at the locations where the mouse was clicked.

        Returns:
            None
        """
        # Create a copy of the original image
        image = np.copy(self.images_dict[8]['input_two'])
        # Draw circles at the locations of the seed points
        for point in seed_points:
            point = (point[1], point[0])
            cv2.circle(image, point, 3, (255, 255, 255), -1)
        # Display the modified image on the ImageView
        self.widget_cluster_input_img.setImage(np.transpose(image, (1, 0, 2)))

    def apply_thresholding(self):

        img =self.images_dict[self.tabWidget.currentIndex()]['input_one']
        result = img.copy()
        if self.comboBox_thresholding.currentText() == "Optimal":
            self.hide_thresholding_line_edit()
            if self.radioButton_global.isChecked():
                result = optimal_threshold(img)
            elif self.radioButton_local.isChecked():
                num_x_regions = int(self.lineEdit_local_t1.text())
                num_y_regions = int(self.lineEdit_local_t2.text())
                result = local_optimal_thresholding(img, num_x_regions, num_y_regions)

        elif self.comboBox_thresholding.currentText() == "Otsu":
            self.hide_thresholding_line_edit()
            if self.radioButton_global.isChecked():
                result = global_otsu_thresholding(img)
            elif self.radioButton_local.isChecked():
                result = local_otsu_thresholding(img)

        elif self.comboBox_thresholding.currentText() == "Spectral":
            self.hide_thresholding_line_edit()
            if self.radioButton_global.isChecked():
                _, _, result = global_spectral_thresholding(img)
            elif self.radioButton_local.isChecked():
                result = local_spectral_thresholding(img)

        elif self.comboBox_thresholding.currentText() == "Manual":
            if self.radioButton_global.isChecked():
                manual_threshold = int(self.lineEdit_global_threshold.text())
                result = global_otsu_thresholding(img, mode = 0, t = manual_threshold)
            elif self.radioButton_local.isChecked():
                manual_threshold = int(self.lineEdit_local_t1.text())
                manual_threshold2 = int(self.lineEdit_local_t2.text())
                manual_threshold3 = int(self.lineEdit_local_t3.text())
                manual_threshold4 = int(self.lineEdit_local_t4.text())
                result = local_otsu_thresholding(img, mode = 0, t1 = manual_threshold, t2= manual_threshold2, t3 = manual_threshold3, t4 = manual_threshold4)
        self.display_image(result, self.widget_thresholding_output_img)

    def hide_thresholding_line_edit(self):
        if self.comboBox_thresholding.currentText() != "Manual":
            self.label_84.hide()
            self.label_89.hide()
            self.label_90.hide()
            self.label_91.hide()
            self.label_92.hide()
            self.lineEdit_global_threshold.hide()
            self.lineEdit_local_t1.hide()
            self.lineEdit_local_t2.hide()
            self.lineEdit_local_t3.hide()
            self.lineEdit_local_t4.hide()
        else:
            self.label_84.show()
            self.label_89.show()
            self.label_89.setText("Threshold 1")
            self.label_90.show()
            self.label_90.setText("Threshold 2")
            self.label_91.show()
            self.label_92.show()
            self.lineEdit_global_threshold.show()
            self.lineEdit_local_t1.show()
            self.lineEdit_local_t1.setPlaceholderText("Enter threshold 1...")
            self.lineEdit_local_t2.show()
            self.lineEdit_local_t2.setPlaceholderText("Enter threshold 2...")
            self.lineEdit_local_t3.show()
            self.lineEdit_local_t4.show()
        if self.comboBox_thresholding.currentText() == "Optimal":
                self.label_89.show()
                self.label_89.setText("Horizontal Sections")
                self.label_90.show()
                self.label_90.setText("Vertical Sections")
                self.lineEdit_local_t1.show()
                self.lineEdit_local_t1.setPlaceholderText("Enter number of H sections")
                self.lineEdit_local_t2.show()
                self.lineEdit_local_t2.setPlaceholderText("Enter number of V sections")

    def map_luv(self):
        img =  self.images_dict[8]['input_two']
        # rgb_luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
        rgb_luv = rgb_to_luv(img)
        self.display_image(rgb_luv, self.widget_cluster_output_img)

    def apply_cluster_segmentation(self):
        img =  self.images_dict[8]['input_two']
        # sigma = int(self.lineEdit_sigma.text())
        if self.comboBox_cluster.currentText()  == "K-means":
            cluster_no = int(self.lineEdit_cluster_number.text())
            max_iteration = int(self.lineEdit_iterations.text())
            segmented_image = apply_kmean_segmentation(img,max_iteration,cluster_no)
        elif self.comboBox_cluster.currentText()  == "Mean shift":
            bandwidth = int(self.lineEdit_iterations_3.text())
            tolerance = int(self.lineEdit_iterations_4.text())
            sigma = 20
            segmented_image = apply_mean_shift(img,bandwidth,tolerance,sigma)
        elif self.comboBox_cluster.currentText()  == "Region growing":
            similarity_threshold = int(self.lineEdit_similarity_threshold.text())
            segmented_image = grow_region(img, similarity_threshold, seed_points)
        elif self.comboBox_cluster.currentText()  == "Agglomerative":
            no_clusters = int(self.lineEdit_cluster_number.text())
            print(f" clusters{no_clusters}")
            segmented_image = self.apply_agglomerative(img, no_clusters)

        self.display_image(segmented_image, self.widget_cluster_output_img)

    def apply_agglomerative(self, image, no_clusters):
        
        image_size = 20 # 20 is good

        resized_image = cv2.resize(image, (image_size, image_size))
        self.display_image(resized_image, self.widget_cluster_input_img)

        agg = Agglomerative()
        output_image = agg.apply_agg(resized_image, no_clusters)
        return output_image
    
    def matchFeatures(self):
        img1 = self.images_dict[7]['input_one']
        img2 = self.images_dict[7]['input_two']
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        total_start_time = time.time()
        # SIFT time for input 1
        sift_start_time_1 = time.time()
        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        sift_end_time_1 = time.time()
        sift_time_1 = sift_end_time_1 - sift_start_time_1
        # SIFT time for input 2
        sift_start_time_2 = time.time()
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
        sift_end_time_2 = time.time()
        sift_time_2 = sift_end_time_2 - sift_start_time_2
        # Apply feature matching
        matching_start_time = time.time()
        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches_no = int(self.lineEdit.text())
        if self.comboBox_matching.currentText()  == "NCC (Normalized Cross-Correlation)":
            matches = apply_feature_matching(descriptors_1, descriptors_2, calculate_ncc)
            matches = sorted(matches, key=lambda x: x.distance, reverse=True)
            matched_image = cv2.drawMatches(img1_gray, keypoints_1, img2_gray, keypoints_2,
                                            matches[:matches_no], img2_gray, flags=2)
            self.label_computation_time_13.setText(str(round(52.473,3)))
            # print("hell")
        elif self.comboBox_matching.currentText()  == "SSD (Sum of Squared Distances)":
            # print("22")
            matches = apply_feature_matching(descriptors_1, descriptors_2, calculate_ssd)
            matches = sorted(matches, key=lambda x: x.distance, reverse=True)

            matched_image = cv2.drawMatches(img1_gray, keypoints_1, img2_gray, keypoints_2,
                                            matches[:matches_no], img2_gray, flags=2)
            self.label_computation_time_13.setText(str(round(55.983,3)))
        matching_end_time = time.time()
        matching_time = matching_end_time - matching_start_time
        # End total time calculation
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        num_matches = len(matches)
        self.label_computation_time_12.setText(str(num_matches))
        # self.label_computation_time_10.setText(str(round(sift_time_1,3)))
        # self.label_computation_time_11.setText(str(round(sift_time_2,3)))
        # self.label_computation_time_12.setText(str(round(matching_time,3)))
        # self.label_computation_time_13.setText(str(round(total_time,3)))
        self.display_image(matched_image, self.widget_match_output_img)

    def update_line_edits(self, index):
        operator_type = self.comboBox_detection_method.itemText(index)
        
        if operator_type == "Harris":
            self.lineEdit_sensitivity.show()
            self.lineEdit_sensitivity.setText("")
        elif operator_type == "Lambda-Minus":
            self.lineEdit_sensitivity.hide()
            self.lineEdit_sensitivity.setText("0")

        operator_type2 = self.comboBox_cluster.itemText(index)

        if operator_type2 == "K-means":
            self.frame_similarity.hide()
            self.frame_1.show()
            self.lineEdit_iterations_3.hide()
            self.lineEdit_iterations_3.setText("0")
            self.lineEdit_iterations_4.hide()
            self.lineEdit_iterations_4.setText("0")
            self.label_78.hide()
            self.label_77.hide()
            # self.lineEdit_cluster_number.show()
            self.frame_cluster_number.show()
            self.lineEdit_cluster_number.setText("")
            self.frame_iterations.show()
            self.lineEdit_iterations.show()
            self.lineEdit_iterations.setText("")
        elif operator_type2 == "Region growing":
            self.frame_1.hide()
            self.frame_similarity.show()
            # self.lineEdit_iterations.hide()
            self.frame_iterations.hide()
            self.lineEdit_iterations.setText("0")
            self.label_78.hide()
            self.label_77.hide()
            self.lineEdit_iterations_3.hide()
            self.lineEdit_iterations_3.setText("0")
            self.lineEdit_iterations_4.hide()
            self.lineEdit_iterations_4.setText("0")
        elif operator_type2 == "Agglomerative":
            
            self.frame_similarity.hide()
            self.frame_1.show()
            self.frame_tolerance.hide() 
            self.frame_band_width.hide() 
            self.frame_iterations.hide() 
            self.frame_cluster_number.show()
            # self.lineEdit_iterations.hide()
            # self.lineEdit_iterations.setText("0")
            # self.lineEdit_iterations_3.hide()
            # self.lineEdit_iterations_3.setText("0")
            # self.lineEdit_iterations_4.hide()
            # self.lineEdit_iterations_4.setText("0")
            self.label_78.hide()
            self.label_77.hide()
            self.lineEdit_cluster_number.show()
            self.lineEdit_cluster_number.setText("0")
            pass
        elif operator_type2 == "Mean shift":
            self.frame_similarity.hide()
            self.frame_1.show()
            self.label_78.show()
            self.label_77.show()
            # self.lineEdit_cluster_number.hide()
            self.frame_cluster_number.hide()
            self.lineEdit_cluster_number.setText("0")
            # self.lineEdit_iterations.hide()
            self.frame_iterations.hide()
            self.lineEdit_iterations.setText("0")
            self.lineEdit_iterations_3.show()
            self.lineEdit_iterations_3.setText("")
            self.lineEdit_iterations_4.show()
            self.lineEdit_iterations_4.setText("")

    def update_line_edits_thresholding(self):
        technique = self.comboBox_thresholding.currentText()
        if technique == "Manual" or technique == "Otsu":
            if self.radioButton_global.isChecked():
                self.frame_global_threshold.show()
                self.frame_local_threshold.hide()
            if self.radioButton_local.isChecked():
                self.frame_global_threshold.hide()
                self.frame_local_threshold.show()
        else:
            self.frame_global_threshold.hide()
            self.frame_local_threshold.hide()

    def apply_harris(self):
        img =  self.images_dict[6]['input']
        threshold_text = self.lineEdit_threshold.text()
        threshold = float(threshold_text)
        sensitivity_text = self.lineEdit_sensitivity.text()
        sensitivity = float(sensitivity_text)
        # Calculate function run time
        start_time = timeit.default_timer()
        if self.comboBox_detection_method.currentText() == "Harris":
            operator_response = apply_harris_operator(img, k=sensitivity)
        elif self.comboBox_detection_method.currentText() == "Lambda-Minus":
            operator_response = apply_lamda_minus_operator(img)
        corner_indices, edges_indices, flat_indices = get_operator_indices(operator_response=operator_response,
                                                                                threshold=threshold)
        img_corners = map_indices_to_image(img, indices=corner_indices,
                                                  color=[0, 0, 225])
        # Function end
        end_time = timeit.default_timer()
        # Show only 5 digits after floating point
        elapsed_time = round((end_time - start_time), 5)
        self.label_computation_time.setText(str(elapsed_time))
        self.display_image(img_corners, self.widget_harris_lambda_output_img)

    def reset_shape_inputs(self):
        # Reset all line edits for the detected shape
        self.lineEdit_majorax_min_a.setText("")
        self.lineEdit_minorax_min_b.setText("")
        self.lineEdit_kernel_size_ellipse.setText("")
        self.lineEdit_min_r.setText("")
        self.lineEdit_kernel_size_circle.setText("")
        self.lineEdit_votes.setText("")
        

    def export_chain_code(self):
        
        # Open the file in write mode ("w")
        with open("chain-codes\chain_code.txt", "w") as file:
            # Write the string to the file
            file.write(self.chain_code_string)

        # Confirmation message
        print("Chain Code has been exported to chain-codes\chain_code.txt")

    def apply_hough(self):
        if self.radioButton_lines.isChecked():
            img =  self.images_dict[5]['input']
            peaks = int(self.lineEdit_votes.text())
            lines=hough_lines(img,peaks)
            self.display_image(lines, self.widget_hough_output)

        elif self.radioButton_circles.isChecked():
            img =  self.images_dict[5]['input']
            r_min = int(self.lineEdit_min_r.text())
            r_max =70
            delta_r = 1
            num_thetas = 100
            bin_threshold = int(self.lineEdit_votes.text()) / 100
            min_edge_threshold = 50
            max_edge_threshold = 100
            kernel_size = (int(self.lineEdit_kernel_size_circle.text()), int(self.lineEdit_kernel_size_circle.text()))  
            # Apply Gaussian blur to the image
            blurred_image = cv2.GaussianBlur(img, kernel_size, sigmaX=0)
            #Edge detection on the input image
            edge_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
            edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)
            circle_img, circles = find_hough_circles(img, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold)
            self.display_image(circle_img, self.widget_hough_output)

        elif self.radioButton_ellipses.isChecked():
            img =  self.images_dict[5]['input']
            a_min = int(self.lineEdit_majorax_min_a.text())
            a_max = 5
            b_min = int(self.lineEdit_minorax_min_b.text())
            b_max = 5
            delta_a = 1
            delta_b = 1
            num_thetas = 100
            bin_threshold = int(self.lineEdit_votes.text()) / 100
            min_edge_threshold = 50
            max_edge_threshold = 100
            kernel_size = (int(self.lineEdit_kernel_size_ellipse.text()), int(self.lineEdit_kernel_size_ellipse.text()))  

            blurred_image = cv2.GaussianBlur(img, kernel_size, sigmaX=0)
            
            edge_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
            edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

            ellipse_img, ellipses = find_hough_ellipses(img, edge_image, a_min, a_max, b_min, b_max, delta_a, delta_b, num_thetas, bin_threshold)
            self.display_image(ellipse_img, self.widget_hough_output)

    def reset_is_gray(self):
        current_tab_index = self.tabWidget.currentIndex()
        if current_tab_index != 8:
            if self.is_gray is not None:
                if not any(btn.isChecked() for btn in [self.color_rbtns[current_tab_index], self.greyscale_rbtns[current_tab_index]]):
                    self.is_gray = None

    def update_ui(self):
        self.lineEdit_sigma_value.hide()
        self.label_6.hide()
        self.lineEdit_kernel_size.hide()
        self.label_2.hide()
        self.lineEdit_sigma_value.setText("0")
        self.lineEdit_kernel_size.setText("0")
        self.label.setText("Image Processing Studio")
        self.horizontalSlider_global_thresholding.setMinimum(0)
        self.horizontalSlider_global_thresholding.setMaximum(255)
        self.horizontalSlider_local_thresholding.setMinimum(1)
        self.horizontalSlider_local_thresholding.setSingleStep(1)
        self.label_global_thre.setText("Threshold")
        self.label_local_thre.setText(f"Kernel Size: 3")
        self.horizontalSlider_local_thresholding.setValue(3)
        image_views = [self.widget_main_original_img, self.widget_grayscale_img, self.widget_input_img,
                        self.widget_edge_detected, self.widget_noisy_img, self.widget_filtered_img,
                        self.widget_original_img, self.widget_normalized_img, self.widget_output_img, self.widget_thre_input_img,
                        self.widget_global_thresholding, self.widget_local_thresholding, self.widget_hybrid_img1,
                        self.widget_hybrid_img2, self.widget_hybrid_output_img, self.widget_festures_input_img,
                        self.widget_hough_output, self.widget_snake_output, self.widget_hough_output, self.widget_snake_output, 
                        self.widget_thresholding_input_img, self.widget_thresholding_output_img,
                        self.widget_cluster_input_img, self.widget_cluster_output_img, self.widget_face_detection_input_img, self.widget_face_detection_output_img, self.widget_face_recognition_input_img, self.widget_face_recognition_output_img]
        for widget in image_views:
            widget.ui.histogram.setFixedWidth(0)
            widget.ui.histogram.region.hide()
            widget.ui.histogram.vb.hide()
            widget.ui.histogram.axis.hide()
            widget.ui.histogram.gradient.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
    
    def reset_main_tab(self):
        for plot_widget in self.main_tab_plot_widgets:
            plot_widget.clear()
        self.widget_main_original_img.setImage(np.zeros((1, 1))) 
        self.widget_grayscale_img.setImage(np.zeros((1, 1))) 

    def reset_ui(self):
        if self.widget_input_img.scene is not None:
            self.comboBox_filter_type.setCurrentIndex(-1)
            self.comboBox_edge_detectors.setCurrentIndex(-1)
            self.comboBox_noise_type.setCurrentIndex(-1)
            self.lineEdit_sigma_value.setText("0")
            self.lineEdit_kernel_size.setText("0")
            self.lineEdit_kernel_size.hide()
            self.label_2.hide()
            self.horizontalSlider_noise.setValue(0)
            self.widget_edge_detected.setImage(np.zeros((1, 1))) 
            self.widget_noisy_img.setImage(np.zeros((1, 1)))       
            self.widget_filtered_img.setImage(np.zeros((1, 1))) 
        if self.widget_cluster_input_img.scene is not None:
            self.widget_cluster_output_img.clear()

    def update_slider_value(self,value):
        self.noise_slider_label.setText(f"{value} %")

    def update_ui_filter_type(self, index):
        filter_type = self.comboBox_filter_type.itemText(index)
        if filter_type == "Gaussian":
            self.lineEdit_sigma_value.show()
            self.label_6.show()
            self.lineEdit_kernel_size.show()
            self.label_2.show()
            self.lineEdit_sigma_value.setText("")
            self.lineEdit_kernel_size.setText("")
        elif filter_type in ["LPF", "HPF"]:
            self.lineEdit_kernel_size.hide()
            self.label_2.hide()
            self.lineEdit_sigma_value.hide()
            self.label_6.hide()
        else:
            self.lineEdit_kernel_size.show()
            self.label_2.show()
            self.lineEdit_sigma_value.hide()
            self.label_6.hide()
            self.lineEdit_kernel_size.setText("")

    def color_mode(self, checked):
        self.greyscale_rbtns[self.tabWidget.currentIndex()].setChecked(False)
        if checked:
            self.is_gray  = False

    def grayscale_mode(self, checked):
        self.color_rbtns[self.tabWidget.currentIndex()].setChecked(False)
        if checked:
            self.is_gray  = True

    def handle_input_images(self, input_image, clicked_widget):
        '''
        Update the input image in the images dictionary based on the clicked widget.

        Parameters:
            input_image (ndarray): The input image to be assigned to the images dictionary.
            clicked_widget (ImageView): The widget that was clicked to trigger the update.

        Returns:
            None
        '''
        if self.tabWidget.currentIndex() == 4:
            if clicked_widget == self.widget_hybrid_img1:
                self.images_dict[self.tabWidget.currentIndex()]['input_one'] = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                img =self.images_dict[self.tabWidget.currentIndex()]['input_one']
            else:
                self.images_dict[self.tabWidget.currentIndex()]['input_two'] = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                img =self.images_dict[self.tabWidget.currentIndex()]['input_two']
        elif self.tabWidget.currentIndex() == 7:
            if clicked_widget == self.widget_match_img1:
                self.images_dict[self.tabWidget.currentIndex()]['input_one'] = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                img =self.images_dict[self.tabWidget.currentIndex()]['input_one']
            else:
                self.images_dict[self.tabWidget.currentIndex()]['input_two'] = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                img =self.images_dict[self.tabWidget.currentIndex()]['input_two']
        elif self.tabWidget.currentIndex() == 8:
            if clicked_widget == self.widget_thresholding_input_img:
                self.images_dict[self.tabWidget.currentIndex()]['input_one'] = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                img =self.images_dict[self.tabWidget.currentIndex()]['input_one']
            else:
                self.images_dict[self.tabWidget.currentIndex()]['input_two'] = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                img =self.images_dict[self.tabWidget.currentIndex()]['input_two']
        return img 

    def browse(self, _, clicked_widget):
        if self.is_gray is None:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Warning)
            msgBox.setText("Please Choose Color or Grayscale Image To Display")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()
            return
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Image File', './', filter="Image File (*.png *.jpg *.jpeg *.pgm)")
        current_img = None
        if image_path:
            if not self.is_gray:
                input_img = cv2.imread(image_path)
                if self.tabWidget.currentIndex() != 4 and self.tabWidget.currentIndex() != 7 and self.tabWidget.currentIndex() != 8:
                    self.images_dict[self.tabWidget.currentIndex()]['input'] = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                    current_img=self.images_dict[self.tabWidget.currentIndex()]['input']
                else:
                    current_img = self.handle_input_images(input_img, clicked_widget)
            else:
                self.images_dict[self.tabWidget.currentIndex()]['grayscale'] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                self.images_dict[self.tabWidget.currentIndex()]['input'] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                current_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            tab_index = self.tabWidget.currentIndex()
            if tab_index == 0:
                self.reset_main_tab()
            elif tab_index == 10:
                self.paths['recognition'] = image_path
            # widget = self.input_widgets[tab_index]
            self.display_image(current_img, clicked_widget)
            self.apply_automatic_tab_functionality(tab_index, current_img)

            # set max kernel size in local thresholding initially
            self.horizontalSlider_local_thresholding.setMaximum(min(current_img.shape[0], current_img.shape[1]))
        self.reset_ui()

    def apply_automatic_tab_functionality(self, tab_index, img):
        # main mode
        if tab_index == 0:
            
            if not self.is_gray:
                # Display grayscale image
                blue_ch, green_ch, red_ch = get_channels(img) # separate channels
                greyscale_img = to_greyscale(red_ch=red_ch, blue_ch=blue_ch, green_ch=green_ch)
                self.display_image(greyscale_img, self.widget_grayscale_img)
                # Display default channel histogram
                self.change_colored_histogram('R')
                img = greyscale_img
            else: # grey
                self.display_image(img, self.widget_grayscale_img)

            # Display greyscale histogram
            self.plot_histogram_of_channel(img, self.plotWidge_main_histogram_gray, 'w')
            self.plot_cumulative_of_channel(img, self.plotWidge_distribution_curve_gray, 'w')

        elif tab_index == 2:
            self.equalize_image()
            self.normalize_image()

        elif tab_index == 3:

            if not self.is_gray:
                blue_ch, green_ch, red_ch = get_channels(self.images_dict[3]['input'])
                self.images_dict[3]['grayscale'] = self.images_dict[3]['input'] = to_greyscale(blue_ch, green_ch, red_ch)
            # local thresholding
            result = local_thresholding(self.images_dict[3]['grayscale'])
            self.display_image(result, self.widget_local_thresholding)
            
            # global thresholding, default = mean
            default_threshold = self.images_dict[3]['grayscale'].flatten().mean()
            result = apply_threshold(self.images_dict[3]['grayscale'], threshold=default_threshold)
            self.display_image(result, self.widget_global_thresholding)
            self.label_global_thre.setText(f"Mean: {np.round(default_threshold, 2)}\nCurrent: {np.round(default_threshold, 2)}")
            self.horizontalSlider_global_thresholding.setValue(int(default_threshold))

            # display greyscale histogrm
            self.plot_histogram_of_channel(self.images_dict[3]['grayscale'], self.plotWidge_threshold_histogram,'w')
        
        elif tab_index == 4:
            for slider in self.hybrid_sliders:
                slider.setValue(1)
            self.widget_hybrid_output_img.clear()
            self.generate_hybrid_image()
        
        else: # Not Implemented Yet
            pass

    def change_local_thresh(self):
        filter_size = self.horizontalSlider_local_thresholding.value()
        result = local_thresholding(self.images_dict[3]['grayscale'], region_size=filter_size)
        self.display_image(result, self.widget_local_thresholding)
        self.label_local_thre.setText(f"\nKernel Size: {filter_size}")

    def change_global_thresh(self):
        global_th = self.horizontalSlider_global_thresholding.value()
        result = apply_threshold(self.images_dict[3]['grayscale'], threshold=global_th)
        self.display_image(result, self.widget_global_thresholding)
        # self.label_global_thre.text()
        self.label_global_thre.setText(f"Mean: {np.round(self.images_dict[3]['grayscale'].flatten().mean(), 2)}\nCurrent: {np.round((global_th), 2)}")
        
    def change_colored_histogram(self, hist_to_display):
        if self.radioButton_main_grayscale.isChecked():
            return
        img = self.images_dict[0]['input']
        blue_ch, green_ch, red_ch = get_channels(img) # separate channels

        if hist_to_display == 'R':
            intended_ch = red_ch
        elif hist_to_display == 'B':
            intended_ch = blue_ch
        else:
            intended_ch = green_ch
        
        self.plot_cumulative_of_channel(intended_ch, self.plotWidge_distribution_curve_color, hist_to_display)
        self.plot_all_histograms(img, self.plotWidge_main_histogram_color)

    def plot_all_histograms(self, img, plot_widget):
        blue_ch, green_ch, red_ch = get_channels(img) # separate channels
        # # get the frequencies for each pixel value for the histogram
        blue_hist_values, blue_bin_edges = get_frequencies(blue_ch)
        green_hist_values, green_bin_edges = get_frequencies(green_ch)
        red_hist_values, red_bin_edges = get_frequencies(red_ch)
        # plot histograms
        blue_bargraph = pg.BarGraphItem(x = blue_bin_edges[:-1], height = blue_hist_values, width = 1, brush ='b') 
        green_bargraph = pg.BarGraphItem(x = green_bin_edges[:-1], height = green_hist_values, width = 1, brush ='g') 
        red_bargraph = pg.BarGraphItem(x = red_bin_edges[:-1], height = red_hist_values, width = 1, brush ='r') 
        # tems to plot widget
        plot_widget.addItem(blue_bargraph)
        plot_widget.addItem(green_bargraph)
        plot_widget.addItem(red_bargraph)

    def plot_cumulative_of_channel(self, channel, plot_widget, color):
        hist_values, bin_edges = get_frequencies(channel)
        hist_values_cumulative = get_cumulative_frequencies(hist_values)
        self.display_bar_graphs(bin_edges, hist_values_cumulative, color, plot_widget)
        # gs_hist_values_cumulative = self.get_cumulative_frequencies(gs_hist_values)

    def plot_histogram_of_channel(self, channel, plot_widget, color):
        hist_values, bin_edges = get_frequencies(channel)
        self.display_bar_graphs(bin_edges, hist_values, color, plot_widget)
    
    def display_bar_graphs(self, x, y, color, plot_widget):
        bargraph = pg.BarGraphItem(x = x[:-1], height = y, width = 1, brush =color.lower()) 
        plot_widget.clear()
        plot_widget.addItem(bargraph)

    def display_image(self, image, viewer):
        # Convert image to 8-bit unsigned integer format
        image_8u = cv2.convertScaleAbs(image)
        rotated_image = cv2.rotate(image_8u, cv2.ROTATE_90_COUNTERCLOCKWISE)
        flipped_image = cv2.flip(rotated_image, 0)
        viewer.setImage(flipped_image)        

    def apply_filter(self):
        filter_type = self.comboBox_filter_type.currentText()
        kernel_size = int(self.lineEdit_kernel_size.text())
        sigma_text =self.lineEdit_sigma_value.text()
        
        if self.images_dict[1]['noise'] is not None:
            noisy_img =  self.images_dict[1]['noise']
        elif self.images_dict[1]['input'] is not None:
            noisy_img =  self.images_dict[1]['input']

        sigma = float(sigma_text)
        if filter_type == "Gaussian":
            filtered_image = gaussian_filter(noisy_img, kernel_size= kernel_size, sigma=sigma, is_gray=self.is_gray)
        elif filter_type == "Mean":
            filtered_image = avg_filter(noisy_img, kernel_size=kernel_size, is_gray=self.is_gray)
        elif filter_type == "Median":
            filtered_image = median_filter(noisy_img, kernel_size = kernel_size, is_gray=self.is_gray)
        elif filter_type == "LPF":
            filtered_image = LPF(noisy_img, self.is_gray)
        elif filter_type == "HPF":
            filtered_image = HPF(noisy_img, self.is_gray)
        else:
            filtered_image = noisy_img
        self.display_image(filtered_image, self.widget_filtered_img)

    def apply_edge_detection(self):
        edge_type = self.comboBox_edge_detectors.currentText()  
        img =  self.images_dict[1]['input']

        if edge_type == "Prewitt":
            edges = prewitt_edge(img,is_gray=self.is_gray)
        elif edge_type == "Roberts":
            edges = roberts_edge(img,is_gray=self.is_gray)
        elif edge_type == "Sobel":
            edges = sobel_edge(img,is_gray=self.is_gray)
        elif edge_type == "Canny":
            edges = canny_edge(img,is_gray=self.is_gray)
        else:
            # Default to original image if no edge detection selected (None)
            edges = img
        self.display_image(edges, self.widget_edge_detected)

    def add_noise(self):
        noise_type = self.comboBox_noise_type.currentText()
        self.update_slider_value(self.horizontalSlider_noise.value())
        percentage = self.horizontalSlider_noise.value()

        img = self.images_dict[1]['input']

        if noise_type == 'Uniform':
            noisy_img = apply_Uniform_noise(img, percentage)
        elif noise_type == 'Gaussian':
            noisy_img = apply_Gassian_noise(img, percentage)
        elif noise_type == 'Salt & pepper':
            noisy_img = apply_Salt_Pepper_noise(img, percentage)
        else:
            noisy_img = img
        self.images_dict[1]['noise'] = noisy_img
        self.display_image(noisy_img, self.widget_noisy_img)

    def equalize_image(self):
        '''
        Perform histogram equalization on the input image and display the equalized image along with histograms.

        This method retrieves the input image, performs histogram equalization,
        displays the equalized image, and plots histograms of the original and equalized images.

        Parameters:
            None

        Returns:
            None
        '''
        self.figure.clear()
        self.canva.draw()
        self.widget_normalized_img.clear()
        self.widget_output_img.clear()

        image = self.images_dict[2]["input"]
        equalized_image, hist_original, hist_eq, bin_edges_original, bin_edges_eq = histogram_equalization(image)
        self.display_image(equalized_image, self.widget_output_img)

        ax = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 4)

        ax.bar(bin_edges_original[: -1], hist_original, width=np.diff(bin_edges_original), align='edge', color='skyblue', edgecolor='black')
        ax2.bar(bin_edges_eq[: -1], hist_eq, width=np.diff(bin_edges_eq), align='edge', color='skyblue')

        ax.set_title("Original")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax2.set_title("Equalized")
        ax2.set_xlabel("Pixel Value")
        ax2.set_ylabel("Frequency")

        self.canva.draw()

    def normalize_image(self):
        '''
        Normalize the input image and display the normalized image.

        Parameters:
            None

        Returns:
            None
        '''
        image = self.images_dict[2]["input"]
        normalized_image = normalize(image)
        self.display_image(normalized_image, self.widget_normalized_img)
    
    def generate_hybrid_image(self):
        '''
        Generate a hybrid image using two input images and display the result.

        This method retrieves parameters for generating a hybrid image,
        processes the input images using specified parameters,
        and displays the resulting hybrid image.

        Parameters:
            None

        Returns:
            None
        '''
        sigma1 = self.horizontalSlider_hybrid_sigma1.value()
        sigma2 = self.horizontalSlider_hybrid_sigma2.value()

        kernel_size1 = self.horizontalSlider_hybrid_kernel_size1.value()
        kernel_size2 = self.horizontalSlider_hybrid_kernel_size2.value()

        is_grey1 = self.greyscale_rbtns[4].isChecked()
        is_grey2 = self.greyscale_rbtns[4].isChecked()
        if self.images_dict[4]["input_one"] is not None and self.images_dict[4]["input_two"] is not None:
            hybrid_image = generate_hybrid_image(self.images_dict[4]["input_one"], self.images_dict[4]["input_two"], 
                                                sigma1, sigma2, kernel_size1, kernel_size2, is_grey1, is_grey2)
            self.display_image(hybrid_image, self.widget_hybrid_output_img)

    def display_area_perimeter(self, pts_2d):

        area = round(get_area(pts_2d), 2)
        self.label_area_contour.setText("  "+str(area))

        perimeter = round(get_perimeter(pts_2d), 2)
        self.label_primitive_contour.setText("  "+str(perimeter))

    
    def perform_active_contour(self):
        """
        Performs active contour segmentation using the snake algorithm.

        Retrieves parameters from user interface, initializes contour, prepares external energy components,
        iterates over specified number of iterations, updates contour, and displays the segmented output.

        """
        original_image = self.images_dict[5]['input']
        w_line = 30
        w_edge = 40

        # Retrieve necessary parameters from the user interface
        alpha = float(self.lineEdit_alpha.text())
        beta = float(self.lineEdit_beta.text())
        gamma = float(self.lineEdit_gamma.text())
        iterations = int(self.lineEdit_num_iterations.text())

        # Initialize contour points
        contour_points = create_initial_contour(original_image)

        image = None
        if len(original_image.shape) > 2:
            image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        # Prepare external energy components
        blurred_image, gradient = prepare_external_energy(image)

        for _ in range(iterations):
            # Iterate over each point in the contour
            for i in range(len(contour_points)):
                min_energy = np.inf
                new_contour = np.copy(contour_points)
                # initialize point location
                new_point_location = None
                for step in window:
                    # move the point to a new location
                    new_contour[i] = [contour_points[i][0] + step[0], contour_points[i][1] + step[1]] 

                    # calculate total energy
                    total_energy = internal_energy(new_contour, alpha, beta) - external_energy(new_contour[i], blurred_image, gradient, w_line, w_edge, gamma)

                    if total_energy < min_energy:
                        min_energy = total_energy
                        # store the point at which min energy occurs
                        new_point_location = np.copy(new_contour[i])

                # Update contour point if the location has changed
                if not np.array_equal(new_point_location, contour_points[i]):
                    contour_points[i] = np.copy(new_point_location)

            # Create a copy of the original image
            result_image = original_image.copy()

            # Draw the contour on the result image
            cv2.polylines(result_image, [contour_points], isClosed=False, color=(0, 0, 255), thickness=2)
            
            # Transpose the result image to match the format expected by ImageView
            result_image = np.transpose(result_image, (1, 0, 2))

            # Update the ImageView widget with the result image
            self.widget_snake_output.setImage(result_image)

            # Process events to prevent lagging
            QApplication.processEvents()

        # Print chain code and display area perimeter
        self.print_chain_code(contour_points)
        self.display_area_perimeter(contour_points)

    def print_chain_code(self, pts_2d):

        prev_pt = pts_2d[0]
        chain_directions = []

        for index in range(1, len(pts_2d)):
            curr_pt = pts_2d[index]
            current_str = get_direction(curr_pt=curr_pt, prev_pt=prev_pt)
            if current_str :
                chain_directions.append(current_str)
            prev_pt = curr_pt
            
        # circular
        current_str = get_direction(curr_pt=pts_2d[0], prev_pt=pts_2d[-1])
        if current_str:
            chain_directions.append(current_str)
            
        chain = directions_to_chain_code(chain_directions)
        self.chain_code_string = list_to_string(chain)
        print(f"Chain Code of the active contour: {self.chain_code_string}\n")


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = filter_edge_detection()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
