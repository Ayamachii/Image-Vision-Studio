import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QGridLayout
import cv2
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage.transform import resize
from math import sqrt
from functions.filters_edges import * # just for the try file, will meed to be modified
# from functions.filters_edges import *
import time
import numpy.linalg as LA
from scipy import signal

class SIFTMatcher:
    def __init__(self):
        pass

    def match_descriptors(self, descriptors1, descriptors2, num_matches, keypoints1, keypoints2):
        matches = []
        for idx1, (desc1, point1)in enumerate(descriptors1):
            best_match_point2 = self.find_best_match(desc1, descriptors2, point1)
            if best_match_point2 is not None:
                # matches.append((idx1, best_match_idx))
                for i, (x_keypoint, y_keypoint) in enumerate(keypoints1):
                    if x_keypoint == point1[0] and y_keypoint == point1[1]:
                        point1_index = i
                for i, (x_keypoint, y_keypoint) in enumerate(keypoints2):
                    if x_keypoint == best_match_point2[0] and y_keypoint == best_match_point2[1]:
                        point2_index = i
                matches.append((point1_index, point2_index))
            if len(matches) >= num_matches:
                break
        return matches

    def find_best_match(self, descriptor, descriptors, point1):
        best_match_idx = None
        best_distance = float('inf')
        for idx, (desc, point2) in enumerate(descriptors):
            # we can change calculate_SSD to calculate_NCC 
            distance = self.calculate_SSD(descriptor, desc)
            if distance < best_distance:
                best_distance = distance
                best_match_idx = idx
                best_matched_point2 = point2
        return best_matched_point2

    def draw_matches(self, image1, keypoints1, image2, keypoints2, matches):
        keypoints_list1 = []
        keypoints_list2 = []
        # Convert matches to list of DMatch objects
        dmatches = [cv2.DMatch(_queryIdx=index1, _trainIdx=index2, _distance=0) for index1, index2 in matches]

        keypoints_list1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints1]
        keypoints_list2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints2]

        img_match = np.empty((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3),dtype=np.uint8)

        # convert keypoints_list to a std::vector<KeyPoint> object
        # keypoints_vector1 = cv2.KeyPoint_convert(keypoints_list1)
        # keypoints_vector2 = cv2.KeyPoint_convert(keypoints_list2)

        img_match = cv2.drawMatches(image1, keypoints_list1, image2, keypoints_list2, dmatches, outImg=img_match, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_match

    def calculate_SSD(self, descriptor1, descriptor2):
            """
            Compute the Sum of Squared Differences (SSD) between two descriptors.

            Args:
            - descriptor1: The first descriptor.
            - descriptor2: The second descriptor.

            Returns:
            - ssd_value: The SSD value between the two descriptors.
            """
            descriptor1 = np.array(descriptor1)
            descriptor2 = np.array(descriptor2)
            ssd_value = np.sum((descriptor1 - descriptor2)**2)
            return ssd_value

    def calculate_NCC(self, descriptor1, descriptor2):
        """
        Compute the Normalized Cross-Correlation (NCC) between two descriptors.

        Args:
        - descriptor1: The first descriptor.
        - descriptor2: The second descriptor.

        Returns:
        - ncc_value: The NCC value between the two descriptors.
        """
        descriptor1 = np.array(descriptor1)
        descriptor2 = np.array(descriptor2)
        
        mean1 = np.mean(descriptor1)
        mean2 = np.mean(descriptor2)
        
        std1 = np.std(descriptor1)
        std2 = np.std(descriptor2)
        
        ncc_value = np.sum((descriptor1 - mean1) * (descriptor2 - mean2)) / (std1 * std2 * descriptor1.size)
        return ncc_value

k = sqrt(2)
sigma = 1.6
num_scales = 5 
num_octaves = 4
# [sigma, sqrt(2)*sigma, 2*sigma, 2sqrt(2)*sigma, 4*sigma]
sigma_values = [(k**i)*sigma for i in range(num_scales)]
# sigma_values = [sigma, k*sigma, 2*sigma, 2*k*sigma, 2*k*k*sigma]
octaves = []
diff_of_gaussian = []

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIFT")
        self.setGeometry(100, 100, 800, 600)
        print("in main")

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.vlayout = QVBoxLayout(self.central_widget)
        self.hlayout = QHBoxLayout(self.central_widget)
        self.imgv1 = pg.ImageView()
        self.imgv2 = pg.ImageView()
        self.imgv1.ui.histogram.hide()
        self.imgv1.ui.roiBtn.hide()
        self.imgv2.ui.histogram.hide()
        self.imgv2.ui.roiBtn.hide()
        # Disable the default context menu
        self.imgv1.ui.menuBtn.hide()
        self.hlayout.addWidget(self.imgv1)
        self.vlayout.addLayout(self.hlayout)
        self.imgv2.ui.menuBtn.hide()
        self.hlayout.addWidget(self.imgv2)
        self.vlayout.addLayout(self.hlayout)
        # Create a grid layout
        self.grid = QGridLayout(self.central_widget)

        # Create a 4x5 array of QLabel
        self.img_views = [[pg.ImageView() for _ in range(5)] for _ in range(4)]

        # Add the QLabel to the grid layout
        for i in range(4):
            for j in range(5):
                self.img_views[i][j].ui.histogram.hide()
                self.img_views[i][j].ui.roiBtn.hide()
                # Disable the default context menu
                self.img_views[i][j].ui.menuBtn.hide()
                self.grid.addWidget(self.img_views[i][j], i, j)

        self.vlayout.addLayout(self.grid)

        # image1 = cv2.imread("test-images/hybrid/bird.jpg")
        image1 = cv2.imread("D:/ComputerVision/Filtering_and_Edge_Detection_Studio/test-images/small.jpg")
        image2 = cv2.imread("D:/ComputerVision/Filtering_and_Edge_Detection_Studio/test-images/12.jpg")
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        self.imgv1.setImage(np.transpose(image1, (1, 0)))
        self.imgv2.setImage(np.transpose(image2, (1, 0)))

        self.dominant_dir_list = []
        start_time = time.time()
        list_DoGs, list_octaves = self.construct_scale_space(image=image1)
# ---------------------------------edited--------------------------------
        list_DoGs = np.array(list_DoGs, dtype=object)
        list_octaves = np.array(list_octaves, dtype=object)

        list_DoGs2 = np.array(list_DoGs, dtype=object)
        list_octaves2 = np.array(list_octaves, dtype=object)

        # new_img = self.compare_with_neighbors(list_DoGs[0])

        for ind, octave in enumerate(list_DoGs.copy()):
            list_DoGs[ind] = self.compare_with_neighbors(octave)
            for index, stage in enumerate(list_DoGs[ind]):
                print(f"Number of key points in octave {ind}, stage {index}: {np.count_nonzero(stage)}")
        

        # list_DoGs2 = np.array(list_DoGs)
        # list_octaves2 = np.array(list_octaves)

        # new_img = self.compare_with_neighbors(list_DoGs[0])

        for ind, octave in enumerate(list_DoGs2.copy()):
            list_DoGs2[ind] = self.compare_with_neighbors(octave)
            for index, stage in enumerate(list_DoGs2[ind]):
                print(f"Number of key points2 in octave {ind}, stage {index}: {np.count_nonzero(stage)}")

        # list_DoGs is a numpy array of shape (4x4)
        # list_DoGs[0] i the first octave (of highest resolution (size), list_DoGs[1] is the
        # second best resolution (half sized) and so on)
        # list_DoGs[0][1] is the first scale (smallest sigma), list_DoGs[0][1] and so on
        # to get the corresponding sigma to a point use the second index of list_DoGs and
        # get the corresponding element in sigma_values list defined above
        # the corresponding scale is the first index of list_DoGs
        #ex list_DoGs[2][3], is the image with the keypoints are the pixels
        # whose values are not equal to zero, octave is 2, scale is three of sigma = 
        # sigma_values[3]

        # (octave index, sigma of stage, np array of keypoints [(x1, y1), (x2, y2), ()])
        # keypoints_list contins 16 elements which are tuples carrying the prev mentioned info
        # keypoints_list = []
        # for octave_ind in range(4):
        #     for stage_ind in range(4):
        #         rows_ind, cols_ind = np.where(np.abs(list_DoGs[octave_ind][stage_ind])>0)
        #         list_of_x_y_tuples = list(zip(rows_ind, cols_ind))

        #         current_tuple = (octave_ind, sigma_values[stage_ind], list_of_x_y_tuples)
        #         keypoints_list.append(current_tuple)

                
        self.orientations = []
        keypoints_list = self.get_keypoints(list_DoGs)
        self.orientation_assignment(keypoints_list, list_octaves)
        # keypoints_list = np.array(keypoints_list)

        descriptors = self.compute_descriptors(keypoints_list, list_octaves)

        keypoints_list2 = self.get_keypoints(list_DoGs2)
        self.orientation_assignment(keypoints_list2, list_octaves2)
        # keypoints_list = np.array(keypoints_list)

        descriptors2 = self.compute_descriptors(keypoints_list2, list_octaves2)

        end_time = time.time()

        runtime = end_time - start_time
        print(f" total runtime is {runtime} seconds.")

        keypoints_list_original1 = []
        for i, (octave_idx, _, coordinates) in enumerate(keypoints_list):
            for coord in (coordinates):
                x, y = coord
                if octave_idx == 0:
                    original_index = (x//2, y//2)
                elif octave_idx == 1:
                    original_index = coord
                elif octave_idx == 2:
                    original_index = (x*2, y*2)
                elif octave_idx == 3:
                    original_index = (x*4, y*4)
                keypoints_list_original1.append(original_index)

        keypoints_list_original2 = []
        for i, (octave_idx, _, coordinates) in enumerate(keypoints_list2):
            for coord in (coordinates):
                x, y = coord
                if octave_idx == 0:
                    original_index = (x//2, y//2)
                elif octave_idx == 1:
                    original_index = coord
                elif octave_idx == 2:
                    original_index = (x*2, y*2)
                elif octave_idx == 3:
                    original_index = (x*4, y*4)
                keypoints_list_original2.append(original_index)

        # Usage example:
        matcher = SIFTMatcher()
        # Assume descriptors1 and descriptors2 are lists of descriptors for two images
        matches = matcher.match_descriptors(descriptors, descriptors2, 30 , keypoints_list_original1, keypoints_list_original2)

        print(f'keypoint 1 trial {keypoints_list_original1[1]}')

        end_time = time.time()

        runtime = end_time - start_time
        print(f" Matching using NCC took {runtime} seconds.")

        # Draw lines between matched keypoints
        matched_keypoints_img = matcher.draw_matches(image1, keypoints_list_original1, image2, keypoints_list_original2, matches)

        # Display the image with matched keypoints and lines
        plt.imshow(matched_keypoints_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt.show()

    def gaussian_kernel1d(self, kernlen=7, std=1.5):
        """Returns a 1D Gaussian window."""
        kernel1d = signal.windows.gaussian(kernlen, std=std)
        kernel1d = kernel1d.reshape(kernlen, 1)
        return kernel1d / kernel1d.sum()

    def gaussian_kernel2d(self, kernlen=7, std=1.5):
        """Returns a 2D Gaussian kernel array."""
        gkern1d = self.gaussian_kernel1d(kernlen,std)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def construct_scale_space(self, image):
        start_time = time.time()

        # image upsampled by a factor of 2
        base_image = np.copy(image)
        base_image = base_image.astype('float32')
        base_image = resize( image=image, output_shape=(image.shape[0]*2, image.shape[1]*2))

        for i in range(0, num_octaves):
            octaves.append([ cv2.GaussianBlur(base_image, (0, 0), sigma) 
                    for sigma in sigma_values ])
            # octaves.append([ gaussian_filter(base_image, kernel_size=3, sigma=sigma, is_gray=True) 
            #         for sigma in sigma_values ])
            diff_of_gaussian.append([ scale2 - scale1
                        for (scale1, scale2) in zip( octaves[i][:-1], octaves[i][1:])])
            base_image = resize(image=octaves[i][2], output_shape=(octaves[i][2].shape[0] // 2, octaves[i][2].shape[1]//2), anti_aliasing=True)

        # ---------------- show octaves ----------------------------
        # for octave_idx in range(num_octaves):
        #     print(octave_idx)
        #     img_octave = octaves[octave_idx]
        #     for scale_idx in range(num_scales):
        #         print(scale_idx)
        #         self.img_views[octave_idx][scale_idx].setImage(np.transpose(img_octave[scale_idx], (1, 0)))
        #         cv2.imwrite(f"oct{octave_idx}_sca{scale_idx}.png", img_octave[scale_idx])
        # ---------------- show DoG ----------------------------
        for octave_idx in range(num_octaves):
            # print(octave_idx)
            dog_octave = diff_of_gaussian[octave_idx]
            for i in range(4):
                # print(i)
                self.img_views[octave_idx][i].setImage(np.transpose(dog_octave[i], (1, 0)))
                cv2.imwrite(f"extra/oct{octave_idx}_sca{i}.jpg", dog_octave[i])
        
        end_time = time.time()

        runtime = end_time - start_time

        print(f"The runtime of construct_scale_space is {runtime} seconds.")
        return diff_of_gaussian , octaves

    def compare_with_neighbors(self, stage_list):
        initial_stage_list = stage_list.copy()
        for current_stage in [1, 2]:
            current_img = initial_stage_list[current_stage]
            prev_img = initial_stage_list[current_stage-1]
            next_img = initial_stage_list[current_stage+1]
            
            img_height = current_img.shape[0]
            img_width = current_img.shape[1]

            for i in range(1, img_height-1):
                for j in range(1, img_width-1):
                    center_value = current_img[i,j]
                    if np.abs(center_value) < 0.03:
                        current_img[i,j] = 0
                        continue

                    prev_kernel = prev_img[i-1:i+2,j-1:j+2]
                    curr_kernel = current_img[i-1:i+2,j-1:j+2]
                    next_kernel = next_img[i-1:i+2,j-1:j+2]
                    
                    all_values = np.concatenate((prev_kernel.flatten(), next_kernel.flatten(), curr_kernel.flatten()))
                    if center_value != np.max(all_values) or center_value != np.min(all_values):
                        current_img[i,j] = 0

            stage_list[current_stage] = current_img
            #check img edges
            stage_list[0] = self.compare_two_stages(initial_stage_list[0], initial_stage_list[1])
            stage_list[3] = self.compare_two_stages(initial_stage_list[3], initial_stage_list[2])
    
        return stage_list

    def compare_two_stages(self, original_stage, second_stage):
            # print("comparing the terminal stages, should run 8 times, wait")
            img_height = original_stage.shape[0]
            img_width = original_stage.shape[1]
            for i in range(1, img_height-1):
                for j in range(1, img_width-1):
                    center_value = original_stage[i,j]

                    sec_kernel = second_stage[i-1:i+2,j-1:j+2]
                    first_kernel = original_stage[i-1:i+2,j-1:j+2]
                    
                    all_values = np.concatenate((first_kernel.flatten(), sec_kernel.flatten()))
                    if center_value != np.max(all_values) or center_value != np.min(all_values):
                        original_stage[i,j] = 0

            return original_stage
    
    def get_keypoints(self, list_DoGs):
        keypoints_list = []
        for octave_ind in range(4):
            for stage_ind in range(4):
                rows_ind, cols_ind = np.where(np.abs(list_DoGs[octave_ind][stage_ind])>0)
                list_of_x_y_tuples = list(zip(rows_ind, cols_ind))

                current_tuple = (octave_ind, sigma_values[stage_ind], list_of_x_y_tuples)
                keypoints_list.append(current_tuple)

        keypoints_list = np.array(keypoints_list, dtype=object)
        return keypoints_list

    def orientation_assignment(self, keypoints_list, list_octaves):
        for octave_index in range(num_octaves):
            for i, keypoint_info in enumerate(keypoints_list):
                octave_ind, sigma_val, keypoint_coords = keypoint_info
                if octave_ind == octave_index:
                    print(f"in orientation assignment point {i}")
                    octave = list_octaves[octave_ind]
                    # scale = sigma_values.index(sigma_val)
                    scale = octave[sigma_values.index(sigma_val)]
                    sigma = 1.5 * sigma_val
                    radius = int(2 * sigma)
                    kernel_size = 2 * radius
                    gauss_kernel = self.gaussian_kernel2d(kernlen=kernel_size, std=sigma)
                    print(f"gaussian_kernel size is {gauss_kernel.shape}")
                    gradient_magnitude, gradient_orientation = self.compute_gradient_magnitude_orientation(scale)
                    for coord in keypoint_coords:
                        x, y = coord
                        orientation_pin_idx = np.round( gradient_orientation * 36 / 360 ).astype(int)
                        window = [x-radius, x+radius, y-radius, y+radius]
                        magnitude_window = self.extract_region( gradient_magnitude , window )
                        orientation_idx = self.extract_region( orientation_pin_idx, window )
                        # **********************************************************************************
                        weighted_magnitude = magnitude_window * gauss_kernel 
                        histogram = np.zeros(36, dtype=np.float32)
                        for bin_idx in range(36):
                            histogram[bin_idx] = np.sum( weighted_magnitude[ orientation_idx == bin_idx ] )
                        dominant_orientation = np.argmax(histogram) * (360/36) % 360
                        # print(f"list of first dominant orientation {[coord, dominant_orientation]}")
                        self.dominant_dir_list.append([coord, dominant_orientation])

    def compute_gradient_magnitude_orientation(self, image):
        dx = np.array([[1,0,-1],
                [2,0,-2],
                [1,0,-1]])
        dy = dx.T
        gx = signal.convolve2d( image , dx , boundary='symm', mode='same' )
        gy = signal.convolve2d( image , dy , boundary='symm', mode='same' )
        magnitude = np.sqrt( gx * gx + gy * gy )
        orientation = np.rad2deg( np.arctan2( gy , gx )) % 360
        return magnitude, orientation

    def extract_region(self, image, window):
        output_shape = np.asarray(np.shape(image))
        output_shape[0] = window[1] - window[0]
        output_shape[1] = window[3] - window[2]
        src = [max(window[0], 0),
            min(window[1], image.shape[0]),
            max(window[2], 0),
            min(window[3], image.shape[1])]
        dst = [src[0] - window[0], src[1] - window[0],
            src[2] - window[2], src[3] - window[2]]
        output = np.zeros(output_shape, dtype=image.dtype)
        output[dst[0]:dst[1],dst[2]:dst[3]] = image[src[0]:src[1],src[2]:src[3]]
        return output
    
    def compute_descriptors(self, keypoints, list_octaves):
        # (octave_ind, sigma_values[stage_ind], list_of_x_y_tuples)
        descriptors = []
        for i, (octave_idx, sigma_val, coordinates) in enumerate(keypoints):
            print(f'in compute descriptors keypoint {i}')
            for coord in (coordinates):
                x, y = coord
                octave = list_octaves[octave_idx]
                scale = octave[sigma_values.index(sigma_val)]
                gradient_magnitude, gradient_orientation = self.compute_gradient_magnitude_orientation(scale)
                kernel = self.gaussian_kernel2d(kernlen=16, std=1.5 * sigma_val)
                print(f'center x is {x}, center y is {y}')
                # magnitude_window = self.extract_window( gradient_magnitude , center_x=x, center_y=y, window_size=16 )
                radius = 8
                window = [x-radius, x+radius, y-radius, y+radius]
                magnitude_window = self.extract_region(gradient_magnitude, window)
                orientation_window = self.extract_region( gradient_orientation, window )
                # orientation_window = self.extract_window( gradient_orientation, center_x=x, center_y=y, window_size=16 )
                weighted_magnitude = magnitude_window * kernel
                # subtract dominant orientation
                orientation_pin_idx = (((orientation_window - self.dominant_dir_list[i][1]) % 360) * 8 / 360).astype(int)
                orientation_pin_idx = np.clip(orientation_pin_idx, 0, None)
                features = []
                for sub_region_i in range(4):
                    for sub_region_j in range(4):
                        sub_magnitude = weighted_magnitude[sub_region_i*4:(sub_region_i+1)*4, sub_region_j*4:(sub_region_j+1)*4]
                        sub_orientation_idx = orientation_pin_idx[sub_region_i*4:(sub_region_i+1)*4, sub_region_j*4:(sub_region_j+1)*4]
                        histogram = np.zeros(8, dtype=np.float32)
                        for bin_idx in range(8):
                            histogram[bin_idx] = np.sum( sub_magnitude[ sub_orientation_idx == bin_idx ] )
                        features.extend( histogram.tolist())
                features = np.array(features) 
                features /= (np.linalg.norm(features))
                np.clip( features, a_min=np.finfo(np.float16).eps, a_max=None, out = features )
                print(f"num of features for {coord} is {features.shape}")
                if octave_idx == 0:
                    original_index = (x//2, y//2)
                elif octave_idx == 1:
                    original_index = coord
                elif octave_idx == 2:
                    original_index = (x*2, y*2)
                elif octave_idx == 3:
                    original_index = (x*4, y*4)

                descriptors.append([features, original_index])
        return descriptors

    def extract_window(self, image, center_x, center_y, window_size=16):
        # Calculate the top-left corner coordinates of the window
        start_x = max(center_x - window_size, 0)
        start_y = max(center_y - window_size, 0)
        
        # Calculate the bottom-right corner coordinates of the window
        end_x = min(center_x + window_size, image.shape[1])
        end_y = min(center_y + window_size, image.shape[0])
        print(f"start is {start_x}, {start_y} end is {end_x}, {end_y}")
        
        # Extract the window from the image
        window = image[start_x:end_x, start_y:end_y]
        
        return window
    
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())