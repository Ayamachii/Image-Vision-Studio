import cv2
from PyQt5.QtGui import QPixmap, QImage
import numpy as np 
import time
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
# Define the folder containing the cascade XML files
cascade_folder = os.path.join(current_dir, "../Data")

def FaceDetection(img, sf):
    """
    Detects faces in the input image using Haar cascades.
    Args:
        img (numpy.ndarray): The input image in BGR or grayscale format.
        sf (float): The scale factor for multi-scale detection.
    Returns:
        tuple: A tuple containing:
            - QPixmap: A QPixmap object representing the image with detected faces.
            - float: The time taken for face detection in seconds.
            - int: The number of faces detected in the image.
    """
    start_time = time.time()  
    image_copy = img.copy()
    haar_cascade_face = cv2.CascadeClassifier(os.path.join(cascade_folder, "haarcascade_frontalface_alt2.xml"))
    haar_cascade_SideFace = cv2.CascadeClassifier(os.path.join(cascade_folder, "haarcascade_frontalface_default.xml"))
    if len(image_copy.shape) > 2 and image_copy.shape[2] == 3:
        gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    elif len(image_copy.shape) == 2:
        gray_image = image_copy
    else:
        raise ValueError("Unsupported image format")    
    faces_rect = haar_cascade_face.detectMultiScale(gray_image, scaleFactor=sf, minNeighbors=3)
    no_faces = len(faces_rect)
    if no_faces == 0:
        SideFace = haar_cascade_SideFace.detectMultiScale(gray_image, scaleFactor=sf, minNeighbors=1)
        print("Side faces Found:", len(SideFace))
        for (sx, sy, sw, sh) in SideFace:
            cv2.rectangle(image_copy, (sx, sy), (sx+sw, sy+sh), (255, 255, 255), 2)
    else:
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 0, 0), 3)
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    end_time = time.time()  
    detection_time = end_time - start_time
    output_pixmap = QPixmap.fromImage(QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], QImage.Format_RGB888))
    return output_pixmap, detection_time, no_faces

def qpixmap_to_nparray(qpixmap):
    """
    Convert a QPixmap object to a numpy array in RGB format.
    Args:
        qpixmap (QPixmap): The QPixmap object to be converted.
    Returns:
        numpy.ndarray: A numpy array representing the image in RGB format.
    """
    image = qpixmap.toImage()
    width = image.width()
    height = image.height()
    bgra = image.bits().asstring(width * height * 4)
    arr = np.frombuffer(bgra, dtype=np.uint8).reshape((height, width, 4))
    arr = arr[..., :3]  # Keep only RGB channels, discard alpha channel
    arr = np.ascontiguousarray(arr)  # Ensure data is contiguous in memory
    return arr
