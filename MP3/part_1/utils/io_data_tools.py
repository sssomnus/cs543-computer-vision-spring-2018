"""Input and output helpers to load and preprocess images."""

import cv2

def load_and_grayscale(root, path):
    # Load image.
    img = cv2.imread(root+path)
    # Convert to grayscale.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    return img

def load_and_normalize(root, path):
    # Load image.
    img = cv2.imread(root+path)
    # Convert to double and normalize.
    img = cv2.normalize(img.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    return img
