import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def sift_descriptors(img):
    """Helper function for get_matched_pixels().
    Find keypoints and descriptors using SIFT.
    Note that img must be grayscale and in CV_8U type, because
    the SIFT funciton only accept this (not double precision)
    """
    
    print("Finding keypoints and descriptors...")
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, desp = sift.detectAndCompute(img,None)
    return keypoints, desp


def plot_save_sift(keypoints, img, root, path):
    out = img.copy()
    out = cv2.drawKeypoints(img, keypoints, out, 
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)   
    plt.imshow(out)
    plt.axis('off')
    cv2.imwrite(root+path, out)
    plt.show()


def get_matched_pixels(threshold, kp1, kp2, desp1, desp2):
    """Find matching descriptors using Euclidean distance.
    
    Args:
        threshold(float): To select matched pairs.
        kp1, kp2(KeyPoint structure): Keypoints of two images.
        desp1, desp2(numpy.ndarray): descriptors, dims (#keypoints, 128).
    Returns:
        match_coords(numpy.ndarray): Coordinates of the matched pixels in 
            pairs, dims (#matched pixels, 4), where each row is in the 
            form of [x1, y1, x2, y2]
    """
    
    print("Matching features...")
    # Pair distance with shape (desp1.shape[0], desp2.shape[0]).
    pair_dist = distance.cdist(desp1, desp2, 'sqeuclidean')
    # Get matched descriptors.
    desp1_idx = np.where(pair_dist < threshold)[0]
    desp2_idx = np.where(pair_dist < threshold)[1]
    # Find the corresponding keypoint coordinates.
    coord1 = np.array([kp1[idx].pt for idx in desp1_idx])
    coord2 = np.array([kp2[idx].pt for idx in desp2_idx])
    match_coords = np.concatenate((coord1, coord2), axis=1)

    return match_coords





