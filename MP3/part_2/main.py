import sys
from utils.io_data_tools import load_plot_data, plot_epipolar
from feature_matching import sift_descriptors, get_matched_pixels
from fit_fundamental_tools import fit_fundamental, get_geo_distance
from ransac import ransac_fitting, plot_inlier_matches
from triangulation_tools import triangulate, get_residual, plot_3d

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    """High level pipeline.
    
    To run the program, type in command line:
        1. If you want to fit fundamental matrix:
            1)  with ground truth labels (normalized & unnormalized)
                "python3 main.py fit_fundamental gt"
            2)  no ground truth labels (using RANSAC)
                "python3 main.py fit_fundamental no_gt"
        2. If you want to perform triangulation:
            "python3 main.py triangulation"  
    """
    # Read arguments and check correctness.
    if len(sys.argv) == 3:
        function = str(sys.argv[1])
        method = str(sys.argv[2])
    elif len(sys.argv) == 2:
        function = str(sys.argv[1])
    else:
        print("Please type in correct arguments.")
        sys.exit(1)


    # Load images and matches.
    root = '/Users/ziyu/Desktop/Study Affairs/2018-Spring/CS-543-Computer-Vision/MP3/part_2/'
    im_name = 'house'  # 'house' or 'library'
    im_1, im_2, matches = load_plot_data(root, im_name)

    # Procedure for different functions.
    if function == 'fit_fundamental' and len(sys.argv) == 3:
        if method == 'gt':
            print("Fundamental fitting with ground truth.")
            # Unnormalize.
            # Use homogeneous setup.
            F = fit_fundamental(matches, normalize=False, setup='non-homogeneous')
            plot_epipolar(matches, F, im_2, root, im_name+'_epipolar_unnorm_nonhomo')
            dist1, dist2 = get_geo_distance(matches, F)
            print("Unnormalized. Avg dist1: {}, Avg dist2: {}"
            .format(dist1, dist2))

            # Normalize.
            F = fit_fundamental(matches, normalize=True, setup='non-homogeneous')
            plot_epipolar(matches, F, im_2, root, im_name+'_epipolar_norm_nonhomo')
            dist1, dist2 = get_geo_distance(matches, F)
            print("Normalized. Avg dist1: {}, Avg dist2: {}"
            .format(dist1, dist2))

        elif method == 'no_gt':
            print("Fundamental fitting without ground truth.")
            # Parameters.
            # t_match = 30000  # Threshold for matching features (house).
            # t_ransac = 150  # Threshold for RANSAC (house).

            # t_match = 20000  # Threshold for matching features (library).
            # t_ransac = 120  # Threshold for RANSAC (library).

            # t_match = 30000  # For house.
            # t_ransac = 0.03

            t_match = 20000  # For library.
            t_ransac = 0.008

            # SIFT feature detection.
            kp_1, desp_1 = sift_descriptors(im_1)
            kp_2, desp_2 = sift_descriptors(im_2)
            
            # Feature matching.
            match_coords = get_matched_pixels(t_match, kp_1, kp_2, 
                                            desp_1, desp_2)
            print(match_coords.shape)

            # RANSAC alignment.
            inliers, F, avg_residual = ransac_fitting(match_coords, t_ransac)
            print("Number of inliers: {}, Average residual: {}"
            .format(len(inliers), avg_residual))
            # Fit fundamental.
            # F = fit_fundamental(inliers, normalize=True, setup='homogeneous')
            # dist1, dist2 = get_geo_distance(inliers, F)
            # print("Avg dist1: {}, Avg dist2: {}".format(dist1, dist2))
            # Plot inlier matches.
            plot_inlier_matches(inliers, root, im_name)
            # Plot epipolar.
            plot_epipolar(inliers, F, im_2, root, im_name+'_epipolar_ransac')

            # # RANSAC alignment.
            # inliers, H, avg_residual = ransac_fitting(match_coords, t_ransac)
            # print("Number of inliers: {}, Average residual: {}"
            # .format(len(inliers), avg_residual))
            # # Fit fundamental.
            # F = fit_fundamental(inliers, normalize=True, setup='homogeneous')
            # dist1, dist2 = get_geo_distance(inliers, F)
            # print("Avg dist1: {}, Avg dist2: {}".format(dist1, dist2))
            # # Plot inlier matches.
            # plot_inlier_matches(inliers, root, im_name)
            # # Plot epipolar.
            # plot_epipolar(inliers, F, im_2, root, im_name+'_epipolar_ransac')

        else:
            print("Please type in correct method name.")

    elif function == 'triangulation' and len(sys.argv) == 2:
        print("Triangulation.")
        # Load data.
        camera1 = np.loadtxt(root+'data/'+im_name+'1_camera.txt')
        camera2 = np.loadtxt(root+'data/'+im_name+'2_camera.txt')
        # Triangulation.
        center1, center2, X_3D = triangulate(camera1, camera2, matches)
        avg_res1, avg_res2 = get_residual(camera1, camera2, X_3D, matches)
        print("Avg residuals: {}, {}".format(avg_res1, avg_res2))
        # Plot 3D reconstruction.
        plot_3d(center1, center2, X_3D)

    else:
        print("Please type in correct arguments.")
        sys.exit(1)



if __name__ == '__main__':
    main()
    print("Done! :)")