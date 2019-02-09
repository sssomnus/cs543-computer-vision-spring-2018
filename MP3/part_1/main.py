from utils.io_data_tools import load_and_grayscale, load_and_normalize
from feature_matching import sift_descriptors, get_matched_pixels, plot_save_sift
from ransac import ransac_fitting, plot_inlier_matches
from stitching import stitch_img, plot_save_stitch


def main():
    """High level pipeline."""
    # Parameters.
    t_match = 7000  # Threshold for matching features.
    t_ransac = 0.5  # Threshold for RANSAC.

    # Read images and convert to grayscale.
    # (Must use full path for imread to work. Don't know why...)
    root = '/Users/ziyu/Desktop/Study Affairs/2018-Spring/CS-543-Computer-Vision/MP3/part_1/'
    path1 = 'images/uttower_left.JPG'
    path2 = 'images/uttower_right.JPG'
    im_left = load_and_grayscale(root, path1)
    im_right = load_and_grayscale(root, path2)
    
    # SIFT feature detection.
    kp_l, desp_l = sift_descriptors(im_left)
    kp_r, desp_r = sift_descriptors(im_right)
    # Plot keypoints.
    plot_save_sift(kp_l, im_left, root, 'outputs/sift_left.jpg')
    plot_save_sift(kp_r, im_right, root, 'outputs/sift_right.jpg')

    # Feature matching.
    match_coords = get_matched_pixels(t_match, kp_l, kp_r, 
                                    desp_l, desp_r)
    
    # RANSAC alignment.
    inliers, H, avg_residual = ransac_fitting(match_coords, t_ransac)
    # Plot inlier matches.
    plot_inlier_matches(inliers.astype(int), root, path1, path2)

    # Load original color images and normalize.
    im_l_color = load_and_normalize(root, path1)
    im_r_color = load_and_normalize(root, path2)
    # Stitch images.
    stitched = stitch_img(im_l_color, im_r_color, H)
    # Plot panorama.
    plot_save_stitch(stitched, root, 'outputs/uttower_stitched.jpg')


if __name__ == '__main__':
    main()
    print("Done! :)")
