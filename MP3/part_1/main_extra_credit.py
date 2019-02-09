from utils.io_data_tools import load_and_grayscale, load_and_normalize
from feature_matching import sift_descriptors, get_matched_pixels, plot_save_sift
from ransac import ransac_fitting, plot_inlier_matches
from stitching import stitch_img, plot_save_stitch
# from stitching_extra import stitch_img_extra


def main():
    """High level pipeline."""
    # Parameters.
    t_match = 7000  # Threshold for matching features.
    t_ransac = 0.5  # Threshold for RANSAC.

    # Read images and convert to grayscale.
    # (Must use full path for imread to work. Don't know why...)
    root = '/Users/ziyu/Desktop/Study Affairs/2018-Spring/CS-543-Computer-Vision/MP3/part_1/'
    images = ['hill', 'ledge', 'pier']
    im = images[1]  # Choose from 0, 1, 2 to change images.


    # -------------------------- Stitch 1 & 2 -------------------------- #
    path1 = 'images/'+im+'/1.JPG'
    path2 = 'images/'+im+'/2.JPG'
    im_1 = load_and_grayscale(root, path1)
    im_2 = load_and_grayscale(root, path2)
    
    # SIFT feature detection.
    kp_1, desp_1 = sift_descriptors(im_1)
    kp_2, desp_2 = sift_descriptors(im_2)
    # Plot keypoints.
    plot_save_sift(kp_1, im_1, root, 'outputs/extra_credit/'+im+'/sift_1.jpg')
    plot_save_sift(kp_2, im_2, root, 'outputs/extra_credit/'+im+'/sift_2.jpg')

    # Feature matching.
    match_coords = get_matched_pixels(t_match, kp_1, kp_2, 
                                    desp_1, desp_2)
    
    # RANSAC alignment.
    inliers, H, avg_residual = ransac_fitting(match_coords, t_ransac)
    # Plot inlier matches.
    plot_inlier_matches(inliers.astype(int), root, path1, path2)

    # Load original color images and normalize.
    im_1_color = load_and_normalize(root, path1)
    im_2_color = load_and_normalize(root, path2)
    # Stitch images.
    stitched = stitch_img(im_1_color, im_2_color, H)
    # Plot panorama.
    print("Please press ESC to quit preview, otherwise OpenCV will be stuck there.")
    plot_save_stitch(stitched, root, 'outputs/extra_credit/'+im+'/stitched_12.jpg')


    # -------------------------- Stitch 12 & 3 -------------------------- #

    path12 = 'outputs/extra_credit/'+im+'/stitched_12.jpg'
    im_12 = load_and_grayscale(root, path12)

    path3 = 'images/'+im+'/3.JPG'
    im_3 = load_and_grayscale(root, path3)

    # SIFT feature detection.
    kp_12, desp_12 = sift_descriptors(im_12)
    kp_3, desp_3 = sift_descriptors(im_3)
    # Plot keypoints.
    plot_save_sift(kp_12, im_12, root, 'outputs/extra_credit/'+im+'/sift_12.jpg')
    plot_save_sift(kp_3, im_3, root, 'outputs/extra_credit/'+im+'/sift_3.jpg')

    # Feature matching.
    match_coords = get_matched_pixels(t_match, kp_12, kp_3, 
                                    desp_12, desp_3)
    
    # RANSAC alignment.
    inliers, H, avg_residual = ransac_fitting(match_coords, t_ransac)
    # Plot inlier matches.
    # plot_inlier_matches(inliers.astype(int), root, path12, path3)

    # Load original color images and normalize.
    im_12_color = load_and_normalize(root, path12)
    im_3_color = load_and_normalize(root, path3)
    # Stitch images.
    stitched = stitch_img(im_12_color, im_3_color, H)
    # Plot panorama.
    print("Please press ESC to quit preview, otherwise OpenCV will be stuck there.")
    plot_save_stitch(stitched, root, 'outputs/extra_credit/'+im+'/stitched_123.jpg')



if __name__ == '__main__':
    main()
    print("Done! :)")
