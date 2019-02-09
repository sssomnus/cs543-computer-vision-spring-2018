import cv2
import numpy as np


def warp_left(img, H):    
    """Warp the left image and make sure no pixel is out of image frame.
    
    Args:
        img(numpy.ndarray): Left image (BGR) read by cv2.imread().
        H(numpy.ndarray): Homography matrix, dims (3, 3).
    Returns:
        warped_img(numpy.ndarray): A warped image with BGR channels.
            Black pixels means there is no pixel in the original image
            at this location. When plot it, use cv2.imshow() instead of 
            matplotlib.
        (x_min, y_min): Max Translation for warped_img.
    """
    # Find locations of top_left, top_right, 
    # bottom_right, bottom_left corners.
    # Note that height means y, width means x.
    height, width, z = img.shape    
    corners = [[0, 0], [width, 0], [width, height], [0, height]]

    # Transform corner locations to locations in new image.
    corners_new = []
    for corner in corners:
        corner = np.append(np.array(corner), 1)
        corners_new.append(np.matmul(H, corner))
    corners_new = np.array(corners_new).T

    # Find transformed shape (min and max x, y) of new image.
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)
    y_max = max(y_news)
    x_max = max(x_news)

    # Translate new image back so that no pixel is out of bound,
    # by multiplying the backward translation with homography.
    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)

    # Get height, width of new image frame.
    height_new = int(round(abs(y_min) + height))
    width_new = int(round(abs(x_min) + width))
    size = (width_new, height_new)

    # Perform warping.
    warped_img = cv2.warpPerspective(src=img, M=H, dsize=size)

    return warped_img, (x_min, y_min)


def move_right(img, translation):
    """Move the right image (param: img, BGR) according to the translation of 
    the left one after warpping.
    """
    x_min = translation[0]
    y_min = translation[1]
    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    height, width, z = img.shape
    
    height_new = int(round(abs(y_min) + height))
    width_new = int(round(abs(x_min) + width))
    size = (width_new, height_new)
    
    # Move img according to translation_mat (affine transformation).
    moved_img = cv2.warpPerspective(src=img, M=translation_mat, dsize=size)

    return moved_img


def stitch_img(img_left, img_right, H):
    """Stitch two images to create panorama.

    Args:
        img_left, img_right: Normalized BGR images.
        H(numpy.ndarray): Homography matrix, dims (3, 3).
    Returns:
        warped_l(numpy.ndarray): Stitched image (BGR).
    """

    print("Stitching images...")
    warped_l, translation = warp_left(img_left, H)
    moved_r = move_right(img_right, translation)
    
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in range(moved_r.shape[0]):
        for j in range(moved_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = moved_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
            
    # return warped_l
    return warped_l[:moved_r.shape[0], :moved_r.shape[1], :]


def plot_save_stitch(stitched, root, path):
    cv2.imshow('stitched_image', stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert back to 255 scale.
    stitched = cv2.convertScaleAbs(stitched, alpha=255)
    cv2.imwrite(root+path, stitched)