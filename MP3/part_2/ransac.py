import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from fit_fundamental_tools import fit_fundamental


def ransac_fitting(match_coords, threshold):
    """Eliminate outliers and fit the homography using RANSAC alg.
    
    Args:
        match_coords(numpy.ndarray): In dims (#matched pixels, 4).
        threshold(float): For determining inliers.
    Returns:
        best_inliers(numpy.ndarray): In dims (#inliers, 4).
        best_F(numpy.ndarray): Fundamental matrix, dims (3, 3).
        avg_residual(float): Average esidual for all inliers. 
    """
    print("Performing RANSAC...")

    max_ite = 1000
    ite = 0

    num_inliers = 0
    num_best_inliers = 0

    # RANSAC procedure.
    while ite < max_ite:

        # Randomly select 4 matched pairs (unique).
        # rand_idx = random.sample(range(match_coords.shape[0]), k=4)
        # select_pairs = match_coords[rand_idx]

        # Fit a homography.
        # H = fit_homography(select_pairs)
        F = fit_fundamental(match_coords, normalize=True)

        # # Jump to next loop if H is degenerate.
        # if np.linalg.matrix_rank(H) < 3:
        #     continue

        # Find and add inliers.
        errors = get_errors(match_coords, F)
        idx = np.where(errors < threshold)[0]
        inliers = match_coords[idx]

        # Save current solution and compute residual if it's the best.
        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_F = F.copy()
            
            avg_residual = errors[idx].sum() / num_best_inliers

        ite += 1
    
    # print("Number of inliers: {}, Average residual: {}"
    #         .format(num_best_inliers, avg_residual))
    
    return best_inliers, best_F, avg_residual


# def fit_homography(pairs):
#     """Use 4 pairs to compute homography matrix."""
#     A_rows = []  # Every row in A is a sublist of A_row.
    
#     # Construct A.
#     for i in range(pairs.shape[0]):
#         p1 = np.append(pairs[i][0:2], 1)  # [x1, y1, 1]
#         p2 = np.append(pairs[i][2:4], 1)  # [x2, y2, 1]
        
#         row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
#         row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
#         A_rows.append(row1)
#         A_rows.append(row2)

#     A = np.array(A_rows)

#     # SVD and find singular_vec of the smallest singular_val.
#     U, s, V = np.linalg.svd(A)
#     H = V[len(V)-1].reshape(3, 3)

#     # Normalize H.
#     H = H / H[2, 2] 

#     return H


# def get_errors(all_pairs, H):
#     """Compute error or distance between original points and 
#     points transformed by H.
#     Return an array of errors for all points, dims (#pairs,)."""
#     num_pairs = len(all_pairs)

#     all_p1 = np.concatenate((all_pairs[:, 0:2], np.ones((num_pairs, 1))), axis=1)
#     all_p2 = all_pairs[:, 2:4]

#     # Transform every point in p1 to estimate p2.
#     estimate_p2 = np.zeros((num_pairs, 2))
#     for i in range(num_pairs):
#         temp = np.matmul(H, all_p1[i])
#         estimate_p2[i] = (temp/temp[2])[0:2]

#     # Compute error.
#     errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2

#     return errors

def get_errors(matches, F):
    """Compute average geometric distances between epipolar line and its 
    corresponding point in both images. Note that matches is all of the 
    matching pair, not the selected ones in fit_fundamental()."""

    ones = np.ones((matches.shape[0], 1))
    all_p1 = np.concatenate((matches[:, 0:2], ones), axis=1)
    all_p2 = np.concatenate((matches[:, 2:4], ones), axis=1)
    # Epipolar lines.
    F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
    # Geometric distances.
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    # Return array (#points,)
    return (d1 + d2) / 2



def plot_inlier_matches(inliers, root, img_name):
    """Adepted from sample code in part 2:
    http://slazebni.cs.illinois.edu/spring18/assignment3/part2_sample_code_python.py"""
    I1 = Image.open(root+'data/'+img_name+'1.jpg').convert('L')
    I2 = Image.open(root+'data/'+img_name+'2.jpg').convert('L')
    I3 = np.zeros((I1.size[1], I1.size[0]*2))
    I3[:,:I1.size[0]] = I1
    I3[:,I1.size[0]:] = I2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(float), cmap='gray')
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2]+I1.size[0], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2]+I1.size[0]],[inliers[:,1], inliers[:,3]], 
            'r', linewidth=0.4)
    plt.axis('off')
    plt.savefig(root+'outputs/'+img_name+'_inlier_matches.svg', format='svg')
    plt.show()
