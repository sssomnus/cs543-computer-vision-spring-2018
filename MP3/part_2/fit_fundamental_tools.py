import numpy as np
from math import sqrt
import random


def fit_fundamental(matches, normalize=False, setup='homogeneous'):
    """Fit fundamental matrix using eight-point alg.
    
    Args:
        matches: Coords of pairs in image 1 and 2, dims (#matches, 4).
        normalize: Boolean flag for using normalized or unnormalized alg.
        setup(str): 'homogeneous' or 'non-homogeneous' setup for solving 
            linear systems.
    Returns:
        F: Fundamental matrix, dims (3, 3).
    """
    # print("Fitting fundamental...")
    all_p1 = matches[:, 0:2]  # All points of image1 in matching pairs.
    all_p2 = matches[:, 2:4]  # All points of image2 in matching pairs.

    # Normalize data.
    if normalize:
        all_p1, T1 = normalization(all_p1)
        all_p2, T2 = normalization(all_p2)

    # Randomly sample 8 matching pairs (unique).
    rand_idx = random.sample(range(matches.shape[0]), k=8)
    select_p1 = all_p1[rand_idx]
    select_p2 = all_p2[rand_idx]

    # Esitimate F.
    F = solve_linear_sys(select_p1, select_p2, setup)

    # Transform F back to original units (denormalize).
    if normalize:
        F = np.dot(np.dot(T2.T, F), T1)
    
    return F


def get_geo_distance(matches, F):
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

    # Final distance.
    dist1 = d1.sum() / matches.shape[0]
    dist2 = d2.sum() / matches.shape[0]

    return dist1, dist2


def normalization(points):
    """Helper function to normalized data in image."""
    # De-mean to center the origin at mean.
    mean = np.mean(points, axis=0)
    # Rescale.
    std_x = np.std(points[:, 0])
    std_y = np.std(points[:, 1])

    # tmp1 = points[:,0]-mean[0]
    # tmp2 = points[:,1]-mean[1]
    # dist = np.sqrt(tmp1**2+tmp2**2)
    # scale = sqrt(2)/np.mean(dist)
    
    # Matrix for transforming points to do normalization.
    transform = np.array([[sqrt(2)/std_x, 0, -sqrt(2)/std_x*mean[0]], 
                          [0, sqrt(2)/std_y, -sqrt(2)/std_y*mean[1]], 
                          [0, 0, 1]])
    # transform = np.array([[scale, 0, -scale*mean[0]], 
    #                       [0, scale, -scale*mean[1]], 
    #                       [0, 0, 1]])
    # Homogeneous coords.
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    normalized = np.dot(transform, points.T).T

    return normalized[:, 0:2], transform


def solve_linear_sys(pairs_p1, pairs_p2, setup):
    """Use 8 pairs to solve linear system to get F,
    with either 'homogeneous' or 'non-homogeneous' setup.
    """
    A_rows = []  # Every row in A is a sublist of A_row.
    
    # Construct A.
    for i in range(pairs_p1.shape[0]):
        p1 = pairs_p1[i]
        p2 = pairs_p2[i]
        
        row = [p2[0]*p1[0], p2[0]*p1[1], p2[0], 
               p2[1]*p1[0], p2[1]*p1[1], p2[1], p1[0], p1[1], 1]
        A_rows.append(row)

    A = np.array(A_rows)

    # Solve linear system.
    if setup == 'homogeneous':
        U, s, V = np.linalg.svd(A)
        F = V[len(V)-1].reshape(3, 3)
        # Normalize F to homogeneous coords.
        F = F / F[2, 2] 
    elif setup == 'non-homogeneous':
        A = A[:, 0:8]  # A is now in dims [8, 8]
        F = np.linalg.solve(A, np.ones(A.shape[1])*(-1))
        F = np.append(F, 1).reshape(3, 3)

    # Enforce rank-2 constraint.
    U, s, Vh = np.linalg.svd(F)
    s_prime = np.diag(s)
    s_prime[-1] = 0
    F = np.dot(U, np.dot(s_prime, Vh))

    return F
